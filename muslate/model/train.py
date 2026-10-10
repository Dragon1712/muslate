"""Training the latent world model from an offline interaction log.

Loss = one-step reward MSE
     + multi-step reward MSE (rolled out through the model's own dynamics)
     + optional latent-state aux MSE (oracle-assisted representation)
     + optional item-score aux MSE (so the proposal policy has a usable prior)
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from muslate.config import EnvConfig, LogConfig, WorldModelConfig
from muslate.env.logger import OfflineLog, ObservedLog, generate_log
from muslate.env.simulator import SlateSim
from muslate.model.world_model import WorldModel
from muslate.planning.validity import validate_slates


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def true_item_scores(sim: SlateSim, states: np.ndarray) -> np.ndarray:
    """The simulator's myopic per-item score for a batch of latent states [B, D]."""
    d = sim.cfg.dim
    h, sat = states[:, :d], states[:, d:]
    base = h @ sim.catalog.item_emb.T
    pen = sim.lt * sim.cfg.sat_penalty * sat[:, sim.catalog.item_coarse]
    return base - pen


def logs_to_tensors(log: OfflineLog, device: torch.device):
    n_ep = int(log.episode_ids.max()) + 1
    t = log.horizon
    k = log.slate_size
    slates = torch.as_tensor(log.slates, dtype=torch.long, device=device).reshape(n_ep, t, k)
    rewards = torch.as_tensor(log.rewards, dtype=torch.float32, device=device).reshape(n_ep, t)
    states = torch.as_tensor(log.states, dtype=torch.float32, device=device).reshape(n_ep, t, -1)
    return slates, rewards, states


def train_world_model(
    sim: SlateSim,
    log: OfflineLog,
    cfg: WorldModelConfig,
    device: torch.device,
    verbose: bool = True,
) -> WorldModel:
    """Legacy privileged training. Do not label this interface strict offline."""
    torch.manual_seed(cfg.seed)
    slates, rewards, states = logs_to_tensors(log, device)
    model = WorldModel(cfg, sim.n_items, sim.slate_size, true_state_dim=states.shape[-1]).to(device)
    if cfg.score_supervision not in ('none', 'oracle_full'):
        raise ValueError('use train_logged_world_model for exposed feedback')
    scores_true = None
    if cfg.score_supervision == 'oracle_full' and cfg.w_score != 0:
        scores_true = torch.as_tensor(true_item_scores(sim, log.states),
                                      dtype=torch.float32, device=device).reshape(
                                          slates.shape[0], log.horizon, sim.n_items)
    return _fit_world_model(model, slates, rewards, cfg,
                           oracle_states=states if cfg.oracle_state else None,
                           oracle_scores=scores_true, verbose=verbose)


def _observed_tensors(log: ObservedLog, device: torch.device):
    if type(log) is not ObservedLog:
        raise ValueError('strict training requires an ObservedLog export')
    rows, t, k = len(log), log.horizon, log.slate_size
    if rows == 0 or t <= 0 or k <= 0 or log.n_items < k or rows % t:
        raise ValueError('positive complete episodes and valid item/slate sizes required')
    n_ep = rows // t
    validate_slates(log.slates,log.n_items,k)
    if (log.slates.shape != (rows, k) or log.rewards.shape != (rows,)
            or not np.issubdtype(log.slates.dtype, np.integer)
            or np.any(log.slates < 0) or np.any(log.slates >= log.n_items)
            or not np.isfinite(log.rewards).all()
            or not np.array_equal(log.episode_ids, np.repeat(np.arange(n_ep), t))
            or not np.array_equal(log.step_ids, np.tile(np.arange(t), n_ep))):
        raise ValueError('log rows must be finite, valid, and in complete episode/step order')
    slates = torch.as_tensor(log.slates, dtype=torch.long, device=device).reshape(n_ep, t, k)
    rewards = torch.as_tensor(log.rewards, dtype=torch.float32, device=device).reshape(n_ep, t)
    return slates, rewards


def train_logged_world_model(log: ObservedLog, cfg: WorldModelConfig,
                             device: torch.device, verbose: bool = True) -> WorldModel:
    """Train only from observations. No simulator, state, or full-score channel."""
    if cfg.oracle_state or cfg.score_supervision not in ('none', 'exposed'):
        raise ValueError('strict training rejects privileged state/full-score supervision')
    slates, rewards = _observed_tensors(log, device)
    feedback = None
    if cfg.score_supervision == 'exposed':
        if log.item_feedback is None or log.item_feedback.shape != log.slates.shape:
            raise ValueError('exposed supervision requires item_feedback shaped like slates')
        feedback = torch.as_tensor(log.item_feedback, dtype=torch.float32,
                                    device=device).reshape(slates.shape)
    torch.manual_seed(cfg.seed)
    model = WorldModel(cfg, log.n_items, log.slate_size).to(device)
    return _fit_world_model(model, slates, rewards, cfg,
                           exposed_feedback=feedback, verbose=verbose)


def _exposed_feedback_loss(model: WorldModel, h: torch.Tensor,
                           slates: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = model.item_value(h, slates)
    mask = torch.isfinite(target)
    # Sanitise before subtraction: NaN multiplied by zero still poisons gradients.
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    error = torch.where(mask, prediction - safe_target, torch.zeros_like(prediction))
    return error.square().sum() / mask.sum().clamp_min(1)


def _fit_world_model(model: WorldModel, slates: torch.Tensor, rewards: torch.Tensor,
                     cfg: WorldModelConfig, *, exposed_feedback: torch.Tensor | None = None,
                     oracle_states: torch.Tensor | None = None,
                     oracle_scores: torch.Tensor | None = None,
                     verbose: bool = True) -> WorldModel:
    if cfg.batch_size <= 0 or cfg.epochs <= 0 or cfg.rollout_depth < 0:
        raise ValueError('positive batch_size/epochs and nonnegative rollout_depth required')
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n_ep = slates.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    model.training_history = []
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n_ep, generator=gen)
        totals = dict(loss=0., reward=0., multistep=0., state=0., score=0.)
        for start in range(0, n_ep, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            sl, rw = slates[idx], rewards[idx]
            states_seq = model(sl, rw)
            t_len = sl.shape[1]
            reward_loss = F.mse_loss(model.predict_reward(states_seq, sl), rw)
            multi = states_seq.new_zeros(())
            for j in range(1, min(cfg.rollout_depth, t_len - 1) + 1):
                h = states_seq[:, : t_len - j]
                for s in range(j):
                    h = model.step_latent(h, sl[:, s : t_len - j + s])
                r_hat = model.predict_reward(h, sl[:, j:])
                multi = multi + F.mse_loss(r_hat, rw[:, j:])
            multi = multi / max(cfg.rollout_depth, 1)
            state_loss = states_seq.new_zeros(())
            if model.state_head is not None:
                if oracle_states is None:
                    raise ValueError('state head requires explicit privileged targets')
                state_loss = F.mse_loss(model.state_head(states_seq), oracle_states[idx])
            score_loss = states_seq.new_zeros(())
            if exposed_feedback is not None:
                score_loss = _exposed_feedback_loss(model, states_seq, sl, exposed_feedback[idx])
            elif oracle_scores is not None:
                score_loss = F.mse_loss(model.item_scores(states_seq), oracle_scores[idx])
            loss = (cfg.w_reward * reward_loss + cfg.w_multistep * multi
                    + cfg.w_state * state_loss + cfg.w_score * score_loss)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for key, value in (('loss', loss), ('reward', reward_loss), ('multistep', multi),
                               ('state', state_loss), ('score', score_loss)):
                totals[key] += float(value.detach()) * idx.numel()
        report = {key: value / n_ep for key, value in totals.items()}
        model.training_history.append(report)
        if verbose and (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            print(f"  [wm] epoch {epoch + 1:>3}/{cfg.epochs}  loss={report['loss']:.4f}", flush=True)
    return model


def train_ensemble(
    sim: SlateSim,
    log_cfgs: list[LogConfig] | None,
    cfg: WorldModelConfig,
    device: torch.device,
    verbose: bool = True,
) -> list[WorldModel]:
    """Train ``cfg.n_ensemble`` models on independently logged datasets.

    Data diversity, not just init diversity, is what makes the ensemble
    disagreement a usable epistemic-uncertainty signal.
    """
    models = []
    for i in range(cfg.n_ensemble):
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.seed = cfg.seed + 1000 * i
        log_cfg = (log_cfgs[i] if log_cfgs else LogConfig(seed=cfg.seed + 1000 * i))
        log_cfg = copy.deepcopy(log_cfg)
        log_cfg.seed = cfg.seed + 1000 * i
        if verbose:
            print(f"[ensemble {i + 1}/{cfg.n_ensemble}] generating log ...")
        log = generate_log(sim, log_cfg)
        models.append(train_world_model(sim, log, sub_cfg, device, verbose=verbose))
    return models


def build_dataset(
    env_cfg: EnvConfig,
    log_cfg: LogConfig,
    path: str | Path | None = None,
) -> tuple[SlateSim, OfflineLog]:
    """Simulator + log, cached on disk if ``path`` is given."""
    sim = SlateSim(env_cfg)
    if path is not None and Path(path).exists():
        return sim, OfflineLog.load(path)
    log = generate_log(sim, log_cfg)
    if path is not None:
        log.save(path)
    return sim, log


# Architecture variations for a heterogeneous ensemble.  ``latent_dim`` is
# deliberately NOT varied: every member must live in the same belief space,
# otherwise the planner cannot feed one state to all of them and the whole
# "ensemble of dynamics models" abstraction collapses.  Everything downstream of
# the belief -- slate encoding capacity, MLP width, multi-step training strength,
# learning rate -- is fair game, and that is where the disagreement comes from.
HETERO_SPECS: list[dict] = [
    dict(slate_emb_dim=32, hidden_dim=64, rollout_depth=1, lr=5e-3),
    dict(slate_emb_dim=64, hidden_dim=128, rollout_depth=3, lr=3e-3),
    dict(slate_emb_dim=96, hidden_dim=256, rollout_depth=5, lr=1e-3),
    dict(slate_emb_dim=48, hidden_dim=96, rollout_depth=2, lr=4e-3),
]


def train_heterogeneous_ensemble(
    sim: SlateSim,
    log_cfgs: list[LogConfig],
    base_cfg: WorldModelConfig,
    device: torch.device,
    specs: list[dict] | None = None,
    verbose: bool = True,
) -> list[WorldModel]:
    """Ensemble whose members differ in architecture, capacity and training.

    Identically-trained members make nearly identical mistakes, so their
    disagreement says little about error (Gate 4 measured ρ ≈ 0.1 for exactly
    that configuration).  Varying what each member can represent, and how hard it
    is pushed on multi-step prediction, is the cheap way to decorrelate them.
    """
    specs = specs if specs is not None else HETERO_SPECS
    models = []
    for i in range(base_cfg.n_ensemble):
        spec = specs[i % len(specs)]
        cfg = copy.deepcopy(base_cfg)
        for key, value in spec.items():
            setattr(cfg, key, value)
        cfg.seed = base_cfg.seed + 100 * i

        log_cfg = copy.deepcopy(log_cfgs[i % len(log_cfgs)])
        log = generate_log(sim, log_cfg)
        if verbose:
            print(f"  [hetero {i + 1}/{base_cfg.n_ensemble}] {spec}")
        models.append(train_world_model(sim, log, cfg, device, verbose=verbose))
    return models
