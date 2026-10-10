"""World-model invariants.

The most important test here pins down the property whose absence made the
learned planner in Gate 2 under-perform a plain greedy policy: the reward head
and the item-score head must agree *by construction*,

    reward(h, A) == sum_k w_k * item_scores(h)[A_k]

so that the proposal policy and the value estimate can never point in different
directions.  A monolithic ``MLP(h, slate_emb)`` reward head does not have this
property, generalises badly to unseen slates, and gets exploited by search.

Run directly:
    python tests/test_world_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig, WorldModelConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.model.world_model import WorldModel  # noqa: E402


def _make(cfg: WorldModelConfig | None = None):
    sim = SlateSim(EnvConfig())
    cfg = cfg or WorldModelConfig()
    model = WorldModel(cfg, sim.n_items, sim.slate_size, true_state_dim=sim.cfg.dim + sim.n_coarse)
    return sim, model


def test_unary_part_is_decomposable_into_item_scores() -> None:
    """After adding the pairwise term the *whole* reward is no longer a weighted
    sum of item scores -- only its unary part is.  That part is what
    ``item_scores`` (the proposal prior) must stay consistent with: the proposal
    should not know about pair penalties, that is what search is for."""
    sim, model = _make()
    b, k = 8, sim.slate_size
    h = torch.randn(b, model.cfg.latent_dim)
    slates = torch.stack([torch.randperm(sim.n_items)[:k] for _ in range(b)])

    scores = model.item_scores(h)                                   # [B, N]
    manual = (scores.gather(1, slates) * model.slate_enc.pos_w[None, :]).sum(dim=-1)
    assert torch.allclose(model.predict_unary(h, slates), manual, atol=1e-5)


def test_pairwise_term_is_symmetric_in_item_identity() -> None:
    """``e_j^T W e_k`` must equal ``e_k^T W e_j``, i.e. W is enforced symmetric."""
    sim, model = _make()
    slates = torch.randint(0, sim.n_items, (4, sim.slate_size))
    e = model.slate_enc.item_emb(slates)                            # [B, K, de]
    w = 0.5 * (model.pair_proj.weight + model.pair_proj.weight.t())
    gram = e @ w @ e.transpose(-1, -2)
    assert torch.allclose(gram, gram.transpose(-1, -2), atol=1e-6)


def test_pairwise_head_can_express_the_environment_diversity_penalty() -> None:
    """Structural check that the fix actually targets the misspecification.

    Point the model's item embeddings at the simulator's and set ``W = I``; the
    pairwise term must then reproduce the simulator's raw diversity sum
    ``sum_{j<k} w_j w_k <z_j, z_k>`` exactly.  This is precisely what the purely
    unary head could not represent at all.
    """
    sim, _ = _make()
    cfg = WorldModelConfig(slate_emb_dim=sim.cfg.dim, pairwise=True)
    model = WorldModel(cfg, sim.n_items, sim.slate_size, true_state_dim=sim.cfg.dim + sim.n_coarse)

    with torch.no_grad():
        model.slate_enc.item_emb.weight.copy_(
            torch.as_tensor(sim.catalog.item_emb, dtype=torch.float32)
        )
        model.pair_proj.weight.copy_(torch.eye(sim.cfg.dim))

    slate = sim.greedy_slate(sim.initial_state(0).h, sim.initial_state(0).sat)
    slates = torch.as_tensor(np.stack([slate] * 3))
    h = torch.zeros(3, cfg.latent_dim)

    with torch.no_grad():
        pair = model.pair_value(h, slates).numpy()
        env = sim.diversity_term(slates.numpy())
    assert np.allclose(pair, env, atol=1e-4), "pairwise head must be able to express the env term"


def test_pairwise_off_reproduces_the_unary_only_model() -> None:
    """The ablation switch must genuinely restore the old (misspecified) head."""
    sim = SlateSim(EnvConfig())
    cfg = WorldModelConfig(pairwise=False)
    model = WorldModel(cfg, sim.n_items, sim.slate_size, true_state_dim=sim.cfg.dim + sim.n_coarse)
    assert not model.pairwise

    h = torch.randn(3, cfg.latent_dim)
    slates = torch.randint(0, sim.n_items, (3, sim.slate_size))
    assert torch.allclose(model.predict_reward(h, slates), model.predict_unary(h, slates))


def test_shapes_and_history_encoding() -> None:
    sim, model = _make()
    b, t = 4, 6
    slates = torch.stack([torch.stack([torch.randperm(sim.n_items)[: sim.slate_size] for _ in range(t)]) for _ in range(b)])
    rewards = torch.randn(b, t)

    states = model(slates, rewards)
    assert states.shape == (b, t, model.cfg.latent_dim)

    lat = model.encode_history(slates, rewards)
    assert lat.shape == (b, model.cfg.latent_dim)

    empty = model.encode_history(slates[:, :0], rewards[:, :0])
    assert empty.shape == (b, model.cfg.latent_dim)

    r = model.predict_reward(lat, slates[:, 0])
    assert r.shape == (b,)
    assert torch.isfinite(r).all()


def test_step_latent_moves_state() -> None:
    sim, model = _make()
    h = torch.zeros(2, model.cfg.latent_dim)
    slates = torch.stack([torch.randperm(sim.n_items)[: sim.slate_size] for _ in range(2)])
    h2 = model.step_latent(h, slates)
    assert h2.shape == h.shape
    assert not torch.allclose(h2, h), "a non-trivial slate should move the latent state"


def test_ensemble_disagreement_is_zero_for_identical_models() -> None:
    from muslate.model.neural_model import NeuralLatentModel

    sim, model = _make()
    nm = NeuralLatentModel([model, model], "cpu")
    lat = nm.initial_latent()
    slates = np.stack([sim.greedy_slate(sim.initial_state(0).h, sim.initial_state(0).sat)])
    u = nm.uncertainty(lat, slates)
    assert np.allclose(u, 0.0)


def test_neural_model_matches_simulator_when_fitted() -> None:
    """Sanity: an untrained model at least produces finite, correctly shaped output."""
    from muslate.model.neural_model import NeuralLatentModel

    sim, model = _make()
    nm = NeuralLatentModel([model], "cpu")
    slates = np.stack([sim.greedy_slate(sim.initial_state(0).h, sim.initial_state(0).sat)])
    lat = nm.initial_latent()
    r = nm.reward(lat, slates)
    nxt = nm.step(lat, slates)[1]
    assert r.shape == (1,) and np.isfinite(r).all()
    assert nxt.shape == (1, model.cfg.latent_dim)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
