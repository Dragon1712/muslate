"""Closed-loop evaluation of a planner inside the simulator.

The planner never touches the simulator.  It only gets a latent state, either

* the ground-truth ``(h, sat)`` (oracle planning, an upper bound), or
* a belief inferred from the interaction history by the learned world model
  (what deployment would actually look like).

Both go through the same interface, so a difference in return is attributable to
the representation, not to the search.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from muslate.env.simulator import SlateSim
from muslate.planning.search import BasePlanner

LatentFn = Callable[[object, list, list], np.ndarray]


@dataclass
class EvalResult:
    returns: np.ndarray
    latency_ms: float = 0.0
    n_decisions: int = 0
    reward_mae: float | None = None
    tail_returns: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(self.returns.mean())

    @property
    def se(self) -> float:
        n = len(self.returns)
        if n < 2:
            return 0.0
        return float(self.returns.std(ddof=1) / np.sqrt(n))

    @property
    def tail_mean(self) -> float:
        """Mean return excluding the cold-start prefix.

        Reported separately because early steps are dominated by the planner's
        inability to know the user yet, which is common to every horizon and
        would otherwise mask the H comparison.
        """
        if self.tail_returns is None:
            return self.mean
        return float(self.tail_returns.mean())

    @property
    def tail_se(self) -> float:
        if self.tail_returns is None or len(self.tail_returns) < 2:
            return 0.0
        return float(self.tail_returns.std(ddof=1) / np.sqrt(len(self.tail_returns)))


def make_oracle_latent_fn(oracle_model, sim: SlateSim) -> LatentFn:
    def fn(state, _hs, _hr) -> np.ndarray:
        return oracle_model.encode(state)[None]

    return fn


def make_learned_latent_fn(neural_model) -> LatentFn:
    def fn(state, hs, hr) -> np.ndarray:
        if len(hs) == 0:
            return neural_model.initial_latent()
        slates = np.stack(hs)[None]                       # [1, T, K]
        rewards = np.asarray(hr, dtype=np.float64)[None]  # [1, T]
        return neural_model.encode_history(slates, rewards)

    return fn


@dataclass
class WarmStartCase:
    state: object
    hist_slates: list[np.ndarray]
    hist_rewards: list[float]
    user_id: int
    episode_seed: int
    prefix_return: float


def build_warm_start_cases(sim: SlateSim, n_users: int, warmup: int = 5,
                           seed: int = 0) -> list[WarmStartCase]:
    if not 0 <= warmup < sim.cfg.horizon or not 0 < n_users <= sim.cfg.n_users:
        raise ValueError('invalid warmup or user count')
    cases = []
    for u in range(n_users):
        episode_seed = sim.cfg.seed + 1000 + u + seed
        state = sim.initial_state(u, np.random.default_rng(episode_seed))
        rng = np.random.default_rng(episode_seed + 98731)
        hs, hr, ret = [], [], 0.
        for t in range(warmup):
            slate = sim.behavior_slate(state.h, state.sat, rng, epsilon=.25, temperature=1.)
            reward, state = sim.step(state, slate, noise=False)
            hs.append(slate.copy())
            hr.append(float(reward))
            ret += sim.cfg.gamma ** t * reward
        cases.append(WarmStartCase(state.clone(), hs, hr, u, episode_seed, float(ret)))
    return cases


def _sync_planner(planner):
    model = getattr(planner, 'model', None)
    model = getattr(model, 'inner', model)
    device = getattr(model, 'device', None)
    if device is not None and str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def evaluate_from_prefixes(sim: SlateSim, planner: BasePlanner, latent_fn: LatentFn,
                           cases: list[WarmStartCase], measure_latency: bool = True,
                           *, observer=None,decision_observer=None) -> EvalResult:
    """Paired suffix evaluation; observer is an evaluation-only diagnostic hook."""
    returns, tails, total_time, decisions = [], [], 0., 0
    for case in cases:
        state = case.state.clone()
        hs = [slate.copy() for slate in case.hist_slates]
        hr = list(case.hist_rewards)
        tail = 0.
        while state.t < sim.cfg.horizon:
            lat = latent_fn(state, hs, hr)
            if observer is not None:
                observer(state, lat)
            if measure_latency:
                _sync_planner(planner)
                start = time.perf_counter()
            action = planner.plan(lat,remaining_steps=sim.cfg.horizon-state.t)[0].copy()
            if measure_latency:
                _sync_planner(planner)
                total_time += time.perf_counter() - start
            discount = sim.cfg.gamma ** state.t
            previous_state=state
            reward, state = sim.step(state, action, noise=False)
            if decision_observer is not None:
                decision_observer(case.user_id,previous_state,action.copy(),float(reward),planner)
            tail += discount * reward
            hs.append(action)
            hr.append(float(reward))
            decisions += 1
        tails.append(tail)
        returns.append(case.prefix_return + tail)
    return EvalResult(np.asarray(returns), 1000 * total_time / max(decisions, 1),
                      decisions, tail_returns=np.asarray(tails))


def evaluate(
    sim: SlateSim,
    planner: BasePlanner,
    latent_fn: LatentFn,
    n_users: int,
    seed: int = 0,
    measure_latency: bool = True,
    tail_from: int = 0,
) -> EvalResult:
    returns = []
    tails = []
    total_time = 0.0
    n_decisions = 0

    for u in range(n_users):
        state = sim.initial_state(u, np.random.default_rng(sim.cfg.seed + 1000 + u + seed))
        hist_slates: list[np.ndarray] = []
        hist_rewards: list[float] = []
        ret, tail, disc = 0.0, 0.0, 1.0

        while state.t < sim.cfg.horizon:
            step_idx = state.t
            lat = latent_fn(state, hist_slates, hist_rewards)
            if measure_latency:
                t0 = time.perf_counter()
            action = planner.plan(lat,remaining_steps=sim.cfg.horizon-state.t)[0]
            if measure_latency:
                total_time += time.perf_counter() - t0
            n_decisions += 1

            r, state = sim.step(state, action, noise=False)
            hist_slates.append(action)
            hist_rewards.append(r)
            ret += disc * r
            if step_idx >= tail_from:
                tail += disc * r
            disc *= sim.cfg.gamma

        returns.append(ret)
        tails.append(tail)

    return EvalResult(
        returns=np.asarray(returns),
        latency_ms=1000.0 * total_time / max(n_decisions, 1),
        n_decisions=n_decisions,
        tail_returns=np.asarray(tails),
    )


def reward_prediction_error(
    sim: SlateSim, neural_model, n_episodes: int = 32, seed: int = 0
) -> dict[str, float]:
    """One-step and multi-step reward prediction error under the *behaviour* policy.

    This is the "did the world model actually learn the dynamics?" probe (Q1).
    """
    from muslate.config import LogConfig
    from muslate.env.logger import generate_log

    log = generate_log(sim, LogConfig(n_episodes=n_episodes, seed=seed))
    n_ep = int(log.episode_ids.max()) + 1
    t_len = log.horizon
    slates = log.slates.reshape(n_ep, t_len, log.slate_size)
    rewards = log.rewards.reshape(n_ep, t_len)

    # one-step error at every position
    errs = []
    for t in range(t_len):
        if t > 0:
            lat = neural_model.encode_history(slates[:, :t], rewards[:, :t])
        else:
            lat = np.repeat(neural_model.initial_latent(), n_ep, axis=0)
        pred = neural_model.reward(lat, slates[:, t])
        errs.append(np.abs(pred - rewards[:, t]).mean())

    out = {"mae_1step": float(np.mean(errs))}
    # k-step rollout error from the first step
    for k in (1, 2, 3, 5):
        if k + 1 >= t_len:
            continue
        lat = neural_model.encode_history(slates[:, :1], rewards[:, :1])
        err = 0.0
        n = 0
        for j in range(k):
            r_pred, lat = neural_model.step(lat, slates[:, j + 1])
            err += np.abs(r_pred - rewards[:, j + 1]).mean()
            n += 1
        out[f"mae_{k + 1}step_rollout"] = float(err / max(n, 1))
    return out
