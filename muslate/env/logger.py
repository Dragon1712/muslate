"""Offline interaction-log generation.

The log is what a real offline recommender dataset would give us: for each
(episode, step) we see the *ordered slate* that was shown and the *scalar
reward* that came back.  The user's latent state is NOT observed -- it is
recorded separately only so that the ``oracle_state`` world model can be
trained as a reference point.

The logging policy is a temperature-controlled myopic greedy with epsilon
exploration, i.e. realistic: competent enough to be informative, biased enough
that naive off-policy evaluation is misleading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from muslate.config import LogConfig
from muslate.env.simulator import SlateSim


@dataclass
class OfflineLog:
    episode_ids: np.ndarray   # [T] int
    step_ids: np.ndarray      # [T] int
    slates: np.ndarray        # [T, K] int
    rewards: np.ndarray       # [T] float
    states: np.ndarray        # [T, D] ground-truth latent (h, sat)
    n_items: int
    slate_size: int
    horizon: int

    def __len__(self) -> int:
        return self.slates.shape[0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            episode_ids=self.episode_ids,
            step_ids=self.step_ids,
            slates=self.slates,
            rewards=self.rewards,
            states=self.states,
            meta=np.asarray([self.n_items, self.slate_size, self.horizon], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OfflineLog":
        d = np.load(path)
        n_items, slate_size, horizon = d["meta"].tolist()
        return cls(
            episode_ids=d["episode_ids"],
            step_ids=d["step_ids"],
            slates=d["slates"],
            rewards=d["rewards"],
            states=d["states"],
            n_items=int(n_items),
            slate_size=int(slate_size),
            horizon=int(horizon),
        )


@dataclass(slots=True)
class ObservedLog:
    """Deployable observations; no latent state or unexposed-item targets."""

    episode_ids: np.ndarray
    step_ids: np.ndarray
    slates: np.ndarray
    rewards: np.ndarray
    n_items: int
    slate_size: int
    horizon: int
    item_feedback: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.slates)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = dict(episode_ids=self.episode_ids, step_ids=self.step_ids,
                      slates=self.slates, rewards=self.rewards,
                      meta=np.asarray([self.n_items, self.slate_size, self.horizon], dtype=np.int64))
        if self.item_feedback is not None:
            fields['item_feedback'] = self.item_feedback
        np.savez_compressed(path, **fields)

    @classmethod
    def load(cls, path: str | Path) -> 'ObservedLog':
        with np.load(path, allow_pickle=False) as data:
            if 'states' in data.files:
                raise ValueError('ObservedLog cannot load privileged state fields')
            n_items, slate_size, horizon = map(int, data['meta'])
            return cls(episode_ids=data['episode_ids'], step_ids=data['step_ids'],
                       slates=data['slates'], rewards=data['rewards'], n_items=n_items,
                       slate_size=slate_size, horizon=horizon,
                       item_feedback=data['item_feedback'] if 'item_feedback' in data.files else None)


def as_observed_log(log: OfflineLog, item_feedback: np.ndarray | None = None) -> ObservedLog:
    """Copy only allowed observations across the training information boundary."""
    if item_feedback is not None and item_feedback.shape != log.slates.shape:
        raise ValueError('item_feedback must have shape [rows, slate_size]')
    return ObservedLog(log.episode_ids.copy(), log.step_ids.copy(), log.slates.copy(),
                       log.rewards.copy(), log.n_items, log.slate_size, log.horizon,
                       None if item_feedback is None else item_feedback.copy())


def _generate_arrays(sim: SlateSim, cfg: LogConfig, *, states: bool, feedback: bool) -> dict:
    if cfg.n_episodes <= 0 or cfg.item_feedback_noise < 0:
        raise ValueError('episodes must be positive and feedback noise nonnegative')
    rng = np.random.default_rng(cfg.seed + 4242)
    feedback_rng = np.random.default_rng(cfg.seed + 99173)

    ep_ids, steps, slates, rewards, latent_states, item_feedback = [], [], [], [], [], []
    for e in range(cfg.n_episodes):
        user = int(rng.integers(sim.cfg.n_users))
        state = sim.initial_state(user, np.random.default_rng(cfg.seed * 7919 + e))
        for t in range(sim.cfg.horizon):
            slate = sim.behavior_slate(
                state.h, state.sat, rng, epsilon=cfg.epsilon, temperature=cfg.temperature
            )
            if states:
                latent_states.append(np.concatenate([state.h, state.sat]))
            if feedback:
                # Only the served IDs receive labels. Simulation is an oracle
                # data generator, but none of its hidden inputs are exported.
                y = sim.catalog.item_emb[slate] @ state.h
                y = y - sim.lt * sim.cfg.sat_penalty * state.sat[sim.catalog.item_coarse[slate]]
                y = y + feedback_rng.normal(0, cfg.item_feedback_noise, size=sim.slate_size)
                item_feedback.append(y)
            slates.append(slate)
            r, state = sim.step(state, slate, rng, noise=True)
            ep_ids.append(e)
            steps.append(t)
            rewards.append(r)

    arrays = dict(
        episode_ids=np.asarray(ep_ids, dtype=np.int64),
        step_ids=np.asarray(steps, dtype=np.int64),
        slates=np.asarray(slates, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float64),
        n_items=sim.n_items,
        slate_size=sim.slate_size,
        horizon=sim.cfg.horizon,
    )
    if states:
        arrays['states'] = np.asarray(latent_states, dtype=np.float64)
    if feedback:
        arrays['item_feedback'] = np.asarray(item_feedback, dtype=np.float64)
    return arrays


def generate_log(sim: SlateSim, cfg: LogConfig) -> OfflineLog:
    return OfflineLog(**_generate_arrays(sim, cfg, states=True, feedback=False))


def generate_observed_log(sim: SlateSim, cfg: LogConfig) -> ObservedLog:
    """Synthetic continuous feedback export; not a real click-log experiment."""
    return ObservedLog(**_generate_arrays(sim, cfg, states=False, feedback=True))
