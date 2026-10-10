"""Latent-dynamics interface shared by the oracle simulator and the learned
world model.

Everything in :mod:`muslate.planning` is written against :class:`LatentModel`
only, so the *identical* planner code can be run with

* ground-truth dynamics (an upper bound on achievable performance), and
* a learned world model (what we would actually deploy).

That separation is what lets us tell "planning cannot help here" apart from
"our model is too weak to plan with".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import time

import numpy as np

from muslate.env import SimState, SlateSim
from muslate.planning.cost import PlanningCost
from muslate.planning.validity import validate_slates


class LatentModel(ABC):
    """Batch interface over a (learned or true) latent dynamics model."""

    state_dim: int
    n_items: int
    slate_size: int

    @abstractmethod
    def reward(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        """states [B, D], slates [B, K] -> reward [B]."""

    @abstractmethod
    def step(self, states: np.ndarray, slates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """states [B, D], slates [B, K] -> (reward [B], next_states [B, D])."""

    @abstractmethod
    def item_scores(self, states: np.ndarray) -> np.ndarray:
        """Myopic per-item score used to bias candidate generation. [B, N]."""


class OracleModel(LatentModel):
    """The simulator exposed through the latent-model interface."""

    def __init__(self, sim: SlateSim):
        self.sim = sim
        self.dim = sim.cfg.dim
        self.state_dim = sim.cfg.dim + sim.n_coarse
        self.n_items = sim.n_items
        self.slate_size = sim.slate_size

    # -- state helpers ----------------------------------------------------
    def encode(self, state: SimState) -> np.ndarray:
        return np.concatenate([state.h, state.sat]).astype(np.float64)

    def _split(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return states[:, : self.dim], states[:, self.dim :]

    # -- LatentModel ------------------------------------------------------
    def reward(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        h, sat = self._split(states)
        return self.sim.reward_batch(h, sat, slates)

    def step(self, states: np.ndarray, slates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, sat = self._split(states)
        r, h2, s2 = self.sim.step_batch(h, sat, slates)
        return r, np.concatenate([h2, s2], axis=1)

    def item_scores(self, states: np.ndarray) -> np.ndarray:
        h, sat = self._split(states)
        base = h @ self.sim.catalog.item_emb.T                       # [B, N]
        pen = self.sim.lt * self.sim.cfg.sat_penalty * sat[:, self.sim.catalog.item_coarse]
        return base - pen


class CountingModel(LatentModel):
    """Wraps a :class:`LatentModel` and counts model evaluations.

    Different planners spend their budget in structurally different ways -- a
    beam expands ``M`` slates at every level, MCTS spreads the same total over
    simulations, CEM re-samples its elite distribution.  Comparing them at
    "the same nominal budget" is therefore meaningless unless the *actual*
    number of model calls is measured.  This wrapper is what makes the Q3
    comparison honest.

    ``n_slate_evals`` counts reward/dynamics evaluations of concrete slates
    (the expensive ones); ``n_score_calls`` counts item-score queries, which cost
    one vector op per state.
    """

    def __init__(self, inner: LatentModel, weights=None):
        self.inner = inner
        self.state_dim = inner.state_dim
        self.n_items = inner.n_items
        self.slate_size = inner.slate_size
        self.n_slate_evals = 0
        self.n_score_calls = 0
        self.weights=weights
        self.cost=PlanningCost();self.last_cost=self.cost
        self._seen=set();self._budget=None

    @property
    def device(self):return getattr(self.inner,'device',None)

    @property
    def estimated_compute_ms(self):
        w=self._budget.weights if self._budget is not None else self.weights
        return self.cost.estimate(w) if w is not None else None

    @contextmanager
    def decision(self,budget=None):
        self.cost=PlanningCost();self._seen=set();self._budget=budget
        start=time.perf_counter()
        try:yield
        finally:
            self.cost.planning_time_ms=1000*(time.perf_counter()-start)
            self.last_cost=self.cost;self._budget=None

    @contextmanager
    def proposal_scope(self):
        model_start=self.cost.model_time_ms;start=time.perf_counter()
        try:yield
        finally:
            total=1000*(time.perf_counter()-start)
            self.cost.proposal_wall_ms+=total
            self.cost.proposal_time_ms+=max(0.,total-(self.cost.model_time_ms-model_start))

    def _charge(self,field,n):
        if self._budget is not None:
            attr={'reward_only_calls':'reward_ms','transition_calls':'transition_ms',
                  'score_queries':'score_ms','proposal_candidates':'proposal_ms'}[field]
            self._budget.spend(n*getattr(self._budget.weights,attr))
        setattr(self.cost,field,getattr(self.cost,field)+n)

    def reserve_proposals(self,n):self._charge('proposal_candidates',int(n))

    def record_emitted(self,pool):self.cost.emitted_candidates+=int(np.prod(pool.shape[:-1]))

    def _slates(self,a):
        validate_slates(a,self.n_items,self.slate_size)
        self._seen.update(tuple(row) for row in np.atleast_2d(a))
        self.cost.unique_slates=len(self._seen)

    def _timed(self,fn,*args):
        start=time.perf_counter()
        try:return fn(*args)
        finally:self.cost.model_time_ms+=1000*(time.perf_counter()-start)

    def reset(self) -> None:
        self.n_slate_evals = 0
        self.n_score_calls = 0
        self.cost=PlanningCost();self.last_cost=self.cost;self._seen=set()

    def reward(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        self._charge('reward_only_calls',int(np.atleast_2d(slates).shape[0]));self._slates(slates)
        self.n_slate_evals += int(np.atleast_2d(slates).shape[0])
        return self._timed(self.inner.reward,states,slates)

    def step(self, states: np.ndarray, slates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._charge('transition_calls',int(np.atleast_2d(slates).shape[0]));self._slates(slates)
        self.n_slate_evals += int(np.atleast_2d(slates).shape[0])
        return self._timed(self.inner.step,states,slates)

    def item_scores(self, states: np.ndarray) -> np.ndarray:
        self._charge('score_queries',int(np.atleast_2d(states).shape[0]))
        self.n_score_calls += int(np.atleast_2d(states).shape[0])
        return self._timed(self.inner.item_scores,states)

    def uncertainty(self,states,slates):
        members=getattr(self.inner,'models',None)
        if members is None or len(members)<2:
            raise ValueError('metered disagreement requires at least two neural members')
        n=len(members)*len(np.atleast_2d(slates))
        self._charge('reward_only_calls',n);self._slates(slates);self.n_slate_evals+=n
        return self._timed(self.inner.uncertainty,states,slates)
