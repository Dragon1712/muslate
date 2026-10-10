"""Candidate slate generation.

This is the ``proposal policy`` of the framework: a small, cheap mechanism that
turns a latent state into a set of ``M`` candidate ordered slates.  Under the
planning-centric framing the generator is *not* the contribution -- it exists
only to make the combinatorial action space searchable.

Several abstraction levels are provided so that Gate 3 can ask whether the
Semantic-ID hierarchy actually buys anything:

``item_perturb``   greedy slate + random single-item substitutions (local search)
``item_softmax``   Gumbel top-K sampling from per-item scores (Plackett-Luce)
``uniform``        uniformly random slates (weak baseline pool)
"""

from __future__ import annotations

import numpy as np

from muslate.planning.latent_model import LatentModel
from muslate.planning.validity import validate_slates, distinct_indices, stack_pools
from muslate.planning.cost import timed_proposal


class ProposalPolicy:
    def __init__(
        self,
        model: LatentModel,
        n_candidates: int = 24,
        seed: int = 0,
        mode: str = "item_perturb",
        temperature: float = 0.7,
        n_perturb: int = 1,
    ):
        self.model = model
        self.n_items = model.n_items
        self.k = model.slate_size
        self.m = n_candidates
        self.mode = mode
        self.temperature = temperature
        self.n_perturb = n_perturb
        self.rng = np.random.default_rng(seed)

    # -- public -----------------------------------------------------------
    @timed_proposal
    def propose(self, states: np.ndarray) -> np.ndarray:
        """states [B, D] -> candidates [B, M, K]."""
        states = np.atleast_2d(states)
        b = states.shape[0]
        if hasattr(self.model,'reserve_proposals'):self.model.reserve_proposals(b*self.m)
        scores = self.model.item_scores(states)                 # [B, N]

        if self.mode == "uniform":pool=self._uniform(b)
        elif self.mode == "item_softmax":pool=self._softmax(scores,b)
        elif self.mode == "item_perturb":pool=self._perturb(scores,b)
        else:raise ValueError(f"unknown proposal mode {self.mode!r}")
        validate_slates(pool,self.n_items,self.k)
        return stack_pools([p[distinct_indices(p)] for p in pool])

    # -- modes ------------------------------------------------------------
    def _uniform(self, b: int) -> np.ndarray:
        out = np.empty((b, self.m, self.k), dtype=np.int64)
        for i in range(b):
            for j in range(self.m):
                out[i, j] = self.rng.choice(self.n_items, size=self.k, replace=False)
        return out

    def _softmax(self, scores: np.ndarray, b: int) -> np.ndarray:
        logits = scores / max(self.temperature, 1e-6)
        out = np.empty((b, self.m, self.k), dtype=np.int64)
        for j in range(self.m):
            gumbel = self.rng.gumbel(size=logits.shape)
            out[:, j, :] = np.argsort(-(logits + gumbel), axis=1)[:, : self.k]
        return out

    def _perturb(self, scores: np.ndarray, b: int) -> np.ndarray:
        base = np.argsort(-scores, axis=1)[:, : self.k]              # [B, K] greedy
        out = np.repeat(base[:, None, :], self.m, axis=1)            # [B, M, K]
        for i in range(b):
            for j in range(1, self.m):
                s = out[i, j].copy()
                for _ in range(self.n_perturb):
                    pos = int(self.rng.integers(self.k))
                    # Draw from the actual complement; full-catalog slates stay
                    # unchanged rather than exhausting retries and duplicating.
                    available=np.setdiff1d(np.arange(self.n_items),s,assume_unique=True)
                    if len(available):s[pos]=int(self.rng.choice(available))
                out[i, j] = s
        return out
