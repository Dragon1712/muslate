"""Corrupt the *prior*, not the tree.

`SIDMap` lets the sweep vary how good the search tree is.  This decorator varies
a different thing: how good the per-item scores are that the tree is walked with.
The two are independent failure modes and the sweep has to separate them --
"the hierarchy failed because the tree was wrong" and "the hierarchy failed
because the scores it sorted by were wrong" call for completely different fixes.

The corruption is a **fixed** per-class offset, drawn once at construction:

    item_scores(h) + bonus[coarse_of[i]],   bonus ~ N(0, noise^2)^K

Two consequences worth stating, because they are the reason this is not simply
"add random noise to the scores":

* It is *systematic*, not jittery.  The model consistently believes one class is
  better than another, which is what a mis-specified world model actually looks
  like.  Resampling per call would instead make `_coarse_plans` and `_complete`
  disagree *within a single `propose()`* -- internal incoherence, a different
  phenomenon.
* It preserves within-class order.  Only the *ranking between classes* is
  damaged.  That is precisely the ordering a coarse-to-fine hierarchy commits to
  first, so if the hierarchy is more sensitive to it than flat search is, this is
  where it shows.

`reward` and `step` delegate untouched, so only the search is affected, never the
environment being searched.
"""

from __future__ import annotations

import numpy as np

from muslate.planning.latent_model import LatentModel


class NoisyScoreModel(LatentModel):
    """Wraps a `LatentModel` and adds a fixed, per-class offset to `item_scores`."""

    def __init__(
        self,
        inner: LatentModel,
        coarse_of: np.ndarray,
        n_coarse: int,
        noise: float,
        seed: int = 0,
    ):
        self.inner = inner
        self.state_dim = inner.state_dim
        self.n_items = inner.n_items
        self.slate_size = inner.slate_size

        self.coarse_of = np.asarray(coarse_of, dtype=np.int64)
        self.n_coarse = int(n_coarse)
        self.noise = float(noise)

        if self.noise > 0.0:
            rng = np.random.default_rng(seed)
            self.bonus = rng.normal(0.0, self.noise, size=self.n_coarse)
        else:
            self.bonus = np.zeros(self.n_coarse, dtype=np.float64)
        # precompute the per-item view so `item_scores` is a single broadcast add
        self._bonus_per_item = self.bonus[self.coarse_of]

    # -- LatentModel --------------------------------------------------------
    def item_scores(self, states: np.ndarray) -> np.ndarray:
        scores = np.asarray(self.inner.item_scores(states), dtype=np.float64)
        if self.noise <= 0.0:
            return scores
        return scores + self._bonus_per_item[None, :]

    def reward(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        return self.inner.reward(states, slates)

    def step(self, states: np.ndarray, slates: np.ndarray):
        return self.inner.step(states, slates)
