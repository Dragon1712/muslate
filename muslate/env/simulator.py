"""Ground-truth slate recommendation environment.

The environment is a partially observable, long-horizon MDP over *ordered
slates*:

    state_t   = (h_t, sat_t)
    action_t  = slate_t = (i_1, ..., i_K)          ordered, K = slate_size
    reward_t  = <h_t, weighted_mean(z_slate)>
                - lt * sat_penalty * <w, sat_t[cat(slate)]>   (delayed!)
                + noise
    state_{t+1} = ( drift(h_t, slate_t), decay(sat_t, slate_t) )

Two mechanisms create genuine long-horizon dependency:

1. **Delayed saturation.**  Showing items from a coarse category makes that
   category *tiresome* from the next step onward.  The penalty is applied at
   step t but is produced by step t-1's exposure, so a myopic policy that
   squeezes the currently-best category pays for it later.
2. **Preference drift.**  The user's interest vector moves toward what is being
   shown, which changes which items are good a few steps later.

Both are scaled by ``long_term_strength`` (lt).  With lt = 0 the problem is a
contextual bandit and immediate greedy is optimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from muslate.config import EnvConfig
from muslate.env.catalog import Catalog
from muslate.planning.validity import validate_slates


@dataclass
class SimState:
    """Full ground-truth state (only the simulator may peek at it)."""

    h: np.ndarray          # [dim]   user interest vector
    sat: np.ndarray        # [n_coarse] saturation per coarse category
    t: int = 0
    slates: list = field(default_factory=list)
    rewards: list = field(default_factory=list)

    def clone(self) -> "SimState":
        return SimState(
            h=self.h.copy(),
            sat=self.sat.copy(),
            t=self.t,
            slates=list(self.slates),
            rewards=list(self.rewards),
        )


class SlateSim:
    def __init__(self, cfg: EnvConfig, catalog: Catalog | None = None):
        self.cfg = cfg
        self.catalog = catalog if catalog is not None else Catalog(cfg.catalog)
        self.pos_w = (cfg.pos_discount ** np.arange(cfg.slate_size)).astype(np.float64)
        # weighted upper-triangular mask for the intra-slate diversity penalty
        pair_w = self.pos_w[:, None] * self.pos_w[None, :]
        self._pair_w = pair_w * np.triu(np.ones_like(pair_w), k=1)
        self.float_rng = np.random.default_rng(cfg.seed + 7717)

        pref = np.random.default_rng(cfg.seed + 101).normal(
            size=(cfg.n_users, cfg.dim)
        )
        pref /= np.linalg.norm(pref, axis=1, keepdims=True)
        self.user_pref = pref

    # -- basic geometry ---------------------------------------------------
    @property
    def n_items(self) -> int:
        return self.catalog.n_items

    @property
    def n_coarse(self) -> int:
        return self.catalog.n_coarse

    @property
    def slate_size(self) -> int:
        return self.cfg.slate_size

    @property
    def lt(self) -> float:
        return self.cfg.long_term_strength

    # -- episode bookkeeping ----------------------------------------------
    def initial_state(self, user_id: int, rng: np.random.Generator | None = None) -> SimState:
        rng = rng if rng is not None else np.random.default_rng(self.cfg.seed + user_id)
        h = self.user_pref[user_id] + 0.3 * rng.normal(size=self.cfg.dim)
        h = h / np.linalg.norm(h)
        sat = np.zeros(self.catalog.n_coarse, dtype=np.float64)
        return SimState(h=h, sat=sat, t=0)

    # -- reward / transition ----------------------------------------------
    def reward(
        self,
        h: np.ndarray,
        sat: np.ndarray,
        slate: np.ndarray,
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> float:
        slate = validate_slates(slate, self.n_items, self.slate_size)
        r = float(self.reward_batch(h, sat, slate[None])[0])
        if noise > 0.0:
            rng = rng if rng is not None else self.float_rng
            r += float(rng.normal(0.0, noise))
        return r

    def reward_batch(
        self, h: np.ndarray, sat: np.ndarray, slates: np.ndarray
    ) -> np.ndarray:
        """Noiseless reward for a batch.  h [B, dim], sat [B, C], slates [B, K] -> [B].

        A single state is broadcast, so ``h [dim]`` with ``slates [M, K]`` also
        works and returns ``[M]``.

        Composition::

            immediate  = sum_k w_k <h, z_k>          (separable over positions)
            satiation  = sum_k w_k sat[c_k]          (delayed, category-level)
            diversity  = sum_{j<k} w_j w_k <z_j, z_k>  (immediate, COMBINATORIAL)

        The diversity term is what makes the slate reward non-separable: no
        per-position greedy can optimise it, which is the regime where a
        coarse-to-fine action space can earn its keep.
        """
        slates = np.atleast_2d(validate_slates(slates, self.n_items, self.slate_size))
        if h.ndim == 1:
            h = np.broadcast_to(h, (slates.shape[0], h.shape[0]))
        if sat.ndim == 1:
            sat = np.broadcast_to(sat, (slates.shape[0], sat.shape[0]))
        z = self.catalog.item_emb[slates]                       # [B, K, dim]
        base = np.einsum("k,bkd,bd->b", self.pos_w, z, h)
        cats = self.catalog.item_coarse[slates]                 # [B, K]
        sat_g = sat[np.arange(slates.shape[0])[:, None], cats]  # [B, K]
        sat_term = (self.pos_w[None, :] * sat_g).sum(axis=1)
        r = base - self.lt * self.cfg.sat_penalty * sat_term

        if self.cfg.div_penalty > 0.0:
            gram = np.einsum("bjd,bkd->bjk", z, z)              # [B, K, K]
            r = r - self.cfg.div_penalty * (gram * self._pair_w).sum(axis=(1, 2))
        return r

    def diversity_term(self, slates: np.ndarray) -> np.ndarray:
        """The raw intra-slate similarity sum, before the penalty weight.  [B]."""
        slates = np.atleast_2d(validate_slates(slates, self.n_items, self.slate_size))
        z = self.catalog.item_emb[slates]
        gram = np.einsum("bjd,bkd->bjk", z, z)
        return (gram * self._pair_w).sum(axis=(1, 2))

    def step_batch(
        self, h: np.ndarray, sat: np.ndarray, slates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised noiseless transition.

        h [B, dim], sat [B, C], slates [B, K] -> (reward [B], h' [B, dim], sat' [B, C])
        """
        single = h.ndim == 1
        if single:
            h = h[None, :]
            sat = sat[None, :]
        slates = np.atleast_2d(validate_slates(slates, self.n_items, self.slate_size))
        b, k = slates.shape

        r = self.reward_batch(h, sat, slates)

        cats = self.catalog.item_coarse[slates]                  # [B, K]
        expo = np.zeros((b, self.catalog.n_coarse), dtype=np.float64)
        rows = np.repeat(np.arange(b), k)
        np.add.at(expo, (rows, cats.ravel()), np.tile(self.pos_w, b))
        sat_next = self.cfg.sat_decay * sat + (1.0 - self.cfg.sat_decay) * expo

        z = self.catalog.item_emb[slates]                        # [B, K, dim]
        mean_z = np.einsum("k,bkd->bd", self.pos_w, z) / self.pos_w.sum()
        lr = self.lt * self.cfg.pref_lr
        h_next = (1.0 - lr) * h + lr * mean_z
        h_next /= np.maximum(np.linalg.norm(h_next, axis=1, keepdims=True), 1e-12)

        if single:
            return r[0], h_next[0], sat_next[0]
        return r, h_next, sat_next

    def step(
        self,
        state: SimState,
        slate: np.ndarray,
        rng: np.random.Generator | None = None,
        noise: bool = True,
    ) -> tuple[float, SimState]:
        slate = validate_slates(slate, self.n_items, self.slate_size).astype(np.int64)
        r, h_next, sat_next = self.step_batch(state.h, state.sat, slate)
        r = float(r)
        if noise and self.cfg.reward_noise > 0.0:
            rng = rng if rng is not None else self.float_rng
            r += float(rng.normal(0.0, self.cfg.reward_noise))

        nxt = SimState(
            h=h_next,
            sat=sat_next,
            t=state.t + 1,
            slates=state.slates + [slate.copy()],
            rewards=state.rewards + [r],
        )
        return r, nxt

    # -- behaviour policies ------------------------------------------------
    def item_scores(self, h: np.ndarray, sat: np.ndarray) -> np.ndarray:
        """Myopic score of every item, **ignoring slate composition**.

        With ``div_penalty > 0`` this is only a per-position prior: the ranking
        of top-K items is no longer the optimal slate, because assembling the K
        highest-scoring items can pile them into one coarse category and incur
        the diversity penalty.
        """
        s = self.catalog.item_emb @ h
        s = s - self.lt * self.cfg.sat_penalty * sat[self.catalog.item_coarse]
        return s

    def greedy_slate(self, h: np.ndarray, sat: np.ndarray) -> np.ndarray:
        """Top-K items by myopic score (a reactive, non-planning policy)."""
        scores = self.item_scores(h, sat)
        return np.argsort(-scores)[: self.cfg.slate_size].astype(np.int64)

    def behavior_slate(
        self,
        h: np.ndarray,
        sat: np.ndarray,
        rng: np.random.Generator,
        epsilon: float = 0.25,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Logging policy: temperature-sampled greedy with epsilon exploration.

        It is deliberately myopic-ish but not degenerate, which is what makes
        the resulting offline log realistic (and biased in the usual way).
        """
        scores = self.item_scores(h, sat)
        chosen: list[int] = []
        for _ in range(self.cfg.slate_size):
            if rng.random() < epsilon:
                p = np.ones(self.n_items, dtype=np.float64)
            else:
                logits = scores / max(temperature, 1e-6)
                logits = logits - logits.max()
                p = np.exp(logits)
            if chosen:
                p[np.asarray(chosen, dtype=np.int64)] = 0.0
            total = p.sum()
            p = p / total if total > 0 else np.full(self.n_items, 1.0 / self.n_items)
            chosen.append(int(rng.choice(self.n_items, p=p)))
        return np.asarray(chosen, dtype=np.int64)
