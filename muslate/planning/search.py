"""Planners.

All planners share one interface:

    plan(states [B, D]) -> slates [B, K]

They differ only in *how* they search, which is exactly what the gate
experiments need to hold constant or vary:

* ``ReactiveGreedy``   no search, myopic top-K items (the do-nothing baseline)
* ``Lookahead``        depth-limited best-first search over the proposal pool;
                      ``depth = H`` is the knob Gate 2 scans
* ``RandomShooting``   sample whole trajectories and keep the best first action
* ``MCTS``             PUCT tree search (added for the Q3 planner comparison)

The optional ``adjust`` hook receives ``(states, slates, rewards)`` and returns
adjusted rewards.  Conservative planning (uncertainty / OOD penalties) plugs in
there without touching the search logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from muslate.planning.latent_model import LatentModel
from muslate.planning.proposal import ProposalPolicy
from muslate.planning.validity import validate_slates,distinct_indices
from muslate.planning.cost import planning_decision,timed_proposal

AdjustFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


class BasePlanner(ABC):
    def __init__(self, model: LatentModel, gamma: float = 0.95, adjust: AdjustFn | None = None):
        self.model = model
        self.gamma = gamma
        self.adjust = adjust

    @abstractmethod
    def plan(self, states: np.ndarray, remaining_steps=None, budget=None) -> np.ndarray:
        """states [B, D] -> slates [B, K]."""

    def _effective_depth(self, configured, remaining_steps):
        if remaining_steps is None:return configured
        if isinstance(remaining_steps,(bool,np.bool_)) or not isinstance(remaining_steps,(int,np.integer)) or remaining_steps<1:
            raise ValueError('remaining_steps must be a positive integer')
        return min(configured,int(remaining_steps)-1)

    # -- shared helper -----------------------------------------------------
    def _expand(self, states: np.ndarray, cands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """states [B, D], cands [B, M, K] -> (rewards [B, M], next states [B, M, D])."""
        b, m, k = cands.shape
        validate_slates(cands,self.model.n_items,self.model.slate_size)
        rep = np.repeat(states, m, axis=0)
        flat = cands.reshape(-1, k)
        r, nxt = self.model.step(rep, flat)
        if self.adjust is not None:
            r = self.adjust(rep, flat, r)
        return r.reshape(b, m), nxt.reshape(b, m, -1)


class ReactiveGreedy(BasePlanner):
    """Myopic top-K over the model's per-item score.  Zero lookahead."""

    @planning_decision
    def plan(self, states: np.ndarray, remaining_steps=None, budget=None) -> np.ndarray:
        self._effective_depth(0,remaining_steps)
        states = np.atleast_2d(states)
        scores = self.model.item_scores(states)
        return np.argsort(-scores, axis=1)[:, : self.model.slate_size]


class Lookahead(BasePlanner):
    """Depth-limited best-first search over the proposal pool.

    ``depth = 0``  myopic: ``argmax_a r(s, a)`` over the pool
    ``depth = H``  ``argmax_a [ r(s, a) + gamma * V_{H-1}(s') ]``

    where ``V_h`` is itself approximated by the same beam-limited recursion.
    All ``H`` share the same pool size ``M`` and beam width, so the *only*
    thing that changes across the curve is the planning horizon.
    """

    def __init__(
        self,
        model: LatentModel,
        proposal: ProposalPolicy,
        depth: int = 0,
        beam_width: int = 3,
        gamma: float = 0.95,
        adjust: AdjustFn | None = None,
    ):
        super().__init__(model, gamma, adjust)
        self.proposal = proposal
        self.depth = depth
        self.beam_width = beam_width

    @planning_decision
    def plan(self, states: np.ndarray, remaining_steps=None, budget=None) -> np.ndarray:
        depth=self._effective_depth(self.depth,remaining_steps)
        states = np.atleast_2d(states)
        b = states.shape[0]
        cands = self.proposal.propose(states)              # [B, M, K]
        r, nxt = self._expand(states, cands)               # [B, M], [B, M, D]

        if depth <= 0:
            idx = np.argmax(r, axis=1)
            return cands[np.arange(b), idx]

        bw = min(self.beam_width, cands.shape[1])
        top = np.argsort(-r, axis=1)[:, :bw]                          # [B, bw]
        sel_r = np.take_along_axis(r, top, axis=1)
        sel_nxt = np.take_along_axis(nxt, top[:, :, None], axis=1)   # [B, bw, D]
        child = self._value(
            sel_nxt.reshape(-1, sel_nxt.shape[-1]), depth - 1
        ).reshape(b, bw)
        vals = sel_r + self.gamma * child
        pick = np.take_along_axis(top, np.argmax(vals, axis=1)[:, None], axis=1)[:, 0]
        return cands[np.arange(b), pick]

    def _value(self, states: np.ndarray, h: int) -> np.ndarray:
        """Best discounted return reachable within an ``h``-step lookahead."""
        b = states.shape[0]
        cands = self.proposal.propose(states)
        r, nxt = self._expand(states, cands)
        if h <= 0:
            return r.max(axis=1)
        bw = min(self.beam_width, cands.shape[1])
        top = np.argsort(-r, axis=1)[:, :bw]
        sel_r = np.take_along_axis(r, top, axis=1)
        sel_nxt = np.take_along_axis(nxt, top[:, :, None], axis=1)
        child = self._value(
            sel_nxt.reshape(-1, sel_nxt.shape[-1]), h - 1
        ).reshape(b, bw)
        return (sel_r + self.gamma * child).max(axis=1)


class RandomShooting(BasePlanner):
    """Sample ``n_samples`` whole trajectories, keep the best first action.

    This is the "is a tree even needed?" baseline for Q3.
    """

    def __init__(
        self,
        model: LatentModel,
        proposal: ProposalPolicy,
        depth: int = 0,
        n_samples: int = 32,
        gamma: float = 0.95,
        seed: int = 0,
        adjust: AdjustFn | None = None,
    ):
        super().__init__(model, gamma, adjust)
        self.proposal = proposal
        self.depth = depth
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    @planning_decision
    def plan(self, states: np.ndarray, remaining_steps=None, budget=None) -> np.ndarray:
        depth=self._effective_depth(self.depth,remaining_steps)
        states = np.atleast_2d(states)
        b = states.shape[0]
        n = self.n_samples
        cur = np.repeat(states, n, axis=0)                       # [B*N, D]
        total = np.zeros(b * n, dtype=np.float64)
        first = np.empty((b * n, self.model.slate_size), dtype=np.int64)
        disc = 1.0

        for d in range(depth + 1):
            cands = self.proposal.propose(cur)                   # [B*N, M, K]
            pick = self.rng.integers(0, cands.shape[1], size=b * n)
            act = cands[np.arange(b * n), pick]
            if d == 0:
                first = act
            r, nxt = self.model.step(cur, act)
            if self.adjust is not None:
                r = self.adjust(cur, act, r)
            total += disc * r
            disc *= self.gamma
            cur = nxt

        total = total.reshape(b, n)
        first = first.reshape(b, n, -1)
        idx = np.argmax(total, axis=1)
        return first[np.arange(b), idx]


class _MCTSNode:
    """A search node over slate actions.

    ``latent`` is the state *before* choosing an action, ``child_reward[a]`` and
    ``child_latent[a]`` are the cached results of taking action ``a``.
    """

    __slots__ = (
        "latent",
        "depth",
        "actions",
        "P",
        "N",
        "W",
        "children",
        "child_reward",
        "child_latent",
        "expanded",
    )

    def __init__(self, latent: np.ndarray, depth: int = 0):
        self.latent = latent
        self.depth = depth
        self.actions: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.N: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.child_reward: np.ndarray | None = None
        self.child_latent: np.ndarray | None = None
        self.children: dict[int, "_MCTSNode"] = {}
        self.expanded = False


class MCTS(BasePlanner):
    """PUCT tree search over the proposal pool.

    Design points that make it comparable to :class:`Lookahead`:

    * the leaf value is the best immediate reward available at that leaf, i.e.
      exactly the bootstrap :class:`Lookahead` uses at its own horizon.  Neither
      planner has a learned value head quietly doing the work, so ``max_depth``
      really is the knob under study.  (A hard zero leaf value would be
      *unfair* to the tree: it systematically under-rates shallow paths and
      biases exploration toward depth, which is not what a tree is for.)
    * the prior ``P`` is the softmax of the model's immediate reward over the
      pool, i.e. the same information ``Lookahead(depth=0)`` uses;
    * Q is min-max normalised per node and unvisited actions are initialised
      optimistically, so the exploration term is not dominated by the arbitrary
      scale of the reward.

    The planner answers Q3's "is a tree even needed?": it is given the *same*
    proposal pool and the *same* depth as the beam, and the runner reports the
    *actual* number of model evaluations each one spends.
    """

    def __init__(
        self,
        model: LatentModel,
        proposal: ProposalPolicy,
        max_depth: int = 3,
        n_sims: int = 64,
        c_puct: float = 1.5,
        prior_temp: float = 1.0,
        bootstrap: bool = True,
        gamma: float = 0.95,
        seed: int = 0,
        adjust: AdjustFn | None = None,
    ):
        super().__init__(model, gamma, adjust)
        self.proposal = proposal
        self.max_depth = max_depth
        self.n_sims = n_sims
        self.c_puct = c_puct
        self.prior_temp = prior_temp
        self.bootstrap = bootstrap
        self.rng = np.random.default_rng(seed)

    # -- public -------------------------------------------------------------
    @planning_decision
    def plan(self, states: np.ndarray, remaining_steps=None, budget=None) -> np.ndarray:
        depth=self._effective_depth(self.max_depth,remaining_steps)
        states = np.atleast_2d(states)
        out = np.empty((states.shape[0], self.model.slate_size), dtype=np.int64)
        for b in range(states.shape[0]):
            out[b] = self._plan_one(states[b],depth)
        return out

    # -- internals ----------------------------------------------------------
    def _plan_one(self, latent: np.ndarray, depth=None) -> np.ndarray:
        depth=self.max_depth if depth is None else depth
        root = _MCTSNode(latent, 0)
        self._expand(root)
        if root.actions is None or root.actions.shape[0] == 0:
            raise RuntimeError("MCTS root expanded to an empty action set")
        for _ in range(self.n_sims):
            self._simulate(root,depth)

        visits = root.N
        if visits.sum() == 0:
            # nothing was backed up (e.g. max_depth == 0); fall back to the prior,
            # which is the softmax of the immediate reward
            return root.actions[int(np.argmax(root.P))]
        q = np.where(visits > 0, root.W / np.maximum(visits, 1.0), -np.inf)
        order = np.lexsort((-q, -visits))       # most-visited, then best-Q
        return root.actions[order[0]]

    def _expand(self, node: _MCTSNode) -> None:
        cands = self.proposal.propose(node.latent[None])[0]        # [M, K]
        m = cands.shape[0]
        rep = np.repeat(node.latent[None], m, axis=0)
        r, nxt = self.model.step(rep, cands)
        if self.adjust is not None:
            r = self.adjust(rep, cands, r)

        z = r / max(self.prior_temp, 1e-6)
        z = z - z.max()
        p = np.exp(z)
        p = p / p.sum()

        node.actions = cands
        node.child_reward = r
        node.child_latent = nxt
        node.P = p
        node.N = np.zeros(m, dtype=np.float64)
        node.W = np.zeros(m, dtype=np.float64)
        node.expanded = True

    def _select(self, node: _MCTSNode) -> int:
        n, w, p = node.N, node.W, node.P
        total = float(n.sum())
        visited = n > 0

        q = np.where(visited, w / np.maximum(n, 1.0), 0.0)
        if visited.any():
            qv = q[visited]
            lo, hi = float(qv.min()), float(qv.max())
            qn = np.where(visited, (q - lo) / max(hi - lo, 1e-8), 1.0)
        else:
            qn = np.ones_like(q)

        u = self.c_puct * p * np.sqrt(total + 1.0) / (1.0 + n)
        return int(np.argmax(qn + u))

    def _simulate(self, root: _MCTSNode, depth=None) -> None:
        depth=self.max_depth if depth is None else depth
        node = root
        path: list[tuple[_MCTSNode, int]] = []

        while node.expanded and node.depth < depth:
            a = self._select(node)
            path.append((node, a))
            child = node.children.get(a)
            if child is None:
                child = _MCTSNode(node.child_latent[a], node.depth + 1)
                node.children[a] = child
            node = child

        # Bootstrap the leaf with the best immediate reward available there, the
        # same way Lookahead evaluates its horizon.  Using a hard zero instead
        # would systematically under-rate shallow paths, which biases the tree
        # toward depth rather than toward good decisions.  ``bootstrap=False``
        # restores that (worse) behaviour so the effect can be ablated.
        leaf_value = 0.0
        if not node.expanded and node.depth <= depth:
            self._expand(node)
        if self.bootstrap and node.expanded and node.depth <= depth:
            leaf_value = float(node.child_reward.max())

        value = leaf_value
        for nd, a in reversed(path):
            value = nd.child_reward[a] + self.gamma * value
            nd.N[a] += 1.0
            nd.W[a] += value


class CEM(BasePlanner):
    """Cross-entropy method over a per-position item distribution.

    Maintains ``K`` categorical distributions ``P[k, .]`` over items, samples
    candidate slates from them, keeps the elite, and refits the distributions
    toward the elite's per-position marginals.  Its value estimate is *identical*
    to :class:`RandomShooting`'s, so the two differ only in how candidates are
    proposed -- random versus adaptive.
    """

    def __init__(
        self,
        model: LatentModel,
        proposal: ProposalPolicy,
        depth: int = 0,
        n_samples: int = 32,
        n_elite: int = 8,
        n_iters: int = 3,
        alpha: float = 0.5,
        temperature: float = 1.0,
        gamma: float = 0.95,
        seed: int = 0,
        adjust: AdjustFn | None = None,
    ):
        super().__init__(model, gamma, adjust)
        self.proposal = proposal
        self.depth = depth
        self.n_samples = n_samples
        self.n_elite = n_elite
        self.n_iters = n_iters
        self.alpha = alpha
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    @planning_decision
    def plan(self, states: np.ndarray, remaining_steps=None, budget=None) -> np.ndarray:
        depth=self._effective_depth(self.depth,remaining_steps)
        states = np.atleast_2d(states)
        out = np.empty((states.shape[0], self.model.slate_size), dtype=np.int64)
        for b in range(states.shape[0]):
            out[b] = self._plan_one(states[b],depth)
        return out

    # -- internals ----------------------------------------------------------
    @timed_proposal
    def _sample(self, logits: np.ndarray, m: int) -> np.ndarray:
        """Sample ``m`` slates without repeating an item inside a slate.

        The no-repeat constraint matters: a duplicated item wastes a position and
        is strictly worse under any reasonable slate reward, so letting CEM emit
        them would handicap it against the flat pools, which never do.
        """
        k = logits.shape[0]
        if hasattr(self.model,'reserve_proposals'):self.model.reserve_proposals(m)
        out = np.empty((m, k), dtype=np.int64)
        for i in range(m):
            taken: list[int] = []
            for pos in range(k):
                z = logits[pos].copy()
                if taken:
                    z[np.asarray(taken, dtype=np.int64)] = -np.inf
                z = z / max(self.temperature, 1e-6)
                z = z - z.max()
                p = np.exp(z)
                total = p.sum()
                if not np.isfinite(total) or total <= 0:
                    p = np.ones_like(z)
                    if taken:
                        p[np.asarray(taken, dtype=np.int64)] = 0.0
                    total = p.sum()
                p = p / total
                item = int(self.rng.choice(self.model.n_items, p=p))
                out[i, pos] = item
                taken.append(item)
        return out[distinct_indices(out)]

    def _value(self, latent: np.ndarray, cands: np.ndarray, depth=None) -> np.ndarray:
        depth=self.depth if depth is None else depth
        """Discounted return of each candidate, with random continuation."""
        b = cands.shape[0]
        rep = np.repeat(latent[None], b, axis=0)
        r, cur = self.model.step(rep, cands)
        if self.adjust is not None:
            r = self.adjust(rep, cands, r)
        total = r.copy()
        disc = self.gamma

        for _ in range(depth):
            pool = self.proposal.propose(cur)                     # [B, M, K]
            pick = self.rng.integers(0, pool.shape[1], size=cur.shape[0])
            act = pool[np.arange(cur.shape[0]), pick]
            rr, nxt = self.model.step(cur, act)
            if self.adjust is not None:
                rr = self.adjust(cur, act, rr)
            total = total + disc * rr
            disc *= self.gamma
            cur = nxt
        return total

    def _plan_one(self, latent: np.ndarray, depth=None) -> np.ndarray:
        k = self.model.slate_size
        scores = self.model.item_scores(latent[None])[0]          # [N]
        logits = np.repeat(scores[None, :], k, axis=0)            # [K, N]

        best_act = None
        best_val = -np.inf
        for _ in range(self.n_iters):
            cands = self._sample(logits, self.n_samples)
            vals = self._value(latent, cands,depth)
            order = np.argsort(-vals)
            elite = cands[order[: self.n_elite]]

            for pos in range(k):
                counts = np.bincount(elite[:, pos], minlength=self.model.n_items).astype(np.float64)
                counts += 1e-3
                logits[pos] = (1.0 - self.alpha) * logits[pos] + self.alpha * np.log(counts)

            if vals[order[0]] > best_val:
                best_val = float(vals[order[0]])
                best_act = cands[order[0]]

        return best_act


def make_planner(kind: str, model: LatentModel, proposal: ProposalPolicy, cfg) -> BasePlanner:
    """Factory used by the experiment runners."""
    kind = kind.lower()
    if kind == "reactive":
        return ReactiveGreedy(model, gamma=cfg.gamma)
    if kind == "lookahead":
        return Lookahead(
            model, proposal, depth=cfg.max_depth, beam_width=getattr(cfg, "beam_width", 3), gamma=cfg.gamma
        )
    if kind == "shooting":
        return RandomShooting(
            model,
            proposal,
            depth=cfg.max_depth,
            n_samples=getattr(cfg, "n_samples", 32),
            gamma=cfg.gamma,
            seed=cfg.seed,
        )
    if kind == "mcts":
        return MCTS(
            model,
            proposal,
            max_depth=cfg.max_depth,
            n_sims=getattr(cfg, "n_sims", 64),
            c_puct=getattr(cfg, "c_puct", 1.5),
            gamma=cfg.gamma,
            seed=cfg.seed,
        )
    if kind == "cem":
        return CEM(
            model,
            proposal,
            depth=cfg.max_depth,
            n_samples=getattr(cfg, "n_samples", 32),
            n_elite=getattr(cfg, "n_elite", 8),
            n_iters=getattr(cfg, "n_iters", 3),
            gamma=cfg.gamma,
            seed=cfg.seed,
        )
    raise ValueError(f"unknown planner {kind!r}")
