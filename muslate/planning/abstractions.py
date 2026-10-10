"""Action abstractions, expressed as drop-in candidate generators.

Gate 3 asks whether a Semantic-ID hierarchy is worth anything compared with
flattening the problem.  To answer that, the *search* must be held fixed and only
the *candidate generator* varied -- otherwise a difference could come from the
planner rather than from the action space.

Every class here therefore exposes the same tiny interface as
:class:`~muslate.planning.proposal.ProposalPolicy`

    propose(states [B, D]) -> candidates [B, M, K]

so it can be dropped into ``Lookahead``, ``RandomShooting`` or any future planner
unchanged.

``out_size`` and ``proposal_cost()`` describe nominal pool settings. Effective
pools can shrink after deduplication, and recursive planning incurs additional
work. Use CountingModel for disjoint reward/transition/score/proposal counts and
measured latency; matching nominal settings does not match total compute.
"""

from __future__ import annotations

import numpy as np

from muslate.planning.latent_model import LatentModel
from muslate.planning.sid_map import SIDMap
from muslate.planning.validity import validate_slates, distinct_indices, stack_pools
from muslate.planning.cost import timed_proposal


class SIDProposal:
    """Coarse-to-fine generation over a two-level Semantic-ID hierarchy.

    ``refine_positions == 0`` is the flat one-level variant: sample coarse
    category plans and complete each greedily.  ``refine_positions > 0`` adds
    progressive widening -- only the ``widen`` best coarse plans are carried
    forward and refined item-by-item inside their fixed category structure.

    The refinement is what a flat sampler cannot do: it commits to a coarse
    structure first and only then spends budget on detail, instead of spreading
    the whole budget over coarse combinations.
    """

    def __init__(
        self,
        model: LatentModel,
        n_coarse: int | None = None,
        n_fine: int | None = None,
        n_sampled: int = 64,
        out_size: int = 64,
        widen: int = 4,
        refine_positions: int = 0,
        fine_per_pos: int = 8,
        seed: int = 0,
        temperature: float = 0.7,
        sid_map: SIDMap | None = None,
    ):
        self.model = model
        if sid_map is None:
            if n_coarse is None or n_fine is None:
                raise ValueError(
                    "SIDProposal needs either `sid_map`, or both `n_coarse` and `n_fine`"
                )
            sid_map = SIDMap.uniform(model.n_items, n_coarse, n_fine)
        # `_coarse_plans` builds a length-`slate_size` anchor from the class
        # ranking, so fewer classes than that would fail later with an opaque
        # numpy broadcast error (Review finding N2).  Fail early and clearly.
        if sid_map.n_coarse < model.slate_size:
            raise ValueError(
                f"SIDProposal needs at least slate_size={model.slate_size} classes, "
                f"got {sid_map.n_coarse} (kind={sid_map.kind!r})"
            )
        self.sid_map = sid_map
        self.n_coarse = sid_map.n_coarse
        self.n_sampled = n_sampled
        self.out_size = out_size
        self.widen = widen
        self.refine_positions = refine_positions
        self.fine_per_pos = fine_per_pos
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    # -- budget -----------------------------------------------------------
    def proposal_cost(self) -> int:
        refine = self.widen * self.refine_positions * self.fine_per_pos if self.refine_positions else 0
        return self.n_sampled + refine

    def output_size(self) -> int:
        return self.out_size

    # -- helpers ----------------------------------------------------------
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x / max(self.temperature, 1e-6)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _items_in(self, c: int) -> np.ndarray:
        return self.sid_map.items_by_coarse[c]

    def _complete(self, scores: np.ndarray, coarse_plan: np.ndarray) -> np.ndarray:
        """Fill a coarse plan into a concrete slate, greedily, without repeats."""
        slate: list[int] = []
        used: set[int] = set()
        for c in coarse_plan:
            items = self._items_in(int(c))
            order = np.argsort(-scores[items])
            pick = None
            for o in order:
                cand = int(items[o])
                if cand not in used:
                    pick = cand
                    break
            if pick is None:
                raise ValueError('coarse group capacity cannot complete slate without repeats')
            used.add(pick)
            slate.append(pick)
        return validate_slates(np.asarray(slate,dtype=np.int64),self.model.n_items,len(coarse_plan))

    def _coarse_plans(self, coarse_scores: np.ndarray, k: int, n: int) -> np.ndarray:
        plans = np.empty((n, k), dtype=np.int64)
        plans[0] = np.argsort(-coarse_scores)[:k]              # greedy anchor
        for m in range(1, n):
            used=np.zeros(self.n_coarse,dtype=int)
            capacity=np.asarray([len(g) for g in self.sid_map.items_by_coarse])
            for pos in range(k):
                available=np.flatnonzero(used<capacity)
                if not len(available):raise ValueError('insufficient group capacity')
                p=self._softmax(coarse_scores[available])
                c=int(self.rng.choice(available,p=p))
                plans[m,pos]=c;used[c]+=1
        return plans

    # -- search -----------------------------------------------------------
    def _search_one(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """state [1, D] -> (evaluated candidates [E, K], their model rewards [E])."""
        k = self.model.slate_size
        scores = self.model.item_scores(state)[0]                       # [N]
        coarse_scores = np.asarray(
            [scores[g].max() for g in self.sid_map.items_by_coarse], dtype=np.float64
        )

        if hasattr(self.model,'reserve_proposals'):self.model.reserve_proposals(self.n_sampled)
        plans = self._coarse_plans(coarse_scores, k, self.n_sampled)
        slates = np.stack([self._complete(scores, p) for p in plans])    # [S, K]
        slates=slates[distinct_indices(slates)]
        r = self.model.reward(np.repeat(state, slates.shape[0], axis=0), slates)
        cache={tuple(a):float(v) for a,v in zip(slates,r)}

        if self.refine_positions == 0:
            return slates, r

        # -- progressive widening: only the best coarse plans get refined ----
        n_widen = min(self.widen, slates.shape[0])
        keep = np.argsort(-r)[:n_widen]
        best = slates[keep].copy()
        best_r = r[keep].copy()

        extra_slates: list[np.ndarray] = []
        extra_r: list[np.ndarray] = []

        for pos in range(min(self.refine_positions, k)):
            cands, owners = [], []
            for w in range(best.shape[0]):
                c = int(self.sid_map.coarse_of[best[w, pos]])
                items = self._items_in(c)
                order = items[np.argsort(-scores[items])][: self.fine_per_pos]
                for it in order:
                    cand = best[w].copy()
                    if it==cand[pos] or it in np.delete(cand,pos):continue
                    if hasattr(self.model,'reserve_proposals'):self.model.reserve_proposals(1)
                    cand[pos] = it
                    cands.append(cand)
                    owners.append(w)
            if not cands:
                continue
            cands = np.stack(cands)
            missing=[a for a in cands[distinct_indices(cands)] if tuple(a) not in cache]
            if missing:
                fresh=np.stack(missing)
                values=self.model.reward(np.repeat(state,len(fresh),axis=0),fresh)
                cache.update((tuple(a),float(v)) for a,v in zip(fresh,values))
            cr=np.asarray([cache[tuple(a)] for a in cands])
            extra_slates.append(cands)
            extra_r.append(cr)
            for idx, w in enumerate(owners):
                if cr[idx] > best_r[w]:
                    best_r[w] = cr[idx]
                    best[w] = cands[idx]

        all_slates = np.concatenate([slates] + extra_slates, axis=0)
        all_r = np.concatenate([r] + extra_r, axis=0)
        idx=distinct_indices(all_slates)
        return all_slates[idx],all_r[idx]

    @timed_proposal
    def propose(self, states: np.ndarray) -> np.ndarray:
        states = np.atleast_2d(states)
        b = states.shape[0]
        pools=[]
        for i in range(b):
            slates, r = self._search_one(states[i : i + 1])
            n_out = min(self.out_size, slates.shape[0])
            order = np.argsort(-r)[:n_out]
            pools.append(slates[order])
        out=stack_pools(pools)
        return validate_slates(out,self.model.n_items,self.model.slate_size)
