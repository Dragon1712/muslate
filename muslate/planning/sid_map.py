"""The Semantic-ID tree as an explicit object.

Why this exists.  In the synthetic catalogue the search tree's hierarchy is the
same `(c1, c2)` that generated the item embeddings: the environment's geometry
and the planner's abstraction come from one generator.  So Gate 3's finding that
`sid_hier` beats flat search under ground-truth dynamics may be partly an artifact
of handing the planner the *correct* tree rather than a property of hierarchical
search.

Making the tree a parameter lets the sweep separate those.  Four kinds:

``oracle``   the catalogue's own `(c1, c2)` -- the ceiling
``noisy``    oracle labels with a fraction `p` re-labelled, class sizes intact
``random``   the oracle label multiset permuted -- a tree with no relation to
             item geometry, but the same class sizes so `fine_per_pos` keeps its
             meaning
``learned``  k-means over a supplied embedding -- the data-driven tree a real
             system would have to use

``purity`` is the same measure for all four, so the sweep's x-axis is comparable:
the fraction of items whose class is the majority class of their cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cluster_purity(labels: np.ndarray, reference: np.ndarray) -> float:
    """Weighted majority-class fraction: 1.0 for the oracle tree, ~1/K for random."""
    labels = np.asarray(labels, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    correct = 0
    total = 0
    for c in np.unique(labels):
        members = reference[labels == c]
        if members.size == 0:
            continue
        correct += int(np.bincount(members).max())
        total += int(members.size)
    return float(correct / total) if total else 0.0


@dataclass
class SIDMap:
    """An item -> coarse-class assignment plus the reverse index the planner needs."""

    coarse_of: np.ndarray                  # [n_items] int64
    n_coarse: int
    items_by_coarse: list[np.ndarray]      # coarse id -> item ids
    purity: float = 1.0
    kind: str = "uniform"

    @classmethod
    def from_labels(cls, labels: np.ndarray, kind: str = "custom", purity: float | None = None,
                    reference: np.ndarray | None = None) -> "SIDMap":
        """Build a map from arbitrary labels, compacting away empty classes."""
        labels = np.asarray(labels, dtype=np.int64)
        present = np.unique(labels)
        remap = {int(c): i for i, c in enumerate(present)}
        compact = np.asarray([remap[int(c)] for c in labels], dtype=np.int64)
        n_coarse = len(present)
        items_by_coarse = [
            np.where(compact == c)[0].astype(np.int64) for c in range(n_coarse)
        ]
        if purity is None:
            purity = (
                cluster_purity(compact, reference)
                if reference is not None
                else 1.0
            )
        return cls(
            coarse_of=compact,
            n_coarse=n_coarse,
            items_by_coarse=items_by_coarse,
            purity=float(purity),
            kind=kind,
        )

    @classmethod
    def uniform(cls, n_items: int, n_coarse: int, n_fine: int) -> "SIDMap":
        """The legacy `item // n_fine` layout, kept for backward compatibility.

        Rejects inconsistent dimensions rather than truncating: a silent shortfall
        drops tail items out of every class (Review finding N1), which breaks the
        partition invariant `_complete` relies on without any visible error.
        """
        if n_coarse * n_fine != n_items:
            raise ValueError(
                f"uniform needs n_coarse * n_fine == n_items, got "
                f"{n_coarse} * {n_fine} = {n_coarse * n_fine} != {n_items}"
            )
        labels = np.repeat(np.arange(n_coarse), n_fine)
        return cls.from_labels(labels, kind="uniform", purity=1.0)


def _kmeans_labels(embedding: np.ndarray, n_clusters: int, seed: int, iters: int = 50) -> np.ndarray:
    """Small Lloyd's algorithm; no sklearn dependency."""
    x = np.asarray(embedding, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # k-means++ style seeding: first centre random, rest biased to be far apart
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, min(n_clusters, len(x))):
        d = np.min(
            np.stack([np.linalg.norm(x - c, axis=1) ** 2 for c in centres]), axis=0
        )
        total = d.sum()
        probs = d / total if total > 0 else np.full(len(x), 1.0 / len(x))
        centres.append(x[rng.choice(len(x), p=probs)])
    centres = np.stack(centres)

    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(iters):
        dist = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=-1)
        new = np.argmin(dist, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(len(centres)):
            members = x[labels == c]
            if len(members) > 0:
                centres[c] = members.mean(axis=0)
    return labels


def build_sid_map(
    n_items: int,
    item_coarse: np.ndarray,
    kind: str,
    p: float = 0.0,
    seed: int = 0,
    embedding: np.ndarray | None = None,
) -> SIDMap:
    """Build one of the four tree kinds.  See the module docstring."""
    item_coarse = np.asarray(item_coarse, dtype=np.int64)
    rng = np.random.default_rng(seed)

    if kind == "oracle":
        return SIDMap.from_labels(item_coarse, kind="oracle", reference=item_coarse)

    if kind == "random":
        # permute the label multiset: no relation to geometry, sizes preserved
        shuffled = rng.permutation(item_coarse)
        return SIDMap.from_labels(shuffled, kind="random", reference=item_coarse)

    if kind == "noisy":
        labels = item_coarse.copy()
        if p > 0:
            mask = rng.random(n_items) < p
            idx = np.where(mask)[0]
            if idx.size > 0:
                # re-permute the selected labels among themselves, so the class-size
                # multiset is untouched and `fine_per_pos` keeps its meaning
                labels[idx] = rng.permutation(labels[idx])
        return SIDMap.from_labels(labels, kind="noisy", reference=item_coarse)

    if kind == "learned":
        if embedding is None:
            raise ValueError("kind='learned' requires an embedding")
        n_clusters = int(item_coarse.max()) + 1
        labels = _kmeans_labels(np.asarray(embedding, dtype=np.float64), n_clusters, seed)
        return SIDMap.from_labels(labels, kind="learned", reference=item_coarse)

    raise ValueError(f"unknown SID map kind {kind!r}")
