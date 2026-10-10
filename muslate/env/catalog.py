"""Item catalogue with a two-level Semantic-ID hierarchy.

Item id ``i`` maps to ``(coarse, fine) = (i // n_fine, i % n_fine)``.  The
embedding of an item is a normalised mixture of a coarse-level prototype and a
fine-level residual, so items sharing a coarse id are genuinely similar.  This
is what makes a coarse-to-fine search tree a sensible action abstraction.
"""

from __future__ import annotations

import numpy as np

from muslate.config import CatalogConfig


class Catalog:
    def __init__(self, cfg: CatalogConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        coarse = rng.normal(size=(cfg.n_coarse, cfg.dim))
        coarse /= np.linalg.norm(coarse, axis=1, keepdims=True)
        fine = rng.normal(size=(cfg.n_fine, cfg.dim))
        fine /= np.linalg.norm(fine, axis=1, keepdims=True)

        self.coarse_emb = coarse.astype(np.float64)
        self.fine_emb = fine.astype(np.float64)

        ids = np.arange(cfg.n_items)
        self.item_coarse = (ids // cfg.n_fine).astype(np.int64)
        self.item_fine = (ids % cfg.n_fine).astype(np.int64)

        emb = (
            cfg.coarse_weight * coarse[self.item_coarse]
            + cfg.fine_weight * fine[self.item_fine]
        )
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        self.item_emb = emb.astype(np.float64)

    # -- convenience ------------------------------------------------------
    @property
    def n_items(self) -> int:
        return self.cfg.n_items

    @property
    def n_coarse(self) -> int:
        return self.cfg.n_coarse

    @property
    def n_fine(self) -> int:
        return self.cfg.n_fine

    @property
    def dim(self) -> int:
        return self.cfg.dim
