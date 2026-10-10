"""Shared action contract and stable pool deduplication."""
import numpy as np


def validate_slates(slates, n_items: int, slate_size: int) -> np.ndarray:
    a=np.asarray(slates)
    if a.ndim<1 or a.shape[-1]!=slate_size or a.size==0:
        raise ValueError('invalid slate length or empty batch')
    if not np.issubdtype(a.dtype,np.integer) or np.any(a<0) or np.any(a>=n_items):
        raise ValueError('item indices must be integers in catalog range')
    ordered=np.sort(a,axis=-1)
    if np.any(np.diff(ordered,axis=-1)==0):
        raise ValueError('duplicated item in slate')
    return a


def distinct_indices(slates: np.ndarray) -> np.ndarray:
    _,indices=np.unique(slates,axis=0,return_index=True)
    return np.sort(indices)


def stack_pools(pools: list[np.ndarray]) -> np.ndarray:
    # Batched planner interface is rectangular; trim, never duplicate-pad.
    count=min(len(p) for p in pools)
    if count<1:raise ValueError('empty legal candidate pool')
    return np.stack([p[:count] for p in pools])
