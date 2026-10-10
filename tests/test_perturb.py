"""NoisyScoreModel invariants.

This decorator isolates a failure mode that a bad *tree* does not cover: the tree
is perfect, but the prior guiding it is wrong.  So the tests care about three
things -- zero noise is the identity, the corruption is a *fixed* per-class shift
(not per-call jitter, which is a different phenomenon), and the dynamics are
untouched so only the search is affected.

Run directly:
    python tests/test_perturb.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.planning.latent_model import OracleModel  # noqa: E402
from muslate.planning.perturb import NoisyScoreModel  # noqa: E402


def _setup(noise: float, lt: float = 3.0):
    sim = SlateSim(EnvConfig(long_term_strength=lt, div_penalty=0.8))
    inner = OracleModel(sim)
    state = sim.initial_state(0)
    lat = inner.encode(state)[None]
    wrapped = NoisyScoreModel(inner, sim.catalog.item_coarse, sim.n_coarse, noise, seed=0)
    return sim, inner, wrapped, lat


def test_zero_noise_is_the_identity() -> None:
    _sim, inner, wrapped, lat = _setup(0.0)
    assert np.allclose(wrapped.item_scores(lat), inner.item_scores(lat))


def test_bias_is_fixed_across_calls() -> None:
    """Review Focus #3: the prior must be consistently wrong, not jittery.

    If the bias resampled per call, `_coarse_plans` and `_complete` would see
    different scores inside one `propose()` -- internal inconsistency, which is a
    different phenomenon from a systematically wrong prior.
    """
    _sim, _inner, wrapped, lat = _setup(1.0)
    a = wrapped.item_scores(lat)
    b = wrapped.item_scores(lat)
    assert np.array_equal(a, b)


def test_bias_is_constant_within_a_class() -> None:
    sim, inner, wrapped, lat = _setup(1.0)
    base = inner.item_scores(lat)[0]
    got = wrapped.item_scores(lat)[0]
    delta = got - base
    for c in range(sim.n_coarse):
        items = np.where(sim.catalog.item_coarse == c)[0]
        assert np.allclose(delta[items], delta[items[0]]), f"class {c} not shifted uniformly"


def test_bias_shifts_classes_differently() -> None:
    """A degenerate all-equal bonus would move nothing relative to other classes."""
    sim, inner, wrapped, lat = _setup(1.5)
    delta = wrapped.item_scores(lat)[0] - inner.item_scores(lat)[0]
    per_class = np.array([delta[sim.catalog.item_coarse == c][0] for c in range(sim.n_coarse)])
    assert per_class.std() > 0, "the corruption must differentiate classes"


def test_reward_and_step_delegate_unchanged() -> None:
    sim, inner, wrapped, lat = _setup(2.0)
    slate = sim.greedy_slate(sim.initial_state(0).h, sim.initial_state(0).sat)
    slates = np.stack([slate])
    assert np.allclose(wrapped.reward(lat, slates), inner.reward(lat, slates))
    r_w, s_w = wrapped.step(lat, slates)
    r_i, s_i = inner.step(lat, slates)
    assert np.allclose(r_w, r_i) and np.allclose(s_w, s_i)


def test_state_dim_and_item_counts_are_mirrored() -> None:
    _sim, inner, wrapped, _lat = _setup(1.0)
    assert wrapped.state_dim == inner.state_dim
    assert wrapped.n_items == inner.n_items
    assert wrapped.slate_size == inner.slate_size


# --------------------------------------------------------------------------
# Review finding B1: the uniform proposal pool NEVER reads item_scores, so it is
# immune to this decorator.  That makes `max(flat_uniform, flat_policy)` a useless
# baseline when the *prior* is the varied axis -- with flat_uniform constant, the
# derived column is just `sid_hier` shifted by a constant.  Pinned here so the
# trap cannot be walked into again.
# --------------------------------------------------------------------------
def test_uniform_pool_is_immune_to_the_score_corruption() -> None:
    from muslate.planning.proposal import ProposalPolicy

    sim, inner, wrapped, lat = _setup(5.0)
    clean = ProposalPolicy(inner, n_candidates=32, seed=1, mode="uniform")
    dirty = ProposalPolicy(wrapped, n_candidates=32, seed=1, mode="uniform")
    assert np.array_equal(clean.propose(lat), dirty.propose(lat)), (
        "uniform sampling must ignore item_scores entirely"
    )


def test_score_driven_pool_does_change() -> None:
    """The contrast that makes the immune baseline a real problem."""
    from muslate.planning.proposal import ProposalPolicy

    sim, inner, wrapped, lat = _setup(5.0)
    clean = ProposalPolicy(inner, n_candidates=32, seed=1, mode="item_perturb")
    dirty = ProposalPolicy(wrapped, n_candidates=32, seed=1, mode="item_perturb")
    assert not np.array_equal(clean.propose(lat), dirty.propose(lat)), (
        "a score-driven pool must react to the corrupted prior"
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
