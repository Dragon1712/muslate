"""Environment invariants.

These are the properties Gate 2 relies on, so they must be pinned down:

1. lt = 0 really is a bandit (state never moves, reward never depends on history)
2. the saturation penalty is *delayed* -- step t's reward uses the saturation
   accumulated up to t-1, not t's own exposure.  Without this there is no
   long-horizon dependency and no reason for planning to exist.
3. the batched transition agrees with the step-by-step one
4. the Semantic-ID hierarchy is real: items in the same coarse group are more
   similar than items in different groups

Run directly (no pytest needed):
    python tests/test_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402


def test_lt0_is_a_bandit() -> None:
    sim = SlateSim(EnvConfig(long_term_strength=0.0))
    st = sim.initial_state(0)
    rng = np.random.default_rng(0)

    state = st
    for _ in range(5):
        slate = sim.behavior_slate(state.h, state.sat, rng, epsilon=0.3)
        _, state = sim.step(state, slate, noise=False)

    # the user preference must be frozen
    assert np.allclose(state.h, st.h), "lt=0 must not drift the user preference"

    # saturation may still be tracked internally, but it must not reach the
    # reward, because the penalty term is scaled by lt
    slate = sim.greedy_slate(st.h, st.sat)
    r_first = sim.reward(st.h, st.sat, slate)
    r_again = sim.reward(state.h, state.sat, slate)
    assert np.isclose(r_again, r_first), "lt=0 reward must not depend on history"

    # and the slate that maximises the immediate reward must be greedy's
    pool = np.stack([slate, sim.behavior_slate(st.h, st.sat, rng, epsilon=1.0)])
    r_pool = sim.reward_batch(st.h, st.sat, pool)
    assert r_pool[0] >= r_pool[1]


def test_saturation_penalty_is_delayed() -> None:
    sim = SlateSim(EnvConfig(long_term_strength=1.0))
    st = sim.initial_state(0)
    slate = sim.greedy_slate(st.h, st.sat)

    r1, st1 = sim.step(st, slate, noise=False)
    # step 0's reward must use the *initial* (zero) saturation, not its own exposure
    expected_r1 = sim.reward_batch(st.h, st.sat, slate)[0]
    assert np.isclose(r1, expected_r1)

    # but saturation must have moved for step 1
    assert st1.sat.sum() > 0.0, "exposure must accumulate saturation"

    # and re-showing the same slate must now be worse
    r2, _ = sim.step(st1, slate, noise=False)
    assert r2 < r1, "showing the same thing twice must be penalised by saturation"


def test_batch_matches_loop() -> None:
    sim = SlateSim(EnvConfig(long_term_strength=2.0))
    rng = np.random.default_rng(3)
    h = rng.normal(size=(4, sim.cfg.dim))
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    sat = rng.random(size=(4, sim.n_coarse))
    slates = np.stack([sim.greedy_slate(h[i], sat[i]) for i in range(4)])

    r_batch, h_batch, s_batch = sim.step_batch(h, sat, slates)
    for i in range(4):
        r_i, h_i, s_i = sim.step_batch(h[i], sat[i], slates[i])
        assert np.isclose(r_batch[i], r_i)
        assert np.allclose(h_batch[i], h_i)
        assert np.allclose(s_batch[i], s_i)


def test_sid_hierarchy_is_meaningful() -> None:
    sim = SlateSim(EnvConfig())
    emb = sim.catalog.item_emb
    coarse = sim.catalog.item_coarse

    same, diff = [], []
    for i in range(0, sim.n_items, 7):
        for j in range(1, 5):
            k = i + j
            if k >= sim.n_items:
                continue
            sim_ij = float(emb[i] @ emb[k])
            (same if coarse[i] == coarse[k] else diff).append(sim_ij)

    assert np.mean(same) > np.mean(diff), "same-coarse items should be more similar"


def _spread_slate(sim: SlateSim, h, sat, k: int | None = None) -> np.ndarray:
    """Pick the best item from each of the k best coarse categories."""
    k = k or sim.cfg.slate_size
    scores = sim.item_scores(h, sat)
    per_cat = scores.reshape(sim.n_coarse, sim.catalog.n_fine)
    cat_order = np.argsort(-per_cat.max(axis=1))[:k]
    return np.asarray(
        [int(c * sim.catalog.n_fine + np.argmax(per_cat[c])) for c in cat_order], dtype=np.int64
    )


def test_diversity_penalty_is_zero_by_default() -> None:
    sim = SlateSim(EnvConfig(div_penalty=0.0))
    st = sim.initial_state(0)
    slate = sim.greedy_slate(st.h, st.sat)
    assert np.isclose(sim.reward_batch(st.h, st.sat, slate)[0], sim.reward(st.h, st.sat, slate))


def test_diversity_breaks_separability() -> None:
    """The point of the term: per-position greedy stops being optimal.

    Without it, the K highest-scoring items form (near-)optimal slate.  With it,
    a category-spread slate can beat the greedy pile-up.
    """
    st = SlateSim(EnvConfig()).initial_state(0)

    flat_sim = SlateSim(EnvConfig(div_penalty=0.0))
    div_sim = SlateSim(EnvConfig(div_penalty=0.8))

    greedy = flat_sim.greedy_slate(st.h, st.sat)
    spread = _spread_slate(flat_sim, st.h, st.sat)

    # sanity: the two slates differ (otherwise the test proves nothing)
    assert not np.array_equal(greedy, spread), "greedy must actually pile items into one category"
    assert len(set(spread // flat_sim.catalog.n_fine)) == len(spread)

    r_greedy_flat = flat_sim.reward_batch(st.h, st.sat, greedy)[0]
    r_spread_flat = flat_sim.reward_batch(st.h, st.sat, spread)[0]
    assert r_greedy_flat >= r_spread_flat, "without diversity, greedy should win"

    r_greedy_div = div_sim.reward_batch(st.h, st.sat, greedy)[0]
    r_spread_div = div_sim.reward_batch(st.h, st.sat, spread)[0]
    assert r_spread_div > r_greedy_div, "with diversity, the spread slate must win"


def test_diversity_penalises_same_category_slates() -> None:
    sim = SlateSim(EnvConfig(div_penalty=1.0))
    st = sim.initial_state(0)
    same_cat = np.arange(sim.cfg.slate_size, dtype=np.int64)              # first K items, one category
    spread = _spread_slate(sim, st.h, st.sat)
    assert sim.diversity_term(same_cat[None])[0] > sim.diversity_term(spread[None])[0]


def test_diversity_is_immediate_not_delayed() -> None:
    """The term is part of the reward at t; it must not leak into the state update."""
    a = SlateSim(EnvConfig(div_penalty=0.0, long_term_strength=0.0))
    b = SlateSim(EnvConfig(div_penalty=0.9, long_term_strength=0.0))
    st = a.initial_state(0)
    slate = a.greedy_slate(st.h, st.sat)

    r_a, st_a = a.step(st, slate, noise=False)
    r_b, st_b = b.step(st, slate, noise=False)

    assert r_b <= r_a, "adding diversity can only lower this slate's reward"
    assert np.allclose(st_a.h, st_b.h), "divergence must not change the state transition"
    assert np.allclose(st_a.sat, st_b.sat)


def test_horizon_and_gamma() -> None:
    sim = SlateSim(EnvConfig(horizon=7))
    state = sim.initial_state(0)
    n = 0
    rng = np.random.default_rng(1)
    while state.t < sim.cfg.horizon:
        slate = sim.behavior_slate(state.h, state.sat, rng)
        _, state = sim.step(state, slate, noise=False)
        n += 1
    assert n == 7


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
