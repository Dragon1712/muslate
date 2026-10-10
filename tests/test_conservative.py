"""Conservative-planning invariants.

The penalty terms are only useful if they are *discriminative* and *pointed the
right way*.  These tests pin both down, because "the penalty does nothing" and
"the penalty is inverted" look identical in a final return number.

Run directly:
    python tests/test_conservative.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig, LogConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.env.logger import generate_log  # noqa: E402
from muslate.planning.conservative import (  # noqa: E402
    AdaptiveHorizon,
    BehaviorSupport,
    ConservativePenalty,
    LatentSupport,
)
from muslate.planning.controlled import BiasedOracleModel  # noqa: E402
from muslate.planning.latent_model import OracleModel  # noqa: E402
from muslate.planning.proposal import ProposalPolicy  # noqa: E402


def _sim_log(lt: float = 3.0, div: float = 0.8):
    sim = SlateSim(EnvConfig(long_term_strength=lt, div_penalty=div))
    log = generate_log(sim, LogConfig(n_episodes=40, seed=3))
    return sim, log


# ---------------------------------------------------------------- support
def test_behavior_support_is_discriminative() -> None:
    """Raw unigram NLL is nearly constant; the z-scoring must give it spread."""
    _sim, log = _sim_log()
    support = BehaviorSupport(log)

    counts = np.bincount(log.slates.ravel(), minlength=log.n_items)
    rare = np.argsort(counts)[: log.slate_size]
    common = np.argsort(-counts)[: log.slate_size]
    pool = np.stack([rare, common])

    scores = support.score(pool)
    assert scores.std() > 0.05, "the OOD score has no spread; it cannot drive a penalty"
    assert scores[0] > scores[1], "rare items must score as more out-of-support"


def test_latent_support_zero_at_reference_and_grows_away() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(128, 8))
    support = LatentSupport(ref)

    assert np.allclose(support.score(ref[:5]), 0.0, atol=1e-6), "a logged point is its own support"
    far = ref[:1] + 10.0
    assert support.score(far)[0] > support.score(ref[:1])[0] + 1.0


# ---------------------------------------------------------------- penalty
def test_zero_penalty_is_identity() -> None:
    sim, log = _sim_log()
    model = OracleModel(sim)
    support = BehaviorSupport(log)
    penalty = ConservativePenalty(model, 0.0, 0.0, support=support)

    states = np.zeros((4, model.state_dim))
    slates = np.stack([sim.greedy_slate(sim.initial_state(0).h, sim.initial_state(0).sat)] * 4)
    rewards = np.array([1.0, -2.0, 3.0, 0.5])
    assert np.allclose(penalty(states, slates, rewards), rewards)


def test_ood_penalty_reduces_rewards_monotonically() -> None:
    sim, log = _sim_log()
    model = OracleModel(sim)
    support = BehaviorSupport(log)

    states = np.zeros((2, model.state_dim))
    counts = np.bincount(log.slates.ravel(), minlength=log.n_items)
    rare = np.argsort(counts)[: sim.slate_size]
    common = np.argsort(-counts)[: sim.slate_size]
    slates = np.stack([rare, common])
    rewards = np.zeros(2)

    weak = ConservativePenalty(model, 0.0, 0.5, support=support)(states, slates, rewards)
    strong = ConservativePenalty(model, 0.0, 2.0, support=support)(states, slates, rewards)
    assert strong[0] < weak[0], "higher beta must penalise the rare slate more"
    assert np.isclose(weak[1], strong[1], atol=1e-6) or strong[1] >= weak[1]


def test_uncertainty_penalty_requires_an_ensemble() -> None:
    sim, _log = _sim_log()
    model = OracleModel(sim)          # no uncertainty()
    try:
        ConservativePenalty(model, lambda_unc=1.0)
    except TypeError:
        return
    raise AssertionError("a model without uncertainty() must be rejected")


# ---------------------------------------------------------------- bias model
def test_biased_model_leaves_the_environment_alone() -> None:
    """The bias must live in the *model*, never in the simulator."""
    sim, _log = _sim_log()
    biased = BiasedOracleModel(sim, bias_strength=3.0, seed=1)
    oracle = OracleModel(sim)

    state = sim.initial_state(0)
    latent = oracle.encode(state)[None]
    slate = sim.greedy_slate(state.h, state.sat)[None]

    r_true = sim.reward_batch(state.h, state.sat, slate)[0]
    assert np.isclose(oracle.reward(latent, slate)[0], r_true)
    assert not np.isclose(biased.reward(latent, slate)[0], r_true), "bias must change the model"
    # dynamics of the underlying simulator are untouched either way
    _r, next_state = sim.step(state, slate[0], noise=False)
    assert np.isclose(next_state.sat.sum(), sim.cfg.sat_decay * state.sat.sum() + (1 - sim.cfg.sat_decay) * sim.pos_w.sum())


def test_biased_model_bias_grows_with_saturation() -> None:
    """The state coupling is what makes the bias exploitable by planning."""
    sim, _log = _sim_log()
    model = BiasedOracleModel(sim, bias_strength=1.0, state_coupling=1.0, seed=0)
    slate = sim.greedy_slate(sim.initial_state(0).h, sim.initial_state(0).sat)[None]

    cold = sim.initial_state(0)
    hot = sim.initial_state(0)
    hot.sat = np.full_like(hot.sat, 3.0)

    b_cold = abs(model.bias(model.encode(cold)[None], slate)[0])
    b_hot = abs(model.bias(model.encode(hot)[None], slate)[0])
    assert b_hot > b_cold, "saturated states must carry a larger bias"


def test_perfect_uncertainty_equals_absolute_bias() -> None:
    sim, _log = _sim_log()
    model = BiasedOracleModel(sim, bias_strength=2.0, seed=0, perfect_uncertainty=True)
    state = sim.initial_state(0)
    latent = model.encode(state)[None]
    slates = ProposalPolicy(model, n_candidates=8, seed=0).propose(latent)[0]

    u = model.uncertainty(np.repeat(latent, 8, axis=0), slates)
    b = np.abs(model.bias(np.repeat(latent, 8, axis=0), slates))
    assert np.allclose(u, b)

    const = BiasedOracleModel(sim, bias_strength=2.0, seed=0, perfect_uncertainty=False)
    assert const.uncertainty(np.repeat(latent, 8, axis=0), slates).std() < 1e-12, "bad proxy is constant"


# ---------------------------------------------------------------- horizon
def test_adaptive_horizon_returns_a_valid_horizon() -> None:
    sim, _log = _sim_log()
    model = BiasedOracleModel(sim, bias_strength=1.0, seed=0)
    proposal = ProposalPolicy(model, n_candidates=16, seed=0)
    gate = AdaptiveHorizon(model, proposal, horizons=(5, 3, 1), u_low=0.0, u_high=1e9)

    latent = model.encode(sim.initial_state(0))[None]
    h = gate.horizon(latent)
    assert h in (5, 3, 1)
    assert gate.state_uncertainty(latent) >= 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
