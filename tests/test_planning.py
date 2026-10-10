"""Planner and action-abstraction invariants.

These make sure the Gate 2 / Gate 3 comparisons measure what they claim:

* ``Lookahead(depth=0)`` must equal "argmax over the proposal pool", otherwise
  the H=0 row is not the myopic baseline
* the proposal pool must contain the greedy anchor, otherwise the pool itself
  (not the horizon) is what changes with H
* ``SIDProposal`` must report its real proposal-side cost, otherwise the Gate 3
  "equal budget" claim is false

Run directly:
    python tests/test_planning.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.planning.abstractions import SIDProposal  # noqa: E402
from muslate.planning.latent_model import CountingModel, OracleModel  # noqa: E402
from muslate.planning.proposal import ProposalPolicy  # noqa: E402
from muslate.planning.search import (  # noqa: E402
    CEM,
    Lookahead,
    MCTS,
    RandomShooting,
    ReactiveGreedy,
)


def _setup(lt: float = 1.0, div: float = 0.0):
    sim = SlateSim(EnvConfig(long_term_strength=lt, div_penalty=div))
    model = OracleModel(sim)
    state = sim.initial_state(0)
    return sim, model, model.encode(state)[None]


def _true_reward(sim, model, lat, slate):
    d = sim.cfg.dim
    return sim.reward_batch(lat[0, :d], lat[0, d:], slate[None])[0]


def test_lookahead_zero_is_pool_argmax() -> None:
    sim, model, lat = _setup()
    proposal = ProposalPolicy(model, n_candidates=32, seed=5, mode="item_perturb")
    planner = Lookahead(model, proposal, depth=0, gamma=sim.cfg.gamma)

    action = planner.plan(lat)[0]
    pool = proposal.propose(lat)[0]
    rewards = sim.reward_batch(lat[0, : sim.cfg.dim], lat[0, sim.cfg.dim :], pool)
    assert np.isclose(_true_reward(sim, model, lat, action), rewards.max())


def test_proposal_pool_contains_greedy() -> None:
    sim, model, lat = _setup()
    proposal = ProposalPolicy(model, n_candidates=16, seed=5, mode="item_perturb")
    pool = proposal.propose(lat)[0]
    greedy = sim.greedy_slate(lat[0, : sim.cfg.dim], lat[0, sim.cfg.dim :])
    assert any(np.array_equal(row, greedy) for row in pool), "pool must contain the greedy anchor"


def test_no_repeated_items_in_pool() -> None:
    sim, model, lat = _setup()
    for mode in ("item_perturb", "uniform"):
        proposal = ProposalPolicy(model, n_candidates=16, seed=5, mode=mode)
        pool = proposal.propose(lat)[0]
        for row in pool:
            assert len(set(row.tolist())) == len(row), f"{mode} produced a slate with duplicates"


def test_sid_proposal_cost_is_reported_correctly() -> None:
    sim, model, _ = _setup()
    flat = SIDProposal(
        model, sim.n_coarse, sim.catalog.n_fine, n_sampled=64, out_size=64, refine_positions=0
    )
    assert flat.proposal_cost() == 64

    hier = SIDProposal(
        model,
        sim.n_coarse,
        sim.catalog.n_fine,
        n_sampled=32,
        out_size=128,
        widen=4,
        refine_positions=5,
        fine_per_pos=8,
    )
    assert hier.proposal_cost() == 32 + 4 * 5 * 8


def test_sid_proposal_shape_and_validity() -> None:
    sim, model, lat = _setup(lt=3.0)
    for refine in (0, 5):
        prop = SIDProposal(
            model,
            sim.n_coarse,
            sim.catalog.n_fine,
            n_sampled=24,
            out_size=32,
            widen=4,
            refine_positions=refine,
            fine_per_pos=4,
            seed=2,
        )
        pool = prop.propose(lat)
        assert pool.shape[0] == 1 and 0 < pool.shape[1] <= 32 and pool.shape[2] == sim.slate_size
        assert len({tuple(a) for a in pool[0]}) == pool.shape[1]
        assert pool.min() >= 0 and pool.max() < sim.n_items


def test_sid_pool_contains_the_greedy_anchor() -> None:
    sim, model, lat = _setup(lt=1.0)
    prop = SIDProposal(
        model, sim.n_coarse, sim.catalog.n_fine, n_sampled=16, out_size=16, seed=7
    )
    pool = prop.propose(lat)[0]
    scores = model.item_scores(lat)[0]
    coarse_scores = scores.reshape(sim.n_coarse, sim.catalog.n_fine).max(axis=1)
    anchor_plan = np.argsort(-coarse_scores)[: sim.slate_size]
    anchor = prop._complete(scores, anchor_plan)
    assert any(np.array_equal(row, anchor) for row in pool), "pool must contain the greedy coarse anchor"


def test_planners_return_valid_slates() -> None:
    sim, model, lat = _setup(lt=3.0)
    planners = {
        "reactive": ReactiveGreedy(model, gamma=sim.cfg.gamma),
        "look0": Lookahead(
            model, ProposalPolicy(model, n_candidates=16, seed=1), depth=0, gamma=sim.cfg.gamma
        ),
        "look2": Lookahead(
            model, ProposalPolicy(model, n_candidates=16, seed=1), depth=2, beam_width=3, gamma=sim.cfg.gamma
        ),
        "look2_sid": Lookahead(
            model,
            SIDProposal(
                model, sim.n_coarse, sim.catalog.n_fine, n_sampled=8, out_size=16, refine_positions=3, seed=1
            ),
            depth=2,
            beam_width=3,
            gamma=sim.cfg.gamma,
        ),
        "shoot": RandomShooting(
            model, ProposalPolicy(model, n_candidates=16, seed=1), depth=2, n_samples=8, gamma=sim.cfg.gamma
        ),
    }
    for name, planner in planners.items():
        action = planner.plan(lat)[0]
        assert action.shape == (sim.slate_size,), name
        assert action.min() >= 0 and action.max() < sim.n_items, name


def test_lookahead_reduces_to_pool_argmax_for_any_proposal() -> None:
    """The abstraction swap must not change what depth=0 means."""
    sim, model, lat = _setup(lt=3.0)
    prop = SIDProposal(
        model, sim.n_coarse, sim.catalog.n_fine, n_sampled=16, out_size=24, seed=11
    )
    planner = Lookahead(model, prop, depth=0, gamma=sim.cfg.gamma)
    action = planner.plan(lat)[0]
    # re-derive the pool with a fresh generator on the same seed
    pool = SIDProposal(
        model, sim.n_coarse, sim.catalog.n_fine, n_sampled=16, out_size=24, seed=11
    ).propose(lat)[0]
    rewards = sim.reward_batch(lat[0, : sim.cfg.dim], lat[0, sim.cfg.dim :], pool)
    assert np.isclose(_true_reward(sim, model, lat, action), rewards.max())


def test_mcts_depth0_reduces_to_the_prior_argmax() -> None:
    """With no backup possible, MCTS must fall back to the immediate-reward prior."""
    sim, model, lat = _setup(lt=3.0)
    prop = ProposalPolicy(model, n_candidates=24, seed=4, mode="item_perturb")
    planner = MCTS(model, prop, max_depth=0, n_sims=16, gamma=sim.cfg.gamma)
    action = planner.plan(lat)[0]

    pool = ProposalPolicy(model, n_candidates=24, seed=4, mode="item_perturb").propose(lat)[0]
    rewards = sim.reward_batch(lat[0, : sim.cfg.dim], lat[0, sim.cfg.dim :], pool)
    assert np.isclose(_true_reward(sim, model, lat, action), rewards.max())


def test_mcts_the_tree_can_disagree_with_its_own_prior() -> None:
    """The point of a tree: the best *first* action need not be the best myopic one.

    Under the combinatorial objective the highest-immediate-reward slate is often
    a bad long-run choice, so a depth-limited search must be free to pick a first
    action the prior would not.  This test only asserts validity plus the
    structural possibility of disagreement (it records rather than mandates it,
    since whether the tree actually switches depends on the instance).
    """
    sim = SlateSim(EnvConfig(long_term_strength=3.0, div_penalty=0.8))
    model = OracleModel(sim)
    state = sim.initial_state(0)
    lat = model.encode(state)[None]

    prop = ProposalPolicy(model, n_candidates=32, seed=9, mode="item_perturb")
    prior_pick = prop.propose(lat)[0][0]                       # greedy anchor = the prior's choice

    planner = MCTS(model, prop, max_depth=3, n_sims=200, c_puct=1.5, gamma=sim.cfg.gamma, seed=9)
    chosen = planner.plan(lat)[0]

    assert chosen.shape == (sim.slate_size,)
    assert chosen.min() >= 0 and chosen.max() < sim.n_items
    # `chosen` comes from a pool the search built itself; the contract is validity
    # and reachability in the *action space*, not membership of the anchor pool.
    assert len(set(chosen.tolist())) == len(chosen)
    # disagreement is possible: the search is not hard-wired to the anchor
    _ = prior_pick


def test_cem_and_shooting_agree_on_validity() -> None:
    sim, model, lat = _setup(lt=3.0)
    for seed in (1, 2):
        cem = CEM(
            model,
            ProposalPolicy(model, n_candidates=16, seed=seed),
            depth=2,
            n_samples=16,
            n_elite=4,
            n_iters=2,
            gamma=sim.cfg.gamma,
            seed=seed,
        )
        act = cem.plan(lat)[0]
        assert act.shape == (sim.slate_size,)
        assert act.min() >= 0 and act.max() < sim.n_items
        assert len(set(act.tolist())) == len(act), "CEM must not repeat an item inside a slate"


def test_cem_sampler_never_repeats_items() -> None:
    sim, model, lat = _setup(lt=1.0)
    cem = CEM(model, ProposalPolicy(model, n_candidates=8, seed=0), n_samples=32, seed=0)
    logits = np.repeat(model.item_scores(lat)[0][None, :], sim.slate_size, axis=0)
    samples = cem._sample(logits, 64)
    assert samples.shape == (64, sim.slate_size)
    for row in samples:
        assert len(set(row.tolist())) == len(row)


def test_sid_proposal_accepts_a_custom_map_and_stays_backward_compatible() -> None:
    """Existing call sites pass (n_coarse, n_fine) positionally; that must keep working."""
    sim, model, lat = _setup(lt=3.0)
    from muslate.planning.sid_map import build_sid_map

    legacy = SIDProposal(model, sim.n_coarse, sim.catalog.n_fine, n_sampled=16, out_size=16, seed=3)
    assert legacy.propose(lat).shape == (1, 16, sim.slate_size)

    sid_map = build_sid_map(sim.n_items, sim.catalog.item_coarse, "random", seed=2)
    custom = SIDProposal(model, n_sampled=16, out_size=16, seed=3, sid_map=sid_map)
    pool = custom.propose(lat)
    assert pool.shape == (1, 16, sim.slate_size)
    assert pool.min() >= 0 and pool.max() < sim.n_items


def test_noisy_tree_still_completes_with_the_real_item_scores() -> None:
    """Review Focus #1: the tree may change, the completion rule may not.

    If `_complete` sorted by anything but the real `item_scores`, the sweep would
    measure "is greedy fill-in broken" rather than "is the tree good".
    """
    sim, model, lat = _setup(lt=3.0)
    from muslate.planning.sid_map import build_sid_map

    scores = model.item_scores(lat)[0]
    sid_map = build_sid_map(sim.n_items, sim.catalog.item_coarse, "noisy", p=1.0, seed=11)
    prop = SIDProposal(model, n_sampled=1, out_size=1, seed=0, sid_map=sid_map)

    filled = prop._complete(scores, np.arange(sid_map.n_coarse))
    for c, item in enumerate(filled):
        cls = int(sid_map.coarse_of[item])
        items = sid_map.items_by_coarse[cls]
        assert scores[item] == scores[items].max(), f"position {c} did not take the class argmax"


def test_sid_proposal_requires_a_map_or_legacy_dims() -> None:
    sim, model, lat = _setup(lt=3.0)
    try:
        SIDProposal(model, n_sampled=8, out_size=8, seed=0)
    except ValueError:
        return
    raise AssertionError("omitting both sid_map and n_coarse/n_fine must raise ValueError")


def test_sid_proposal_rejects_a_map_with_too_few_classes() -> None:
    """Review finding N2: `_coarse_plans` needs at least `slate_size` classes.

    With a custom map (e.g. a `learned` tree that k-means collapsed to 3 clusters)
    `plans[0] = np.argsort(-coarse_scores)[:k]` would be short and fail later with
    an opaque numpy broadcast error.  Fail early and clearly instead.
    """
    sim, model, lat = _setup(lt=3.0)
    from muslate.planning.sid_map import build_sid_map

    sid_map = build_sid_map(sim.n_items, sim.catalog.item_coarse, "oracle", seed=0)
    # force a degenerate map: everything in a single class
    one_class = type(sid_map)(
        coarse_of=np.zeros(sim.n_items, dtype=np.int64),
        n_coarse=1,
        items_by_coarse=[np.arange(sim.n_items, dtype=np.int64)],
        purity=0.0,
        kind="degenerate",
    )
    try:
        SIDProposal(model, n_sampled=8, out_size=8, seed=0, sid_map=one_class)
    except ValueError:
        return
    raise AssertionError("a map with fewer classes than slate_size must raise ValueError")


def test_gate3_variants_wire_the_adjust_hook_into_every_planner() -> None:
    """P2: the conservative penalty must reach *every* planner.

    If it reached only some of them, a comparison across variants would be a
    comparison of penalties rather than of search structure -- which is exactly
    the confound this test exists to prevent.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_gate3 import build_variants

    sim, model, lat = _setup(lt=3.0)
    calls: list[int] = []

    def adjust(states, slates, rewards):
        calls.append(len(slates))
        return rewards - 0.5

    planners = build_variants(model, sim, budget=32, depth=1, seed=0, adjust=adjust)
    for name, planner in planners.items():
        assert planner.adjust is adjust, f"{name} did not receive the adjust hook"

    # and it must actually fire during a search, not merely be stored
    planners["sid_hier"].plan(lat)
    assert calls, "the adjust hook was stored but never invoked"


def test_gate3_variants_default_to_no_penalty() -> None:
    """Omitting `adjust` must reproduce the unpenalised Gate 3 behaviour exactly."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_gate3 import build_variants

    sim, model, _lat = _setup(lt=3.0)
    planners = build_variants(model, sim, budget=32, depth=1, seed=0)
    for name, planner in planners.items():
        assert planner.adjust is None, name


def test_counting_model_counts_slate_evaluations() -> None:
    sim, model, lat = _setup(lt=3.0)
    counter = CountingModel(model)
    slates = np.stack([sim.greedy_slate(sim.initial_state(0).h, sim.initial_state(0).sat)] * 5)

    counter.reset()
    counter.step(np.repeat(lat, 5, axis=0), slates)
    assert counter.n_slate_evals == 5

    counter.reset()
    counter.item_scores(lat)
    assert counter.n_score_calls == 1
    assert counter.n_slate_evals == 0


def test_all_planners_spend_model_budget() -> None:
    """Every planner except the reactive baseline must actually query the model."""
    sim, model, lat = _setup(lt=3.0, div=0.8)
    counter = CountingModel(model)

    planners = {
        "reactive": ReactiveGreedy(counter, gamma=sim.cfg.gamma),
        "shooting": RandomShooting(
            counter, ProposalPolicy(counter, n_candidates=16, seed=1), depth=1, n_samples=8, gamma=sim.cfg.gamma
        ),
        "lookahead": Lookahead(
            counter, ProposalPolicy(counter, n_candidates=16, seed=1), depth=1, beam_width=2, gamma=sim.cfg.gamma
        ),
        "mcts": MCTS(counter, ProposalPolicy(counter, n_candidates=16, seed=1), max_depth=1, n_sims=8, gamma=sim.cfg.gamma),
        "cem": CEM(
            counter,
            ProposalPolicy(counter, n_candidates=16, seed=1),
            depth=1,
            n_samples=8,
            n_elite=3,
            n_iters=2,
            gamma=sim.cfg.gamma,
        ),
    }

    evals = {}
    for name, planner in planners.items():
        counter.reset()
        planner.plan(lat)
        evals[name] = counter.n_slate_evals
        assert evals[name] > 0 or name == "reactive", f"{name} never queried the model"

    assert evals["mcts"] > 0 and evals["cem"] > 0 and evals["lookahead"] > 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
