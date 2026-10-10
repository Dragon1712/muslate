# Semantic Purity Sweeps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run two orthogonal purity sweeps that isolate *why* `sid_hier` loses to flat search under learned dynamics — one over the **tree's** semantic fidelity, one over the **prior's** fidelity — under ground-truth dynamics so neither is confounded by world-model error.

**Architecture:** Introduce a `SIDMap` value object that decouples "which items belong to which category" from the proposal algorithm, so `SIDProposal` can be driven by an oracle / noisy / random / learned tree. Separately, introduce `NoisyScoreModel`, a `LatentModel` decorator that adds a **fixed, coarse-level** bias to `item_scores`, so every planner sees the same corrupted prior. `run_purity.py` then sweeps each axis with the same equal-budget planners Gate 3 already uses.

**Tech Stack:** Python (3.14 local / 3.10 server), numpy 2.3.5, torch 2.12 CPU. No new dependencies.

**Spec:** `SUMMARY.md` §2.5.1 (SID purity knob), §3.3 (why `sid_hier` fails under learned dynamics), and §第五部分 P0b (the sweep's purpose and acceptance criterion).

## Global Constraints

- **Not a git repository.** Do not add `git commit` steps; the equivalent closing step is "run the full test suite and record results".
- **No new dependencies.** `sklearn` is not installed — k-means must be hand-written in numpy.
- **Tests run without pytest**: each file under `tests/` is standalone (`python tests/test_x.py`) and ends with the `if __name__ == "__main__":` runner block that collects `test_*` functions. Follow that exact pattern.
- **Backward compatibility is mandatory**: every existing call site uses the positional form `SIDProposal(model, n_coarse, n_fine, ...)`. That signature must keep working after this change.
- **Equal-budget invariant preserved**: `proposal_cost()` and `output_size()` must keep returning the same values as before for the uniform map.
- Environment MDP is unchanged by this plan; all new code is planner-side.

## Review Focus

1. **`noisy` trees must perturb only the tree's layering, never the completion step.** If `_complete` sorts by anything other than the real `item_scores`, the sweep measures "is greedy fill-in broken", not "is the tree good". The test asserts that a `noisy` map with `p=1.0` and a `random` map differ from `oracle` in *which* items are grouped, while all three still pick the highest-scoring item within whatever group they are given.
2. **`SIDMap.items_by_coarse` must be a partition of `range(n_items)`** — every item exactly once. A gap makes `_complete` silently return fewer than K items; a duplicate makes `widen`/`fine_per_pos` semantics drift. Asserted directly.
3. **`NoisyScoreModel`'s bias must be fixed for the lifetime of the object.** If it resamples per call, `_coarse_plans` and `_complete` see different scores inside one `propose()`, which is internal inconsistency rather than a wrong prior — a different phenomenon from the one being studied.
4. **The `random` tree must preserve the oracle's per-class sizes.** Otherwise `n_fine` changes meaning and `fine_per_pos=8` samples a different fraction of the class, confounding tree quality with calibration of the refinement budget.
5. **`learned` k-means must be deterministically seeded.** An unseeded init makes sweep points irreproducible; the test calls `build_sid_map` twice with the same seed and asserts identical output.

---

### Task 1: `SIDMap` and its builders

**Files:**
- Create: `muslate/planning/sid_map.py`
- Test: `tests/test_sid_map.py`

**Interfaces:**
- Consumes: nothing (leaf module; may import `numpy`)
- Produces:
  - `SIDMap` dataclass with fields `coarse_of: np.ndarray` (shape `[n_items]`, int64), `n_coarse: int`, `items_by_coarse: list[np.ndarray]`, `purity: float`, `kind: str`
  - `SIDMap.uniform(n_items: int, n_coarse: int, n_fine: int) -> SIDMap` (classmethod)
  - `build_sid_map(n_items: int, item_coarse: np.ndarray, kind: str, p: float = 0.0, seed: int = 0, embedding: np.ndarray | None = None) -> SIDMap`
  - supported `kind` values: `"oracle"`, `"noisy"`, `"random"`, `"learned"`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sid_map.py
import numpy as np
from muslate.planning.sid_map import SIDMap, build_sid_map

def _fake_catalog(n_items=40, n_coarse=4):
    coarse = np.repeat(np.arange(n_coarse), n_items // n_coarse)
    return n_items, coarse, n_items // n_coarse

def test_items_by_coarse_is_a_partition():
    n_items, coarse, n_fine = _fake_catalog()
    for kind, p in [("oracle", 0.0), ("random", 0.0), ("noisy", 0.5), ("noisy", 1.0)]:
        m = build_sid_map(n_items, coarse, kind, p=p, seed=3)
        flat = np.concatenate(m.items_by_coarse)
        assert np.array_equal(np.sort(flat), np.arange(n_items)), kind
        assert m.coarse_of.shape == (n_items,)
        assert m.n_coarse == int(m.coarse_of.max()) + 1

def test_random_tree_preserves_class_sizes():
    n_items, coarse, n_fine = _fake_catalog()
    m = build_sid_map(n_items, coarse, "random", seed=1)
    assert sorted(len(g) for g in m.items_by_coarse) == sorted(np.bincount(coarse).tolist())

def test_noisy_is_a_controlled_departure_from_oracle():
    n_items, coarse, n_fine = _fake_catalog()
    oracle = build_sid_map(n_items, coarse, "oracle", seed=0)
    assert np.array_equal(oracle.coarse_of, coarse)
    for p in (0.25, 0.5, 1.0):
        m = build_sid_map(n_items, coarse, "noisy", p=p, seed=5)
        assert m.purity == 1.0 - p
        # a noisy tree must lose some oracle agreement, more so as p grows
        agree = float((m.coarse_of == coarse).mean())
        assert agree <= 1.0 - p + 1e-9

def test_learned_is_deterministic_given_seed():
    n_items, coarse, n_fine = _fake_catalog()
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n_items, 8))
    a = build_sid_map(n_items, coarse, "learned", seed=7, embedding=emb)
    b = build_sid_map(n_items, coarse, "learned", seed=7, embedding=emb)
    assert np.array_equal(a.coarse_of, b.coarse_of)

def test_uniform_matches_the_legacy_layout():
    m = SIDMap.uniform(n_items=40, n_coarse=4, n_fine=10)
    assert np.array_equal(m.coarse_of, np.repeat(np.arange(4), 10))
    assert m.n_coarse == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_sid_map.py`
Expected: `ModuleNotFoundError: No module named 'muslate.planning.sid_map'`

- [ ] **Step 3: Implement `muslate/planning/sid_map.py`**

`noisy` reassigns each item's coarse label with probability `p` by swapping labels between randomly chosen pairs (keeps class sizes roughly balanced); `purity` is stored as `1.0 - p`. `random` permutes the label vector generated by `uniform`, which preserves sizes exactly. `learned` runs a hand-written Lloyd k-means on `embedding` with `n_coarse` centroids, seeded from `np.random.default_rng(seed)`, then relabels clusters in descending size order to keep `n_coarse` stable.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_sid_map.py`
Expected: `5 tests passed`

---

### Task 2: `SIDProposal` driven by a `SIDMap`

**Files:**
- Modify: `muslate/planning/abstractions.py` (constructor at lines 48-70; `_items_in` at 87-88; `_complete` at 90-101; `_coarse_plans`)
- Test: `tests/test_planning.py` (append)

**Interfaces:**
- Consumes: `SIDMap` from Task 1
- Produces: `SIDProposal(model, n_coarse=None, n_fine=None, ..., sid_map: SIDMap | None = None)`. When `sid_map` is given it wins and `n_coarse`/`n_fine` are ignored; when it is `None` a `SIDMap.uniform` is built from them, reproducing today's behaviour exactly. `_items_in(c)` becomes a lookup into `sid_map.items_by_coarse[c]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_planning.py
def test_sid_proposal_accepts_a_custom_map_and_stays_backward_compatible() -> None:
    sim, model, lat = _setup(lt=3.0)
    from muslate.planning.sid_map import build_sid_map

    # legacy positional form must keep working
    legacy = SIDProposal(model, sim.n_coarse, sim.catalog.n_fine, n_sampled=16, out_size=16, seed=3)
    assert legacy.propose(lat).shape == (1, 16, sim.slate_size)

    # a random tree yields a different, still-valid pool
    sid_map = build_sid_map(
        sim.n_items, sim.catalog.item_coarse, "random", seed=2
    )
    custom = SIDProposal(model, n_sampled=16, out_size=16, seed=3, sid_map=sid_map)
    pool = custom.propose(lat)
    assert pool.shape == (1, 16, sim.slate_size)
    assert pool.min() >= 0 and pool.max() < sim.n_items

def test_noisy_tree_still_completes_with_the_real_item_scores() -> None:
    """Review Focus #1: the tree may change, the completion rule may not."""
    sim, model, lat = _setup(lt=3.0)
    from muslate.planning.sid_map import build_sid_map

    scores = model.item_scores(lat)[0]
    sid_map = build_sid_map(sim.n_items, sim.catalog.item_coarse, "noisy", p=1.0, seed=11)
    prop = SIDProposal(model, n_sampled=1, out_size=1, seed=0, sid_map=sid_map)
    plan = sid_map.coarse_of.copy()          # one item per class, in class order
    filled = prop._complete(scores, np.arange(sid_map.n_coarse))
    for c, item in enumerate(filled):
        items = sid_map.items_by_coarse[sid_map.coarse_of[item]]
        assert scores[item] == scores[items].max(), f"position {c} did not take the class argmax"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_planning.py`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'sid_map'`

- [ ] **Step 3: Implement the `sid_map` parameter in `muslate/planning/abstractions.py`**

Signature becomes `__init__(self, model, n_coarse=None, n_fine=None, n_sampled=64, out_size=64, widen=4, refine_positions=0, fine_per_pos=8, seed=0, temperature=0.7, sid_map=None)`. Store `self.sid_map`; set `self.n_coarse = sid_map.n_coarse`. Replace `_items_in` with `return self.sid_map.items_by_coarse[c]`. If `sid_map is None`, build `SIDMap.uniform(model.n_items, n_coarse, n_fine)` first, raising a clear `ValueError` when both are missing. `coarse_scores` in `_search_one` must reshape by the map, so replace `scores.reshape(self.n_coarse, self.n_fine).max(axis=1)` with a per-class max computed from `items_by_coarse`.

- [ ] **Step 4: Run tests to verify they pass, plus the full regression**

Run: `python tests/test_planning.py` then `python tests/test_env.py`
Expected: `16 tests passed` (14 existing + 2 new) and `9 tests passed`

---

### Task 3: `NoisyScoreModel` — a corrupted prior that all planners share

**Files:**
- Create: `muslate/planning/perturb.py`
- Test: `tests/test_perturb.py`

**Interfaces:**
- Consumes: `LatentModel` from `muslate.planning.latent_model`, `SIDMap` from Task 1
- Produces: `NoisyScoreModel(inner: LatentModel, coarse_of: np.ndarray, n_coarse: int, noise: float, seed: int = 0)` implementing `LatentModel`; `item_scores(states)` returns `inner.item_scores(states) + self.bonus[coarse_of]` where `bonus = np.random.default_rng(seed).normal(0.0, noise, size=n_coarse)` is drawn **once** in `__init__`; `reward` and `step` delegate unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_perturb.py
import numpy as np
from muslate.config import EnvConfig
from muslate.env import SlateSim
from muslate.planning.latent_model import OracleModel
from muslate.planning.perturb import NoisyScoreModel

def _setup(noise: float):
    sim = SlateSim(EnvConfig(long_term_decouples if False else 3.0, div_penalty=0.8))
    inner = OracleModel(sim)
    state = sim.initial_state(0)
    lat = inner.encode(state)[None]
    return sim, inner, NoisyScoreModel(inner, sim.catalog.item_coarse, sim.n_coarse, noise, seed=0), lat

def test_zero_noise_is_the_identity():
    _sim, inner, wrapped, lat = _setup(0.0)
    assert np.allclose(wrapped.item_scores(lat), inner.item_scores(lat))

def test_bias_is_fixed_across_calls():
    """Review Focus #3: the prior must be consistently wrong, not jittery."""
    _sim, _inner, wrapped, lat = _setup(1.0)
    a = wrapped.item_scores(lat)
    b = wrapped.item_scores(lat)
    assert np.array_equal(a, b)

def test_bias_is_constant_within_a_class():
    sim, inner, wrapped, lat = _setup(1.0)
    base = inner.item_scores(lat)[0]
    got = wrapped.item_scores(lat)[0]
    delta = got - base
    for c in range(sim.n_coarse):
        items = np.where(sim.catalog.item_coarse == c)[0]
        assert np.allclose(delta[items], delta[items[0]]), f"class {c} not shifted uniformly"

def test_reward_and_step_delegate_unchanged():
    sim, inner, wrapped, lat = _setup(2.0)
    slates = np.stack([sim.greedy_slate(sim.initial_state(0).h, sim.initial_state(0).sat)])
    assert np.allclose(wrapped.reward(lat, slates), inner.reward(lat, slates))
    r_w, s_w = wrapped.step(lat, slates)
    r_i, s_i = inner.step(lat, slates)
    assert np.allclose(r_w, r_i) and np.allclose(s_w, s_i)
```

Note: fix the `EnvConfig(long_term_decouples if False else 3.0, ...)` typo when writing the real test — it should read `EnvConfig(long_term_strength=3.0, div_penalty=0.8)`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/test_perturb.py`
Expected: `ModuleNotFoundError: No module named 'muslate.planning.perturb'`

- [ ] **Step 3: Implement `muslate/planning/perturb.py`**

Subclass `LatentModel`, mirror `state_dim`/`n_items`/`slate_size` from `inner`, and store the precomputed `bonus` plus the indexing array `coarse_of` so `item_scores` is a single gather-and-add.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/test_perturb.py`
Expected: `4 tests passed`

---

### Task 4: `run_purity.py` — the two sweeps

**Files:**
- Create: `scripts/run_purity.py`
- Test: manual smoke run (no unit test; it is a thin experiment driver over tested components)

**Interfaces:**
- Consumes: `SIDMap`/`build_sid_map` (Task 1), `SIDProposal` with `sid_map` (Task 2), `NoisyScoreModel` (Task 3), plus existing `Lookahead`, `ProposalPolicy`, `ReactiveGreedy`, `OracleModel`, `evaluate`
- Produces: `results/purity_sweeps.json` and `.md`, with two tables keyed `sweep_a_tree` and `sweep_b_prior`

- [ ] **Step 1: Write the runner**

Both sweeps use **oracle dynamics** and the same planner set at equal budget, `--budget 512 --depth 2`, matching Gate 3. Sweep A varies the tree with the prior held perfect; Sweep B varies the prior with the tree held at oracle. Report, for each x value, the four planners' returns plus the derived `sid_hier - max(flat_uniform, flat_policy)`. Accept `--budget`, `--depth`, `--users`, `--lts`, `--div`, `--purities`, `--noises`, `--seed`, `--out`.

Planner construction mirrors `run_gate3.build_variants`; read that function and reuse its budget arithmetic verbatim so the two runners stay comparable.

- [ ] **Step 2: Smoke-test it at tiny scale**

Run: `python scripts/run_purity.py --lts 3.0 --users 3 --budget 32 --depth 1 --purities 1.0 0.0 --noises 0.0 1.0 --out results/_purity_smoke.json`
Expected: writes the file, prints two tables, finishes in well under a minute.

- [ ] **Step 3: Delete the smoke artefacts and record nothing yet**

Run: `Remove-Item results\_purity_smoke.json, results\_purity_smoke.md -Force -ErrorAction SilentlyContinue`

---

### Task 5: Run the sweeps and record the outcome

**Files:**
- Modify: `SUMMARY.md` (§2.5.1 and §第五部分 P0b — replace the plan-as-intent with plan-as-result)
- Create: `results/purity_sweeps.json`, `results/purity_sweeps.md` (produced by Task 4)

**Interfaces:**
- Consumes: the runner from Task 4
- Produces: the numbers that decide whether the "hierarchy amplifies ranking error" hypothesis survives

- [ ] **Step 1: Run the full sweep**

Run: `python scripts/run_purity.py --lts 3.0 --div 0.8 --users 16 --budget 512 --depth 2 --purities 1.0 0.75 0.5 0.25 0.0 --noises 0.0 0.25 0.5 1.0 2.0 --out results/purity_sweeps.json`
Expected: `results/purity_sweeps.json` + `.md`.

- [ ] **Step 2: Run the full test suite**

Run each of: `python tests/test_env.py`, `test_planning.py`, `test_world_model.py`, `test_conservative.py`, `test_error_model.py`, `test_sid_map.py`, `test_perturb.py`
Expected: all pass, with `test_planning.py` at 16 and the two new files at 5 and 4.

- [ ] **Step 3: Update `SUMMARY.md`**

Under §2.5.1 record the realized `purity -> hierarchical gain` curve; under P0b replace the "判据" line with the observed result. If `sid_hier`'s curve is materially steeper than the flat baselines' in Sweep B, record the "hierarchy amplifies prior error" finding and note that it is measured under oracle dynamics (so it is a property of the planner, not of the world model). State the null result plainly if the curves are parallel.

---

## Self-Review

**Spec coverage.** §2.5.1 asks for a purity knob and the four tree variants → Task 1 (`oracle`/`noisy`/`random`/`learned`) + Task 2 (wiring) + Task 4 Sweep A. §3.3's diagnosis asks whether the hierarchy's failure is about model quality → Task 3 + Task 4 Sweep B isolate the prior as a separate axis. P0b's acceptance criterion ("purity ↑ ⇒ hierarchical gain ↑, with a threshold") is computed by Task 4's `sid_hier - max(flat)` column and recorded in Task 5.

**Step scan.** Every step names one action with a checkable result. No step says "handle edge cases" or "write tests for the above". The k-means step is the only one whose body is not fully determined by its signature, and the plan states its required properties (seeded, size-ordered relabelling) rather than leaving it open.

**Type consistency.** `SIDMap.coarse_of` (int64 array), `n_coarse` (int), `items_by_coarse` (list of int arrays) are used identically in Tasks 1, 2, 3 and 4. `build_sid_map(n_items, item_coarse, kind, p, seed, embedding)` keeps the same parameter order everywhere. `NoisyScoreModel.item_scores` returns the same `[B, N]` float shape as `LatentModel.item_scores`.

**Review Focus.** Each of the five lines has a test: #1 → `test_noisy_tree_still_completes_with_the_real_item_scores`; #2 → `test_items_by_coarse_is_a_partition`; #3 → `test_bias_is_fixed_across_calls`; #4 → `test_random_tree_preserves_class_sizes`; #5 → `test_learned_is_deterministic_given_seed`.

**Proportion.** Four source files (one leaf value object, one decorator, two small edits) against five focused tests and two experiment tables. No task's code is transcribed in full; bodies are specified only where the signature and tests leave a choice (k-means properties, the noise drawing rule).
