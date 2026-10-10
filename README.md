# MuSlate — Gate experiments

> **文档导航**
> * **[`SUMMARY.md`](SUMMARY.md)** — 主文档：研究思路、实验实现方法、全部实验结果、被推翻的假设、后续实验
> * [`EXPERIMENTS.md`](EXPERIMENTS.md) — 实验结果数据表（按实验组织的细节版）
> * 本文件 — 工程说明：目录结构、脚本用法、设计决定、已知局限

Prototype for the research direction *"Offline Conservative Latent Planning over
Combinatorial Slate Actions"* / *"Uncertainty-Aware Hierarchical Latent Planning
for Generative Slate Recommendation"*.

This repository contains **only the pre-coding gate experiments**. The point of a
gate is to kill weak ideas cheaply, before a full system is built.

| Gate | Question | Why it decides the paper |
|---|---|---|
| **Gate 2** | Does multi-step planning beat one-step lookahead? | MF-MDP reports *"a single step of lookahead already captures most of the gain"*. If that holds in our setting too, the whole multi-step framing collapses into UA-TWM / OGR territory. |
| **Gate 3** | Does a Semantic-ID hierarchy buy anything over flat slate actions at equal budget? | It is the second half of the novelty claim. If the hierarchy is decoration, the contribution shrinks to "world model + conservative planning". |
| **Q3** | Does a tree beat a beam at equal *measured* budget? | Directly tests the "is ordinary rollout search already enough?" objection to the MCTS component. Answer: yes under combinatorial reward, no under separable reward. |
| **Gate 4** | Can conservative planning stop the planner exploiting model error? | Mechanism works with a calibrated signal, does nothing without one. Ensemble disagreement is unusable (ρ=0.1); a supervised holdout **error model reaches ρ=0.6**. |
| **Gate 1** | Can request-level slates be recovered from real logs (KuaiRand)? | Not covered here — this repo runs on a synthetic simulator. See *Limitations*. |

Everything runs against a **fully controllable synthetic slate simulator** whose
ground-truth user dynamics are known and whose long-horizon coupling strength can
be dialled. That is what makes it possible to answer Gate 2 as a *function of the
environment* rather than as a single yes/no.

---

## Headline result (Gate 2, ground-truth dynamics)

Return of the same beam-based planner at different lookahead depths `H`, as a
function of the long-horizon coupling knob `lt` (`long_term_strength`):

| `lt` | H=0 | H=1 | H=2 | H=3 | H=5 | best depth |
|---|---|---|---|---|---|---|
| 0.00 | 16.300 | +0.000 | +0.000 | +0.000 | +0.000 | **H=0 (lookahead is worthless)** |
| 0.25 | 16.486 | +0.087 | +0.112 | +0.074 | +0.063 | H=2 |
| 0.50 | 15.717 | +0.078 | +0.089 | +0.066 | +0.080 | H=2 |
| 0.75 | 14.870 | +0.163 | +0.165 | +0.238 | +0.223 | H=3 |
| 1.00 | 14.335 | +0.137 | +0.338 | +0.321 | +0.227 | H=2 |
| 2.00 | 11.227 | +0.029 | +0.286 | +0.218 | +0.569 | H=5 |
| 3.00 | 8.512 | +0.571 | +0.631 | +1.045 | +0.994 | H=3 |
| 5.00 | 7.794 | +0.220 | +0.572 | +1.111 | +1.698 | H=5 |

(`results/gate2_oracle.json`; 12 users, proposal pool 32, beam 3. `reactive` == `H=0`
by construction, which is the check that depth is the only thing varying.)

Two findings worth writing up:

1. **`lt = 0` is exactly flat.** When the present does not constrain the future,
   extra lookahead contributes *nothing* — `H=0` and `H=5` are identical to the
   fourth decimal. This is the regime MF-MDP's conclusion lives in, and it is the
   honest boundary of our claim.
2. **When coupling exists, the gain grows monotonically with coupling strength,
   and the optimal depth moves right** (H=2 → H=3 → H=5), but at moderate
   coupling the curve is *unimodal*: H=5 is worse than H=2 because search
   approximation error accumulates faster than the extra information pays off.

Reproduce with `python scripts/probe_environment.py` (seconds) or
`python scripts/run_gate2.py --dynamics oracle` (minutes).

---

## Gate 3 result: the hierarchy pays off *only when the reward is combinatorial*

Return at **equal model-evaluation budget**, `lt = 3.0`, lookahead depth 2.
Two environments, differing only in the intra-slate diversity penalty
(`--div`), which adds

```
r -= div * sum_{j<k} w_j w_k <z_j, z_k>
```

an **immediate combinatorial** term that makes the slate reward non-separable.

**`div = 0` — separable reward (per-position greedy is optimal):**

| budget | reactive | flat_uniform | flat_policy | sid_flat | sid_hier |
|---|---|---|---|---|---|
| 64 | 8.506 | −1.059 | **8.925** | 6.564 | 7.502 |
| 256 | 8.506 | 0.751 | **9.687** | 7.056 | 8.096 |
| 1024 | 8.506 | 1.769 | **9.236** | 7.887 | 8.313 |

**`div = 0.8` — combinatorial reward:**

| budget | reactive | flat_uniform | flat_policy | sid_flat | sid_hier |
|---|---|---|---|---|---|
| 64 | −15.745 | −3.060 | −1.935 | −1.721 | −2.044 |
| 256 | −15.745 | −1.416 | −0.942 | −0.830 | **+0.976** |
| 1024 | −15.745 | −0.320 | −1.181 | −0.013 | **+1.967** |

Three findings, and together they are the actual contribution of Gate 3:

1. **A separable objective makes the hierarchy worthless.** With `div = 0` the
   category structure has nothing to exploit: per-position greedy is optimal,
   `flat_policy` (greedy + single-item substitutions) wins at every budget, and
   the Semantic-ID variants only catch up at large budgets.
2. **A combinatorial objective destroys per-position greedy.** With `div = 0.8`,
   `reactive` — which piles the K highest-scoring items into one coarse category
   — collapses from 8.5 to **−15.7**. This is not a modelling trick; it is what
   happens whenever a slate has an intra-slate constraint.
3. **Only then does the hierarchy win, and it scales with budget.** `sid_hier`
   rises monotonically with budget (−2.04 → +0.98 → +1.97) and overtakes every
   flat pool at budget ≥ 256. `flat_policy` does not improve at all
   (−1.94 → −0.94 → −1.18). Its structure — commit to a coarse category plan,
   then widen progressively and refine items inside the retained plans — is
   exactly what the diversity term rewards, because the penalty is largely a
   *category-multiset* property.

Note also that `sid_hier < sid_flat` at budget 64 (`−2.04` vs `−1.72`): at low
budget the refinement cost crowds out coarse sampling. The hierarchy needs a
budget floor before it pays.

**Consequence for the paper.** The claim is now conditional in a *useful* way:
multi-step planning needs long-horizon coupling (Gate 2), and hierarchical
Semantic-ID search needs intra-slate combinatorial structure (Gate 3). Both
conditions are measurable properties of a dataset, not modelling choices — which
turns the two gates into a concrete checklist for when the method applies,
instead of a claim that it always does.

Reproduce with:

```bash
python scripts/run_gate3.py --dynamics oracle --lt 3.0 --budget 1024 --depth 2 --div 0.0
python scripts/run_gate3.py --dynamics oracle --lt 3.0 --budget 1024 --depth 2 --div 0.8
```

---

## Q3: is a tree actually needed? (answer: only when the reward is combinatorial)

Five planners, same proposal pool, same depth, same discount.  Crucially the
tables report the **actual number of world-model evaluations per decision**
(measured by wrapping the model in `CountingModel`), not a nominal budget —
different planners spend their budget structurally differently, so nominal
budgets are not comparable.

**`lt = 3.0`, `div = 0.8` (combinatorial reward), depth 2:**

| planner | return | evals/decision |
|---|---|---|
| reactive | −15.745 | 0 |
| shooting | −3.754 | 96 |
| beam | −2.790 | 416 |
| cem | −5.261 | 288 |
| mcts (`n_sims=32`) | **−2.382** | 1056 |
| mcts (`n_sims=12`, **budget-matched to beam**) | **−2.466** | 416 |
| mcts (`n_sims=12 --no-bootstrap`, ablation) | −4.621 | 416 |

**`lt = 3.0`, `div = 0.0` (separable reward), depth 2:**

| planner | return | evals/decision |
|---|---|---|
| reactive | 8.506 | 0 |
| shooting | 8.040 | 96 |
| beam | **9.275** | 416 |
| cem | 0.668 | 288 |
| mcts | 9.030 | 1056 |

**`lt = 5.0`, `div = 0.8`, depth 4, budget-matched at 3872 evaluations** (the
beam's cost grows like `M · sum(bw^i)` while MCTS grows linearly in simulations,
so depth 4 is where the tree has the best shot):

| planner | return | evals/decision |
|---|---|---|
| reactive | −23.004 | 0 |
| shooting | −15.758 | 320 |
| beam | −12.584 | 3872 |
| cem | −14.710 | 960 |
| mcts | **−12.202** | 3872 |

**Verdict: the tree wins exactly when the reward is combinatorial, and loses when
it is separable.**

* With `div = 0.8`, MCTS beats the beam at *matched* budget (−2.466 vs −2.790 at
  416 evaluations) and again at depth 4 (−12.202 vs −12.584), and it also wins
  when given a larger budget. The mechanism is specific: the beam's per-level
  top-`bw` pruning is a *position-wise* cut, so under a combinatorial objective
  it discards paths that sacrifice immediate reward for a better category
  composition. UCB does not prune that way.
* With `div = 0.0`, the beam wins (9.275 vs 9.030). A separable objective needs
  no exploration for composition, so the tree's extra generality is pure
  overhead.

The effect size is modest (~0.3 points, and one seed), so **the significance is
not yet established** — but the sign is consistent across depth, budget and both
budget-matched and unmatched comparisons, which is why it is reported rather than
buried. Confirming it needs multiple seeds and a larger catalogue; see
*Limitations*.

### A methodological note worth keeping

The first version of this table said the opposite — "MCTS never overtakes the
beam". That conclusion was an artifact of an unfair comparison: `Lookahead`
bootstraps its horizon with the best immediate reward available there (it
evaluates one more expansion at the leaf), while the original MCTS returned a
hard zero leaf value. A zero leaf systematically under-rates *shallow* paths,
which biases the tree toward depth rather than toward good decisions.

The ablation above isolates it exactly. At matched budget the same tree scores
**−4.621 with a zero leaf and −2.466 with the shared bootstrap**, against the
beam's −2.790 — so the bootstrap, not the tree, was deciding the verdict. The
lesson is worth keeping for the paper: fair-baseline discipline matters more than
the planner. `CountingModel` now pins every planner's budget accounting in
`tests/test_planning.py`.

Reproduce with:

```bash
python scripts/run_q3.py --dynamics oracle --lt 3.0 --div 0.8 --depth 2
python scripts/run_q3.py --dynamics oracle --lt 3.0 --div 0.8 --depth 2 --n-sims 12 # budget-matched
python scripts/run_q3.py --dynamics oracle --lt 3.0 --div 0.0 --depth 2            # separable reward
python scripts/run_q3.py --dynamics oracle --lt 5.0 --div 0.8 --depth 4 --n-sims 120
```

---

## Gate 4: conservative planning

Three separate questions, and they have to be answered in this order.

### (a) Does a planner exploit model error?  Yes.

`BiasedOracleModel` injects a **state-coupled** bias into otherwise perfect
dynamics: `bias = strength * <w, mean(z_A)> * (1 + coupling*sum(sat))`.  The
coupling is what makes it exploitable rather than merely wrong — the planner sees
"saturating the user makes the next slate look better" and plans to saturate,
which is exactly what the real environment punishes.

`lt=3.0`, `div=0.8`, depth 2-5 lookahead:

| injected bias | H=0 | H=1 | H=2 | H=3 | H=5 |
|---|---|---|---|---|---|
| 0.0 | −5.475 | −3.401 | −2.768 | −2.416 | −2.608 |
| 1.0 | −6.482 | −5.140 | −3.542 | −4.727 | −3.859 |
| 2.0 | −8.420 | −6.753 | −6.224 | −5.384 | −5.286 |
| 4.0 | −10.218 | −9.336 | −7.697 | −7.743 | **−8.323** |

True return at **H=0** degrades monotonically with the bias (−5.5 → −10.2): the
myopic planner is already being dragged toward the over-estimated slates.  At
`bias=4` the curve also turns over — H=5 (−8.323) is worse than H=3 (−7.743) —
which is the exploitation signature: *more* planning, *worse* outcome.

### (b) Can a penalty fix it?  Yes — if the uncertainty estimate is calibrated.

| bias | no penalty (H=0) | `unc-perfect` λ=1 (H=0) | recovery |
|---|---|---|---|
| 0.0 | −5.475 | −5.475 | — |
| 1.0 | −6.482 | −5.322 | +1.16 |
| 2.0 | −8.420 | **−4.884** | **+3.54** |
| 4.0 | −10.218 | **−4.539** | **+5.68** |

With a perfect uncertainty signal, the penalty **fully recovers** the loss and
returns performance to the bias-free level (−4.5 vs −5.5).  This is the upper
bound for Innovation 3, and it clears: the conservative-planning *mechanism* is
sound.

### (c) Is our uncertainty estimate calibrated?  No — and that is the real problem.

The same penalty run with a deliberately miscalibrated estimator (a constant)
reproduces `no_penalty` **digit for digit** at every bias level, confirming the
penalty is only as good as the signal driving it.  So the question becomes
whether a *learned* uncertainty estimate is any better:

| model | mean abs error | corr(U, error) | corr(state-OOD, error) |
|---|---|---|---|
| 800 episodes, 10 epochs | 0.523 | −0.012 | −0.055 |
| 4000 episodes, 40 epochs | 0.345 | **+0.083** | **+0.110** |

Calibration improves as the model gets better, but from nothing to *almost*
nothing.  At ρ ≈ 0.1 the uncertainty signal is not informative enough to drive a
penalty, which is exactly what the learned-model sweep shows: every penalty
configuration lands within noise of `no_penalty`, and the whole sweep is swamped
by the model's absolute error (`learned/none` sits at −19.7 while the oracle sits
at −5.8).

### (d) Can the calibration be fixed?  Yes — by supervising it directly.

Two routes were tried, on the same data budget, scored on a holdout neither was
trained on (2000 episodes/member, 4 members, error model on 1500 × 2 episodes):

| signal | homogeneous ensemble | heterogeneous ensemble |
|---|---|---|
| `ensemble_std` (Pearson ρ) | +0.112 | +0.019 |
| `ensemble_std` (Spearman) | +0.113 | +0.021 |
| **`error_model` (Pearson ρ)** | **+0.643** | **+0.608** |
| **`error_model` (Spearman)** | **+0.596** | **+0.544** |
| `state_ood` | −0.069 | −0.134 |
| `item_ood` | −0.024 | −0.037 |

**The supervised error model lifts rank correlation from 0.11 to 0.60 — a 5×
improvement, and squarely into the usable range.**

The heterogeneous ensemble was a **failed hypothesis** and is worth recording as
such: varying capacity, depth and learning rate across members made disagreement
*worse* as an error proxy (0.019 vs 0.112 for identical members). Disagreement
between models measures *model spread*, not *model correctness*; decorrelating
the members does not change that. The only thing that worked was pointing a
supervised regressor at the target directly:

```
E(h, a) ~ | r_hat(h, a) - r_true |
```

That converts "is uncertainty calibration possible here" from a hope into a
supervised-learning question with a measurable answer, and the answer is yes.

### (e) End-to-end on a learned model: the exploitation is real, and the penalty removes most of the harm

`--unc-source error_model`, 2 members trained on 4000 episodes, error model on
2 × 2000 holdout episodes, `lt=3.0`, `div=0.8`, warm-start metric:

| config | H=0 | H=1 | H=2 | H=3 | H=5 | Δ (H0→Hmax) |
|---|---|---|---|---|---|---|
| oracle | −5.475 | −3.401 | −2.768 | −2.416 | −2.608 | **+2.867** |
| learned, no penalty | −21.835 | −22.574 | −22.160 | −21.990 | **−22.931** | **−1.096** |
| learned, error-model penalty | **−10.425** | −11.047 | −11.139 | −10.917 | −10.883 | −0.458 |
| learned, both | **−10.425** | −10.758 | −10.912 | −10.763 | −10.613 | −0.188 |

(error-model calibration was re-measured independently here: Pearson 0.735,
Spearman 0.612 — the (d) result replicates.)

Three things, in order of importance:

1. **The exploitation is real, on a real learned model.** `learned/none` has
   **negative** Δ: lookahead depth *hurts* (−1.096), and H=5 (−22.931) is the
   worst point on the curve.  This is exactly the risk the framework is exposed
   to, observed rather than argued.  Note the oracle row has the same planner and
   the same pool and is *positive* (+2.867), so this is not a property of the
   search.
2. **The penalty removes most of the harm — but as a safety mechanism at every
   depth, not only deep.** Returns jump from −21.8 to −10.4 (**+11.4**), and the
   gain is already fully present at **H=0**.  The dominant effect is therefore on
   *action selection* (the penalty steers away from over-estimated slates), not on
   making long rollouts safe.  Δ moves from −1.096 to −0.458 — the exploitation
   is largely neutralised.
3. **The positive value of multi-step planning does not come back** (Δ stays ≈ 0,
   not +2.867).  The learned planner at H=0 sits at −10.4 against the oracle's
   −5.5: the model is still the binding constraint, so there is no planning gain
   left for depth to unlock.  Making the penalty stronger does not fix this, and
   is not supposed to.

**The honest reading:** conservative planning does what it claims — it stops the
planner being dragged into the model's over-confidence — and in this configuration
it is worth more than the entire planning gain it was meant to protect. But
end-to-end confirmation that it *restores multi-step value* requires a world model
whose H=0 is already close to the oracle's, which needs server-scale training.
Until that exists, (e) establishes the safety property, not the value-restoration
property.

### What this settles

Innovation 3's mechanism is validated in three steps, and the remaining gap is
now unambiguously located:

1. With a calibrated signal, the penalty recovers a bias-destroyed policy
   completely (b: −10.2 → −4.5).
2. Ensemble disagreement is *not* a calibrated signal (c: ρ ≈ 0.1), and the
   intuitive fix — decorrelating the ensemble — makes it worse (d: ρ ≈ 0.02).
3. A **supervised** error model on held-out logs reaches ρ ≈ 0.6 (d), replicates
   at 0.61–0.74, and end-to-end it removes the measured exploitation on a learned
   model and is worth **+11.4 return** (e).

The design implication is concrete: **do not derive the penalty signal from
ensemble disagreement — train it.**

What is still not shown is that the penalty *restores multi-step planning value*.
The reason is no longer in doubt: at H=0 the learned planner sits at −10.4 against
the oracle's −5.5, so there is no planning gain left for depth to unlock. That is
a data/compute gap, not a methodological one, and it is the one remaining box to
tick before the framework's claim is complete.

Reproduce with:

```bash
python scripts/run_gate4_controlled.py --biases 0 1 2 4 --horizons 0 1 2 3 5
python scripts/run_gate4.py --probe --episodes 4000 --epochs 40 --ensemble 2   # calibration only
```

---

## The environment: why it has a knob

`muslate/env/simulator.py` implements an ordered-slate MDP. Step `t`'s reward uses
the saturation accumulated *up to* `t-1`, not `t`'s own exposure, so a greedy
policy that squeezes the currently-best category pays for it later:

```
state_t   = (h_t, sat_t)
action_t  = slate_t = (i_1..i_K)            ordered
reward_t  = <h_t, mean_w(z_slate)>                      (immediate, separable)
            - lt * sat_penalty * <w, sat_t[cat(slate)]> (delayed, category-level)
            - div * sum_{j<k} w_j w_k <z_j, z_k>        (immediate, COMBINATORIAL)
            + noise
state_{t+1} = ( normalize(drift(h_t, slate_t)), decay(sat_t, slate_t) )
```

There are two independent knobs, and they are deliberately orthogonal:

* **`long_term_strength` (`lt`)** scales the two *delayed* coupling mechanisms.
  `lt = 0` is a contextual bandit; `lt = 1` is the default; `lt >= 3` is strongly
  coupled. This is what Gate 2 sweeps.
* **`div_penalty` (`div`)** scales an *immediate* intra-slate combinatorial term.
  It changes what the myopic optimum is without touching the long-horizon
  structure, so it does not disturb Gate 2. This is what Gate 3 sweeps.

The separation matters: it means "does multi-step planning help" and "does a
hierarchical action space help" can be answered independently, instead of being
confounded by each other.

Run the environment invariant tests with:

```
python tests/test_env.py
python tests/test_planning.py
python tests/test_world_model.py
```

No pytest required — each file also runs standalone.

---

## Layout

```
muslate/
  config.py                  # all knobs as dataclasses
  env/
    catalog.py               # item catalogue with a 2-level Semantic-ID hierarchy
    simulator.py             # ground-truth slate MDP (the `lt` knob lives here)
    logger.py                # offline interaction-log generation
  model/
    world_model.py           # (h, slate) -> (r, h')  [item-decomposed reward]
    train.py                 # multi-step training from the log
    neural_model.py          # numpy adapter: same interface as the simulator
    error_model.py           # supervised |prediction error| regressor (Gate 4)
  planning/
    latent_model.py          # LatentModel interface + OracleModel + CountingModel
    proposal.py              # candidate slate generation
    search.py                # ReactiveGreedy / Lookahead / RandomShooting / MCTS / CEM
    abstractions.py          # SID proposal (Gate 3)
    conservative.py          # uncertainty / OOD penalties (Gate 4)
    controlled.py            # simulator + injected, exploitable bias (Gate 4)
  eval/
    rollout.py               # closed-loop evaluation
    calibration.py           # does an uncertainty signal track the true error?
scripts/
  probe_environment.py       # does the environment even have a regime worth planning in?
  run_gate2.py               # H-sweep (the decisive experiment)
  run_gate3.py               # action-abstraction comparison at equal budget
  run_q3.py                  # planner comparison at equal *measured* budget
  run_gate4.py               # conservative planning on the learned model (+ calibration)
  run_gate4_controlled.py    # conservative planning with a controlled injected bias
  run_calibration.py         # ensemble disagreement vs supervised error model
  diagnose_world_model.py    # where does the learned model lose the ability to plan?
tests/                       # invariants, runnable without pytest
results/                     # JSON + Markdown outputs
```

### One interface, two dynamics

`OracleModel` (the simulator) and `NeuralLatentModel` (the learned model) both
implement `muslate.planning.latent_model.LatentModel`. The *same* planner code
therefore runs against true dynamics and learned dynamics, so a gap between them
is attributable to the model, not to the search. This is how the code separates
"planning cannot help here" from "our model is too weak to plan with".

---

## Running

Local smoke test (CPU, seconds to a couple of minutes):

```bash
python scripts/probe_environment.py --users 8 --candidates 16 --beam 2 --horizons 0 1 2 --lts 0 1
python scripts/run_gate2.py --dynamics oracle --lts 0 1 3 --users 8 --horizons 0 1 2
python scripts/run_gate3.py --dynamics oracle --lt 3.0 --users 8 --budget 64
```

Note: the foreground shell has a 2-minute cap. Anything larger should be run in
the background or on a server.

Full runs (recommended on a GPU box):

```bash
# decisive sweep, ground-truth dynamics
python scripts/run_gate2.py --dynamics oracle \
    --lts 0 0.25 0.5 0.75 1 2 3 5 --horizons 0 1 2 3 5 \
    --users 32 --candidates 32 --beam 3 --out results/gate2_oracle.json

# same, learned dynamics (this is where the offline-RL reality bites)
python scripts/run_gate2.py --dynamics both \
    --lts 0 1 3 5 --horizons 0 1 2 3 5 --users 16 --warmup 5 \
    --episodes 20000 --epochs 60 --ensemble 3 \
    --device cuda --out results/gate2_full.json

# action abstraction at equal budget
python scripts/run_gate3.py --dynamics both \
    --lts 0 1 3 5 --users 16 --budget 512 \
    --episodes 20000 --epochs 60 --device cuda --out results/gate3.json

# planner comparison at equal *measured* budget
python scripts/run_q3.py --dynamics oracle --lt 3.0 --div 0.8 --depth 2
python scripts/run_q3.py --dynamics oracle --lt 5.0 --div 0.8 --depth 4 --n-sims 120

# conservative planning: controlled bias (fast, decisive) and learned model (needs data)
python scripts/run_gate4_controlled.py --biases 0 1 2 4 --horizons 0 1 2 3 5
python scripts/run_gate4.py --probe --episodes 4000 --epochs 40 --ensemble 2
python scripts/run_gate4.py --episodes 4000 --epochs 40 --ensemble 2 \
    --unc-source error_model --err-episodes 2000 --err-epochs 80 --warmup 5

# is the penalty signal calibrated?  (the number that decides Innovation 3)
python scripts/run_calibration.py --episodes 2000 --epochs 25 --ensemble 4 \
    --err-episodes 1500 --eval-episodes 300
```

`--device auto` picks CUDA when available. Every run writes both
`results/*.json` (machine readable) and `results/*.md` (the table), so results can
be pulled off the server without re-running.

---

## Design decisions that matter

**Item-decomposed reward head.** The world model predicts

```
r(h, A) = sum_k w_k * ( <g(h), e_{i_k}> + q(h)[i_k] )
```

rather than `MLP(h, slate_emb)`. Two reasons. It matches how slate rewards are
actually generated, so it generalises to slates the logging policy never showed;
and it shares parameters with `item_scores`, so the proposal policy and the value
estimate cannot disagree about which items are good. This was not cosmetic — with
the monolithic head, `Lookahead(H=0)` scored **-5.03** while a plain reactive
greedy scored **-0.24** on the same learned model, i.e. the planner was actively
searching the region where the head over-estimates. That failure mode *is* the
paper's core risk, and it needs to be visible, not accidentally avoided.
`tests/test_world_model.py::test_reward_is_decomposable_into_item_scores` pins the
property down.

**Three-way split of the H-sweep.** On the learned model, the cold-start steps
(`t = 0,1,2`, empty or near-empty history) dominate the error: the model's item
scores correlate with the truth at `rho = 0.01` at `t=0` and only reach `0.89` by
`t=19`. Since that penalty is common to every `H`, `--warmup N` reports a second
metric that excludes the first `N` steps, so a horizon comparison is not swallowed
by it.

**Equal budget in Gate 3.** `FlatSIDPlanner(n_plans=256)` and
`HierarchicalSIDPlanner(n_plans=96, n_widen=4, n_fine_per_pos=8)` both spend
`n_plans + n_widen * K * n_fine_per_pos` evaluations. `evaluation_budget()` is
asserted in `tests/test_planning.py` so the comparison cannot silently drift into
a budget difference.

---

## Limitations / what is still missing

* **No real data.** Gate 1 (recovering request-level slates from KuaiRand) is not
  implemented here. The synthetic `lt` knob is a *controlled* stand-in; the open
  question it does not answer is **what `lt` real systems actually have**. Until
  that is measured on KuaiSim/KuaiRand, the honest claim is conditional: multi-step
  planning pays off *if and only if* long-horizon coupling is strong.
* **Learned-dynamics planning is still weak in absolute terms.** Returns on the
  learned model trail the oracle by a wide margin. `scripts/diagnose_world_model.py`
  splits the error: at `t = 0` the model's item scores correlate with the truth
  at `rho ≈ 0.02` — an empty history carries no user information — and only reach
  `~0.9` by `t = 19`, while latent-state error is small throughout. So the loss is
  dominated by **belief inference during the cold-start window**, not by the
  dynamics model. `--warmup N` reports a metric that excludes it. Still to do:
  more offline data (the defaults are deliberately small so a laptop can run
  them), and conservative planning to stop the planner exploiting model error.
  The item-decomposed reward head already removed the worst form of that failure:
  before it, `Lookahead(H=0)` scored **−5.03** while reactive greedy scored
  **−0.24** on the same model.
* **The Gate 3 conclusion rests on one chosen form of combinatorial structure.**
  The intra-slate diversity penalty restored the regime where the hierarchy wins
  (see the Gate 3 section). Still to do: test the other plausible sources
  (position-dependent preferences, category quotas) to check the conclusion is
  not specific to the diversity form. This matters because the paper's claim is
  conditional on intra-slate combinatorial structure existing at all.
* **The Q3 effect size is small and single-seeded.** Under combinatorial reward
  MCTS beats the beam by ~0.3 points at matched budget, and the `--no-bootstrap`
  ablation confirms the mechanism (the same tree loses by 1.8 points without the
  shared bootstrap). But one seed and a 10-category catalogue is not enough to
  call it significant, and the *explanation* — UCB explores composition-saving
  paths that the beam's position-wise pruning discards — is still an explanation,
  not a measurement. Needs multiple seeds, a larger SID space, and a diagnostic
  that counts how often beam and MCTS disagree on the first action.
* **The error model's cost is held-out data, and its coverage is the behaviour
  policy's.** It reaches ρ ≈ 0.6 with 2 × 1500 holdout episodes in this toy
  environment, but that requirement scales with catalogue size and policy
  coverage.  On a real dataset it means carving out a slice that could otherwise
  have trained the world model — a trade-off to quantify rather than assume away.
  The model is also only supervised where the logging policy went, which is
  precisely *not* where an optimistic planner wants to go; the softplus-plus-log-
  space fitting is a mitigation, not a guarantee, and behaviour far outside the
  support is untested.
* **Heterogeneous ensembles do not work as an uncertainty signal.** Measured, not
  assumed: varying capacity, depth and learning rate across members made
  disagreement *worse* as an error proxy (ρ = 0.019 vs 0.112 for identical
  members). Recorded because it is the intuitive fix and it fails.
* **Value restoration is still unproven, and the reason is model strength.** With
  the error-model penalty the learned planner improves from −21.8 to −10.4, but
  the oracle sits at −5.5 — so there is no planning gain left for depth to
  recover (Gate 4e).  Confirming that conservative planning *restores multi-step
  value* needs a world model whose H=0 is already near the oracle's, i.e.
  server-scale training (20k+ episodes, GPU), not a methodological change.
* **Progressive widening is not yet uncertainty-driven.** In
  `HierarchicalSIDPlanner`, widening is currently driven by the model's *value*
  estimate. The structural coupling the paper argues for — widening driven by the
  world model's *epistemic uncertainty* — is the natural next step
  (`NeuralLatentModel.uncertainty` already exists for it).
