"""B: push `item_scores` precision to its ceiling, then ask whether the
hierarchy recovers.

Motivation
----------
D5 separated the two possible sources of score error:

    ceiling (score head sees the TRUE state)  rho = 0.993
    model's own scores                        rho = 0.720  (steady state 0.84)

so the score *head* is essentially not the problem (0.007 of the loss); the
**belief** is (0.272).  Raising model capacity or retuning the head is therefore
wasted effort unless it actually moves the belief.  Two knobs could:

  * ``latent_dim`` -- raw information capacity of the belief
  * ``w_state``    -- how hard the latent is pushed onto the true state

This script sweeps them, reports the steady-state D2 per setting, and -- the part
that actually matters -- re-runs the learned-dynamics Gate 3 comparison for each.
A high rho on its own says nothing about the method: the question is whether
``sid_hier - flat_uniform`` turns positive once the scores are good.

Run (server):
  python -u scripts/sweep_score_precision.py --device cuda --out results/score_precision.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig, LogConfig, WorldModelConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.env.logger import generate_log  # noqa: E402
from muslate.eval.calibration import spearman  # noqa: E402
from muslate.model.neural_model import NeuralLatentModel  # noqa: E402
from muslate.model.train import get_device, train_world_model  # noqa: E402
from scripts.diagnose_semantics import d2_score_rank, d5_head_ceiling  # noqa: E402
from scripts.run_gate3 import build_variants, make_learned_latent_fn  # noqa: E402


def steady_rho(d2: dict, skip: int = 3) -> float:
    """Mean per-timestep rho ignoring the cold start.

    ``t=0`` has no history, so its rho is ~0.01 for information-theoretic reasons
    and averaging it in would conflate 'cannot be known' with 'not learned'.
    """
    by_t = d2["spearman_by_t"]
    vals = [v for t, v in by_t.items() if int(t) >= skip]
    return round(float(np.mean(vals)), 4) if vals else float("nan")


def evaluate_arm(sim: SlateSim, nm: NeuralLatentModel, eval_cfg, holdout, budget, depth, users):
    """Gate 3 under learned dynamics, budget-matched."""
    from muslate.eval.rollout import evaluate as rollout_eval

    planners = build_variants(nm, sim, budget, depth, 0)
    latent_fn = make_learned_latent_fn(nm)
    res = rollout_eval(sim, planners["flat_uniform"], latent_fn, users, tail_from=0)
    rows = {"flat_uniform": round(res.mean, 4)}
    for name in ("flat_policy", "sid_hier"):
        r = rollout_eval(sim, planners[name], latent_fn, users, tail_from=0)
        rows[name] = round(r.mean, 4)
    rows["hier_minus_flat"] = round(rows["sid_hier"] - rows["flat_uniform"], 4)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lt", type=float, default=3.0)
    ap.add_argument("--div", type=float, default=0.8)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--latent-dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--w-states", type=float, nargs="+", default=[1.0, 10.0])
    ap.add_argument(
        "--w-scores",
        type=float,
        nargs="+",
        default=[],
        help="with --no-oracle-state, sweep the direct item_scores loss weight "
        "instead, since there is no state target to weight",
    )
    ap.add_argument(
        "--oracle-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="supervise the latent against the true simulator state. Disabling it "
        "is the honest partially-observed setting and the control that says whether "
        "the belief can be sharpened WITHOUT privileged state access.",
    )
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--users", type=int, default=16)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="results/score_precision.json")
    args = ap.parse_args()

    sim = SlateSim(EnvConfig(long_term_strength=args.lt, div_penalty=args.div))
    device = get_device(args.device)
    log = generate_log(sim, LogConfig(n_episodes=args.episodes, seed=4000))
    holdout = generate_log(sim, LogConfig(n_episodes=args.holdout, seed=9000))
    ceiling = d5_head_ceiling(sim, holdout)
    print(f"ceiling (head sees the true state): rho={ceiling['ceiling_spearman']}", flush=True)

    # With state supervision there is a state target to weight; without it the only
    # direct handle on the belief is the score loss, so that is what gets swept.
    if args.oracle_state:
        knobs = [("w_state", v) for v in args.w_states]
    else:
        knobs = [("w_score", v) for v in (args.w_scores or [1.0])]
        if not args.w_scores:
            print("note: --no-oracle-state with no --w-scores; using w_score=1.0", flush=True)

    results: dict = {"config": vars(args), "ceiling": ceiling, "arms": {}}
    for ld in args.latent_dims:
        for kname, kval in knobs:
            tag = f"ld{ld}_{kname[2:]}{kval:g}"
            print(f"\n=== {tag} ===", flush=True)
            cfg = WorldModelConfig(
                latent_dim=ld,
                oracle_state=args.oracle_state,
                epochs=args.epochs,
                pairwise=True,
                seed=1,
                **{kname: kval},
            )
            model = train_world_model(sim, log, cfg, device, verbose=False)
            nm = NeuralLatentModel([model], device)

            d2 = d2_score_rank(nm, sim, holdout)
            rows = evaluate_arm(sim, nm, cfg, holdout, args.budget, args.depth, args.users)
            entry = {
                "latent_dim": ld,
                "oracle_state": args.oracle_state,
                kname: kval,
                "d2_mean": d2["spearman_mean"],
                "d2_t0": d2["spearman_t0"],
                "d2_steady": steady_rho(d2),
                "by_t": d2["spearman_by_t"],
                "gate3": rows,
            }
            results["arms"][tag] = entry
            print(
                f"  D2 mean={entry['d2_mean']:+.4f}  t0={entry['d2_t0']:+.4f}  "
                f"steady={entry['d2_steady']:+.4f}",
                flush=True,
            )
            print(
                f"  gate3 flat_uniform={rows['flat_uniform']:+.3f}  "
                f"flat_policy={rows['flat_policy']:+.3f}  sid_hier={rows['sid_hier']:+.3f}  "
                f"hier-flat={rows['hier_minus_flat']:+.3f}",
                flush=True,
            )

    lines = [
        f"Score-precision sweep  (lt={args.lt}, div={args.div}, {args.episodes} ep, "
        f"{args.epochs} epochs, budget={args.budget}, depth={args.depth})",
        "",
        f"ceilings: head-with-true-state rho={ceiling['ceiling_spearman']}",
        "'steady' ignores t=0, whose rho is ~0 for information-theoretic reasons.",
        "",
        f"{'arm':<16} | {'D2 mean':>8} | {'D2 steady':>9} | {'flat_uniform':>12} | "
        f"{'flat_policy':>11} | {'sid_hier':>9} | {'hier-flat':>9}",
    ]
    lines.append("-" * len(lines[-1]))
    for tag, e in results["arms"].items():
        g = e["gate3"]
        lines.append(
            f"{tag:<16} | {e['d2_mean']:>+8.4f} | {e['d2_steady']:>+9.4f} | "
            f"{g['flat_uniform']:>+12.3f} | {g['flat_policy']:>+11.3f} | "
            f"{g['sid_hier']:>+9.3f} | {g['hier_minus_flat']:>+9.3f}"
        )
    table = "\n".join(lines)
    print("\n" + table, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(
        "# Score-precision sweep\n\n```\n" + table + "\n```\n", encoding="utf-8"
    )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
