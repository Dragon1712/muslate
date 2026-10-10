"""Gate 3: does the Semantic-ID hierarchy buy anything over flat slate actions?

Held fixed: the planner (``Lookahead`` at a fixed depth) and nominal proposal
settings. Actual reward/transition/score work differs and must be measured.
Varied: the candidate generator.

  flat_uniform   sample complete slates uniformly, keep the best
  flat_policy    sample complete slates around the greedy anchor
  sid_flat       one coarse level: category plans, completed greedily
  sid_hier       category plans + progressive widening + item refinement

The depth matters.  At ``--depth 0`` every variant is capped by the myopic
optimum, which greedy already attains, so all four tie by construction and the
comparison says nothing.  The abstraction only matters once the planner has to
*evaluate consequences* -- use ``--depth 2`` or higher.

Usage:
  python scripts/run_gate3.py --dynamics oracle --lt 3.0 --depth 2
  python scripts/run_gate3.py --dynamics both --lts 0 1 3 5 --depth 2 --episodes 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig, LogConfig, WorldModelConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.env.logger import generate_log  # noqa: E402
from muslate.eval.rollout import (  # noqa: E402
    evaluate,
    make_learned_latent_fn,
    make_oracle_latent_fn,
)
from muslate.model.neural_model import NeuralLatentModel  # noqa: E402
from muslate.model.error_model import NeuralErrorModel, train_error_model  # noqa: E402
from muslate.model.train import get_device, train_world_model  # noqa: E402
from muslate.planning.conservative import ConservativePenalty  # noqa: E402
from muslate.planning.abstractions import SIDProposal  # noqa: E402
from muslate.planning.latent_model import OracleModel  # noqa: E402
from muslate.planning.proposal import ProposalPolicy  # noqa: E402
from muslate.planning.search import Lookahead, ReactiveGreedy  # noqa: E402

WIDEN = 4
REFINE_POSITIONS = 5


def build_variants(
    model, sim: SlateSim, budget: int, depth: int, seed: int, adjust=None
) -> dict:
    """Allocate nominal proposal settings from ``budget``.

    This is not a total compute budget or strict equal-cost comparison. Pools
    shrink after deduplication, refinement attempts can be skipped, and recursive
    calls and joint transitions incur additional work. CountingModel records
    actual work; PlanningBudget can bound calibrated estimated operation cost.

    ``adjust`` is the conservative-planning hook: every planner here shares it, so
    a penalty comparison stays a comparison of *search structure*, not of penalties.
    """
    refine_target = budget // 2
    fine_per_pos = max(1, refine_target // (WIDEN * REFINE_POSITIONS))
    refine = WIDEN * REFINE_POSITIONS * fine_per_pos

    flat_out = budget

    sid_sampled = max(8, budget // 2)
    sid_out = max(8, budget - sid_sampled)

    hier_sampled = max(8, (budget - refine) // 2)
    hier_out = max(8, budget - hier_sampled - refine)

    def look(prop):
        return Lookahead(
            model, prop, depth=depth, beam_width=3, gamma=sim.cfg.gamma, adjust=adjust
        )

    return {
        "reactive": ReactiveGreedy(model, gamma=sim.cfg.gamma, adjust=adjust),
        "flat_uniform": look(
            ProposalPolicy(model, n_candidates=flat_out, seed=seed, mode="uniform")
        ),
        "flat_policy": look(
            ProposalPolicy(model, n_candidates=flat_out, seed=seed, mode="item_perturb", n_perturb=1)
        ),
        "sid_flat": look(
            SIDProposal(
                model,
                sim.n_coarse,
                sim.catalog.n_fine,
                n_sampled=sid_sampled,
                out_size=sid_out,
                refine_positions=0,
                seed=seed,
            )
        ),
        "sid_hier": look(
            SIDProposal(
                model,
                sim.n_coarse,
                sim.catalog.n_fine,
                n_sampled=hier_sampled,
                out_size=hier_out,
                widen=WIDEN,
                refine_positions=REFINE_POSITIONS,
                fine_per_pos=fine_per_pos,
                seed=seed,
            )
        ),
    }


def run_variants(sim: SlateSim, model, latent_fn, args, adjust=None) -> dict:
    planners = build_variants(model, sim, args.budget, args.depth, args.seed, adjust=adjust)
    rows = {}
    for name, planner in planners.items():
        res = evaluate(sim, planner, latent_fn, args.users, tail_from=args.warmup)
        rows[name] = {
            "return": round(res.mean, 4),
            "tail": round(res.tail_mean, 4),
            "se": round(res.se, 4),
            "latency_ms": round(res.latency_ms, 3),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics", choices=["oracle", "learned", "both"], default="oracle")
    ap.add_argument("--lts", type=float, nargs="+", default=[0.0, 1.0, 3.0, 5.0])
    ap.add_argument(
        "--div",
        type=float,
        default=0.0,
        help="intra-slate diversity penalty; >0 makes the slate reward non-separable, "
        "which is the regime where a coarse-to-fine action space can win",
    )
    ap.add_argument("--users", type=int, default=16)
    ap.add_argument("--budget", type=int, default=256, help="model evaluations per decision")
    ap.add_argument("--depth", type=int, default=2, help="lookahead depth (0 makes all variants tie)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--wm-seed",
        type=int,
        default=1,
        help="world-model training seed; vary it to get error bars (planning-level "
        "comparisons are otherwise dominated by single-run model-quality noise)",
    )
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument(
        "--pairwise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="add the intra-slate pairwise term to the reward head (P0a fix). "
        "--no-pairwise reproduces the old unary-only model for the ablation.",
    )
    ap.add_argument(
        "--lambda-unc",
        type=float,
        default=0.0,
        help="P2: weight on the learned error model in the planning value. 0 disables "
        "the penalty; >0 fits an error model on held-out logs and subtracts it, which "
        "is what stops the planner exploiting model error.",
    )
    ap.add_argument("--err-episodes", type=int, default=800)
    ap.add_argument("--err-epochs", type=int, default=60)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="results/gate3.json")
    args = ap.parse_args()

    results: dict = {"budget": args.budget, "depth": args.depth, "div_penalty": args.div, "rows": {}}

    for lt in args.lts:
        key = f"lt={lt}"
        results["rows"][key] = {}
        print(f"\n=== long_term_strength = {lt}, div_penalty = {args.div} ===")
        sim = SlateSim(EnvConfig(long_term_strength=lt, div_penalty=args.div))

        if args.dynamics in ("oracle", "both"):
            oracle = OracleModel(sim)
            rows = run_variants(sim, oracle, make_oracle_latent_fn(oracle, sim), args)
            results["rows"][key]["oracle"] = rows
            print("  oracle: " + "  ".join(f"{k}:{v['return']:.3f}" for k, v in rows.items()))

        if args.dynamics in ("learned", "both"):
            device = get_device(args.device)
            log = generate_log(sim, LogConfig(n_episodes=args.episodes, seed=1000 + int(lt * 10)))
            wm_cfg = WorldModelConfig(
                epochs=args.epochs, seed=args.wm_seed, pairwise=args.pairwise
            )
            print(f"  training world model on {device} ...")
            model = train_world_model(sim, log, wm_cfg, device, verbose=True)
            neural = NeuralLatentModel([model], device)

            # P2: conservative planning.  The error model is fitted on held-out logs
            # and answers "how wrong is the world model about this (state, slate)?" --
            # the signal Gate 4 showed *is* calibrated (rho ~ 0.8) where ensemble
            # disagreement is not (rho ~ 0.09).
            adjust = None
            if args.lambda_unc > 0:
                err_logs = [
                    generate_log(sim, LogConfig(n_episodes=args.err_episodes, seed=8000 + i))
                    for i in range(2)
                ]
                print(
                    f"  fitting error model on {args.err_episodes} x {len(err_logs)} "
                    f"holdout episodes ..."
                )
                em = train_error_model(
                    neural, sim, err_logs, epochs=args.err_epochs, device=device, seed=1
                )
                adjust = ConservativePenalty(
                    neural, lambda_unc=args.lambda_unc, uncertainty_fn=NeuralErrorModel(em, device).score
                )

            rows = run_variants(sim, neural, make_learned_latent_fn(neural), args, adjust=adjust)
            results["rows"][key]["learned"] = rows
            print("  learned: " + "  ".join(f"{k}:{v['return']:.3f}" for k, v in rows.items()))

    variants = ["reactive", "flat_uniform", "flat_policy", "sid_flat", "sid_hier"]
    lines = [
        f"Gate 3 - action abstraction at equal budget {args.budget}, lookahead depth {args.depth}",
        "",
        f"{'lt':>5} | {'dyn':<8} | " + " | ".join(f"{v:>12}" for v in variants),
    ]
    lines.append("-" * len(lines[-1]))
    for lt_key, by_dyn in results["rows"].items():
        for dyn, rows in by_dyn.items():
            cells = " | ".join(f"{rows[v]['return']:>12.3f}" for v in variants)
            lines.append(f"{lt_key.replace('lt=', ''):>5} | {dyn:<8} | {cells}")

    if args.warmup > 0:
        lines.append("")
        lines.append(f"[warm-start, steps >= {args.warmup}]")
        for lt_key, by_dyn in results["rows"].items():
            for dyn, rows in by_dyn.items():
                cells = " | ".join(f"{rows[v]['tail']:>12.3f}" for v in variants)
                lines.append(f"{lt_key.replace('lt=', ''):>5} | {dyn:<8} | {cells}")

    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text("# Gate 3\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
