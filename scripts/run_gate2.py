"""Gate 2: is multi-step planning worth anything over one-step lookahead?

The question that decides the paper.  MF-MDP reports that "a single step of
lookahead already captures most of the gain" in a retrieval setting.  This runner
answers it *as a function of how strongly the environment couples the present to
the future*, which is the only honest way to settle it:

    for lt in long_term_strength grid:
        for H in {0, 1, 2, 3, 5}:
            return(planner with lookahead depth H)

Run with ground-truth dynamics first (fast, no model error) and then with the
learned world model (what deployment looks like).  If the two curves agree, the
conclusion is about the problem, not about the model.

Usage:
  python scripts/run_gate2.py --dynamics oracle
  python scripts/run_gate2.py --dynamics learned --episodes 2000 --epochs 30
  python scripts/run_gate2.py --dynamics both --lts 0 1 3 5 --horizons 0 1 2 3 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    reward_prediction_error,
)
from muslate.model.neural_model import NeuralLatentModel  # noqa: E402
from muslate.model.train import get_device, train_world_model  # noqa: E402
from muslate.planning.latent_model import OracleModel  # noqa: E402
from muslate.planning.proposal import ProposalPolicy  # noqa: E402
from muslate.planning.search import Lookahead, ReactiveGreedy  # noqa: E402


def run_one_dynamics(sim: SlateSim, model, latent_fn, args, horizons: list[int]) -> dict:
    rows: dict[str, dict] = {}

    greedy = ReactiveGreedy(model, gamma=sim.cfg.gamma)
    t0 = time.perf_counter()
    res = evaluate(sim, greedy, latent_fn, args.users, tail_from=args.warmup)
    rows["reactive"] = {
        "return": round(res.mean, 4),
        "tail": round(res.tail_mean, 4),
        "se": round(res.se, 4),
        "latency_ms": round(res.latency_ms, 3),
        "wall_s": round(time.perf_counter() - t0, 1),
    }

    for h in horizons:
        proposal = ProposalPolicy(
            model,
            n_candidates=args.candidates,
            seed=args.proposal_seed,
            mode="item_perturb",
            n_perturb=1,
        )
        planner = Lookahead(model, proposal, depth=h, beam_width=args.beam, gamma=sim.cfg.gamma)
        res = evaluate(sim, planner, latent_fn, args.users, tail_from=args.warmup)
        rows[f"H={h}"] = {
            "return": round(res.mean, 4),
            "tail": round(res.tail_mean, 4),
            "se": round(res.se, 4),
            "tail_se": round(res.tail_se, 4),
            "latency_ms": round(res.latency_ms, 3),
            "n_decisions": res.n_decisions,
        }
    return rows


def _table(results: dict, metric: str, title: str) -> list[str]:
    lines = [title]
    header = (
        f"{'lt':>5} | {'dyn':<8} | "
        + " | ".join(f"{'H=' + str(h):>8}" for h in results["horizons"])
        + f" | {'reactive':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for lt_key, by_dyn in results["rows"].items():
        for dyn, rows in by_dyn.items():
            cells = " | ".join(
                f"{rows[f'H={h}'][metric]:>8.3f}" if f"H={h}" in rows else f"{'-':>8}"
                for h in results["horizons"]
            )
            lines.append(
                f"{lt_key.replace('lt=', ''):>5} | {dyn:<8} | {cells} | "
                f"{rows['reactive'][metric]:>8.3f}"
            )
    return lines


def format_table(results: dict) -> str:
    lines = _table(results, "return", "[full episode]")
    lines.append("")
    lines.append(f"gain over H=0, full episode (warmup excluded from the metric only when --warmup > 0):")
    for lt_key, by_dyn in results["rows"].items():
        for dyn, rows in by_dyn.items():
            base = rows["H=0"]["return"]
            gains = "  ".join(
                f"H={h}:{rows[f'H={h}']['return'] - base:+.3f}"
                for h in results["horizons"]
                if f"H={h}" in rows
            )
            lines.append(f"  {lt_key:<10} {dyn:<8} {gains}")

    if results.get("warmup", 0) > 0:
        lines.append("")
        lines.extend(_table(results, "tail", f"[warm-start, steps >= {results['warmup']} only]"))
        lines.append("")
        lines.append("gain over H=0, warm start:")
        for lt_key, by_dyn in results["rows"].items():
            for dyn, rows in by_dyn.items():
                base = rows["H=0"]["tail"]
                gains = "  ".join(
                    f"H={h}:{rows[f'H={h}']['tail'] - base:+.3f}"
                    for h in results["horizons"]
                    if f"H={h}" in rows
                )
                lines.append(f"  {lt_key:<10} {dyn:<8} {gains}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics", choices=["oracle", "learned", "both"], default="oracle")
    ap.add_argument("--lts", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument(
        "--div",
        type=float,
        default=0.0,
        help="intra-slate diversity penalty (an immediate combinatorial term; "
        "independent of long_term_strength, so it does not change Gate 2's semantics)",
    )
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    ap.add_argument("--users", type=int, default=16)
    ap.add_argument("--candidates", type=int, default=32)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--proposal-seed", type=int, default=13)
    ap.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="also report a warm-start metric that excludes the first N steps (cold-start belief inference)",
    )
    # learned-model knobs
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--rollout-depth", type=int, default=3)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="results/gate2.json")
    args = ap.parse_args()

    results: dict = {
        "horizons": args.horizons,
        "rows": {},
        "model_errors": {},
        "warmup": args.warmup,
    }

    for lt in args.lts:
        key = f"lt={lt}"
        results["rows"][key] = {}
        print(f"\n=== long_term_strength = {lt} ===")
        sim = SlateSim(EnvConfig(long_term_strength=lt, div_penalty=args.div))

        if args.dynamics in ("oracle", "both"):
            oracle = OracleModel(sim)
            t0 = time.perf_counter()
            rows = run_one_dynamics(sim, oracle, make_oracle_latent_fn(oracle, sim), args, args.horizons)
            results["rows"][key]["oracle"] = rows
            print(f"  oracle dynamics done in {time.perf_counter() - t0:.1f}s")
            print("  " + "  ".join(f"H={h}:{rows[f'H={h}']['return']:.3f}" for h in args.horizons))

        if args.dynamics in ("learned", "both"):
            device = get_device(args.device)
            log_cfg = LogConfig(n_episodes=args.episodes, seed=1000 + int(lt * 10))
            log = generate_log(sim, log_cfg)
            models = []
            for i in range(args.ensemble):
                wm_cfg = WorldModelConfig(
                    n_ensemble=args.ensemble,
                    epochs=args.epochs,
                    rollout_depth=args.rollout_depth,
                    seed=100 * i + 1,
                )
                print(f"  training world model {i + 1}/{args.ensemble} on {device} ...")
                models.append(train_world_model(sim, log, wm_cfg, device, verbose=True))
            neural = NeuralLatentModel(models, device)

            errs = reward_prediction_error(sim, neural, n_episodes=32, seed=7)
            results["model_errors"][key] = {k: round(v, 4) for k, v in errs.items()}
            print(f"  world-model error: {results['model_errors'][key]}")

            rows = run_one_dynamics(sim, neural, make_learned_latent_fn(neural), args, args.horizons)
            results["rows"][key]["learned"] = rows
            print("  " + "  ".join(f"H={h}:{rows[f'H={h}']['return']:.3f}" for h in args.horizons))

    table = format_table(results)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text("# Gate 2 - multi-step vs one-step lookahead\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"\nwrote {out} and {md}")


if __name__ == "__main__":
    main()
