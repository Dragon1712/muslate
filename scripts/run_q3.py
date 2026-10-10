"""Q3: is a tree actually needed?

The reviewer question the document flags is "is ordinary rollout search already
enough?"  This runner puts five planners on the same problem and, crucially,
reports the **actual number of world-model evaluations** each one spends, not a
nominal budget.  Different planners allocate their budget structurally
differently -- a beam expands the pool at *every* level, MCTS spreads the same
total over simulations, CEM re-samples its elite distribution -- so comparing
them at "the same settings" is meaningless without measuring.

  reactive   top-K by the model's item score, no search at all
  shooting   sample whole trajectories, keep the best first action
  beam       depth-limited best-first search over the pool (the Gate 2 planner)
  cem        cross-entropy method over a per-position item distribution
  mcts       PUCT tree search, zero leaf value

All of them share the proposal pool, the depth, and the discount, so the only
differences are *how* the budget is spent and *what* the value estimate is.

Usage:
  python scripts/run_q3.py --dynamics oracle --lt 3.0 --div 0.8 --depth 2
  python scripts/run_q3.py --dynamics oracle --lt 3.0 --div 0.0 --depth 2
  python scripts/run_q3.py --dynamics both --lt 3.0 --div 0.8 --episodes 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from muslate.model.train import get_device, train_world_model  # noqa: E402
from muslate.planning.latent_model import CountingModel, OracleModel  # noqa: E402
from muslate.planning.proposal import ProposalPolicy  # noqa: E402
from muslate.planning.search import (  # noqa: E402
    CEM,
    Lookahead,
    MCTS,
    RandomShooting,
    ReactiveGreedy,
)


def build_planners(model, sim: SlateSim, args) -> dict:
    def prop():
        return ProposalPolicy(model, n_candidates=args.candidates, seed=args.seed, mode="item_perturb")

    return {
        "reactive": ReactiveGreedy(model, gamma=sim.cfg.gamma),
        "shooting": RandomShooting(
            model, prop(), depth=args.depth, n_samples=args.samples, gamma=sim.cfg.gamma, seed=args.seed
        ),
        "beam": Lookahead(model, prop(), depth=args.depth, beam_width=args.beam, gamma=sim.cfg.gamma),
        "cem": CEM(
            model,
            prop(),
            depth=args.depth,
            n_samples=args.samples,
            n_elite=max(2, args.samples // 4),
            n_iters=args.cem_iters,
            gamma=sim.cfg.gamma,
            seed=args.seed,
        ),
        "mcts": MCTS(
            model,
            prop(),
            max_depth=args.depth,
            n_sims=args.n_sims,
            c_puct=args.c_puct,
            bootstrap=not args.no_bootstrap,
            gamma=sim.cfg.gamma,
            seed=args.seed,
        ),
    }


def run_suite(sim: SlateSim, base_model, latent_fn, args) -> dict:
    counter = CountingModel(base_model)
    planners = build_planners(counter, sim, args)

    rows = {}
    for name, planner in planners.items():
        counter.reset()
        res = evaluate(sim, planner, latent_fn, args.users, tail_from=args.warmup)
        evals = counter.n_slate_evals / max(res.n_decisions, 1)
        rows[name] = {
            "return": round(res.mean, 4),
            "tail": round(res.tail_mean, 4),
            "se": round(res.se, 4),
            "evals_per_decision": round(evals, 2),
            "latency_ms": round(res.latency_ms, 3),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamics", choices=["oracle", "learned", "both"], default="oracle")
    ap.add_argument("--lts", type=float, nargs="+", default=[3.0])
    ap.add_argument("--div", type=float, default=0.0)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=32, help="proposal pool size per node")
    ap.add_argument("--samples", type=int, default=32, help="samples for shooting / CEM")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--n-sims", type=int, default=32, help="MCTS simulations per decision")
    ap.add_argument("--cem-iters", type=int, default=3)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="ablation: give MCTS a hard zero leaf value instead of the same "
        "bootstrap Lookahead uses",
    )
    ap.add_argument("--users", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="results/q3.json")
    args = ap.parse_args()

    names = ["reactive", "shooting", "beam", "cem", "mcts"]
    results: dict = {"config": vars(args), "rows": {}}

    for lt in args.lts:
        key = f"lt={lt}"
        results["rows"][key] = {}
        print(f"\n=== lt={lt}, div={args.div}, depth={args.depth} ===")
        sim = SlateSim(EnvConfig(long_term_strength=lt, div_penalty=args.div))

        if args.dynamics in ("oracle", "both"):
            oracle = OracleModel(sim)
            rows = run_suite(sim, oracle, make_oracle_latent_fn(oracle, sim), args)
            results["rows"][key]["oracle"] = rows

        if args.dynamics in ("learned", "both"):
            device = get_device(args.device)
            log = generate_log(sim, LogConfig(n_episodes=args.episodes, seed=1000 + int(lt * 10)))
            wm_cfg = WorldModelConfig(epochs=args.epochs, seed=1)
            print(f"  training world model on {device} ...")
            wm = train_world_model(sim, log, wm_cfg, device, verbose=True)
            neural = NeuralLatentModel([wm], device)
            rows = run_suite(sim, neural, make_learned_latent_fn(neural), args)
            results["rows"][key]["learned"] = rows

    lines = [
        f"Q3 - planner comparison  (lt={args.lts}, div={args.div}, depth={args.depth}, pool={args.candidates})",
        "",
        f"{'lt':>5} | {'dyn':<8} | " + " | ".join(f"{n:>10}" for n in names) + "   <- return",
        "-" * 78,
    ]
    for lt_key, by_dyn in results["rows"].items():
        for dyn, rows in by_dyn.items():
            lines.append(
                f"{lt_key.replace('lt=', ''):>5} | {dyn:<8} | "
                + " | ".join(f"{rows[n]['return']:>10.3f}" for n in names)
            )
    lines.append("")
    lines.append(f"{'lt':>5} | {'dyn':<8} | " + " | ".join(f"{n:>10}" for n in names) + "   <- model evals / decision")
    lines.append("-" * 78)
    for lt_key, by_dyn in results["rows"].items():
        for dyn, rows in by_dyn.items():
            lines.append(
                f"{lt_key.replace('lt=', ''):>5} | {dyn:<8} | "
                + " | ".join(f"{rows[n]['evals_per_decision']:>10.1f}" for n in names)
            )

    best = {}
    for lt_key, by_dyn in results["rows"].items():
        for dyn, rows in by_dyn.items():
            b = max(names, key=lambda n: rows[n]["return"])
            best[f"{lt_key}/{dyn}"] = b
    lines.append("")
    lines.append("best return: " + ", ".join(f"{k}={v}" for k, v in best.items()))

    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text("# Q3\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
