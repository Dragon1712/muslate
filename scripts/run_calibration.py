"""Can uncertainty calibration be fixed?  Homogeneous vs heterogeneous ensemble
vs a supervised holdout error model.

Gate 4 established that the conservative-planning *mechanism* works with a
calibrated signal and does nothing without one, and that our ensemble
disagreement sat at ρ ≈ 0.1.  This runner attacks that number directly:

  signal                       where it comes from
  ---------------------------  ------------------------------------------------
  ensemble_std (homogeneous)   members identical except for init/data seed
  ensemble_std (heterogeneous) members differ in capacity, depth and LR
  error_model                  supervised regression of |pred - true| on holdout
  state_ood                    distance from the logged belief distribution

Everything is scored on a holdout used for neither training.  The decision rule is
simple: a signal is usable for planning if its rank correlation with the true
error is materially above the homogeneous baseline.

Usage:
  python scripts/run_calibration.py --lt 3.0 --div 0.8 --episodes 1500 --ensemble 4
  python scripts/run_calibration.py --episodes 6000 --ensemble 4 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig, LogConfig, WorldModelConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.env.logger import generate_log  # noqa: E402
from muslate.eval.calibration import collect_latents, signal_report  # noqa: E402
from muslate.model.error_model import NeuralErrorModel, train_error_model  # noqa: E402
from muslate.model.neural_model import NeuralLatentModel  # noqa: E402
from muslate.model.train import (  # noqa: E402
    get_device,
    train_heterogeneous_ensemble,
    train_world_model,
)
from muslate.planning.conservative import BehaviorSupport, LatentSupport  # noqa: E402


def train_homogeneous(sim, log_cfgs, base_cfg, device, verbose):
    models = []
    for i in range(base_cfg.n_ensemble):
        cfg = WorldModelConfig(
            epochs=base_cfg.epochs,
            rollout_depth=base_cfg.rollout_depth,
            seed=base_cfg.seed + 100 * i,
        )
        log = generate_log(sim, log_cfgs[i % len(log_cfgs)])
        if verbose:
            print(f"  [homo {i + 1}/{base_cfg.n_ensemble}] seed={cfg.seed}")
        models.append(train_world_model(sim, log, cfg, device, verbose=verbose))
    return models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lt", type=float, default=3.0)
    ap.add_argument("--div", type=float, default=0.8)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--rollout-depth", type=int, default=3)
    ap.add_argument("--err-episodes", type=int, default=600)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--err-epochs", type=int, default=60)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default="results/calibration.json")
    args = ap.parse_args()

    sim = SlateSim(EnvConfig(long_term_strength=args.lt, div_penalty=args.div))
    device = get_device(args.device)
    results: dict = {"config": vars(args)}

    base_cfg = WorldModelConfig(
        n_ensemble=args.ensemble, epochs=args.epochs, rollout_depth=args.rollout_depth
    )
    train_log_cfgs = [
        LogConfig(n_episodes=args.episodes, seed=1000 + 137 * i) for i in range(args.ensemble)
    ]
    # error model is fit on data the *world models* never saw
    err_logs = [
        generate_log(sim, LogConfig(n_episodes=args.err_episodes, seed=7000 + i)) for i in range(2)
    ]
    # and everything is scored on a third, independent holdout
    eval_logs = [generate_log(sim, LogConfig(n_episodes=args.eval_episodes, seed=9500))]
    item_support = BehaviorSupport(err_logs[0])

    for kind in ("homogeneous", "heterogeneous"):
        print(f"\n=== {kind} ensemble ===")
        t0 = time.perf_counter()
        if kind == "homogeneous":
            models = train_homogeneous(sim, train_log_cfgs, base_cfg, device, args.verbose)
        else:
            models = train_heterogeneous_ensemble(
                sim, train_log_cfgs, base_cfg, device, verbose=args.verbose
            )
        nm = NeuralLatentModel(models, device)
        # support built from in-distribution data the ensemble did not train on
        state_support = LatentSupport(collect_latents(nm, err_logs[0]))
        print(f"  trained in {time.perf_counter() - t0:.1f}s")

        t0 = time.perf_counter()
        em = train_error_model(
            nm,
            sim,
            err_logs,
            epochs=args.err_epochs,
            device=device,
            seed=1,
            verbose=args.verbose,
        )
        nem = NeuralErrorModel(em, device)
        print(f"  error model fitted in {time.perf_counter() - t0:.1f}s")

        report = signal_report(
            nm,
            eval_logs,
            error_model=nem,
            state_support=state_support,
            item_support=item_support,
            stride=2,
        )
        results[kind] = report
        print(f"  target: {json.dumps(report['_target'])}")
        for name, row in report.items():
            if name.startswith("_"):
                continue
            print(
                f"    {name:<16} pearson={row['pearson']:+.4f}  spearman={row['spearman']:+.4f}"
                f"  std={row['signal_std']:.5f}"
            )

    # -- comparison table --------------------------------------------------
    signals = sorted({k for kind in ("homogeneous", "heterogeneous") for k in results[kind] if not k.startswith("_")})
    lines = [
        f"Uncertainty calibration (lt={args.lt}, div={args.div}, {args.ensemble} members, "
        f"{args.episodes} episodes each)",
        "",
        f"target error: mean={results['homogeneous']['_target']['mean_abs_error']}, "
        f"std={results['homogeneous']['_target']['error_std']}, "
        f"n={results['homogeneous']['_target']['n_pairs']}",
        "",
        f"{'signal':<16} | {'homo rho':>9} | {'hetero rho':>10} | {'homo rank':>10} | {'hetero rank':>11}",
    ]
    lines.append("-" * len(lines[-1]))
    for name in signals:
        h = results["homogeneous"].get(name, {})
        t = results["heterogeneous"].get(name, {})
        lines.append(
            f"{name:<16} | {h.get('pearson', float('nan')):>+9.4f} | {t.get('pearson', float('nan')):>+10.4f}"
            f" | {h.get('spearman', float('nan')):>+10.4f} | {t.get('spearman', float('nan')):>+11.4f}"
        )
    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text("# Calibration\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
