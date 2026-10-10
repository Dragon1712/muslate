"""Direct diagnostic: did the pairwise reward head actually learn the div term?

The planning-level comparison (Gate 3 on/off) is too noisy to settle this: a
single run's model-quality difference between the two arms is larger than the
effect being measured.  This script removes planning entirely and asks the
minimal question:

    does `pairwise=True` predict the environment reward better than
    `pairwise=False`, and *only* when the environment has a diversity term?

Design: a 2x2 grid over (environment div, reward head), same data budget and
seed.  The pattern that confirms the fix is a *cross-over*:

    env div = 0.0 :  on ~= off      (nothing extra to represent)
    env div = 0.8 :  on  <  off     (the pairwise term was the missing structure)

If `on` does not help even with div > 0, the misspecification hypothesis is wrong
and the learned-Gate-3 failure has some other cause.

Usage:
  python scripts/diagnose_pairwise.py --episodes 3000 --epochs 40
  python scripts/diagnose_pairwise.py --lt 3.0 --divs 0 0.8 --seeds 3 --device cuda
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
from muslate.model.neural_model import NeuralLatentModel  # noqa: E402
from muslate.model.train import get_device, train_world_model  # noqa: E402


def reward_prediction_error(nm: NeuralLatentModel, holdout) -> dict[str, float]:
    """Mean |predicted - true| reward on held-out logged transitions.

    Uses the same (belief, slate) pairing the planner queries, so this measures
    exactly the quantity planning consumes.
    """
    n_ep = int(holdout.episode_ids.max()) + 1
    t_len = holdout.horizon
    slates = holdout.slates.reshape(n_ep, t_len, holdout.slate_size)
    rewards = holdout.rewards.reshape(n_ep, t_len)

    latent = nm.encode_sequence(slates, rewards)                    # [E, T, dh]
    pred = nm.reward(
        latent.reshape(-1, latent.shape[-1]), slates.reshape(-1, holdout.slate_size)
    ).reshape(n_ep, t_len)

    err = np.abs(pred - rewards)
    # split cold-start from the rest, since t=0 belief carries no history
    return {
        "mae_all": float(err.mean()),
        "mae_t0": float(err[:, 0].mean()),
        "mae_after_t0": float(err[:, 1:].mean()),
        "bias": float((pred - rewards).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lt", type=float, default=3.0)
    ap.add_argument("--divs", type=float, nargs="+", default=[0.0, 0.8])
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--rollout-depth", type=int, default=3)
    ap.add_argument("--holdout-episodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1, help="repeat the whole grid this many times")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default="results/pairwise_diagnostic.json")
    args = ap.parse_args()

    device = get_device(args.device)
    results: dict = {"config": vars(args), "grid": {}}

    for div in args.divs:
        sim = SlateSim(EnvConfig(long_term_strength=args.lt, div_penalty=div))
        key = f"div={div}"
        results["grid"][key] = {}

        for pairwise in (False, True):
            arm = "pairwise" if pairwise else "unary_only"
            errs = []
            for s in range(args.seeds):
                log = generate_log(
                    sim, LogConfig(n_episodes=args.episodes, seed=4000 + 17 * s)
                )
                holdout = generate_log(
                    sim, LogConfig(n_episodes=args.holdout_episodes, seed=9000 + 17 * s)
                )
                cfg = WorldModelConfig(
                    epochs=args.epochs,
                    rollout_depth=args.rollout_depth,
                    pairwise=pairwise,
                    seed=1 + 11 * s,
                )
                print(f"[{key} / {arm} / seed {s}] training ...")
                model = train_world_model(sim, log, cfg, device, verbose=args.verbose)
                nm = NeuralLatentModel([model], device)
                errs.append(reward_prediction_error(nm, holdout))

            agg = {
                k: {"mean": float(np.mean([e[k] for e in errs])),
                    "std": float(np.std([e[k] for e in errs]))}
                for k in errs[0]
            }
            results["grid"][key][arm] = agg
            print(
                f"  {arm:<11} mae_all={agg['mae_all']['mean']:.4f} "
                f"mae_t0={agg['mae_t0']['mean']:.4f} "
                f"mae_after={agg['mae_after_t0']['mean']:.4f}"
            )

    lines = [
        f"Pairwise reward-head diagnostic  (lt={args.lt}, {args.episodes} ep, "
        f"{args.epochs} epochs, {args.seeds} seed(s))",
        "",
        "|pred - true| on held-out transitions.  The fix predicts a cross-over:",
        "with div = 0 the two heads tie, with div > 0 pairwise should win.",
        "",
        f"{'env':<10} | {'head':<11} | {'MAE(all)':>9} | {'MAE(t=0)':>9} | {'MAE(t>0)':>9} | {'bias':>8}",
    ]
    lines.append("-" * len(lines[-1]))
    for key, arms in results["grid"].items():
        for arm, agg in arms.items():
            lines.append(
                f"{key:<10} | {arm:<11} | {agg['mae_all']['mean']:>9.4f} | "
                f"{agg['mae_t0']['mean']:>9.4f} | {agg['mae_after_t0']['mean']:>9.4f} | "
                f"{agg['bias']['mean']:>+8.4f}"
            )
        if "unary_only" in arms and "pairwise" in arms:
            d = arms["unary_only"]["mae_all"]["mean"] - arms["pairwise"]["mae_all"]["mean"]
            lines.append(f"{'':<10} | {'-> gain':<11} | {d:>+9.4f} | {'':>9} | {'':>9} |")
    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text("# Pairwise diagnostic\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
