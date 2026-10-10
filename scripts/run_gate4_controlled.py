"""Gate 4 (controlled): does conservative planning work when model error is real
and exactly measurable?

The learned-model version of this experiment is confounded -- a weak world model
destroys the planning gain with ordinary error, hiding the specific failure the
framework must survive.  This runner uses :class:`BiasedOracleModel`, which
injects a known, exploitable bias into otherwise perfect dynamics, and answers
two questions separately:

  (a) **Does exploitation happen?**  With a bias injected along an unknown slate
      direction, deeper planning should find that direction and get *worse* in
      truth (even as it gets better under the model).
  (b) **Would a good uncertainty estimate stop it?**  With the *perfect*
      uncertainty signal available, does the ``Q - lambda*U`` penalty remove the
      exploitation?  And what happens with a deliberately miscalibrated one?

(b) is an upper bound on Innovation 3: if conservative planning cannot work with
a perfect uncertainty estimate, no estimator will save it.  If it works with the
perfect estimate and fails with the miscalibrated one, the contribution is
precisely "we need a calibrated estimator", which is a well-posed problem.

Usage:
  python scripts/run_gate4_controlled.py
  python scripts/run_gate4_controlled.py --biases 0 0.5 1 2 --lambdas 0.25 0.5 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.eval.rollout import evaluate, make_oracle_latent_fn  # noqa: E402
from muslate.planning.conservative import ConservativePenalty  # noqa: E402
from muslate.planning.controlled import BiasedOracleModel  # noqa: E402
from muslate.planning.proposal import ProposalPolicy  # noqa: E402
from muslate.planning.search import Lookahead  # noqa: E402


def run_bias(sim: SlateSim, args, bias: float) -> dict:
    configs: list[tuple[str, float, bool]] = [("no_penalty", 0.0, True)]
    configs += [(f"unc-perfect@l={lam}", lam, True) for lam in args.lambdas]
    configs += [(f"unc-bad@l={lam}", lam, False) for lam in args.lambdas]

    rows: dict[str, dict[str, float]] = {}
    for name, lam, perfect in configs:
        model = BiasedOracleModel(
            sim, bias_strength=bias, seed=args.bias_seed, perfect_uncertainty=perfect
        )
        penalty = (
            None
            if lam == 0.0
            else ConservativePenalty(model, lambda_unc=lam, unc_scale=1.0)
        )
        latent_fn = make_oracle_latent_fn(model, sim)

        row: dict[str, float] = {}
        for h in args.horizons:
            proposal = ProposalPolicy(
                model, n_candidates=args.candidates, seed=args.seed, mode="item_perturb"
            )
            planner = Lookahead(
                model, proposal, depth=h, beam_width=args.beam, gamma=sim.cfg.gamma, adjust=penalty
            )
            res = evaluate(sim, planner, latent_fn, args.users)
            row[f"H={h}"] = round(res.mean, 4)
        rows[name] = row
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--biases", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--lt", type=float, default=3.0)
    ap.add_argument("--div", type=float, default=0.8)
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--bias-seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/gate4_controlled.json")
    args = ap.parse_args()

    sim = SlateSim(EnvConfig(long_term_strength=args.lt, div_penalty=args.div))
    results: dict = {"config": vars(args), "rows": {}}

    for bias in args.biases:
        print(f"\n=== injected bias = {bias} ===")
        rows = run_bias(sim, args, bias)
        results["rows"][f"bias={bias}"] = rows
        for name, row in rows.items():
            delta = row[f"H={args.horizons[-1]}"] - row["H=0"]
            print(
                f"  {name:<20} "
                + "  ".join(f"H={h}:{row[f'H={h}']:.3f}" for h in args.horizons)
                + f"   Δ={delta:+.3f}"
            )

    lines = [
        f"Gate 4 (controlled bias)  lt={args.lt}  div={args.div}",
        "",
        "Δ = return(H=max) - return(H=0).  Positive = lookahead helps;",
        "negative = the planner is exploiting the injected bias.",
        "",
    ]
    for key, rows in results["rows"].items():
        lines.append(f"[{key}]")
        lines.append(f"{'config':<20} | " + " | ".join(f"{'H=' + str(h):>8}" for h in args.horizons) + " |      Δ")
        lines.append("-" * (len(lines[-1]) + 2))
        for name, row in rows.items():
            cells = " | ".join(f"{row[f'H={h}']:>8.3f}" for h in args.horizons)
            delta = row[f"H={args.horizons[-1]}"] - row["H=0"]
            lines.append(f"{name:<20} | {cells} | {delta:>+7.3f}")
        lines.append("")

    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text("# Gate 4 (controlled)\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
