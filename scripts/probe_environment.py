"""Diagnostic probe: does the synthetic environment contain a regime where
multi-step planning beats immediate greedy?

Run with **ground-truth dynamics** (OracleModel) so that no learned world model
can be blamed for a negative result.  This isolates a property of the
environment: how much does lookahead depth H buy, as a function of the
long-horizon coupling knob ``lt``?

Expected signature:
  * lt = 0  -> flat curve.  A contextual bandit: myopic greedy is optimal and
    lookahead cannot help.  This is the regime behind MF-MDP's "one step is
    enough".
  * lt > 0  -> rising curve with an elbow: long-horizon coupling exists, so a
    deeper planner wins, and the elbow tells you how deep is worth paying for.

If lt = 0 does not come out flat, the environment is broken and Gate 2 would be
meaningless.

Usage:
    python scripts/probe_environment.py
    python scripts/probe_environment.py --users 24 --candidates 32 --beam 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.planning.latent_model import OracleModel  # noqa: E402
from muslate.planning.proposal import ProposalPolicy  # noqa: E402
from muslate.planning.search import Lookahead, ReactiveGreedy  # noqa: E402


def evaluate(sim: SlateSim, model: OracleModel, planner, n_users: int, gamma: float) -> tuple[float, float]:
    rets = []
    for u in range(n_users):
        state = sim.initial_state(u, np.random.default_rng(sim.cfg.seed + 1000 + u))
        ret, disc = 0.0, 1.0
        while state.t < sim.cfg.horizon:
            lat = model.encode(state)[None]
            action = planner.plan(lat,remaining_steps=sim.cfg.horizon-state.t)[0]
            r, state = sim.step(state, action, noise=False)
            ret += disc * r
            disc *= gamma
        rets.append(ret)
    rets = np.asarray(rets)
    return float(rets.mean()), float(rets.std(ddof=1) / np.sqrt(len(rets)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    ap.add_argument("--lts", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--out", type=str, default="results/probe_environment.json")
    args = ap.parse_args()

    table: dict[str, dict[str, float]] = {}
    header = f"{'lt':>6} | " + " | ".join(f"{'H=' + str(h):>7}" for h in args.horizons) + " |  greedy"
    print(header)
    print("-" * len(header))

    for lt in args.lts:
        cfg = EnvConfig(long_term_strength=lt)
        sim = SlateSim(cfg)
        model = OracleModel(sim)
        row: dict[str, float] = {}

        for h in args.horizons:
            proposal = ProposalPolicy(
                model, n_candidates=args.candidates, seed=13, mode="item_perturb", n_perturb=1
            )
            planner = Lookahead(model, proposal, depth=h, beam_width=args.beam, gamma=cfg.gamma)
            m, _se = evaluate(sim, model, planner, args.users, cfg.gamma)
            row[f"H={h}"] = round(m, 4)

        greedy = ReactiveGreedy(model, gamma=cfg.gamma)
        g, _se = evaluate(sim, model, greedy, args.users, cfg.gamma)
        row["greedy"] = round(g, 4)

        table[f"lt={lt}"] = row
        print(
            f"{lt:>6} | "
            + " | ".join(f"{row[f'H={h}']:>7.3f}" for h in args.horizons)
            + f" | {row['greedy']:>7.3f}"
        )

    print()
    print("gain over H=0 (oracle planner, same proposal pool):")
    for lt, row in table.items():
        base = row["H=0"]
        gains = {k: round(v - base, 4) for k, v in row.items() if k.startswith("H=")}
        print(f"  {lt}: {gains}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
