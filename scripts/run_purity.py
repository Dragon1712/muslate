"""Two orthogonal purity sweeps: is `sid_hier`'s learned-dynamics failure about
the *tree*, or about the *prior* that walks it?

Both sweeps run under **ground-truth dynamics** so that neither axis is
confounded by world-model error -- the point is to characterise the planner, not
the model.  That matters because Gate 3's learned arm mixes three causes at once:
a wrong tree, a wrong prior, and a wrong reward estimate.  The P0a pairwise fix
removed the third; this script separates the first two.

Sweep A -- tree purity.  The tree is varied (oracle / noisy(p) / random / learned)
while the prior stays perfect.  Answers: does hierarchical search need a good
tree?
Sweep B -- prior purity.  The tree stays oracle while `item_scores` get a fixed
per-class offset.  Answers: does hierarchical search amplify a bad prior more
than flat search does?

The headline column is `sid_hier - best_flat`: the hierarchy's *relative* value.
A curve that falls faster in B than in A means the hierarchy is more sensitive to
the prior than to the tree, which would relocate the learned-arm failure from
"the model's semantics are wrong" to "the hierarchy amplifies ranking error".

Usage:
  python scripts/run_purity.py --lts 3.0 --users 16 --budget 512 --depth 2
  python scripts/run_purity.py --purities 1 0.5 0 --noises 0 1 3 --out results/purity_sweeps.json
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
from muslate.eval.rollout import evaluate, make_oracle_latent_fn  # noqa: E402
from muslate.planning.latent_model import OracleModel  # noqa: E402
from muslate.planning.perturb import NoisyScoreModel  # noqa: E402
from muslate.planning.proposal import ProposalPolicy  # noqa: E402
from muslate.planning.search import Lookahead, ReactiveGreedy  # noqa: E402
from muslate.planning.sid_map import build_sid_map  # noqa: E402
from muslate.planning.abstractions import SIDProposal  # noqa: E402

WIDEN = 4
REFINE_POSITIONS = 5
PLANNERS = ["flat_uniform", "flat_policy", "sid_flat", "sid_hier"]


def build_planners(model, sim: SlateSim, sid_map, budget: int, depth: int, seed: int) -> dict:
    """Budget arithmetic copied verbatim from `run_gate3.build_variants` so the
    two runners stay comparable; only the tree is allowed to differ."""
    refine_target = budget // 2
    fine_per_pos = max(1, refine_target // (WIDEN * REFINE_POSITIONS))
    refine = WIDEN * REFINE_POSITIONS * fine_per_pos

    flat_out = budget
    sid_sampled = max(8, budget // 2)
    sid_out = max(8, budget - sid_sampled)
    hier_sampled = max(8, (budget - refine) // 2)
    hier_out = max(8, budget - hier_sampled - refine)

    def look(prop):
        return Lookahead(model, prop, depth=depth, beam_width=3, gamma=sim.cfg.gamma)

    return {
        "flat_uniform": look(ProposalPolicy(model, n_candidates=flat_out, seed=seed, mode="uniform")),
        "flat_policy": look(
            ProposalPolicy(model, n_candidates=flat_out, seed=seed, mode="item_perturb", n_perturb=1)
        ),
        "sid_flat": look(
            SIDProposal(model, n_sampled=sid_sampled, out_size=sid_out,
                        refine_positions=0, seed=seed, sid_map=sid_map)
        ),
        "sid_hier": look(
            SIDProposal(model, n_sampled=hier_sampled, out_size=hier_out, widen=WIDEN,
                        refine_positions=REFINE_POSITIONS, fine_per_pos=fine_per_pos,
                        seed=seed, sid_map=sid_map)
        ),
    }


def summarize_row(returns: dict) -> dict:
    """Derive the comparison columns from one point's planner returns.

    Two baselines, deliberately:

    * ``hier_minus_flat`` uses ``max(flat_uniform, flat_policy)``, matching what
      `run_gate3` reports.  That is the right baseline when the *tree* is varied.
    * ``hier_minus_policy`` uses ``flat_policy`` alone.  ``flat_uniform`` samples
      uniformly and never reads ``item_scores``, so it is **immune** to prior
      corruption; when the prior is the varied axis it is a constant, and
      ``hier_minus_flat`` degenerates into ``sid_hier`` shifted by a constant --
      it measures nothing relative.  Only ``hier_minus_policy`` is comparable
      across prior settings.

    Pinned by
    `tests/test_perturb.py::test_uniform_pool_is_immune_to_the_score_corruption`.
    """
    missing = [p for p in PLANNERS if p not in returns]
    if missing:
        raise KeyError(f"missing planner returns: {missing}")

    row = {p: round(float(returns[p]), 4) for p in PLANNERS}
    row["best_flat"] = max(row["flat_uniform"], row["flat_policy"])
    row["hier_minus_flat"] = round(row["sid_hier"] - row["best_flat"], 4)
    row["hier_minus_policy"] = round(row["sid_hier"] - row["flat_policy"], 4)
    return row


def run_point(sim: SlateSim, model, latent_fn, sid_map, args) -> dict:
    planners = build_planners(model, sim, sid_map, args.budget, args.depth, args.seed)
    returns = {}
    for name in PLANNERS:
        res = evaluate(sim, planners[name], latent_fn, args.users)
        returns[name] = res.mean
    return summarize_row(returns)


def sweep_a_tree(sim: SlateSim, args) -> dict:
    """Vary the tree; keep the prior perfect (oracle dynamics)."""
    oracle = OracleModel(sim)
    latent_fn = make_oracle_latent_fn(oracle, sim)
    rows: dict[str, dict] = {}

    for p in args.purities:
        if p >= 1.0:
            sid_map = build_sid_map(sim.n_items, sim.catalog.item_coarse, "oracle", seed=args.seed)
            label = "oracle"
        elif p <= 0.0:
            sid_map = build_sid_map(sim.n_items, sim.catalog.item_coarse, "random", seed=args.seed)
            label = "random"
        else:
            sid_map = build_sid_map(sim.n_items, sim.catalog.item_coarse, "noisy", p=1.0 - p, seed=args.seed)
            label = f"noisy(p={1.0 - p:.2f})"
        rows[label] = {"purity": round(sid_map.purity, 4), **run_point(sim, oracle, latent_fn, sid_map, args)}

    if args.include_learned:
        emb = sim.catalog.item_emb
        sid_map = build_sid_map(sim.n_items, sim.catalog.item_coarse, "learned", seed=args.seed, embedding=emb)
        rows["learned(env-emb)"] = {
            "purity": round(sid_map.purity, 4),
            **run_point(sim, oracle, latent_fn, sid_map, args),
        }
    return rows


def sweep_b_prior(sim: SlateSim, args) -> dict:
    """Vary the prior; keep the tree at oracle."""
    sid_map = build_sid_map(sim.n_items, sim.catalog.item_coarse, "oracle", seed=args.seed)
    rows: dict[str, dict] = {}

    for noise in args.noises:
        oracle = OracleModel(sim)
        model = NoisyScoreModel(oracle, sim.catalog.item_coarse, sim.n_coarse, noise, seed=args.seed)
        # state encoding must come from the un-corrupted model: the decorator only
        # corrupts item_scores, so anything else would leak into the environment
        latent_fn = make_oracle_latent_fn(oracle, sim)
        rows[f"noise={noise:g}"] = {"noise": noise, **run_point(sim, model, latent_fn, sid_map, args)}
    return rows


def format_table(title: str, x_label: str, rows: dict) -> str:
    lines = [title, ""]
    header = (
        f"{x_label:<18} | {'purity':>7} | "
        + " | ".join(f"{n:>12}" for n in PLANNERS)
        + f" | {'h-flat':>8} | {'h-policy':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for label, row in rows.items():
        purity = row.get("purity", row.get("noise", float("nan")))
        cells = " | ".join(f"{row[n]:>12.3f}" for n in PLANNERS)
        lines.append(
            f"{label:<18} | {purity:>7.3f} | {cells} | "
            f"{row['hier_minus_flat']:>+8.3f} | {row['hier_minus_policy']:>+9.3f}"
        )
    lines.append("")
    lines.append(
        "h-flat   = sid_hier - max(flat_uniform, flat_policy)   [right baseline for Sweep A]"
    )
    lines.append(
        "h-policy = sid_hier - flat_policy                       [right baseline for Sweep B:"
    )
    lines.append(
        "           flat_uniform never reads item_scores, so it is immune to prior corruption]"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lts", type=float, nargs="+", default=[3.0])
    ap.add_argument("--div", type=float, default=0.8)
    ap.add_argument("--purities", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    ap.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--users", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--include-learned", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", type=str, default="results/purity_sweeps.json")
    args = ap.parse_args()

    results: dict = {"config": vars(args), "sweep_a_tree": {}, "sweep_b_prior": {}}
    tables: list[str] = []

    for lt in args.lts:
        sim = SlateSim(EnvConfig(long_term_strength=lt, div_penalty=args.div))
        print(f"\n=== lt={lt}, div={args.div}, budget={args.budget}, depth={args.depth} ===")

        print("  sweep A (tree purity, oracle dynamics) ...")
        a = sweep_a_tree(sim, args)
        results["sweep_a_tree"][f"lt={lt}"] = a
        tables.append(format_table(f"Sweep A - tree purity (lt={lt})", "tree", a))

        print("  sweep B (prior purity, oracle tree) ...")
        b = sweep_b_prior(sim, args)
        results["sweep_b_prior"][f"lt={lt}"] = b
        tables.append(format_table(f"Sweep B - prior purity (lt={lt})", "prior", b))

    table = "\n\n".join(tables)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text("# Purity sweeps\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
