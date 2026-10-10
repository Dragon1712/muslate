"""Semantic diagnostics: after the pairwise fix, is the model's *representation*
good enough for structure-seeking search?

Plan
----
P0a (reward head + pairwise) confirmed the misspecification and recovered roughly
half the damage, but `sid_hier` still did not overtake `flat_uniform`.  P0a also
showed a residual: prediction error on a div>0 environment is 0.179 against 0.088
on a div=0 environment.  This script tests the leading explanation -- that the
pairwise term's expressive power is capped by the *learned* item embedding `e`,
because the environment's term uses a fixed `z` while the model must discover one.

Three independent measurements, each on held-out data, each with the pairwise
head on and off:

D1  class structure of `item_emb`
      mean cosine within a true coarse class minus mean cosine across classes.
      Reference points: the environment's own `z` (the ceiling) and a random
      embedding (the floor).  If the model sits near the floor, the embedding
      carries no usable class geometry -- and a hierarchy built on it is being
      asked to search a structure that is not there.

D2  rank agreement of `item_scores` with the true myopic score
      per-timestep Spearman between the model's per-item scores and the
      simulator's. This is what the SID proposal uses to build and complete
      coarse plans, so it bounds how good the candidate pool can be.

D3  correlation between the learned pairwise term and the true diversity term
      `pair_value` vs `sim.diversity_term`. Quantifies how much of the
      combinatorial structure the new head actually captured (P0a's residual).

Run:
  python scripts/diagnose_semantics.py --episodes 3000 --epochs 40
  python scripts/diagnose_semantics.py --seeds 3 --device cuda
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
from muslate.eval.calibration import pearson, spearman  # noqa: E402
from muslate.model.neural_model import NeuralLatentModel  # noqa: E402
from muslate.model.train import get_device, train_world_model, true_item_scores  # noqa: E402


# ------------------------------------------------------------------ D1
def class_structure(emb: np.ndarray, coarse: np.ndarray) -> dict[str, float]:
    """Mean cosine similarity within a coarse class minus across classes."""
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    unit = emb / np.maximum(norm, 1e-12)
    sim = unit @ unit.T

    same = coarse[:, None] == coarse[None, :]
    np.fill_diagonal(same, False)
    intra = float(sim[same].mean())
    inter = float(sim[~same].mean())
    return {"intra": round(intra, 4), "inter": round(inter, 4), "gap": round(intra - inter, 4)}


def d1_embedding_structure(nm: NeuralLatentModel, sim: SlateSim) -> dict:
    coarse = sim.catalog.item_coarse
    env = class_structure(sim.catalog.item_emb, coarse)

    # the model's item embedding table (first ensemble member; they share the space)
    with torch.no_grad():
        model_emb = nm.models[0].slate_enc.item_emb.weight.detach().cpu().numpy().astype(np.float64)
    learned = class_structure(model_emb, coarse)

    # floor reference: the same geometry on random directions
    rng = np.random.default_rng(0)
    rand = rng.normal(size=model_emb.shape)
    floor = class_structure(rand, coarse)

    return {"env": env, "model": learned, "random_floor": floor}


# ------------------------------------------------------------------ D2
def d2_score_rank(nm: NeuralLatentModel, sim: SlateSim, holdout, stride: int = 3) -> dict:
    n_ep = int(holdout.episode_ids.max()) + 1
    t_len = holdout.horizon
    slates = holdout.slates.reshape(n_ep, t_len, holdout.slate_size)
    rewards = holdout.rewards.reshape(n_ep, t_len)
    states = holdout.states.reshape(n_ep, t_len, -1)

    latent = nm.encode_sequence(slates, rewards)                      # [E, T, dh]
    model_scores = nm.item_scores(latent.reshape(-1, latent.shape[-1]))
    model_scores = model_scores.reshape(n_ep, t_len, -1)

    t_idx = list(range(0, t_len, stride))
    true_scores = true_item_scores(sim, states[:, t_idx].reshape(-1, states.shape[-1]))
    true_scores = true_scores.reshape(n_ep, len(t_idx), -1)
    ms = model_scores[:, t_idx]

    rhos, t0_rho, by_t = [], [], {}
    for i in range(n_ep):
        for j, t in enumerate(t_idx):
            r = spearman(ms[i, j], true_scores[i, j])
            rhos.append(r)
            by_t.setdefault(int(t), []).append(r)
            if t == 0:
                t0_rho.append(r)
    return {
        "spearman_mean": round(float(np.mean(rhos)), 4),
        "spearman_t0": round(float(np.mean(t0_rho)) if t0_rho else 0.0, 4),
        "spearman_by_t": {int(t): round(float(np.mean(v)), 4) for t, v in sorted(by_t.items())},
        "n": len(rhos),
    }


# ------------------------------------------------------------------ D3
def d3_pairwise_vs_diversity(nm: NeuralLatentModel, sim: SlateSim, holdout, n: int = 400) -> dict:
    """Correlate the learned pairwise term with the true diversity term."""
    if not nm.models[0].pairwise:
        return {"pearson": None, "spearman": None, "note": "head has no pairwise term"}

    n_ep = int(holdout.episode_ids.max()) + 1
    t_len = holdout.horizon
    slates = holdout.slates.reshape(n_ep, t_len, holdout.slate_size).reshape(-1, holdout.slate_size)

    rng = np.random.default_rng(0)
    idx = rng.choice(slates.shape[0], size=min(n, slates.shape[0]), replace=False)
    picked = slates[idx]

    # mix in deliberately same-class and spread slates so the diversity term varies
    same_class = np.stack([np.arange(sim.slate_size) for _ in range(n // 4)])
    spread = np.stack(
        [
            [
                int(rng.choice(np.arange(c * sim.catalog.n_fine, (c + 1) * sim.catalog.n_fine), size=1)[0])
                for c in rng.permutation(sim.catalog.n_coarse)[: sim.slate_size]
            ]
            for _ in range(n // 4)
        ]
    )
    picked = np.concatenate([picked, same_class.astype(np.int64), spread.astype(np.int64)], axis=0)

    latent = nm.initial_latent()
    with torch.no_grad():
        st = torch.as_tensor(np.repeat(latent, picked.shape[0], axis=0), dtype=torch.float32)
        sl = torch.as_tensor(picked, dtype=torch.long)
        pv = nm.models[0].pair_value(st, sl).cpu().numpy()

    dt = sim.diversity_term(picked)
    return {
        "pearson": round(pearson(pv, dt), 4),
        "spearman": round(spearman(pv, dt), 4),
        "pair_std": round(float(pv.std()), 4),
        "div_std": round(float(dt.std()), 4),
        "n": int(picked.shape[0]),
    }


# ------------------------------------------------------------------ D4
def d4_coarse_rank(nm: NeuralLatentModel, sim: SlateSim, holdout, stride: int = 3) -> dict:
    """Coarse-level rank agreement -- the quantity the hierarchy actually consumes.

    ``SIDProposal`` builds coarse plans from ``scores.reshape(C, F).max(-1)``, i.e.
    it needs the *category-level* preference order, not the item-level one.  D2
    measured item-level agreement; this measures the level above it, which is
    where a weak class structure (D1) should show up first.
    """
    n_ep = int(holdout.episode_ids.max()) + 1
    t_len = holdout.horizon
    slates = holdout.slates.reshape(n_ep, t_len, holdout.slate_size)
    rewards = holdout.rewards.reshape(n_ep, t_len)
    states = holdout.states.reshape(n_ep, t_len, -1)

    latent = nm.encode_sequence(slates, rewards).reshape(-1, nm.state_dim)
    model_scores = nm.item_scores(latent).reshape(n_ep, t_len, -1)

    c_out, c_in = sim.n_coarse, sim.catalog.n_fine
    t_idx = list(range(0, t_len, stride))
    true_scores = true_item_scores(sim, states[:, t_idx].reshape(-1, states.shape[-1]))
    true_scores = true_scores.reshape(n_ep, len(t_idx), -1)

    rhos = []
    for i in range(n_ep):
        for j in range(len(t_idx)):
            ms = model_scores[i, t_idx[j]].reshape(c_out, c_in).max(axis=1)
            ts = true_scores[i, j].reshape(c_out, c_in).max(axis=1)
            rhos.append(spearman(ms, ts))
    return {"coarse_spearman": round(float(np.mean(rhos)), 4), "n": len(rhos)}


def _cached_model(path: Path, sim: SlateSim, cfg: WorldModelConfig, log, device):
    """Train once, reuse across diagnostic runs."""
    if path.exists():
        model = torch.load(path, weights_only=False)
        model.to(device)
        print(f"  loaded cached model from {path}")
        return model
    model = train_world_model(sim, log, cfg, device, verbose=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)
    return model


# ------------------------------------------------------------------ D5
def d5_head_ceiling(sim: SlateSim, holdout, epochs: int = 400, seed: int = 0) -> dict:
    """How well could *any* score head do if the belief were perfect?

    D2 measures the model's own `item_scores` accuracy (~0.72) but cannot say
    *why* it is not 1.0: the latent may have lost the information, or the head
    may simply never have learned to extract it.  Those call for opposite
    fixes -- a better history encoder versus a better head/objective -- so they
    must be separated *before* any tuning effort is spent.

    This trains `MLP(true_state) -> true_item_scores` on half the holdout and
    reports Spearman on the other half.  Because the input *is* the state, this
    is an upper bound on what a head could reach, and the shortfall of D2 from
    it is the part of the error attributable to belief inference.

    ceiling high + model low  ->  the belief is losing information
    ceiling itself low        ->  the head / objective is the limit
    """
    states = holdout.states.reshape(-1, holdout.states.shape[-1])
    scores = true_item_scores(sim, states)                        # [N, n_items]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(states))
    half = len(states) // 2
    tr, te = perm[:half], perm[half:]

    x = torch.as_tensor(states, dtype=torch.float32)
    y = torch.as_tensor(scores, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(states.shape[-1], 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, sim.n_items),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss = torch.tensor(0.0)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(x[tr]), y[tr])
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = net(x[te]).numpy()
    truth = scores[te]
    rhos = [spearman(pred[i], truth[i]) for i in range(len(te))]
    return {
        "ceiling_spearman": round(float(np.mean(rhos)), 4),
        "final_train_mse": round(float(loss.item()), 6),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lt", type=float, default=3.0)
    ap.add_argument("--div", type=float, default=0.8)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--holdout-episodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--cache-dir", type=str, default="results/_semantic_cache")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default="results/semantic_diagnostic.json")
    args = ap.parse_args()

    sim = SlateSim(EnvConfig(long_term_strength=args.lt, div_penalty=args.div))
    device = get_device(args.device)
    results: dict = {"config": vars(args), "arms": {}}

    for pairwise in (False, True):
        arm = "pairwise" if pairwise else "unary_only"
        print(f"\n=== {arm} ===")
        acc = []
        for s in range(args.seeds):
            log = generate_log(sim, LogConfig(n_episodes=args.episodes, seed=4000 + 17 * s))
            holdout = generate_log(sim, LogConfig(n_episodes=args.holdout_episodes, seed=9000 + 17 * s))
            cfg = WorldModelConfig(epochs=args.epochs, pairwise=pairwise, seed=1 + 11 * s)
            print(f"  [seed {s}] preparing model ...")
            model = _cached_model(
                Path(args.cache_dir) / f"{arm}_s{s}_ep{args.episodes}_e{args.epochs}.pt",
                sim,
                cfg,
                log,
                device,
            )
            nm = NeuralLatentModel([model], device)
            acc.append({
                "d1_embedding": d1_embedding_structure(nm, sim),
                "d2_scores": d2_score_rank(nm, sim, holdout),
                "d3_pairwise": d3_pairwise_vs_diversity(nm, sim, holdout),
                "d4_coarse": d4_coarse_rank(nm, sim, holdout),
                "d5_ceiling": d5_head_ceiling(sim, holdout),
            })
        results["arms"][arm] = acc
        d1 = acc[0]["d1_embedding"]
        print(f"  D1 structure  model gap={d1['model']['gap']:+.4f}  "
              f"env gap={d1['env']['gap']:+.4f}  random gap={d1['random_floor']['gap']:+.4f}")
        print(f"  D2 item rank  rho={acc[0]['d2_scores']['spearman_mean']:+.4f}")
        print(f"  D3 pair vs div rho={acc[0]['d3_pairwise']['pearson']}")
        print(f"  D4 coarse rank rho={acc[0]['d4_coarse']['coarse_spearman']:+.4f}")
        d5 = acc[0]["d5_ceiling"]
        d2 = acc[0]["d2_scores"]["spearman_mean"]
        print(f"  D5 head ceiling rho={d5['ceiling_spearman']:+.4f}  "
              f"(model D2={d2:+.4f} -> belief loses {d5['ceiling_spearman'] - d2:+.4f}; "
              f"head/objective loses {1.0 - d5['ceiling_spearman']:+.4f} from 1.0)")

    lines = [
        f"Semantic diagnostics  (lt={args.lt}, div={args.div}, {args.episodes} ep, {args.epochs} epochs)",
        "",
        "D1  class structure of the item embedding: intra-class minus inter-class cosine.",
        "    'env' is the ceiling (the simulator's own z), 'random' the floor.",
        "D2  per-timestep Spearman between the model's item_scores and the true ones.",
        "D4  the same but at the COARSE level -- what SIDProposal actually consumes.",
        "D3  correlation between the learned pairwise term and the true diversity term.",
        "",
        f"{'arm':<12} | {'D1 model':>9} | {'D1 env':>8} | {'D2 item':>8} | {'D4 coarse':>9} | {'D3 rho':>8}",
    ]
    lines.append("-" * len(lines[-1]))
    for arm, accs in results["arms"].items():
        d1 = accs[0]["d1_embedding"]
        lines.append(
            f"{arm:<12} | {d1['model']['gap']:>+9.4f} | {d1['env']['gap']:>+8.4f} | "
            f"{accs[0]['d2_scores']['spearman_mean']:>+8.4f} | "
            f"{accs[0]['d4_coarse']['coarse_spearman']:>+9.4f} | "
            f"{str(accs[0]['d3_pairwise']['pearson']):>8}"
        )
    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text("# Semantic diagnostics\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
