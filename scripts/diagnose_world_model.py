"""Diagnose where the learned world model loses its planning ability.

Reports, as a function of the step index t:

* latent-state error        ``||state_head(h_hat_t) - true_state_t||``
* reward error              ``|model_reward(h_hat_t, slate_t) - true_r_t|``
* proposal quality          rank correlation between the model's per-item scores
                            and the simulator's per-item scores

Splitting t = 0 (cold start, empty history) from t > 0 is the point: if the
error is concentrated at t = 0, the bottleneck is initial state inference.  If
the rank correlation is low overall, the model cannot propose good slates and
no amount of search depth will help.

Usage:
  python scripts/diagnose_world_model.py --lt 1.0 --episodes 2000 --epochs 30
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
from muslate.model.train import get_device, train_world_model, true_item_scores  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lt", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="results/diagnose_wm.json")
    args = ap.parse_args()

    sim = SlateSim(EnvConfig(long_term_strength=args.lt))
    device = get_device(args.device)
    log = generate_log(sim, LogConfig(n_episodes=args.episodes, seed=11))
    wm_cfg = WorldModelConfig(epochs=args.epochs, seed=1)
    model = train_world_model(sim, log, wm_cfg, device, verbose=True)
    nm = NeuralLatentModel([model], device)

    # fresh behaviour trajectories for evaluation
    ev = generate_log(sim, LogConfig(n_episodes=args.eval_episodes, seed=999))
    n_ep = int(ev.episode_ids.max()) + 1
    t_len = ev.horizon
    slates = ev.slates.reshape(n_ep, t_len, ev.slate_size)
    rewards = ev.rewards.reshape(n_ep, t_len)
    states = ev.states.reshape(n_ep, t_len, -1)

    per_t = []
    for t in range(t_len):
        if t > 0:
            lat = nm.encode_history(slates[:, :t], rewards[:, :t])
        else:
            lat = np.repeat(nm.initial_latent(), n_ep, axis=0)

        # latent error via the aux head
        import torch

        with torch.no_grad():
            st = torch.as_tensor(lat, dtype=torch.float32, device=nm.device)
            pred_state = model.state_head(st).cpu().numpy()
        state_mae = float(np.abs(pred_state - states[:, t]).mean())

        r_pred = nm.reward(lat, slates[:, t])
        reward_mae = float(np.abs(r_pred - rewards[:, t]).mean())

        scores_model = nm.item_scores(lat)
        scores_true = true_item_scores(sim, states[:, t])
        rho = float(np.mean([spearman(scores_model[i], scores_true[i]) for i in range(n_ep)]))

        per_t.append(
            {"t": t, "state_mae": round(state_mae, 4), "reward_mae": round(reward_mae, 4), "score_rho": round(rho, 4)}
        )

    print(f"\n{'t':>3} | {'state_mae':>10} | {'reward_mae':>10} | {'score_rho':>10}")
    for row in per_t:
        print(f"{row['t']:>3} | {row['state_mae']:>10.4f} | {row['reward_mae']:>10.4f} | {row['score_rho']:>10.4f}")

    cold = per_t[0]
    warm = [r for r in per_t if r["t"] > 0]
    summary = {
        "t0_state_mae": cold["state_mae"],
        "t0_score_rho": cold["score_rho"],
        "warm_state_mae": round(float(np.mean([r["state_mae"] for r in warm])), 4),
        "warm_reward_mae": round(float(np.mean([r["reward_mae"] for r in warm])), 4),
        "warm_score_rho": round(float(np.mean([r["score_rho"] for r in warm])), 4),
    }
    print("\nsummary:", json.dumps(summary, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_t": per_t, "summary": summary}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
