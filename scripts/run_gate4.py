"""Gate 4: can conservative planning stop the planner exploiting model error?

The failure mode this measures.  A planner optimises the *model's* estimate, not
reality.  Where the model over-estimates, the planner goes there.  So with a
learned world model, increasing lookahead depth can *reduce* true return even
though it increases the model's estimated return -- the opposite of what happens
with ground-truth dynamics:

    oracle dynamics   : return rises with depth H
    learned dynamics  : return falls (or flattens early) with depth H
    learned + penalty : the fall is reduced or removed

The penalty is ``Q - lambda*U - beta*D`` where U is ensemble disagreement and D is
distance from a support estimate.

**This runner first checks that the penalty terms are pointed at the error at
all.**  A penalty can only help if it correlates with the model's true
prediction error; if the correlations come out near zero, no choice of lambda or
beta can work and the negative result belongs to the *uncertainty estimator*, not
to the planning framework.  That check is reported before any planning happens.

Usage:
  python scripts/run_gate4.py --probe                      # scales + calibration only
  python scripts/run_gate4.py --lt 3.0 --div 0.8 --users 6
  python scripts/run_gate4.py --lt 3.0 --episodes 20000 --ensemble 3 --device cuda
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
from muslate.env.logger import OfflineLog, generate_log  # noqa: E402
from muslate.eval.calibration import signal_report  # noqa: E402
from muslate.eval.rollout import (  # noqa: E402
    evaluate,
    make_learned_latent_fn,
    make_oracle_latent_fn,
)
from muslate.model.neural_model import NeuralLatentModel  # noqa: E402
from muslate.model.error_model import NeuralErrorModel, train_error_model  # noqa: E402
from muslate.model.train import get_device, train_world_model  # noqa: E402
from muslate.planning.conservative import (  # noqa: E402
    BehaviorSupport,
    ConservativePenalty,
    LatentSupport,
)
from muslate.planning.latent_model import OracleModel  # noqa: E402
from muslate.planning.proposal import ProposalPolicy  # noqa: E402
from muslate.planning.search import Lookahead  # noqa: E402

# name -> (lambda_unc, beta_ood)
PENALTY_CONFIGS: dict[str, tuple[float, float]] = {
    "none": (0.0, 0.0),
    "unc": (1.0, 0.0),
    "ood": (0.0, 1.0),
    "both": (1.0, 1.0),
}


# ---------------------------------------------------------------- helpers
def train_ensemble_with_logs(sim, args, device):
    models, logs = [], []
    for i in range(args.ensemble):
        log = generate_log(sim, LogConfig(n_episodes=args.episodes, seed=args.log_seed + 1000 * i))
        wm_cfg = WorldModelConfig(
            epochs=args.epochs, rollout_depth=args.rollout_depth, seed=1 + 100 * i
        )
        print(f"  training world model {i + 1}/{args.ensemble} ...")
        models.append(train_world_model(sim, log, wm_cfg, device, verbose=args.verbose))
        logs.append(log)
    return models, logs


def collect_latents(nm: NeuralLatentModel, log: OfflineLog) -> np.ndarray:
    """Beliefs the model forms on the logged data: the reference support."""
    n_ep = int(log.episode_ids.max()) + 1
    t_len = log.horizon
    slates = log.slates.reshape(n_ep, t_len, log.slate_size)
    rewards = log.rewards.reshape(n_ep, t_len)
    lats = [nm.encode_history(slates[:, :t], rewards[:, :t]) for t in range(1, t_len)]
    return np.concatenate(lats, axis=0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def calibration_report(
    sim: SlateSim,
    nm: NeuralLatentModel,
    item_support: BehaviorSupport,
    state_support: LatentSupport,
    n_episodes: int = 48,
    stride: int = 3,
) -> dict:
    """Is each penalty term actually pointed at the model's error?

    Measured on *held-out* behaviour data, against the reference support built
    from the *training* log, so a sample is never its own nearest neighbour.
    """
    log = generate_log(sim, LogConfig(n_episodes=n_episodes, seed=777))
    t_len = log.horizon
    slates = log.slates.reshape(n_episodes, t_len, log.slate_size)
    rewards = log.rewards.reshape(n_episodes, t_len)

    us, di, ds, errs = [], [], [], []
    for t in range(1, t_len, stride):
        lat = nm.encode_history(slates[:, :t], rewards[:, :t])
        cur = slates[:, t]
        us.append(np.asarray(nm.uncertainty(lat, cur), dtype=np.float64))
        di.append(item_support.score(cur))
        ds.append(state_support.score(lat))
        pred = np.asarray(nm.reward(lat, cur), dtype=np.float64)
        errs.append(np.abs(pred - rewards[:, t]))

    u, di, ds, e = (np.concatenate(x) for x in (us, di, ds, errs))
    return {
        "corr_unc_error": round(_corr(u, e), 4),
        "corr_item_ood_error": round(_corr(di, e), 4),
        "corr_state_ood_error": round(_corr(ds, e), 4),
        "mean_abs_error": round(float(e.mean()), 4),
        "n_samples": int(len(e)),
    }


def term_stats(penalty: ConservativePenalty, states, slates) -> dict:
    comps = penalty.components(states, slates)
    return {
        k: {"mean": float(v.mean()), "std": float(v.std()), "max": float(v.max())}
        for k, v in comps.items()
    }


# ---------------------------------------------------------------- sweeps
def run_sweep(
    sim: SlateSim,
    model,
    latent_fn,
    item_support,
    state_support,
    arg_lambda,
    arg_beta,
    args,
    scales: dict,
    unc_fn=None,
) -> dict:
    penalty = None
    if arg_lambda > 0 or arg_beta > 0:
        penalty = ConservativePenalty(
            model,
            lambda_unc=arg_lambda,
            beta_ood=arg_beta,
            state_support=state_support,
            uncertainty_fn=unc_fn,
            unc_scale=scales.get("unc_scale", 1.0),
            ood_scale=scales.get("ood_scale", 1.0),
        )

    rows = {}
    for h in args.horizons:
        proposal = ProposalPolicy(
            model, n_candidates=args.candidates, seed=args.seed, mode="item_perturb"
        )
        planner = Lookahead(
            model, proposal, depth=h, beam_width=args.beam, gamma=sim.cfg.gamma, adjust=penalty
        )
        res = evaluate(sim, planner, latent_fn, args.users, tail_from=args.warmup)
        rows[f"H={h}"] = {
            "return": round(res.mean, 4),
            "tail": round(res.tail_mean, 4),
            "se": round(res.se, 4),
            "latency_ms": round(res.latency_ms, 3),
        }
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lt", type=float, default=3.0)
    ap.add_argument("--div", type=float, default=0.8)
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 2, 3, 5])
    ap.add_argument("--users", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=24)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--rollout-depth", type=int, default=3)
    ap.add_argument("--log-seed", type=int, default=555)
    ap.add_argument("--calib-episodes", type=int, default=48)
    ap.add_argument("--support-points", type=int, default=2000)
    ap.add_argument(
        "--unc-source",
        choices=["ensemble", "error_model"],
        default="ensemble",
        help="which signal drives the U term: ensemble disagreement, or a "
        "supervised error model fitted on held-out data",
    )
    ap.add_argument("--err-episodes", type=int, default=800)
    ap.add_argument("--err-epochs", type=int, default=60)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--probe", action="store_true", help="scales + calibration only, no planning")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default="results/gate4.json")
    args = ap.parse_args()

    sim = SlateSim(EnvConfig(long_term_strength=args.lt, div_penalty=args.div))
    device = get_device(args.device)

    print(f"=== lt={args.lt}, div={args.div}, ensemble={args.ensemble} ===")
    t0 = time.perf_counter()
    models, logs = train_ensemble_with_logs(sim, args, device)
    nm = NeuralLatentModel(models, device)
    item_support = BehaviorSupport(logs[0])
    state_support = LatentSupport(collect_latents(nm, logs[0]), max_points=args.support_points)
    print(f"  trained {len(models)} models in {time.perf_counter() - t0:.1f}s")

    # -- uncertainty source ------------------------------------------------
    unc_fn = None
    if args.unc_source == "error_model":
        err_logs = [
            generate_log(sim, LogConfig(n_episodes=args.err_episodes, seed=8000 + i)) for i in range(2)
        ]
        print(f"  fitting error model on {args.err_episodes} x {len(err_logs)} holdout episodes ...")
        em = train_error_model(
            nm, sim, err_logs, epochs=args.err_epochs, device=device, seed=1, verbose=args.verbose
        )
        nem = NeuralErrorModel(em, device)
        unc_fn = nem.score
        em_calib = signal_report(
            nm, [generate_log(sim, LogConfig(n_episodes=32, seed=9100))], error_model=nem, stride=3
        )
        print(f"  error-model calibration: {json.dumps(em_calib.get('error_model', {}))}")
        results_extra = {"error_model_calibration": em_calib}
    else:
        results_extra = {}

    calib = calibration_report(sim, nm, item_support, state_support, args.calib_episodes)
    print(f"  calibration: {json.dumps(calib)}")
    informative = abs(calib["corr_unc_error"]) > 0.05 or abs(calib["corr_state_ood_error"]) > 0.05
    if not informative:
        print(
            "  WARNING: neither U nor D correlates with the true prediction error.\n"
            "           A penalty cannot help by construction; any flat curve below is\n"
            "           a property of the uncertainty estimator, not of conservative planning."
        )

    # penalty terms are reported in units of one standard deviation.  The probe
    # must use a *diverse* set of states, otherwise the state-level term is
    # constant by construction and its scale is meaningless.
    ref = collect_latents(nm, logs[0])
    rng = np.random.default_rng(0)
    n_probe = min(32, ref.shape[0])
    probe_states = ref[rng.choice(ref.shape[0], n_probe, replace=False)]
    probe_pool = ProposalPolicy(nm, n_candidates=args.candidates, seed=0).propose(probe_states)
    # match how the planners call `adjust`: states and slates are flat pairs
    flat_states = np.repeat(probe_states, probe_pool.shape[1], axis=0)
    flat_slates = probe_pool.reshape(-1, probe_pool.shape[-1])
    probe_pen = ConservativePenalty(
        nm, 1.0, 1.0, state_support=state_support, uncertainty_fn=unc_fn
    )
    stats = term_stats(probe_pen, flat_states, flat_slates)
    scales = {
        "unc_scale": 1.0 / max(stats["uncertainty"]["std"], 1e-6),
        "ood_scale": 1.0 / max(stats["state_ood"]["std"], 1e-6),
        "term_stats": stats,
    }
    print(f"  term stats : {json.dumps(stats)}")
    print(f"  scales     : unc={scales['unc_scale']:.2f}  ood={scales['ood_scale']:.2f}")

    if args.probe:
        return

    results: dict = {
        "config": vars(args),
        "calibration": calib,
        "scales": scales,
        "rows": {},
        **results_extra,
    }

    oracle = OracleModel(sim)
    results["rows"]["oracle"] = run_sweep(
        sim, oracle, make_oracle_latent_fn(oracle, sim), item_support, None, 0.0, 0.0, args, {}
    )
    print("  oracle  " + "  ".join(f"H={h}:{results['rows']['oracle'][f'H={h}']['return']:.3f}" for h in args.horizons))

    for name, (lam, beta) in PENALTY_CONFIGS.items():
        key = f"learned/{name}"
        results["rows"][key] = run_sweep(
            sim,
            nm,
            make_learned_latent_fn(nm),
            item_support,
            state_support,
            lam,
            beta,
            args,
            scales,
            unc_fn=unc_fn,
        )
        row = results["rows"][key]
        print(f"  {name:<7} " + "  ".join(f"H={h}:{row[f'H={h}']['return']:.3f}" for h in args.horizons))

    lines = [
        f"Gate 4 - conservative planning (lt={args.lt}, div={args.div}, ensemble={args.ensemble})",
        "",
        f"calibration: {json.dumps(calib)}",
        "",
        f"{'config':<18} | " + " | ".join(f"{'H=' + str(h):>8}" for h in args.horizons) + " |  H0 -> Hmax",
    ]
    lines.append("-" * (len(lines[-1]) + 2))
    for key, rows in results["rows"].items():
        cells = " | ".join(f"{rows[f'H={h}']['return']:>8.3f}" for h in args.horizons)
        delta = rows[f"H={args.horizons[-1]}"]["return"] - rows["H=0"]["return"]
        lines.append(f"{key:<18} | {cells} |   {delta:+.3f}")
    lines.append("")
    lines.append("A falling H0->Hmax row means lookahead is making things WORSE (model exploitation).")
    table = "\n".join(lines)
    print("\n" + table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text("# Gate 4\n\n```\n" + table + "\n```\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
