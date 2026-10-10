"""Calibration diagnostics: does an uncertainty signal track the true model error?

A penalty term of the form ``Q - lambda*U`` is only as good as the correlation
between ``U`` and the actual error.  Gate 4 found ensemble disagreement sitting
at ρ ≈ 0.1, i.e. useless.  This module makes that comparison a first-class,
reusable measurement so candidate signals can be ranked before any planning run.

Signals compared:

* ``ensemble_std``  disagreement between dynamics models
* ``error_model``   supervised regression of |predicted - true| on held-out data
* ``state_ood``     distance from the logged belief distribution

Everything is measured on a holdout that neither the world model nor the error
model was trained on.
"""

from __future__ import annotations

import numpy as np

from muslate.env.logger import OfflineLog
from muslate.model.error_model import build_error_dataset


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation.  A penalty only needs to *rank* actions, not rescale them."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ra, rb)


def collect_latents(neural_model, log: OfflineLog) -> np.ndarray:
    """Beliefs the model forms on logged data: the reference support."""
    n_ep = int(log.episode_ids.max()) + 1
    t_len = log.horizon
    slates = log.slates.reshape(n_ep, t_len, log.slate_size)
    rewards = log.rewards.reshape(n_ep, t_len)
    latent_seq = neural_model.encode_sequence(slates, rewards)          # [E, T, dh]
    return latent_seq[:, 1:].reshape(-1, latent_seq.shape[-1])


def signal_report(
    neural_model,
    logs: list[OfflineLog],
    error_model=None,
    state_support=None,
    item_support=None,
    stride: int = 2,
) -> dict:
    """Correlate each candidate uncertainty signal with the true absolute error."""
    lat, sls, err = build_error_dataset(neural_model, logs, stride=stride)

    signals: dict[str, np.ndarray] = {}
    signals["ensemble_std"] = np.asarray(neural_model.uncertainty(lat, sls), dtype=np.float64)
    if error_model is not None:
        signals["error_model"] = np.asarray(error_model.score(lat, sls), dtype=np.float64)
    if state_support is not None:
        signals["state_ood"] = np.asarray(state_support.score(lat), dtype=np.float64)
    if item_support is not None:
        signals["item_ood"] = np.asarray(item_support.score(sls), dtype=np.float64)

    report: dict[str, dict] = {}
    for name, sig in signals.items():
        report[name] = {
            "pearson": round(pearson(sig, err), 4),
            "spearman": round(spearman(sig, err), 4),
            "signal_std": round(float(sig.std()), 6),
        }
    report["_target"] = {
        "mean_abs_error": round(float(err.mean()), 4),
        "error_std": round(float(err.std()), 4),
        "n_pairs": int(err.shape[0]),
    }
    return report
