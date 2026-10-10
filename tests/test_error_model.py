"""Error-model invariants.

The error model is the component that decides whether Innovation 3 is viable, so
its contract needs to be pinned down: it must output a non-negative estimate, its
training set must be built from the same (belief, slate) pairing the planner
queries at planning time, and it must be trainable end-to-end without blowing up.

Run directly:
    python tests/test_error_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muslate.config import EnvConfig, LogConfig, WorldModelConfig  # noqa: E402
from muslate.env import SlateSim  # noqa: E402
from muslate.env.logger import generate_log  # noqa: E402
from muslate.eval.calibration import collect_latents, pearson, spearman  # noqa: E402
from muslate.model.error_model import (  # noqa: E402
    ErrorModel,
    NeuralErrorModel,
    build_error_dataset,
    train_error_model,
)
from muslate.model.neural_model import NeuralLatentModel  # noqa: E402
from muslate.model.world_model import WorldModel  # noqa: E402


def _tiny_setup(lt: float = 3.0):
    sim = SlateSim(EnvConfig(long_term_strength=lt, div_penalty=0.8))
    cfg = WorldModelConfig(latent_dim=16, slate_emb_dim=8, hidden_dim=32, epochs=1)
    wm = WorldModel(cfg, sim.n_items, sim.slate_size, true_state_dim=sim.cfg.dim + sim.n_coarse)
    nm = NeuralLatentModel([wm], "cpu")
    log = generate_log(sim, LogConfig(n_episodes=8, seed=1))
    return sim, nm, log


def test_error_model_output_is_non_negative_and_shaped() -> None:
    m = ErrorModel(latent_dim=8, n_items=20, slate_size=3)
    h = torch.randn(5, 8)
    slates = torch.randint(0, 20, (5, 3))
    out = m(h, slates)
    assert out.shape == (5,)
    assert (out >= 0).all(), "an error estimate cannot be negative"


def test_error_model_accepts_leading_batch_dims() -> None:
    """The planners call heads with sequence-shaped tensors."""
    m = ErrorModel(latent_dim=8, n_items=20, slate_size=3)
    h = torch.randn(4, 7, 8)
    slates = torch.randint(0, 20, (4, 7, 3))
    assert m(h, slates).shape == (4, 7)


def test_error_dataset_pairs_match_the_planner_query() -> None:
    """state[:, t] must be the belief used to predict r_t, or the model is
    supervised on a different distribution than it will be queried on."""
    sim, nm, log = _tiny_setup()
    lat, sls, err = build_error_dataset(nm, [log], stride=2)

    assert lat.shape[0] == sls.shape[0] == err.shape[0]
    assert lat.shape[1] == nm.state_dim
    assert sls.shape[1] == sim.slate_size
    assert (err >= 0).all()
    assert not np.allclose(err, 0.0), "errors must be non-degenerate"


def test_error_dataset_stride_shrinks_it() -> None:
    sim, nm, log = _tiny_setup()
    full, _, _ = build_error_dataset(nm, [log], stride=1)
    half, _, _ = build_error_dataset(nm, [log], stride=2)
    assert half.shape[0] < full.shape[0]


def test_training_runs_and_predictions_vary() -> None:
    sim, nm, log = _tiny_setup()
    model = train_error_model(nm, sim, [log], epochs=5, batch_size=64, device="cpu", seed=0)
    nem = NeuralErrorModel(model, "cpu")

    lat, sls, _err = build_error_dataset(nm, [log], stride=4)
    pred = nem.score(lat, sls)
    assert pred.shape[0] == lat.shape[0]
    assert (pred >= 0).all()
    assert pred.std() > 0, "a constant error estimator cannot drive a penalty"


def test_collect_latents_matches_state_dim() -> None:
    _sim, nm, log = _tiny_setup()
    lat = collect_latents(nm, log)
    assert lat.shape[1] == nm.state_dim
    assert lat.shape[0] > 0


def test_correlation_helpers() -> None:
    x = np.arange(50.0)
    assert pearson(x, x) > 0.99
    assert pearson(x, -x) < -0.99
    assert pearson(x, np.zeros_like(x)) == 0.0
    # spearman is rank-based, so a monotone nonlinearity does not hurt it
    assert spearman(x, np.exp(x)) > 0.99
    assert spearman(x, -x) < -0.99


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
