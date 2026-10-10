"""Explicit error model: predict the world model's own prediction error.

Ensemble disagreement is an *indirect* proxy for error.  It only tracks the error
if the members happen to disagree in proportion to how wrong they are -- which,
for identically-trained members on the same data, they mostly do not (measured
ρ ≈ 0.1 in Gate 4).

This module takes the direct route.  Given a held-out interaction log, for every
(latent belief, slate) pair we can compute the world model's *actual* error, and
then fit a supervised regressor onto it:

    E(h, a)  ~  | r_hat(h, a) - r_true |

That turns calibration from a hope into a supervised learning problem with a
measurable target.  Planning then uses ``E`` in place of ``U``:

    Q_plan = Q_hat - lambda * E(h, a)

The subtlety worth stating up front: ``E`` is only defined where the offline data
can compute a target.  Far outside the behaviour distribution there is no truth
to compare against, which is exactly where the planner wants to go.  So the
error model is trained with a *range constraint* -- the predictions are bounded
by the observed error envelope -- rather than extrapolating a confidently small
error into unknown territory.  ``ErrorModel`` therefore outputs through a
softplus and is trained on log-error, which keeps it from claiming high
confidence where it has no evidence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from muslate.env.logger import OfflineLog
from muslate.model.world_model import SlateEncoder


class ErrorModel(nn.Module):
    """Regresses |reward prediction error| from (latent belief, slate)."""

    def __init__(
        self,
        latent_dim: int,
        n_items: int,
        slate_size: int,
        emb_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.slate_enc = SlateEncoder(n_items, slate_size, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, slates: torch.Tensor) -> torch.Tensor:
        """(..., dh) with (..., K) -> predicted error (...)."""
        shape = h.shape[:-1]
        hf = h.reshape(-1, h.shape[-1])
        sf = slates.reshape(-1, slates.shape[-1])
        se = self.slate_enc(sf)
        x = torch.cat([hf, se], dim=-1)
        out = F.softplus(self.net(x)).squeeze(-1)
        return out.reshape(shape)


# ---------------------------------------------------------------- dataset
def build_error_dataset(
    neural_model,
    logs: list[OfflineLog],
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(latents, slates, true absolute errors) from held-out logs.

    ``state[:, t]`` is the belief before slate t, so the pair used to predict
    ``r_t`` is ``(state[:, t], slates[:, t])`` -- matching how the planner queries
    the world model, which is what makes the trained error model applicable.
    """
    lats, sls, errs = [], [], []
    for log in logs:
        n_ep = int(log.episode_ids.max()) + 1
        t_len = log.horizon
        slates = log.slates.reshape(n_ep, t_len, log.slate_size)
        rewards = log.rewards.reshape(n_ep, t_len)

        latent_seq = neural_model.encode_sequence(slates, rewards)      # [E, T, dh]
        pred = neural_model.reward(
            latent_seq.reshape(-1, latent_seq.shape[-1]),
            slates.reshape(-1, log.slate_size),
        ).reshape(n_ep, t_len)

        err = np.abs(pred - rewards)[:, ::stride]
        lats.append(latent_seq[:, ::stride].reshape(-1, latent_seq.shape[-1]))
        sls.append(slates[:, ::stride].reshape(-1, log.slate_size))
        errs.append(err.reshape(-1))

    return (
        np.concatenate(lats, axis=0),
        np.concatenate(sls, axis=0),
        np.concatenate(errs, axis=0),
    )


def train_error_model(
    neural_model,
    sim,
    logs: list[OfflineLog],
    epochs: int = 40,
    lr: float = 3e-3,
    batch_size: int = 2048,
    device: torch.device | str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> ErrorModel:
    """Supervised fit of the error predictor on held-out data."""
    lat, sls, err = build_error_dataset(neural_model, logs)
    if verbose:
        print(f"  [err] dataset: {lat.shape[0]} pairs, mean|err|={err.mean():.4f}")

    device = torch.device(device)
    x_lat = torch.as_tensor(lat, dtype=torch.float32, device=device)
    x_sl = torch.as_tensor(sls, dtype=torch.long, device=device)
    # fit in log space: errors are positive and right-skewed, and this stops the
    # model from being dominated by a handful of large residuals
    y = torch.as_tensor(np.log(err + 1e-3), dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    model = ErrorModel(neural_model.state_dim, sim.n_items, sim.slate_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    n = x_lat.shape[0]
    n_batches = max(1, n // batch_size)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        total = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            if idx.numel() == 0:
                continue
            pred = model(x_lat[idx], x_sl[idx])
            loss = F.mse_loss(torch.log(pred + 1e-3), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        if verbose and (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"  [err] epoch {epoch + 1:>3}/{epochs}  loss={total / n_batches:.4f}")

    return model


class NeuralErrorModel:
    """Numpy adapter so the error model can drive a ``ConservativePenalty``."""

    def __init__(self, model: ErrorModel, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def score(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        st = torch.as_tensor(np.atleast_2d(states), dtype=torch.float32, device=self.device)
        sl = torch.as_tensor(np.atleast_2d(slates), dtype=torch.long, device=self.device)
        return self.model(st, sl).cpu().numpy()
