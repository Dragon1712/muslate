"""Numpy adapter that exposes the trained (ensemble) world model through the same
:class:`~muslate.planning.latent_model.LatentModel` interface as the simulator.

Having both behind one interface is what makes the Gate 2 comparison honest:
the *same* planner code runs against true dynamics and learned dynamics, so any
gap is attributable to the model rather than to the search.
"""

from __future__ import annotations

import numpy as np
import torch

from muslate.model.world_model import WorldModel
from muslate.planning.latent_model import LatentModel


class NeuralLatentModel(LatentModel):
    def __init__(self, models: list[WorldModel], device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.models = [m.eval().to(self.device) for m in models]
        first = self.models[0]
        self.state_dim = first.cfg.latent_dim
        self.n_items = first.n_items
        self.slate_size = first.slate_size

    # -- helpers -----------------------------------------------------------
    def _t(self, x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    @torch.no_grad()
    def encode_history(self, slates: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """slates [B, T, K], rewards [B, T] -> initial beliefs [B, dh]."""
        sl = self._t(np.atleast_2d(slates) if slates.ndim == 2 else slates, torch.long)
        rw = self._t(rewards)
        if sl.ndim == 2:
            sl = sl[:, None, :]
            rw = rw[:, None]
        hs = [m.encode_history(sl, rw) for m in self.models]
        return torch.stack(hs).mean(0).cpu().numpy()

    @torch.no_grad()
    def initial_latent(self) -> np.ndarray:
        """Cold-start belief with an empty history.  [1, dh]"""
        hs = [m.h0 for m in self.models]
        return torch.stack(hs).mean(0).cpu().numpy()[None].astype(np.float64)

    @torch.no_grad()
    def encode_sequence(self, slates: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Teacher-forced beliefs for a whole batch of sequences.

        slates [E, T, K], rewards [E, T] -> [E, T, dh], where ``[:, t]`` is the
        belief before slate t.  One pass instead of T separate ``encode_history``
        calls, which matters when building an error-model dataset.
        """
        sl = self._t(np.asarray(slates), torch.long)
        rw = self._t(np.asarray(rewards))
        hs = [m.rollout(sl, rw) for m in self.models]
        return torch.stack(hs).mean(0).cpu().numpy()

    # -- LatentModel -------------------------------------------------------
    @torch.no_grad()
    def reward(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        st, sl = self._t(np.atleast_2d(states)), self._t(np.atleast_2d(slates), torch.long)
        outs = [m.predict_reward(st, sl) for m in self.models]
        return torch.stack(outs).mean(0).cpu().numpy()

    @torch.no_grad()
    def step(self, states: np.ndarray, slates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        st, sl = self._t(np.atleast_2d(states)), self._t(np.atleast_2d(slates), torch.long)
        rs, ns = [], []
        for m in self.models:
            rs.append(m.predict_reward(st, sl))
            ns.append(m.step_latent(st, sl))
        r = torch.stack(rs).mean(0).cpu().numpy()
        nxt = torch.stack(ns).mean(0).cpu().numpy()
        return r, nxt

    @torch.no_grad()
    def item_scores(self, states: np.ndarray) -> np.ndarray:
        st = self._t(np.atleast_2d(states))
        outs = [m.item_scores(st) for m in self.models]
        return torch.stack(outs).mean(0).cpu().numpy()

    # -- extras used by conservative planning ------------------------------
    @torch.no_grad()
    def uncertainty(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        """Ensemble disagreement on the predicted reward.  [B]."""
        st, sl = self._t(np.atleast_2d(states)), self._t(np.atleast_2d(slates), torch.long)
        outs = [m.predict_reward(st, sl) for m in self.models]
        return torch.stack(outs).std(0).cpu().numpy()

    @torch.no_grad()
    def latent_uncertainty(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        """Ensemble disagreement on the predicted next latent state.  [B]."""
        st, sl = self._t(np.atleast_2d(states)), self._t(np.atleast_2d(slates), torch.long)
        outs = [m.step_latent(st, sl) for m in self.models]
        return torch.stack(outs).std(0).norm(dim=-1).cpu().numpy()
