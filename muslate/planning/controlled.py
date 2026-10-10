"""Controlled model error: ground-truth dynamics with an injected, exploitable bias.

The learned-model version of Gate 4 is confounded.  When the world model is weak
the planning gain is wiped out by ordinary prediction error, so the specific
failure the framework is exposed to -- the planner *actively hunting* the region
where the model over-estimates -- cannot be separated from "the model is just
bad".  Two questions get tangled:

    (a) does a planner exploit model error when error is present?
    (b) is our uncertainty estimate good enough to stop it?

This module separates them.  It wraps the simulator and adds a bias that the
planner *can* find:

    reward_model(s, A) = reward_true(s, A) + bias_strength * <w, mean(z_A)>

``w`` is a fixed random direction, unknown to the planner.  A planner optimising
``reward_model`` will prefer slates aligned with ``w``; their true reward is
unbiased, so those slates are simply worse.  Increasing lookahead therefore
*reduces* true return -- exploitation, on demand, with a single dial.

``uncertainty`` returns the exact magnitude of the bias a slate can receive, i.e.
a *perfect* uncertainty estimate.  Running the penalty with it answers the upper
bound question: if the uncertainty estimate were ideal, would conservative
planning work?  That is the right thing to know before blaming the estimator.
"""

from __future__ import annotations

import numpy as np

from muslate.planning.latent_model import OracleModel


class BiasedOracleModel(OracleModel):
    """Simulator + a systematic, state-coupled reward bias.

    The bias is not a constant offset.  It is modulated by the accumulated
    saturation::

        bias(s, A) = strength * <w, mean(z_A)> * (1 + coupling * sum(sat))

    That coupling is what makes it *exploitable by planning* rather than merely
    wrong.  A model-blind planner sees "saturating the user makes the next slate
    look better", so it plans to saturate; the real environment penalises exactly
    that.  The deeper the lookahead, the more thoroughly it walks into the trap --
    which is the failure mode this experiment needs to reproduce on demand.  A
    state-independent bias cannot do this: it is just a wrong constant, and
    multi-step planning still adds its usual value on top.
    """

    def __init__(
        self,
        sim,
        bias_strength: float = 1.0,
        state_coupling: float = 0.5,
        seed: int = 0,
        perfect_uncertainty: bool = True,
    ):
        super().__init__(sim)
        rng = np.random.default_rng(seed)
        w = rng.normal(size=sim.cfg.dim)
        self.w = w / np.linalg.norm(w)
        self.bias_strength = float(bias_strength)
        self.state_coupling = float(state_coupling)
        self.perfect_uncertainty = perfect_uncertainty

    # -- the injected error ------------------------------------------------
    def bias(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        """The (unknown to the planner) additive bias for each (state, slate). [B]"""
        states = np.atleast_2d(states)
        slates = np.atleast_2d(slates)
        z = self.sim.catalog.item_emb[slates].mean(axis=1)     # [B, dim]
        direction = z @ self.w
        _h, sat = self._split(states)
        mod = 1.0 + self.state_coupling * sat.sum(axis=1)
        return self.bias_strength * direction * mod

    # -- LatentModel with the bias applied --------------------------------
    def reward(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        return super().reward(states, slates) + self.bias(states, slates)

    def step(self, states: np.ndarray, slates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r, nxt = super().step(states, slates)
        return r + self.bias(states, slates), nxt

    # -- uncertainty --------------------------------------------------------
    def uncertainty(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        """Perfect epistemic uncertainty: exactly how much bias a slate can carry.

        With ``perfect_uncertainty=False`` it returns a *miscalibrated* proxy
        instead -- a fixed constant -- which is what a badly behaving estimator
        looks like, so the two can be compared honestly.
        """
        if not self.perfect_uncertainty:
            return np.full(np.atleast_2d(slates).shape[0], 0.5, dtype=np.float64)
        return np.abs(self.bias(states, slates))
