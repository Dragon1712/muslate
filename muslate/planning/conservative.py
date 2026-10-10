"""Conservative planning: penalise the planner for exploiting world-model error.

The risk this addresses is the one the framework is most exposed to.  A planner
does not search for the action that is best in reality; it searches for the
action that *scores highest under the model*.  Those coincide only where the
model is accurate.  Where it is not, the planner finds the region of maximum
over-estimation -- model error is not merely propagated, it is actively
*optimised against*.

The remedy is to make the planning objective pessimistic in exactly the two
places where the model is untrustworthy:

    Q_plan(h, a) = Q_hat(h, a) - lambda * U(h, a) - beta * D(a, pi_beta)

* ``U(h, a)``  epistemic uncertainty -- ensemble disagreement.  Where the
  members disagree, the model does not know, and an optimistic planner will
  exploit that.
* ``D(a, pi_beta)``  distance from the behaviour support -- how unlikely this
  slate would have been under the logging policy.  Model predictions far outside
  the logged distribution are extrapolations, and extrapolations are where the
  over-estimation lives.

Both plug into the ``adjust`` hook that every planner in :mod:`muslate.planning.search`
already accepts, so no search code has to change.
"""

from __future__ import annotations

import numpy as np

from muslate.env.logger import OfflineLog


class BehaviorSupport:
    """Novelty of a slate relative to the offline log, as a z-score.

    ``score`` is the mean negative log-probability of the slate's items under the
    empirical item distribution of the log, rescaled to standard deviations.

    The rescaling is not cosmetic.  Raw NLL is almost constant across candidate
    slates -- with a few hundred items the empirical distribution is flat enough
    that every slate scores around ``-log(1/N)`` with a standard deviation under
    0.02, so a raw penalty would spend its budget distinguishing nothing.  In
    z-score units, "one unit" means "one standard deviation rarer than typical",
    which is both interpretable and actually discriminative.

    This is a deliberately simple density proxy; a slate-level (rather than
    item-level) density estimate would be better and is a natural extension, but
    the quantity that matters -- whether the candidate is an extrapolation -- is
    already captured.
    """

    def __init__(self, log: OfflineLog, smooth: float = 1.0):
        counts = np.bincount(log.slates.ravel(), minlength=log.n_items).astype(np.float64)
        probs = (counts + smooth) / (counts.sum() + smooth * log.n_items)
        self.logp = np.log(probs)
        self.mu = float(self.logp.mean())
        self.sigma = float(self.logp.std()) + 1e-8
        self.smooth = smooth

    def score(self, slates: np.ndarray) -> np.ndarray:
        """slates [B, K] -> novelty z-score [B]; positive means outside the support."""
        slates = np.atleast_2d(slates)
        vals = self.logp[slates].mean(axis=1)
        return (self.mu - vals) / self.sigma


class LatentSupport:
    """Novelty in *belief* space: distance from the nearest logged latent state.

    This is the OOD term that matters.  Item-frequency novelty is nearly useless
    here because the logging policy is close to uniform across items, so almost
    every slate looks equally (im)plausible.  What actually happens under
    planning is that rollouts drive the *belief* into regions the offline data
    never covered, and the model's reward head is an extrapolation there.  So the
    useful question is "has the model ever seen a state like this one", not "has
    it seen an item like this one".
    """

    def __init__(self, latents: np.ndarray, max_points: int = 2000, seed: int = 0):
        latents = np.asarray(latents, dtype=np.float64)
        if latents.shape[0] > max_points:
            idx = np.random.default_rng(seed).choice(latents.shape[0], max_points, replace=False)
            latents = latents[idx]
        self.ref = latents

    def score(self, states: np.ndarray) -> np.ndarray:
        """states [B, D] -> distance to the nearest logged belief [B]."""
        states = np.atleast_2d(states)
        # [B, R, D] -> min over R
        d = np.linalg.norm(states[:, None, :] - self.ref[None, :, :], axis=-1)
        return d.min(axis=1)


class ConservativePenalty:
    """Callable that turns a reward estimate into a pessimistic planning value.

    Signature matches the ``adjust`` hook used by the planners:
    ``(states [B, D], slates [B, K], rewards [B]) -> adjusted rewards [B]``.
    """

    def __init__(
        self,
        model,
        lambda_unc: float = 0.0,
        beta_ood: float = 0.0,
        support: BehaviorSupport | None = None,
        state_support: LatentSupport | None = None,
        uncertainty_fn=None,
        unc_scale: float = 1.0,
        ood_scale: float = 1.0,
    ):
        self.model = model
        self.lambda_unc = lambda_unc
        self.beta_ood = beta_ood
        self.support = support
        self.state_support = state_support
        self.uncertainty_fn = uncertainty_fn
        self.unc_scale = unc_scale
        self.ood_scale = ood_scale
        if lambda_unc > 0 and uncertainty_fn is None and not hasattr(model, "uncertainty"):
            raise TypeError(
                "model has no uncertainty(); pass uncertainty_fn or use an ensemble"
            )
        if beta_ood > 0 and support is None and state_support is None:
            raise ValueError("beta_ood > 0 requires a support estimate")

    # -- uncertainty source -------------------------------------------------
    def uncertainty(self, states: np.ndarray, slates: np.ndarray) -> np.ndarray:
        """Either the ensemble disagreement or a supplied (e.g. learned) estimate.

        Letting this be swapped is the point: Gate 4 showed the penalty is only
        as good as its signal, and a supervised holdout error model is a
        different -- better calibrated -- signal than ensemble disagreement.
        """
        if self.uncertainty_fn is not None:
            return np.asarray(self.uncertainty_fn(states, slates), dtype=np.float64)
        return np.asarray(self.model.uncertainty(states, slates), dtype=np.float64)

    # -- diagnostics --------------------------------------------------------
    def components(self, states: np.ndarray, slates: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if self.lambda_unc > 0:
            out["uncertainty"] = self.uncertainty(states, slates)
        if self.beta_ood > 0:
            if self.state_support is not None:
                out["state_ood"] = self.state_support.score(states)
            if self.support is not None:
                out["item_ood"] = self.support.score(slates)
        return out

    # -- the hook -----------------------------------------------------------
    def __call__(self, states: np.ndarray, slates: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        out = np.array(rewards, dtype=np.float64, copy=True)
        if self.lambda_unc > 0:
            out = out - self.lambda_unc * self.unc_scale * self.uncertainty(states, slates)
        if self.beta_ood > 0:
            if self.state_support is not None:
                out = out - self.beta_ood * self.ood_scale * self.state_support.score(states)
            if self.support is not None:
                out = out - self.beta_ood * self.ood_scale * self.support.score(slates)
        return out


class AdaptiveHorizon:
    """State-level uncertainty gate for the planning horizon.

    Estimates how uncertain the model is about *this state* (by scoring a small
    probe pool) and returns a horizon:

        U < u_low            -> full horizon
        u_low <= U < u_high  -> reduced horizon
        U >= u_high          -> single step

    This is the "uncertainty-adaptive planning horizon" idea: where the model is
    confident, plan deep; where it is not, do not trust long rollouts.  It is
    cheaper and more interpretable than a continuous penalty, at the cost of
    being discrete.
    """

    def __init__(
        self,
        model,
        proposal,
        horizons: tuple[int, int, int] = (5, 3, 1),
        u_low: float = 0.05,
        u_high: float = 0.2,
        n_probe: int = 16,
        seed: int = 0,
    ):
        self.model = model
        self.proposal = proposal
        self.horizons = horizons
        self.u_low = u_low
        self.u_high = u_high
        self.n_probe = n_probe
        self.rng = np.random.default_rng(seed)

    def state_uncertainty(self, latent: np.ndarray) -> float:
        lat = np.atleast_2d(latent)
        pool = self.proposal.propose(lat)[0][: self.n_probe]
        rep = np.repeat(lat, pool.shape[0], axis=0)
        u = np.asarray(self.model.uncertainty(rep, pool), dtype=np.float64)
        return float(u.mean())

    def horizon(self, latent: np.ndarray) -> int:
        u = self.state_uncertainty(latent)
        if u < self.u_low:
            return self.horizons[0]
        if u < self.u_high:
            return self.horizons[1]
        return self.horizons[2]
