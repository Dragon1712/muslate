"""Configuration dataclasses for the MuSlate research prototype."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CatalogConfig:
    """Item catalogue with a two-level (coarse, fine) Semantic-ID structure."""

    n_coarse: int = 10
    n_fine: int = 20
    dim: int = 32
    # The coarse level dominates the item embedding so that the hierarchy is
    # actually meaningful (items in the same coarse group are similar).
    coarse_weight: float = 1.0
    fine_weight: float = 0.5
    seed: int = 0

    @property
    def n_items(self) -> int:
        return self.n_coarse * self.n_fine


@dataclass
class EnvConfig:
    """Ground-truth slate environment.

    ``long_term_strength`` (lt) is the single knob that controls how much the
    current slate changes the future.  It scales both the delayed saturation
    penalty and the preference drift:

    * lt == 0  -> a contextual bandit.  Immediate greedy is optimal and *no*
      amount of lookahead can help.  This is the regime MF-MDP reports.
    * lt == 1  -> strong long-horizon coupling.  Lookahead should pay off.
    """

    catalog: CatalogConfig = field(default_factory=CatalogConfig)

    # slate geometry
    slate_size: int = 5
    pos_discount: float = 0.8

    # dynamics
    sat_decay: float = 0.7
    sat_penalty: float = 0.5
    pref_lr: float = 0.3
    long_term_strength: float = 1.0

    # Intra-slate diversity penalty.  Penalises pairs of similar items showing
    # up in the same slate:
    #     r -= div_penalty * sum_{j<k} w_j w_k <z_j, z_k>
    # This is an *immediate* combinatorial term, deliberately independent of
    # `long_term_strength`: it changes what the myopic optimum is (so per-position
    # greedy stops being optimal and a coarse-to-fine search has something to
    # exploit) without changing the long-horizon structure that Gate 2 measures.
    # Items from the same coarse category are more similar, so this term is what
    # makes the Semantic-ID hierarchy potentially useful.
    div_penalty: float = 0.0

    # observation noise on the logged reward
    reward_noise: float = 0.02

    # episode
    horizon: int = 20
    gamma: float = 0.95

    # user population
    n_users: int = 64
    seed: int = 0

    @property
    def n_items(self) -> int:
        return self.catalog.n_items

    @property
    def dim(self) -> int:
        return self.catalog.dim

    @property
    def n_coarse(self) -> int:
        return self.catalog.n_coarse


@dataclass
class WorldModelConfig:
    """Learned latent world model."""

    latent_dim: int = 64
    slate_emb_dim: int = 64
    hidden_dim: int = 128
    n_ensemble: int = 3
    # Supervise the latent state directly against the simulator state.
    # This is the "oracle-assisted" setting: it isolates planning quality from
    # representation quality.  Set to False for the honest partially-observed
    # setting where the state must be inferred from the interaction history.
    oracle_state: bool = True
    lr: float = 3e-3
    batch_size: int = 256
    epochs: int = 30
    rollout_depth: int = 3  # multi-step training horizon
    # Reward-head structure.  With `pairwise` the head becomes
    #     r = sum_k w_k u(h, i_k)  +  sum_{j<k} w_j w_k * e_j^T W e_k
    # The environment's diversity penalty is a sum over item PAIRS, so a purely
    # unary head cannot represent it at all -- it is a *structural* mismatch, not
    # a capacity or data problem.  Set False to reproduce the old model for the
    # ablation.
    pairwise: bool = True
    # loss weights
    w_reward: float = 1.0
    w_multistep: float = 1.0
    w_state: float = 1.0
    w_score: float = 1.0
    # oracle_full is the legacy privileged upper bound; strict logged paths
    # accept only none or exposed (observed feedback for served item IDs).
    score_supervision: str = 'oracle_full'
    seed: int = 0


@dataclass
class PlanConfig:
    """Search / planning configuration."""

    n_sims: int = 32          # MCTS simulations per decision
    n_candidates: int = 24    # candidate slates expanded per node
    c_puct: float = 1.5
    max_depth: int = 3        # lookahead depth H
    gamma: float = 0.95
    # conservative planning terms (Gate 4)
    unc_penalty: float = 0.0   # lambda * U(h, a)
    ood_penalty: float = 0.0   # beta * D(a, behaviour support)
    seed: int = 0


@dataclass
class LogConfig:
    """Offline interaction log generation."""

    n_episodes: int = 2000
    epsilon: float = 0.25
    temperature: float = 1.0
    seed: int = 0
    # Independent, observed item-level continuous feedback in synthetic logs.
    item_feedback_noise: float = 0.02
