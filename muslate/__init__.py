"""MuSlate: uncertainty-aware hierarchical latent planning for generative slate
recommendation.

This package holds the *gate* experiments that must pass before the full system
is worth building:

* Gate 2 -- is multi-step planning worth anything over one-step lookahead?
  Answer: only when the environment has non-trivial long-horizon dependency.
* Gate 3 -- does a Semantic-ID hierarchy buy anything over flat slate actions?

Everything runs on a fully controllable synthetic slate simulator so that the
ground-truth user dynamics are known and the long-term dependency strength can
be dialled up and down.
"""

__version__ = "0.1.0"
