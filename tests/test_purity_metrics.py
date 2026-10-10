"""Sweep-metric invariants.

Review finding B1: the derived column `hier - flat` used `max(flat_uniform,
flat_policy)` as its baseline, but `flat_uniform` samples uniformly and never
reads `item_scores` -- so it is *immune* to prior corruption.  When the prior is
the axis being varied, that baseline is a constant, and the "gain" column is just
`sid_hier` shifted by a constant: it measures the y-value it was supposed to
normalise away.

The fix reports both baselines explicitly, and this file pins the arithmetic so a
future edit cannot silently reintroduce the single-baseline form.

Run directly:
    python tests/test_purity_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_purity import summarize_row  # noqa: E402


def test_summarize_row_reports_both_baselines() -> None:
    """Sweep B, noise=1: flat_uniform is pinned at -0.896 by construction."""
    row = summarize_row(
        {
            "flat_uniform": -0.896,
            "flat_policy": -18.221,
            "sid_flat": -3.741,
            "sid_hier": -1.709,
        }
    )
    assert np.isclose(row["hier_minus_flat"], -1.709 - (-0.896))
    assert np.isclose(row["hier_minus_policy"], -1.709 - (-18.221))


def test_the_two_baselines_disagree_exactly_when_the_prior_is_bad() -> None:
    """The whole point of B1: with a clean prior the two columns agree, with a
    corrupted prior they diverge -- and only the second one is a relative claim."""
    clean = summarize_row(
        {"flat_uniform": -0.896, "flat_policy": -1.306, "sid_flat": -0.225, "sid_hier": 1.474}
    )
    dirty = summarize_row(
        {"flat_uniform": -0.896, "flat_policy": -18.221, "sid_flat": -3.741, "sid_hier": -1.709}
    )

    # clean prior: flat_uniform is the better flat, so both baselines coincide
    assert np.isclose(clean["hier_minus_flat"], clean["hier_minus_policy"], atol=0.5)
    # corrupted prior: the immune baseline holds flat up while the score-driven
    # baseline collapses -- the two columns now disagree in SIGN, which is exactly
    # the failure mode B1 warned about
    assert dirty["hier_minus_flat"] < 0.0 < dirty["hier_minus_policy"]


def test_summarize_row_rejects_a_missing_planner() -> None:
    try:
        summarize_row({"flat_uniform": 0.0, "sid_hier": 1.0})
    except KeyError:
        return
    raise AssertionError("summarize_row must require every planner's return")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
