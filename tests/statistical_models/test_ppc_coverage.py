# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard tests for the posterior-predictive coverage suite (issue #318).

The coverage statistic is a decidable check that band-gazing is not, so it must be
validated against synthetic fits with *known* behaviour: a perfectly-calibrated
predictive should recover ≈ nominal coverage; a point-mass predictive at / far from
the observed value should give coverage 1 / 0; and the closed-interval convention
(observed exactly on a quantile edge counts as inside) must hold. The off-floor
rate path is validated on hand-constructed cells with a known in/out verdict.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.reporting import (
    ppc_calibration_table,
    ppc_coverage_markdown,
    ppc_interval_coverage,
    ppc_offfloor_cell_table,
    ppc_offfloor_rate_coverage,
)
from language_reading_predictors.statistical_models.reporting import (
    _ppc_node_arrays,
)


def _count_trace(rep, obs, *, node="y_post"):
    """A DataTree with a ``posterior_predictive`` + ``observed_data`` count node.

    ``rep`` is ``(chain, draw, obs_id)`` and ``obs`` is ``(obs_id,)``.
    """
    return xr.DataTree.from_dict(
        {
            "posterior_predictive": xr.Dataset(
                {node: (("chain", "draw", "obs_id"), np.asarray(rep, dtype=float))}
            ),
            "observed_data": xr.Dataset(
                {node: (("obs_id",), np.asarray(obs, dtype=float))}
            ),
        }
    )


# --- count-interval coverage -------------------------------------------------


def test_point_mass_predictive_gives_zero_and_one_coverage():
    # Every draw equals 3 for both observations. Observed [3, 5]: the closed interval
    # is the degenerate [3, 3], so obs 3 is INSIDE and obs 5 is OUTSIDE -> 1/2.
    rep = np.full((2, 50, 2), 3.0)
    cov = ppc_interval_coverage(_count_trace(rep, [3, 5]))
    for _, row in cov.iterrows():
        assert row["n_total"] == 2
        assert row["n_inside"] == 1
        assert row["coverage"] == pytest.approx(0.5)

    # All observed exactly on the point mass -> full coverage (closed convention).
    cov_all_in = ppc_interval_coverage(_count_trace(rep, [3, 3]))
    assert (cov_all_in["coverage"] == 1.0).all()

    # All observed off the point mass -> zero coverage.
    cov_none = ppc_interval_coverage(_count_trace(rep, [0, 9]))
    assert (cov_none["coverage"] == 0.0).all()


def test_closed_interval_convention_counts_edge_as_inside():
    # One obs, draws 0..9 (single chain). Level 0.9 -> [q(0.05), q(0.95)] = [0.45, 8.55];
    # level 0.5 -> [q(0.25), q(0.75)] = [2.25, 6.75]. ci_levels order is (0.5, 0.9).
    draws = np.arange(10, dtype=float).reshape(1, 10, 1)
    # observed 0: outside both intervals (0 < 0.45).
    assert ppc_interval_coverage(_count_trace(draws, [0]))["coverage"].tolist() == [0.0, 0.0]
    # observed 1: outside the 50% [2.25, 6.75] but inside the 90% [0.45, 8.55].
    assert ppc_interval_coverage(_count_trace(draws, [1]))["coverage"].tolist() == [0.0, 1.0]
    # observed 5: inside both.
    assert ppc_interval_coverage(_count_trace(draws, [5]))["coverage"].tolist() == [1.0, 1.0]

    # Edge exactly on a quantile endpoint: draws {0, 10} over 2 samples ->
    # q(0.05)=0.5, q(0.95)=9.5 for level .9; construct a case whose lower endpoint is
    # an integer and equals the observed value. draws 0..20 step giving q(0.25)=5.0.
    d = np.array([0, 5, 10, 15, 20], dtype=float).reshape(1, 5, 1)
    lo50 = float(np.quantile(d.ravel(), 0.25))  # 5.0 exactly
    assert lo50 == 5.0
    cov = ppc_interval_coverage(_count_trace(d, [5]), ci_levels=(0.5,))
    assert cov["coverage"].iloc[0] == 1.0  # observed == lower edge -> inside (closed)


def test_wellcalibrated_predictive_recovers_nominal_coverage():
    rng = np.random.default_rng(3)
    n_obs = 400
    lam = rng.integers(3, 25, size=n_obs)
    rep = rng.poisson(lam[None, None, :], size=(4, 400, n_obs)).astype(float)
    obs = rng.poisson(lam).astype(float)  # from the same predictive law
    cov = ppc_interval_coverage(_count_trace(rep, obs)).set_index("level_pct")
    assert cov.loc[90, "coverage"] == pytest.approx(0.90, abs=0.06)
    assert cov.loc[50, "coverage"] == pytest.approx(0.50, abs=0.08)


def test_interval_coverage_matches_direct_recomputation():
    rng = np.random.default_rng(7)
    rep = rng.integers(0, 30, size=(3, 120, 40)).astype(float)
    obs = rng.integers(0, 30, size=40).astype(float)
    cov = ppc_interval_coverage(_count_trace(rep, obs)).set_index("level_pct")
    flat = rep.reshape(-1, 40)  # (samples, obs)
    for pct, p in ((50, 0.5), (90, 0.9)):
        lo = np.quantile(flat, (1 - p) / 2, axis=0)
        hi = np.quantile(flat, (1 + p) / 2, axis=0)
        ref = float(np.mean((obs >= lo) & (obs <= hi)))
        assert cov.loc[pct, "coverage"] == pytest.approx(ref)


def test_interval_coverage_ignores_nonfinite_observed():
    rep = np.full((1, 20, 3), 5.0)
    cov = ppc_interval_coverage(_count_trace(rep, [5.0, np.nan, 5.0]))
    assert (cov["n_total"] == 2).all()  # the NaN row is dropped
    assert (cov["coverage"] == 1.0).all()


# --- calibration table -------------------------------------------------------


def test_calibration_table_matches_coverage_inside_flag():
    rng = np.random.default_rng(11)
    rep = rng.integers(0, 40, size=(2, 100, 25)).astype(float)
    obs = rng.integers(0, 40, size=25).astype(float)
    trace = _count_trace(rep, obs)
    cal = ppc_calibration_table(trace, ci_prob=0.9)
    assert list(cal.columns) == ["observed", "pp_median", "pp_lo", "pp_hi", "inside"]
    assert len(cal) == 25
    # The 90% coverage equals the mean of the calibration table's ``inside`` flag.
    cov90 = ppc_interval_coverage(trace, ci_levels=(0.9,))["coverage"].iloc[0]
    assert cov90 == pytest.approx(cal["inside"].mean())
    assert (cal["pp_lo"] <= cal["pp_median"]).all()
    assert (cal["pp_median"] <= cal["pp_hi"]).all()


# --- node-array extraction ---------------------------------------------------


def test_node_arrays_flatten_multidim_obs_in_order():
    # A (chain, draw, obs_id, domain) predictive with a matching 2-D observed block.
    rep = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
    obs = np.array([[10.0, 11.0], [12.0, 13.0]])  # (obs_id, domain)
    trace = xr.DataTree.from_dict(
        {
            "posterior_predictive": xr.Dataset(
                {"y_obs": (("chain", "draw", "obs_id", "domain"), rep)}
            ),
            "observed_data": xr.Dataset({"y_obs": (("obs_id", "domain"), obs)}),
        }
    )
    y_rep, y_obs = _ppc_node_arrays(trace, "y_obs")
    assert y_rep.shape == (4, 6)  # 2*2 obs cells, 2*3 samples
    assert y_obs.tolist() == [10.0, 11.0, 12.0, 13.0]  # row-major obs order preserved


def test_node_arrays_raises_on_missing_group():
    trace = xr.DataTree.from_dict(
        {"posterior_predictive": xr.Dataset({"y_post": (("chain", "draw", "obs_id"), np.ones((1, 2, 3)))})}
    )
    with pytest.raises(KeyError, match="observed_data"):
        _ppc_node_arrays(trace, "y_post")


# --- off-floor rate coverage -------------------------------------------------


def _offfloor_trace(rep, obs, *, node="y_offfloor"):
    return xr.DataTree.from_dict(
        {
            "posterior_predictive": xr.Dataset(
                {node: (("chain", "draw", "obs_id"), np.asarray(rep, dtype=float))}
            ),
            "observed_data": xr.Dataset(
                {node: (("obs_id",), np.asarray(obs, dtype=float))}
            ),
        }
    )


def test_offfloor_rate_coverage_by_group_cell_known_verdict():
    # Two arms, 4 obs each. Immediate observed off-floor rate 0.5; waitlist 0.0.
    # Predictive: immediate cell always off-floor (rate 1.0) so observed 0.5 sits
    # well below its predictive band -> OUTSIDE. Waitlist predictive is a coin flip
    # (rate ~0.5 with spread) so observed 0.0... construct deterministically instead.
    obs = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # immediate rate .5, waitlist rate 0
    group = np.array(["immediate"] * 4 + ["waitlist"] * 4)
    # Replicated: immediate always 1 (rate 1.0), waitlist always 0 (rate 0.0).
    rep = np.zeros((1, 30, 8))
    rep[:, :, :4] = 1.0
    cov = ppc_offfloor_rate_coverage(
        _offfloor_trace(rep, obs), group=group
    ).set_index("level_pct")
    # 2 cells: immediate observed .5 vs predictive point mass 1.0 -> OUTSIDE;
    # waitlist observed 0 vs predictive point mass 0 -> INSIDE. So 1/2 covered.
    assert cov.loc[90, "n_total"] == 2
    assert cov.loc[90, "n_inside"] == 1
    assert cov.loc[90, "coverage"] == pytest.approx(0.5)
    assert (cov["mode"] == "offfloor_rate").all()
    assert (cov["quantity"] == "observed off-floor rate").all()


def test_offfloor_cell_table_reports_rates_and_counts():
    obs = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    group = np.array(["immediate"] * 4 + ["waitlist"] * 4)
    rep = np.zeros((1, 30, 8))
    rep[:, :, :4] = 1.0
    cells = ppc_offfloor_cell_table(_offfloor_trace(rep, obs), group=group)
    assert cells["cell"].tolist() == ["immediate", "waitlist"]
    assert cells["n"].tolist() == [4, 4]
    assert cells.set_index("cell").loc["immediate", "observed_rate"] == pytest.approx(0.5)
    assert cells.set_index("cell").loc["immediate", "pp_rate_median"] == pytest.approx(1.0)
    assert cells.set_index("cell").loc["immediate", "inside"] == np.False_
    assert cells.set_index("cell").loc["waitlist", "inside"] == np.True_


def test_offfloor_rate_coverage_falls_back_to_single_cell():
    obs = np.array([1, 0, 0, 0])
    rep = np.zeros((1, 20, 4))
    rep[:, :, 0] = 1.0  # predictive rate 0.25, matching observed rate 0.25
    cov = ppc_offfloor_rate_coverage(_offfloor_trace(rep, obs), group=None)
    assert (cov["n_total"] == 1).all()
    assert (cov["unit"] == "off-floor rate").all()


def test_offfloor_treats_raw_counts_as_indicator():
    # A node carrying raw counts (not 0/1) is reduced to the >0 indicator, so a
    # count trace and its off-floor indicator give identical rate coverage.
    counts = np.array([0, 3, 7, 0])
    rep_counts = np.array([[[0, 2, 5, 0]] * 15]).reshape(1, 15, 4)
    ind = ppc_offfloor_rate_coverage(_offfloor_trace((rep_counts > 0), (counts > 0)))
    raw = ppc_offfloor_rate_coverage(_offfloor_trace(rep_counts, counts))
    assert ind["coverage"].tolist() == raw["coverage"].tolist()


# --- markdown renderer -------------------------------------------------------


def test_coverage_markdown_renders_counts_and_verdict():
    import pandas as pd

    cov = pd.DataFrame(
        {
            "mode": ["count_interval"] * 2,
            "node": ["y_post"] * 2,
            "unit": ["observations"] * 2,
            "quantity": ["observed score"] * 2,
            "level": [0.5, 0.9],
            "level_pct": [50, 90],
            "n_total": [153, 153],
            "n_inside": [84, 138],
            "coverage": [84 / 153, 138 / 153],
        }
    )
    md = ppc_coverage_markdown(cov)
    assert "138 of 153" in md
    assert "observations" in md
    assert "expected ≈ 90%" in md
    assert "84 of 153" in md
    assert "same-children" in md  # conditional / in-sample caveat present
    # 138/153 = 0.902 -> close to nominal verdict.
    assert "close to the nominal level" in md


def test_coverage_markdown_flags_undercoverage():
    import pandas as pd

    cov = pd.DataFrame(
        {
            "mode": ["count_interval"],
            "node": ["y_post"],
            "unit": ["observations"],
            "quantity": ["observed score"],
            "level": [0.9],
            "level_pct": [90],
            "n_total": [100],
            "n_inside": [60],
            "coverage": [0.60],
        }
    )
    md = ppc_coverage_markdown(cov)
    assert "too narrow" in md  # 0.60 is well below nominal 0.90


def test_coverage_markdown_flags_overcoverage():
    # Review fix: coverage well ABOVE nominal (100% inside a 90% interval) must not be
    # labelled "close to the nominal level" — it means the ranges are wider than needed.
    import pandas as pd

    cov = pd.DataFrame(
        {
            "mode": ["count_interval"],
            "node": ["y_post"],
            "unit": ["observations"],
            "quantity": ["observed score"],
            "level": [0.9],
            "level_pct": [90],
            "n_total": [100],
            "n_inside": [100],
            "coverage": [1.0],
        }
    )
    md = ppc_coverage_markdown(cov)
    assert "above the nominal level" in md
    assert "wider than the data need" in md
    assert "close to the nominal level" not in md


def test_coverage_markdown_empty_on_none_or_degenerate():
    # Review fix: a zero-observation / NaN-coverage frame must not render "nan%".
    import pandas as pd

    assert ppc_coverage_markdown(None) == ""
    assert ppc_coverage_markdown(pd.DataFrame()) == ""
    degenerate = pd.DataFrame(
        {
            "mode": ["count_interval"] * 2,
            "node": ["y_post"] * 2,
            "unit": ["observations"] * 2,
            "quantity": ["observed score"] * 2,
            "level": [0.5, 0.9],
            "level_pct": [50, 90],
            "n_total": [0, 0],
            "n_inside": [0, 0],
            "coverage": [float("nan"), float("nan")],
        }
    )
    assert ppc_coverage_markdown(degenerate) == ""


def test_interval_coverage_zero_observations_renders_empty():
    # An all-NaN observed vector -> n_total 0, coverage NaN -> markdown renders nothing
    # rather than a "nan%" sentence (end-to-end of the review guard).
    rep = np.full((1, 10, 2), 5.0)
    cov = ppc_interval_coverage(_count_trace(rep, [np.nan, np.nan]))
    assert (cov["n_total"] == 0).all()
    assert ppc_coverage_markdown(cov) == ""
