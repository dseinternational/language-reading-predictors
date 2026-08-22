# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`statistical_models.arm_overlap`.

The overlap figures reuse ``predicted_scores.counterfactual_predictive_contrast``,
so the key guard is that the ``arm_overlap_mean.csv`` average-marginal-effect
row equals the same guard-tested AME that drives ``rope_summary.csv`` /
``predicted_scores.csv``. The rest pins the overlapping-coefficient behaviour and
the two-individual-files (never a shared panel) contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.arm_overlap import (
    overlap_curves,
    write_arm_overlap_artifacts,
)
from language_reading_predictors.statistical_models.predicted_scores import (
    counterfactual_predictive_contrast,
)


def _trace(eta, tau, *, kappa=None):
    n_chain, n_draw, n_obs = eta.shape
    data = {
        "eta": (("chain", "draw", "obs_id"), eta),
        "tau": (("chain", "draw"), tau),
    }
    if kappa is not None:
        data["kappa"] = (("chain", "draw"), kappa)
    coords = {
        "chain": np.arange(n_chain),
        "draw": np.arange(n_draw),
        "obs_id": np.arange(n_obs),
    }
    return SimpleNamespace(posterior=xr.Dataset(data, coords=coords))


def _rng_trace(n_chain=2, n_draw=60, n_obs=12, *, seed=7, kappa=True):
    rng = np.random.default_rng(seed)
    eta = rng.normal(0.0, 1.0, size=(n_chain, n_draw, n_obs))
    tau = rng.normal(0.4, 0.2, size=(n_chain, n_draw))
    kp = rng.uniform(20.0, 60.0, size=(n_chain, n_draw)) if kappa else None
    return eta, tau, kp


# ---------------------------------------------------------------------------
# Overlapping coefficient
# ---------------------------------------------------------------------------


def test_overlap_coefficient_bounds_and_extremes():
    rng = np.random.default_rng(0)
    a = rng.normal(30.0, 3.0, size=4000)
    identical = overlap_curves(a, a.copy())
    assert identical.overlap_coefficient == pytest.approx(1.0, abs=0.05)

    far = overlap_curves(rng.normal(5.0, 1.0, size=4000),
                         rng.normal(90.0, 1.0, size=4000))
    assert far.overlap_coefficient == pytest.approx(0.0, abs=0.02)

    partial = overlap_curves(rng.normal(40.0, 5.0, size=4000),
                            rng.normal(50.0, 5.0, size=4000))
    assert 0.0 < partial.overlap_coefficient < 1.0
    assert partial.density_overlap.shape == partial.grid.shape


# ---------------------------------------------------------------------------
# Drift guard + file contract
# ---------------------------------------------------------------------------


def test_mean_figure_ame_matches_contrast(tmp_path):
    eta, tau, kappa = _rng_trace()
    trace = _trace(eta, tau, kappa=kappa)
    G = np.array([0, 1] * 6, dtype=float)

    tables = write_arm_overlap_artifacts(
        str(tmp_path),
        trace,
        outcome_symbol="W",
        item_label="Word reading (WR)",
        G=G,
        n_trials=79,
        term="tau",
        likelihood="beta_binomial",
        ci_prob=0.89,
        random_seed=0,
    )

    # Independent contrast with the same seed reproduces the plotted AME exactly.
    contrast = counterfactual_predictive_contrast(
        trace, G=G, n_trials=79, term="tau", likelihood="beta_binomial",
        rng=np.random.default_rng(0),
    )
    expected_ame_pp = float(np.median(contrast.ame_prob)) * 100.0

    mean = tables["arm_overlap_mean"]
    ame_row = mean.loc[mean["quantity"] == "average_marginal_effect"].iloc[0]
    assert ame_row["scale"] == "percentage_points"
    assert ame_row["median"] == pytest.approx(expected_ame_pp, rel=1e-9, abs=1e-9)

    overlap = mean.loc[mean["quantity"] == "overlap_coefficient", "median"].iloc[0]
    assert 0.0 <= overlap <= 1.0


def test_guessing_floor_is_forwarded_to_overlap_artifacts(tmp_path):
    eta, tau, kappa = _rng_trace(seed=17)
    trace = _trace(eta, tau, kappa=kappa)
    G = np.array([0, 1] * 6, dtype=float)

    tables = write_arm_overlap_artifacts(
        str(tmp_path),
        trace,
        outcome_symbol="B",
        item_label="Phoneme blending",
        G=G,
        n_trials=40,
        term="tau",
        likelihood="beta_binomial",
        score_mean_link="three_choice_guessing_floor",
        ci_prob=0.89,
        random_seed=3,
    )
    guessing_floor = counterfactual_predictive_contrast(
        trace,
        G=G,
        n_trials=40,
        term="tau",
        likelihood="beta_binomial",
        score_mean_link="three_choice_guessing_floor",
        rng=np.random.default_rng(3),
    )
    ordinary = counterfactual_predictive_contrast(
        trace,
        G=G,
        n_trials=40,
        term="tau",
        likelihood="beta_binomial",
        rng=np.random.default_rng(3),
    )

    mean = tables["arm_overlap_mean"]
    ame_row = mean.loc[mean["quantity"] == "average_marginal_effect"].iloc[0]
    expected_ame_pp = float(np.median(guessing_floor.ame_prob)) * 100.0
    assert ame_row["median"] == pytest.approx(expected_ame_pp, abs=1e-12)
    assert expected_ame_pp == pytest.approx(
        (2.0 / 3.0) * float(np.median(ordinary.ame_prob)) * 100.0,
        abs=1e-12,
    )
    for table in tables.values():
        assert set(table["score_mean_link"]) == {
            "three_choice_guessing_floor"
        }


@pytest.mark.parametrize(
    ("likelihood", "expect_predictive"),
    [("beta_binomial", True), ("bernoulli", False)],
)
def test_individual_files_written(tmp_path, likelihood, expect_predictive):
    eta, tau, kappa = _rng_trace()
    trace = _trace(eta, tau, kappa=kappa)
    G = np.array([0, 1] * 6, dtype=float)

    tables = write_arm_overlap_artifacts(
        str(tmp_path),
        trace,
        outcome_symbol="W",
        item_label="Word reading (WR)",
        G=G,
        n_trials=79 if likelihood == "beta_binomial" else 1,
        term="tau",
        varying_term="" if likelihood == "bernoulli" else "tau_i",
        likelihood=likelihood,
        ci_prob=0.89,
        random_seed=0,
    )

    # Figure 1 is always emitted, as its own PNG + SVG + CSV (never a panel).
    for suffix in (".png", ".svg", ".csv"):
        assert (tmp_path / f"arm_overlap_mean{suffix}").exists()
    assert "arm_overlap_mean" in tables

    predictive_files = [(tmp_path / f"arm_overlap_predictive{s}").exists()
                        for s in (".png", ".svg", ".csv")]
    if expect_predictive:
        assert all(predictive_files)
        assert "arm_overlap_predictive" in tables
    else:
        assert not any(predictive_files)
        assert "arm_overlap_predictive" not in tables


def test_summary_csv_is_tidy(tmp_path):
    eta, tau, kappa = _rng_trace()
    trace = _trace(eta, tau, kappa=kappa)
    tables = write_arm_overlap_artifacts(
        str(tmp_path),
        trace,
        outcome_symbol="W",
        item_label="Word reading (WR)",
        G=np.array([0, 1] * 6, dtype=float),
        n_trials=79,
        term="tau",
        likelihood="beta_binomial",
        ci_prob=0.89,
        random_seed=0,
    )
    saved = pd.read_csv(tmp_path / "arm_overlap_mean.csv")
    assert set(saved["quantity"]) == {
        "no_intervention_level",
        "intervention_level",
        "average_marginal_effect",
        "overlap_coefficient",
        "p_intervention_higher",
    }
    # The saved sidecar matches the returned table.
    pd.testing.assert_frame_equal(saved, tables["arm_overlap_mean"])


# ---------------------------------------------------------------------------
# 2026-08-22 ITT audit regressions (issue #577, finding 4)
# ---------------------------------------------------------------------------


def test_the_two_overlap_figures_do_not_share_contrast_quantity_names():
    """The mean and predictive figures publish different contrasts.

    Both tables used to write ``average_marginal_effect`` and
    ``p_intervention_higher``, so one fit directory carried two contradictory
    values under each name - for lrp-rli-itt-001, 1.368 items [0.19, 2.53] beside
    1.0 items [-5, 8], and 0.968 beside 0.582 - distinguishable only by filename.
    """
    from language_reading_predictors.statistical_models.arm_overlap import (
        arm_overlap_summary,
    )

    draws = np.linspace(-1.0, 3.0, 400)
    shared = {
        "outcome_symbol": "W",
        "control": draws,
        "intervention": draws + 1.0,
        "effect": draws,
        "level_scale": "percent",
        "overlap_coefficient": 0.5,
        "ci_prob": 0.89,
        "population": "pop",
        "contrast_status": "status",
    }
    mean = arm_overlap_summary(effect_scale="percentage_points", **shared)
    predictive = arm_overlap_summary(
        effect_scale="items",
        effect_quantity="predicted_score_difference_independent_children",
        superiority_quantity="p_intervention_child_higher",
        **shared,
    )
    assert "average_marginal_effect" in set(mean.quantity)
    assert "p_intervention_higher" in set(mean.quantity)
    assert not set(mean.quantity) & {
        "predicted_score_difference_independent_children",
        "p_intervention_child_higher",
    }
    assert not set(predictive.quantity) & {
        "average_marginal_effect",
        "p_intervention_higher",
    }
