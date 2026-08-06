# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the #125 reporting additions: prior pushforward, ROPE markdown,
and the evidence-label rollout into the plain summary cards."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models.reporting import (
    evidence_label,
    factor_summary,
    joint_prior_pushforward,
    labelled_pushforward,
    marginal_prior_pushforward,
    prior_pushforward,
    pushforward_scale_for,
    pushforward_values,
    rope_markdown,
    tau_summary_itt,
    unavailable_pushforward,
)
from language_reading_predictors.statistical_models.pipeline import (
    _save_contrast_heatmap,
)


def _ds(eta, tau, *, extra=None):
    n_chain, n_draw, n_obs = eta.shape
    data = {
        "eta": (("chain", "draw", "obs_id"), eta),
        "tau": (("chain", "draw"), tau),
    }
    for k, v in (extra or {}).items():
        data[k] = (("chain", "draw"), v)
    return xr.Dataset(
        data,
        coords={
            "chain": np.arange(n_chain),
            "draw": np.arange(n_draw),
            "obs_id": np.arange(n_obs),
        },
    )


def test_prior_pushforward_reads_prior_group():
    rng = np.random.default_rng(1)
    eta = rng.normal(0.0, 1.0, (1, 500, 10))
    tau = rng.normal(0.0, 0.5, (1, 500))
    G = (rng.random(10) > 0.5).astype(float)
    trace = SimpleNamespace(prior=_ds(eta, tau))
    out = prior_pushforward(trace, G=G, n_trials=79, ci_prob=0.95)
    assert out["n_trials"] == 79
    # Prior centred at 0 -> items median near 0; 95% range wide but finite.
    assert abs(out["prior_items_median"]) < 5
    assert out["prior_items_lo"] < out["prior_items_hi"]
    assert out["prior_items_lo50"] >= out["prior_items_lo"]
    assert out["prior_items_hi50"] <= out["prior_items_hi"]


def test_tau_summary_itt_has_direction_label():
    rng = np.random.default_rng(2)
    eta = rng.normal(0.0, 1.0, (2, 300, 8))
    tau = rng.normal(0.5, 0.2, (2, 300))
    G = (rng.random(8) > 0.5).astype(float)
    out = tau_summary_itt(SimpleNamespace(posterior=_ds(eta, tau)), ci_prob=0.95, G=G)
    assert "direction_label" in out
    assert out["direction_label"] == evidence_label(out["prob_tau_pos"])


def test_factor_summary_has_direction_label():
    rng = np.random.default_rng(3)
    beta = rng.normal(0.3, 0.2, (2, 300))
    ds = xr.Dataset(
        {"beta_trt": (("chain", "draw"), beta)},
        coords={"chain": np.arange(2), "draw": np.arange(300)},
    )
    df = factor_summary(
        SimpleNamespace(posterior=ds), ["beta_trt"], ci_prob=0.95, causal_terms=("beta_trt",)
    )
    assert "direction_label" in df.columns
    row = df.iloc[0]
    assert row["role"] == "causal"
    assert row["median"] == np.median(beta)
    assert row["direction_label"] == evidence_label(row["prob_positive"])


def test_rope_markdown_items_and_risk_difference():
    base = {
        "tau_logit_median": 0.3, "tau_logit_lo50": 0.1, "tau_logit_hi50": 0.5,
        "tau_logit_lo": -0.1, "tau_logit_hi": 0.7,
        "items_median": 2.0, "items_lo50": 1.0, "items_hi50": 3.0,
        "items_lo": -0.5, "items_hi": 4.5, "delta_items": 1.0,
        "pd": 0.97, "prob_benefit_ge_delta": 0.85, "prob_in_rope": 0.05,
        "prob_harm_ge_delta": 0.01, "direction_label": "moderate",
        "benefit_label": "suggestive",
    }
    md = rope_markdown(pd.DataFrame([base]), "letter sounds")
    assert "letter sounds" in md
    assert "+2.0 items" in md
    assert "δ" in md

    # Floored risk-difference scale: median ×100 to percentage points, flagged provisional.
    rd = dict(base, delta_scale="risk_difference", provisional_delta=True, items_median=0.12,
              items_lo50=0.05, items_hi50=0.20, items_lo=-0.02, items_hi=0.30, delta_items=0.10)
    md2 = rope_markdown(pd.DataFrame([rd]), "P(off-floor)")
    assert "percentage points" in md2
    assert "provisional" in md2
    assert "+12.0" in md2


def test_probability_contrast_heatmap_saves_with_colorbar(tmp_path):
    contrast = pd.DataFrame(
        [[np.nan, 0.8], [0.2, np.nan]],
        index=["TE", "TR"],
        columns=["TE", "TR"],
    )

    _save_contrast_heatmap(SimpleNamespace(output_dir=str(tmp_path)), contrast)

    assert (tmp_path / "contrast_heatmap.png").exists()
    assert (tmp_path / "contrast_heatmap.svg").exists()


def test_floor_report_renders_same_estimand_bounds_and_gates_secondaries():
    repo = Path(__file__).resolve().parents[2]
    floor_results = (
        repo / "docs/models/_partials/_results_floored.qmd"
    ).read_text(encoding="utf-8")
    diagnostics = (repo / "docs/models/_partials/_diagnostics.qmd").read_text(
        encoding="utf-8"
    )

    assert "floor_transition_missingness_bounds.csv" in floor_results
    assert "full_randomised_population" not in floor_results  # driven by the CSV
    assert "ppc_two_sided_tail" in floor_results
    assert "1 - _p" not in floor_results
    assert "_floor_ppc if _graded_ok else None" in floor_results
    assert "if _graded_ok and _floor_ppc is not None" in floor_results
    assert "Graded posterior-predictive check suppressed" in floor_results
    assert "typically cannot reproduce" not in floor_results
    assert "tau_summary_hurdle.csv" in floor_results
    assert "_hurdle_ok" in floor_results
    assert "Post-outcome selection" in floor_results
    assert "_is_floor_binary" in diagnostics
    assert 'config.get("kind") == "joint"' in diagnostics
    assert "off-floor event rates" in diagnostics


def test_itt_evidence_callout_is_separated_from_preceding_output():
    repo = Path(__file__).resolve().parents[2]
    itt_results = (repo / "docs/models/_partials/_results_itt.qmd").read_text(
        encoding="utf-8"
    )

    assert "'\\n::: {.callout-note title=\"Reading the evidence labels\"}" in itt_results
    assert "point.\\n\\n:::\\n" in itt_results


def test_joint_report_warns_when_persisted_shape_check_flags_p():
    repo = Path(__file__).resolve().parents[2]
    joint_results = (repo / "docs/models/_partials/_results_joint.qmd").read_text(
        encoding="utf-8"
    )
    diagnostics = (repo / "docs/models/_partials/_diagnostics.qmd").read_text(
        encoding="utf-8"
    )

    assert "posterior_predictive_shape_calibration.csv" in joint_results
    assert "ppc_shape_flag" in joint_results
    assert "Posterior-predictive shape misfit limits cross-outcome ranking" in joint_results
    assert 'if _shape_metric_flag(_e, "interquartile_range_count")' in joint_results
    assert 'if _shape_metric_flag(_p, "upper_quartile_count")' in joint_results
    assert "P is the **graded secondary** outcome" in joint_results
    assert "Do not use P's graded joint-model AME unqualified" in joint_results
    assert "posterior_predictive_shape_calibration.csv" in diagnostics
    assert "upper quartile" in diagnostics


def test_diagnostics_report_surfaces_unreliable_pareto_k():
    repo = Path(__file__).resolve().parents[2]
    diagnostics = (repo / "docs/models/_partials/_diagnostics.qmd").read_text(
        encoding="utf-8"
    )

    assert '_csv("pareto_k.csv")' in diagnostics
    assert "_pareto_k > _pareto_thresholds" in diagnostics
    assert "PSIS-LOO requires influence sensitivity" in diagnostics
    assert '_csv("influence_sensitivity.csv")' in diagnostics
    assert "PSIS-LOO unreliable; effect sensitivity completed" in diagnostics
    assert "not reliable for model comparison" in diagnostics
    assert "direct refit excluding all flagged children" in diagnostics
    assert "scripts/influence_sensitivity.py" in diagnostics
    assert "evaluate_influence_bundle" in diagnostics
    assert "_influence_status['reason']" in diagnostics
    assert "passed the trace-recomputed" in diagnostics
    assert "same retained" in diagnostics
    assert "max_refit_ame_shift" in diagnostics
    assert "max_composition_ame_shift" in diagnostics
    assert "max_total_ame_shift" in diagnostics
    assert "ame_prob_median_full_retained" in diagnostics
    assert "refit_shift_ame_prob_median" in diagnostics
    assert "composition_shift_ame_prob_median" in diagnostics
    assert "total_shift_ame_prob_median" in diagnostics
    assert "_influence_view = _influence[" in diagnostics
    assert diagnostics.count("_influence_view\n```") == 1
    assert "PSIS-LOO requires observation-level sensitivity" in diagnostics
    assert "high-k point is a child × phase/period row" in diagnostics
    assert "exact or moment-matched LOO" in diagnostics
    assert "same conditional row-level predictive target" in diagnostics
    assert "leaving out a whole child changes the target" in diagnostics


# ---------------------------------------------------------------------------
# Generalised estimand-scale prior pushforward (#381)
# ---------------------------------------------------------------------------


def test_marginal_pushforward_conventions_differ_and_match_their_transforms():
    """The two conventions must be the transforms the families actually report.

    ``forward`` is ``expit(eta + beta) - expit(eta)`` — the effect of one *more*
    unit, which is what ``concurrent_marginals`` and the dose marginal compute.
    ``net_out`` is ``expit(eta) - expit(eta - beta)`` — the unit already in the
    linear predictor, the ``_itt_ame_draws`` convention. They are evaluated at
    different operating points, so a family must not silently get the other one.
    """
    eta = np.full((1, 4, 3), 0.8)
    beta = np.full((1, 4), 0.5)
    trace = SimpleNamespace(prior=_ds(eta, beta.reshape(1, 4), extra={"b": beta}))

    fwd = marginal_prior_pushforward(
        trace, term="b", n_trials=10, ci_prob=0.9, convention="forward"
    )
    net = marginal_prior_pushforward(
        trace, term="b", n_trials=10, ci_prob=0.9, convention="net_out"
    )
    expected_fwd = 10 * (expit(0.8 + 0.5) - expit(0.8))
    expected_net = 10 * (expit(0.8) - expit(0.8 - 0.5))
    assert fwd["prior_items_median"] == pytest.approx(expected_fwd)
    assert net["prior_items_median"] == pytest.approx(expected_net)
    # Concave inverse link above zero: one more unit buys less than the last one.
    assert fwd["prior_items_median"] < net["prior_items_median"]
    # The coefficient itself is convention-free.
    assert fwd["prior_logit_median"] == pytest.approx(net["prior_logit_median"])


def test_marginal_pushforward_selects_one_element_of_a_vector_term():
    """The horseshoe families rank a ``beta`` indexed by predictor.

    Without ``term_index`` the whole vector would stack into the draw axis and the
    row would summarise a mixture of every predictor's coefficient, which is not a
    quantity the model reports.
    """
    n_draw = 200
    rng = np.random.default_rng(11)
    beta = np.stack(
        [rng.normal(-2.0, 0.01, n_draw), rng.normal(2.0, 0.01, n_draw)], axis=-1
    )[None, ...]  # (1, n_draw, 2)
    ds = xr.Dataset(
        {
            "eta": (("chain", "draw", "obs_id"), np.zeros((1, n_draw, 3))),
            "beta": (("chain", "draw", "predictor"), beta),
        },
        coords={
            "chain": [0],
            "draw": np.arange(n_draw),
            "obs_id": np.arange(3),
            "predictor": ["low", "high"],
        },
    )
    trace = SimpleNamespace(prior=ds)
    low = marginal_prior_pushforward(
        trace, term="beta", n_trials=10, ci_prob=0.9, term_index={"predictor": "low"}
    )
    high = marginal_prior_pushforward(
        trace, term="beta", n_trials=10, ci_prob=0.9, term_index={"predictor": "high"}
    )
    assert low["prior_logit_median"] == pytest.approx(-2.0, abs=0.05)
    assert high["prior_logit_median"] == pytest.approx(2.0, abs=0.05)


def test_labelled_pushforward_keeps_the_numeric_keys_the_blending_bundle_matches():
    """Labels must not enter the numeric dict ``blending_sensitivity`` recomputes.

    That validator intersects the saved CSV's columns with the keys
    ``prior_pushforward`` returns and compares each with a float conversion, so a
    string key in the returned dict would fail the released phoneme-blending
    bundle. The labels therefore live only on the *written* row.
    """
    values = pushforward_values(
        np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]), n_trials=10, ci_prob=0.9
    )
    assert all(isinstance(v, (int, float)) for v in values.values())
    row = labelled_pushforward(
        values, estimand="tau", estimand_label="the treatment effect", role="causal"
    )
    assert set(values) <= set(row)
    assert row["status"] == "ok"
    assert row["role"] == "causal"
    # Every added key is a label, never a number the validator would compare.
    assert set(row) - set(values) == {
        "estimand",
        "estimand_label",
        "role",
        "scale",
        "status",
        "reason",
    }


def test_unavailable_pushforward_records_the_reason_rather_than_vanishing():
    """A check that could not run must stay visible — that is the #381 finding."""
    row = unavailable_pushforward(
        estimand="beta_cohort",
        estimand_label="the cohort contrast",
        role="association",
        reason="this variant fits a single pooled cohort",
    )
    assert row["status"] == "unavailable"
    assert "pooled cohort" in row["reason"]
    assert np.isnan(row["prior_items_median"])


def test_joint_pushforward_gives_each_outcome_its_own_row_and_denominator():
    """A joint fit has one tau and one item ceiling per outcome.

    A single-row schema would have to pick one outcome's denominator and apply it
    to all of them, so the joint family gets one labelled row each.
    """
    n_draw = 300
    rng = np.random.default_rng(7)
    ds = xr.Dataset(
        {
            "eta": (
                ("chain", "draw", "obs_id", "outcome"),
                np.zeros((1, n_draw, 6, 2)),
            ),
            "tau": (("chain", "draw", "outcome"), rng.normal(0, 0.5, (1, n_draw, 2))),
        },
        coords={
            "chain": [0],
            "draw": np.arange(n_draw),
            "obs_id": np.arange(6),
            "outcome": ["W", "L"],
        },
    )
    rows = joint_prior_pushforward(
        SimpleNamespace(prior=ds),
        outcomes=["W", "L"],
        G=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        n_trials={"W": 79, "L": 24},
        ci_prob=0.89,
    )
    assert [r["estimand"] for r in rows] == ["tau[W]", "tau[L]"]
    assert [r["n_trials"] for r in rows] == [79, 24]
    assert all(r["role"] == "causal" for r in rows)
    # Same logit prior, different denominators -> the wider ceiling gives the
    # wider items range.
    assert rows[0]["prior_items_hi"] > rows[1]["prior_items_hi"]


def test_scale_is_derived_from_the_denominator_so_it_cannot_disagree():
    """A denominator of 1 means a probability difference, never an item count.

    Eight pre-#381 fits (the floor-rule ITT, gain-factor, level-factor and DiD
    outcomes) published an off-floor risk difference described as items, and so
    a hundred times too small. Deriving the scale from ``n_trials`` in one place
    means no call site can reintroduce that by passing a mismatched pair.
    """
    assert pushforward_scale_for(79) == "items"
    assert pushforward_scale_for(1) == "percentage points"

    values = pushforward_values(
        np.array([0.0]), np.array([0.05]), n_trials=1, ci_prob=0.9
    )
    row = labelled_pushforward(
        values,
        estimand="tau",
        estimand_label="the off-floor treatment effect",
        role="causal",
    )
    assert row["scale"] == "percentage points"
    # An explicit scale still wins, for a family whose units are neither.
    override = labelled_pushforward(
        values, estimand="tau", estimand_label="x", role="causal", scale="words/year"
    )
    assert override["scale"] == "words/year"
