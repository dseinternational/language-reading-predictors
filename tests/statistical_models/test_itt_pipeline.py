# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fast end-to-end regression tests for the ITT-family pipeline branches.

These tests exercise the real orchestration and artifact writers while replacing
posterior sampling, diagnostic plotting, and Quarto copying with deterministic
stubs. Factory construction itself is covered separately by the registered-model
build tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import pipeline
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.factories import BuiltModel
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.preprocessing import (
    PreparedData,
    Standardiser,
)


class _FakeModel:
    """Small context-manager stand-in for the secondary-fit PyMC models."""

    def __init__(self, names=("alpha", "tau", "gamma_A", "kappa")):
        self.free_RVs = [SimpleNamespace(name=name) for name in names]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeTrace:
    """Trace surface used by the pipeline after the sampler has been stubbed."""

    def __init__(self, outcomes=()):
        self.posterior = {"outcome": SimpleNamespace(values=np.asarray(outcomes, dtype=object))}

    def to_netcdf(self, path):
        Path(path).write_text("mock trace\n")


def _prepared(
    outcomes: dict[str, np.ndarray],
    pre: dict[str, np.ndarray],
    *,
    group: np.ndarray,
    n_trials: dict[str, int],
    covariates: dict[str, np.ndarray] | None = None,
) -> PreparedData:
    n = len(group)
    age = np.linspace(72.0, 108.0, n)
    age_std = (age - age.mean()) / age.std(ddof=1)
    covariates = covariates or {}
    return PreparedData(
        subject_ids=np.asarray([f"child-{i:02d}" for i in range(n)]),
        child_idx=np.arange(n),
        phase=np.zeros(n, dtype=int),
        G=np.asarray(group, dtype=int),
        A_months=age,
        A_std=age_std,
        age_scaler=Standardiser(mean=float(age.mean()), sd=float(age.std(ddof=1))),
        pre_logit={key: np.asarray(value, dtype=float) for key, value in pre.items()},
        post_counts={key: np.asarray(value, dtype=float) for key, value in outcomes.items()},
        n_trials=n_trials,
        n_obs=n,
        n_children=n,
        n_phases=1,
        dropped_rows=0,
        phase_mode="itt",
        column_map={key: key for key in outcomes},
        covariates=covariates,
        covariate_time={key: "baseline" for key in covariates},
        data_path="synthetic-rli.csv",
        data_sha256="synthetic-sha256",
    )


def test_write_loo_influence_persists_subject_and_reliability_gate(tmp_path):
    ctx = SimpleNamespace(
        loo=SimpleNamespace(pareto_k=np.array([0.21, 0.82, 0.47]), good_k=0.7),
        prepared=SimpleNamespace(subject_ids=np.array(["a", "b", "c"])),
        output_dir=str(tmp_path),
        tables={},
    )

    result = pipeline._write_loo_influence(ctx)

    assert result is not None
    assert result["subject_id"].tolist() == ["b", "c", "a"]
    assert result["observation_index"].tolist() == [1, 2, 0]
    assert result["loo_reliable"].tolist() == [False, True, True]
    persisted = pd.read_csv(tmp_path / "pareto_k.csv")
    assert persisted["good_k_threshold"].eq(0.7).all()
    assert "pareto_k" in ctx.tables


@pytest.fixture
def fast_pipeline(monkeypatch, tmp_path):
    """Keep real branch and artifact logic, replacing only expensive fit phases."""

    root = tmp_path / "models"

    def make_context(spec, config="dev"):
        out = root / f"{spec.model_id}-{config}"
        out.mkdir(parents=True)
        return SimpleNamespace(
            spec=spec,
            reporting=SimpleNamespace(output_dir=str(out), ci_prob=0.95),
            sampling=SimpleNamespace(
                draws=2,
                tune=2,
                chains=1,
                cores=1,
                target_accept=0.9,
                random_seed=47,
            ),
            prepared=None,
            model=None,
            model_vars=None,
            prior_samples=None,
            trace=None,
            loo=None,
            tables={},
            resolved_plan=None,
            output_dir=str(out),
        )

    def sample_and_loo(ctx, *, compute_loo=True):
        outcomes = (
            ctx.resolved_plan.outcomes
            if ctx.resolved_plan is not None
            else tuple(ctx.spec.extra.get("outcomes", ()))
        )
        ctx.trace = _FakeTrace(outcomes)
        ctx.loo = SimpleNamespace(elpd=-12.5)

    def write_analysis_audit(ctx, prepared, outcomes):
        table = pd.DataFrame([{"outcome": symbol, "fitted_n": prepared.n_obs} for symbol in outcomes])
        table.to_csv(Path(ctx.output_dir) / "analysis_set.csv", index=False)
        table.to_csv(Path(ctx.output_dir) / "attrition_bounds.csv", index=False)
        ctx.tables["analysis_set"] = table

    def write_ppc_audit(ctx, prepared, outcomes, **kwargs):
        table = pd.DataFrame([{"outcome": symbol, "n": prepared.n_obs} for symbol in outcomes])
        filename = kwargs.get("filename", "posterior_predictive_calibration.csv")
        table.to_csv(Path(ctx.output_dir) / filename, index=False)
        ctx.tables[Path(filename).stem] = table
        return table

    monkeypatch.setattr(pipeline, "make_context", make_context)
    monkeypatch.setattr(pipeline, "_run_sampling_and_loo", sample_and_loo)
    monkeypatch.setattr(pipeline, "_run_ppc", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_write_itt_analysis_audit", write_analysis_audit)
    monkeypatch.setattr(pipeline, "_write_itt_ppc_calibration", write_ppc_audit)
    monkeypatch.setattr(pipeline, "_finalize_report", lambda ctx: ctx)

    for name in (
        "_print_header",
        "_render_model_graph",
        "_emit_priors",
        "_emit_itt_extras",
        "_save_rope_plot",
        "_save_proportion_at_zero_plot",
        "_save_forest_plot",
        "_save_contrast_heatmap",
        "print_table",
        "section_header",
    ):
        monkeypatch.setattr(pipeline, name, lambda *args, **kwargs: None)

    for name in (
        "run_prior_predictive",
        "save_prior_predictive_plot",
        "summary_diagnostics",
        "sample_posterior_predictive",
        "save_joint_posterior_predictive_plot",
        "save_joint_loo_pit_plot",
        "write_diagnostics_summary",
        "run_extended_diagnostics",
        "save_prior_posterior_plot",
    ):
        monkeypatch.setattr(pipeline._diag, name, lambda *args, **kwargs: None)

    def save_trace(ctx):
        Path(ctx.output_dir, "trace.nc").write_text("mock primary trace\n")

    monkeypatch.setattr(pipeline._diag, "save_trace", save_trace)
    monkeypatch.setattr(
        pipeline._diag,
        "subfit_convergence",
        lambda *args, **kwargs: {
            "converged": True,
            "max_rhat": 1.0,
            "min_ess": 800.0,
            "min_bfmi": 0.8,
            "n_divergences": 0,
        },
    )

    monkeypatch.setattr(pm, "sample", lambda *args, **kwargs: _FakeTrace())
    monkeypatch.setattr(
        pm,
        "sample_posterior_predictive",
        lambda trace, *args, **kwargs: trace,
    )

    monkeypatch.setattr(
        pipeline._report,
        "tau_summary_itt",
        lambda *args, **kwargs: {
            "tau_prob_median": 0.08,
            "tau_prob_lo": -0.02,
            "tau_prob_hi": 0.18,
            "prob_ame_pos": 0.92,
            "prob_tau_pos": 0.92,
            "prob_tau_logit_pos": 0.90,
        },
    )
    monkeypatch.setattr(
        pipeline._report,
        "tau_summary_offfloor",
        lambda *args, **kwargs: {
            "tau_prob_median": 0.10,
            "tau_prob_lo": -0.05,
            "tau_prob_hi": 0.25,
            "prob_ame_pos": 0.88,
            "prob_tau_pos": 0.88,
            "prob_tau_logit_pos": 0.84,
        },
    )
    monkeypatch.setattr(
        pipeline._report,
        "rope_summary",
        lambda *args, **kwargs: {
            "delta": kwargs["delta"],
            "prob_benefit_at_least_delta": 0.70,
        },
    )
    monkeypatch.setattr(
        pipeline._report,
        "rope_sensitivity",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "delta": list(kwargs["deltas"]),
                "prob_benefit_at_least_delta": 0.5,
            }
        ),
    )
    monkeypatch.setattr(pipeline._report, "tau_moderation_summary", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        pipeline._report,
        "proportion_at_zero_ppc",
        lambda *args, **kwargs: {
            "obs_prop_at_zero": 0.75,
            "ppc_mean_prop_at_zero": 0.60,
            "ppc_upper_tail": 0.20,
            "ppc_lower_tail": 0.85,
            "ppc_two_sided_tail": 0.40,
            "ppc_p_value": 0.20,
            "rep": np.array([0.5, 0.7]),
        },
    )

    return root


@pytest.mark.parametrize(
    ("symbol", "group", "fitted_post"),
    [
        (
            "W",
            np.array([1, 1, 1, 1, 0, 0, 0]),
            np.array([1, 2, 3, 4, 2, 3, 4], dtype=float),
        ),
        (
            "P",
            np.array([1, 1, 1, 0, 0]),
            np.array([0, 1, 0, 0, 2], dtype=float),
        ),
    ],
    ids=("ordinary", "floor-restricted"),
)
def test_write_itt_analysis_audit_real_writer_for_single_outcome(tmp_path, monkeypatch, symbol, group, fitted_post):
    """The real writer maps both ordinary and floor-restricted fitted frames."""

    fitted = _prepared(
        {symbol: fitted_post},
        {symbol: np.linspace(-2.0, 2.0, group.size)},
        group=group,
        n_trials={symbol: 10},
    )
    archive_group = np.array([1] * 6 + [0] * 6)
    archive = _prepared(
        {symbol: np.arange(12, dtype=float) % 8},
        {symbol: np.linspace(-2.0, 2.0, 12)},
        group=archive_group,
        n_trials={symbol: 10},
    )
    load_calls = []
    monkeypatch.setattr(
        pipeline,
        "load_and_prepare",
        lambda **kwargs: load_calls.append(kwargs) or archive,
    )
    ctx = SimpleNamespace(output_dir=str(tmp_path), tables={})

    pipeline._write_itt_analysis_audit(ctx, fitted, (symbol,))

    assert load_calls == [
        {
            "path": "synthetic-rli.csv",
            "phase_mode": "itt",
            "outcomes": (symbol,),
            "pre_required": (),
        }
    ]
    analysis = pd.read_csv(tmp_path / "analysis_set.csv")
    assert "outcome" not in analysis.columns
    expected_fitted = {
        "intervention": int(np.sum(group == 1)),
        "control": int(np.sum(group == 0)),
    }
    assert analysis.set_index("arm")["fitted_n"].to_dict() == expected_fitted
    bounds = pd.read_csv(tmp_path / "attrition_bounds.csv")
    assert bounds.loc[0, "outcome"] == symbol
    assert bounds.loc[0, "observed_intervention_n"] == 6
    assert bounds.loc[0, "observed_control_n"] == 6
    pd.testing.assert_frame_equal(ctx.tables["analysis_set"], analysis)
    pd.testing.assert_frame_equal(ctx.tables["attrition_bounds"], bounds)


def test_write_itt_analysis_audit_real_writer_maps_joint_outcomes(tmp_path, monkeypatch):
    group = np.array([1, 1, 0, 0])
    fitted = _prepared(
        {
            "W": np.array([1, 2, 3, 4], dtype=float),
            "L": np.array([4, np.nan, 6, 7], dtype=float),
        },
        {
            "W": np.linspace(-1.5, 1.5, 4),
            "L": np.linspace(-1.0, 1.0, 4),
        },
        group=group,
        n_trials={"W": 10, "L": 8},
    )
    archive_group = np.array([1] * 6 + [0] * 6)
    archive = _prepared(
        {
            "W": np.arange(12, dtype=float) % 10,
            "L": np.arange(12, dtype=float) % 8,
        },
        {
            "W": np.linspace(-1.5, 1.5, 12),
            "L": np.linspace(-1.0, 1.0, 12),
        },
        group=archive_group,
        n_trials={"W": 10, "L": 8},
    )
    load_calls = []
    monkeypatch.setattr(
        pipeline,
        "load_and_prepare",
        lambda **kwargs: load_calls.append(kwargs) or archive,
    )
    ctx = SimpleNamespace(output_dir=str(tmp_path), tables={})

    pipeline._write_itt_analysis_audit(ctx, fitted, ("W", "L"))

    assert [call["outcomes"] for call in load_calls] == [("W",), ("L",)]
    assert all(call["pre_required"] == () for call in load_calls)
    analysis = pd.read_csv(tmp_path / "analysis_set.csv")
    fitted_counts = analysis.set_index(["outcome", "arm"])["fitted_n"].to_dict()
    assert fitted_counts == {
        ("W", "intervention"): 2,
        ("W", "control"): 2,
        ("L", "intervention"): 1,
        ("L", "control"): 2,
    }
    bounds = pd.read_csv(tmp_path / "attrition_bounds.csv")
    assert bounds["outcome"].tolist() == ["W", "L"]


def _ppc_context(output_dir: Path, predictive: xr.Dataset, constant=None):
    trace = SimpleNamespace(posterior_predictive=predictive)
    if constant is not None:
        trace.constant_data = constant
    return SimpleNamespace(
        trace=trace,
        reporting=SimpleNamespace(ci_prob=0.95),
        output_dir=str(output_dir),
        tables={},
    )


def test_write_itt_ppc_calibration_real_writer_maps_single_outcome(tmp_path):
    group = np.array([1, 1, 1, 0, 0, 0])
    observed = np.array([0, 2, 4, 6, 8, 10], dtype=float)
    prepared = _prepared(
        {"W": observed},
        {"W": np.linspace(-2.0, 2.0, 6)},
        group=group,
        n_trials={"W": 10},
    )
    replicated = np.tile(observed, (2, 4, 1))
    predictive = xr.Dataset({"y_post": (("chain", "draw", "obs_id"), replicated)})
    ctx = _ppc_context(tmp_path, predictive)

    calibration = pipeline._write_itt_ppc_calibration(ctx, prepared, ("W",))

    assert (tmp_path / "posterior_predictive_calibration.csv").is_file()
    assert calibration["n"].sum() == prepared.n_obs
    assert set(calibration["outcome"]) == {"W"}
    assert not calibration.filter(like="outside_interval").to_numpy().any()
    pd.testing.assert_frame_equal(ctx.tables["posterior_predictive_calibration"], calibration)


def test_write_itt_ppc_calibration_real_writer_maps_floor_indicator(tmp_path):
    group = np.array([1, 1, 1, 0, 0, 0])
    post = np.array([0, 1, 2, 0, 0, 3], dtype=float)
    prepared = _prepared(
        {"P": post},
        {"P": np.full(6, -3.0)},
        group=group,
        n_trials={"P": 10},
    )
    observed_offfloor = (post > 0).astype(float)
    replicated = np.tile(observed_offfloor, (2, 4, 1))
    predictive = xr.Dataset({"y_offfloor": (("chain", "draw", "obs_id"), replicated)})
    ctx = _ppc_context(tmp_path, predictive)

    calibration = pipeline._write_itt_ppc_calibration(
        ctx,
        prepared,
        ("P",),
        node="y_offfloor",
        filename="posterior_predictive_calibration_offfloor.csv",
    )

    assert (tmp_path / "posterior_predictive_calibration_offfloor.csv").is_file()
    assert calibration["n"].sum() == prepared.n_obs
    weighted_mean = np.average(calibration["observed_mean_proportion"], weights=calibration["n"])
    assert weighted_mean == pytest.approx(float(observed_offfloor.mean()))
    assert set(calibration["baseline_band"]) == {"baseline_all"}
    assert not calibration.filter(like="outside_interval").to_numpy().any()


def test_write_itt_ppc_calibration_real_writer_maps_joint_cells(tmp_path):
    group = np.array([1, 1, 0, 0])
    prepared = _prepared(
        {
            "W": np.array([1, 2, 3, 4], dtype=float),
            "L": np.array([4, np.nan, 6, 7], dtype=float),
        },
        {
            "W": np.linspace(-1.5, 1.5, 4),
            "L": np.linspace(-1.0, 1.0, 4),
        },
        group=group,
        n_trials={"W": 10, "L": 8},
    )
    cell_rows = np.array([0, 0, 1, 2, 2, 3, 3])
    cell_outcomes = np.array([0, 1, 0, 0, 1, 0, 1])
    observed_cells = np.array([1, 4, 2, 3, 6, 4, 7], dtype=float)
    replicated = np.tile(observed_cells, (2, 4, 1))
    predictive = xr.Dataset({"y_post": (("chain", "draw", "cell"), replicated)})
    constant = xr.Dataset(
        {
            "y_post_cell_row": (("cell",), cell_rows),
            "y_post_cell_outcome": (("cell",), cell_outcomes),
        }
    )
    ctx = _ppc_context(tmp_path, predictive, constant)

    calibration = pipeline._write_itt_ppc_calibration(ctx, prepared, ("W", "L"))

    assert (tmp_path / "posterior_predictive_calibration.csv").is_file()
    assert (tmp_path / "posterior_predictive_shape_calibration.csv").is_file()
    assert calibration.groupby("outcome")["n"].sum().to_dict() == {"L": 3, "W": 4}
    assert not calibration.filter(like="outside_interval").to_numpy().any()
    shape = pd.read_csv(
        tmp_path / "posterior_predictive_shape_calibration.csv", keep_default_na=False
    )
    assert shape.set_index("outcome")["n"].to_dict() == {"L": 3, "W": 4}
    assert not shape["ppc_shape_flag"].any()
    pd.testing.assert_frame_equal(
        ctx.tables["posterior_predictive_shape_calibration"], shape
    )


def test_tau_summary_itt_names_probability_scale_direction_and_keeps_alias():
    """Coefficient and AME directions remain distinct under moderation."""

    n_draw, n_obs = 40, 3
    group = np.array([1.0, 0.0, 1.0])
    tau = np.full((1, n_draw), 0.2)
    interaction = np.full((1, n_draw), -1.0)
    moderator = np.ones(n_obs)
    delta = 0.2 - moderator
    eta = np.broadcast_to((delta * group)[None, None, :], (1, n_draw, n_obs)).copy()
    posterior = xr.Dataset(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "tau": (("chain", "draw"), tau),
            "gamma_tau_int": (("chain", "draw"), interaction),
        }
    )

    result = pipeline._report.tau_summary_itt(
        SimpleNamespace(posterior=posterior),
        ci_prob=0.95,
        G=group,
        moderators=(("gamma_tau_int", moderator),),
    )

    assert result["prob_tau_logit_pos"] == pytest.approx(1.0)
    assert result["prob_ame_pos"] == pytest.approx(0.0)
    assert result["prob_tau_pos"] == result["prob_ame_pos"]
    assert result["favoured_direction"] == "negative"


def test_fit_itt_ordinary_writes_headline_and_effective_spec_artifacts(fast_pipeline, monkeypatch):
    group = np.array([1] * 6 + [0] * 6)
    prepared = _prepared(
        {"W": np.arange(12) % 8},
        {"W": np.linspace(-2.0, 2.0, 12)},
        group=group,
        n_trials={"W": 79},
        covariates={"kept_adjuster": np.linspace(-1.0, 1.0, 12)},
    )
    load_calls = []
    build_calls = []
    monkeypatch.setattr(
        pipeline,
        "load_and_prepare",
        lambda **kwargs: load_calls.append(kwargs) or prepared,
    )

    def build(data, **kwargs):
        build_calls.append(kwargs)
        return BuiltModel(_FakeModel(), {}, data)

    monkeypatch.setattr(pipeline._factories, "build_itt_model", build)

    spec = ModelSpec(
        model_id="lrp-rli-itt-901",
        kind="itt",
        title="ordinary pipeline regression",
        outcome_symbol="W",
        model_settings=IttModelSettings(
            adjust_for=("kept_adjuster", "dropped_adjuster"),
            tau_sigma=0.35,
            alpha_sigma=1.25,
            gamma_own_sigma=0.45,
            kappa_sigma=0.80,
        ),
    )

    ctx = pipeline.fit_itt(spec, config="dev")
    out = Path(ctx.output_dir)

    assert load_calls == [
        {
            "phase_mode": "itt",
            "outcomes": ("W",),
            "covariates": ("kept_adjuster", "dropped_adjuster"),
            "restrict_complete": (),
            "drop_missing_pre": True,
            "pre_required": None,
        }
    ]
    assert build_calls[0]["adjust_for"] == ("kept_adjuster",)
    assert build_calls[0]["tau_sigma"] == pytest.approx(0.35)
    assert build_calls[0]["alpha_sigma"] == pytest.approx(1.25)
    assert build_calls[0]["gamma_own_sigma"] == pytest.approx(0.45)
    assert build_calls[0]["kappa_sigma"] == pytest.approx(0.80)

    for filename in (
        "trace.nc",
        "analysis_set.csv",
        "attrition_bounds.csv",
        "posterior_predictive_calibration.csv",
        "tau_summary.csv",
        "rope_summary.csv",
        "rope_sensitivity.csv",
        "model_recipe.md",
        "config.json",
    ):
        assert (out / filename).is_file(), filename

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["effective_model_settings"]["effective_adjustment"] == ["kept_adjuster"]
    assert cfg["effective_model_settings"]["floor_rule"] is False
    assert cfg["effective_model_settings"]["age_effect"] == "linear"
    assert cfg["effective_model_settings"]["baseline_terms"] == {
        "use_own_baseline": True,
        "use_own_baseline_gp": False,
        "cross_symbols": [],
        "pre_required": None,
    }
    assert cfg["spec_extra"] == {}
    assert cfg["model_settings"]["source"] == "typed"
    assert cfg["resolved_run_plan"]["adjust_for"] == [
        "kept_adjuster",
        "dropped_adjuster",
    ]
    assert cfg["model_recipe_file"] == "model_recipe.md"
    recipe = (out / "model_recipe.md").read_text()
    assert "supports the causal interpretation" in recipe
    assert "missing-outcome assumption" in recipe
    assert "does not make the other coefficients causal" in recipe
    assert "machine-readable form" in recipe
    assert cfg["extra"]["adjust_for"] == ["kept_adjuster"]
    assert cfg["extra"]["tau_summary"]["tau_prob_median"] == pytest.approx(0.08)


def test_fit_itt_rejects_an_invalid_plan_before_context_or_data(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline,
        "make_context",
        lambda *args, **kwargs: calls.append("context"),
    )
    monkeypatch.setattr(
        pipeline,
        "load_and_prepare",
        lambda *args, **kwargs: calls.append("data"),
    )
    spec = ModelSpec(
        model_id="lrp-rli-itt-903",
        kind="itt",
        title="invalid settings regression",
        outcome_symbol="W",
        model_settings=IttModelSettings(use_age_gp=True),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        pipeline.fit_itt(spec, config="dev")

    assert calls == []


def test_fit_itt_floor_rule_persists_missing_eligibility_and_secondary_audit(fast_pipeline, monkeypatch):
    from language_reading_predictors.statistical_models import floor as floor_module

    n_trials = 10
    floor_logit = np.log(0.5 / (n_trials + 0.5))
    above_floor_logit = np.log(2.5 / (n_trials - 2 + 0.5))
    pre_arm = np.array([floor_logit] * 5 + [above_floor_logit, np.nan], dtype=float)
    prepared = _prepared(
        {"P": np.array([0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float)},
        {"P": np.concatenate([pre_arm, pre_arm])},
        group=np.array([1] * 7 + [0] * 7),
        n_trials={"P": n_trials},
    )
    load_calls = []
    likelihoods = []
    monkeypatch.setattr(
        pipeline,
        "load_and_prepare",
        lambda **kwargs: load_calls.append(kwargs) or prepared,
    )
    transition_bounds = pd.DataFrame(
        [
            {
                "scope": "archived_dataset",
                "risk_difference_lower": -0.1,
                "risk_difference_upper": 0.3,
            },
            {
                "scope": "full_randomised_population",
                "risk_difference_lower": -0.4,
                "risk_difference_upper": 0.5,
            },
        ]
    )
    monkeypatch.setattr(
        floor_module,
        "binary_transition_missingness_bounds",
        lambda *args, **kwargs: transition_bounds,
    )

    def build(data, **kwargs):
        likelihood = kwargs.get("likelihood", "beta_binomial")
        likelihoods.append(likelihood)
        names = ("alpha", "tau", "gamma_A")
        if likelihood == "beta_binomial":
            names += ("kappa",)
        return BuiltModel(_FakeModel(names), {}, data)

    monkeypatch.setattr(pipeline._factories, "build_itt_model", build)

    spec = ModelSpec(
        model_id="lrp-rli-itt-902",
        kind="itt",
        title="floor pipeline regression",
        outcome_symbol="P",
        model_settings=IttModelSettings.for_floor_outcome(),
    )

    ctx = pipeline.fit_itt(spec, config="dev")
    out = Path(ctx.output_dir)

    assert load_calls == [
        {
            "phase_mode": "itt",
            "outcomes": ("P",),
            "covariates": (),
            "restrict_complete": (),
            "drop_missing_pre": True,
            "pre_required": (),
        }
    ]
    assert likelihoods == ["bernoulli_offfloor", "beta_binomial"]

    eligibility = pd.read_csv(out / "baseline_floor_eligibility.csv")
    assert eligibility["n_pre_missing"].tolist() == [1, 1]
    assert eligibility["n_exploratory_eligible"].tolist() == [5, 5]
    sensitivity = pd.read_csv(out / "floor_eligibility_sensitivity.csv")
    assert sensitivity.loc[0, "intervention_unknown_eligibility_n"] == 1
    assert sensitivity.loc[0, "control_unknown_eligibility_n"] == 1

    for filename in (
        "trace.nc",
        "trace_graded_secondary.nc",
        "tau_summary.csv",
        "tau_summary_graded.csv",
        "floor_transition_missingness_bounds.csv",
        "offfloor_movers.csv",
        "proportion_at_zero_ppc.csv",
        "model_recipe.md",
        "config.json",
    ):
        assert (out / filename).is_file(), filename
    assert not (out / "trace_hurdle_secondary.nc").exists()

    graded = pd.read_csv(out / "tau_summary_graded.csv").iloc[0]
    assert bool(graded["converged"])
    assert graded["trace_file"] == "trace_graded_secondary.nc"
    floor_ppc = pd.read_csv(out / "proportion_at_zero_ppc.csv").iloc[0]
    assert floor_ppc["ppc_upper_tail"] == pytest.approx(0.20)
    assert floor_ppc["ppc_lower_tail"] == pytest.approx(0.85)
    assert floor_ppc["ppc_two_sided_tail"] == pytest.approx(0.40)

    cfg = json.loads((out / "config.json").read_text())
    floor_meta = cfg["extra"]["floor_rule"]
    assert floor_meta["status"] == "post_hoc_data_adaptive"
    assert floor_meta["arm_blind_gate"] is True
    assert floor_meta["at_risk_n"] == 10
    assert floor_meta["total_n"] == 14
    assert floor_meta["baseline_missing_n"] == 2
    assert [row["n_pre_missing"] for row in floor_meta["eligibility_by_arm"]] == [
        1,
        1,
    ]
    assert floor_meta["eligibility_status_sensitivity"][0]["scale"] == ("off_floor_risk_difference")
    assert [row["scope"] for row in floor_meta["transition_missingness_bounds"]] == [
        "archived_dataset",
        "full_randomised_population",
    ]


def test_fit_joint_persists_probability_and_logit_contrasts_with_report_metadata(fast_pipeline, monkeypatch):
    from language_reading_predictors.statistical_models.lrp_rli_itt_015 import SPEC

    prepared = _prepared(
        {
            "TE": np.array([4, 5, 6, 3, 2, 4, 1, 2, 3, 2, 1, 3]),
            "UE": np.array([2, 3, 3, 2, 1, 2, 1, 1, 2, 1, 0, 2]),
        },
        {
            "TE": np.linspace(-1.5, 1.5, 12),
            "UE": np.linspace(-1.0, 1.0, 12),
        },
        group=np.array([1] * 6 + [0] * 6),
        n_trials={"TE": 12, "UE": 12},
    )
    load_calls = []
    build_calls = []
    marginal_calls = []
    contrast_scales = []
    difference_calls = []
    monkeypatch.setattr(
        pipeline,
        "load_and_prepare",
        lambda **kwargs: load_calls.append(kwargs) or prepared,
    )

    def build(data, **kwargs):
        build_calls.append(kwargs)
        return BuiltModel(
            _FakeModel(),
            {},
            data,
            extras={
                "joint_dependence": "factorised_outcome_marginals",
                "loo_unit": "child",
            },
        )

    monkeypatch.setattr(pipeline._factories, "build_joint_model", build)
    monkeypatch.setattr(
        pipeline._report,
        "tau_summary_joint",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "outcome": ["TE", "UE"],
                "ame_prob_median": [0.12, 0.04],
                "ame_prob_lo": [0.01, -0.03],
                "ame_prob_hi": [0.23, 0.11],
                "prob_ame_pos": [0.98, 0.84],
            }
        ),
    )

    def marginals(*args, **kwargs):
        marginal_calls.append(kwargs)
        return pd.DataFrame(
            {
                "outcome": ["TE", "UE"],
                "items_median": [1.4, 0.5],
                "items_lo": [0.1, -0.4],
                "items_hi": [2.7, 1.3],
                "prob_pos": [0.98, 0.84],
            }
        )

    monkeypatch.setattr(
        pipeline._report,
        "joint_treatment_marginals",
        marginals,
    )

    def contrast(*args, **kwargs):
        contrast_scales.append(kwargs["scale"])
        return pd.DataFrame(
            [[0.0, 0.08], [-0.08, 0.0]],
            index=["TE", "UE"],
            columns=["TE", "UE"],
        )

    monkeypatch.setattr(pipeline._report, "tau_contrast_matrix", contrast)

    def difference(*args, **kwargs):
        difference_calls.append(kwargs)
        metadata = kwargs["metadata"]
        return {
            "contrast": "TE - UE",
            "headline_scale": "proportion_correct_risk_difference",
            "diff_prob_median": 0.08,
            "diff_logit_median": 0.20,
            **metadata,
        }

    monkeypatch.setattr(pipeline._report, "tau_difference_summary", difference)

    ctx = pipeline.fit_joint(SPEC, config="dev")
    out = Path(ctx.output_dir)

    assert load_calls == [{"phase_mode": "itt", "outcomes": ("TE", "UE")}]
    assert build_calls[0]["outcomes"] == ("TE", "UE")
    assert build_calls[0]["use_cross_baselines"] is False
    assert build_calls[0]["use_residual_correlation"] is False
    assert marginal_calls[0]["outcomes"] == ["TE", "UE"]
    assert marginal_calls[0]["G"] is prepared.G
    assert marginal_calls[0]["n_trials"] is prepared.n_trials
    assert contrast_scales == ["probability", "logit"]
    assert difference_calls[0]["G"] is prepared.G
    assert difference_calls[0]["metadata"] == SPEC.extra["difference_metadata"]

    for filename in (
        "trace.nc",
        "tau_summary.csv",
        "joint_treatment_marginal.csv",
        "tau_contrast_matrix.csv",
        "tau_contrast_matrix_logit.csv",
        "tau_difference.csv",
        "config.json",
    ):
        assert (out / filename).is_file(), filename

    difference_df = pd.read_csv(out / "tau_difference.csv").iloc[0]
    assert difference_df["headline_scale"] == "proportion_correct_risk_difference"
    assert difference_df["contrast_kind"] == "generalisation"
    assert "does not by itself show" in difference_df["positive_interpretation"]
    assert difference_df["transfer_outcome"] == "UE"
    assert "does not estimate" in difference_df["dependence_note"]

    cfg = json.loads((out / "config.json").read_text())
    assert cfg["extra"]["joint_structure"] == "factorised_outcome_marginals"
    assert cfg["extra"]["loo_unit"] == "child"
    assert cfg["extra"]["tau_difference"]["headline_scale"] == ("proportion_correct_risk_difference")
    assert cfg["extra"]["difference_metadata"] == SPEC.extra["difference_metadata"]
