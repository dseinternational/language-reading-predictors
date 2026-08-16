# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the historical group-by-wave growth model (RLMHG, #165).

Smoke tests (build + tiny prior predictive) on a synthetic Byrne-shaped panel,
plus a check that the new ModelSpec dataset/estimand metadata reaches config.json.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from language_reading_predictors.statistical_models import diagnostics as _diagnostics
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.datasets import RLM_MEASURES
from language_reading_predictors.statistical_models.factories import (
    build_historical_growth_model,
)
from language_reading_predictors.statistical_models.historical_growth import (
    HistoricalGrowthModelSettings,
    evaluate_historical_growth_influence_bundle,
    exclude_historical_growth_observations,
    historical_growth_influence_summary,
    historical_growth_pareto_table,
    resolve_historical_growth_run_plan,
)
from language_reading_predictors.statistical_models.sensitivity import sha256_file
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
)
from language_reading_predictors.statistical_models.reporting import write_run_metadata

from .test_datasets import _dataset, _write_synthetic


def _panel(tmp_path, *, extension=False, extension_waves=()):
    path = _write_synthetic(tmp_path, extension=extension)
    return load_longitudinal_panel(
        _dataset(path),
        [RLM_MEASURES["basread"]],
        waves=(1, 2, 3),
        extension_waves=extension_waves,
    )


def test_build_historical_growth_model(tmp_path):
    panel = _panel(tmp_path)
    built = build_historical_growth_model(panel, measure="basread")

    names = {v.name for v in built.model.free_RVs}
    assert {"eta_cell", "sigma_subject", "z_subject", "kappa"}.issubset(names)
    dets = {v.name for v in built.model.deterministics}
    assert {
        "subject_offset",
        "mean_items",
        "growth_first_next_items",
        "growth_next_last_items",
        "growth_first_last_items",
    }.issubset(dets)
    assert "score" in {v.name for v in built.model.observed_RVs}
    assert built.prepared is panel  # panel carried through for the summaries
    # #338: the random-effect scales are indexed by group.
    assert built.model.named_vars["sigma_subject"].eval().shape == (3,)
    assert built.model.named_vars["kappa"].eval().shape == (3,)
    # Rectangular panel -> one eta per (group, wave) cell.
    assert built.model.named_vars["eta_cell"].eval().shape == (9,)

    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    assert pp.prior_predictive["score"].shape[-1] == panel.n_obs


def test_build_historical_growth_model_ragged_extension(tmp_path):
    # #338: extension waves add only *supported* cells - the group-1-only wave 5
    # contributes one eta cell, not a prior-only row for every group.
    panel = _panel(tmp_path, extension=True, extension_waves=(4, 5))
    built = build_historical_growth_model(panel, measure="basread")

    cells = panel.cells("basread")
    assert built.model.named_vars["eta_cell"].eval().shape == (len(cells),)
    assert (1, 5) in cells and (2, 5) not in cells
    # Growth deterministics span the common (all-group) window: waves 1-4.
    coords = built.model.coords
    assert len(coords["cell"]) == len(cells)
    dets = {v.name for v in built.model.deterministics}
    assert {
        "growth_first_next_items",
        "growth_next_last_items",
        "growth_first_last_items",
    }.issubset(dets)

    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    assert pp.prior_predictive["score"].shape[-1] == panel.n_obs


def test_build_rejects_measure_absent_from_panel(tmp_path):
    # The factory rejects a measure that was not loaded into the panel, whether or
    # not it is registered in RLM_MEASURES. ``bpvs`` is now a registered Phase-A
    # measure but this panel loads only ``basread``, so it is absent here.
    panel = _panel(tmp_path)
    with pytest.raises(KeyError, match="not in panel"):
        build_historical_growth_model(panel, measure="bpvs")


def test_historical_growth_pareto_maps_likelihood_rows(tmp_path):
    panel = _panel(tmp_path, extension=True, extension_waves=(4, 5))
    values = np.linspace(0.1, 0.9, panel.n_obs)
    loo = SimpleNamespace(pareto_k=values, good_k=0.7)

    result = historical_growth_pareto_table(panel, loo, measure="basread")

    worst = result.iloc[0]
    source = panel.long.iloc[int(worst["observation_index"])]
    assert worst["pareto_k"] == pytest.approx(0.9)
    assert worst["subject_id"] == source[panel.dataset.subject_col]
    assert worst["wave"] == source[panel.dataset.wave_col]
    assert worst["group_code"] == source[panel.dataset.group_col]
    assert worst["score"] == source["basread"]
    assert result["loo_reliable"].eq(result["pareto_k"] <= 0.7).all()


def test_shared_influence_maps_historical_growth_rows(tmp_path):
    panel = _panel(tmp_path)
    values = np.linspace(0.1, 0.9, panel.n_obs)
    context = SimpleNamespace(
        loo=SimpleNamespace(pareto_k=values, good_k=0.7),
        prepared=panel,
    )

    result, threshold, n_flagged = _diagnostics.influence_diagnostics(context)

    assert threshold == 0.7
    assert n_flagged == int((values > 0.7).sum())
    assert len(result) == panel.n_obs
    worst = result.iloc[0]
    assert worst["subject_id"] == panel.long.iloc[-1][panel.dataset.subject_col]


def test_exclude_historical_growth_rows_rebuilds_panel(tmp_path):
    panel = _panel(tmp_path)
    subject_col = panel.dataset.subject_col
    wave_col = panel.dataset.wave_col
    first_subject = panel.subject_ids[0]
    first_rows = panel.long.index[panel.long[subject_col] == first_subject].to_numpy()

    one_row_out = exclude_historical_growth_observations(panel, first_rows[:1])
    assert one_row_out.n_obs == panel.n_obs - 1
    assert one_row_out.n_subjects == panel.n_subjects
    subject_position = one_row_out.subject_ids.index(first_subject)
    wave_position = panel.waves.index(
        int(panel.long.iloc[int(first_rows[0])][wave_col])
    )
    assert np.isnan(one_row_out.counts["basread"][subject_position, wave_position])
    assert not one_row_out.obs_mask["basread"][subject_position, wave_position]

    child_out = exclude_historical_growth_observations(panel, first_rows)
    assert child_out.n_obs == panel.n_obs - len(first_rows)
    assert child_out.n_subjects == panel.n_subjects - 1
    assert first_subject not in child_out.subject_ids
    assert child_out.dropped_subjects == panel.dropped_subjects + 1


def test_historical_growth_influence_summary_compares_separate_fits(tmp_path):
    panel = _panel(tmp_path)
    excluded_index = np.array([0])
    sensitivity_panel = exclude_historical_growth_observations(
        panel, excluded_index
    )

    primary_built = build_historical_growth_model(panel, measure="basread")
    with primary_built.model:
        primary_prior = pm.sample_prior_predictive(draws=20, random_seed=11)
    sensitivity_built = build_historical_growth_model(
        sensitivity_panel, measure="basread"
    )
    with sensitivity_built.model:
        sensitivity_prior = pm.sample_prior_predictive(draws=20, random_seed=12)
    primary_trace = SimpleNamespace(posterior=primary_prior.prior)
    sensitivity_trace = SimpleNamespace(posterior=sensitivity_prior.prior)
    excluded = historical_growth_pareto_table(
        panel,
        SimpleNamespace(
            pareto_k=np.r_[0.8, np.repeat(0.1, panel.n_obs - 1)],
            good_k=0.7,
        ),
        measure="basread",
    ).query("loo_reliable == False")

    result = historical_growth_influence_summary(
        primary_trace,
        sensitivity_trace,
        primary_panel=panel,
        sensitivity_panel=sensitivity_panel,
        measure="basread",
        excluded_rows=excluded,
        sensitivity_converged=True,
    )

    assert not result.empty
    assert result["n_excluded_rows"].eq(1).all()
    assert result["n_excluded_children"].eq(1).all()
    assert result["n_fully_excluded_children"].eq(0).all()
    assert result["max_excluded_pareto_k"].eq(0.8).all()
    assert result["sensitivity_converged"].all()
    assert np.allclose(
        result["median_shift"],
        result["sensitivity_q50"] - result["primary_q50"],
    )


def test_historical_growth_influence_bundle_is_hash_bound(tmp_path):
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "trace.nc").write_text("primary trace\n", encoding="utf-8")
    pareto = pd.DataFrame(
        {
            "observation_index": [0, 1],
            "subject_id": ["child-a", "child-a"],
            "pareto_k": [0.8, 0.2],
            "good_k_threshold": [0.7, 0.7],
        }
    )
    pareto.to_csv(tmp_path / "pareto_k.csv", index=False)
    trace_name = "trace_historical_growth_influence_sensitivity.nc"
    (tmp_path / trace_name).write_text("sensitivity trace\n", encoding="utf-8")
    summary = pd.DataFrame(
        {
            "model_id": ["lrp-rlm-hg-009"],
            "config": ["reporting"],
            "median_shift": [0.34],
            "median_direction_stable": [True],
            "intervals_overlap": [True],
            "n_excluded_rows": [1],
            "max_excluded_pareto_k": [0.8],
            "sensitivity_converged": [True],
            "primary_config_sha256": [sha256_file(tmp_path / "config.json")],
            "primary_trace_sha256": [sha256_file(tmp_path / "trace.nc")],
            "primary_pareto_k_sha256": [sha256_file(tmp_path / "pareto_k.csv")],
            "sensitivity_trace_file": [trace_name],
            "sensitivity_trace_sha256": [sha256_file(tmp_path / trace_name)],
        }
    )
    summary_path = tmp_path / "historical_growth_influence_sensitivity.csv"
    summary.to_csv(summary_path, index=False)
    provenance = {
        "status": "completed",
        "model_id": "lrp-rlm-hg-009",
        "config": "reporting",
        "primary_config_sha256": sha256_file(tmp_path / "config.json"),
        "primary_trace_sha256": sha256_file(tmp_path / "trace.nc"),
        "primary_pareto_k_sha256": sha256_file(tmp_path / "pareto_k.csv"),
        "flagged_observation_indices": [0],
        "sensitivity_trace_sha256": sha256_file(tmp_path / trace_name),
        "sensitivity_summary_sha256": sha256_file(summary_path),
        "convergence": {"converged": True},
    }
    (tmp_path / "historical_growth_influence_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    report_config = {
        "model_id": "lrp-rlm-hg-009",
        "kind": "historical_growth",
        "n_obs": 2,
    }

    valid = evaluate_historical_growth_influence_bundle(
        summary, tmp_path, report_config, "reporting"
    )
    assert valid["ready"] is True
    assert valid["max_median_shift"] == pytest.approx(0.34)

    (tmp_path / trace_name).write_text("tampered\n", encoding="utf-8")
    invalid = evaluate_historical_growth_influence_bundle(
        summary, tmp_path, report_config, "reporting"
    )
    assert invalid["ready"] is False
    assert "hash-mismatched" in invalid["reason"]


def test_dataset_metadata_reaches_config_json(tmp_path):
    """The new ModelSpec dataset/estimand fields round-trip to config.json (#165)."""
    spec = ModelSpec(
        model_id="lrp-rlm-hg-001",
        kind="historical_growth",
        title="t",
        outcome_symbol="basread",
        study_id="rlm",
        family="historical_growth",
        design="historical_cohort",
        estimand_type="descriptive",
        causal_status="none",
        dataset_ref="rlm:reading_language_memory_data_long",
        audit_baseline="table2_complete_case_summary",
    )
    ctx = SimpleNamespace(
        spec=spec,
        prepared=SimpleNamespace(n_obs=27, n_children=9, n_phases=2, dropped_rows=0),
        reporting=SimpleNamespace(output_dir=str(tmp_path), ci_prob=0.94),
        sampling=SimpleNamespace(
            draws=1, tune=1, chains=1, target_accept=0.9, random_seed=47
        ),
        output_dir=str(tmp_path),
    )
    write_run_metadata(ctx, extra={"measure": "basread"})

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["study_id"] == "rlm"
    assert cfg["family"] == "historical_growth"
    assert cfg["design"] == "historical_cohort"
    assert cfg["estimand_type"] == "descriptive"
    assert cfg["causal_status"] == "none"
    assert cfg["dataset_ref"] == "rlm:reading_language_memory_data_long"
    assert cfg["audit_baseline"] == "table2_complete_case_summary"
    contract = cfg["publication_input_contract"]
    assert contract["study_id"] == "rlm"
    assert contract["publication_ready"] is True
    assert set(contract["measures"]) == {"basread"}
    assert contract["blockers"] == []
    assert contract["dataset"]["source_provenance_manifest"].endswith(
        "source_provenance.json"
    )


def test_non_itt_typed_settings_reach_config_json(tmp_path):
    """Historical growth records its typed settings and resolved plan, not legacy ``extra``."""

    spec = ModelSpec(
        model_id="lrp-rlm-hg-999",
        kind="historical_growth",
        title="typed metadata test",
        outcome_symbol="basread",
        study_id="rlm",
        model_settings=HistoricalGrowthModelSettings(
            measure="basread", waves=(1, 2, 3)
        ),
    )
    ctx = SimpleNamespace(
        spec=spec,
        prepared=SimpleNamespace(n_obs=27, n_children=9, n_phases=2, dropped_rows=0),
        reporting=SimpleNamespace(output_dir=str(tmp_path), ci_prob=0.89),
        sampling=SimpleNamespace(
            draws=1, tune=1, chains=1, target_accept=0.9, random_seed=47
        ),
        output_dir=str(tmp_path),
    )

    write_run_metadata(ctx)

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["model_settings"] == {
        "source": "typed",
        "measure": "basread",
        "waves": [1, 2, 3],
        "extension_waves": [],
        "eta_prior_sigma": 1.5,
        "sigma_subject_prior_sigma": 1.0,
        "kappa_prior_sigma": 50.0,
    }
    assert cfg["spec_extra"] == {}
    assert cfg["resolved_run_plan"]["settings_source"] == "typed"
    assert cfg["resolved_run_plan"]["measure"] == "basread"


def test_rlm_input_contract_includes_predictors_not_only_the_outcome(tmp_path):
    spec = ModelSpec(
        model_id="lrp-rlm-adj-001",
        kind="adjusted",
        title="input contract fixture",
        outcome_symbol="basread",
        study_id="rlm",
        extra={"predictor_measures": ("bpvs", "basnum")},
    )
    prepared = SimpleNamespace(
        n_obs=20,
        n_children=20,
        n_phases=1,
        dropped_rows=0,
        outcome="basread",
        n_trials={"basread": 90},
        predictors={"bpvs": np.zeros(20), "basnum": np.zeros(20), "age": np.zeros(20)},
    )
    ctx = SimpleNamespace(
        spec=spec,
        prepared=prepared,
        reporting=SimpleNamespace(output_dir=str(tmp_path), ci_prob=0.89),
        sampling=SimpleNamespace(
            draws=1, tune=1, chains=1, target_accept=0.9, random_seed=47
        ),
        output_dir=str(tmp_path),
    )

    write_run_metadata(ctx)

    contract = json.loads((tmp_path / "config.json").read_text())[
        "publication_input_contract"
    ]
    assert set(contract["measures"]) == {"basread", "bpvs", "basnum"}
    assert "age" not in contract["measures"]
    assert any("basnum" in blocker for blocker in contract["blockers"])


# --- #164 Phase A models (lrp-rlm-hg-001..009), with their #338 wave windows:
# (measure, complete-case core waves, extension waves).
_PHASE_A_MODELS = {
    "lrp-rlm-hg-001": ("basread", (1, 2, 3), (4, 5)),
    "lrp-rlm-hg-002": ("basspel", (1, 2, 3), (4, 5)),
    "lrp-rlm-hg-003": ("woco", (1, 2, 3), (4, 5)),
    "lrp-rlm-hg-004": ("bpvs", (1, 2, 3), (4, 5)),
    "lrp-rlm-hg-005": ("trog", (1, 2, 3), (4, 5)),
    "lrp-rlm-hg-006": ("basdig", (1, 2, 3), (4, 5)),
    "lrp-rlm-hg-007": ("bassim", (1, 2, 3), (4, 5)),
    # basnum was not assessed at wave 5; basmat is wave-3+ only (own core).
    "lrp-rlm-hg-008": ("basnum", (1, 2, 3), (4,)),
    "lrp-rlm-hg-009": ("basmat", (3, 4), (5,)),
}


@pytest.mark.parametrize(
    "model_id, measure, waves, extension_waves",
    [(mid, *cfg) for mid, cfg in sorted(_PHASE_A_MODELS.items())],
)
def test_phase_a_specs_well_formed(model_id, measure, waves, extension_waves):
    """Each Phase-A hg model is discoverable and carries the right descriptive metadata."""
    from language_reading_predictors.statistical_models.datasets import resolve_dataset
    from language_reading_predictors.statistical_models.registry import discover_models

    models = discover_models()
    assert model_id in models, f"{model_id} not auto-discovered"
    spec = models[model_id].SPEC
    assert spec.model_id == model_id
    assert spec.kind == "historical_growth"
    assert spec.study_id == "rlm"
    assert spec.outcome_symbol == measure
    # Descriptive, non-causal: readgrp is a cohort factor, never a treatment.
    assert spec.estimand_type == "descriptive"
    assert spec.causal_status == "none"
    assert isinstance(spec.model_settings, HistoricalGrowthModelSettings)
    assert spec.extra == {}
    plan = resolve_historical_growth_run_plan(spec)
    assert plan.waves == waves
    assert plan.extension_waves == extension_waves
    assert plan.measure == measure
    # The measure the spec names must be registered for the study.
    _dataset_spec, measures = resolve_dataset("rlm")
    assert measure in measures


def test_itt_spec_defaults_and_effective_settings_reach_config_json(tmp_path):
    """ITT metadata records requested/effective settings and source provenance."""
    spec = ModelSpec(
        model_id="lrp-rli-itt-010",
        kind="itt",
        title="t",
        outcome_symbol="W",
        model_settings=IttModelSettings(adjust_for=("age",)),
    )
    ctx = SimpleNamespace(
        spec=spec,
        prepared=SimpleNamespace(
            n_obs=4,
            n_children=4,
            n_phases=1,
            dropped_rows=0,
            G=np.array([1, 1, 0, 0]),
            post_counts={"W": np.array([1.0, 2.0, 3.0, np.nan])},
            n_trials={"W": 79},
            covariates={"age": np.arange(4.0)},
            covariate_time={"age": "pre"},
            dropped_covariates=("constant",),
            phase_mode="itt",
            data_path="/study/rli_data_long.csv",
            data_sha256="abc123",
        ),
        reporting=SimpleNamespace(output_dir=str(tmp_path), ci_prob=0.95),
        sampling=SimpleNamespace(
            draws=1, tune=1, chains=1, target_accept=0.9, random_seed=47
        ),
        output_dir=str(tmp_path),
    )
    write_run_metadata(ctx)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["study_id"] == "rli"
    assert cfg["family"] == "itt"
    assert cfg["estimand_type"] == "available_case_modified_itt_estimate"
    assert cfg["spec_extra"] == {}
    assert cfg["model_settings"]["source"] == "typed"
    assert cfg["model_settings"]["adjust_for"] == ["age"]
    assert cfg["resolved_run_plan"]["outcomes"] == ["W"]
    assert cfg["model_recipe_file"] == "model_recipe.md"
    assert (tmp_path / "model_recipe.md").is_file()
    assert cfg["effective_model_settings"]["likelihood"] == "beta_binomial"
    assert cfg["effective_model_settings"]["effective_adjustment"] == ["age"]
    assert cfg["data_path"] == "/study/rli_data_long.csv"
    assert cfg["data_sha256"] == "abc123"
    assert set(cfg["provenance"]) == {
        "recorded_at_utc",
        "invocation",
        "source",
        "runtime",
        "packages",
    }
    assert cfg["provenance"]["runtime"]["python_version"]
    assert "pymc" in cfg["provenance"]["packages"]
    counts = {row["arm"]: row for row in cfg["analysis_set_by_arm"]}
    assert counts["intervention"]["randomised_n"] == 29
    assert counts["intervention"]["fitted_n"] == 2
    assert counts["control"]["randomised_n"] == 28
    assert counts["control"]["fitted_n"] == 1
