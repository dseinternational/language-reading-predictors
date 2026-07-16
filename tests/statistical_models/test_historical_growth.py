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
import pymc as pm
import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.datasets import RLM_MEASURES
from language_reading_predictors.statistical_models.factories import (
    build_historical_growth_model,
)
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
    assert tuple(spec.extra["waves"]) == waves
    assert tuple(spec.extra["extension_waves"]) == extension_waves
    assert spec.extra["measure"] == measure
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
        extra={
            "outcomes": ("W",),
            "cross_symbols": (),
            "use_age_linear": True,
            "numpy_setting": np.int64(3),
            "numpy_array_setting": np.array([1, 2, 3]),
        },
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
    assert cfg["estimand_type"] == "causal_available_case_randomised_effect"
    assert cfg["spec_extra"]["outcomes"] == ["W"]
    assert cfg["spec_extra"]["numpy_setting"] == 3
    assert cfg["spec_extra"]["numpy_array_setting"] == [1, 2, 3]
    assert cfg["effective_model_settings"]["likelihood"] == "beta_binomial"
    assert cfg["effective_model_settings"]["effective_adjustment"] == ["age"]
    assert cfg["data_path"] == "/study/rli_data_long.csv"
    assert cfg["data_sha256"] == "abc123"
    counts = {row["arm"]: row for row in cfg["analysis_set_by_arm"]}
    assert counts["intervention"]["randomised_n"] == 29
    assert counts["intervention"]["fitted_n"] == 2
    assert counts["control"]["randomised_n"] == 28
    assert counts["control"]["fitted_n"] == 1
