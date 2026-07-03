# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the historical group-by-wave growth model (RLMHG, #165).

Smoke tests (build + tiny prior predictive) on a synthetic Byrne-shaped panel,
plus a check that the new ModelSpec dataset/estimand metadata reaches config.json.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

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


def _panel(tmp_path):
    path = _write_synthetic(tmp_path)
    return load_longitudinal_panel(
        _dataset(path), [RLM_MEASURES["basread"]], waves=(1, 2, 3)
    )


def test_build_historical_growth_model(tmp_path):
    panel = _panel(tmp_path)
    built = build_historical_growth_model(panel, measure="basread")

    names = {v.name for v in built.model.free_RVs}
    assert {"eta_group_wave", "sigma_subject", "z_subject", "kappa"}.issubset(names)
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

    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    assert pp.prior_predictive["score"].shape[-1] == panel.n_obs


def test_build_rejects_unknown_measure(tmp_path):
    panel = _panel(tmp_path)
    with pytest.raises(KeyError, match="not in panel"):
        build_historical_growth_model(panel, measure="bpvs")


def test_dataset_metadata_reaches_config_json(tmp_path):
    """The new ModelSpec dataset/estimand fields round-trip to config.json (#165)."""
    spec = ModelSpec(
        model_id="rlmhg01",
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
        reporting=SimpleNamespace(output_dir=str(tmp_path), hdi=0.94),
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


def test_existing_spec_defaults_are_rli(tmp_path):
    """An intervention-style spec (no dataset metadata) defaults to study rli."""
    spec = ModelSpec(model_id="lrpitt10", kind="itt", title="t", outcome_symbol="W")
    ctx = SimpleNamespace(
        spec=spec,
        prepared=SimpleNamespace(n_obs=10, n_children=5, n_phases=1, dropped_rows=0),
        reporting=SimpleNamespace(output_dir=str(tmp_path), hdi=0.95),
        sampling=SimpleNamespace(
            draws=1, tune=1, chains=1, target_accept=0.9, random_seed=47
        ),
        output_dir=str(tmp_path),
    )
    write_run_metadata(ctx)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["study_id"] == "rli"
    assert cfg["family"] is None and cfg["causal_status"] is None
