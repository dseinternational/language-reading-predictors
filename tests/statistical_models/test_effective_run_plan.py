# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Final metadata must retain the fitted recipe without changing the declaration."""

from dataclasses import replace
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models import run_metadata
from language_reading_predictors.statistical_models.adjusted import resolve_adjusted_run_plan
from language_reading_predictors.statistical_models.context import StatisticalFitContext, _reporting, _sampling
from language_reading_predictors.statistical_models.factories.adjusted import build_adjusted_model
from language_reading_predictors.statistical_models.family_registry import resolve_run_plan
from language_reading_predictors.statistical_models.lrp_rli_adj_065 import get_spec
from language_reading_predictors.statistical_models.preprocessing import load_and_prepare

from .test_prior_inventory import _write_synthetic


def _context(tmp_path, name, spec, plan):
    ctx = StatisticalFitContext(
        spec=spec,
        resolved_plan=plan,
        reporting=_reporting.ReportingConfiguration(
            model_name=spec.model_id, config_name="dev", output_root_dir=str(tmp_path / name), interval_kind="eti"
        ),
        sampling=_sampling.SamplingConfiguration(
            draws=100, tune=100, chains=2, cores=1, target_accept=0.9, random_seed=47
        ),
    )
    ctx.ensure_output_dir()
    return ctx


def test_dropped_adjuster_recipe_survives_metadata_and_reuse(tmp_path):
    path = _write_synthetic(tmp_path)
    frame = pd.read_csv(path)
    frame[V.ERBTO] = 1 + np.arange(len(frame)) % 17
    frame[V.DEAPP_C] = 50 + np.arange(len(frame)) % 31
    frame.to_csv(path, index=False)
    spec = get_spec()
    declared = resolve_adjusted_run_plan(spec)
    prepared = load_and_prepare(path=path, **declared.rli_prepare_kwargs())
    active = declared.with_active_covariates(tuple(c for c in declared.declared_covariates if c in prepared.covariates))
    assert "erbto_missing" in declared.active_covariates
    assert "erbto_missing" not in active.active_covariates
    built = build_adjusted_model(prepared, **active.rli_factory_kwargs())
    assert "beta_erbto_missing" not in built.model.named_vars
    source = _context(tmp_path, "source", spec, declared)
    current = _context(tmp_path, "current", spec, declared)
    for ctx in (source, current):
        ctx.model, ctx.prepared = built.model, built.prepared
        run_metadata.write_model_recipe(ctx, plan=active)
    output = Path(source.output_dir)
    # Compatibility checks inspect bytes before loading a posterior.
    (output / "trace.nc").write_bytes(b"Test checksum fixture; not a sampled posterior.\n")
    run_metadata.write_run_metadata(source)
    expected_recipe = active.recipe_markdown(title=spec.title)
    assert (output / "model_recipe.md").read_text(encoding="utf-8") == expected_recipe
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["resolved_run_plan"] == json.loads(json.dumps(declared.as_dict()))
    assert config["effective_model_settings"]["active_covariates"] == list(active.active_covariates)
    assert source.resolved_plan is declared
    run_metadata.require_reuse_compatibility(current, output)

    current.resolved_plan = replace(declared, predictor_slope_sigma=0.7)
    with pytest.raises(ValueError, match="resolved_run_plan"):
        run_metadata.require_reuse_compatibility(current, output)
    current.resolved_plan = declared
    recipe = Path(current.output_dir) / "model_recipe.md"
    recipe.write_text(expected_recipe + "Changed recipe.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model_recipe_sha256"):
        run_metadata.require_reuse_compatibility(current, output)


@pytest.mark.parametrize("model_id", ["jm_001", "jm_002", "med_059", "med_066", "med_092", "mm_001"])
def test_effective_plan_is_persisted_for_each_filtering_family(tmp_path, model_id):
    spec = importlib.import_module(f"language_reading_predictors.statistical_models.lrp_rli_{model_id}").SPEC
    declared = resolve_run_plan(spec)
    if spec.kind == "joint_mechanism":
        field = "active_adjustment"
        active = declared.with_active_adjustment(declared.active_adjustment[:-1])
    elif spec.kind == "corr_factor":
        field = "active_structural_covariates"
        active = declared.with_active_structural_covariates(declared.active_structural_covariates[:-1])
    else:
        field = "effective_confounders"
        active = declared.with_effective_confounders(declared.effective_confounders[:-1])
    assert active != declared
    ctx = _context(tmp_path, model_id, spec, declared)
    run_metadata.write_model_recipe(ctx, plan=active)
    run_metadata.write_run_metadata(ctx)
    output = Path(ctx.output_dir)
    assert (output / "model_recipe.md").read_text(encoding="utf-8") == active.recipe_markdown(title=spec.title)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["effective_model_settings"][field] == list(getattr(active, field))
    assert config["resolved_run_plan"] == json.loads(json.dumps(declared.as_dict()))
    assert config[run_metadata.REUSE_CONTRACT_KEY]["effective_model_settings"] == config["effective_model_settings"]
