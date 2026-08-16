# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Confirmed-measure Byrne concurrent-family port (#409 C1)."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from language_reading_predictors.statistical_models.concurrent import (
    ConcurrentModelSettings,
    resolve_concurrent_run_plan,
)
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.factories import (
    build_rlm_concurrent_model,
)
from language_reading_predictors.statistical_models.lrp_rlm_ca_001 import (
    SPEC as RLM_CA_001,
)
from language_reading_predictors.statistical_models.lrp_rlm_ca_002 import (
    SPEC as RLM_CA_002,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_rlm_concurrent_frames,
)


def _rlm_spec(*, outcome: str, predictors: tuple[str, ...]) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rlm-ca-999",
        kind="concurrent",
        title="test",
        outcome_symbol=outcome,
        study_id="rlm",
        model_settings=ConcurrentModelSettings(
            predictor_symbols=predictors,
            waves=(1, 2, 3, 4),
        ),
    )


def test_registered_rlm_concurrent_specs_use_confirmed_five_measure_battery():
    core = {"basread", "bpvs", "trog", "basdig", "bassim"}
    for spec in (RLM_CA_001, RLM_CA_002):
        plan = resolve_concurrent_run_plan(spec)
        assert plan.port == "rlm"
        assert plan.study_id == "rlm"
        assert plan.waves == (1, 2, 3, 4)
        assert {plan.outcome_symbol, *plan.predictor_symbols} == core
        assert plan.causal_status.startswith("Associational")


@pytest.mark.parametrize("unresolved", ["basspel", "woco", "basnum"])
def test_rlm_concurrent_plan_rejects_provisional_measures(unresolved):
    spec = _rlm_spec(outcome="basread", predictors=("bpvs", unresolved))
    with pytest.raises(ValueError, match=rf"unresolved:.*{unresolved}"):
        resolve_concurrent_run_plan(spec)


def test_rlm_concurrent_plan_rejects_basmat_outside_its_source_window():
    spec = _rlm_spec(outcome="basread", predictors=("bpvs", "basmat"))
    with pytest.raises(ValueError, match=r"basmat is not available.*1, 2"):
        resolve_concurrent_run_plan(spec)


def test_rlm_concurrent_plan_rejects_rli_only_covariates():
    spec = ModelSpec(
        model_id="lrp-rlm-ca-999",
        kind="concurrent",
        title="test",
        outcome_symbol="basread",
        study_id="rlm",
        model_settings=ConcurrentModelSettings(
            predictor_symbols=("bpvs",), covariates=("blocks",)
        ),
    )
    with pytest.raises(ValueError, match="does not support RLI trait covariates"):
        resolve_concurrent_run_plan(spec)


def test_rlm_concurrent_plan_rejects_empty_predictor_set():
    spec = _rlm_spec(outcome="basread", predictors=())
    with pytest.raises(ValueError, match="predictor_symbols cannot be empty"):
        resolve_concurrent_run_plan(spec)


def test_live_rlm_concurrent_frames_are_outcome_complete_and_predictor_available_case():
    frames = load_rlm_concurrent_frames(
        outcome="basread",
        predictor_measures=("bpvs", "trog", "basdig", "bassim"),
    )
    assert {wave: frame.n_obs for wave, frame in frames.items()} == {
        1: 96,
        2: 88,
        3: 78,
        4: 61,
    }
    for wave, frame in frames.items():
        assert np.isfinite(frame.post_counts["basread"]).all(), wave
        assert frame.n_obs == frame.n_children
        assert set(frame.group_code) == {1, 2, 3}
    assert np.isfinite(frames[1].post_counts["bpvs"]).sum() == 87
    assert np.isnan(frames[1].post_counts["bpvs"]).sum() == 9


def test_rlm_concurrent_factory_builds_mutual_and_single_skill_models():
    frame = load_rlm_concurrent_frames(
        outcome="basread",
        predictor_measures=("bpvs", "trog", "basdig", "bassim"),
        waves=(3,),
    )[3]
    mutual = build_rlm_concurrent_model(
        frame,
        predictor_symbols=("bpvs", "trog", "basdig", "bassim"),
    )
    names = {rv.name for rv in mutual.model.free_RVs}
    assert {"beta_bpvs", "beta_trog", "beta_basdig", "beta_bassim"} <= names
    assert "beta_age" in names
    assert len({name for name in names if name.startswith("beta_group_nuisance_")}) == 2
    assert "gamma_own" not in names
    with mutual.model:
        prior = pm.sample_prior_predictive(draws=3, random_seed=17)
    assert prior.prior_predictive["y_post"].shape[-1] == frame.n_obs

    single = build_rlm_concurrent_model(
        frame,
        predictor_symbols=("bpvs",),
        include_age=False,
        include_group=False,
    )
    single_names = {rv.name for rv in single.model.free_RVs}
    assert {name for name in single_names if name.startswith("beta_")} == {
        "beta_bpvs"
    }
