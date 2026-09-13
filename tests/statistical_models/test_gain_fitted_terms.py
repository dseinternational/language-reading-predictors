# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gain association summaries must use the design that reached the likelihood."""

import importlib

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.factories.gain_factors import build_gain_factors_model
from language_reading_predictors.statistical_models.gain_factors import resolve_gain_factors_run_plan
from language_reading_predictors.statistical_models.pipelines.gain_factors import _gf_association_terms
from language_reading_predictors.statistical_models.preprocessing import load_and_prepare

from .test_prior_inventory import _write_synthetic


@pytest.mark.parametrize("number", ["001", "101", "201", "205", "306"])
def test_summary_scales_and_interactions_match_the_final_model(tmp_path, number):
    spec = importlib.import_module(f"language_reading_predictors.statistical_models.lrp_rli_gf_{number}").SPEC
    plan = resolve_gain_factors_run_plan(spec)
    path = _write_synthetic(tmp_path)
    frame = pd.read_csv(path)
    frame[V.ERBTO] = 1 + np.arange(len(frame)) % 17
    frame[V.DEAPP_C] = 50 + np.arange(len(frame)) % 31
    frame.to_csv(path, index=False)
    prepared = load_and_prepare(path=path, **plan.prepare_kwargs())
    own = plan.outcome_symbol
    prepared.post_counts[own] = prepared.post_counts[own].astype(float)
    prepared.post_counts[own][0] = np.nan
    adjust = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    built = build_gain_factors_model(prepared, **plan.factory_kwargs(effective_adjustment=adjust))
    assert built.prepared.n_obs < prepared.n_obs
    payload = built.payload
    terms = {term.label: term for term in _gf_association_terms(plan, built)}
    assert all(term.coef in built.model.named_vars for term in terms.values())

    np.testing.assert_array_equal(payload.term_vectors["age"], built.model["A_std"].get_value())
    assert terms["age"].main_scale == 1.0
    if plan.ability_covariate:
        np.testing.assert_array_equal(
            payload.term_vectors["ability"], built.model[f"{plan.ability_covariate}_std"].get_value()
        )
        assert terms["ability"].main_scale == 1.0
    if plan.off_floor:
        np.testing.assert_array_equal(terms["own"].toggle_vector, built.model["own_pre_offfloor"].get_value())
        assert terms["own"].main_scale == 1.0
    else:
        assert terms["own"].main_scale == pytest.approx(np.std(built.model["own_pre_logit"].get_value(), ddof=1))
    for symbol in plan.skill_symbols:
        assert terms[symbol].main_scale == pytest.approx(np.std(built.model[f"{symbol}_pre_logit"].get_value(), ddof=1))
    for left, right in payload.active_interactions:
        np.testing.assert_array_equal(
            built.model[f"int_{left}_{right}"].get_value(), payload.term_vectors[left] * payload.term_vectors[right]
        )
        for focal, partner in ((left, right), (right, left)):
            if focal in terms:
                partners = dict(terms[focal].interactions)
                np.testing.assert_array_equal(partners[f"gamma_int_{left}_{right}"], payload.term_vectors[partner])
    if plan.treated_only:
        assert all("trt" not in pair for pair in payload.active_interactions)
        assert all(not name.startswith("gamma_int_trt_") for term in terms.values() for name, _ in term.interactions)
