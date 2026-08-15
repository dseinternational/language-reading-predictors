# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contracts for the remaining confirmed-input Byrne gain models (#409 D1)."""

from language_reading_predictors.statistical_models.adjusted import (
    resolve_adjusted_run_plan,
)
from language_reading_predictors.statistical_models.horseshoe import (
    resolve_horseshoe_run_plan,
)
from language_reading_predictors.statistical_models.lrp_rlm_adj_004 import (
    SPEC as TROG_ADJUSTED_SPEC,
)
from language_reading_predictors.statistical_models.lrp_rlm_adj_005 import (
    SPEC as BASDIG_ADJUSTED_SPEC,
)
from language_reading_predictors.statistical_models.lrp_rlm_hs_003 import (
    SPEC as TROG_HORSESHOE_SPEC,
)


def test_trog_gain_pair_has_identical_confirmed_analysis_contract():
    adjusted = resolve_adjusted_run_plan(TROG_ADJUSTED_SPEC)
    horseshoe = resolve_horseshoe_run_plan(TROG_HORSESHOE_SPEC)

    assert adjusted.outcome_symbol == horseshoe.outcome_symbol == "trog"
    assert (
        adjusted.predictor_measures
        == horseshoe.predictor_measures
        == ("basread", "bpvs", "basdig", "bassim")
    )
    assert adjusted.use_age_predictor is horseshoe.use_age_predictor is True
    assert (
        (adjusted.pre_wave, adjusted.post_wave)
        == (horseshoe.pre_wave, horseshoe.post_wave)
        == (1, 3)
    )
    assert adjusted.group_codes is None
    assert adjusted.require_confirmed_inputs is True
    assert horseshoe.require_confirmed_inputs is True
    assert (
        TROG_ADJUSTED_SPEC.causal_status
        == TROG_HORSESHOE_SPEC.causal_status
        == "none"
    )
    assert TROG_ADJUSTED_SPEC.audit_baseline == TROG_HORSESHOE_SPEC.audit_baseline


def test_basdig_gain_has_confirmed_adjusted_analysis_contract():
    adjusted = resolve_adjusted_run_plan(BASDIG_ADJUSTED_SPEC)

    assert adjusted.outcome_symbol == "basdig"
    assert adjusted.predictor_measures == ("basread", "bpvs", "trog", "bassim")
    assert adjusted.use_age_predictor is True
    assert (adjusted.pre_wave, adjusted.post_wave) == (1, 3)
    assert adjusted.group_codes is None
    assert adjusted.require_confirmed_inputs is True
    assert BASDIG_ADJUSTED_SPEC.causal_status == "none"
    assert BASDIG_ADJUSTED_SPEC.audit_baseline == "complete_case_summary"
