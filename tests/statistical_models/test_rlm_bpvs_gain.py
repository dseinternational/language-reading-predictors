# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for the confirmed-input Byrne BPVS gain pair (#409 D1)."""

from language_reading_predictors.statistical_models.adjusted import (
    resolve_adjusted_run_plan,
)
from language_reading_predictors.statistical_models.horseshoe import (
    resolve_horseshoe_run_plan,
)
from language_reading_predictors.statistical_models.lrp_rlm_adj_003 import (
    SPEC as ADJUSTED_SPEC,
)
from language_reading_predictors.statistical_models.lrp_rlm_hs_002 import (
    SPEC as HORSESHOE_SPEC,
)


def test_bpvs_gain_pair_has_identical_confirmed_analysis_contract():
    adjusted = resolve_adjusted_run_plan(ADJUSTED_SPEC)
    horseshoe = resolve_horseshoe_run_plan(HORSESHOE_SPEC)

    assert adjusted.outcome_symbol == horseshoe.outcome_symbol == "bpvs"
    assert (
        adjusted.predictor_measures
        == horseshoe.predictor_measures
        == (
            "basread",
            "trog",
            "basdig",
            "bassim",
        )
    )
    assert adjusted.use_age_predictor is horseshoe.use_age_predictor is True
    assert (
        (adjusted.pre_wave, adjusted.post_wave)
        == (
            horseshoe.pre_wave,
            horseshoe.post_wave,
        )
        == (1, 3)
    )
    assert adjusted.group_codes is None
    assert adjusted.require_confirmed_inputs is True
    assert horseshoe.require_confirmed_inputs is True
    assert ADJUSTED_SPEC.causal_status == HORSESHOE_SPEC.causal_status == "none"
    assert ADJUSTED_SPEC.audit_baseline == HORSESHOE_SPEC.audit_baseline
