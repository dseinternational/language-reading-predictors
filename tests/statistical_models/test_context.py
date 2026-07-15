# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from language_reading_predictors.statistical_models.context import ModelSpec


def test_itt_spec_records_available_case_randomised_design_by_default():
    spec = ModelSpec(
        model_id="lrp-rli-itt-010",
        kind="itt",
        title="Word reading",
        outcome_symbol="W",
    )

    assert spec.family == "itt"
    assert spec.design == "waitlist_randomised_t1_to_t2_available_case"
    assert spec.estimand_type == "causal_available_case_randomised_effect"
    assert "observed_analysis_set" in spec.causal_status
    assert "54 available of 57 randomised" in spec.dataset_ref


def test_explicit_itt_audit_metadata_is_not_overwritten():
    spec = ModelSpec(
        model_id="lrp-rli-itt-999",
        kind="itt",
        title="Custom",
        design="custom-design",
        estimand_type="custom-estimand",
        causal_status="custom-status",
        dataset_ref="custom-data",
    )

    assert spec.design == "custom-design"
    assert spec.estimand_type == "custom-estimand"
    assert spec.causal_status == "custom-status"
    assert spec.dataset_ref == "custom-data"


def test_non_itt_spec_defaults_remain_unchanged():
    spec = ModelSpec(model_id="lrp-rli-mech-001", kind="mechanism", title="Mechanism")

    assert spec.family is None
    assert spec.design is None
    assert spec.estimand_type is None
    assert spec.causal_status is None
    assert spec.dataset_ref is None
