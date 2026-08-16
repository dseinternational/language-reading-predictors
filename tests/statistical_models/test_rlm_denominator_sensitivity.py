# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guards for the Byrne denominator/likelihood sensitivity audit (#338)."""

from __future__ import annotations

import pandas as pd
import pytest

from language_reading_predictors.statistical_models.datasets import resolve_dataset
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
)
from language_reading_predictors.statistical_models.rlm_denominator_sensitivity import (
    MAX_MEDIAN_RANGE_FRACTION,
    aggregate_sensitivity,
    build_negative_binomial_historical_growth_model,
    denominator_grid,
    panel_with_denominator,
    sensitivity_variants,
)


def _basspel_panel():
    dataset, measures = resolve_dataset("rlm")
    return load_longitudinal_panel(
        dataset,
        [measures["basspel"]],
        waves=(1, 2, 3),
        complete_case=True,
        extension_waves=(4, 5),
    )


def test_denominator_grid_is_fixed_observed_maximum_stress_not_manual_claim():
    assert denominator_grid(18) == (18, 36, 72)
    variants = sensitivity_variants(18)
    assert [variant.name for variant in variants] == [
        "beta_binomial_1x",
        "beta_binomial_2x",
        "beta_binomial_4x",
        "negative_binomial",
    ]
    assert variants[-1].denominator is None


@pytest.mark.parametrize("value", [0, -1])
def test_denominator_grid_rejects_nonpositive_maxima(value):
    with pytest.raises(ValueError, match="positive integer"):
        denominator_grid(value)


def test_panel_denominator_override_is_non_mutating_and_fail_closed():
    panel = _basspel_panel()
    widened = panel_with_denominator(panel, measure="basspel", denominator=72)

    assert panel.n_trials["basspel"] == 18
    assert widened.n_trials["basspel"] == 72
    too_small = int(panel.long["basspel"].max()) - 1
    with pytest.raises(ValueError, match="below observed"):
        panel_with_denominator(panel, measure="basspel", denominator=too_small)


def test_negative_binomial_prototype_exposes_growth_contract_without_ceiling():
    panel = _basspel_panel()
    widened = panel_with_denominator(panel, measure="basspel", denominator=72)

    original = build_negative_binomial_historical_growth_model(
        panel,
        measure="basspel",
    )
    changed_metadata = build_negative_binomial_historical_growth_model(
        widened,
        measure="basspel",
    )

    expected = {
        "score",
        "eta_cell",
        "alpha",
        "fitted_mean_items_obs",
        "mean_items",
        "growth_first_last_items",
    }
    assert expected <= set(original.model.named_vars)
    assert expected <= set(changed_metadata.model.named_vars)
    assert "kappa" not in original.model.named_vars
    assert original.model["score"].owner.op.name == "negative_binomial"


def _growth_table(*, shifted: bool = False) -> pd.DataFrame:
    variants = [
        "beta_binomial_1x",
        "beta_binomial_2x",
        "beta_binomial_4x",
        "negative_binomial",
    ]
    medians = [2.0, 2.1, 2.2, 2.3]
    if shifted:
        medians[-1] = 5.0
    return pd.DataFrame(
        {
            "variant": variants,
            "quantity": ["growth_1_3_items"] * 4,
            "label": ["Wave 1 to wave 3"] * 4,
            "readgrp_label": ["Down syndrome"] * 4,
            "q_lo": [1.0, 1.1, 1.2, 1.3],
            "q50": medians,
            "q_hi": [3.2, 3.3, 3.4, 3.5],
        }
    )


def test_aggregate_sensitivity_passes_only_complete_stable_bundle():
    diagnostics = pd.DataFrame(
        {"variant": _growth_table()["variant"], "converged": [True] * 4}
    )
    comparison, decision = aggregate_sensitivity(
        _growth_table(),
        diagnostics,
        observed_maximum=18,
    )

    assert decision["status"] == "pass"
    assert comparison.loc[0, "median_range_fraction_observed_max"] < (
        MAX_MEDIAN_RANGE_FRACTION
    )

    _comparison, failed = aggregate_sensitivity(
        _growth_table(shifted=True),
        diagnostics,
        observed_maximum=18,
    )
    assert failed["status"] == "no_go"
