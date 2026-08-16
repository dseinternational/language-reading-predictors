# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guards for the denominator-free Byrne growth bootstrap (#338)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from language_reading_predictors.statistical_models.datasets import RLM_MEASURES
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
)
from language_reading_predictors.statistical_models.rlm_growth_bootstrap import (
    BOOTSTRAP_VARIANT,
    compare_bootstrap_with_likelihoods,
    monte_carlo_stability,
    participant_bayesian_bootstrap_growth,
)

from .test_datasets import _dataset, _write_synthetic


def _panel(tmp_path, *, extension: bool = True):
    path = _write_synthetic(tmp_path, extension=extension)
    return load_longitudinal_panel(
        _dataset(path),
        [RLM_MEASURES["basread"]],
        waves=(1, 2, 3),
        extension_waves=(4, 5) if extension else (),
    )


def test_participant_bootstrap_is_reproducible_and_uses_paired_extensions(tmp_path):
    panel = _panel(tmp_path)
    first = participant_bayesian_bootstrap_growth(
        panel,
        measure="basread",
        draws=5_000,
        seed=123,
    )
    second = participant_bayesian_bootstrap_growth(
        panel,
        measure="basread",
        draws=5_000,
        seed=123,
    )

    pd.testing.assert_frame_equal(first, second)
    assert set(first["variant"]) == {BOOTSTRAP_VARIANT}
    assert first["denominator"].isna().all()
    assert first["bootstrap_draws"].eq(5_000).all()
    down_4_5 = first[
        (first["readgrp_label"] == "Down syndrome")
        & (first["quantity"] == "growth_4_5_items")
    ].iloc[0]
    down_1_5 = first[
        (first["readgrp_label"] == "Down syndrome")
        & (first["quantity"] == "growth_1_5_items")
    ].iloc[0]
    assert down_4_5["n_subjects"] == 2
    assert down_1_5["n_subjects"] == 3
    assert len(first[first["quantity"].str.startswith("total_growth_")]) == 3


def test_participant_bootstrap_recovers_a_constant_paired_change(tmp_path):
    panel = _panel(tmp_path, extension=False)
    long = panel.long.copy()
    wave_col = panel.dataset.wave_col
    subject_col = panel.dataset.subject_col
    wave_1 = long[long[wave_col] == 1].set_index(subject_col)["basread"]
    wave_2 = long[wave_col] == 2
    long.loc[wave_2, "basread"] = (
        long.loc[wave_2, subject_col].map(wave_1) + 2
    )
    fixed = replace(panel, long=long)

    result = participant_bayesian_bootstrap_growth(
        fixed,
        measure="basread",
        draws=1_000,
        seed=321,
    )
    first_interval = result[result["quantity"] == "growth_1_2_items"]
    assert first_interval["mean"].eq(2.0).all()
    assert first_interval["q_lo"].eq(2.0).all()
    assert first_interval["q_hi"].eq(2.0).all()


@pytest.mark.parametrize("draws,seed", [(0, 1), (100, 0)])
def test_participant_bootstrap_rejects_nonpositive_controls(tmp_path, draws, seed):
    with pytest.raises(ValueError, match="positive integer"):
        participant_bayesian_bootstrap_growth(
            _panel(tmp_path, extension=False),
            measure="basread",
            draws=draws,
            seed=seed,
        )


def _summary_row(*, variant: str, q50: float = 2.0) -> dict[str, object]:
    return {
        "variant": variant,
        "quantity": "growth_1_3_items",
        "label": "Wave 1 to wave 3",
        "readgrp_label": "Down syndrome",
        "q_lo": 1.0,
        "q50": q50,
        "q_hi": 3.0,
    }


def test_monte_carlo_stability_is_fail_closed():
    primary = pd.DataFrame([_summary_row(variant=BOOTSTRAP_VARIANT)])
    replicate = primary.copy()
    comparison, passed = monte_carlo_stability(
        primary,
        replicate,
        observed_maximum=18,
    )
    assert passed["status"] == "pass"
    assert comparison.loc[0, "maximum_quantile_difference"] == 0

    replicate.loc[0, "q_hi"] = 4.0
    _comparison, failed = monte_carlo_stability(
        primary,
        replicate,
        observed_maximum=18,
    )
    assert failed["status"] == "no_go"


def test_five_method_comparison_requires_stable_complete_reference():
    variants = (
        "beta_binomial_1x",
        "beta_binomial_2x",
        "beta_binomial_4x",
        "negative_binomial",
    )
    reference = pd.DataFrame(
        [_summary_row(variant=variant, q50=2.0 + 0.1 * index) for index, variant in enumerate(variants)]
    )
    bootstrap = pd.DataFrame(
        [_summary_row(variant=BOOTSTRAP_VARIANT, q50=2.4)]
    )
    comparison, passed = compare_bootstrap_with_likelihoods(
        bootstrap,
        reference,
        observed_maximum=18,
        expected_likelihood_variants=variants,
        likelihood_reference_passed=True,
        monte_carlo_passed=True,
    )
    assert passed["status"] == "pass"
    assert comparison.loc[0, "n_methods"] == 5

    reversed_bootstrap = bootstrap.copy()
    reversed_bootstrap.loc[0, "q50"] = -0.1
    _comparison, failed = compare_bootstrap_with_likelihoods(
        reversed_bootstrap,
        reference,
        observed_maximum=18,
        expected_likelihood_variants=variants,
        likelihood_reference_passed=True,
        monte_carlo_passed=True,
    )
    assert failed["status"] == "no_go"

    with pytest.raises(ValueError, match="pre-specified reference set"):
        compare_bootstrap_with_likelihoods(
            bootstrap,
            reference.iloc[:-1],
            observed_maximum=18,
            expected_likelihood_variants=variants,
            likelihood_reference_passed=True,
            monte_carlo_passed=True,
        )
