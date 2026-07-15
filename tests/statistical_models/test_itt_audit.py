# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import numpy as np
import pytest

from language_reading_predictors.statistical_models.itt_audit import (
    analysis_set_table,
    randomised_postscore_bounds,
)


def _prepared(*, G, post, n_trials=10):
    return SimpleNamespace(
        G=np.asarray(G, dtype=int),
        post_counts={"W": np.asarray(post, dtype=float)},
        n_trials={"W": n_trials},
    )


def test_analysis_set_table_distinguishes_randomised_available_and_fitted():
    prepared = _prepared(G=[1] * 27 + [0] * 25, post=[1] * 51 + [np.nan])

    table = analysis_set_table(prepared, outcome_symbol="W").set_index("arm")

    assert table.loc["intervention", "randomised_n"] == 29
    assert table.loc["intervention", "available_t1_n"] == 28
    assert table.loc["intervention", "fitted_n"] == 27
    assert table.loc["intervention", "absent_from_archive_n"] == 1
    assert table.loc["intervention", "not_in_fitted_analysis_n"] == 2
    assert table.loc["intervention", "excluded_after_archive_n"] == 1
    assert table.loc["control", "randomised_n"] == 28
    assert table.loc["control", "available_t1_n"] == 26
    assert table.loc["control", "fitted_n"] == 24
    assert table.loc["control", "absent_from_archive_n"] == 2
    assert table.loc["control", "not_in_fitted_analysis_n"] == 4
    assert table.loc["control", "excluded_after_archive_n"] == 2


def test_randomised_postscore_bounds_assign_extreme_missing_scores():
    # Observed: intervention 8/10 and 6/10; control 4/10.  The remaining 27 and
    # 27 randomised outcomes per arm are bounded at 0/1 rather than imputed.
    prepared = _prepared(G=[1, 1, 0], post=[8, 6, 4])

    row = randomised_postscore_bounds(prepared, "W").iloc[0]

    assert row.observed_mean_difference == pytest.approx(0.3)
    assert row.missing_intervention_n == 27
    assert row.missing_control_n == 27
    assert row.worst_case_lower == pytest.approx((1.4 / 29) - (27.4 / 28))
    assert row.worst_case_upper == pytest.approx((28.4 / 29) - (0.4 / 28))
    assert row.worst_case_items_lower == pytest.approx(row.worst_case_lower * 10)


def test_randomised_postscore_bounds_rejects_out_of_range_scores():
    prepared = _prepared(G=[1, 0], post=[11, 0])

    with pytest.raises(ValueError, match="outside"):
        randomised_postscore_bounds(prepared, "W")
