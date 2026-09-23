# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Huber threshold rule shared by the tuner and the model registry."""

import numpy as np
import pytest

from language_reading_predictors.models.objective import (
    HUBER_TUNING_CONSTANT,
    MAD_TO_SIGMA,
    ROBUST_MAD_RULE,
    robust_huber_delta,
)


def test_delta_is_huber_constant_times_normal_consistent_mad():
    y = np.array([1.0, 2.0, 4.0, 7.0, 11.0])  # median 4, |dev| = 3,2,0,3,7 -> MAD 3
    out = robust_huber_delta(y)
    assert out.rule == ROBUST_MAD_RULE
    assert out.scale_source == "mad"
    assert out.mad == 3.0
    assert out.delta == pytest.approx(HUBER_TUNING_CONSTANT * MAD_TO_SIGMA * 3.0)
    assert out.n_rows == 5


def test_floored_target_falls_back_to_mean_absolute_deviation():
    # More than half the rows at the floor: MAD is zero, the mean deviation is not.
    y = np.array([0.0] * 6 + [1.0, 2.0, 3.0, 6.0])
    out = robust_huber_delta(y)
    assert out.mad == 0.0
    assert out.scale_source == "mean_abs_dev"
    assert out.delta == pytest.approx(HUBER_TUNING_CONSTANT * 1.2)


def test_reproduces_the_objective_sensitivity_thresholds():
    """The 22 September check derived 12.96 for word-reading level (MAD 6.5)."""
    y = np.concatenate([np.zeros(10), np.full(10, 13.0)])  # median 6.5 with even n
    out = robust_huber_delta(y)
    assert out.median == 6.5
    assert out.delta == pytest.approx(HUBER_TUNING_CONSTANT * MAD_TO_SIGMA * 6.5)


@pytest.mark.parametrize("y", [[], [np.nan, 1.0], [2.0, 2.0, 2.0]])
def test_degenerate_targets_are_rejected(y):
    with pytest.raises(ValueError):
        robust_huber_delta(np.array(y, dtype=float))
