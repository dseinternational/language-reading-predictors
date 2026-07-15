# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from types import SimpleNamespace

import numpy as np
import pytest

from language_reading_predictors.statistical_models.ppc_audit import (
    score_ppc_by_arm_and_baseline,
)


def _prepared():
    return SimpleNamespace(
        n_obs=6,
        G=np.array([1, 1, 1, 0, 0, 0]),
        pre_logit={"W": np.array([-2.0, 0.0, 2.0, -1.0, 0.5, 3.0])},
        post_counts={"W": np.array([0.0, 5.0, 10.0, 1.0, 4.0, 9.0])},
        n_trials={"W": 10},
    )


def test_score_ppc_reports_arm_and_baseline_calibration():
    prepared = _prepared()
    observed = prepared.post_counts["W"]
    # Replicates equal the observations, so all observed metrics sit inside the
    # predictive intervals and the table still separates both arms/baseline bands.
    replicated = np.broadcast_to(observed, (2, 20, observed.size))

    table = score_ppc_by_arm_and_baseline(prepared, "W", replicated)

    assert set(table.arm) == {"intervention", "control"}
    assert table.n.sum() == prepared.n_obs
    assert not table.ppc_mean_proportion_outside_interval.any()
    assert not table.ppc_floor_rate_outside_interval.any()
    assert not table.ppc_ceiling_rate_outside_interval.any()


def test_score_ppc_supports_flattened_joint_row_mapping():
    prepared = _prepared()
    row_indices = np.array([0, 2, 3, 5])
    observed = prepared.post_counts["W"][row_indices]
    replicated = np.broadcast_to(observed, (10, observed.size))

    table = score_ppc_by_arm_and_baseline(
        prepared,
        "W",
        replicated,
        row_indices=row_indices,
    )

    assert table.n.sum() == row_indices.size
    assert set(table.arm) == {"intervention", "control"}


def test_score_ppc_rejects_invalid_mapping_and_scores():
    prepared = _prepared()
    replicated = np.zeros((10, 2))
    with pytest.raises(ValueError, match="align"):
        score_ppc_by_arm_and_baseline(prepared, "W", replicated)

    with pytest.raises(ValueError, match="outside"):
        score_ppc_by_arm_and_baseline(
            prepared,
            "W",
            np.zeros((10, 1)),
            row_indices=np.array([0]),
            observed_counts=np.array([11.0]),
        )
