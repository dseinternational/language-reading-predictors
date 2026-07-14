# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the pooled treatment x baseline moderation meta-analysis (#228 item 9).

The model build is checked structurally and the summary is checked against a
synthetic posterior (no MCMC), matching the suite's fast-unit-test convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from language_reading_predictors.statistical_models.pooled_moderation import (
    _fake_idata,
    build_pooled_moderation_model,
    summarise,
)


def test_build_model_structure_and_guard():
    m = build_pooled_moderation_model(
        [-0.2, -0.1, 0.0, -0.15], [0.1, 0.08, 0.12, 0.1]
    )
    assert {"mu", "tau", "theta", "obs"} <= set(m.named_vars)
    with pytest.raises(ValueError):
        build_pooled_moderation_model([0.1], [0.1])  # need >= 2 outcomes
    with pytest.raises(ValueError):
        build_pooled_moderation_model([0.1, 0.2], [0.1])  # length mismatch


def test_summarise_shapes_and_probability():
    rng = np.random.default_rng(0)
    chain, draw, K = 2, 500, 3
    mu = rng.normal(-0.15, 0.03, size=(chain, draw))  # clearly negative
    tau = np.abs(rng.normal(0.05, 0.01, size=(chain, draw)))
    theta = rng.normal(-0.15, 0.05, size=(chain, draw, K))
    idata = _fake_idata(mu, tau, theta)

    pooled, by = summarise(
        idata, labels=["W", "L", "B"], effects=[-0.1, -0.2, -0.3], ses=[0.1, 0.1, 0.1]
    )

    assert set(pooled.columns) >= {"term", "median", "lo", "hi", "prob_lt_0"}
    mu_row = pooled[pooled["term"].str.startswith("pooled")].iloc[0]
    # bracket access: "median" collides with the pandas Series.median method
    assert mu_row["lo"] <= mu_row["median"] <= mu_row["hi"]
    assert 0.9 < mu_row["prob_lt_0"] <= 1.0  # mu draws are all negative
    tau_row = pooled[pooled["term"].str.startswith("hetero")].iloc[0]
    assert np.isnan(tau_row["prob_lt_0"])  # one-sided probability only meaningful for mu

    assert set(by.columns) >= {
        "outcome",
        "raw_mean",
        "raw_se",
        "shrunk_median",
        "shrunk_lo",
        "shrunk_hi",
    }
    assert len(by) == 3
    assert (by["shrunk_lo"] <= by["shrunk_median"]).all()
    assert (by["shrunk_median"] <= by["shrunk_hi"]).all()


def test_summarise_label_mismatch_raises():
    rng = np.random.default_rng(1)
    idata = _fake_idata(
        rng.normal(0, 0.1, (2, 50)),
        np.abs(rng.normal(0.1, 0.02, (2, 50))),
        rng.normal(0, 0.1, (2, 50, 3)),
    )
    with pytest.raises(ValueError):
        summarise(idata, labels=["W", "L"], effects=[0.0, 0.0], ses=[0.1, 0.1])
