# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Natural-scale fit artefacts consumed by the key-findings builders (#320)."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models import pipeline
from language_reading_predictors.statistical_models.preprocessing import (
    logit_safe,
    standardise,
)


def _posterior(**variables) -> xr.Dataset:
    data = {}
    coords: dict[str, object] = {"chain": [0], "draw": np.arange(3)}
    for name, (dims, values) in variables.items():
        data[name] = (dims, values)
        for dim, size in zip(dims, np.asarray(values).shape, strict=True):
            if dim not in coords:
                coords[dim] = np.arange(size)
    return xr.Dataset(data, coords=coords)


def test_mechanism_curve_writes_posterior_items_range_summary(tmp_path, monkeypatch):
    counts = np.array([0.0, 16.0, 32.0])
    z_mech, _ = standardise(logit_safe(counts, 32))
    beta = np.array([[0.2, 0.4, 0.6]])
    eta = beta[:, :, None] * z_mech[None, None, :]
    posterior = _posterior(
        beta_mech=(("chain", "draw"), beta),
        eta=(("chain", "draw", "obs_id"), eta),
    )
    ctx = SimpleNamespace(
        trace=SimpleNamespace(posterior=posterior, constant_data=xr.Dataset()),
        spec=SimpleNamespace(
            mechanism_symbol="L",
            outcome_symbol="W",
            extra={"mechanism_is_covariate": False},
        ),
        prepared=SimpleNamespace(
            post_counts={"L": counts},
            n_trials={"L": 32, "W": 100},
            covariates={},
            covariate_scalers={},
        ),
        reporting=SimpleNamespace(ci_prob=0.95),
        output_dir=str(tmp_path),
        tables={},
    )
    monkeypatch.setattr(
        pipeline,
        "save_styled_figure",
        lambda *_args, **_kwargs: plt.close("all"),
    )

    pipeline._write_mechanism_curve(ctx)

    summary = pd.read_csv(tmp_path / "mechanism_summary.csv").iloc[0]
    expected = (
        expit(beta.reshape(-1) * z_mech[-1])
        - expit(beta.reshape(-1) * z_mech[0])
    ) * 100
    assert summary.items_median == pytest.approx(float(np.median(expected)))
    assert summary.prob_pos == pytest.approx(1.0)
    assert summary.exposure_low == pytest.approx(0.0)
    assert summary.exposure_high == pytest.approx(32.0)
    assert "mechanism_summary" in ctx.tables


def test_dose_summary_writes_items_scale_marginal(tmp_path, monkeypatch):
    beta = np.array([[0.2, 0.4, 0.6]])
    eta = np.zeros((1, 3, 2))
    posterior = _posterior(
        beta_dose=(("chain", "draw"), beta),
        eta=(("chain", "draw", "obs_id"), eta),
    )
    ctx = SimpleNamespace(
        trace=SimpleNamespace(posterior=posterior),
        spec=SimpleNamespace(outcome_symbol="W", extra={}),
        prepared=SimpleNamespace(
            phase=np.array([0, 1]),
            n_trials={"W": 100},
            covariate_scalers={"attend": SimpleNamespace(mean=12.0, sd=4.0)},
        ),
        reporting=SimpleNamespace(ci_prob=0.95),
        output_dir=str(tmp_path),
        tables={},
    )
    monkeypatch.setattr(pipeline, "print_table", lambda *_args, **_kwargs: None)

    pipeline._write_dose_slope_summary(ctx, period_varying=False)

    summary = pd.read_csv(tmp_path / "dose_marginal_summary.csv").iloc[0]
    expected = (expit(beta.reshape(-1)) - 0.5) * 100
    assert summary.items_median == pytest.approx(float(np.median(expected)))
    assert summary.prob_pos == pytest.approx(1.0)
    assert "dose_marginal_summary" in ctx.tables
    slope = pd.read_csv(tmp_path / "dose_slope_summary.csv").iloc[0]
    assert slope.dose_mean_sessions == pytest.approx(12.0)
    assert slope.dose_sd_sessions == pytest.approx(4.0)
