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

from language_reading_predictors.statistical_models.pipelines import (
    dose_response as dose_pipeline,
    mechanism as mech_pipeline,
)
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
    resolve_mechanism_run_plan,
)
from language_reading_predictors.statistical_models.preprocessing import (
    Standardiser,
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


def test_mechanism_summary_writes_both_declared_contrasts(tmp_path, monkeypatch):
    """``mechanism_summary.csv`` carries the headline first, the secondary second.

    #602: the family declares one natural-scale estimand — the interquartile
    exposure contrast standardised over the fitted rows — and the observed-range
    contrast is retained beside it under an explicit label. Both rows must name
    their estimand machine-readably, and the headline must be the first row so
    ``_kf_csv_row`` picks it up.
    """
    from language_reading_predictors.statistical_models import mechanism_items as mi

    counts = np.array([0.0, 16.0, 32.0])
    z_mech, scaler = standardise(logit_safe(counts, 32))
    beta = np.array([[0.2, 0.4, 0.6]])
    eta = beta[:, :, None] * z_mech[None, None, :]
    posterior = _posterior(
        beta_mech=(("chain", "draw"), beta),
        eta=(("chain", "draw", "obs_id"), eta),
    )
    constant = xr.Dataset(
        {"z_mech_logit": (("obs_id",), z_mech)},
        coords={"obs_id": np.arange(counts.size)},
    )
    spec = ModelSpec(
        model_id="test-mechanism-summary",
        kind="mechanism",
        title="test",
        outcome_symbol="W",
        mechanism_symbol="L",
        adjustment=["G", "W_pre"],
        model_settings=MechanismModelSettings(
            outcomes=("W", "L"), linear_mechanism=True
        ),
    )
    ctx = SimpleNamespace(
        trace=SimpleNamespace(posterior=posterior, constant_data=constant),
        spec=spec,
        resolved_plan=resolve_mechanism_run_plan(spec),
        prepared=SimpleNamespace(
            post_counts={"L": counts},
            n_trials={"L": 32, "W": 79},
            covariates={},
            covariate_scalers={},
            n_obs=counts.size,
        ),
        reporting=SimpleNamespace(ci_prob=0.95),
        output_dir=str(tmp_path),
        tables={},
    )
    monkeypatch.setattr(
        mech_pipeline, "save_styled_figure", lambda *_a, **_k: plt.close("all")
    )
    monkeypatch.setattr(mi, "save_styled_figure", lambda *_a, **_k: plt.close("all"))

    mech_pipeline._write_mechanism_items(ctx)

    table = pd.read_csv(tmp_path / "mechanism_summary.csv")
    assert list(table["contrast"]) == [
        "headline_interquartile",
        "secondary_observed_range",
    ]
    assert list(table["estimand"]) == [mi.HEADLINE_ESTIMAND, mi.SECONDARY_ESTIMAND]
    assert (table["reference_population"] == "fitted_rows").all()
    assert (table["child_intercept"] == "retained_at_fitted_value").all()

    def _z(count: float) -> float:
        return float((logit_safe(np.array([count]), 32)[0] - scaler.mean) / scaler.sd)

    headline, secondary = table.iloc[0], table.iloc[1]
    # Quartiles of [0, 16, 32] are 8 and 24, rounded to whole items.
    assert headline.exposure_low == pytest.approx(8.0)
    assert headline.exposure_high == pytest.approx(24.0)
    expected_headline = (
        expit(beta.reshape(-1) * _z(24.0)) - expit(beta.reshape(-1) * _z(8.0))
    ) * 79
    assert headline.items_median == pytest.approx(float(np.median(expected_headline)))
    assert headline.prob_pos == pytest.approx(1.0)

    assert secondary.exposure_low == pytest.approx(0.0)
    assert secondary.exposure_high == pytest.approx(32.0)
    expected_secondary = (
        expit(beta.reshape(-1) * z_mech[-1]) - expit(beta.reshape(-1) * z_mech[0])
    ) * 79
    assert secondary.items_median == pytest.approx(float(np.median(expected_secondary)))
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
    monkeypatch.setattr(dose_pipeline, "print_table", lambda *_args, **_kwargs: None)

    dose_pipeline.write_dose_slope_summary(ctx, period_varying=False)

    summary = pd.read_csv(tmp_path / "dose_marginal_summary.csv").iloc[0]
    expected = (expit(beta.reshape(-1)) - 0.5) * 100
    assert summary.items_median == pytest.approx(float(np.median(expected)))
    assert summary.prob_pos == pytest.approx(1.0)
    assert "dose_marginal_summary" in ctx.tables
    slope = pd.read_csv(tmp_path / "dose_slope_summary.csv").iloc[0]
    assert slope.dose_mean_sessions == pytest.approx(12.0)
    assert slope.dose_sd_sessions == pytest.approx(4.0)


def test_dose_summary_persists_explicit_treated_rows_scaler(tmp_path, monkeypatch):
    """The DiD dose companions fit slopes per 1 SD of the *treated-rows* session
    scale (``build_did_model`` re-standardises among treated P1/P2 rows), so
    ``fit_did`` passes the fitted payload's scaler explicitly. The persisted
    ``dose_mean_sessions`` / ``dose_sd_sessions`` must come from that scaler on
    every row — never from the loader's all-rows ``covariate_scalers`` entry,
    whose SD is diluted by the untreated zero-session cell. The natural-scale
    marginal likewise averages over the treated rows only (a dose step on an
    untreated waitlist-P1 row is not a supported counterfactual of the
    treated-centred design), recorded via ``n_rows`` / ``row_population``."""
    mu = np.array([[0.2, 0.4, 0.6]])
    sigma = np.array([[0.1, 0.1, 0.1]])
    bdp = np.stack([mu, mu + 0.1], axis=-1)  # (chain, draw, dose_phase)
    # Rows 0/1 (P1, treated at eta = 0) versus rows 2/3 (untreated, parked at a
    # far-from-0.5 operating point): if the mask leaked, the marginal would mix
    # the +5 rows' near-zero probability change into the average.
    eta = np.zeros((1, 3, 4))
    eta[:, :, 2:] = 5.0
    posterior = _posterior(
        mu_dose=(("chain", "draw"), mu),
        sigma_dose=(("chain", "draw"), sigma),
        beta_dose_phase=(("chain", "draw", "dose_phase"), bdp),
        eta=(("chain", "draw", "obs_id"), eta),
    )
    ctx = SimpleNamespace(
        trace=SimpleNamespace(posterior=posterior),
        spec=SimpleNamespace(outcome_symbol="L", extra={}),
        prepared=SimpleNamespace(
            phase=np.array([0, 0, 1, 1]),
            n_trials={"L": 32},
            # The loader scaler that must NOT reach the persisted columns.
            covariate_scalers={"attend": SimpleNamespace(mean=53.5, sd=31.0)},
        ),
        reporting=SimpleNamespace(ci_prob=0.95),
        output_dir=str(tmp_path),
        tables={},
    )
    monkeypatch.setattr(dose_pipeline, "print_table", lambda *_args, **_kwargs: None)

    dose_pipeline.write_dose_slope_summary(
        ctx,
        period_varying=True,
        dose_scaler=Standardiser(mean=70.1, sd=17.1),
        marginal_row_mask=np.array([True, True, False, False]),
    )

    slope = pd.read_csv(tmp_path / "dose_slope_summary.csv")
    assert slope["dose_mean_sessions"].tolist() == pytest.approx(
        [70.1] * len(slope)
    )
    assert slope["dose_sd_sessions"].tolist() == pytest.approx([17.1] * len(slope))

    marginal = pd.read_csv(tmp_path / "dose_marginal_summary.csv").iloc[0]
    # Masked rows are both phase 0 at eta = 0, so per draw the marginal is
    # expit(mu) - 0.5 exactly; any leakage of the eta = 5 rows would drag it down.
    expected = (expit(mu.reshape(-1)) - 0.5) * 32
    assert marginal.items_median == pytest.approx(float(np.median(expected)))
    assert marginal.n_rows == 2
    assert "treated" in marginal.row_population
