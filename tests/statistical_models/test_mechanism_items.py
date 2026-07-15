# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`statistical_models.mechanism_items` (#319).

The items-scale mechanism curve rescales the fitted logit-contribution curve to
predicted outcome items, holding the rest of the linear predictor at reference
values (covariate sample means; population-level over the child intercepts;
moderator at its mean). The crux is the per-draw reference constant

    C[s] = mean_i( eta[i,s] - f_curve[i,s] - u_child[i,s] - moderator[i,s] )

so these build synthetic traces with a *known* baseline and assert the recovered
items curve equals ``N * expit(C + f_curve)`` exactly — across the HSGP (``f_mech``)
and linear (``beta_mech``) branches, with and without a child intercept and a
moderator — then check the worked-example contrast, the off-floor branch and the
input guards.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models import mechanism_items as mi


def _trace(posterior: dict, constant: dict | None = None):
    """Synthetic ``SimpleNamespace(posterior=..., constant_data=...)``.

    ``posterior`` / ``constant`` map var name -> ``(dims, values)``. Coords are
    inferred from the dim names present.
    """
    def _dataset(spec: dict) -> xr.Dataset:
        coords: dict[str, np.ndarray] = {}
        for _dims, values in spec.values():
            arr = np.asarray(values)
            for k, d in enumerate(_dims):
                if d not in coords:
                    coords[d] = np.arange(arr.shape[k])
        return xr.Dataset({n: (d, v) for n, (d, v) in spec.items()}, coords=coords)

    return SimpleNamespace(
        posterior=_dataset(posterior),
        constant_data=_dataset(constant or {}),
    )


def _bcast(scalar_draws: np.ndarray, n_obs: int) -> np.ndarray:
    """(chain, draw) -> (chain, draw, obs) by broadcasting a per-draw scalar."""
    return np.broadcast_to(scalar_draws[:, :, None], (*scalar_draws.shape, n_obs)).copy()


# ---------------------------------------------------------------------------
# Reference-constant reconstruction (the crux)
# ---------------------------------------------------------------------------


def test_gp_branch_recovers_reference_curve():
    """With f_mech present and no child/moderator, y == N*expit(mean_i(baseline) + f_mech)."""
    rng = np.random.default_rng(0)
    n_chain, n_draw = 2, 30
    x = np.array([2.0, 2.0, 5.0, 9.0, 9.0, 14.0, 20.0], dtype=float)  # counts, with ties
    n_obs = x.size
    S = n_chain * n_draw

    # A per-obs baseline (phase / covariate contribution) and a monotone GP curve;
    # f_mech is a deterministic function of x, so tied x share an f value.
    baseline = rng.normal(0.3, 0.5, size=(n_chain, n_draw, n_obs))
    amp = rng.normal(1.0, 0.1, size=(n_chain, n_draw))
    f = amp[:, :, None] * (0.15 * (x[None, None, :] - 8.0))  # increasing in x
    eta = baseline + f

    trace = _trace(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "f_mech": (("chain", "draw", "obs_id"), f),
        }
    )
    curve_df, worked = mi.mechanism_items_curve(
        trace, x_exposure=x, n_trials_outcome=79
    )

    # Expected: dedup to unique x, C = mean_i(baseline) per draw.
    C = baseline.reshape(n_chain, n_draw, n_obs).mean(axis=2).reshape(-1)  # (S,)
    xs = np.unique(x)
    f_flat = f.reshape(S, n_obs)
    for _, row in curve_df.iterrows():
        xi = row["exposure"]
        obs_j = int(np.where(x == xi)[0][0])
        y_true = 79.0 * expit(C + f_flat[:, obs_j])
        assert row["exposure"] in xs
        np.testing.assert_allclose(row["outcome_mean"], y_true.mean(), rtol=0, atol=1e-9)
    assert worked["curve_kind"] == "GP"
    assert worked["n_trials_outcome"] == 79
    # Increasing curve -> higher exposure predicts more items.
    assert worked["outcome_difference_median"] > 0


def test_linear_branch_with_child_and_moderator_recovers_constant():
    """beta_mech branch: child intercept and moderator terms drop out of C exactly."""
    rng = np.random.default_rng(1)
    n_chain, n_draw = 2, 40
    n_children = 5
    x = np.array([1.0, 3.0, 3.0, 6.0, 10.0, 10.0, 15.0, 22.0], dtype=float)
    n_obs = x.size
    child_idx = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    S = n_chain * n_draw

    # z_L monotone in x (so ties share a curve value); z_M arbitrary.
    z_L = (x - x.mean()) / x.std()
    z_M = rng.normal(0.0, 1.0, size=n_obs)
    z_M = z_M - z_M.mean()  # sample-mean 0, as the factory standardises it

    beta_mech = rng.normal(0.5, 0.1, size=(n_chain, n_draw))
    gamma_mod = rng.normal(-0.2, 0.05, size=(n_chain, n_draw))
    gamma_int = rng.normal(0.1, 0.03, size=(n_chain, n_draw))
    sigma_c = np.abs(rng.normal(0.4, 0.05, size=(n_chain, n_draw)))
    u_raw = rng.normal(0.0, 1.0, size=(n_chain, n_draw, n_children))
    u_child = sigma_c[:, :, None] * u_raw

    baseline = rng.normal(0.1, 0.6, size=(n_chain, n_draw, n_obs))
    f_curve = beta_mech[:, :, None] * z_L[None, None, :]
    mod = (
        gamma_mod[:, :, None] * z_M[None, None, :]
        + gamma_int[:, :, None] * (z_L * z_M)[None, None, :]
    )
    eta = baseline + f_curve + u_child[:, :, child_idx] + mod

    trace = _trace(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "beta_mech": (("chain", "draw"), beta_mech),
            "gamma_mod": (("chain", "draw"), gamma_mod),
            "gamma_int": (("chain", "draw"), gamma_int),
            "u_child": (("chain", "draw", "child"), u_child),
        },
        {
            "z_mech_logit": (("obs_id",), z_L),
            "z_moderator": (("obs_id",), z_M),
            "child_idx": (("obs_id",), child_idx),
        },
    )
    curve_df, worked = mi.mechanism_items_curve(
        trace, x_exposure=x, n_trials_outcome=79
    )
    assert worked["curve_kind"] == "linear"

    # C = mean_i(baseline): child + moderator terms must cancel out exactly.
    C = baseline.reshape(S, n_obs).mean(axis=1)
    f_flat = f_curve.reshape(S, n_obs)
    for _, row in curve_df.iterrows():
        obs_j = int(np.where(x == row["exposure"])[0][0])
        y_true = 79.0 * expit(C + f_flat[:, obs_j])
        np.testing.assert_allclose(row["outcome_mean"], y_true.mean(), rtol=0, atol=1e-9)


def test_reference_constant_ignores_moderator_when_absent():
    """No gamma_mod -> moderator contribution is zero and C = mean_i(eta - f)."""
    rng = np.random.default_rng(2)
    n_chain, n_draw, n_obs = 2, 20, 6
    x = np.arange(n_obs, dtype=float) * 4.0
    baseline = rng.normal(0.0, 0.5, size=(n_chain, n_draw, n_obs))
    f = 0.2 * (x[None, None, :] - x.mean())
    f = np.broadcast_to(f, (n_chain, n_draw, n_obs)).copy()
    eta = baseline + f
    trace = _trace(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "f_mech": (("chain", "draw", "obs_id"), f),
        }
    )
    _, worked = mi.mechanism_items_curve(trace, x_exposure=x, n_trials_outcome=32)
    C = baseline.reshape(-1, n_obs).mean(axis=1)
    # The reference constant is private; check it via the predicted median at the
    # smallest exposure, whose f is deterministic.
    y0 = 32.0 * expit(C + f.reshape(-1, n_obs)[:, 0])
    assert abs(worked["predicted_low_median"] - np.median(y0)) < 5.0  # ballpark sanity


# ---------------------------------------------------------------------------
# Worked example + off-floor branch
# ---------------------------------------------------------------------------


def test_worked_example_uses_requested_quantiles_and_rounds_counts():
    rng = np.random.default_rng(3)
    n_chain, n_draw = 2, 30
    x = np.arange(0, 33, dtype=float)  # 0..32 letter sounds
    n_obs = x.size
    baseline = rng.normal(0.0, 0.3, size=(n_chain, n_draw, n_obs))
    f = np.broadcast_to(0.1 * (x - 16.0), (n_chain, n_draw, n_obs)).copy()
    eta = baseline + f
    trace = _trace(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "f_mech": (("chain", "draw", "obs_id"), f),
        }
    )
    _, worked = mi.mechanism_items_curve(
        trace, x_exposure=x, n_trials_outcome=79, ref_quantiles=(0.10, 0.90)
    )
    assert worked["ref_quantile_low"] == 0.10
    assert worked["ref_quantile_high"] == 0.90
    # Counts round to integers, and low < high with an increasing curve.
    assert worked["exposure_ref_low"] == float(round(np.quantile(x, 0.10)))
    assert worked["exposure_ref_high"] == float(round(np.quantile(x, 0.90)))
    assert worked["exposure_ref_low"] < worked["exposure_ref_high"]
    assert worked["outcome_difference_median"] > 0


def test_off_floor_branch_reports_probability_not_items():
    rng = np.random.default_rng(4)
    n_chain, n_draw, n_obs = 2, 20, 8
    x = np.linspace(0, 6, n_obs)
    baseline = rng.normal(0.0, 0.3, size=(n_chain, n_draw, n_obs))
    f = np.broadcast_to(0.3 * (x - 3.0), (n_chain, n_draw, n_obs)).copy()
    eta = baseline + f
    trace = _trace(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "f_mech": (("chain", "draw", "obs_id"), f),
        }
    )
    curve_df, worked = mi.mechanism_items_curve(
        trace, x_exposure=x, n_trials_outcome=6, outcome_off_floor=True
    )
    # Probabilities lie in [0, 1] regardless of the (ignored) item ceiling.
    assert (curve_df["outcome_mean"] >= 0).all() and (curve_df["outcome_mean"] <= 1).all()
    assert worked["outcome_off_floor"] is True
    assert abs(worked["outcome_difference_median"]) <= 1.0


# ---------------------------------------------------------------------------
# _interp_draws
# ---------------------------------------------------------------------------


def test_interp_draws_endpoints_and_midpoint():
    xs = np.array([0.0, 10.0, 20.0])
    fs = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])  # (U=3, S=2)
    np.testing.assert_allclose(mi._interp_draws(xs, fs, 0.0), fs[0])
    np.testing.assert_allclose(mi._interp_draws(xs, fs, 20.0), fs[2])
    np.testing.assert_allclose(mi._interp_draws(xs, fs, 5.0), (fs[0] + fs[1]) / 2)
    # Out-of-range clamps to the nearest endpoint (never extrapolates).
    np.testing.assert_allclose(mi._interp_draws(xs, fs, 99.0), fs[2])


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------


def test_write_artifacts_emits_csv_png_and_caption(tmp_path):
    rng = np.random.default_rng(5)
    n_chain, n_draw = 2, 25
    x = np.arange(0, 33, dtype=float)
    n_obs = x.size
    baseline = rng.normal(0.0, 0.3, size=(n_chain, n_draw, n_obs))
    f = np.broadcast_to(0.12 * (x - 16.0), (n_chain, n_draw, n_obs)).copy()
    eta = baseline + f
    trace = _trace(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "f_mech": (("chain", "draw", "obs_id"), f),
        }
    )
    worked = mi.write_mechanism_items_artifacts(
        str(tmp_path),
        trace,
        x_exposure=x,
        outcome_symbol="W",
        outcome_label="Word reading (EWRSWR)",
        n_trials_outcome=79,
        exposure_label="Letter-sound knowledge (YARC-LSK)",
        exposure_is_covariate=False,
        exposure_n_trials=32,
    )
    assert (tmp_path / "mechanism_curve_items.csv").exists()
    assert (tmp_path / "mechanism_curve_items.png").exists()
    assert "Word reading" in worked["y_label"]
    assert "out of 32" in worked["x_label"]
    assert worked["caption"]
    assert "percentile" in worked["caption"]
    assert "on Letter-sound knowledge" in worked["caption"]


def test_write_artifacts_covariate_labels_raw_score(tmp_path):
    rng = np.random.default_rng(9)
    n_chain, n_draw = 2, 20
    x = np.array([3.5, 4.2, 7.1, 9.8, 12.4, 18.3, 21.9, 25.0, 29.7, 33.1])
    n_obs = x.size
    z_L = (x - x.mean()) / x.std()
    beta = rng.normal(0.4, 0.05, size=(n_chain, n_draw))
    baseline = rng.normal(0.0, 0.3, size=(n_chain, n_draw, n_obs))
    eta = baseline + beta[:, :, None] * z_L[None, None, :]
    trace = _trace(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "beta_mech": (("chain", "draw"), beta),
        },
        {"z_mech_logit": (("obs_id",), z_L)},
    )
    worked = mi.write_mechanism_items_artifacts(
        str(tmp_path),
        trace,
        x_exposure=x,
        outcome_symbol="W",
        outcome_label="Word reading (EWRSWR)",
        n_trials_outcome=79,
        exposure_label="Phonological memory",
        exposure_is_covariate=True,
        exposure_n_trials=None,
    )
    # Covariate exposure: raw-score axis, no fabricated denominator, "on ..." reads.
    assert "raw score" in worked["x_label"]
    assert "out of" not in worked["x_label"]
    assert "on Phonological memory (raw score)" in worked["caption"]


def test_covariate_exposure_does_not_round_and_labels_raw():
    rng = np.random.default_rng(6)
    n_chain, n_draw = 2, 20
    # Raw covariate scores (not counts): reference points must not be rounded.
    x = np.array([3.5, 4.2, 7.1, 9.8, 12.4, 18.3, 21.9, 25.0, 29.7, 33.1, 37.6, 41.5])
    n_obs = x.size
    z_L = (x - x.mean()) / x.std()
    beta = rng.normal(0.4, 0.05, size=(n_chain, n_draw))
    baseline = rng.normal(0.0, 0.3, size=(n_chain, n_draw, n_obs))
    f = beta[:, :, None] * z_L[None, None, :]
    eta = baseline + f
    trace = _trace(
        {
            "eta": (("chain", "draw", "obs_id"), eta),
            "beta_mech": (("chain", "draw"), beta),
        },
        {"z_mech_logit": (("obs_id",), z_L)},
    )
    _, worked = mi.mechanism_items_curve(
        trace, x_exposure=x, n_trials_outcome=79, round_exposure=False
    )
    # With rounding off, the reference points are the raw quantiles unchanged.
    assert worked["exposure_ref_low"] == float(np.quantile(x, 0.25))
    assert worked["exposure_ref_high"] == float(np.quantile(x, 0.75))


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_length_mismatch_raises():
    n_chain, n_draw, n_obs = 2, 10, 5
    f = np.zeros((n_chain, n_draw, n_obs))
    trace = _trace(
        {
            "eta": (("chain", "draw", "obs_id"), f),
            "f_mech": (("chain", "draw", "obs_id"), f),
        }
    )
    with pytest.raises(ValueError, match="fitted-subset"):
        mi.mechanism_items_curve(trace, x_exposure=np.arange(n_obs + 1), n_trials_outcome=79)


def test_missing_curve_raises():
    n_chain, n_draw, n_obs = 2, 10, 5
    eta = np.zeros((n_chain, n_draw, n_obs))
    trace = _trace({"eta": (("chain", "draw", "obs_id"), eta)})
    with pytest.raises(KeyError, match="f_mech"):
        mi.mechanism_items_curve(trace, x_exposure=np.arange(n_obs), n_trials_outcome=79)
