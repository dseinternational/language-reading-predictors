# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Smoke tests for the longitudinal dynamic models (LRP67 LCSM + lagged suite).

These check the wave-panel loader and that each factory *builds* and can draw a
small prior predictive sample — including the #250 generalisations (multi-target
couplings, arm x window intercepts, adjuster covariate block). Full
posterior-sampling correctness is validated by the end-to-end fits in
``scripts/fit_statistical_model.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.factories import (
    build_lcsm_model,
)
from language_reading_predictors.statistical_models.preprocessing import load_wave_panel


def _write_panel_csv(tmp_path, n_children: int = 20, seed: int = 3):
    """Four-wave long CSV with the dynamic-model measures, dose and adjusters.

    Drops two outcome cells (one ewrswr, one yarclet) so the observation mask
    is exercised, plus one erbto cell and one child's hearing_c so the
    missing-indicator covariate policy is exercised too.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_children):
        sid = f"S{i:03d}"
        age0 = int(rng.integers(60, 110))
        g = int(rng.integers(1, 3))
        hearing = float(rng.integers(0, 2)) if i > 0 else np.nan
        for t in (1, 2, 3, 4):
            rows.append(
                {
                    V.SUBJECT_ID: sid,
                    V.TIME: t,
                    V.GROUP: g,
                    V.AGE: age0 + 6 * (t - 1),
                    V.HEARING_C: hearing,
                    V.EWRSWR: int(rng.integers(0, 80)),
                    V.YARCLET: int(rng.integers(0, 33)),
                    V.EOWPVT: int(rng.integers(0, 171)),
                    V.B1EXTAU: int(rng.integers(0, 25)),
                    V.B1RETAU: int(rng.integers(0, 25)),
                    "erbto": float(rng.integers(0, 37)),
                    "deapp_c": float(rng.integers(100, 300)),
                    V.ATTEND: int(rng.integers(0, 10)),
                }
            )
    df = pd.DataFrame(rows)
    # Inject missing cells: S000 reading at t2, S001 letter-sounds at t3, and
    # one phonological-memory cell for the wave-covariate indicator.
    df.loc[(df[V.SUBJECT_ID] == "S000") & (df[V.TIME] == 2), V.EWRSWR] = np.nan
    df.loc[(df[V.SUBJECT_ID] == "S001") & (df[V.TIME] == 3), V.YARCLET] = np.nan
    df.loc[(df[V.SUBJECT_ID] == "S002") & (df[V.TIME] == 4), "erbto"] = np.nan
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    return p


def _n_observed(panel) -> int:
    return int(sum(int(panel.obs_mask[s].sum()) for s in panel.outcomes))


def test_load_wave_panel_shapes_mask_and_denominators(tmp_path):
    p = _write_panel_csv(tmp_path, n_children=20)
    panel = load_wave_panel(path=p)

    assert panel.n_children == 20
    assert panel.n_waves == 4
    assert panel.outcomes == ("W", "L", "E")
    for s, denom in (("W", 79), ("L", 32), ("E", 170)):
        assert panel.counts[s].shape == (20, 4)
        assert panel.n_trials[s] == denom
        # logit is NaN exactly where the count is missing.
        assert np.array_equal(np.isnan(panel.logit[s]), ~panel.obs_mask[s])

    # Injected missing cells are masked out (S000/S001 sort to indices 0/1).
    assert not panel.obs_mask["W"][0, 1]  # S000, wave t2
    assert not panel.obs_mask["L"][1, 2]  # S001, wave t3
    assert _n_observed(panel) == 20 * 4 * 3 - 2

    # Age (interpolated) and dose (NaN->0) are fully populated after prep.
    assert not np.isnan(panel.age_std).any()
    assert not np.isnan(panel.dose_std).any()
    assert panel.age_std.shape == (20, 4)


def test_lcsm_factory_builds_and_samples_prior(tmp_path):
    p = _write_panel_csv(tmp_path, n_children=20)
    panel = load_wave_panel(path=p)
    built = build_lcsm_model(panel)
    names = {v.name for v in built.model.free_RVs}
    assert {"g_L", "g_E", "b_self", "kappa", "sigma_proc"}.issubset(names)
    assert any(v.name == "x_latent" for v in built.model.deterministics)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    assert pp.prior_predictive["y_obs"].shape[-1] == _n_observed(panel)


def test_lcsm_factory_process_noise_toggle(tmp_path):
    p = _write_panel_csv(tmp_path, n_children=15)
    panel = load_wave_panel(path=p)
    built = build_lcsm_model(panel, use_process_noise=False)
    names = {v.name for v in built.model.free_RVs}
    assert "sigma_proc" not in names
    assert not any(n.startswith("zproc_") for n in names)


# --- #250 lagged-suite generalisations --------------------------------------


def test_load_wave_panel_group_and_adjuster_covariates(tmp_path):
    p = _write_panel_csv(tmp_path, n_children=20)
    panel = load_wave_panel(
        path=p,
        outcomes=("TE", "TR", "W"),
        wave_covariates=("erbto", "deapp_c"),
        include_hearing=True,
    )
    assert panel.group is not None and set(panel.group) <= {1, 2}
    assert panel.group.shape == (20,)
    # hs dummies: S000's hearing_c is NaN -> hs 0 (clear reference) + missing 1.
    assert panel.child_covariates["hs"][0] == 0.0
    assert panel.child_covariates["hs_missing"][0] == 1.0
    assert panel.child_covariates["hs_missing"][1:].sum() == 0.0
    # Wave covariates: mean-filled + standardised + indicator, all NaN-free
    # (S002's erbto at t4 was injected missing).
    for key in ("erbto", "erbto_missing", "deapp_c", "deapp_c_missing"):
        assert panel.wave_covariates[key].shape == (20, 4)
        assert not np.isnan(panel.wave_covariates[key]).any()
    assert panel.wave_covariates["erbto_missing"][2, 3] == 1.0
    assert panel.wave_covariates["erbto_missing"].sum() == 1.0
    assert "erbto" in panel.wave_covariate_scaler


def test_lcsm_factory_multi_target_couplings_and_arm_window(tmp_path):
    p = _write_panel_csv(tmp_path, n_children=20)
    panel = load_wave_panel(
        path=p,
        outcomes=("TE", "TR", "W"),
        wave_covariates=("erbto", "deapp_c"),
        include_hearing=True,
    )
    block = (
        "hs", "hs_missing", "erbto", "erbto_missing", "deapp_c", "deapp_c_missing",
    )
    built = build_lcsm_model(
        panel,
        reading_symbol="W",
        couplings={"TE": ("W", "TR"), "TR": ("W",)},
        arm_window_intercepts=True,
        covariate_block=block,
        covariate_targets=("TE", "TR"),
    )
    names = {v.name for v in built.model.free_RVs}
    # Multi-target couplings carry the target in the name.
    assert {"g_W_TE", "g_TR_TE", "g_W_TR"}.issubset(names)
    assert "g_W" not in names and "g_TR" not in names
    # One shared slope per covariate-block entry.
    assert {f"b_{n}" for n in block}.issubset(names)
    det = {v.name for v in built.model.deterministics}
    assert "itt_w1_contrast" in det
    with built.model:
        pp = pm.sample_prior_predictive(draws=3, random_seed=1)
    # a_change is arm x trans x outcome; the contrast is per outcome.
    assert pp.prior["a_change"].shape[-3:] == (2, 3, 3)
    assert pp.prior["itt_w1_contrast"].shape[-1] == 3
    assert pp.prior_predictive["y_obs"].shape[-1] == _n_observed(panel)


def test_lcsm_factory_validation_guards(tmp_path):
    p = _write_panel_csv(tmp_path, n_children=10)
    panel = load_wave_panel(path=p)
    with pytest.raises(ValueError, match="couple to itself"):
        build_lcsm_model(panel, couplings={"W": ("W",)})
    with pytest.raises(KeyError, match="not on the panel"):
        build_lcsm_model(
            panel, covariate_block=("hs",), covariate_targets=("W",)
        )
    with pytest.raises(ValueError, match="given together"):
        build_lcsm_model(panel, covariate_block=("hs",))
    from dataclasses import replace

    no_group = replace(panel, group=None)
    with pytest.raises(ValueError, match="group column"):
        build_lcsm_model(no_group, arm_window_intercepts=True)
