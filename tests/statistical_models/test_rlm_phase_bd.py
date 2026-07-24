# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Byrne Phase B/D models (#338): frames, factories, specs.

Synthetic Byrne-shaped CSVs only - the real participant data is never read in
unit tests. The frame loaders take a ``path`` override for exactly this reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from language_reading_predictors.statistical_models.factories import (
    build_rlm_adjusted_model,
    build_rlm_corr_factor_model,
    build_rlm_horseshoe_model,
    build_rlm_joint_growth_model,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
    load_rlm_span_frame,
    load_rlm_wave_battery,
)

_MEASURE_COLS = [
    "basread", "basspel", "woco", "bpvs", "trog", "basdig", "bassim",
    "basmat", "basnum",
]


def _write_battery_csv(tmp_path, *, drop_one=False):
    """12 children (4 per group) x 3 waves of the full nine-measure battery."""
    rng = np.random.default_rng(7)
    rows = []
    for grp in (1, 2, 3):
        for k in range(4):
            sid = f"S{grp}{k}"
            for t in (1, 2, 3):
                row = {"subject_id": sid, "time": t, "readgrp": grp,
                       "age": 60 + 12 * t + int(rng.integers(0, 6))}
                for col in _MEASURE_COLS:
                    row[col] = int(rng.integers(0, 15)) + 2 * (t - 1)
                rows.append(row)
    df = pd.DataFrame(rows)
    if drop_one:
        # S10 loses its wave-1 bpvs -> not complete-case for the span frame.
        df.loc[(df.subject_id == "S10") & (df.time == 1), "bpvs"] = np.nan
    path = tmp_path / "rlm_synth_battery.csv"
    df.to_csv(path, index=False)
    return path


def test_span_frame_complete_case(tmp_path):
    path = _write_battery_csv(tmp_path, drop_one=True)
    frame = load_rlm_span_frame(path=path)
    # S10 misses a wave-1 predictor -> dropped from the one-row-per-child frame.
    assert frame.n_obs == 11
    assert "S10" not in list(frame.subject_ids)
    assert frame.dropped_rows == 1
    assert set(frame.predictors) == {"bpvs", "trog", "basdig", "bassim", "basnum", "age"}
    # Standardised predictors: mean ~ 0, sd ~ 1.
    for k, z in frame.predictors.items():
        assert abs(float(np.mean(z))) < 1e-8, k
    assert frame.outcome == "basread"
    assert frame.n_trials["basread"] == 90
    assert frame.post_counts["basread"].dtype.kind == "i"


def test_duplicate_subject_wave_row_raises(tmp_path):
    # #358 review: a duplicated (subject, wave) row would silently multiply rows
    # through the wide-frame join - it must be rejected at load.
    path = _write_battery_csv(tmp_path)
    df = pd.read_csv(path)
    dup = df[(df.subject_id == "S20") & (df.time == 1)]
    pd.concat([df, dup], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate rows for subjects"):
        load_rlm_span_frame(path=path)
    with pytest.raises(ValueError, match="Duplicate rows for subjects"):
        load_rlm_wave_battery(wave=1, path=path)


def test_wave_battery_complete_case(tmp_path):
    path = _write_battery_csv(tmp_path)
    battery = load_rlm_wave_battery(wave=3, path=path)
    assert battery.n_obs == 12
    assert set(battery.indicators) == set(_MEASURE_COLS)
    for k, z in battery.indicators.items():
        assert abs(float(np.mean(z))) < 1e-8, k


def test_build_rlm_adjusted_and_horseshoe(tmp_path):
    path = _write_battery_csv(tmp_path)
    frame = load_rlm_span_frame(path=path)

    adj = build_rlm_adjusted_model(frame)
    names = {v.name for v in adj.model.free_RVs}
    assert {"alpha", "gamma_own", "kappa"}.issubset(names)
    assert {f"beta_{k}" for k in frame.predictors}.issubset(names)
    # Group nuisance: exactly two dummies (three groups, largest = reference).
    assert sum(n.startswith("beta_group_nuisance_") for n in names) == 2

    hs = build_rlm_horseshoe_model(frame)
    hs_names = {v.name for v in hs.model.free_RVs}
    assert {"hs_tau", "hs_c2", "hs_lambda", "hs_z"}.issubset(hs_names)
    assert "beta" in {v.name for v in hs.model.deterministics}

    with adj.model:
        pp = pm.sample_prior_predictive(draws=3, random_seed=1)
    assert pp.prior_predictive["y_post"].shape[-1] == frame.n_obs


def test_build_rlm_corr_factor_single_indicator_fixed(tmp_path):
    path = _write_battery_csv(tmp_path)
    battery = load_rlm_wave_battery(wave=3, path=path)
    domains = {
        "reading": ("basread", "basspel", "woco"),
        "language": ("bpvs", "trog"),
        "memory": ("basdig",),
        "ability": ("bassim", "basmat", "basnum"),
    }
    built = build_rlm_corr_factor_model(
        battery, domains=domains, single_indicator_reliability=0.8
    )
    names = {v.name for v in built.model.free_RVs}
    # Communality parameterisation (#409 item B): the free parameter is the
    # communality; the loading sqrt(c) and residual sqrt(1 - c) are derived
    # deterministics (enforcing lambda**2 + sigma**2 = 1, the fix for the Heywood
    # loading-residual ridge).
    assert {"factor_cov", "communality_free"}.issubset(names)
    det_names = {v.name for v in built.model.deterministics}
    assert {"lambda_free", "sigma_free"}.issubset(det_names)
    # The single-indicator memory domain contributes NO free loading/residual:
    # 9 indicators, 1 fixed -> 8 free.
    assert built.model.named_vars["communality_free"].eval().shape == (8,)
    assert built.model.named_vars["lambda_free"].eval().shape == (8,)
    # The fixed indicator's loading/residual are sqrt(r) / sqrt(1 - r).
    loading = built.model.named_vars["loading"].eval()
    idx = list(built.model.coords["indicator"]).index("basdig")
    assert np.isclose(loading[idx], np.sqrt(0.8))
    with built.model:
        pp = pm.sample_prior_predictive(draws=3, random_seed=1)
    corr = pp.prior["factor_corr"].values[0]
    assert np.allclose(np.diagonal(corr, axis1=-2, axis2=-1), 1.0)


def test_build_rlm_corr_factor_rejects_bad_reliability(tmp_path):
    path = _write_battery_csv(tmp_path)
    battery = load_rlm_wave_battery(wave=3, path=path)
    with pytest.raises(ValueError, match="reliability"):
        build_rlm_corr_factor_model(
            battery,
            domains={"memory": ("basdig",), "reading": ("basread", "basspel")},
            single_indicator_reliability=1.0,
        )


def test_build_rlm_corr_factor_rejects_bad_communality_shapes(tmp_path):
    path = _write_battery_csv(tmp_path)
    battery = load_rlm_wave_battery(wave=3, path=path)
    with pytest.raises(ValueError, match="comm_alpha and comm_beta"):
        build_rlm_corr_factor_model(
            battery,
            domains={"memory": ("basdig",), "reading": ("basread", "basspel")},
            comm_alpha=0.0,
        )


def test_build_rlm_joint_growth(tmp_path):
    from language_reading_predictors.statistical_models.datasets import (
        RLM_MEASURES,
    )

    from .test_datasets import _dataset

    path = _write_battery_csv(tmp_path)
    panel = load_longitudinal_panel(
        _dataset(path),
        [RLM_MEASURES[m] for m in ("basread", "bpvs", "basdig")],
        waves=(1, 2, 3),
    )
    built = build_rlm_joint_growth_model(
        panel, measures=("basread", "bpvs", "basdig")
    )
    names = {v.name for v in built.model.free_RVs}
    assert {
        "eta_cell", "sigma_subject", "kappa", "measure_corr_chol", "z_subject",
    }.issubset(names)
    # Group-indexed scales per measure: (measure, group) = (3, 3).
    assert built.model.named_vars["sigma_subject"].eval().shape == (3, 3)
    assert built.model.named_vars["kappa"].eval().shape == (3, 3)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    corr = pp.prior["measure_corr"].values[0]
    assert np.allclose(np.diagonal(corr, axis1=-2, axis2=-1), 1.0)
    assert np.allclose(corr, np.swapaxes(corr, -1, -2))


_PHASE_BD_SPECS = {
    "lrp-rlm-adj-001": ("adjusted", "basread"),
    "lrp-rlm-hs-001": ("horseshoe", "basread"),
    "lrp-rlm-mm-001": ("corr_factor", None),
    "lrp-rlm-jc-001": ("historical_joint", None),
}


@pytest.mark.parametrize("model_id, expected", sorted(_PHASE_BD_SPECS.items()))
def test_phase_bd_specs_well_formed(model_id, expected):
    from language_reading_predictors.statistical_models.registry import (
        discover_models,
    )

    kind, outcome = expected
    models = discover_models()
    assert model_id in models, f"{model_id} not auto-discovered"
    spec = models[model_id].SPEC
    assert spec.model_id == model_id
    assert spec.kind == kind
    assert spec.study_id == "rlm"
    assert spec.outcome_symbol == outcome
    assert spec.causal_status == "none"
    assert spec.design == "historical_cohort"
