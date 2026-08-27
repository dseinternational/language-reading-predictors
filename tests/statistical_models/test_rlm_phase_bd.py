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
    build_rlm_transition_adjusted_model,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
    load_rlm_span_frame,
    load_rlm_transition_frame,
    load_rlm_wave_battery,
)

_MEASURE_COLS = [
    "basread", "basspel", "woco", "bpvs", "trog", "basdig", "bassim",
    "basmat", "basnum",
]


def _write_battery_csv(tmp_path, *, drop_one=False, waves=(1, 2, 3)):
    """12 children (4 per group) x 3 waves of the full nine-measure battery."""
    rng = np.random.default_rng(7)
    rows = []
    for grp in (1, 2, 3):
        for k in range(4):
            sid = f"S{grp}{k}"
            for t in waves:
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


def test_span_frame_group_subset_precedes_complete_case_filter(tmp_path):
    path = _write_battery_csv(tmp_path, drop_one=True)
    frame = load_rlm_span_frame(
        path=path,
        predictor_measures=("basdig", "bpvs", "bassim"),
        include_age=False,
        group_codes=(1,),
    )
    assert frame.n_obs == 3
    assert set(frame.group_code) == {1}
    assert frame.group_labels == {1: "Down syndrome"}
    assert set(frame.predictors) == {"basdig", "bpvs", "bassim"}
    assert frame.source_n_children == 12
    assert frame.eligible_n_children == 4
    assert frame.dropped_rows == 1
    assert frame.dropped_by_reason == {
        "design_group_exclusion": 8,
        "missing_required_values": 1,
    }

    model = build_rlm_adjusted_model(frame)
    names = {variable.name for variable in model.model.free_RVs}
    assert not any(name.startswith("beta_group_nuisance_") for name in names)


def test_span_frame_rejects_unknown_or_empty_group_subset(tmp_path):
    path = _write_battery_csv(tmp_path)
    with pytest.raises(ValueError, match="cannot be empty"):
        load_rlm_span_frame(path=path, group_codes=())
    with pytest.raises(ValueError, match="Unknown RLM group code"):
        load_rlm_span_frame(path=path, group_codes=(99,))


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


@pytest.mark.parametrize(
    ("value", "match"),
    [(-4.0, "below the valid lower bound"), (3.5, "integer counts")],
)
def test_wave_battery_rejects_negative_and_fractional_scores(tmp_path, value, match):
    """#631 finding 5: only the upper ceiling was checked, so a negative or
    fractional score reached the Haldane logit unnoticed."""
    path = _write_battery_csv(tmp_path)
    df = pd.read_csv(path)
    df["bpvs"] = df["bpvs"].astype(float)
    df.loc[(df.subject_id == "S20") & (df.time == 3), "bpvs"] = value
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_rlm_wave_battery(wave=3, path=path)


def test_rlm_wave_wide_rejects_fractional_group_code(tmp_path):
    """#631 finding 5: ``astype(int)`` truncated a raw 1.5 into a valid-looking
    cohort before the unknown-code check could see it; the shared wide-frame
    loader must validate the raw codes first (exercised via the battery)."""
    path = _write_battery_csv(tmp_path)
    df = pd.read_csv(path)
    df["readgrp"] = df["readgrp"].astype(float)
    df.loc[df.subject_id == "S20", "readgrp"] = 1.5
    df.to_csv(path, index=False)
    with pytest.raises(
        ValueError, match=r"reading-group code\(s\) at wave 3.*1\.5"
    ):
        load_rlm_wave_battery(wave=3, path=path)


def test_build_rlm_adjusted_and_horseshoe(tmp_path):
    path = _write_battery_csv(tmp_path)
    frame = load_rlm_span_frame(path=path)

    adj = build_rlm_adjusted_model(frame)
    names = {v.name for v in adj.model.free_RVs}
    # The concentration takes the dispersion-scale prior of the RLM historical
    # families (2026-08-22 adjusted review, finding 4): the free RV is
    # inv_sqrt_kappa and kappa is its Deterministic re-expression.
    assert {"alpha", "gamma_own", "inv_sqrt_kappa"}.issubset(names)
    assert "kappa" not in names
    assert "kappa" in {v.name for v in adj.model.deterministics}
    assert {f"beta_{k}" for k in frame.predictors}.issubset(names)
    # Group nuisance: exactly two dummies (three groups, largest = reference).
    assert sum(n.startswith("beta_group_nuisance_") for n in names) == 2

    hs = build_rlm_horseshoe_model(frame)
    hs_names = {v.name for v in hs.model.free_RVs}
    assert {"hs_tau", "hs_c2", "hs_lambda", "hs_z"}.issubset(hs_names)
    assert "beta" in {v.name for v in hs.model.deterministics}
    # The horseshoe partner shares the adjusted fit's frame and, since
    # 2026-08-22, its dispersion-scale concentration prior.
    assert "inv_sqrt_kappa" in hs_names
    assert "kappa" not in hs_names
    assert "kappa" in {v.name for v in hs.model.deterministics}

    with adj.model:
        pp = pm.sample_prior_predictive(draws=3, random_seed=1)
    assert pp.prior_predictive["y_post"].shape[-1] == frame.n_obs


def test_transition_frame_and_factory_preserve_child_loo_unit(tmp_path):
    path = _write_battery_csv(tmp_path, waves=(1, 2, 3, 4, 5))
    df = pd.read_csv(path)
    required = ["basread", "bpvs", "trog", "basdig", "bassim", "age"]
    df.loc[(df["time"] == 5) & (df["readgrp"] != 1), required] = np.nan
    df.to_csv(path, index=False)

    frame = load_rlm_transition_frame(path=path)
    assert frame.transition_waves == (1, 2, 3, 4, 5)
    assert frame.transition_n_obs == {
        "w1->w2": 12,
        "w2->w3": 12,
        "w3->w4": 12,
        "w4->w5": 4,
    }
    assert frame.transition_group_counts["w4->w5"] == {1: 4}
    assert (frame.n_obs, frame.n_children) == (40, 12)
    assert len(np.unique(frame.child_idx)) == frame.n_children
    for values in frame.predictors.values():
        for phase in range(frame.n_phases):
            assert abs(float(np.mean(values[frame.phase == phase]))) < 1e-8

    built = build_rlm_transition_adjusted_model(frame)
    names = {variable.name for variable in built.model.free_RVs}
    assert {
        "alpha_transition", "gamma_own", "sigma_child", "inv_sqrt_kappa"
    }.issubset(names)
    assert "kappa" in {v.name for v in built.model.deterministics}
    assert {f"beta_{key}" for key in frame.predictors}.issubset(names)
    assert "loo_child_idx" in built.model.named_vars
    with built.model:
        prior = pm.sample_prior_predictive(draws=3, random_seed=1)
    assert prior.prior_predictive["y_post"].shape[-1] == frame.n_obs

    varying = build_rlm_transition_adjusted_model(frame, varying_slopes=True)
    varying_names = {variable.name for variable in varying.model.free_RVs}
    assert "beta_transition" in varying_names
    assert not any(name.startswith("beta_bpvs") for name in varying_names)


def test_rlm_adjusted_factories_take_the_dispersion_scale_and_own_baseline_priors(
    tmp_path,
):
    """2026-08-22 adjusted-family review, findings 4 and 5.

    Both Byrne adjusted factories put the Beta-Binomial concentration on the
    dispersion scale (``inv_sqrt_kappa ~ HalfNormal(dispersion_prior_sigma)``,
    ``kappa`` a Deterministic) exactly as the RLM historical factories do, and
    expose the own-baseline prior SD for the family's 0.25-vs-0.5 sweep.
    """
    from pymc.printing import str_for_dist

    path = _write_battery_csv(tmp_path, waves=(1, 2, 3, 4, 5))
    span = load_rlm_span_frame(path=path)
    transition = load_rlm_transition_frame(path=path, transition_waves=(1, 2, 3))
    for built in (
        build_rlm_adjusted_model(span, gamma_own_sigma=0.5, dispersion_prior_sigma=0.4),
        build_rlm_transition_adjusted_model(
            transition, gamma_own_sigma=0.5, dispersion_prior_sigma=0.4
        ),
    ):
        free = {v.name: v for v in built.model.free_RVs}
        assert "kappa" not in free
        assert "0.4" in str_for_dist(free["inv_sqrt_kappa"], formatting="plain")
        assert "0.5" in str_for_dist(free["gamma_own"], formatting="plain")
        assert "kappa" in {v.name for v in built.model.deterministics}
        with built.model:
            prior = pm.sample_prior_predictive(draws=50, random_seed=2)
        kappa = prior.prior["kappa"].values.ravel()
        inv = prior.prior["inv_sqrt_kappa"].values.ravel()
        np.testing.assert_allclose(kappa, 1.0 / (inv**2 + 1e-6))


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
    assert {"factor_corr_chol", "communality_free"}.issubset(names)
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
    # The sampled dispersion parameter is 1/sqrt(kappa) since the 2026-08-21
    # review (finding 8); kappa stays available as the interpretable
    # Deterministic the reports and diagnostics use.
    assert {
        "eta_cell",
        "sigma_subject",
        "inv_sqrt_kappa",
        "measure_corr_chol",
        "z_subject",
    }.issubset(names)
    assert "kappa" in {v.name for v in built.model.deterministics}
    # Group-indexed scales per measure: (measure, group) = (3, 3).
    assert built.model.named_vars["sigma_subject"].eval().shape == (3, 3)
    assert built.model.named_vars["kappa"].eval().shape == (3, 3)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=1)
    corr = pp.prior["measure_corr"].values[0]
    assert np.allclose(np.diagonal(corr, axis1=-2, axis2=-1), 1.0)
    assert np.allclose(corr, np.swapaxes(corr, -1, -2))


def test_build_rlm_joint_growth_within_child_layer_is_double_centred(tmp_path):
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
        panel,
        measures=("basread", "bpvs", "basdig"),
        within_correlation=True,
    )
    names = {v.name for v in built.model.free_RVs}
    assert {"sigma_within", "within_corr_chol", "z_within"}.issubset(names)
    assert "kappa" not in names
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=2)

    corr = pp.prior["within_corr"].values[0]
    assert np.allclose(np.diagonal(corr, axis1=-2, axis2=-1), 1.0)
    assert np.allclose(corr, np.swapaxes(corr, -1, -2))

    offsets = pp.prior["within_offset"].values[0]
    subject_col = panel.dataset.subject_col
    group_col = panel.dataset.group_col
    wave_col = panel.dataset.wave_col
    for _subject, idx in panel.long.groupby(subject_col).indices.items():
        assert np.allclose(offsets[:, idx, :].mean(axis=1), 0.0, atol=1e-6)
    for _cell, idx in panel.long.groupby([group_col, wave_col]).indices.items():
        assert np.allclose(offsets[:, idx, :].mean(axis=1), 0.0, atol=1e-6)


_PHASE_BD_SPECS = {
    "lrp-rlm-adj-001": ("adjusted", "basread"),
    "lrp-rlm-adj-003": ("adjusted", "bpvs"),
    "lrp-rlm-adj-004": ("adjusted", "trog"),
    "lrp-rlm-adj-005": ("adjusted", "basdig"),
    "lrp-rlm-adj-006": ("adjusted", "basread"),
    "lrp-rlm-hs-001": ("horseshoe", "basread"),
    "lrp-rlm-hs-002": ("horseshoe", "bpvs"),
    "lrp-rlm-hs-003": ("horseshoe", "trog"),
    "lrp-rlm-mm-001": ("corr_factor", None),
    "lrp-rlm-jc-001": ("historical_joint", None),
    "lrp-rlm-jc-002": ("historical_joint", None),
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
    assert spec.design == (
        "historical_stacked_transitions"
        if model_id == "lrp-rlm-adj-006"
        else "historical_cohort"
    )


def test_span_frame_rejects_fractional_and_out_of_range_counts(tmp_path):
    """2026-08-21 review, finding 9: a fractional count was silently truncated by
    the factory's int64 cast and an out-of-range predictor surfaced only as an
    opaque NaN sampler failure."""
    path = _write_battery_csv(tmp_path)
    df = pd.read_csv(path)
    frac_mask = (df.subject_id == "S10") & (df.time == 1)
    df["bpvs"] = df["bpvs"].astype(float)
    df.loc[frac_mask, "bpvs"] = 3.5
    frac = tmp_path / "frac.csv"
    df.to_csv(frac, index=False)
    with pytest.raises(ValueError, match="integer counts"):
        load_rlm_span_frame(path=frac)

    df2 = pd.read_csv(path)
    df2.loc[frac_mask, "trog"] = 10_000
    over = tmp_path / "over.csv"
    df2.to_csv(over, index=False)
    with pytest.raises(ValueError, match="outside 0"):
        load_rlm_span_frame(path=over)
