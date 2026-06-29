# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`statistical_models.preprocessing`."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
    unconfirmed_ceilings,
)
from language_reading_predictors.statistical_models.preprocessing import (
    logit_safe,
    standardise,
    load_and_prepare,
    load_and_prepare_aligned,
    load_and_prepare_lagged_outcome,
)


def test_logit_safe_haldane_correction():
    # y=0 with N=10 -> log((0.5)/(10.5)) = log(1/21)
    assert logit_safe(np.array([0]), 10)[0] == pytest.approx(np.log(0.5 / 10.5))
    # y=N -> symmetric
    assert logit_safe(np.array([10]), 10)[0] == pytest.approx(np.log(10.5 / 0.5))
    # monotone increasing
    ys = np.arange(11)
    vals = logit_safe(ys, 10)
    assert np.all(np.diff(vals) > 0)


def test_standardise_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5, scale=3, size=200)
    z, scaler = standardise(x)
    assert z.mean() == pytest.approx(0.0, abs=1e-10)
    assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-10)
    back = scaler.inverse(z)
    assert np.allclose(back, x)


def _make_synthetic_long(n_children: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_children):
        sid = f"S{i:03d}"
        # integer age in months
        age_base = int(rng.integers(60, 110))
        mumedu = int(rng.integers(0, 8))
        dadedu = int(rng.integers(0, 8))
        agebooks = int(rng.integers(0, 48))
        for t in (1, 2, 3, 4):
            row = {
                V.SUBJECT_ID: sid,
                V.TIME: t,
                V.GROUP: int(rng.integers(1, 3)),
                V.AGE: age_base + 6 * (t - 1),
                V.MUMEDUPOST16: mumedu,
                V.DADEDUPOST16: dadedu,
                V.AGEBOOKS: agebooks,
            }
            for s in ITT_OUTCOMES:
                m = MEASURES[s]
                row[m.column] = int(rng.integers(0, m.n_trials + 1))
            row[V.NONWORD] = int(rng.integers(0, 7))
            rows.append(row)
    return pd.DataFrame(rows)


def test_load_and_prepare_itt(tmp_path):
    df = _make_synthetic_long(n_children=20, seed=1)
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no rows should be dropped
        prep = load_and_prepare(path=p, phase_mode="itt")
    assert prep.n_obs == 20
    assert prep.n_phases == 1
    assert set(prep.pre_logit) == set(ITT_OUTCOMES)
    for s in ITT_OUTCOMES:
        assert prep.pre_logit[s].shape == (20,)
        assert prep.post_counts[s].shape == (20,)
    assert np.issubdtype(prep.G.dtype, np.integer)
    assert set(np.unique(prep.G)).issubset({0, 1})


def test_group_recode_intervention_is_positive(tmp_path):
    """Sign-convention guard (positive = intervention benefit).

    The immediate-intervention arm (``group == 1``) must recode to ``G == 1``
    and the wait-list control (``group == 2``) to ``G == 0``, so every
    coefficient on ``G`` (tau, tau_i/tau_k, beta_G, b_G, b_GM, a_G) reads
    *positive* when the intervention raises the outcome. Guards the
    ``G = 2 - group`` recode in :func:`load_and_prepare` against silent
    reversal. See the "Sign convention" section of METHODS.md.
    """
    for group_code, expected_g in ((1, 1), (2, 0)):
        df = _make_synthetic_long(n_children=8, seed=group_code)
        df[V.GROUP] = group_code
        p = tmp_path / f"rli_group{group_code}.csv"
        df.to_csv(p, index=False)
        prep = load_and_prepare(path=p, phase_mode="itt")
        assert set(np.unique(prep.G)) == {expected_g}, (
            f"group=={group_code} must recode to G=={expected_g} "
            "(positive = intervention benefit)"
        )


def test_load_and_prepare_all_phases(tmp_path):
    df = _make_synthetic_long(n_children=15, seed=2)
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    prep = load_and_prepare(path=p, phase_mode="all")
    assert prep.n_phases == 3
    assert prep.n_obs == 15 * 3
    phase_counts = np.bincount(prep.phase, minlength=3)
    assert (phase_counts == 15).all()


def test_load_and_prepare_span_one_row_per_child(tmp_path):
    df = _make_synthetic_long(n_children=18, seed=5)
    # Block design is a t1-only baseline (NaN at later waves), like the real data.
    df[V.BLOCKS] = np.nan
    df.loc[df[V.TIME] == 1, V.BLOCKS] = np.arange(18, dtype=float)
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    prep = load_and_prepare(
        path=p, phase_mode="span", post_time=4, covariates=(V.BLOCKS,)
    )
    # One row per child; the t1-only block design survives (span pre = t1).
    assert prep.phase_mode == "span"
    assert prep.n_phases == 1
    assert prep.n_obs == 18
    assert prep.n_children == 18
    assert V.BLOCKS in prep.covariates
    assert prep.covariates[V.BLOCKS].shape == (18,)
    assert prep.covariates[V.BLOCKS].mean() == pytest.approx(0.0, abs=1e-9)


def test_load_and_prepare_drops_missing_pre(tmp_path):
    df = _make_synthetic_long(n_children=10, seed=3)
    # Introduce missing pre-score for one child at t=1.
    df.loc[(df[V.SUBJECT_ID] == "S000") & (df[V.TIME] == 1), V.EWRSWR] = np.nan
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        prep = load_and_prepare(path=p, phase_mode="itt")
    assert prep.n_obs == 9
    assert prep.dropped_rows == 1
    assert any("dropped" in str(w.message) for w in ws)


def test_load_and_prepare_covariates_are_standardised(tmp_path):
    df = _make_synthetic_long(n_children=10, seed=4)
    df.loc[(df[V.SUBJECT_ID] == "S000") & (df[V.TIME] == 1), V.AGEBOOKS] = np.nan
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        prep = load_and_prepare(
            path=p,
            phase_mode="itt",
            covariates=(V.MUMEDUPOST16, V.DADEDUPOST16, V.AGEBOOKS),
        )

    assert prep.n_obs == 9
    assert prep.dropped_rows == 1
    assert set(prep.covariates) == {V.MUMEDUPOST16, V.DADEDUPOST16, V.AGEBOOKS}
    for z in prep.covariates.values():
        assert z.mean() == pytest.approx(0.0, abs=1e-10)
        assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-10)
    assert any("dropped" in str(w.message) for w in ws)


def test_lagged_outcome_takes_outcome_from_later_wave(tmp_path):
    """The temporal-ordering sensitivity prep (issue #84): outcome post-counts
    come from the later wave, while baselines, treatment, and the mediator's
    post (t2) are exactly the ITT design."""
    df = _make_synthetic_long(n_children=20, seed=5)
    # Make the outcome (W = ewrswr) distinguishable by wave.
    df.loc[df[V.TIME] == 2, V.EWRSWR] = 10
    df.loc[df[V.TIME] == 3, V.EWRSWR] = 40
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)

    itt = load_and_prepare(path=p, phase_mode="itt")
    lagged = load_and_prepare_lagged_outcome("W", outcome_time=3, path=p)

    assert np.all(itt.post_counts["W"] == 10)  # ITT outcome is t2
    assert np.all(lagged.post_counts["W"] == 40)  # sensitivity outcome is t3
    # Mediator post (L) stays at t2; baselines / treatment / age are unchanged.
    assert np.array_equal(lagged.post_counts["L"], itt.post_counts["L"])
    assert np.allclose(lagged.pre_logit["W"], itt.pre_logit["W"])
    assert np.array_equal(lagged.G, itt.G)
    assert np.allclose(lagged.A_std, itt.A_std)
    assert lagged.n_obs == itt.n_obs


def test_lagged_outcome_rejects_randomised_wave(tmp_path):
    df = _make_synthetic_long(n_children=8, seed=6)
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="post-RCT wave"):
        load_and_prepare_lagged_outcome("W", outcome_time=2, path=p)


def test_lagged_outcome_missing_later_wave_is_nan(tmp_path):
    """A subject present in the ITT base but missing the later-wave outcome gets
    a NaN post-count (the mediation factory then drops that row)."""
    df = _make_synthetic_long(n_children=10, seed=7)
    df.loc[(df[V.SUBJECT_ID] == "S000") & (df[V.TIME] == 3), V.EWRSWR] = np.nan
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    lagged = load_and_prepare_lagged_outcome("W", outcome_time=3, path=p)
    assert np.isnan(lagged.post_counts["W"]).sum() == 1


def test_measure_ceilings_documented():
    """#80: ITT-outcome ceilings are documented (W=79, P=92). The only
    unconfirmed ceilings are the not-taught taught-vocabulary measures (UE/UR),
    whose item counts are not tabulated in the paper (LRP74-76; pending the data
    dictionary) — see ``measures.py`` and the LRP74-76 note."""
    assert set(unconfirmed_ceilings()) == {"UE", "UR"}
    assert MEASURES["W"].n_trials == 79
    assert MEASURES["P"].n_trials == 92


def test_load_and_prepare_rejects_count_above_ceiling(tmp_path):
    """#80: a count above n_trials raises a clear error naming the measure."""
    df = _make_synthetic_long(n_children=10, seed=5)
    over = MEASURES["W"].n_trials + 5
    df.loc[df[V.SUBJECT_ID] == "S000", V.EWRSWR] = over
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match=r"ewrswr.*ceiling"):
        load_and_prepare(path=p, phase_mode="itt")


# ---------------------------------------------------------------------------
# levels phase mode + baseline_covariates broadcast (LRPLF / LRPGF, issue #127)
# ---------------------------------------------------------------------------


def test_load_and_prepare_levels(tmp_path):
    """``levels`` yields one row per (child, timepoint t1-t4): four phases, the
    score at each timepoint as the post outcome, and no autoregressive pre."""
    df = _make_synthetic_long(n_children=15, seed=6)
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    prep = load_and_prepare(path=p, phase_mode="levels")
    assert prep.n_phases == 4
    assert prep.n_obs == 15 * 4
    assert (np.bincount(prep.phase, minlength=4) == 15).all()
    assert prep.pre_logit == {}  # levels has no pre-score
    assert set(prep.post_counts) == set(ITT_OUTCOMES)
    assert set(np.unique(prep.G)).issubset({0, 1})


def test_load_and_prepare_levels_rejects_pre_required(tmp_path):
    df = _make_synthetic_long(n_children=8, seed=7)
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="pre_required"):
        load_and_prepare(path=p, phase_mode="levels", pre_required=("W",))


def test_baseline_covariate_broadcast_from_t1(tmp_path):
    """A t1-only baseline (here AGEBOOKS, blanked at t2-t4) is broadcast across
    every row by subject merge — so no rows drop and the covariate is finite and
    standardised on every (child, phase) row. This is what lets the gain model
    use t1 ability/SES in ``all`` mode without collapsing to one phase."""
    for mode, n_phases in (("all", 3), ("levels", 4)):
        df = _make_synthetic_long(n_children=12, seed=8)
        df.loc[df[V.TIME] != 1, V.AGEBOOKS] = np.nan  # genuine t1-only baseline
        p = tmp_path / f"rli_{mode}.csv"
        df.to_csv(p, index=False)
        prep = load_and_prepare(
            path=p, phase_mode=mode, baseline_covariates=(V.AGEBOOKS,)
        )
        assert prep.n_obs == 12 * n_phases  # broadcast -> no rows dropped
        z = prep.covariates[V.AGEBOOKS]
        assert np.all(np.isfinite(z))  # filled on every row
        assert z.mean() == pytest.approx(0.0, abs=1e-10)  # standardised
        assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-10)


def test_load_and_prepare_rejects_covariate_baseline_overlap(tmp_path):
    """A column listed as both a per-row covariate and a t1 baseline_covariate is
    rejected up front (#127/#128): the baseline merge would suffix the duplicate
    (``_x``/``_y``) and the later lookup would ``KeyError``, so the loader fails fast
    with a clear message naming the offending column."""
    df = _make_synthetic_long(n_children=8, seed=9)
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="exactly one role"):
        load_and_prepare(
            path=p,
            phase_mode="all",
            covariates=(V.AGEBOOKS,),
            baseline_covariates=(V.AGEBOOKS,),
        )


# ---------------------------------------------------------------------------
# load_and_prepare_aligned: onset-aligned per-protocol single gain (LRPAL)
# ---------------------------------------------------------------------------


def test_load_and_prepare_aligned(tmp_path):
    """One row per child, no phases, group recoded to the intervention indicator."""
    df = _make_synthetic_long(n_children=20, seed=21)
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    prep = load_and_prepare_aligned(path=p)
    assert prep.phase_mode == "aligned"
    assert prep.n_phases == 1
    assert prep.n_obs == prep.n_children == 20
    assert bool((prep.phase == 0).all())
    assert set(np.unique(prep.G)).issubset({0, 1})
    assert set(prep.post_counts) == set(ITT_OUTCOMES)
    assert set(prep.pre_logit) == set(ITT_OUTCOMES)


def test_load_and_prepare_aligned_onset_windows(tmp_path):
    """The crux: the immediate arm's aligned post is t3, the wait-list arm's is t4.

    With each score set equal to its wave index, the loaded post-count reveals
    which wave the aligned window ended on per arm.
    """
    rows = []
    for i in range(8):
        grp = 1 if i % 2 == 0 else 2  # alternate immediate / wait-list
        for t in (1, 2, 3, 4):
            row = {
                V.SUBJECT_ID: f"S{i:03d}", V.TIME: t, V.GROUP: grp,
                V.AGE: 80 + 6 * (t - 1),
            }
            for s in ITT_OUTCOMES:
                row[MEASURES[s].column] = t  # score == wave index
            rows.append(row)
    p = tmp_path / "rli.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    prep = load_and_prepare_aligned(path=p)
    post = prep.post_counts["W"]
    # G == 1 immediate -> aligned post from t3 (==3); G == 0 wait-list -> t4 (==4)
    assert set(post[prep.G == 1].astype(int)) == {3}
    assert set(post[prep.G == 0].astype(int)) == {4}


def test_load_and_prepare_aligned_ability_merged_from_t1(tmp_path):
    """Ability (a t1-only baseline) is merged from t1 for BOTH arms, so even the
    wait-list arm (onset t2) gets a finite, standardised value and no row drops."""
    df = _make_synthetic_long(n_children=12, seed=22)
    df[V.BLOCKS] = np.nan
    t1 = df[V.TIME] == 1
    df.loc[t1, V.BLOCKS] = np.linspace(10.0, 40.0, int(t1.sum()))  # t1-only, varying
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    prep = load_and_prepare_aligned(path=p, ability_covariate=V.BLOCKS)
    assert prep.n_obs == 12  # nothing dropped despite blocks blank at t2-t4
    z = prep.covariates[V.BLOCKS]
    assert np.all(np.isfinite(z))  # filled for every child incl. wait-list onset rows
    assert z.mean() == pytest.approx(0.0, abs=1e-10)


def test_load_and_prepare_aligned_requires_dose_when_requested(tmp_path):
    """Dose variants drop rows with missing aligned-window cumulative sessions."""
    df = _make_synthetic_long(n_children=12, seed=23)
    dose_by_child = {
        sid: 80.0 + i for i, sid in enumerate(sorted(df[V.SUBJECT_ID].unique()))
    }
    df[V.ATTEND_CUMUL] = df[V.SUBJECT_ID].map(dose_by_child)
    sid = df[V.SUBJECT_ID].iloc[0]
    grp = int(df.loc[df[V.SUBJECT_ID] == sid, V.GROUP].iloc[0])
    aligned_post_t = 3 if grp == 1 else 4
    df.loc[
        (df[V.SUBJECT_ID] == sid) & (df[V.TIME] == aligned_post_t),
        V.ATTEND_CUMUL,
    ] = np.nan
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)

    with pytest.warns(UserWarning, match="dropped 1"):
        prep = load_and_prepare_aligned(path=p, include_dose=True)

    assert prep.n_obs == 11
    assert "dose" in prep.covariates
    assert np.all(np.isfinite(prep.covariates["dose"]))
