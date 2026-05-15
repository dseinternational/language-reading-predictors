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
)
from language_reading_predictors.statistical_models.preprocessing import (
    logit_safe,
    standardise,
    load_and_prepare,
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


def test_load_and_prepare_all_phases(tmp_path):
    df = _make_synthetic_long(n_children=15, seed=2)
    p = tmp_path / "rli.csv"
    df.to_csv(p, index=False)
    prep = load_and_prepare(path=p, phase_mode="all")
    assert prep.n_phases == 3
    assert prep.n_obs == 15 * 3
    phase_counts = np.bincount(prep.phase, minlength=3)
    assert (phase_counts == 15).all()


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
