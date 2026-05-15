# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Smoke tests for the model factories on synthetic data.

These tests only check that each factory *builds* and can draw a small prior
predictive sample. Full posterior-sampling correctness is validated by the
end-to-end fits in ``scripts/fit_statistical_model.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.factories import (
    build_itt_model,
    build_joint_model,
    build_mechanism_model,
)
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)


def _write_synthetic(tmp_path, n_children: int = 25, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_children):
        sid = f"S{i:03d}"
        age_base = int(rng.integers(60, 110))
        g = int(rng.integers(1, 3))
        mumedu = int(rng.integers(0, 8))
        dadedu = int(rng.integers(0, 8))
        agebooks = int(rng.integers(0, 48))
        for t in (1, 2, 3, 4):
            row = {
                V.SUBJECT_ID: sid,
                V.TIME: t,
                V.GROUP: g,
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
    p = tmp_path / "rli.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


@pytest.mark.parametrize("outcome", ["W", "R", "E"])
def test_itt_factory_builds(tmp_path, outcome):
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_itt_model(prep, outcome_symbol=outcome)
    assert any(v.name == "tau" for v in built.model.free_RVs)
    with built.model:
        pp = pm.sample_prior_predictive(draws=10, random_seed=1)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_itt_factory_builds_with_adjusters(tmp_path):
    p = _write_synthetic(tmp_path)
    adjusters = (V.MUMEDUPOST16, V.DADEDUPOST16, V.AGEBOOKS)
    prep = load_and_prepare(path=p, phase_mode="itt", covariates=adjusters)
    built = build_itt_model(prep, outcome_symbol="W", adjust_for=adjusters)
    names = {v.name for v in built.model.free_RVs}
    assert {f"gamma_{c}" for c in adjusters}.issubset(names)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=11)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_joint_factory_builds(tmp_path):
    """Default build: no LKJ residual (dropped 2026-04-18)."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_joint_model(prep)
    names = {v.name for v in built.model.free_RVs}
    assert "tau" in names
    assert "u_chol" not in names  # LKJ residual off by default
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=2)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs * len(ITT_OUTCOMES)


def test_joint_factory_residual_correlation_flag(tmp_path):
    """Opt-in LKJ residual adds u_chol; sigma_outcome is the LKJ-derived SD."""
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_joint_model(prep, use_residual_correlation=True)
    free = {v.name for v in built.model.free_RVs}
    dets = {v.name for v in built.model.deterministics}
    assert "u_chol" in free
    # sigma_outcome is now a Deterministic alias of the LKJCholeskyCov SDs
    # (the previous double-scaled HalfNormal was dropped).
    assert "sigma_outcome" in dets
    assert "u_corr" in dets


def test_mechanism_factory_builds(tmp_path):
    """Default mechanism build: includes subject random intercept."""
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="all")
    built = build_mechanism_model(
        prep,
        mechanism_symbol="R",
        outcome_symbol="W",
        confounder_symbols=(),
    )
    names = {v.name for v in built.model.free_RVs}
    assert "beta_G" in names
    assert "sigma_child" in names  # on by default
    assert "u_child_raw" in names
    assert any(v.name == "f_mech" for v in built.model.deterministics)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=3)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_mechanism_factory_without_random_intercept(tmp_path):
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="all")
    built = build_mechanism_model(
        prep,
        mechanism_symbol="R",
        outcome_symbol="W",
        confounder_symbols=(),
        use_subject_random_intercept=False,
    )
    names = {v.name for v in built.model.free_RVs}
    assert "sigma_child" not in names
    assert "u_child_raw" not in names


def test_itt_factory_rejects_wrong_phase(tmp_path):
    p = _write_synthetic(tmp_path, n_children=10)
    prep = load_and_prepare(path=p, phase_mode="all")
    with pytest.raises(ValueError):
        build_itt_model(prep, outcome_symbol="W")
