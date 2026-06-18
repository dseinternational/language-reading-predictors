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
    build_adjusted_model,
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
        blocks = int(rng.integers(5, 40))  # block design — a single t1 assessment
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
            # Block design is t1-only (NaN at later waves); behaviour is t1-t3.
            if t == 1:
                row[V.BLOCKS] = blocks
            if t in (1, 2, 3):
                row[V.BEHAV] = float(rng.integers(1, 6))
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


def test_mechanism_factory_adjusts_for_age_linearly(tmp_path):
    """Age in the adjustment set enters eta as a linear ``gamma_A`` term when the
    age GP is off. Regression test for the silent age-drop bug that left
    LRP56-58 / LRP71 / LRP72 unadjusted for the age confounder."""
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="all")
    built = build_mechanism_model(
        prep,
        mechanism_symbol="L",
        outcome_symbol="W",
        confounder_symbols=("G", "A"),
        use_age_gp=False,
    )
    names = {v.name for v in built.model.free_RVs}
    assert "gamma_A" in names
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=4)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_mechanism_factory_age_moderator_not_double_counted(tmp_path):
    """When age is the moderator (LRP73), the moderator main effect
    ``gamma_mod * z(age)`` represents age, so no separate ``gamma_A`` is added —
    the two would be collinear. Mirrors the pipeline, which strips the moderator
    from ``confounder_symbols``."""
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="all")
    built = build_mechanism_model(
        prep,
        mechanism_symbol="L",
        outcome_symbol="W",
        confounder_symbols=("G",),
        moderator_symbol="A",
        moderator_is_covariate=True,
        use_age_gp=False,
    )
    names = {v.name for v in built.model.free_RVs}
    assert "gamma_mod" in names
    assert "gamma_A" not in names


def test_mechanism_factory_age_gp_skips_linear_term(tmp_path):
    """With the age GP on, age is represented by ``f_A``, so the linear
    ``gamma_A`` term is not added (no double adjustment)."""
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="all")
    built = build_mechanism_model(
        prep,
        mechanism_symbol="L",
        outcome_symbol="W",
        confounder_symbols=("G", "A"),
        use_age_gp=True,
    )
    names = {v.name for v in built.model.free_RVs}
    assert "gamma_A" not in names


def test_adjusted_factory_builds(tmp_path):
    """Between-child adjusted build: standardised T1 predictors, no random intercept."""
    p = _write_synthetic(tmp_path, n_children=30)
    prep = load_and_prepare(
        path=p,
        phase_mode="span",
        post_time=4,
        outcomes=("W", "L", "B", "R", "E", "F"),
        covariates=(V.BLOCKS, V.BEHAV),
    )
    built = build_adjusted_model(
        prep,
        outcome_symbol="W",
        predictors=["L", "lang", "B", "age", V.BLOCKS, V.BEHAV],
    )
    names = {v.name for v in built.model.free_RVs}
    assert {
        "beta_L", "beta_lang", "beta_B", "beta_age",
        f"beta_{V.BLOCKS}", f"beta_{V.BEHAV}",
    }.issubset(names)
    assert "gamma_own" in names
    # Genuinely between-child: no phase intercept, no child random intercept.
    assert "alpha_phase" not in names
    assert "sigma_child" not in names and "u_child_raw" not in names
    assert built.prepared.n_phases == 1
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=4)
    assert pp.prior_predictive["y_post"].shape[-1] == built.prepared.n_obs


def test_adjusted_factory_bivariate_single_predictor(tmp_path):
    """A single-element predictor list yields just that one slope (bivariate fit)."""
    p = _write_synthetic(tmp_path, n_children=25)
    prep = load_and_prepare(
        path=p, phase_mode="span", outcomes=("W", "L", "B", "R", "E", "F")
    )
    built = build_adjusted_model(prep, predictors=["lang"])
    betas = {v.name for v in built.model.free_RVs if v.name.startswith("beta_")}
    assert betas == {"beta_lang"}


def test_adjusted_factory_rejects_pooled_phase(tmp_path):
    """The between-child factory must refuse pooled all-phase data."""
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="all")
    with pytest.raises(ValueError):
        build_adjusted_model(prep, predictors=["L"])
