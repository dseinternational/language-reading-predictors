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
    build_aligned_model,
    build_correlated_factor_model,
    build_did_model,
    build_dose_response_model,
    build_gain_factors_model,
    build_itt_model,
    build_joint_model,
    build_level_factors_model,
    build_mechanism_model,
    build_mediation_model,
    build_two_mediator_model,
)
from language_reading_predictors.statistical_models import priors
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
    TAUGHT_BLOCK1_OUTCOMES,
    is_distal,
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
            for s in (*ITT_OUTCOMES, *TAUGHT_BLOCK1_OUTCOMES):
                m = MEASURES[s]
                row[m.column] = int(rng.integers(0, m.n_trials + 1))
            row[V.NONWORD] = int(rng.integers(0, 7))
            # Intervention dose covariates (used by the dose-response factory).
            row[V.ATTEND] = int(rng.integers(0, 90))
            row[V.ATTEND_CUMUL] = int(rng.integers(0, 200))
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


def test_taught_block1_measures_registered():
    """The taught-vocabulary family is defined with honest denominators and kept
    out of ITT_OUTCOMES (so the eight-outcome joint model stays stable)."""
    assert TAUGHT_BLOCK1_OUTCOMES == ("TE", "TR", "UE", "UR")
    assert not set(TAUGHT_BLOCK1_OUTCOMES) & set(ITT_OUTCOMES)
    assert MEASURES["TE"].n_trials == 24 and MEASURES["TE"].n_trials_confirmed
    assert MEASURES["TR"].n_trials == 24 and MEASURES["TR"].n_trials_confirmed
    # Not-taught sets are the half-size 3x4 control (12 items), confirmed (#214).
    assert MEASURES["UE"].n_trials == 12 and MEASURES["UE"].n_trials_confirmed
    assert MEASURES["UR"].n_trials == 12 and MEASURES["UR"].n_trials_confirmed


def test_itt_factory_taught_outcome_with_cross_symbols(tmp_path):
    """LRP74-style: an ITT outcome outside ITT_OUTCOMES, conditioned only on its
    own baseline plus a chosen cross-baseline (the matched standardised vocab)."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt", outcomes=("TE", "E"))
    built = build_itt_model(prep, outcome_symbol="TE", cross_symbols=("E",))
    names = {v.name for v in built.model.free_RVs}
    assert {"tau", "gamma_own", "gamma_E"}.issubset(names)
    # Only the requested cross-baseline enters - not all eight ITT baselines.
    assert "gamma_W" not in names and "gamma_R" not in names
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=12)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_itt_factory_default_cross_is_all_itt_outcomes(tmp_path):
    """Regression: with cross_symbols=None (the default), the LRP52-LRP54
    behaviour of conditioning on every other ITT outcome is preserved."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_itt_model(prep, outcome_symbol="W")  # cross_symbols defaults None
    names = {v.name for v in built.model.free_RVs}
    assert {f"gamma_{s}" for s in ITT_OUTCOMES if s != "W"}.issubset(names)


def test_itt_factory_rejects_unknown_cross_symbol(tmp_path):
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt", outcomes=("TE", "E"))
    with pytest.raises(KeyError):
        build_itt_model(prep, outcome_symbol="TE", cross_symbols=("E", "R"))


def test_joint_factory_two_outcome_taught_contrast(tmp_path):
    """LRP76-style: a two-outcome joint model (taught vs not-taught) with the
    within-child residual correlation block, which is identifiable at K=2."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(path=p, phase_mode="itt", outcomes=("TE", "UE"))
    built = build_joint_model(
        prep, outcomes=("TE", "UE"), use_residual_correlation=True
    )
    names = {v.name for v in built.model.free_RVs}
    assert "tau" in names and "u_chol" in names
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=13)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs * 2


def test_tau_difference_summary_contrast():
    """The taught-minus-not-taught difference parameter is summarised per draw
    from the joint posterior (unit test on a synthetic trace)."""
    import xarray as xr
    from types import SimpleNamespace

    from language_reading_predictors.statistical_models.reporting import (
        tau_difference_summary,
    )

    rng = np.random.default_rng(0)
    n_draws = 800
    te = rng.normal(0.8, 0.2, size=(1, n_draws))
    ue = rng.normal(0.1, 0.2, size=(1, n_draws))
    tau = xr.DataArray(
        np.stack([te, ue], axis=-1),
        dims=("chain", "draw", "outcome"),
        coords={"outcome": ["TE", "UE"]},
    )
    trace = SimpleNamespace(posterior=xr.Dataset({"tau": tau}))
    s = tau_difference_summary(trace, ["TE", "UE"], ("TE", "UE"), ci_prob=0.95)
    assert s["contrast"] == "TE_minus_UE"
    assert s["diff_logit_mean"] > 0.4
    assert 0.9 < s["prob_diff_pos"] <= 1.0
    assert s["diff_logit_lo"] < s["diff_logit_mean"] < s["diff_logit_hi"]


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


# ---------------------------------------------------------------------------
# LRPITT suite (#119): age-linear, own-baseline toggle, off-floor likelihood,
# tau-moderator plumbing
# ---------------------------------------------------------------------------


def test_itt_factory_age_linear_adds_gamma_A(tmp_path):
    """``use_age_linear`` adds a plain linear ``gamma_A`` age term (the LRPITT
    suite's age precision term, in place of the off-by-default age GP)."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_itt_model(
        prep,
        outcome_symbol="W",
        use_age_gp=False,
        use_own_baseline_gp=False,
        cross_symbols=(),
        use_age_linear=True,
    )
    assert "gamma_A" in {v.name for v in built.model.free_RVs}
    # Default (legacy LRP52 behaviour) has no linear age term.
    base = build_itt_model(
        prep, outcome_symbol="W", use_age_gp=False, use_own_baseline_gp=False
    )
    assert "gamma_A" not in {v.name for v in base.model.free_RVs}


def test_itt_factory_age_gp_and_linear_mutually_exclusive(tmp_path):
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    with pytest.raises(ValueError):
        build_itt_model(
            prep, outcome_symbol="W", use_age_gp=True, use_age_linear=True
        )


def test_itt_factory_age_only_drops_own_baseline(tmp_path):
    """``use_own_baseline=False`` drops ``gamma_own`` and never indexes
    ``pre_logit[own]`` — so a post-only / floored outcome can be modelled
    age-only even without a baseline."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt", outcomes=("N",))
    del prep.pre_logit["N"]  # simulate a genuinely post-only baseline
    built = build_itt_model(
        prep,
        outcome_symbol="N",
        use_age_gp=False,
        use_own_baseline_gp=False,
        cross_symbols=(),
        use_age_linear=True,
        use_own_baseline=False,
    )
    names = {v.name for v in built.model.free_RVs}
    assert "gamma_own" not in names
    assert {"tau", "gamma_A"}.issubset(names)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=21)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_itt_factory_bernoulli_offfloor(tmp_path):
    """The floor-rule PRIMARY: a Bernoulli on the binary off-floor indicator,
    age-only, with no ``kappa`` and a ``y_offfloor`` observed node."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt", outcomes=("N",))
    built = build_itt_model(
        prep,
        outcome_symbol="N",
        likelihood="bernoulli_offfloor",
        use_age_gp=False,
        use_own_baseline_gp=False,
        cross_symbols=(),
        use_age_linear=True,
        use_own_baseline=False,
    )
    names = {v.name for v in built.model.free_RVs}
    assert {"alpha", "tau", "gamma_A"}.issubset(names)
    assert "kappa" not in names and "gamma_own" not in names
    obs = {v.name for v in built.model.observed_RVs}
    assert obs == {"y_offfloor"}
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=22)
    yof = pp.prior_predictive["y_offfloor"].values
    assert set(np.unique(yof)).issubset({0, 1})


def test_itt_factory_bad_likelihood_raises(tmp_path):
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    with pytest.raises(ValueError):
        build_itt_model(prep, outcome_symbol="W", likelihood="poisson")


def test_itt_factory_tau_moderator_covariate(tmp_path):
    """Part B plumbing: an age tau-moderator adds ``gamma_tau_mod`` +
    ``gamma_tau_int``; the no-interaction baseline drops ``gamma_tau_int``."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_itt_model(
        prep,
        outcome_symbol="W",
        use_age_gp=False,
        use_own_baseline_gp=False,
        cross_symbols=(),
        use_age_linear=True,
        tau_moderator_symbol="A",
        tau_moderator_is_covariate=True,
    )
    assert {"gamma_tau_mod", "gamma_tau_int"}.issubset(
        {v.name for v in built.model.free_RVs}
    )
    base = build_itt_model(
        prep,
        outcome_symbol="W",
        use_age_gp=False,
        use_own_baseline_gp=False,
        cross_symbols=(),
        use_age_linear=True,
        tau_moderator_symbol="A",
        tau_moderator_is_covariate=True,
        tau_moderator_interaction=False,
    )
    bnames = {v.name for v in base.model.free_RVs}
    assert "gamma_tau_mod" in bnames and "gamma_tau_int" not in bnames


def test_itt_factory_tau_moderator_baseline(tmp_path):
    """A baseline-ability tau-moderator uses the pre-randomisation baseline logit
    ``pre_logit[symbol]`` and builds a valid model."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_itt_model(
        prep,
        outcome_symbol="W",
        use_age_gp=False,
        use_own_baseline_gp=False,
        cross_symbols=(),
        use_age_linear=True,
        tau_moderator_symbol="E",
    )
    assert {"gamma_tau_mod", "gamma_tau_int"}.issubset(
        {v.name for v in built.model.free_RVs}
    )
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=23)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def _assert_itt_diag_vars_subset(built, *, extra, adjust_for=(), likelihood="beta_binomial"):
    """``_itt_diag_vars`` must only name RVs the factory actually builds, else
    ``summary_diagnostics`` (``az.summary``) raises ``KeyError`` at diagnostics time."""
    from types import SimpleNamespace

    from language_reading_predictors.statistical_models.pipeline import _itt_diag_vars

    spec = SimpleNamespace(extra=extra)
    diag = _itt_diag_vars(spec, adjust_for, likelihood=likelihood)
    built_names = {v.name for v in built.model.free_RVs} | {
        v.name for v in built.model.deterministics
    }
    missing = set(diag) - built_names
    assert not missing, f"_itt_diag_vars names RVs the model never builds: {missing}"
    return set(diag)


def test_itt_diag_vars_match_graded_build(tmp_path):
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_itt_model(
        prep, outcome_symbol="W", use_age_gp=False, use_own_baseline_gp=False,
        cross_symbols=(),
    )
    diag = _assert_itt_diag_vars_subset(built, extra={})
    assert {"alpha", "tau", "gamma_own", "kappa"}.issubset(diag)


def test_itt_diag_vars_match_offfloor_age_only_build(tmp_path):
    # A desync here would silently crash diagnostics on the age-only /
    # bernoulli_offfloor floored fits (P, N) — the case this guards.
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt", outcomes=("N",))
    built = build_itt_model(
        prep, outcome_symbol="N", likelihood="bernoulli_offfloor",
        use_age_gp=False, use_own_baseline_gp=False, cross_symbols=(),
        use_age_linear=True, use_own_baseline=False,
    )
    diag = _assert_itt_diag_vars_subset(
        built,
        extra={"use_own_baseline": False, "use_age_linear": True},
        likelihood="bernoulli_offfloor",
    )
    assert {"alpha", "tau", "gamma_A"}.issubset(diag)
    assert "kappa" not in diag and "gamma_own" not in diag


def test_itt_diag_vars_match_tau_moderator_build(tmp_path):
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_itt_model(
        prep, outcome_symbol="W", use_age_gp=False, use_own_baseline_gp=False,
        cross_symbols=(), use_age_linear=True,
        tau_moderator_symbol="A", tau_moderator_is_covariate=True,
    )
    diag = _assert_itt_diag_vars_subset(
        built, extra={"use_age_linear": True, "tau_moderator_symbol": "A"}
    )
    assert {"gamma_tau_mod", "gamma_tau_int"}.issubset(diag)


def test_joint_factory_dag_faithful_flags(tmp_path):
    """The DAG-faithful joint (LRPITT12 / the generalisation contrasts) drops the
    cross-baseline matrix and adds a per-outcome linear age term, mirroring the
    single-outcome suite so the joint tau_k reproduce the single-outcome tau_k."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_joint_model(prep, use_cross_baselines=False, use_age_linear=True)
    names = {v.name for v in built.model.free_RVs}
    assert "gamma_A" in names  # per-outcome linear age
    assert "gamma_own" in names  # own baseline retained
    assert "gamma_cross" not in names  # cross-baseline matrix dropped
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=14)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs * len(ITT_OUTCOMES)


def test_joint_factory_cross_baselines_on_by_default(tmp_path):
    """Regression: the legacy LRP55 behaviour (cross-baseline matrix on, no linear
    age) is preserved by default."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built = build_joint_model(prep)
    names = {v.name for v in built.model.free_RVs}
    assert "gamma_cross" in names
    assert "gamma_A" not in names


def test_joint_factory_age_gp_and_linear_mutually_exclusive(tmp_path):
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="itt")
    with pytest.raises(ValueError):
        build_joint_model(prep, use_age_gp=True, use_age_linear=True)


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
# ---------------------------------------------------------------------------
# Dose-response factory (LRP77, #104 Phase 2)
# ---------------------------------------------------------------------------


def test_dose_response_factory_builds_period_varying(tmp_path):
    """Default build: partial-pooled per-period dose slopes + design adjusters."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(
        path=p, phase_mode="all", outcomes=("W",), covariates=("attend", "attend_cumul")
    )
    built = build_dose_response_model(prep, outcome_symbol="W", period_varying_dose=True)
    free = {v.name for v in built.model.free_RVs}
    dets = {v.name for v in built.model.deterministics}
    # period-varying dose slope (partial pooled), arm, age, dose-stage, subject RI
    assert {"mu_dose", "sigma_dose", "beta_dose_phase_raw", "beta_G", "gamma_A",
            "gamma_dose_stage", "sigma_child"}.issubset(free)
    assert "beta_dose_phase" in dets
    assert "beta_dose" not in free  # pooled slope only in the comparator
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=5)
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_dose_response_factory_pooled_slope(tmp_path):
    """``period_varying_dose=False`` gives a single pooled slope, no phase slopes."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(
        path=p, phase_mode="all", outcomes=("W",), covariates=("attend", "attend_cumul")
    )
    built = build_dose_response_model(prep, outcome_symbol="W", period_varying_dose=False)
    free = {v.name for v in built.model.free_RVs}
    assert "beta_dose" in free
    assert "mu_dose" not in free
    assert not any(v.name == "beta_dose_phase" for v in built.model.deterministics)


def test_dose_response_factory_ability_adjusters(tmp_path):
    """The sensitivity fit adds a ``gamma_{s}_pre`` term per baseline-skill symbol."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(
        path=p,
        phase_mode="all",
        outcomes=("W", "L", "E", "B"),
        covariates=("attend", "attend_cumul"),
    )
    built = build_dose_response_model(
        prep, outcome_symbol="W", ability_adjust_symbols=("L", "E", "B")
    )
    free = {v.name for v in built.model.free_RVs}
    assert {"gamma_L_pre", "gamma_E_pre", "gamma_B_pre"}.issubset(free)


def test_dose_response_factory_rejects_wrong_phase(tmp_path):
    p = _write_synthetic(tmp_path, n_children=12)
    prep = load_and_prepare(
        path=p, phase_mode="itt", outcomes=("W",), covariates=("attend", "attend_cumul")
    )
    with pytest.raises(ValueError):
        build_dose_response_model(prep, outcome_symbol="W")


def test_dose_response_factory_requires_dose_covariate(tmp_path):
    """A missing dose covariate is a clear KeyError, not a late failure."""
    p = _write_synthetic(tmp_path, n_children=12)
    prep = load_and_prepare(path=p, phase_mode="all", outcomes=("W",))
    with pytest.raises(KeyError):
        build_dose_response_model(prep, outcome_symbol="W")


# ---------------------------------------------------------------------------
# Mediation factories (LRP59 Beta-Binomial + LRP62 Gaussian composite)
# ---------------------------------------------------------------------------

# Outcome-leg coefficients shared by both mediation factories (the
# ``_build_outcome_leg`` helper). Guards the extraction in issue #85: any drift
# in the shared leg shows up as a missing/renamed RV here.
_OUTCOME_LEG_RVS = {"b0", "b_G", "b_M", "b_GM", "b_W", "b_A", "kappa_Y"}


def test_mediation_factory_builds_beta_binomial(tmp_path):
    """LRP59: single Beta-Binomial mediator + shared outcome leg builds and
    draws a prior predictive sample for both observation nodes."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built, med = build_mediation_model(
        prep, mediator_symbol="L", outcome_symbol="W", confounder_symbols=("E", "R")
    )
    names = {v.name for v in built.model.free_RVs}
    # Shared outcome leg + the confounder coefficients + the Beta-Binomial
    # mediator leg (a_L / kappa_M).
    assert _OUTCOME_LEG_RVS.issubset(names)
    assert {"b_E", "b_R", "a_E", "a_R", "a_L", "kappa_M"}.issubset(names)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=3)
    assert pp.prior_predictive["L_post"].shape[-1] == prep.n_obs
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_mediation_factory_builds_gaussian_composite(tmp_path):
    """LRP62: Gaussian route-composite mediator reuses the *same* outcome leg
    (a_comp / sigma_M instead of a_L / kappa_M)."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built, med = build_mediation_model(
        prep,
        outcome_symbol="W",
        confounder_symbols=("E", "R"),
        mediator_kind="gaussian_composite",
        route_symbols=("L", "B"),
    )
    names = {v.name for v in built.model.free_RVs}
    assert _OUTCOME_LEG_RVS.issubset(names)
    assert {"a_comp", "sigma_M"}.issubset(names)
    assert not {"a_L", "kappa_M"} & names  # the Beta-Binomial mediator leg is absent
    assert med.mediator_kind == "gaussian_composite"
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=4)
    assert pp.prior_predictive["M_post"].shape[-1] == prep.n_obs
    assert pp.prior_predictive["y_post"].shape[-1] == prep.n_obs


def test_mediation_data_carries_generic_confounders(tmp_path):
    """``MediationData`` carries confounders generically (issue #85): a
    ``conf_logit`` dict keyed by symbol plus ``confounder_symbols`` — not the old
    hardcoded ``E1_logit`` / ``R1_logit`` fields."""
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="itt")
    _, med = build_mediation_model(prep, confounder_symbols=("E", "R"))
    assert med.confounder_symbols == ("E", "R")
    assert set(med.conf_logit) == {"E", "R"}
    assert med.conf_logit["E"].shape == (prep.n_obs,)
    assert not hasattr(med, "E1_logit")
    assert not hasattr(med, "R1_logit")


def test_mediation_factory_custom_confounder_set(tmp_path):
    """A non-default adjustment set is honoured end to end: the model builds
    only the requested confounder coefficients and ``MediationData`` records
    exactly that set (so the g-formula adjusts for the fitted confounders)."""
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built, med = build_mediation_model(prep, confounder_symbols=("E",))
    names = {v.name for v in built.model.free_RVs}
    assert {"b_E", "a_E"}.issubset(names)
    assert not {"b_R", "a_R"} & names
    assert med.confounder_symbols == ("E",)
    assert set(med.conf_logit) == {"E"}


def test_two_mediator_factory_builds(tmp_path):
    """LRP64: two-mediator joint model builds with both mediator legs + interactions."""
    p = _write_synthetic(tmp_path, n_children=25)
    prep = load_and_prepare(path=p, phase_mode="itt")
    built, med = build_two_mediator_model(
        prep, outcome_symbol="W", mediator_symbols=("L", "E"), confounder_symbols=("R",)
    )
    names = {v.name for v in built.model.free_RVs}
    # Two mediator legs, the outcome paths, the interactions, and the R confounder.
    assert {"aL_G", "aE_G", "b_L", "b_E", "b_GL", "b_GE", "b_R"}.issubset(names)
    assert med.mediator_symbols == ("L", "E")
    assert med.n_trials_L == MEASURES["L"].n_trials
    assert med.n_trials_E == MEASURES["E"].n_trials
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=5)
    for node in ("L_post", "E_post", "y_post"):
        assert pp.prior_predictive[node].shape[-1] == built.prepared.n_obs


# ---------------------------------------------------------------------------
# Waitlist-crossover / difference-in-differences factory (kind="did")
# ---------------------------------------------------------------------------


def test_did_factory_builds(tmp_path):
    """Default DiD build: child RE + period anchor + binary treated indicator."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(path=p, phase_mode="all")
    built = build_did_model(prep, outcome_symbol="W")
    names = {v.name for v in built.model.free_RVs}
    assert {"alpha", "beta_period", "delta", "gamma_own", "gamma_A", "kappa",
            "sigma_child"}.issubset(names)
    assert "eta_base" in {v.name for v in built.model.deterministics}
    # Only P1/P2 are kept; phase 2 (t3->t4) is dropped.
    assert set(np.unique(built.prepared.phase)).issubset({0, 1})
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=31)
    assert pp.prior_predictive["y_post"].shape[-1] == built.prepared.n_obs


def test_did_factory_toggles_and_dose(tmp_path):
    """Toggling off child RE and age drops their RVs; dose swaps delta -> beta_dose."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(path=p, phase_mode="all")
    base = build_did_model(prep, outcome_symbol="W", use_child_re=False, use_age=False)
    bnames = {v.name for v in base.model.free_RVs}
    assert "delta" in bnames
    assert "sigma_child" not in bnames and "gamma_A" not in bnames

    # Dose variant needs an 'attend' covariate; inject a synthetic standardised one.
    prep.covariates["attend"] = np.linspace(-1.0, 1.0, prep.n_obs)
    dosed = build_did_model(prep, outcome_symbol="W", dose=True)
    dnames = {v.name for v in dosed.model.free_RVs}
    assert "beta_dose" in dnames and "delta" not in dnames


def test_did_factory_period_varying_dose(tmp_path):
    """period_varying_dose swaps the pooled beta_dose for partial-pooled per-period slopes (#135)."""
    p = _write_synthetic(tmp_path, n_children=20)
    prep = load_and_prepare(path=p, phase_mode="all")
    prep.covariates["attend"] = np.linspace(-1.0, 1.0, prep.n_obs)
    pv = build_did_model(
        prep, outcome_symbol="W", dose=True, period_varying_dose=True
    )
    free = {v.name for v in pv.model.free_RVs}
    dets = {v.name for v in pv.model.deterministics}
    assert {"mu_dose", "sigma_dose", "beta_dose_phase_raw"}.issubset(free)
    assert "beta_dose_phase" in dets and "beta_dose" not in free
    # One partial-pooled slope per kept period (P1, P2).
    assert pv.model["beta_dose_phase"].eval().shape == (2,)
    with pv.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=33)
    assert pp.prior_predictive["y_post"].shape[-1] == pv.prepared.n_obs
    # period_varying_dose requires dose=True.
    with pytest.raises(ValueError):
        build_did_model(prep, outcome_symbol="W", period_varying_dose=True)


def test_did_factory_requires_all_phase_mode(tmp_path):
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="itt")
    with pytest.raises(ValueError):
        build_did_model(prep, outcome_symbol="W")


# ---------------------------------------------------------------------------
# Correlated-domain-factor measurement model (kind="corr_factor", #134)
# ---------------------------------------------------------------------------


def test_correlated_factor_model_builds(tmp_path):
    """Correlated-domain-factor CFA: per-indicator loadings + LKJ factor correlation (#134)."""
    p = _write_synthetic(tmp_path, n_children=30)
    prep = load_and_prepare(path=p, phase_mode="itt")
    prep.covariates["blocks"] = np.linspace(-1.0, 1.0, prep.n_obs)
    built = build_correlated_factor_model(
        prep,
        outcome_symbol="W",
        domains={"vocabulary": ("R", "E"), "code": ("L", "B"), "grammar": ("F", "T")},
        structural_covariates=("blocks",),
    )
    free = {v.name for v in built.model.free_RVs}
    dets = {v.name for v in built.model.deterministics}
    assert {
        "factor_z", "lambda_load", "sigma_indicator", "beta_factor", "beta_blocks"
    }.issubset(free)
    assert {"factors", "factor_corr", "communality"}.issubset(dets)
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=9)
    # 6 indicators across 3 correlated domain factors.
    assert pp.prior["lambda_load"].sizes["indicator"] == 6
    assert pp.prior["factor_corr"].sizes["domain"] == 3
    assert pp.prior_predictive["y_post"].shape[-1] == built.prepared.n_obs


def test_correlated_factor_model_requires_two_indicators(tmp_path):
    """A domain with < 2 indicators cannot identify a factor."""
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="itt")
    with pytest.raises(ValueError):
        build_correlated_factor_model(
            prep, outcome_symbol="W", domains={"single": ("F",)}
        )


# ---------------------------------------------------------------------------
# Gain-factors / level-factors families (LRPGF / LRPLF, issue #127)
# ---------------------------------------------------------------------------


def _prep_all(tmp_path, **kw):
    return load_and_prepare(path=_write_synthetic(tmp_path, **kw), phase_mode="all")


def _prep_levels(tmp_path, **kw):
    return load_and_prepare(path=_write_synthetic(tmp_path, **kw), phase_mode="levels")


def test_gain_factors_factory_builds(tmp_path):
    """Core gain build: period anchor + own baseline + age + treatment, a child
    random intercept and a Beta-Binomial likelihood."""
    prep = _prep_all(tmp_path, n_children=20)
    built = build_gain_factors_model(prep, outcome_symbol="W")
    names = {v.name for v in built.model.free_RVs}
    assert {"alpha", "alpha_phase", "beta_trt", "gamma_own", "gamma_A", "kappa",
            "sigma_child"}.issubset(names)
    assert {v.name for v in built.model.observed_RVs} == {"y_post"}
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=41)
    assert pp.prior_predictive["y_post"].shape[-1] == built.prepared.n_obs


def test_gain_factors_skills_ability_interactions(tmp_path):
    """Skills, an ability covariate and focal interactions each add a coefficient
    (gamma_<skill> / gamma_ability / gamma_int_<a>_<b>)."""
    prep = _prep_all(tmp_path, n_children=20)
    prep.covariates["blocks"] = np.linspace(-1.0, 1.0, prep.n_obs)
    built = build_gain_factors_model(
        prep,
        outcome_symbol="W",
        skill_symbols=("L", "R"),
        ability_covariate="blocks",
        interactions=(("trt", "ability"), ("age", "ability")),
    )
    names = {v.name for v in built.model.free_RVs}
    assert {"gamma_L", "gamma_R", "gamma_ability",
            "gamma_int_trt_ability", "gamma_int_age_ability"}.issubset(names)
    # The trt×ability interaction is exposed for the interaction-aware AME (the
    # non-trt age×ability interaction is NOT — it cancels in the toggle). The
    # moderator vector must equal the standardised ability the factory used
    # (re-standardised on the kept rows, so mean 0 / unit SD on the fitted sample).
    from language_reading_predictors.statistical_models.preprocessing import standardise

    mods = dict(built.extras["trt_interaction_moderators"])
    assert set(mods) == {"gamma_int_trt_ability"}
    expected_ability, _ = standardise(built.prepared.covariates["blocks"])
    np.testing.assert_allclose(mods["gamma_int_trt_ability"], expected_ability)
    assert mods["gamma_int_trt_ability"].shape[0] == built.prepared.n_obs
    assert mods["gamma_int_trt_ability"].mean() == pytest.approx(0.0, abs=1e-9)


def test_gain_factors_treated_only_drops_treatment(tmp_path):
    """treated_only restricts to on-intervention rows; the then-constant beta_trt
    and every trt interaction drop out while non-trt interactions survive."""
    prep = _prep_all(tmp_path, n_children=20)
    prep.covariates["blocks"] = np.linspace(-1.0, 1.0, prep.n_obs)
    built = build_gain_factors_model(
        prep,
        outcome_symbol="W",
        ability_covariate="blocks",
        interactions=(("trt", "ability"), ("age", "ability")),
        treated_only=True,
    )
    names = {v.name for v in built.model.free_RVs}
    assert "beta_trt" not in names
    assert "gamma_int_trt_ability" not in names
    assert "gamma_int_age_ability" in names
    # No treatment term ⇒ no treatment-interaction moderators to net out.
    assert built.extras["trt_interaction_moderators"] == []
    # every retained row is on intervention
    on = (built.prepared.G == 1) | (built.prepared.phase >= 1)
    assert on.all()


def test_gain_factors_bernoulli_offfloor(tmp_path):
    """The floor rule: a Bernoulli on the off-floor indicator, no kappa, a
    y_offfloor node taking only 0/1."""
    prep = _prep_all(tmp_path, n_children=20)
    built = build_gain_factors_model(
        prep, outcome_symbol="P", likelihood="bernoulli_offfloor"
    )
    names = {v.name for v in built.model.free_RVs}
    assert "kappa" not in names
    assert {v.name for v in built.model.observed_RVs} == {"y_offfloor"}
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=42)
    yof = pp.prior_predictive["y_offfloor"].values
    assert set(np.unique(yof)).issubset({0, 1})


def test_gain_factors_rejects_bad_likelihood_and_phase(tmp_path):
    prep = _prep_all(tmp_path, n_children=15)
    with pytest.raises(ValueError):
        build_gain_factors_model(prep, outcome_symbol="W", likelihood="poisson")
    itt = load_and_prepare(path=_write_synthetic(tmp_path), phase_mode="itt")
    with pytest.raises(ValueError):
        build_gain_factors_model(itt, outcome_symbol="W")


def test_level_factors_factory_builds(tmp_path):
    """Level build: per-timepoint intercepts + group x time + ability x time +
    group x ability over the four timepoints, with no own baseline."""
    prep = _prep_levels(tmp_path, n_children=20)
    prep.covariates["blocks"] = np.linspace(-1.0, 1.0, prep.n_obs)
    built = build_level_factors_model(prep, outcome_symbol="W", ability_covariate="blocks")
    names = {v.name for v in built.model.free_RVs}
    assert {"alpha", "alpha_time", "b_grp_time", "gamma_A", "gamma_ability_time",
            "gamma_grp_ability", "kappa", "sigma_child"}.issubset(names)
    assert "gamma_own" not in names  # levels carries no own baseline
    assert built.prepared.n_phases == 4
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=43)
    # group x time is a per-timepoint vector over the four timepoints
    assert pp.prior["b_grp_time"].shape[-1] == 4
    assert pp.prior_predictive["y_post"].shape[-1] == built.prepared.n_obs


def test_level_factors_bernoulli_offfloor(tmp_path):
    prep = _prep_levels(tmp_path, n_children=20)
    prep.covariates["blocks"] = np.linspace(-1.0, 1.0, prep.n_obs)
    built = build_level_factors_model(
        prep, outcome_symbol="P", ability_covariate="blocks",
        likelihood="bernoulli_offfloor",
    )
    names = {v.name for v in built.model.free_RVs}
    assert "kappa" not in names
    assert {v.name for v in built.model.observed_RVs} == {"y_offfloor"}


def test_level_factors_group_ability_requires_ability(tmp_path):
    prep = _prep_levels(tmp_path, n_children=15)
    with pytest.raises(ValueError):
        build_level_factors_model(prep, outcome_symbol="W", group_ability=True)


def test_level_factors_requires_levels_phase_mode(tmp_path):
    prep = _prep_all(tmp_path, n_children=15)
    with pytest.raises(ValueError):
        build_level_factors_model(prep, outcome_symbol="W", group_ability=False)


def test_gf_lf_diag_vars_match_offfloor_builds(tmp_path):
    """_gf_diag_vars / _lf_diag_vars must name only RVs the off-floor factories
    build (kappa dropped), else summary_diagnostics raises KeyError at run time."""
    from types import SimpleNamespace

    from language_reading_predictors.statistical_models.pipeline import (
        _gf_diag_vars,
        _lf_diag_vars,
    )

    gp = _prep_all(tmp_path, n_children=15)
    g_built = build_gain_factors_model(
        gp, outcome_symbol="P", likelihood="bernoulli_offfloor"
    )
    g_names = {v.name for v in g_built.model.free_RVs} | {
        v.name for v in g_built.model.deterministics
    }
    g_diag = _gf_diag_vars(SimpleNamespace(extra={"likelihood": "bernoulli_offfloor"}))
    assert "kappa" not in g_diag
    assert not (set(g_diag) - g_names)

    lp = _prep_levels(tmp_path, n_children=15)
    lp.covariates["blocks"] = np.linspace(-1.0, 1.0, lp.n_obs)
    l_built = build_level_factors_model(
        lp, outcome_symbol="P", ability_covariate="blocks",
        likelihood="bernoulli_offfloor",
    )
    l_names = {v.name for v in l_built.model.free_RVs} | {
        v.name for v in l_built.model.deterministics
    }
    l_diag = _lf_diag_vars(SimpleNamespace(extra={
        "likelihood": "bernoulli_offfloor", "ability_covariate": "blocks",
    }))
    assert "kappa" not in l_diag
    assert not (set(l_diag) - l_names)


# ---------------------------------------------------------------------------
# Aligned-40-week per-protocol family (LRPAL, issue #127 follow-on)
# ---------------------------------------------------------------------------


def _prep_aligned(tmp_path, **kw):
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare_aligned,
    )
    return load_and_prepare_aligned(path=_write_synthetic(tmp_path, **kw))


def test_aligned_factory_builds(tmp_path):
    """Cross-sectional onset-aligned ANCOVA: cohort + own onset baseline + age,
    Beta-Binomial, and NO child random intercept (one row per child)."""
    prep = _prep_aligned(tmp_path, n_children=24)
    assert prep.phase_mode == "aligned" and prep.n_phases == 1
    built = build_aligned_model(prep, outcome_symbol="W")
    names = {v.name for v in built.model.free_RVs}
    assert {"alpha", "beta_cohort", "gamma_own", "gamma_A", "kappa"}.issubset(names)
    assert "sigma_child" not in names  # one row per child -> no random intercept
    assert "gamma_ability" not in names and "gamma_dose" not in names
    assert {v.name for v in built.model.observed_RVs} == {"y_post"}
    with built.model:
        pp = pm.sample_prior_predictive(draws=5, random_seed=51)
    assert pp.prior_predictive["y_post"].shape[-1] == built.prepared.n_obs


def test_aligned_factory_ability_dose_and_no_cohort(tmp_path):
    """Ability and dose each add a coefficient; use_cohort=False drops beta_cohort."""
    prep = _prep_aligned(tmp_path, n_children=24)
    prep.covariates["blocks"] = np.linspace(-1.0, 1.0, prep.n_obs)
    prep.covariates["dose"] = np.linspace(1.0, -1.0, prep.n_obs)
    built = build_aligned_model(
        prep, outcome_symbol="W", ability_covariate="blocks", use_dose=True
    )
    names = {v.name for v in built.model.free_RVs}
    assert {"gamma_ability", "gamma_dose", "beta_cohort"}.issubset(names)
    base = build_aligned_model(
        prep, outcome_symbol="W", ability_covariate="blocks", use_cohort=False
    )
    assert "beta_cohort" not in {v.name for v in base.model.free_RVs}


def test_aligned_factory_bernoulli_offfloor(tmp_path):
    prep = _prep_aligned(tmp_path, n_children=24)
    built = build_aligned_model(
        prep, outcome_symbol="P", likelihood="bernoulli_offfloor"
    )
    names = {v.name for v in built.model.free_RVs}
    assert "kappa" not in names
    assert {v.name for v in built.model.observed_RVs} == {"y_offfloor"}


def test_aligned_factory_rejects_dose_without_covariate_and_wrong_phase(tmp_path):
    prep = _prep_aligned(tmp_path, n_children=15)
    with pytest.raises(KeyError):
        build_aligned_model(prep, outcome_symbol="W", use_dose=True)  # no 'dose'
    itt = load_and_prepare(path=_write_synthetic(tmp_path), phase_mode="itt")
    with pytest.raises(ValueError):
        build_aligned_model(itt, outcome_symbol="W")


def test_al_diag_vars_match_build(tmp_path):
    """_al_diag_vars names only RVs the aligned factory builds: kappa present,
    no sigma_child (no random intercept)."""
    from types import SimpleNamespace

    from language_reading_predictors.statistical_models.pipeline import _al_diag_vars

    prep = _prep_aligned(tmp_path, n_children=20)
    prep.covariates["blocks"] = np.linspace(-1.0, 1.0, prep.n_obs)
    built = build_aligned_model(prep, outcome_symbol="W", ability_covariate="blocks")
    diag = _al_diag_vars(
        SimpleNamespace(extra={"ability_covariate": "blocks", "use_cohort": True})
    )
    built_names = {v.name for v in built.model.free_RVs} | {
        v.name for v in built.model.deterministics
    }
    assert not (set(diag) - built_names)
    assert "kappa" in diag and "sigma_child" not in diag


# ---------------------------------------------------------------------------
# Two-tier treatment-effect prior (issue #141)
# ---------------------------------------------------------------------------


def _tau_dist(model, name: str = "tau") -> str:
    rv = next(v for v in model.free_RVs if v.name == name)
    return priors._dist_from_rv(rv)


def test_itt_tau_prior_tiered_by_outcome(tmp_path):
    """A distal outcome (R) gets the tighter Normal(0, 0.3); a proximal outcome
    (W) keeps the wider Normal(0, 0.5)."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    assert is_distal("R") and not is_distal("W")
    r = build_itt_model(prep, outcome_symbol="R", cross_symbols=(), use_age_linear=True)
    w = build_itt_model(prep, outcome_symbol="W", cross_symbols=(), use_age_linear=True)
    assert _tau_dist(r.model) == "Normal(0, 0.3)"
    assert _tau_dist(w.model) == "Normal(0, 0.5)"


def test_itt_tau_sigma_override(tmp_path):
    """``tau_sigma`` overrides the tier default (for the sensitivity sweep)."""
    p = _write_synthetic(tmp_path)
    prep = load_and_prepare(path=p, phase_mode="itt")
    # Override a distal outcome up to 0.75 and a proximal one down to 0.25.
    r = build_itt_model(
        prep, outcome_symbol="R", cross_symbols=(), use_age_linear=True, tau_sigma=0.75
    )
    w = build_itt_model(
        prep, outcome_symbol="W", cross_symbols=(), use_age_linear=True, tau_sigma=0.25
    )
    assert _tau_dist(r.model) == "Normal(0, 0.75)"
    assert _tau_dist(w.model) == "Normal(0, 0.25)"


def test_did_and_gain_and_level_treatment_terms_tiered(tmp_path):
    """The DiD ``delta``, gain-factors ``beta_trt`` and level-factors group
    contrast all follow the outcome tier."""
    p = _write_synthetic(tmp_path)
    allp = load_and_prepare(path=p, phase_mode="all")
    levels = load_and_prepare(path=p, phase_mode="levels")
    levels.covariates["blocks"] = np.linspace(-1.0, 1.0, levels.n_obs)
    # distal E
    assert _tau_dist(build_did_model(allp, outcome_symbol="E").model, "delta") == "Normal(0, 0.3)"
    assert _tau_dist(
        build_gain_factors_model(allp, outcome_symbol="E").model, "beta_trt"
    ) == "Normal(0, 0.3)"
    assert _tau_dist(
        build_level_factors_model(levels, outcome_symbol="E", ability_covariate="blocks").model,
        "b_grp_time",
    ) == "Normal(0, 0.3)"
    # proximal L
    assert _tau_dist(build_did_model(allp, outcome_symbol="L").model, "delta") == "Normal(0, 0.5)"
    assert _tau_dist(
        build_gain_factors_model(allp, outcome_symbol="L").model, "beta_trt"
    ) == "Normal(0, 0.5)"


def test_association_group_terms_not_tiered(tmp_path):
    """Adjusted-association group terms keep the default 0.5 even for a distal
    outcome: mechanism ``beta_G`` and aligned ``beta_cohort`` are not the
    randomised effect and must not be tightened."""
    p = _write_synthetic(tmp_path)
    allp = load_and_prepare(path=p, phase_mode="all")
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare_aligned,
    )

    mech = build_mechanism_model(
        allp, mechanism_symbol="L", outcome_symbol="E", confounder_symbols=("G",)
    )
    assert _tau_dist(mech.model, "beta_G") == "Normal(0, 0.5)"
    aligned = load_and_prepare_aligned(path=p)
    al = build_aligned_model(aligned, outcome_symbol="E")
    assert _tau_dist(al.model, "beta_cohort") == "Normal(0, 0.5)"
