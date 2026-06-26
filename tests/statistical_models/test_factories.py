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
    build_did_model,
    build_itt_model,
    build_joint_model,
    build_mechanism_model,
    build_mediation_model,
)
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
    TAUGHT_BLOCK1_OUTCOMES,
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
            for s in (*ITT_OUTCOMES, *TAUGHT_BLOCK1_OUTCOMES):
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


def test_taught_block1_measures_registered():
    """The taught-vocabulary family is defined with honest denominators and kept
    out of ITT_OUTCOMES (so the eight-outcome joint model stays stable)."""
    assert TAUGHT_BLOCK1_OUTCOMES == ("TE", "TR", "UE", "UR")
    assert not set(TAUGHT_BLOCK1_OUTCOMES) & set(ITT_OUTCOMES)
    assert MEASURES["TE"].n_trials == 24 and MEASURES["TE"].n_trials_confirmed
    assert MEASURES["TR"].n_trials == 24 and MEASURES["TR"].n_trials_confirmed
    # Not-taught ceilings are observed-max and flagged unconfirmed.
    assert MEASURES["UE"].n_trials == 12 and not MEASURES["UE"].n_trials_confirmed
    assert MEASURES["UR"].n_trials == 12 and not MEASURES["UR"].n_trials_confirmed


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


def test_did_factory_requires_all_phase_mode(tmp_path):
    p = _write_synthetic(tmp_path, n_children=15)
    prep = load_and_prepare(path=p, phase_mode="itt")
    with pytest.raises(ValueError):
        build_did_model(prep, outcome_symbol="W")
