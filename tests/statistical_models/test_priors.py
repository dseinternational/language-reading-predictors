# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the prior role registry + per-model table (issue #125 Area 1)."""

from __future__ import annotations

from types import SimpleNamespace

from language_reading_predictors.statistical_models import priors


def _rv(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def test_prior_info_roles_and_distribution():
    assert priors.prior_info_for_rv("tau")["role"] == "causal"
    assert priors.prior_info_for_rv("beta_trt")["role"] == "causal"  # tau-backed
    assert priors.prior_info_for_rv("b_grp_time")["role"] == "causal"
    assert priors.prior_info_for_rv("gamma_own")["role"] == "precision"
    assert priors.prior_info_for_rv("gamma_A")["role"] == "precision"
    assert priors.prior_info_for_rv("alpha")["role"] == "nuisance"
    assert priors.prior_info_for_rv("kappa")["role"] == "nuisance"
    assert priors.prior_info_for_rv("gamma_cross")["role"] == "association"
    # Unlisted gamma_*/b_*/a_* fall back to the cross (association) prior.
    assert priors.prior_info_for_rv("gamma_grp_ability")["role"] == "association"
    assert priors.prior_info_for_rv("b_M")["role"] == "association"
    # Inline priors are captured (they have no constructor).
    assert priors.prior_info_for_rv("sigma_child")["role"] == "nuisance"
    assert priors.prior_info_for_rv("alpha_phase")["role"] == "nuisance"
    # Distribution is extracted from the constructor docstring (source of truth).
    assert priors.prior_info_for_rv("tau")["distribution"] == "Normal(0, 0.5)"
    assert priors.prior_info_for_rv("kappa")["distribution"] == "HalfNormal(50)"


def test_prior_info_context_overrides():
    info = priors.prior_info_for_rv(
        "beta_dose",
        ctor_overrides={"beta_dose": "beta_mech"},
        role_overrides={"beta_dose": "association"},
    )
    assert info["role"] == "association"
    assert info["panel"] == "beta_mech"
    assert info["distribution"] == "Normal(0, 1)"

    sigma = priors.prior_info_for_rv("sigma_dose")
    assert sigma["role"] == "nuisance"
    assert sigma["panel"] == "sigma_dose"
    assert sigma["distribution"] == "HalfNormal(0.5)"


def test_prior_info_panel_mapping():
    assert priors.prior_info_for_rv("gamma_A")["panel"] == "gamma_age"
    assert priors.prior_info_for_rv("tau")["panel"] == "tau"
    # Inline priors have no panel file.
    assert priors.prior_info_for_rv("sigma_child")["panel"] == ""


def test_used_prior_keys_prunes_unused():
    model = SimpleNamespace(
        free_RVs=[_rv("alpha"), _rv("tau"), _rv("gamma_own"), _rv("gamma_A"), _rv("kappa")],
        deterministics=[_rv("eta")],
    )
    keys = priors.used_prior_keys(model)
    assert set(keys) == {"alpha", "tau", "gamma_own", "gamma_age", "kappa"}
    # GP panels are not used by a plain ITT model -> pruned.
    assert "ell" not in keys
    assert "eta_main" not in keys
    assert "gamma_cross" not in keys


def test_used_prior_keys_skips_inline_noncentred_offsets():
    model = SimpleNamespace(
        free_RVs=[
            _rv("mu_dose"),
            _rv("sigma_dose"),
            _rv("beta_dose_phase_raw"),
        ],
        deterministics=[_rv("beta_dose_phase")],
    )
    keys = priors.used_prior_keys(
        model,
        ctor_overrides={"mu_dose": "tau", "beta_dose_phase": "tau"},
    )
    assert keys == ["tau", "sigma_dose"]
    assert priors.prior_info_for_rv("beta_dose_phase_raw")["panel"] == ""


def test_priors_table_columns_and_rows():
    model = SimpleNamespace(
        free_RVs=[_rv("alpha"), _rv("tau"), _rv("gamma_own"), _rv("gamma_A"), _rv("kappa")],
        deterministics=[],
    )
    df = priors.priors_table(model)
    assert list(df.columns) == ["parameter", "distribution", "role", "rationale", "panel"]
    assert set(df["parameter"]) == {"alpha", "tau", "gamma_own", "gamma_A", "kappa"}
    by_param = df.set_index("parameter")
    assert by_param.loc["tau", "role"] == "causal"
    assert by_param.loc["gamma_A", "role"] == "precision"
    assert by_param.loc["tau", "panel"] == "tau"


def test_priors_table_applies_context_overrides():
    model = SimpleNamespace(
        free_RVs=[_rv("alpha"), _rv("beta_dose"), _rv("mu_dose"), _rv("sigma_dose")],
        deterministics=[],
    )
    df = priors.priors_table(
        model,
        ctor_overrides={"beta_dose": "beta_mech", "mu_dose": "beta_mech"},
        role_overrides={"beta_dose": "association", "mu_dose": "association"},
    )
    by_param = df.set_index("parameter")
    assert by_param.loc["beta_dose", "panel"] == "beta_mech"
    assert by_param.loc["beta_dose", "role"] == "association"
    assert by_param.loc["mu_dose", "panel"] == "beta_mech"
    assert by_param.loc["sigma_dose", "panel"] == "sigma_dose"


def test_concurrent_group_term_is_documented_as_nuisance():
    model = SimpleNamespace(
        free_RVs=[_rv("beta_L"), _rv("beta_group_nuisance")],
        deterministics=[],
    )
    by_param = priors.priors_table(model).set_index("parameter")
    assert by_param.loc["beta_L", "role"] == "association"
    assert by_param.loc["beta_group_nuisance", "role"] == "nuisance"
    assert by_param.loc["beta_group_nuisance", "distribution"] == "Normal(0, 1)"
    assert by_param.loc["beta_group_nuisance", "panel"] == ""


def test_level_factor_prior_role_is_conservative_for_group_time_vector():
    from language_reading_predictors.statistical_models.pipeline import (
        _prior_table_overrides,
    )

    ctx = SimpleNamespace(
        spec=SimpleNamespace(
            kind="level_factors",
            outcome_symbol="W",
            extra={"group_by_time": True},
        ),
        model=None,
    )
    _ctor, role, rationale = _prior_table_overrides(ctx)
    assert role["b_grp_time"] == "association"
    assert "only b_grp_time[1]" in rationale["b_grp_time"]


def test_priors_table_applies_rationale_overrides():
    model = SimpleNamespace(free_RVs=[_rv("b_grp_time")], deterministics=[])
    df = priors.priors_table(
        model,
        role_overrides={"b_grp_time": "association"},
        rationale_overrides={"b_grp_time": "Only b_grp_time[1] is randomised."},
    )
    row = df.iloc[0]
    assert row["role"] == "association"
    assert row["rationale"] == "Only b_grp_time[1] is randomised."
