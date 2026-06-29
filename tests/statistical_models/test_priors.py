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
