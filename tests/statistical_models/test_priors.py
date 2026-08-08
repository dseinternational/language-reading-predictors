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
    from language_reading_predictors.statistical_models.prior_artifacts import (
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


def test_gain_factor_moderation_variant_demotes_beta_trt_role():
    """#490 review: a moderation variant's ``beta_trt`` must not reach the priors
    table as "causal" — every artefact of a variant fit presents it as a
    model-dependent association, priors_table.csv included. A headline primary is
    untouched (no override entry), so its ``beta_trt`` keeps the causal role."""
    from language_reading_predictors.statistical_models.prior_artifacts import (
        _prior_table_overrides,
    )

    variant = SimpleNamespace(
        spec=SimpleNamespace(
            kind="gain_factors",
            outcome_symbol="W",
            extra={"moderation_variant": True},
        ),
        model=None,
    )
    _ctor, role, rationale = _prior_table_overrides(variant)
    assert role["beta_trt"] == "association"
    assert "interaction-free" in rationale["beta_trt"]

    primary = SimpleNamespace(
        spec=SimpleNamespace(kind="gain_factors", outcome_symbol="W", extra={}),
        model=None,
    )
    _ctor, role, _rationale = _prior_table_overrides(primary)
    assert "beta_trt" not in role


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


def test_empirical_bayes_rationale_matches_only_anchored_locations():
    """The EB label fires on a computed prior *mean*, not on any ``<constant>``.

    Keyed on the distribution rather than the name because ``alpha`` is anchored in
    the growth family and a free zero-centred deviation everywhere else (#390 P1).
    The negatives are the other ``<constant>`` renderings in the suite — LKJ
    dimensions and a ZeroSumNormal shape — which are structural arguments, not
    locations, and must not be labelled empirical Bayes.
    """
    eb = priors.empirical_bayes_rationale
    assert "grand mean observed logit" in eb("alpha", "Normal(<constant>, 1.5)")
    assert "observed wave-1 mean logit" in eb("mu1", "Normal(<constant>, 1)")
    assert "arm-blind observed t1 logit" in eb("alpha_offset", "Normal(0, 1.5)")
    for anchored in ("alpha", "mu1", "alpha_offset"):
        rationale = eb(anchored, "Normal(<constant>, 1)")
        if anchored != "alpha_offset":
            assert priors.EMPIRICAL_BAYES_SENTENCE in rationale

    assert eb("alpha", "Normal(0, 1.5)") == ""
    assert eb("mu1", "Normal(0, 1)") == ""
    assert eb("factor_mean", "ZeroSumNormal(1, <constant>)") == ""
    assert eb("trait_corr_chol", "LKJCorrRV(<constant>, 2)") == ""
    assert eb("beta", "Normal(0, 0.5)") == ""


def test_anchored_intercept_row_replaces_the_zero_centred_docstring():
    """An anchored ``alpha`` must not inherit ``alpha_prior``'s docstring.

    That docstring reads "Intercept alpha ~ Normal(0, 1.5)", which is the prior the
    growth family does *not* fit — its mean is the grand mean observed logit. The
    rationale is therefore replaced, not appended to.
    """
    import numpy as np
    import pymc as pm

    with pm.Model() as anchored:
        pm.Normal("alpha", mu=np.array([0.3, -0.2]), sigma=1.5, shape=2)
    row = priors.priors_table(anchored).iloc[0]
    assert "<constant>" in row["distribution"]
    assert priors.EMPIRICAL_BAYES_SENTENCE in row["rationale"]
    assert "Normal(0, 1.5)" not in row["rationale"]

    with pm.Model() as free:
        pm.Normal("alpha", mu=0.0, sigma=1.5)
    free_row = priors.priors_table(free).iloc[0]
    assert priors.EMPIRICAL_BAYES_SENTENCE not in free_row["rationale"]
