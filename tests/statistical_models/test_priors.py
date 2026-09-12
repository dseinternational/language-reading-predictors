# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the prior role registry + per-model table (issue #125 Area 1)."""

from __future__ import annotations

from types import SimpleNamespace

from language_reading_predictors.statistical_models import priors


def _rv(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _described(*names: str, deterministics: tuple[str, ...] = ()) -> SimpleNamespace:
    """A stub model whose variables carry the descriptors a real build records.

    Since #637 a variable's published role and rationale come from the descriptor
    recorded when it was created, not from its name — so a stub model that carries
    no descriptors is not a model with unnamed priors, it is a model whose priors
    were never declared, and ``priors_table`` says so rather than guessing.
    """
    recorded = {}
    for name in names:
        if name in _INLINE_STUBS:
            role, panel = _INLINE_STUBS[name]
            recorded[name] = priors.PriorDescriptor(
                parameter=name,
                constructor="inline",
                distribution="Normal(0, 1)",
                role=role,
                rationale=f"Inline prior for {name}.",
                panel=panel,
                provenance="inline",
            )
            continue
        key = _CONSTRUCTOR_FOR[name]
        constructor = priors.ALL_PRIORS[key]
        recorded[name] = priors.PriorDescriptor(
            parameter=name,
            constructor=key,
            distribution=priors._dist_from_doc(constructor),
            role=constructor.prior_role,
            rationale=priors._first_docline(constructor),
            panel=key,
            provenance="constructor",
        )
    model = SimpleNamespace(
        free_RVs=[_rv(name) for name in names],
        deterministics=[_rv(name) for name in deterministics],
    )
    model._dse_prior_descriptors = recorded
    return model


#: Stub variables declared inline rather than through a named constructor:
#: ``(role, panel)``, as their ``declare`` call records.
_INLINE_STUBS = {
    "beta_dose_phase_raw": ("nuisance", ""),
    "beta_group_nuisance": ("nuisance", ""),
    "beta_L": ("association", "predictor_slope"),
    "b_grp_time": ("causal", "tau"),
}

#: Which constructor built each stub variable, as a real factory would record.
_CONSTRUCTOR_FOR = {
    "alpha": "alpha",
    "tau": "tau",
    "gamma_own": "gamma_own",
    "gamma_A": "gamma_age",
    "kappa": "kappa",
    "mu_dose": "beta_mech",
    "sigma_dose": "sigma_dose",
    "beta_dose": "beta_mech",
    "beta_dose_phase_raw": "beta_mech",
}


def test_used_prior_keys_prunes_unused():
    model = _described("alpha", "tau", "gamma_own", "gamma_A", "kappa", deterministics=("eta",))
    keys = priors.used_prior_keys(model)
    assert set(keys) == {"alpha", "tau", "gamma_own", "gamma_age", "kappa"}
    # GP panels are not used by a plain ITT model -> pruned.
    assert "ell" not in keys
    assert "eta_main" not in keys
    assert "gamma_cross" not in keys


def test_used_prior_keys_skips_inline_noncentred_offsets():
    model = _described(
        "mu_dose",
        "sigma_dose",
        "beta_dose_phase_raw",
        deterministics=("beta_dose_phase",),
    )
    keys = priors.used_prior_keys(
        model,
        ctor_overrides={"mu_dose": "tau", "beta_dose_phase": "tau"},
    )
    assert keys == ["beta_mech", "sigma_dose"]
    # The non-centred offset names no panel: its meaning is carried by the scale
    # it is multiplied by, and its declaration says so.
    assert priors.described_prior_row(model, _rv("beta_dose_phase_raw"))["panel"] == ""


def test_priors_table_columns_and_rows():
    model = _described("alpha", "tau", "gamma_own", "gamma_A", "kappa")
    df = priors.priors_table(model)
    assert list(df.columns) == ["parameter", "distribution", "role", "rationale", "panel"]
    assert set(df["parameter"]) == {"alpha", "tau", "gamma_own", "gamma_A", "kappa"}
    by_param = df.set_index("parameter")
    assert by_param.loc["tau", "role"] == "causal"
    assert by_param.loc["gamma_A", "role"] == "precision"
    assert by_param.loc["tau", "panel"] == "tau"


def test_priors_table_applies_context_overrides():
    model = _described("alpha", "beta_dose", "mu_dose", "sigma_dose")
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
    model = _described("beta_L", "beta_group_nuisance")
    by_param = priors.priors_table(model).set_index("parameter")
    assert by_param.loc["beta_L", "role"] == "association"
    assert by_param.loc["beta_group_nuisance", "role"] == "nuisance"
    assert by_param.loc["beta_group_nuisance", "distribution"] == "Normal(0, 1)"
    assert by_param.loc["beta_group_nuisance", "panel"] == ""


def test_priors_table_applies_rationale_overrides():
    model = _described("b_grp_time")
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


def test_anchor_mean_alone_does_not_infer_empirical_bayes():
    """A vector location can be chosen externally; its source must be declared."""
    import numpy as np
    import pymc as pm

    with pm.Model() as model:
        priors.declare(
            pm.Normal("alpha", mu=np.array([0.3, -0.2]), sigma=1.5),
            role="nuisance",
            rationale="Externally supplied locations.",
        )
    row = priors.priors_table(model).iloc[0]
    assert row["rationale"] == "Externally supplied locations."
    assert priors.EMPIRICAL_BAYES_SENTENCE not in row["rationale"]
