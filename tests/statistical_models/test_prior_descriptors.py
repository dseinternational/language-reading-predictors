# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prior meaning is recorded when a variable is created (#637 stage 2).

``priors.py`` derived a parameter's published role and rationale from its **name**
— an exact-name map, then a name prefix, a name suffix, and in several branches
the rendered distribution string, so a coefficient was routed to the association
role because it happened to read ``Normal(0, 0.3)``. Renaming a variable, or
widening a prior for a sensitivity fit, could therefore change what the report
said a parameter *means* without any change to its statistical role.

Every variable built through a named constructor now carries a
:class:`~priors.PriorDescriptor` recorded at ``to_pymc`` time. These tests pin
that: the descriptor comes from the constructor rather than the name, a call site
can declare a different scientific role where its family reuses a prior, and the
whole thing is invariant to renaming.
"""

from __future__ import annotations

import warnings

import pymc as pm

from language_reading_predictors.statistical_models import priors as P


def _model(build):
    with pm.Model() as model:
        build()
    return model


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_a_constructor_records_what_its_variable_means():
    model = _model(lambda: P.tau_prior().to_pymc("tau"))
    descriptor = P.descriptors_for(model)["tau"]

    assert descriptor.constructor == "tau"
    assert descriptor.role == "causal"
    assert descriptor.panel == "tau"
    assert descriptor.provenance == "constructor"
    assert descriptor.distribution == "Normal(0, 0.5)"
    assert "Treatment effect" in descriptor.rationale


def test_the_recorded_distribution_is_the_one_registered_not_the_default():
    """A constructor called with a non-default scale must not report the default."""
    model = _model(lambda: P.tau_prior(0.3).to_pymc("tau"))
    assert P.descriptors_for(model)["tau"].distribution == "Normal(0, 0.3)"


def test_meaning_survives_a_rename():
    """The defect this replaces: meaning followed the name, not the prior.

    ``beta_G`` was mapped to the treatment prior by name, so a measurement-model
    coefficient of that name published role ``causal``; a variable renamed out of
    the map fell through to a prefix rule instead.
    """
    original = _model(lambda: P.gamma_own_prior().to_pymc("gamma_own"))
    renamed = _model(lambda: P.gamma_own_prior().to_pymc("zzz_unmapped_name"))

    a = P.descriptors_for(original)["gamma_own"]
    b = P.descriptors_for(renamed)["zzz_unmapped_name"]
    assert (a.constructor, a.role, a.rationale, a.panel) == (
        b.constructor,
        b.role,
        b.rationale,
        b.panel,
    )
    assert a.parameter != b.parameter


def test_a_scale_change_does_not_move_the_role():
    """Several fallback branches keyed the role off the rendered scale string."""
    tight = _model(lambda: P.gamma_cross_prior(0.3).to_pymc("g_x"))
    wide = _model(lambda: P.gamma_cross_prior(1.5).to_pymc("g_x"))
    assert P.descriptors_for(tight)["g_x"].role == P.descriptors_for(wide)["g_x"].role
    assert (
        P.descriptors_for(tight)["g_x"].distribution
        != P.descriptors_for(wide)["g_x"].distribution
    )


def test_a_call_site_may_declare_a_different_scientific_role():
    """A family that reuses a prior for a different quantity says so where it builds it.

    The aligned family carries its cohort contrast on the treatment prior but
    reports no causal term at all; the mediation legs carry a randomised-arm
    coefficient whose causal deliverable is the decomposition, not the leg.
    """
    model = _model(
        lambda: P.tau_prior().to_pymc(
            "beta_cohort", role="association", rationale="Per-protocol cohort contrast."
        )
    )
    descriptor = P.descriptors_for(model)["beta_cohort"]
    assert descriptor.role == "association"
    assert descriptor.rationale == "Per-protocol cohort contrast."
    assert descriptor.constructor == "tau"
    assert descriptor.provenance == "call-site"


def test_declare_records_an_inline_prior():
    def build():
        P.declare(
            pm.Normal("u_child_raw", mu=0.0, sigma=1.0),
            role="nuisance",
            rationale="Non-centred child offset.",
        )

    descriptor = P.descriptors_for(_model(build))["u_child_raw"]
    assert descriptor.role == "nuisance"
    assert descriptor.constructor == "inline"
    assert descriptor.provenance == "inline"
    assert descriptor.distribution == "Normal(0, 1)"


def test_a_constructor_outside_a_model_block_records_nothing_and_still_builds():
    """Panels and plots call constructors with no model open."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = P.alpha_prior()
    assert spec.role == "nuisance"
    assert spec.distribution.params_dict["sigma"] == P.ALPHA_SIGMA_PROXIMAL


def test_the_spec_delegates_the_rest_of_the_preliz_surface():
    """``plot_and_save`` and the panel writer call straight through."""
    spec = P.kappa_prior()
    assert hasattr(spec, "plot_pdf")
    assert spec.params_dict == spec.distribution.params_dict


# ---------------------------------------------------------------------------
# The published table
# ---------------------------------------------------------------------------


def test_the_table_describes_a_variable_from_its_descriptor():
    def build():
        P.tau_prior().to_pymc("some_unmapped_effect")
        pm.Binomial("y", n=5, p=0.5, observed=[1, 2, 3])

    table = P.priors_table(_model(build))
    row = table[table["parameter"] == "some_unmapped_effect"].iloc[0]
    assert row["role"] == "causal"
    assert row["panel"] == "tau"
    assert "Treatment effect" in row["rationale"]


def test_an_undescribed_variable_still_falls_back_to_inference():
    """The inline half of the model graph is not migrated yet, and must still report.

    Kept explicit so the remaining gap is visible rather than silent: a variable
    built by a bare ``pm.*`` call, or by the shared HSGP builder inside
    ``dse_research_utils``, has no descriptor and is still classified from its
    name and scale.
    """

    def build():
        pm.HalfNormal("sigma_child", sigma=0.5)
        pm.Binomial("y", n=5, p=0.5, observed=[1, 2, 3])

    model = _model(build)
    assert "sigma_child" not in P.descriptors_for(model)
    row = P.priors_table(model).iloc[0]
    assert row["parameter"] == "sigma_child"
    assert row["role"] == "nuisance"


def test_an_explicit_override_still_outranks_a_descriptor():
    """A family-level correction remains available while the migration finishes."""

    def build():
        P.tau_prior().to_pymc("beta_dose")
        pm.Binomial("y", n=5, p=0.5, observed=[1, 2, 3])

    model = _model(build)
    table = P.priors_table(model, role_overrides={"beta_dose": "association"})
    assert table.iloc[0]["role"] == "association"


def test_every_named_constructor_declares_a_key_and_a_role():
    for key, constructor in P.ALL_PRIORS.items():
        assert getattr(constructor, "prior_key", None) == key, key
        assert getattr(constructor, "prior_role", None) in {
            "causal",
            "precision",
            "association",
            "nuisance",
            "gp",
        }, key
