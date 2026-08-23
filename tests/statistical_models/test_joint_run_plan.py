# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed joint settings and pre-I/O run plan (#394 pillar 4)."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import joint as J
from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.context import ModelSpec

_JOINT_MODULES = (
    "lrp_rli_itt_012",
    "lrp_rli_itt_015",
    "lrp_rli_itt_016",
    "lrp_rli_itt_115",
)
#: #551: the LKJ residual-correlation dependence-sensitivity companions of the
#: three two-outcome contrasts, keyed companion -> parent.
_DEPENDENCE_COMPANIONS = {
    "lrp_rli_itt_215": "lrp_rli_itt_015",
    "lrp_rli_itt_315": "lrp_rli_itt_115",
    "lrp_rli_itt_216": "lrp_rli_itt_016",
}
_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _spec(*, settings=None, **extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-itt-000",
        kind="joint",
        title="test joint",
        model_settings=settings,
        extra=extra,
    )


def _registered_specs() -> list[ModelSpec]:
    return [
        importlib.import_module(f"language_reading_predictors.statistical_models.{name}").SPEC
        for name in _JOINT_MODULES
    ]


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown joint setting.*use_age_gpp"):
        J.JointModelSettings.from_legacy_extra({"use_age_gpp": True}, model_id="lrp-rli-itt-999")


def test_settings_accept_global_target_accept_without_owning_it():
    settings = J.JointModelSettings.from_legacy_extra(
        {"target_accept": 0.99, "use_age_linear": True},
        model_id="lrp-rli-itt-999",
    )
    assert settings.use_age_linear is True
    assert "target_accept" not in settings.__dataclass_fields__


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"use_age_gp": 1}, "use_age_gp"),
        (
            {"use_age_gp": True, "use_age_linear": True},
            "mutually exclusive",
        ),
        ({"loo_unit": "cell"}, "loo_unit must be 'child'"),
        ({"outcomes": ("W", "W")}, "duplicate"),
    ],
)
def test_settings_reject_misshaped_or_contradictory_values(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        J.JointModelSettings(**kwargs)


def test_contrast_rejects_identical_outcomes():
    with pytest.raises(ValueError, match="must be different"):
        J.JointContrastSettings("W", "W")


def test_legacy_contrast_rejects_metadata_without_pair_and_unknown_keys():
    with pytest.raises(ValueError, match="requires a difference pair"):
        J.JointModelSettings.from_legacy_extra(
            {"difference_metadata": {"contrast_kind": "x"}},
            model_id="x",
        )
    with pytest.raises(ValueError, match="unknown joint contrast metadata"):
        J.JointModelSettings.from_legacy_extra(
            {
                "difference": ("W", "L"),
                "difference_metadata": {"contrast_knd": "x"},
            },
            model_id="x",
        )


def test_resolve_rejects_wrong_kind_unknown_outcome_and_incoherent_structure():
    wrong = ModelSpec(model_id="x", kind="itt", title="x")
    with pytest.raises(ValueError, match="expected kind 'joint'"):
        J.resolve_joint_run_plan(wrong)

    with pytest.raises(ValueError, match="unrecognised bounded outcome"):
        J.resolve_joint_run_plan(_spec(outcomes=("W", "ZZ")))

    with pytest.raises(ValueError, match="contradicts use_residual_correlation"):
        J.resolve_joint_run_plan(
            _spec(
                use_residual_correlation=True,
                joint_structure="factorised_outcome_marginals",
            )
        )


def test_resolve_rejects_contrast_outcomes_outside_model():
    settings = J.JointModelSettings(
        outcomes=("W", "L"),
        contrast=J.JointContrastSettings("W", "R"),
    )
    with pytest.raises(ValueError, match="contrast outcome.*not in outcomes"):
        J.resolve_joint_run_plan(_spec(settings=settings))


def test_default_legacy_plan_preserves_loader_and_factory_defaults():
    plan = J.resolve_joint_run_plan(_spec())
    assert plan.settings_source == "legacy_extra"
    assert plan.prepare_kwargs() == {"phase_mode": "itt"}
    assert plan.outcomes_explicit is False
    assert plan.factory_kwargs() == {
        "outcomes": ("W", "R", "E", "L", "P", "B", "F", "T"),
        "use_age_gp": False,
        "partial_pool_age_gp": True,
        "use_residual_correlation": False,
        "use_cross_baselines": True,
        "use_age_linear": False,
    }
    assert plan.diagnostic_vars() == ["alpha", "tau", "gamma_own", "kappa"]


def test_correlated_age_gp_plan_drives_factory_and_diagnostics():
    plan = J.resolve_joint_run_plan(
        _spec(
            outcomes=("W", "L"),
            use_age_gp=True,
            partial_pool_age_gp=False,
            use_residual_correlation=True,
            joint_structure="residual_correlated",
        )
    )
    assert plan.prepare_kwargs() == {
        "phase_mode": "itt",
        "outcomes": ("W", "L"),
    }
    assert plan.factory_kwargs()["partial_pool_age_gp"] is False
    assert plan.joint_structure == "residual_correlated"
    # The dependence block reports its per-outcome residual SDs and the free
    # pairwise correlations (#551), and power scaling covers them beside tau.
    assert plan.diagnostic_vars() == [
        "alpha",
        "tau",
        "gamma_own",
        "kappa",
        "sigma_outcome",
        "u_corr_pair",
    ]
    assert plan.psense_vars == ["tau", "sigma_outcome", "u_corr_pair"]
    assert "residual-correlation block" in plan.recipe_markdown(title="t")
    assert "LKJ" in plan.design
    assert "dependence model" in plan.causal_status
    factorised = J.resolve_joint_run_plan(_spec(outcomes=("W", "L")))
    assert factorised.psense_vars == ["tau"]
    assert "does not estimate paired cross-outcome residual covariance" in factorised.causal_status


def test_split_settings_between_typed_and_extra_is_rejected():
    spec = _spec(
        settings=J.JointModelSettings(outcomes=("W", "L")),
        use_age_linear=True,
    )
    with pytest.raises(ValueError, match="cannot be split"):
        J.resolve_joint_run_plan(spec)


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import joint as P

    touched = {"context": False, "data": False}

    def _context(*args, **kwargs):
        touched["context"] = True
        raise AssertionError("make_context must not run")

    def _data(*args, **kwargs):
        touched["data"] = True
        raise AssertionError("load_and_prepare must not run")

    monkeypatch.setattr(P, "make_context", _context)
    monkeypatch.setattr(P, "load_and_prepare", _data)
    with pytest.raises(ValueError, match="unknown joint setting"):
        P.fit_joint(_spec(use_age_gpp=True))
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_the_attached_plan(tmp_path):
    spec = _spec(
        settings=J.JointModelSettings(
            outcomes=("TE", "UE"),
            use_cross_baselines=False,
            use_age_linear=True,
            contrast=J.JointContrastSettings("TE", "UE"),
        )
    )
    plan = J.resolve_joint_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))
    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated joint run plan" in text
    assert "available-case modified intention-to-treat" in text
    assert "`TE - UE`" in text


def test_pipeline_has_no_direct_joint_setting_reads():
    from language_reading_predictors.statistical_models.pipelines import joint as P

    source = inspect.getsource(P)
    assert "spec.extra" not in source
    assert "ctx.spec.extra" not in source


def test_dependence_companions_match_their_parents_except_the_block():
    """#551: each LKJ companion is the parent's fit with the residual block on —
    same outcomes, precision terms, LOO unit and contrast — and its recipe and
    causal status describe the block; its dependence note names the parent."""
    for companion_name, parent_name in _DEPENDENCE_COMPANIONS.items():
        companion = importlib.import_module(
            f"language_reading_predictors.statistical_models.{companion_name}"
        ).SPEC
        parent = importlib.import_module(
            f"language_reading_predictors.statistical_models.{parent_name}"
        ).SPEC
        assert companion.kind == "joint"
        cs, ps = companion.model_settings, parent.model_settings
        assert isinstance(cs, J.JointModelSettings)
        assert cs.use_residual_correlation is True
        assert cs.joint_structure == "residual_correlated"
        assert ps.use_residual_correlation is False
        # Everything except the block and the note is identical to the parent.
        for field in ("outcomes", "use_age_gp", "partial_pool_age_gp",
                      "use_cross_baselines", "use_age_linear", "loo_unit"):
            assert getattr(cs, field) == getattr(ps, field), (companion_name, field)
        for field in ("left", "right", "contrast_kind", "contrast_label",
                      "positive_interpretation", "negative_interpretation",
                      "transfer_outcome", "transfer_interpretation"):
            assert getattr(cs.contrast, field) == getattr(ps.contrast, field), (companion_name, field)
        assert parent.model_id in cs.contrast.dependence_note
        assert "residual-correlation block is on" in cs.contrast.dependence_note
        # And the parent's note points at the companion (#551 acceptance: the
        # parents' dependence caveat cites the companion).
        assert companion.model_id in ps.contrast.dependence_note
        # The machine-readable pairing (2026-08-21 review, finding 3): the parent
        # names its registered companion so the release decision can verify it;
        # the companion — itself the dependence model — must not name one.
        assert ps.contrast.dependence_companion == companion.model_id
        assert cs.contrast.dependence_companion is None
        plan = J.resolve_joint_run_plan(companion)
        parent_plan = J.resolve_joint_run_plan(parent)
        assert plan.joint_structure == "residual_correlated"
        assert plan.outcomes == parent_plan.outcomes
        assert plan.difference == parent_plan.difference
        assert plan.factory_kwargs() == {
            **parent_plan.factory_kwargs(),
            "use_residual_correlation": True,
        }
        assert plan.diagnostic_vars()[-2:] == ["sigma_outcome", "u_corr_pair"]
        assert plan.psense_vars == ["tau", "sigma_outcome", "u_corr_pair"]


def test_the_release_pairing_constant_matches_the_registered_declarations():
    """2026-08-23 joint audit, finding 2. ``release`` derives the pairing from
    ``JOINT_DEPENDENCE_COMPANIONS`` because every stored parent artefact predates
    the plan field, so the constant is a second source of truth and must not drift
    from the modules that own the declaration."""
    declared = {}
    for companion_name, parent_name in _DEPENDENCE_COMPANIONS.items():
        parent = importlib.import_module(
            f"language_reading_predictors.statistical_models.{parent_name}"
        ).SPEC
        companion = importlib.import_module(
            f"language_reading_predictors.statistical_models.{companion_name}"
        ).SPEC
        declared[parent.model_id] = companion.model_id
    assert J.JOINT_DEPENDENCE_COMPANIONS == declared


def test_residual_correlation_needs_at_least_two_outcomes():
    """A one-dimensional LKJ block has no free correlation and cannot be built;
    the resolver rejects it before any output directory is reset or data loaded
    (2026-08-23 joint audit, lower-priority API correction)."""
    settings = J.JointModelSettings(
        outcomes=("TE",),
        use_residual_correlation=True,
        joint_structure="residual_correlated",
    )
    with pytest.raises(ValueError, match="at least two outcomes"):
        J.resolve_joint_run_plan(_spec(settings=settings))


def test_the_correlated_estimand_is_recorded_as_latent_conditional():
    """Finding 1: the companion's average marginal effect is standardised over the
    fitted children conditional on draws of their own residuals — a different
    estimand from the parent's, not the parent's with its covariance corrected.
    The plan must say so, and must not promise point-estimate invariance."""
    correlated = J.resolve_joint_run_plan(
        _spec(
            settings=J.JointModelSettings(
                outcomes=("TE", "UE"),
                use_residual_correlation=True,
                joint_structure="residual_correlated",
            )
        )
    )
    factorised = J.resolve_joint_run_plan(
        _spec(settings=J.JointModelSettings(outcomes=("TE", "UE")))
    )
    assert "conditional on posterior draws of their own residuals" in correlated.estimand
    assert "NOT invariant by construction" in correlated.estimand
    assert "latent-conditional" not in factorised.estimand
    assert "conditional on posterior draws" not in factorised.estimand


def test_the_blending_link_policy_scope_is_recorded_for_a_joint_b_fit():
    """Finding 12: the 008/108 pairing governs the B model of record; a joint B row
    is a secondary structural cross-check. The plan records that scope so the
    release decision can verify it instead of relying on the findings-box prose."""
    with_b = J.resolve_joint_run_plan(
        _spec(settings=J.JointModelSettings(outcomes=("W", "B")))
    )
    without_b = J.resolve_joint_run_plan(
        _spec(settings=J.JointModelSettings(outcomes=("W", "R")))
    )
    assert without_b.link_sensitivity_scope is None
    assert with_b.link_sensitivity_scope is not None
    assert "lrp-rli-itt-008" in with_b.link_sensitivity_scope
    assert "not independently release-qualified" in with_b.link_sensitivity_scope
    assert with_b.as_dict()["link_sensitivity_scope"] == with_b.link_sensitivity_scope


def test_a_residual_correlated_fit_must_not_declare_a_dependence_companion():
    """A correlated fit IS the dependence model (2026-08-21 review, finding 3);
    naming a further companion would send the release gate chasing a chain."""
    settings = J.JointModelSettings(
        outcomes=("TE", "UE"),
        use_cross_baselines=False,
        use_age_linear=True,
        use_residual_correlation=True,
        joint_structure="residual_correlated",
        contrast=J.JointContrastSettings(
            left="TE", right="UE", dependence_companion="lrp-rli-itt-999"
        ),
    )
    with pytest.raises(ValueError, match="must not declare a dependence_companion"):
        J.resolve_joint_run_plan(_spec(settings=settings))


def test_the_dependence_companion_is_plan_only_metadata():
    """The companion id drives the release decision through the resolved plan; it
    must reach ``config.json`` but never the ``tau_difference.csv`` metadata."""
    settings = J.JointModelSettings(
        outcomes=("TE", "UE"),
        use_cross_baselines=False,
        use_age_linear=True,
        contrast=J.JointContrastSettings(
            left="TE", right="UE", dependence_companion="lrp-rli-itt-215"
        ),
    )
    plan = J.resolve_joint_run_plan(_spec(settings=settings))
    assert plan.as_dict()["contrast"]["dependence_companion"] == "lrp-rli-itt-215"
    metadata = plan.difference_metadata()
    assert metadata is None or "dependence_companion" not in metadata


def test_every_registered_joint_model_is_typed_and_preserves_legacy_contract():
    expected = {
        "lrp-rli-itt-012": (
            ("TR", "TE", "UR", "UE", "R", "E", "L", "B", "P", "W"),
            None,
        ),
        "lrp-rli-itt-015": (("TE", "UE"), ("TE", "UE")),
        "lrp-rli-itt-016": (("TE", "TR"), ("TE", "TR")),
        "lrp-rli-itt-115": (("TR", "UR"), ("TR", "UR")),
    }
    specs = _registered_specs()
    assert len(specs) == 4
    for spec in specs:
        assert isinstance(spec.model_settings, J.JointModelSettings), spec.model_id
        assert spec.extra == {}, spec.model_id
        plan = J.resolve_joint_run_plan(spec)
        outcomes, difference = expected[spec.model_id]
        assert plan.settings_source == "typed"
        assert plan.outcomes == outcomes
        assert plan.difference == difference
        assert plan.prepare_kwargs() == {
            "phase_mode": "itt",
            "outcomes": outcomes,
        }
        assert plan.factory_kwargs() == {
            "outcomes": outcomes,
            "use_age_gp": False,
            "partial_pool_age_gp": True,
            "use_residual_correlation": False,
            "use_cross_baselines": False,
            "use_age_linear": True,
        }
        assert plan.diagnostic_vars() == [
            "alpha",
            "tau",
            "gamma_own",
            "kappa",
            "gamma_A",
        ]
        assert plan.joint_structure == "factorised_outcome_marginals"
        assert plan.loo_unit == "child"
        for field in _META_FIELDS:
            assert isinstance(plan.as_dict()[field], str) and plan.as_dict()[field]

        metadata = plan.difference_metadata()
        if difference is None:
            assert metadata is None
        else:
            assert metadata is not None
            assert "contrast_kind" in metadata
            assert "dependence_note" in metadata


# ---------------------------------------------------------------------------
# 2026-08-22 ITT audit regressions (issue #577, finding 3)
# ---------------------------------------------------------------------------


def _dependence_trace(*, post_sd: float, prior_sd: float = 1 / 3, seed: int = 0):
    """A two-outcome trace carrying the dependence block's reported parameters."""
    import arviz as az
    import numpy as np
    import xarray as xr

    rng = np.random.default_rng(seed)
    nc, nd, npr = 4, 2000, 2000
    coords_post = {"chain": np.arange(nc), "draw": np.arange(nd), "outcome_pair": ["UE|TE"], "outcome": ["TE", "UE"]}
    posterior = xr.Dataset(
        {
            "u_corr_pair": (("chain", "draw", "outcome_pair"), rng.normal(0.0, post_sd, size=(nc, nd, 1))),
            "sigma_outcome": (("chain", "draw", "outcome"), np.abs(rng.normal(0.15, 0.10, size=(nc, nd, 2)))),
        },
        coords=coords_post,
    )
    coords_prior = {"chain": [0], "draw": np.arange(npr), "outcome_pair": ["UE|TE"], "outcome": ["TE", "UE"]}
    prior = xr.Dataset(
        {
            "u_corr_pair": (("chain", "draw", "outcome_pair"), rng.normal(0.0, prior_sd, size=(1, npr, 1))),
            "sigma_outcome": (("chain", "draw", "outcome"), np.abs(rng.normal(0.0, 0.5, size=(1, npr, 2)))),
        },
        coords=coords_prior,
    )
    # ``az.from_dict`` renames labelled dims to ``<var>_dim_0``; build the tree
    # directly so ``outcome_pair`` / ``outcome`` survive, as they do on a real fit.
    del az
    return xr.DataTree.from_dict({"posterior": posterior, "prior": prior})


def test_dependence_summary_flags_a_correlation_that_never_left_its_prior():
    """The registered companions' correlation posterior *is* their prior.

    Their prose invited the reader to treat the companion's interval as the
    data's verdict on within-child covariance. Posterior-to-prior SD ratios of
    1.002, 1.008 and 1.001 say otherwise, and the table has to make that legible
    rather than leaving it to be reconstructed from a wide interval.
    """
    frame = R.dependence_identification_summary(
        _dependence_trace(post_sd=1 / 3), ci_prob=0.89
    )
    correlation = frame.loc[frame["role"] == "residual correlation"].iloc[0]
    assert correlation["prior_source"] == "fitted prior draws"
    assert correlation["posterior_prior_sd_ratio"] == pytest.approx(1.0, abs=0.05)
    assert correlation["verdict"] == "prior-dominated"
    # The residual SDs are a different story, and the table must say so per
    # parameter rather than with one verdict for the whole block.
    assert (frame.loc[frame["role"] == "residual SD", "verdict"] == "informed").all()


def test_dependence_summary_reports_an_informed_correlation_as_informed():
    frame = R.dependence_identification_summary(
        _dependence_trace(post_sd=0.05), ci_prob=0.89
    )
    correlation = frame.loc[frame["role"] == "residual correlation"].iloc[0]
    assert correlation["posterior_prior_sd_ratio"] < 0.75
    assert correlation["verdict"] == "informed"


def test_dependence_summary_is_none_without_the_block():
    import arviz as az
    import numpy as np
    import xarray as xr

    posterior = xr.Dataset(
        {"tau": (("chain", "draw"), np.zeros((2, 10)))},
        coords={"chain": np.arange(2), "draw": np.arange(10)},
    )
    del az
    assert R.dependence_identification_summary(
        xr.DataTree.from_dict({"posterior": posterior}), ci_prob=0.89
    ) is None


def test_the_lkj_prior_sd_closed_form_matches_the_two_outcome_case():
    """Verified against draws; only used when no prior group was persisted.

    For d = 2 the closed form and this environment's sampler agree exactly. For
    d > 2 they do not — ``pm.LKJCorr(n=3, eta=4)`` yields off-diagonal SDs of
    0.316, 0.302 and 0.301 where a true LKJ is exchangeable — which is why the
    summary prefers the fit's own prior draws.
    """
    from language_reading_predictors.statistical_models.priors import (
        JOINT_RESIDUAL_LKJ_ETA,
        residual_correlation_prior_sd,
    )

    assert JOINT_RESIDUAL_LKJ_ETA == 4.0
    assert residual_correlation_prior_sd(2) == pytest.approx(1 / 3, abs=1e-12)
    with pytest.raises(ValueError):
        residual_correlation_prior_sd(1)
