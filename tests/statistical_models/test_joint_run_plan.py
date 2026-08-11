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
    assert plan.diagnostic_vars() == [
        "alpha",
        "tau",
        "gamma_own",
        "kappa",
        "sigma_outcome",
    ]


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
