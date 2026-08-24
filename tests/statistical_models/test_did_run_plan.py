# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed difference-in-differences settings and run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import os

import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.did import (
    DiDModelSettings,
    DiDRunPlan,
    resolve_did_run_plan,
)

_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _did_specs() -> list[ModelSpec]:
    """Every registered difference-in-differences model's SPEC."""
    root = os.path.dirname(
        importlib.import_module(
            "language_reading_predictors.statistical_models.did"
        ).__file__
    )
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_did_*.py"))):
        mod = importlib.import_module(
            "language_reading_predictors.statistical_models." + os.path.basename(path)[:-3]
        )
        spec = getattr(mod, "SPEC", None)
        if spec is not None and spec.kind == "did":
            specs.append(spec)
    return specs


# --- settings validation ------------------------------------------------------


def test_settings_reject_unknown_likelihood():
    with pytest.raises(ValueError, match="likelihood"):
        DiDModelSettings(likelihood="poisson")


def test_settings_reject_non_bool_dose():
    with pytest.raises(TypeError, match="dose"):
        DiDModelSettings(dose=1)  # type: ignore[arg-type]


def test_settings_reject_string_outcomes():
    with pytest.raises(TypeError, match="outcomes"):
        DiDModelSettings(outcomes="W")  # type: ignore[arg-type]


def test_settings_reject_non_int_waves():
    with pytest.raises(TypeError, match="waves"):
        DiDModelSettings(waves=(0, "1"))  # type: ignore[list-item]


def test_settings_reject_period_varying_without_dose():
    with pytest.raises(ValueError, match="period_varying_dose requires dose"):
        DiDModelSettings(period_varying_dose=True)  # dose defaults False


# --- #576 lower-severity 4 and 5: fail before any I/O -------------------------


def test_settings_reject_boolean_prior_widths():
    """``bool`` is an ``int`` subclass, so ``True`` used to become ``1.0`` silently."""
    for name in (
        "tau_t2_prior_sigma",
        "arm_gap_t1_prior_sigma",
        "sigma_child_prior_sigma",
        "kappa_prior_sigma",
    ):
        with pytest.raises(TypeError, match="bool is not a prior width"):
            DiDModelSettings(**{name: True})


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_settings_reject_non_positive_prior_widths(value):
    with pytest.raises(ValueError, match="finite and positive"):
        DiDModelSettings(tau_t2_prior_sigma=value)


def test_settings_reject_a_binary_window_the_factory_cannot_fit():
    with pytest.raises(ValueError, match=r"requires waves=\(0, 1, 2\)"):
        DiDModelSettings(waves=(0, 1))


def test_settings_reject_waves_on_a_dose_model():
    """A dose variant never reads ``waves``; declaring one must not be ignored."""
    with pytest.raises(ValueError, match="never\s+reads"):
        DiDModelSettings(dose=True, waves=(0, 1))


def test_settings_reject_a_guessing_floor_on_a_dose_or_off_floor_model():
    with pytest.raises(ValueError, match="no score mean to map"):
        DiDModelSettings(
            likelihood="bernoulli_offfloor",
            score_mean_link="three_choice_guessing_floor",
        )
    with pytest.raises(ValueError, match="response-link sensitivity"):
        DiDModelSettings(dose=True, score_mean_link="three_choice_guessing_floor")


def test_settings_reject_a_kappa_prior_on_the_off_floor_branch():
    with pytest.raises(ValueError, match="no dispersion parameter"):
        DiDModelSettings(
            likelihood="bernoulli_offfloor",
            kappa_prior_family="halfnormal_inverse_sqrt",
        )


def test_resolution_rejects_outcomes_omitting_the_focal_outcome():
    """The one column the model cannot be built without (#576 lower-severity 4).

    This used to resolve cleanly and fail inside the factory with ``KeyError:
    Outcome 'W' missing from prepared data`` — after ``make_context`` had reset an
    output directory and the loader had read the whole panel.
    """
    spec = ModelSpec(
        model_id="lrp-rli-did-999",
        kind="did",
        title="t",
        outcome_symbol="W",
        family="did",
        design="d",
        estimand_type="mixed",
        causal_status="c",
        extra={"outcomes": ("L",)},
    )
    with pytest.raises(ValueError, match="do not include the focal outcome"):
        resolve_did_run_plan(spec)


def test_resolution_rejects_a_guessing_floor_on_a_non_blending_outcome():
    spec = ModelSpec(
        model_id="lrp-rli-did-999",
        kind="did",
        title="t",
        outcome_symbol="W",
        family="did",
        design="d",
        estimand_type="mixed",
        causal_status="c",
        model_settings=DiDModelSettings(
            outcomes=("W",), score_mean_link="three_choice_guessing_floor"
        ),
    )
    with pytest.raises(ValueError, match="property of the phoneme-blending items"):
        resolve_did_run_plan(spec)


def test_dose_recipe_names_periods_not_inherited_waves():
    """#576 lower-severity 3: a dose recipe must not print ``waves=(0, 1, 2)``."""
    spec = ModelSpec(
        model_id="lrp-rli-did-999",
        kind="did",
        title="t",
        outcome_symbol="L",
        family="did",
        design="d",
        estimand_type="association",
        causal_status="none",
        model_settings=DiDModelSettings(outcomes=("L",), dose=True, periods=(0, 1)),
    )
    recipe = resolve_did_run_plan(spec).recipe_markdown(title="t")
    assert "Periods: P1, P2" in recipe
    assert "Waves:" not in recipe


def test_every_registered_plan_names_one_focal_estimand():
    """#576 finding 1: the published quantity is recorded, not inferred."""
    for spec in _did_specs():
        plan = resolve_did_run_plan(spec)
        assert plan.focal_estimand
        assert plan.focal_estimand_scale == "natural"
        assert plan.focal_estimand_artifact.endswith(".csv")
        recorded = plan.as_dict()
        assert recorded["focal_estimand"] == plan.focal_estimand
        assert recorded["run_plan_digest"] == plan.run_plan_digest


def test_from_legacy_extra_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown DiD setting"):
        DiDModelSettings.from_legacy_extra(
            {"dose": True, "doze": True},  # typo
            model_id="lrp-rli-did-999",
        )


def test_from_legacy_extra_round_trips_the_dose_keys():
    # Split from the binary case below because dose excludes bernoulli_offfloor and
    # use_varying_delta (#455); one settings object cannot carry every non-default at
    # once and still describe a model the factory would build.
    settings = DiDModelSettings.from_legacy_extra(
        {
            "dose": True,
            "period_varying_dose": True,
            "likelihood": "beta_binomial",
            "outcomes": ("W",),
            # ``waves`` stays at the default: a dose model fits the P1/P2 transition
            # frame and never reads it, so a non-default declaration is rejected
            # rather than silently ignored (#576 lower-severity 4, below).
            "periods": (0, 1),
            "use_child_re": True,
            "use_age": False,
        },
        model_id="lrp-rli-did-999",
    )
    assert settings.dose is True
    assert settings.period_varying_dose is True
    assert settings.outcomes == ("W",)
    assert settings.waves == (0, 1, 2)
    assert settings.periods == (0, 1)
    assert settings.use_age is False


def test_from_legacy_extra_round_trips_the_binary_keys():
    settings = DiDModelSettings.from_legacy_extra(
        {
            "dose": False,
            "likelihood": "bernoulli_offfloor",
            "outcomes": ("W", "L"),
            "waves": (0, 1, 2),
            "use_child_re": True,
            "use_varying_delta": True,
        },
        model_id="lrp-rli-did-999",
    )
    assert settings.dose is False
    assert settings.likelihood == "bernoulli_offfloor"
    assert settings.outcomes == ("W", "L")
    assert settings.waves == (0, 1, 2)
    assert settings.use_child_re is True
    assert settings.use_varying_delta is True


def test_settings_reject_off_floor_with_dose():
    with pytest.raises(ValueError, match="bernoulli_offfloor is the binary prevalence"):
        DiDModelSettings(dose=True, likelihood="bernoulli_offfloor")


def test_settings_reject_varying_delta_with_dose():
    with pytest.raises(ValueError, match="use_varying_delta is unavailable for dose"):
        DiDModelSettings(dose=True, use_varying_delta=True)


def test_settings_reject_varying_delta_without_a_child_random_intercept():
    with pytest.raises(ValueError, match="use_varying_delta=True requires use_child_re"):
        DiDModelSettings(use_varying_delta=True, use_child_re=False)


def test_settings_reject_dose_with_non_transition_periods():
    with pytest.raises(ValueError, match=r"dose variants require periods=\(0, 1\)"):
        DiDModelSettings(dose=True, periods=(0, 1, 2))


# --- resolve ------------------------------------------------------------------


def _spec(**extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-did-000",
        kind="did",
        title="test",
        outcome_symbol="W",
        extra=extra,
    )


def test_resolve_rejects_wrong_kind():
    spec = ModelSpec(model_id="x", kind="itt", title="t", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'did'"):
        resolve_did_run_plan(spec)


def test_resolve_binary_is_levels_frame_and_tau_t2_estimand():
    plan = resolve_did_run_plan(_spec())
    assert plan.settings_source == "legacy_extra"
    assert not plan.dose and not plan.period_varying and not plan.off_floor
    assert plan.effect_term == "tau_t2"
    assert plan.obs_node == "y_post"
    assert "randomised" in plan.causal_status
    assert "tau_t2" in plan.estimand
    prep = plan.prepare_kwargs()
    assert prep["phase_mode"] == "levels"
    assert prep["outcomes"] == ("W",)  # defaults to the outcome symbol
    assert prep["require_any_post"] is False
    assert "covariates" not in prep  # binary loads no session covariate


def test_resolve_dose_is_transition_frame_with_attend_and_associational():
    plan = resolve_did_run_plan(_spec(dose=True))
    assert plan.dose and not plan.period_varying
    assert plan.effect_term == "beta_dose"
    prep = plan.prepare_kwargs()
    assert prep["phase_mode"] == "all"
    assert prep["covariates"] == ("attend",)
    assert prep["pre_required"] == ()
    assert plan.causal_status.startswith("Associational")
    assert "observational" in plan.estimand.lower()


def test_resolve_period_varying_dose_focal_term():
    plan = resolve_did_run_plan(_spec(dose=True, period_varying_dose=True))
    assert plan.period_varying
    assert plan.effect_term == "mu_dose"
    # The factory receives the *resolved* period_varying under period_varying_dose.
    assert plan.factory_kwargs()["period_varying_dose"] is True


def test_resolve_off_floor_sets_bernoulli_node():
    plan = resolve_did_run_plan(_spec(likelihood="bernoulli_offfloor"))
    assert plan.off_floor
    assert plan.obs_node == "y_offfloor"
    assert plan.factory_kwargs()["likelihood"] == "bernoulli_offfloor"


def test_resolve_explicit_outcomes_are_kept():
    plan = resolve_did_run_plan(_spec(outcomes=("W", "L")))
    assert plan.prepare_kwargs()["outcomes"] == ("W", "L")


def test_typed_settings_are_accepted_and_sourced():
    spec = ModelSpec(
        model_id="lrp-rli-did-000",
        kind="did",
        title="test",
        outcome_symbol="W",
        model_settings=DiDModelSettings(dose=True),
    )
    plan = resolve_did_run_plan(spec)
    assert plan.settings_source == "typed"
    assert plan.dose is True


def test_split_settings_between_typed_and_extra_is_rejected():
    spec = ModelSpec(
        model_id="lrp-rli-did-000",
        kind="did",
        title="test",
        outcome_symbol="W",
        model_settings=DiDModelSettings(),
        extra={"dose": True},
    )
    with pytest.raises(ValueError, match="cannot be split"):
        resolve_did_run_plan(spec)


# --- registered-specification coverage (acceptance criterion) -----------------


def test_every_registered_did_model_resolves_with_metadata():
    """Every registered DiD model resolves to a validated plan that records the
    design, estimand, causal status, analysis population and missing-data
    assumption (#394 pillar 4)."""
    specs = _did_specs()
    assert len(specs) >= 14, f"expected the full DiD suite, found {len(specs)}"
    saw_dose = saw_binary = False
    for spec in specs:
        plan = resolve_did_run_plan(spec)
        assert isinstance(plan, DiDRunPlan)
        recorded = plan.as_dict()
        for field in _META_FIELDS:
            assert isinstance(recorded[field], str) and recorded[field], (
                f"{spec.model_id}: {field} not recorded"
            )
        assert plan.prepare_kwargs()["outcomes"][0] == spec.outcome_symbol
        saw_dose |= plan.dose
        saw_binary |= not plan.dose
    assert saw_dose, "no dose DiD model found"
    assert saw_binary, "no binary DiD model found"


# --- power-scaling coverage of variant-defining terms (#390 P2) ---------------


def test_psense_terms_cover_the_focal_effect_on_a_plain_binary_model():
    plan = resolve_did_run_plan(_spec())
    assert plan.psense_terms == ("tau_t2",)


def test_psense_terms_add_the_period_varying_dose_structure():
    """DID-007's variant *is* its period-resolved dose slope, so leaving
    ``sigma_dose`` / ``beta_dose_phase`` unmeasured showed a reader no flag on the
    thing the model exists to claim."""
    plan = resolve_did_run_plan(_spec(dose=True, period_varying_dose=True))
    assert plan.psense_terms == ("mu_dose", "sigma_dose", "beta_dose_phase")


def test_psense_terms_add_the_waitlist_catch_up_scale():
    """DID-013's ``sigma_delta`` is informed by one t3 observation per waitlist
    child, which is exactly where a prior-dominated posterior is most likely."""
    plan = resolve_did_run_plan(_spec(use_varying_delta=True))
    assert plan.psense_terms == ("tau_t2", "sigma_delta")


def test_psense_terms_stop_at_variant_defining_parameters():
    """Nuisance scales stay out on purpose: at n ~ 54 they flag suite-wide and
    would bury the rows a reader should act on."""
    plan = resolve_did_run_plan(_spec(use_varying_delta=True))
    assert "kappa" not in plan.psense_terms
    assert "sigma_child" not in plan.psense_terms


def test_every_registered_did_model_power_scales_its_variant_terms():
    """No registered variant may report a structure it never power-scaled."""
    for spec in _did_specs():
        plan = resolve_did_run_plan(spec)
        assert plan.psense_terms[0] == plan.effect_term, spec.model_id
        assert len(set(plan.psense_terms)) == len(plan.psense_terms), spec.model_id
        if plan.period_varying:
            assert {"sigma_dose", "beta_dose_phase"} <= set(plan.psense_terms), (
                spec.model_id
            )
        if plan.use_varying_delta:
            assert "sigma_delta" in plan.psense_terms, spec.model_id


def test_tau_t2_prior_sigma_flows_to_factory_kwargs():
    """#382 rec 3 (LRPDID102): the widened causal-term prior is a typed setting
    that reaches build_did_model, and stays None on the reference model."""
    from language_reading_predictors.statistical_models import lrp_rli_did_002, lrp_rli_did_102
    from language_reading_predictors.statistical_models.did import resolve_did_run_plan

    ref = resolve_did_run_plan(lrp_rli_did_002.SPEC)
    wide = resolve_did_run_plan(lrp_rli_did_102.SPEC)
    assert ref.factory_kwargs()["tau_t2_prior_sigma"] is None
    assert wide.factory_kwargs()["tau_t2_prior_sigma"] == 1.0
    # Identical apart from the sensitivity knob.
    a, b = ref.factory_kwargs(), wide.factory_kwargs()
    a.pop("tau_t2_prior_sigma"), b.pop("tau_t2_prior_sigma")
    assert {k: v for k, v in a.items() if k != "outcome_symbol"} == {
        k: v for k, v in b.items() if k != "outcome_symbol"
    }


def test_use_intercept_anchor_flows_to_factory_kwargs():
    """#390 P1 condition 1 (LRPDID101): the independent-prior intercept is a
    typed setting that reaches build_did_model, and stays True (anchored) on
    the reference model — with no other difference between the two specs."""
    from language_reading_predictors.statistical_models import (
        lrp_rli_did_001,
        lrp_rli_did_101,
    )
    from language_reading_predictors.statistical_models.did import resolve_did_run_plan

    ref = resolve_did_run_plan(lrp_rli_did_001.SPEC)
    free = resolve_did_run_plan(lrp_rli_did_101.SPEC)
    assert ref.factory_kwargs()["use_intercept_anchor"] is True
    assert free.factory_kwargs()["use_intercept_anchor"] is False
    assert free.effect_term == "tau_t2"
    a, b = ref.factory_kwargs(), free.factory_kwargs()
    a.pop("use_intercept_anchor"), b.pop("use_intercept_anchor")
    assert a == b


def test_use_intercept_anchor_rejected_on_dose_models():
    """The dose variants already build a free intercept, so the companion
    setting there would claim a change that is not one."""
    from language_reading_predictors.statistical_models.did import DiDModelSettings

    with pytest.raises(ValueError, match="free intercept"):
        DiDModelSettings(dose=True, use_intercept_anchor=False)


def test_tau_t2_prior_sigma_rejected_on_dose_and_bad_values():
    """A dose model has no tau_t2, so the setting is incoherent there; and the
    scale must be a positive finite number (settings-time, before any IO)."""
    import pytest

    from language_reading_predictors.statistical_models.did import DiDModelSettings

    with pytest.raises(ValueError, match="no tau_t2"):
        DiDModelSettings(dose=True, tau_t2_prior_sigma=1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        DiDModelSettings(tau_t2_prior_sigma=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        DiDModelSettings(tau_t2_prior_sigma=float("nan"))
