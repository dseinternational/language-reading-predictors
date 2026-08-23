# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the typed mechanism settings and pre-I/O run plan (#394 pillar 4)."""

from __future__ import annotations

import glob
import importlib
import inspect
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models import mechanism as M
from language_reading_predictors.statistical_models import reporting as R
from language_reading_predictors.statistical_models.context import ModelSpec

_META_FIELDS = (
    "design",
    "estimand",
    "causal_status",
    "analysis_population",
    "missing_data_assumption",
)


def _spec(*, settings=None, adjustment=None, **extra) -> ModelSpec:
    return ModelSpec(
        model_id="lrp-rli-mech-000",
        kind="mechanism",
        title="test mechanism",
        outcome_symbol="W",
        mechanism_symbol="L",
        adjustment=["G", "A", "W_pre"] if adjustment is None else adjustment,
        model_settings=settings,
        extra=extra,
    )


def _mechanism_specs() -> list[ModelSpec]:
    """Every registered RLI mechanism model's ``SPEC``."""
    root = os.path.dirname(M.__file__)
    specs: list[ModelSpec] = []
    for path in sorted(glob.glob(os.path.join(root, "lrp_rli_mech_*.py"))):
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + os.path.basename(path)[:-3]
        )
        spec = getattr(module, "SPEC", None)
        if spec is not None and spec.kind == "mechanism":
            specs.append(spec)
    return specs


def _registered_spec(model_id: str) -> ModelSpec:
    """One registered mechanism spec, failing loudly if the catalogue drifts."""
    matches = [spec for spec in _mechanism_specs() if spec.model_id == model_id]
    assert len(matches) == 1, model_id
    return matches[0]


# --- settings validation ------------------------------------------------------


def test_settings_reject_unknown_legacy_key():
    with pytest.raises(ValueError, match="unknown mechanism setting.*use_age_gpp"):
        M.MechanismModelSettings.from_legacy_extra(
            {"use_age_gpp": True}, model_id="lrp-rli-mech-999"
        )


def test_settings_accept_global_target_accept_without_owning_it():
    settings = M.MechanismModelSettings.from_legacy_extra(
        {"target_accept": 0.999, "linear_mechanism": True},
        model_id="lrp-rli-mech-999",
    )
    assert settings.linear_mechanism is True
    assert "target_accept" not in settings.__dataclass_fields__


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"use_age_gp": 1}, "use_age_gp"),
        ({"mech_hsgp_m": 0}, "mech_hsgp_m"),
        ({"mech_hsgp_m": 2.5}, "mech_hsgp_m"),
        ({"items_ref_quantiles": (0.8, 0.2)}, "items_ref_quantiles"),
        ({"items_ref_quantiles": (0.25,)}, "items_ref_quantiles"),
        ({"moderator_is_covariate": True}, "requires moderator_symbol"),
        (
            {"linear_mechanism": True, "mech_hsgp_m": 6},
            "cannot declare HSGP",
        ),
        (
            {"linear_mechanism": True, "phase_specific_mechanism": True},
            "cannot be combined with phase_specific_mechanism",
        ),
    ],
)
def test_settings_reject_misshaped_or_contradictory_values(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        M.MechanismModelSettings(**kwargs)


def test_settings_coerce_legacy_sequences_without_changing_values():
    settings = M.MechanismModelSettings.from_legacy_extra(
        {
            "outcomes": ["W", "L"],
            "adjust_for": ["hs", "hs_missing"],
            "items_ref_quantiles": [0.2, 0.8],
        },
        model_id="lrp-rli-mech-999",
    )
    assert settings.outcomes == ("W", "L")
    assert settings.adjust_for == ("hs", "hs_missing")
    assert settings.items_ref_quantiles == (0.2, 0.8)


# --- pure resolution ----------------------------------------------------------


def test_resolve_rejects_wrong_kind_and_missing_required_symbols():
    wrong = ModelSpec(model_id="x", kind="itt", title="x", outcome_symbol="W")
    with pytest.raises(ValueError, match="expected kind 'mechanism'"):
        M.resolve_mechanism_run_plan(wrong)

    no_outcome = ModelSpec(
        model_id="x", kind="mechanism", title="x", mechanism_symbol="L"
    )
    with pytest.raises(ValueError, match="outcome_symbol is required"):
        M.resolve_mechanism_run_plan(no_outcome)

    no_exposure = ModelSpec(
        model_id="x", kind="mechanism", title="x", outcome_symbol="W"
    )
    with pytest.raises(ValueError, match="mechanism_symbol is required"):
        M.resolve_mechanism_run_plan(no_exposure)


@pytest.mark.parametrize(
    "spec",
    [
        _spec(outcomes=("W", "L", "ZZ")),
        _spec(
            outcomes=("W", "L", "ZZ"),
            adjust_baseline_symbol="ZZ",
            adjustment=["G", "A", "ZZ_pre"],
        ),
        replace(_spec(outcomes=("W", "YY")), mechanism_symbol="YY"),
        _spec(outcomes=("W", "L", "QQ"), moderator_symbol="QQ"),
        replace(
            _spec(
                outcomes=("ZZ", "L"),
                adjust_baseline_symbol="ZZ",
                adjustment=["G", "A", "ZZ_pre"],
            ),
            outcome_symbol="ZZ",
        ),
    ],
)
def test_resolve_rejects_unknown_bounded_measure_symbols(spec):
    with pytest.raises(ValueError, match="unrecognised bounded measure symbol"):
        M.resolve_mechanism_run_plan(spec)


def test_measure_exposure_plan_reproduces_loader_factory_and_diagnostics_contract():
    plan = M.resolve_mechanism_run_plan(
        _spec(
            outcomes=("W", "L", "TR"),
            adjust_baseline_symbol="W",
            adjust_for=("hs", "hs_missing"),
            linear_mechanism=True,
            adjustment=["G", "A", "TR", "W_pre"],
        )
    )
    assert plan.settings_source == "legacy_extra"
    assert plan.likelihood == "beta_binomial"
    assert plan.observation_node == "y_post"
    assert plan.prepare_kwargs() == {
        "phase_mode": "all",
        "covariates": (),
        "post_covariates": ("hs", "hs_missing"),
        "baseline_covariates": (),
        "require_observed": (),
        # Only the autoregressive baseline: the exposure (L) and the measure
        # confounder (TR) are contemporaneous post scores the model never reads at
        # pre, so complete-casing on their baselines dropped rows for nothing
        # (#586 finding 4).
        "pre_required": ("W",),
        "outcomes": ("W", "L", "TR"),
    }
    factory = plan.factory_kwargs()
    assert factory["mechanism_symbol"] == "L"
    assert factory["outcome_symbol"] == "W"
    assert factory["confounder_symbols"] == ("G", "A", "TR")
    assert factory["linear_mechanism"] is True
    assert factory["adjust_for"] == ("hs", "hs_missing")
    assert factory["mech_lengthscale_prior"] is None
    assert plan.diagnostic_vars() == [
        "alpha",
        "beta_G",
        "gamma_own",
        "kappa",
        "gamma_TR",
        "gamma_hs",
        "gamma_hs_missing",
        "gamma_A",
        "sigma_child",
        "beta_mech",
    ]


def test_covariate_exposure_complete_case_loads_parent_and_indicator():
    spec = ModelSpec(
        model_id="lrp-rli-mech-000",
        kind="mechanism",
        title="test",
        outcome_symbol="N",
        mechanism_symbol="erbto",
        adjustment=["G", "A", "N_pre"],
        extra={
            "outcomes": ("N",),
            "adjust_baseline_symbol": "N",
            "adjust_for": ("hs", "hs_missing"),
            "require_observed": ("erbto",),
            "mechanism_is_covariate": True,
            "linear_mechanism": True,
        },
    )
    plan = M.resolve_mechanism_run_plan(spec)
    prep = plan.prepare_kwargs()
    loaded = set(prep["covariates"]) | set(prep["post_covariates"])
    assert {"hs", "hs_missing", "erbto", "erbto_missing"} <= loaded
    assert prep["require_observed"] == ("erbto",)
    assert plan.factory_kwargs()["mechanism_is_covariate"] is True


def test_covariate_moderator_complete_case_loads_parent_and_indicator():
    plan = M.resolve_mechanism_run_plan(
        _spec(
            outcomes=("W", "L"),
            adjust_for=("hs", "hs_missing"),
            moderator_symbol="erbto",
            moderator_is_covariate=True,
            require_observed=("erbto",),
        )
    )
    prep = plan.prepare_kwargs()
    loaded = set(prep["covariates"]) | set(prep["post_covariates"])
    assert {"erbto", "erbto_missing"} <= loaded
    assert plan.diagnostic_vars()[-2:] == ["gamma_mod", "gamma_int"]


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"adjust_for": ("hs",)}, "requires companion 'hs_missing'"),
        ({"adjust_for": ("hs_missing",)}, "orphan missingness indicator"),
        ({"adjust_for": ("fake_missing",)}, "unsupported missingness indicator"),
        (
            {
                "outcomes": ("W",),
                "mechanism_is_covariate": True,
                "linear_mechanism": True,
            },
            "filled covariate exposure 'erbto'.*require_observed",
        ),
        (
            {
                "outcomes": ("W", "L"),
                "moderator_symbol": "erbto",
                "moderator_is_covariate": True,
            },
            "filled covariate moderator 'erbto'.*require_observed",
        ),
    ],
)
def test_missing_covariate_policy_rejects_unqualified_filled_values(extra, match):
    mechanism_symbol = "erbto" if extra.get("mechanism_is_covariate") else "L"
    with pytest.raises(ValueError, match=match):
        M.resolve_mechanism_run_plan(
            replace(_spec(**extra), mechanism_symbol=mechanism_symbol)
        )


def test_complete_case_adjuster_auto_loads_filter_indicator_without_fitting_it():
    plan = M.resolve_mechanism_run_plan(
        _spec(
            adjust_for=("hs",),
            require_observed=("hs",),
        )
    )
    assert plan.require_observed == ("hs",)
    assert set(plan.prepare_kwargs()["post_covariates"]) == {"hs", "hs_missing"}
    assert plan.factory_kwargs()["adjust_for"] == ("hs",)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        (
            M.MechanismModelSettings(
                mechanism_is_covariate=True, mechanism_at_pre=True
            ),
            "mechanism_at_pre is incompatible",
        ),
        (
            M.MechanismModelSettings(
                mechanism_is_covariate=True, adjust_for=("L",)
            ),
            "must not also appear in adjust_for",
        ),
        (
            M.MechanismModelSettings(
                moderator_symbol="erbto",
                moderator_is_covariate=True,
                adjust_for=("erbto",),
            ),
            "moderator.*must not also appear",
        ),
        (
            M.MechanismModelSettings(require_observed=("erbto",)),
            "not loaded by the mechanism plan",
        ),
    ],
)
def test_resolve_rejects_cross_field_contradictions(settings, match):
    with pytest.raises(ValueError, match=match):
        M.resolve_mechanism_run_plan(_spec(settings=settings))


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (
            _spec(
                outcomes=("W", "L"),
                mechanism_is_covariate=True,
                linear_mechanism=True,
            ),
            "bounded measure exposure 'L'.*raw covariate",
        ),
        (
            _spec(
                outcomes=("W", "L", "B"),
                moderator_symbol="B",
                moderator_is_covariate=True,
            ),
            "bounded measure moderator 'B'.*raw covariate",
        ),
        (
            _spec(outcomes=("W", "L"), adjust_for=("R",)),
            "bounded measure adjuster.*raw adjust_for: R",
        ),
    ],
)
def test_bounded_measures_cannot_use_raw_covariate_roles(spec, match):
    with pytest.raises(ValueError, match=match):
        M.resolve_mechanism_run_plan(spec)


@pytest.mark.parametrize(
    ("adjustment", "match"),
    [
        (["A", "W_pre"], "must declare 'G'"),
        (["G", "A"], "autoregressive baseline 'W_pre'"),
        (["G", "A", "L_pre"], "autoregressive baseline 'W_pre'"),
        (["G", "A", "W_pre", "L_pre"], "exactly.*'W_pre'"),
    ],
)
def test_adjustment_must_match_always_fitted_group_and_baseline(adjustment, match):
    with pytest.raises(ValueError, match=match):
        M.resolve_mechanism_run_plan(_spec(adjustment=adjustment))


def test_outcome_subset_must_cover_every_bounded_model_term():
    with pytest.raises(ValueError, match=r"omit required mechanism measure\(s\): TR"):
        M.resolve_mechanism_run_plan(
            _spec(outcomes=("W", "L"), adjustment=["G", "A", "TR", "W_pre"])
        )


def test_typed_settings_are_accepted_and_cannot_be_split_with_extra():
    settings = M.MechanismModelSettings(
        outcomes=("W", "L"), linear_mechanism=True
    )
    plan = M.resolve_mechanism_run_plan(_spec(settings=settings))
    assert plan.settings_source == "typed"
    assert plan.linear_mechanism is True

    mixed = _spec(settings=settings, use_age_gp=True)
    with pytest.raises(ValueError, match="cannot be split"):
        M.resolve_mechanism_run_plan(mixed)


def test_typed_and_legacy_declarations_resolve_to_the_same_design():
    values = {
        "outcomes": ("W", "L"),
        "adjust_baseline_symbol": "W",
        "adjust_for": ("hs", "hs_missing"),
        "linear_mechanism": True,
        "items_ref_quantiles": (0.2, 0.8),
    }
    legacy = M.resolve_mechanism_run_plan(_spec(**values))
    typed = M.resolve_mechanism_run_plan(
        _spec(settings=M.MechanismModelSettings(**values))
    )
    assert replace(legacy, settings_source="typed") == typed


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (
            "lrp-rli-mech-058",
            {
                "linear_mechanism": False,
                "mechanism_is_covariate": False,
                "moderator_symbol": None,
                "mech_hsgp_m": 6,
                "mech_lengthscale_tight": True,
                "require_observed": (),
            },
        ),
        (
            "lrp-rli-mech-090",
            {
                "linear_mechanism": True,
                "mechanism_is_covariate": True,
                "moderator_symbol": None,
                "mech_hsgp_m": None,
                "require_observed": ("erbto",),
            },
        ),
        (
            "lrp-rli-mech-104",
            {
                "linear_mechanism": False,
                "mechanism_is_covariate": False,
                "moderator_symbol": "erbto",
                "mech_hsgp_m": None,
                "require_observed": ("erbto",),
            },
        ),
        (
            "lrp-rli-mech-158",
            {
                "linear_mechanism": False,
                "mechanism_is_covariate": False,
                "moderator_symbol": None,
                # Matched to its mech-058 baseline in #586 finding 5; it used to
                # inherit the shared defaults (m=10, InverseGamma(5, 5)) and so
                # differed from its own comparator in functional form.
                "mech_hsgp_m": 6,
                "mech_lengthscale_tight": True,
                "require_observed": ("hs", "deapp_c"),
            },
        ),
        (
            "lrp-rli-mech-191",
            {
                "linear_mechanism": False,
                "mechanism_is_covariate": True,
                "moderator_symbol": None,
                "mech_hsgp_m": 6,
                "mech_lengthscale_tight": True,
                "require_observed": (),
            },
        ),
    ],
)
def test_registered_branch_contracts_reach_loader_factory_and_diagnostics(
    model_id, expected
):
    """Lock representative registered branches at the pure plan boundary."""
    plan = M.resolve_mechanism_run_plan(_registered_spec(model_id))
    prepare = plan.prepare_kwargs()
    factory = plan.factory_kwargs()
    diagnostics = plan.diagnostic_vars()

    for key, value in expected.items():
        assert getattr(plan, key) == value, (model_id, key)
    assert factory["linear_mechanism"] is expected["linear_mechanism"]
    assert factory["mechanism_is_covariate"] is expected["mechanism_is_covariate"]
    assert factory["moderator_symbol"] == expected["moderator_symbol"]
    assert factory["mech_hsgp_m"] == expected["mech_hsgp_m"]
    assert prepare["require_observed"] == expected["require_observed"]

    if expected.get("mech_lengthscale_tight"):
        prior = factory["mech_lengthscale_prior"]
        assert type(prior).__name__ == "InverseGamma"
        assert tuple(float(value) for value in prior.params) == (8.0, 8.0)
    else:
        assert factory["mech_lengthscale_prior"] is None

    if expected["linear_mechanism"]:
        assert "beta_mech" in diagnostics
    else:
        assert "beta_mech" not in diagnostics
    if expected["moderator_symbol"] is not None:
        assert diagnostics[-2:] == ["gamma_mod", "gamma_int"]


def test_lagged_measure_exposure_is_preserved_in_factory_contract():
    plan = M.resolve_mechanism_run_plan(
        _spec(
            outcomes=("W", "L"),
            mechanism_at_pre=True,
            linear_mechanism=True,
        )
    )
    factory = plan.factory_kwargs()
    assert plan.mechanism_at_pre is True
    assert plan.mechanism_is_covariate is False
    assert factory["mechanism_at_pre"] is True
    assert factory["mechanism_is_covariate"] is False


# --- pre-I/O ordering, reporting and coverage --------------------------------


def test_invalid_setting_fails_before_context_reset_or_data_loading(monkeypatch):
    from language_reading_predictors.statistical_models.pipelines import mechanism as P

    touched = {"context": False, "data": False}

    def _context(*args, **kwargs):
        touched["context"] = True
        raise AssertionError("make_context must not run")

    def _data(*args, **kwargs):
        touched["data"] = True
        raise AssertionError("load_and_prepare must not run")

    monkeypatch.setattr(P, "make_context", _context)
    monkeypatch.setattr(M, "load_and_prepare", _data)
    bad = _spec(use_age_gpp=True)
    with pytest.raises(ValueError, match="unknown mechanism setting"):
        P.fit_mechanism(bad)
    assert touched == {"context": False, "data": False}


def test_reporting_dispatch_and_recipe_use_the_attached_plan(tmp_path):
    spec = _spec(outcomes=("W", "L"), linear_mechanism=True)
    plan = M.resolve_mechanism_run_plan(spec)
    ctx = SimpleNamespace(spec=spec, resolved_plan=plan, output_dir=str(tmp_path))
    assert R._resolved_run_plan(ctx) is plan
    path = R.write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "validated mechanism run plan" in text
    assert "Associational only" in text
    assert "beta_binomial (`y_post`)" in text


def test_stale_attached_plan_is_rejected_before_data_loading(monkeypatch):
    current = _spec(outcomes=("W", "L"), linear_mechanism=True)
    stale_spec = replace(
        current,
        outcome_symbol="N",
        adjustment=["G", "A", "N_pre"],
        extra={
            "outcomes": ("N", "L"),
            "adjust_baseline_symbol": "N",
            "linear_mechanism": True,
        },
    )
    stale = M.resolve_mechanism_run_plan(stale_spec)
    loaded = False

    def _load(*args, **kwargs):
        nonlocal loaded
        loaded = True
        raise AssertionError("stale plan must fail before loading")

    monkeypatch.setattr(M, "load_and_prepare", _load)
    with pytest.raises(ValueError, match="does not match the current model specification"):
        M.resolve_mechanism_plan(current, run_plan=stale)
    assert loaded is False


def test_reporting_rejects_stale_attached_plan():
    current = _spec(outcomes=("W", "L"), linear_mechanism=True)
    stale_spec = replace(
        current,
        outcome_symbol="N",
        adjustment=["G", "A", "N_pre"],
        extra={
            "outcomes": ("N", "L"),
            "adjust_baseline_symbol": "N",
            "linear_mechanism": True,
        },
    )
    ctx = SimpleNamespace(
        spec=current,
        resolved_plan=M.resolve_mechanism_run_plan(stale_spec),
    )
    with pytest.raises(ValueError, match="does not match the current model specification"):
        R._resolved_run_plan(ctx)


def test_pipeline_has_no_direct_mechanism_setting_reads():
    from language_reading_predictors.statistical_models.pipelines import mechanism as P

    source = inspect.getsource(P)
    assert "spec.extra" not in source
    assert "ctx.spec.extra" not in source


def test_every_registered_mechanism_model_resolves_with_audit_metadata():
    specs = _mechanism_specs()
    # 34 original + the six-model ability-adjusted Tier-1 panel (196-201)
    # + mech-258, the ability-adjusted counterpart of the mech-058 curve.
    assert len(specs) == 41
    for spec in specs:
        plan = M.resolve_mechanism_run_plan(spec)
        assert isinstance(plan, M.MechanismRunPlan)
        recorded = plan.as_dict()
        for field in _META_FIELDS:
            assert isinstance(recorded[field], str) and recorded[field], (
                f"{spec.model_id}: {field} not recorded"
            )
        assert plan.likelihood == "beta_binomial"
        assert plan.observation_node == "y_post"


def test_ability_covariate_loads_from_t1_not_the_post_row():
    """Block design is recorded once at t1, so it must broadcast, not be pulled per row.

    ``load_and_prepare`` reads the raw CSV, where ``blocks`` is populated only at
    time 1. Routing it through ``post_covariates`` makes it NaN on every transition
    and the complete-case filter then drops all 162 rows, which surfaces only as a
    cryptic "Standard deviation of x must be positive" from age standardisation.
    The typed ability setting must therefore reach ``baseline_covariates``.
    """
    plan = M.resolve_mechanism_run_plan(_spec(ability_covariate="blocks"))
    kwargs = plan.prepare_kwargs()

    assert plan.ability_covariate == "blocks"
    assert kwargs["baseline_covariates"] == ("blocks",)
    assert "blocks" not in kwargs["post_covariates"]
    assert "blocks" not in kwargs["covariates"]


def test_ability_covariate_is_absent_by_default():
    plan = M.resolve_mechanism_run_plan(_spec())

    assert plan.ability_covariate is None
    assert plan.prepare_kwargs()["baseline_covariates"] == ()


def test_ability_covariate_becomes_a_fitted_adjustment_coefficient():
    """It loads by a different route but is an ordinary adjustment term in the model."""
    spec = _spec(ability_covariate="blocks", adjust_for=("hs", "hs_missing"))
    plan = M.resolve_mechanism_plan(spec)

    assert "blocks" in plan.prepared.covariates
    assert "blocks" in plan.adjust_for
    assert plan.factory_kwargs["adjust_for"] == plan.adjust_for


# ---------------------------------------------------------------------------
# #586: sample contract, paired comparators, populations and dormant designs.
# ---------------------------------------------------------------------------


def test_pre_required_covers_only_the_scores_the_model_consumes():
    """Complete-casing on an unused pre-score drops rows for nothing.

    The factory's linear predictor reads exactly one period-start score — the
    autoregressive baseline — while the exposure, measure confounders and moderator
    are all contemporaneous post measurements. ``pre_required`` used to be every
    loaded outcome (#586 finding 4).
    """
    plan = M.resolve_mechanism_run_plan(
        _spec(outcomes=("W", "L", "N"), moderator_symbol="N")
    )
    assert plan.pre_required == ("W",)
    assert plan.prepare_kwargs()["pre_required"] == ("W",)


def test_pre_required_adds_the_exposure_only_when_it_is_read_at_pre():
    lagged = M.resolve_mechanism_run_plan(
        _spec(outcomes=("W", "L"), mechanism_at_pre=True)
    )
    assert lagged.pre_required == ("W", "L")


def test_pre_required_baseline_must_be_loaded():
    """The autoregressive baseline is the one pre-score the model reads, so a
    declared outcome set that omits it is rejected before any I/O."""
    with pytest.raises(ValueError, match=r"omit required mechanism measure\(s\): W"):
        M.resolve_mechanism_run_plan(
            _spec(outcomes=("L",), adjust_baseline_symbol="W", adjustment=["G", "A", "W_pre"])
        )


def test_unused_pre_score_no_longer_drops_an_eligible_row():
    """Data-backed row contract for the models the change moves.

    mech-063/163 fit W on L moderated by N. Every fitted term of four transitions was
    observed, but a missing ``N_pre`` — a score the model never reads — removed them,
    taking both fits from 155 rows to 151.
    """
    moderated = M.resolve_mechanism_plan(
        _spec(outcomes=("W", "L", "N"), moderator_symbol="N")
    )
    unmoderated = M.resolve_mechanism_plan(_spec(outcomes=("W", "L")))
    # Requiring the moderator's *post* score is legitimate (the factory fits it);
    # requiring its *pre* score is not, so the two frames now differ only by the
    # rows whose N_post is missing.
    assert moderated.prepared.n_obs >= unmoderated.prepared.n_obs - 5
    assert moderated.run_plan.pre_required == ("W",)


def test_registered_mechanism_row_counts_are_pinned():
    """Every registered model's fitted rows, so a contract change cannot slip through."""
    expected = {
        "lrp-rli-mech-058": (156, 53),
        "lrp-rli-mech-063": (155, 53),  # was 151 before #586 finding 4
        "lrp-rli-mech-163": (155, 53),
        "lrp-rli-mech-158": (128, 44),
        "lrp-rli-mech-191": (128, 52),  # on-intervention only, #586 finding 2
    }
    for model_id, (n_obs, n_children) in expected.items():
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            + model_id.replace("lrp-rli-mech-", "lrp_rli_mech_")
        )
        built = M.build_mechanism_for_plan(M.resolve_mechanism_plan(module.SPEC))
        assert len(built.model.coords["obs_id"]) == n_obs, model_id
        assert len(built.model.coords["child"]) == n_children, model_id


def test_mech_158_differs_from_mech_058_only_by_its_missing_data_policy():
    """The complete-case comparator must isolate one thing (#586 finding 5).

    mech-158 silently omitted mech-058's ``outcomes``, HSGP basis and tight
    lengthscale, so it also differed in loading contract and functional form.
    """
    from dataclasses import asdict

    base = importlib.import_module(
        "language_reading_predictors.statistical_models.lrp_rli_mech_058"
    ).SPEC
    comparator = importlib.import_module(
        "language_reading_predictors.statistical_models.lrp_rli_mech_158"
    ).SPEC
    a = asdict(M.resolve_mechanism_run_plan(base))
    b = asdict(M.resolve_mechanism_run_plan(comparator))
    differing = {k for k in a if a[k] != b[k]}
    assert differing <= {"model_id", "require_observed", "analysis_population"}, differing
    assert b["require_observed"] == ("hs", "deapp_c")


def test_mech_158_prose_does_not_claim_an_unfitted_confounder():
    """Its docstring described a phonological-memory (erbto) restriction it never ran."""
    module = importlib.import_module(
        "language_reading_predictors.statistical_models.lrp_rli_mech_158"
    )
    plan = M.resolve_mechanism_run_plan(module.SPEC)
    assert "erbto" not in plan.adjust_for
    assert "erbto" not in plan.require_observed
    doc = module.__doc__ or ""
    # It may *mention* erbto to correct the record, but never as a live adjuster.
    assert "identical to LRP58 in every respect" in doc.replace("**", "")
    assert "not** in either model's adjustment set" in doc


def test_mech_191_fits_only_on_intervention_periods():
    """Its documented population and its fitted rows must agree (#586 finding 2)."""
    module = importlib.import_module(
        "language_reading_predictors.statistical_models.lrp_rli_mech_191"
    )
    plan = M.resolve_mechanism_plan(module.SPEC)
    prepared = plan.prepared
    scaler = prepared.covariate_scalers["attend"]
    sessions = scaler.inverse(prepared.covariates["attend"])

    assert plan.run_plan.exposure_positive_only is True
    assert sessions.min() > 0, "a zero-session row survived the restriction"
    # The period-1 wait-list arm was entirely at zero sessions, so it must be gone.
    period_one_waitlist = (prepared.phase == 0) & (prepared.G == 0)
    assert not period_one_waitlist.any()


def test_positive_exposure_restriction_is_covariate_only():
    with pytest.raises(ValueError, match="continuous covariate exposure only"):
        M.resolve_mechanism_run_plan(_spec(exposure_positive_only=True))


@pytest.mark.parametrize(
    ("label", "extra", "match"),
    [
        ("outcome is its own exposure", {"outcome_symbol": "W", "mechanism_symbol": "W",
                                         "outcomes": ("W",)}, "regress a measure on itself"),
        ("moderator is the exposure", {"outcomes": ("W", "L"), "moderator_symbol": "L"},
         "exposure squared"),
        ("moderator is the outcome", {"outcomes": ("W", "L"), "moderator_symbol": "W"},
         "moderate its own predictor"),
        ("phase-specific curves", {"outcomes": ("W", "L"), "phase_specific_mechanism": True},
         "not supported"),
        ("age GP plus age moderation", {"outcomes": ("W", "L"), "use_age_gp": True,
                                        "moderator_symbol": "A",
                                        "moderator_is_covariate": True},
         "cannot be combined with age moderation"),
        ("unknown ability covariate", {"outcomes": ("W", "L"),
                                       "ability_covariate": "nonsense_col"},
         "unsupported ability_covariate"),
        ("non-default bounded exposure", {"mechanism_symbol": "TR"},
         "omit required mechanism measure"),
    ],
)
def test_unsupported_designs_fail_before_any_io(label, extra, match):
    """Each of these resolved cleanly and failed late, or not at all (#586 dormant).

    They are rejected in the pure run-plan stage, before ``make_context`` resets an
    output directory and before the loader reads the data.
    """
    outcome = extra.pop("outcome_symbol", "W")
    mechanism = extra.pop("mechanism_symbol", "L")
    spec = ModelSpec(
        model_id="lrp-rli-mech-000",
        kind="mechanism",
        title="test mechanism",
        outcome_symbol=outcome,
        mechanism_symbol=mechanism,
        adjustment=["G", "A", "W_pre"],
        extra=extra,
    )
    with pytest.raises((ValueError, TypeError), match=match):
        M.resolve_mechanism_run_plan(spec)


def test_bounded_exposure_cannot_also_be_a_declared_confounder():
    spec = ModelSpec(
        model_id="lrp-rli-mech-000",
        kind="mechanism",
        title="test mechanism",
        outcome_symbol="W",
        mechanism_symbol="L",
        adjustment=["G", "A", "W_pre", "L"],
        extra={"outcomes": ("W", "L")},
    )
    with pytest.raises(ValueError, match="fitted twice"):
        M.resolve_mechanism_run_plan(spec)


def test_ability_covariate_type_is_validated_before_io():
    with pytest.raises(TypeError, match="non-empty column name"):
        M.resolve_mechanism_run_plan(_spec(ability_covariate=123))


def test_every_registered_mechanism_spec_still_resolves():
    """The new rejections must not catch a model that is legitimately registered."""
    paths = sorted(
        glob.glob(
            os.path.join(
                os.path.dirname(inspect.getfile(M)), "lrp_rli_mech_*.py"
            )
        )
    )
    assert len(paths) >= 41
    for path in paths:
        name = os.path.basename(path)[:-3]
        spec = importlib.import_module(
            f"language_reading_predictors.statistical_models.{name}"
        ).SPEC
        M.resolve_mechanism_run_plan(spec)


def test_moderator_terms_reach_the_fitted_adjustment_record():
    """gamma_mod / gamma_int carry coefficients, so they must be named (#586 finding 9)."""
    from language_reading_predictors.statistical_models.adjustment import (
        effective_adjustment,
    )

    spec = _spec(outcomes=("W", "L"), moderator_symbol="A", moderator_is_covariate=True)
    plan = M.resolve_mechanism_plan(spec)
    record = effective_adjustment(
        spec,
        plan.prepared,
        measure_confounders=("G", "A"),
        adjust_for=plan.adjust_for,
        baseline_symbol="W",
        moderator_symbol=plan.run_plan.moderator_symbol,
        moderator_is_covariate=plan.run_plan.moderator_is_covariate,
        moderator_interaction=plan.run_plan.include_interaction,
    )
    kinds = [t["kind"] for t in record["fitted"]]
    assert kinds.count("moderator_main_effect") == 1
    assert kinds.count("moderator_interaction") == 1
    moderation = [t for t in record["fitted"] if t["kind"].startswith("moderator")]
    # Recorded as moderation, never relabelled as a backdoor confounder.
    assert all(t["moderator"] == "A" for t in moderation)
    assert {t["term"] for t in moderation} == {"gamma_mod", "gamma_int"}


def test_main_effect_only_companion_records_no_interaction():
    from language_reading_predictors.statistical_models.adjustment import (
        effective_adjustment,
    )

    spec = _spec(outcomes=("W", "L", "B"), moderator_symbol="B", include_interaction=False)
    plan = M.resolve_mechanism_plan(spec)
    record = effective_adjustment(
        spec,
        plan.prepared,
        measure_confounders=("G", "A"),
        adjust_for=plan.adjust_for,
        baseline_symbol="W",
        moderator_symbol=plan.run_plan.moderator_symbol,
        moderator_is_covariate=plan.run_plan.moderator_is_covariate,
        moderator_interaction=plan.run_plan.include_interaction,
    )
    kinds = [t["kind"] for t in record["fitted"]]
    assert kinds.count("moderator_main_effect") == 1
    assert "moderator_interaction" not in kinds
