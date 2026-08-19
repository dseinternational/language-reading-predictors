# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contracts for the wave-pooled level-association family (``kind='pooled_levels'``)."""

from __future__ import annotations

import numpy as np
import pytest

from language_reading_predictors.statistical_models import pooled_levels as P
from language_reading_predictors.statistical_models.context import ModelSpec


def _spec(**extra) -> ModelSpec:
    base = {
        "adjust_for": ("hs", "hs_missing"),
        "ability_covariate": "blocks",
    }
    base.update(extra)
    return ModelSpec(
        model_id="lrp-rli-pl-000",
        kind="pooled_levels",
        title="test pooled levels",
        outcome_symbol="W",
        mechanism_symbol="L",
        extra=base,
    )


def test_ability_covariate_broadcasts_from_t1_not_per_row():
    """Block design is recorded once at t1; a per-row pull is NaN after t1."""
    plan = P.resolve_pooled_levels_run_plan(_spec())
    kwargs = plan.prepare_kwargs()

    assert kwargs["phase_mode"] == "levels"
    assert kwargs["baseline_covariates"] == ("blocks",)
    assert "blocks" not in kwargs["post_covariates"]


def test_exposure_and_outcome_must_differ():
    with pytest.raises(ValueError, match="trivially 1"):
        P.resolve_pooled_levels_run_plan(
            ModelSpec(
                model_id="lrp-rli-pl-000",
                kind="pooled_levels",
                title="degenerate",
                outcome_symbol="W",
                mechanism_symbol="W",
                extra={},
            )
        )


def test_pooling_without_a_child_random_intercept_is_refused():
    """The defect this family exists to avoid must not be reachable by setting."""
    with pytest.raises(ValueError, match="understates"):
        P.PooledLevelsModelSettings(use_subject_random_intercept=False)


def test_single_wave_cannot_ask_for_wave_intercepts():
    with pytest.raises(ValueError, match="at least two waves"):
        P.PooledLevelsModelSettings(waves=(2,), use_subject_random_intercept=True)


def test_unknown_setting_fails_fast():
    with pytest.raises(ValueError, match="unknown pooled_levels setting"):
        P.PooledLevelsModelSettings.from_extra({"nonsense": 1}, model_id="x")


def test_decomposition_is_the_default_and_names_the_between_term_focal():
    plan = P.resolve_pooled_levels_run_plan(_spec())

    assert plan.decompose_between_within is True
    assert plan.focal_term == "beta_between"
    assert {"beta_between", "beta_within"} <= set(plan.diagnostic_vars(("hs",)))


def test_blended_variant_reports_a_single_slope():
    plan = P.resolve_pooled_levels_run_plan(_spec(decompose_between_within=False))

    assert plan.focal_term == "beta_mech"
    names = set(plan.diagnostic_vars(("hs",)))
    assert "beta_mech" in names and "beta_between" not in names


def test_causal_status_refuses_a_causal_reading():
    plan = P.resolve_pooled_levels_run_plan(_spec())

    assert "Association only" in plan.causal_status
    assert "contemporaneous" in plan.causal_status


def test_between_and_within_regressors_are_orthogonal_by_construction():
    """The Mundlak split must give a child mean and a mean-zero within deviation."""
    rng = np.random.default_rng(0)
    child_idx = np.repeat(np.arange(20), 4)
    x = rng.normal(size=child_idx.size)
    bar = np.zeros_like(x)
    for c in np.unique(child_idx):
        m = child_idx == c
        bar[m] = x[m].mean()
    dev = x - bar

    for c in np.unique(child_idx):
        assert dev[child_idx == c].sum() == pytest.approx(0.0, abs=1e-12)
    assert np.corrcoef(bar, dev)[0, 1] == pytest.approx(0.0, abs=1e-8)


def test_estimand_names_the_coefficients_the_posterior_carries():
    """config.json must never name a coefficient the fitted model lacks: the
    decomposed fit has ``beta_between``/``beta_within`` and no ``beta_mech``."""
    split = P.resolve_pooled_levels_run_plan(_spec())
    assert "beta_between" in split.estimand and "beta_within" in split.estimand
    assert "beta_mech" not in split.estimand

    blended = P.resolve_pooled_levels_run_plan(_spec(decompose_between_within=False))
    assert blended.estimand.startswith("beta_mech")
    assert "beta_between" not in blended.estimand


def test_priors_table_presents_nothing_in_the_family_as_causal():
    """``beta_G`` reuses the tau constructor, so without an override the priors
    table would label a term pooled over post-crossover waves as "causal"; the
    exposure slopes reuse the beta_mech constructor under names the name-based
    lookup does not know."""
    from types import SimpleNamespace

    from language_reading_predictors.statistical_models.prior_artifacts import (
        _prior_table_overrides,
    )

    spec = _spec()
    ctx = SimpleNamespace(
        spec=spec,
        resolved_plan=P.resolve_pooled_levels_run_plan(spec),
        model=SimpleNamespace(
            free_RVs=[
                SimpleNamespace(name=n)
                for n in ("beta_between", "beta_within", "beta_G", "gamma_hs",
                          "gamma_hs_missing", "gamma_blocks", "gamma_A", "alpha_wave")
            ]
        ),
    )
    ctor, role, rationale = _prior_table_overrides(ctx)
    assert "causal" not in role.values()
    assert role["beta_G"] == "association"
    assert "not the randomised treatment effect" in rationale["beta_G"]
    assert ctor["beta_between"] == ctor["beta_within"] == "beta_mech"
    assert role["alpha_wave"] == "nuisance"
    assert role["gamma_hs"] == role["gamma_blocks"] == "association"
    # missing-indicator terms keep the universal nuisance treatment; age keeps
    # its own precision-covariate constructor and role
    assert role["gamma_hs_missing"] == "nuisance"
    assert "gamma_A" not in role


def test_model_recipe_is_written_for_the_family(tmp_path):
    """Every typed-plan family writes model_recipe.md; the reporting dispatch now
    resolves a pooled-levels plan, so the plan must be able to render one."""
    from types import SimpleNamespace

    from language_reading_predictors.statistical_models.reporting import (
        write_model_recipe,
    )

    spec = _spec()
    ctx = SimpleNamespace(spec=spec, resolved_plan=None, output_dir=str(tmp_path))
    path = write_model_recipe(ctx)
    assert path is not None
    text = (tmp_path / "model_recipe.md").read_text(encoding="utf-8")
    assert "beta_between" in text and "beta_within" in text
    assert "adjusted association" in text
    assert "blocks" in text and "hs" in text


# --- #553: covariate exposures and same-wave skill adjusters --------------------


def _covariate_spec(mech="erbto", **extra) -> ModelSpec:
    base = {
        "adjust_for": ("hs", "hs_missing"),
        "ability_covariate": "blocks",
        "mechanism_is_covariate": True,
        "require_observed": (mech,),
    }
    base.update(extra)
    return ModelSpec(
        model_id="lrp-rli-pl-000",
        kind="pooled_levels",
        title="test covariate exposure",
        outcome_symbol="W",
        mechanism_symbol=mech,
        extra=base,
    )


def test_covariate_exposure_plan_loads_the_raw_score_complete_case():
    """A raw-score exposure (#553) loads as a same-wave covariate with its
    missingness flag, is complete-case through ``require_observed``, is not a
    requested outcome, and is excluded from the ``gamma_`` adjuster set."""
    plan = P.resolve_pooled_levels_run_plan(_covariate_spec())
    assert plan.mechanism_is_covariate
    assert plan.exposure_kind == "raw_covariate"
    assert plan.require_observed == ("erbto",)
    kwargs = plan.prepare_kwargs()
    assert kwargs["outcomes"] == ("W",)
    assert "erbto" in kwargs["post_covariates"]
    assert "erbto_missing" in kwargs["post_covariates"]
    assert kwargs["require_observed"] == ("erbto",)
    assert plan.factory_kwargs()["mechanism_is_covariate"] is True
    diag = plan.diagnostic_vars(("hs", "erbto", "blocks"))
    assert "gamma_erbto" not in diag
    assert "gamma_hs" in diag and "gamma_blocks" in diag
    assert plan.exposure_label.startswith("phonological memory")
    assert "standardised raw score" in plan.estimand
    assert "complete-case" in plan.missing_data_assumption
    recipe = plan.recipe_markdown(title="t")
    assert "standardised raw score" in recipe
    assert "`require_observed`" in recipe and "`erbto`" in recipe


def test_covariate_exposure_must_be_complete_case_and_not_a_measure():
    """The exposure itself is never imputed: ``require_observed`` must name it;
    a bounded measure cannot be declared raw; an unsupported raw covariate is
    refused; and a raw symbol without the flag is refused — all before any I/O."""
    with pytest.raises(ValueError, match="must be declared in require_observed"):
        P.resolve_pooled_levels_run_plan(_covariate_spec(require_observed=()))
    with pytest.raises(ValueError, match="cannot be declared as a raw covariate"):
        P.resolve_pooled_levels_run_plan(
            _covariate_spec(mech="L", require_observed=())
        )
    with pytest.raises(ValueError, match="not a supported filled covariate"):
        P.resolve_pooled_levels_run_plan(
            _covariate_spec(mech="attend", require_observed=("attend",))
        )
    with pytest.raises(ValueError, match="unknown measure symbol"):
        P.resolve_pooled_levels_run_plan(
            _covariate_spec(mechanism_is_covariate=False, require_observed=())
        )
    with pytest.raises(ValueError, match="must not also appear in adjust_for"):
        P.resolve_pooled_levels_run_plan(
            _covariate_spec(adjust_for=("hs", "hs_missing", "erbto", "erbto_missing"))
        )


def test_skill_adjusters_are_loaded_as_same_wave_measures():
    """Same-wave skill adjusters (#553) load as further outcomes in the levels
    frame, get a ``gamma_<symbol>`` coefficient each, and are recorded in the
    estimand prose; the outcome, the exposure and a bounded ``adjust_for`` entry
    are refused."""
    plan = P.resolve_pooled_levels_run_plan(_spec(skill_symbols=("TR", "TE", "R")))
    assert plan.skill_symbols == ("TR", "TE", "R")
    kwargs = plan.prepare_kwargs()
    assert kwargs["outcomes"] == ("W", "L", "TR", "TE", "R")
    assert plan.factory_kwargs()["skill_symbols"] == ("TR", "TE", "R")
    diag = plan.diagnostic_vars(("hs", "blocks"))
    for sym in ("TR", "TE", "R"):
        assert f"gamma_{sym}" in diag
    assert "skill adjusters" in plan.design
    assert "Table-2 fallacy" in plan.causal_status
    with pytest.raises(ValueError, match="is the outcome"):
        P.resolve_pooled_levels_run_plan(_spec(skill_symbols=("W",)))
    with pytest.raises(ValueError, match="is the exposure"):
        P.resolve_pooled_levels_run_plan(_spec(skill_symbols=("L",)))
    with pytest.raises(ValueError, match="unknown skill adjuster"):
        P.resolve_pooled_levels_run_plan(_spec(skill_symbols=("ZZ",)))
    with pytest.raises(ValueError, match="skill_symbols, not adjust_for"):
        P.resolve_pooled_levels_run_plan(_spec(adjust_for=("hs", "hs_missing", "TR")))
    with pytest.raises(ValueError, match="must not repeat"):
        P.PooledLevelsModelSettings(skill_symbols=("TR", "TR"))


def test_typed_settings_and_extra_cannot_be_mixed():
    with pytest.raises(ValueError, match="cannot be split"):
        P.resolve_pooled_levels_run_plan(
            ModelSpec(
                model_id="lrp-rli-pl-000",
                kind="pooled_levels",
                title="t",
                outcome_symbol="W",
                mechanism_symbol="L",
                model_settings=P.PooledLevelsModelSettings(ability_covariate="blocks"),
                extra={"adjust_for": ("hs", "hs_missing")},
            )
        )


def test_registered_pl_003_to_006_resolve_as_the_issue_specifies():
    """#553: E and R are bounded-count exposures with same-wave skill adjusters
    (the mechanism adjustment sets minus the own baseline); erbto and deapp_c
    are raw-score covariate exposures, complete-case on the exposure; all four
    target word reading with blocks and age and no attend."""
    import importlib

    expected = {
        "lrp_rli_pl_003": ("E", "bounded_count", ("TR", "TE", "R"), (), ("hs", "hs_missing", "erbto", "erbto_missing", "deapp_c", "deapp_c_missing")),
        "lrp_rli_pl_004": ("R", "bounded_count", ("TR",), (), ("hs", "hs_missing", "erbto", "erbto_missing")),
        "lrp_rli_pl_005": ("erbto", "raw_covariate", (), ("erbto",), ("hs", "hs_missing")),
        "lrp_rli_pl_006": ("deapp_c", "raw_covariate", (), ("deapp_c",), ("hs", "hs_missing", "erbto", "erbto_missing")),
    }
    for name, (mech, kind, skills, required, adjust) in expected.items():
        spec = importlib.import_module(
            f"language_reading_predictors.statistical_models.{name}"
        ).SPEC
        plan = P.resolve_pooled_levels_run_plan(spec)
        assert plan.outcome_symbol == "W", name
        assert plan.mechanism_symbol == mech, name
        assert plan.exposure_kind == kind, name
        assert plan.skill_symbols == skills, name
        assert plan.require_observed == required, name
        assert plan.adjust_for == adjust, name
        assert plan.ability_covariate == "blocks", name
        assert "attend" not in plan.adjust_for, name
        assert plan.use_wave_intercepts and plan.decompose_between_within, name
        assert plan.settings_source == "typed_settings", name


def test_priors_table_documents_skill_adjusters_and_raw_exposures():
    from types import SimpleNamespace

    from language_reading_predictors.statistical_models.prior_artifacts import (
        _prior_table_overrides,
    )

    spec = _spec(skill_symbols=("TR",))
    ctx = SimpleNamespace(
        spec=spec,
        resolved_plan=P.resolve_pooled_levels_run_plan(spec),
        model=SimpleNamespace(
            free_RVs=[SimpleNamespace(name=n) for n in ("beta_between", "gamma_TR", "gamma_hs")]
        ),
    )
    ctor, role, rationale = _prior_table_overrides(ctx)
    assert ctor["gamma_TR"] == "gamma_cross"
    assert role["gamma_TR"] == "association"
    assert "same-wave logit of TR" in rationale["gamma_TR"]
    assert "exposure logit" in rationale["beta_between"]

    cspec = _covariate_spec()
    cctx = SimpleNamespace(
        spec=cspec,
        resolved_plan=P.resolve_pooled_levels_run_plan(cspec),
        model=SimpleNamespace(free_RVs=[SimpleNamespace(name="beta_between")]),
    )
    _, _, crationale = _prior_table_overrides(cctx)
    assert "exposure raw score" in crationale["beta_between"]
