# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registry-wide construction checks for the randomised ITT model family.

These tests deliberately stop before prior or posterior sampling.  Their job is
to catch drift between a registered ``ModelSpec``, preprocessing, and the PyMC
factory: every advertised ITT or joint model must be constructible from the
study data through the same effective options used by the fitting pipeline.
"""

from __future__ import annotations

import pytest

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.factories import (
    build_itt_model,
    build_joint_model,
)
from language_reading_predictors.statistical_models.itt import (
    IttModelSettings,
    resolve_itt_run_plan,
)
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    restrict_to_baseline_floored,
)
from language_reading_predictors.statistical_models.registry import discover_models


def _registered_specs() -> list[tuple[str, ModelSpec]]:
    specs = []
    for model_id, module in discover_models().items():
        spec = getattr(module, "SPEC", None)
        if isinstance(spec, ModelSpec) and spec.kind in {"itt", "joint"}:
            specs.append((model_id, spec))
    return specs


def _prepare_itt(spec: ModelSpec):
    plan = resolve_itt_run_plan(spec)
    return load_and_prepare(**plan.prepare_kwargs())


def _build_itt(spec: ModelSpec):
    plan = resolve_itt_run_plan(spec)
    prepared = _prepare_itt(spec)
    adjust_for = tuple(
        name for name in plan.adjust_for if name in prepared.covariates
    )
    common = plan.factory_kwargs(effective_adjustment=adjust_for)
    if plan.floor_rule:
        at_risk = restrict_to_baseline_floored(prepared, spec.outcome_symbol)
        primary = build_itt_model(
            at_risk, likelihood="bernoulli_offfloor", **common
        )
        secondary = build_itt_model(prepared, likelihood="beta_binomial", **common)
        return primary, secondary
    return (build_itt_model(prepared, **common),)


def _build_joint(spec: ModelSpec):
    outcomes = tuple(spec.extra.get("outcomes") or ITT_OUTCOMES)
    prepared = load_and_prepare(phase_mode="itt", outcomes=outcomes)
    return build_joint_model(
        prepared,
        outcomes=outcomes,
        use_age_gp=spec.extra.get("use_age_gp", False),
        partial_pool_age_gp=spec.extra.get("partial_pool_age_gp", True),
        use_residual_correlation=spec.extra.get("use_residual_correlation", False),
        use_cross_baselines=spec.extra.get("use_cross_baselines", True),
        use_age_linear=spec.extra.get("use_age_linear", False),
    )


_REGISTERED_SPECS = _registered_specs()


@pytest.mark.parametrize(
    "model_id,spec", _REGISTERED_SPECS, ids=[model_id for model_id, _ in _REGISTERED_SPECS]
)
def test_registered_itt_family_model_builds(model_id: str, spec: ModelSpec):
    if spec.kind == "itt":
        assert isinstance(spec.model_settings, IttModelSettings), model_id
        assert spec.extra == {}, model_id
        built_models = _build_itt(spec)
    else:
        built_models = (_build_joint(spec),)

    for built in built_models:
        free_names = {variable.name for variable in built.model.free_RVs}
        observed_names = {variable.name for variable in built.model.observed_RVs}
        assert "tau" in free_names, model_id
        assert observed_names & {"y_post", "y_offfloor"}, model_id
        assert built.prepared.n_obs > 0, model_id
