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
    extra = spec.extra
    outcomes = extra.get("outcomes")
    adjust_for = tuple(extra.get("adjust_for", ()))
    kwargs = {
        "phase_mode": "itt",
        "covariates": adjust_for,
        "restrict_complete": tuple(extra.get("restrict_complete", ())),
        "drop_missing_pre": bool(extra.get("drop_missing_pre", True)),
        "pre_required": extra.get("pre_required"),
    }
    if outcomes is not None:
        kwargs["outcomes"] = tuple(outcomes)
    return load_and_prepare(**kwargs)


def _build_itt(spec: ModelSpec):
    extra = spec.extra
    prepared = _prepare_itt(spec)
    adjust_for = tuple(
        name for name in extra.get("adjust_for", ()) if name in prepared.covariates
    )
    common = {
        "outcome_symbol": spec.outcome_symbol,
        "use_age_gp": extra.get("use_age_gp", False),
        "use_own_baseline_gp": extra.get("use_own_baseline_gp", False),
        "use_varying_tau": extra.get("use_varying_tau", False),
        "adjust_for": adjust_for,
        "cross_symbols": extra.get("cross_symbols"),
        "use_age_linear": extra.get("use_age_linear", False),
        "use_own_baseline": extra.get("use_own_baseline", True),
        "tau_moderator_symbol": extra.get("tau_moderator_symbol"),
        "tau_moderator_is_covariate": extra.get(
            "tau_moderator_is_covariate", False
        ),
        "tau_moderator_interaction": extra.get("tau_moderator_interaction", True),
        "tau_sigma": extra.get("tau_sigma"),
        "alpha_sigma": extra.get("alpha_sigma"),
        "gamma_own_sigma": extra.get("gamma_own_sigma"),
    }
    if extra.get("floor_rule", False):
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
        built_models = _build_itt(spec)
    else:
        built_models = (_build_joint(spec),)

    for built in built_models:
        free_names = {variable.name for variable in built.model.free_RVs}
        observed_names = {variable.name for variable in built.model.observed_RVs}
        assert "tau" in free_names, model_id
        assert observed_names & {"y_post", "y_offfloor"}, model_id
        assert built.prepared.n_obs > 0, model_id
