# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for joint-mechanism models.

The family has two deliberately different designs: per-wave bivariate levels
fits with an observation-row residual correlation, and a phase-stacked ANCOVA
with a bivariate child intercept.  This module replaces the free-form
``ModelSpec.extra`` boundary for both designs and resolves their data, factory,
diagnostic and reporting contracts before a fit context is created or RLI data
are loaded (#394 pillar 4).

The migration does not change the fitted equations or scientific warrant.  The
letter-sound slopes, decoding-specificity contrast and conditional-slope ratio
remain adjusted associations rather than causal mechanism effects.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.preprocessing import (
    split_covariates_by_wave,
)

__all__ = [
    "JointMechanismModelSettings",
    "JointMechanismRunPlan",
    "declared_joint_mechanism_settings",
    "resolve_joint_mechanism_run_plan",
]


_DESIGNS = frozenset({"levels", "transition"})
_GLOBAL_KEYS = frozenset({"target_accept"})
_FAMILY_KEYS = frozenset(
    {
        "design",
        "outcome_symbols",
        "contrast",
        "confounder_symbols",
        "include_group",
        "covariates",
        "adjust_for",
        "predictor_slope_sigma",
    }
)
_LEGACY_KEYS = _FAMILY_KEYS | _GLOBAL_KEYS


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    if any(not isinstance(item, str) or not item for item in out):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate symbols: {out!r}")
    return out


def _optional_positive_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number or None")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number or None")
    return out


@dataclass(frozen=True, slots=True)
class JointMechanismModelSettings:
    """Immutable declaration for one bivariate joint-mechanism model."""

    design: Literal["levels", "transition"] = "levels"
    outcome_symbols: tuple[str, ...] = ("W", "N")
    contrast: tuple[str, ...] = ("N", "W")
    confounder_symbols: tuple[str, ...] = ("G", "A")
    include_group: bool = True
    covariates: tuple[str, ...] = ()
    adjust_for: tuple[str, ...] = ()
    predictor_slope_sigma: float | None = None

    def __post_init__(self) -> None:
        if self.design not in _DESIGNS:
            raise ValueError(
                f"design must be 'levels' or 'transition', got {self.design!r}"
            )
        for name in (
            "outcome_symbols",
            "contrast",
            "confounder_symbols",
            "covariates",
            "adjust_for",
        ):
            object.__setattr__(
                self,
                name,
                _tuple_of_strings(getattr(self, name), name=name),
            )
        if not isinstance(self.include_group, bool):
            raise TypeError("include_group must be a boolean")
        object.__setattr__(
            self,
            "predictor_slope_sigma",
            _optional_positive_float(
                self.predictor_slope_sigma,
                name="predictor_slope_sigma",
            ),
        )

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> JointMechanismModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown joint-mechanism setting(s): "
                f"{', '.join(unknown)}. Declare JointMechanismModelSettings so "
                "misspellings fail fast."
            )
        return cls(
            design=extra.get("design", "levels"),
            outcome_symbols=extra.get("outcome_symbols", ("W", "N")),
            contrast=extra.get("contrast", ("N", "W")),
            confounder_symbols=extra.get("confounder_symbols", ("G", "A")),
            include_group=extra.get("include_group", True),
            covariates=extra.get("covariates", ()),
            adjust_for=extra.get("adjust_for", ()),
            predictor_slope_sigma=extra.get("predictor_slope_sigma"),
        )


@dataclass(frozen=True, slots=True)
class JointMechanismRunPlan:
    """Concrete, validated instructions for either joint-mechanism design."""

    model_id: str
    settings_source: str
    study_id: str
    design: Literal["levels", "transition"]
    mechanism_symbol: str
    outcome_symbols: tuple[str, str]
    contrast: tuple[str, str]
    confounder_symbols: tuple[str, ...]
    include_group: bool
    declared_adjustment: tuple[str, ...]
    active_adjustment: tuple[str, ...]
    predictor_slope_sigma: float | None
    phase_mode: Literal["levels", "all"]
    pre_covariates: tuple[str, ...]
    post_covariates: tuple[str, ...]
    likelihood: Literal["binomial", "beta_binomial"]
    joint_dependence: Literal["lkj_residual_within_wave", "lkj_child_intercept"]
    observation_node: str
    compute_loo: bool
    loo_unit: str
    min_wave_rows: int | None
    matched_comparators: tuple[str, str]
    design_description: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def with_active_adjustment(
        self, active_adjustment: tuple[str, ...]
    ) -> JointMechanismRunPlan:
        """Record constant requested terms removed on the fitted rows."""
        unknown = sorted(set(active_adjustment) - set(self.declared_adjustment))
        if unknown:
            raise ValueError(
                f"active joint-mechanism adjustment was not declared: {unknown!r}"
            )
        return replace(self, active_adjustment=active_adjustment)

    @property
    def fits_group_nuisance(self) -> bool:
        """Whether the model includes its design-specific group term."""
        return self.include_group or "G" in self.confounder_symbols

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan."""
        kwargs: dict[str, Any] = {
            "phase_mode": self.phase_mode,
            "outcomes": (*self.outcome_symbols, self.mechanism_symbol),
        }
        if self.design == "levels":
            kwargs["baseline_covariates"] = self.declared_adjustment
        else:
            kwargs["covariates"] = self.pre_covariates
            kwargs["post_covariates"] = self.post_covariates
        return kwargs

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_joint_mechanism_model`` from the same plan."""
        kwargs: dict[str, Any] = {
            "design": self.design,
            "mechanism_symbol": self.mechanism_symbol,
            "outcome_symbols": self.outcome_symbols,
            "contrast": self.contrast,
            "adjust_for": self.active_adjustment,
            "confounder_symbols": self.confounder_symbols,
            "include_group": self.include_group,
        }
        if self.design == "levels":
            assert self.predictor_slope_sigma is not None
            kwargs["predictor_slope_sigma"] = self.predictor_slope_sigma
        return kwargs

    def diagnostic_vars(self, available_names: set[str]) -> list[str]:
        """Reported parameters present in a built model, in stable gate order."""
        names = ["alpha", "beta_mech", "delta_ls_decoding"]
        if self.fits_group_nuisance:
            names.append(
                "beta_group_nuisance" if self.design == "levels" else "beta_G"
            )
        if "A" in self.confounder_symbols:
            names.append("gamma_A")
        names.extend(f"gamma_{name}" for name in self.active_adjustment)
        if self.design == "levels":
            names.extend(
                [
                    "sigma_u_resid",
                    "rho_outcome",
                    "beta_mech_focal_given_held",
                    "share_retained",
                ]
            )
        else:
            names.extend(
                [
                    "gamma_own",
                    "alpha_phase",
                    "kappa",
                    "sigma_u_child",
                    "rho_outcome",
                ]
            )
        return [name for name in names if name in available_names]

    def psense_vars(self, available_names: set[str]) -> list[str]:
        """Stable parameters suitable for power-scaling sensitivity."""
        candidates = ["beta_mech", "delta_ls_decoding", "rho_outcome"]
        if self.design == "levels":
            candidates.append("beta_mech_focal_given_held")
        return [name for name in candidates if name in available_names]

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language account generated from the validated run plan."""
        adjusters = ", ".join(self.declared_adjustment) or "none"
        return (
            "Note: Generated from the validated joint-mechanism run plan; template "
            "drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design_description}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"The standardised `{self.mechanism_symbol}` exposure is fitted jointly "
            f"against `{self.outcome_symbols[0]}` and "
            f"`{self.outcome_symbols[1]}`. The reported contrast is "
            f"`beta({self.contrast[0]}) - beta({self.contrast[1]})`. Declared "
            f"adjustment terms: {adjusters}. Confounder flags: "
            f"{', '.join(self.confounder_symbols) or 'none'}. Likelihood: "
            f"`{self.likelihood}`; dependence block: `{self.joint_dependence}`.\n\n"
            "## Uncertainty and checks\n\n"
            f"The observation node is `{self.observation_node}` and PSIS-LOO uses "
            f"the `{self.loo_unit}` unit. Interpret the posterior only after every "
            "published fit passes the zero-divergence convergence gate, predictive "
            "checks and power-scaling sensitivity diagnostics. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_joint_mechanism_settings(
    spec: ModelSpec,
) -> tuple[JointMechanismModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: joint-mechanism settings cannot be split between "
                f"model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, JointMechanismModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='joint_mechanism' requires "
                f"JointMechanismModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        JointMechanismModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
        ),
        "legacy_extra",
    )


def resolve_joint_mechanism_run_plan(spec: ModelSpec) -> JointMechanismRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "joint_mechanism":
        raise ValueError(
            f"{spec.model_id}: expected kind 'joint_mechanism', got {spec.kind!r}"
        )
    if spec.study_id != "rli":
        raise ValueError(
            f"{spec.model_id}: joint_mechanism requires study_id='rli', got "
            f"{spec.study_id!r}"
        )
    mechanism_symbol = spec.mechanism_symbol or "L"
    if not isinstance(mechanism_symbol, str) or not mechanism_symbol:
        raise TypeError("joint_mechanism mechanism_symbol must be a non-empty string")

    settings, source = declared_joint_mechanism_settings(spec)
    if len(settings.outcome_symbols) != 2:
        raise ValueError("joint_mechanism requires exactly two outcome_symbols")
    outcome_symbols = settings.outcome_symbols
    if mechanism_symbol in outcome_symbols:
        raise ValueError("mechanism_symbol must differ from both outcome_symbols")
    if len(settings.contrast) != 2 or set(settings.contrast) != set(outcome_symbols):
        raise ValueError(
            "contrast must contain the two outcome_symbols exactly once"
        )
    unknown_confounders = sorted(set(settings.confounder_symbols) - {"G", "A"})
    if unknown_confounders:
        raise ValueError(
            "confounder_symbols supports only the group and age flags G/A; got "
            f"{unknown_confounders!r}"
        )

    if settings.design == "levels":
        if settings.adjust_for:
            raise ValueError("adjust_for is transition-only; use covariates for levels")
        adjustment = settings.covariates
        slope_sigma = settings.predictor_slope_sigma or 0.3
        pre_covariates: tuple[str, ...] = ()
        post_covariates: tuple[str, ...] = ()
        phase_mode: Literal["levels", "all"] = "levels"
        likelihood: Literal["binomial", "beta_binomial"] = "binomial"
        dependence: Literal[
            "lkj_residual_within_wave", "lkj_child_intercept"
        ] = "lkj_residual_within_wave"
        min_wave_rows = 10
        comparators = ("lrp-rli-ca-010", "lrp-rli-ca-011")
        design_description = (
            "Separate cross-sectional bivariate fits at each RLI wave. Both bounded "
            "outcomes share the same standardised mechanism exposure and an LKJ "
            "observation-row residual block; the best-populated latest wave is the "
            "diagnostic anchor and every other usable wave is a recorded sub-fit."
        )
        estimand = (
            "Per-wave mechanism slopes for both outcomes, their within-model "
            "decoding-specificity difference, the residual outcome correlation, and "
            "the share of the focal slope retained after partialling the other latent "
            "outcome."
        )
        population = (
            "At each of four waves, archived RLI children with the mechanism score "
            "and at least one outcome observed, plus all retained baseline trait "
            "covariates. Waves with fewer than ten usable children are named and "
            "skipped rather than fitted."
        )
    else:
        if settings.covariates:
            raise ValueError("covariates is levels-only; use adjust_for for transition")
        if settings.predictor_slope_sigma is not None:
            raise ValueError("predictor_slope_sigma is levels-only")
        adjustment = settings.adjust_for
        slope_sigma = None
        pre_covariates, post_covariates = split_covariates_by_wave(adjustment)
        phase_mode = "all"
        likelihood = "beta_binomial"
        dependence = "lkj_child_intercept"
        min_wave_rows = None
        comparators = ("lrp-rli-mech-096", "lrp-rli-mech-101")
        design_description = (
            "One phase-stacked bivariate ANCOVA over the three RLI transitions. Each "
            "outcome retains its own baseline and phase intercepts, while an LKJ "
            "bivariate child intercept represents between-child outcome dependence."
        )
        estimand = (
            "The two adjusted mechanism slopes conditional on each outcome's own "
            "baseline and their within-model decoding-specificity difference on the "
            "same parameterisation as the two matched mechanism models."
        )
        population = (
            "Available RLI transition rows with the mechanism exposure, at least one "
            "outcome, both outcome baselines and all retained requested covariates."
        )

    return JointMechanismRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        design=settings.design,
        mechanism_symbol=mechanism_symbol,
        outcome_symbols=(outcome_symbols[0], outcome_symbols[1]),
        contrast=(settings.contrast[0], settings.contrast[1]),
        confounder_symbols=settings.confounder_symbols,
        include_group=settings.include_group,
        declared_adjustment=adjustment,
        active_adjustment=adjustment,
        predictor_slope_sigma=slope_sigma,
        phase_mode=phase_mode,
        pre_covariates=pre_covariates,
        post_covariates=post_covariates,
        likelihood=likelihood,
        joint_dependence=dependence,
        observation_node="y_post",
        compute_loo=True,
        loo_unit="child",
        min_wave_rows=min_wave_rows,
        matched_comparators=comparators,
        design_description=design_description,
        estimand=estimand,
        causal_status=(
            "Adjusted association only. Randomised group is a nuisance term here; "
            "unobserved general ability can still confound the mechanism-outcome "
            "slopes, and the dependence block does not repair that backdoor path."
        ),
        analysis_population=population,
        missing_data_assumption=(
            "The focal mechanism is never imputed. A row enters when that exposure "
            "and at least one outcome are observed, subject to the design-specific "
            "baseline and covariate requirements. Outcome-specific missing cells are "
            "masked, under ignorable missingness conditional on fitted terms."
        ),
    )
