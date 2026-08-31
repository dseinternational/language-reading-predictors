# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for block-exposure models.

The block-exposure family estimates a staggered block-2 teaching association from
the RLI levels panel.  This module replaces its free-form ``ModelSpec.extra``
boundary with immutable settings and a validated plan resolved before an output
transaction is opened or study data are loaded (#394 pillar 4).

The migration is deliberately structural: registered models keep the same rows,
covariate timing, likelihood, priors, fitted equation, diagnostics and artefacts.
``delta`` remains an adjusted association under a parallel-trends assumption, not
a randomised treatment effect.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.preprocessing import (
    split_confounders_by_timing,
    split_covariates_by_wave,
)
from language_reading_predictors.statistical_models.settings_validation import (
    require_declared_booleans,
)

__all__ = [
    "BlockExposureModelSettings",
    "BlockExposureRunPlan",
    "declared_block_exposure_settings",
    "resolve_block_exposure_run_plan",
]


_LEGACY_KEYS = frozenset(
    {
        "ability_covariate",
        "adjust_for",
        "use_child_re",
        "likelihood",
        "drop_ceiling_violations",
        "delta_prior_sigma",
        # Global sampler option resolved by ``make_context`` rather than this family.
        "target_accept",
    }
)
_LIKELIHOODS = frozenset({"beta_binomial", "bernoulli_offfloor"})
_OUTCOMES = frozenset({"TE2", "TR2", "UE2", "UR2"})


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} must contain non-empty strings, got {item!r}")
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate symbols: {out!r}")
    return out


@dataclass(frozen=True, slots=True)
class BlockExposureModelSettings:
    """Immutable declaration for one staggered block-exposure model."""

    ability_covariate: str | None = None
    adjust_for: tuple[str, ...] = ()
    use_child_re: bool = True
    likelihood: Literal["beta_binomial", "bernoulli_offfloor"] = "beta_binomial"
    drop_ceiling_violations: tuple[str, ...] = ()
    # ``None`` retains the factory's outcome-tier default without duplicating it.
    delta_prior_sigma: float | None = None

    def __post_init__(self) -> None:
        require_declared_booleans(self)
        if self.ability_covariate is not None and (
            not isinstance(self.ability_covariate, str) or not self.ability_covariate
        ):
            raise TypeError("ability_covariate must be a non-empty string or None")
        object.__setattr__(
            self,
            "adjust_for",
            _tuple_of_strings(self.adjust_for, name="adjust_for"),
        )
        object.__setattr__(
            self,
            "drop_ceiling_violations",
            _tuple_of_strings(
                self.drop_ceiling_violations,
                name="drop_ceiling_violations",
            ),
        )
        if self.likelihood not in _LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(_LIKELIHOODS)}, "
                f"got {self.likelihood!r}"
            )
        sigma = self.delta_prior_sigma
        if sigma is not None:
            if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
                raise TypeError("delta_prior_sigma must be a number or None")
            if not math.isfinite(sigma) or sigma <= 0:
                raise ValueError("delta_prior_sigma must be positive and finite")
            object.__setattr__(self, "delta_prior_sigma", float(sigma))

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> BlockExposureModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown block-exposure setting(s): "
                f"{', '.join(unknown)}. Declare BlockExposureModelSettings so "
                "misspellings fail fast."
            )
        return cls(
            ability_covariate=extra.get("ability_covariate"),
            adjust_for=extra.get("adjust_for", ()),
            use_child_re=extra.get("use_child_re", True),
            likelihood=extra.get("likelihood", "beta_binomial"),
            drop_ceiling_violations=extra.get("drop_ceiling_violations", ()),
            delta_prior_sigma=extra.get("delta_prior_sigma"),
        )


@dataclass(frozen=True, slots=True)
class BlockExposureRunPlan:
    """Concrete, validated instructions for a complete block-exposure fit."""

    model_id: str
    settings_source: str
    study_id: str
    outcome_symbol: str
    ability_covariate: str | None
    adjust_for: tuple[str, ...]
    use_child_re: bool
    likelihood: Literal["beta_binomial", "bernoulli_offfloor"]
    off_floor: bool
    drop_ceiling_violations: tuple[str, ...]
    delta_prior_sigma: float | None
    phase_mode: str
    baseline_covariates: tuple[str, ...]
    pre_covariates: tuple[str, ...]
    post_covariates: tuple[str, ...]
    observation_node: str
    compute_loo: bool
    loo_unit: str
    focal_term: str
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan."""
        return {
            "phase_mode": self.phase_mode,
            "outcomes": (self.outcome_symbol,),
            "baseline_covariates": self.baseline_covariates,
            "covariates": self.pre_covariates,
            "post_covariates": self.post_covariates,
            "drop_ceiling_violations": self.drop_ceiling_violations,
        }

    def factory_kwargs(
        self, *, effective_adjustment: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """Arguments for ``build_block_exposure_model``."""
        return {
            "outcome_symbol": self.outcome_symbol,
            "ability_covariate": self.ability_covariate,
            "adjust_for": self.adjust_for
            if effective_adjustment is None
            else effective_adjustment,
            "use_child_re": self.use_child_re,
            "likelihood": self.likelihood,
            "delta_prior_sigma": self.delta_prior_sigma,
        }

    def coefficient_names(
        self, *, effective_adjustment: tuple[str, ...] | None = None
    ) -> list[str]:
        """Reported structural coefficients in stable display order."""
        adjustment = (
            self.adjust_for
            if effective_adjustment is None
            else effective_adjustment
        )
        names = ["delta", "gamma_A"]
        if self.ability_covariate:
            names.append("gamma_ability")
        names.extend(f"gamma_{name}" for name in adjustment)
        return names

    def diagnostic_vars(
        self, *, effective_adjustment: tuple[str, ...] | None = None
    ) -> list[str]:
        """Variables scanned by summaries and the convergence gate."""
        tail: list[str] = [] if self.off_floor else ["kappa"]
        if self.use_child_re:
            tail.append("sigma_child")
        return [
            "alpha",
            "alpha_time",
            *self.coefficient_names(effective_adjustment=effective_adjustment),
            *tail,
        ]

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language recipe generated from the validated plan."""
        adjusters = ", ".join(self.adjust_for) if self.adjust_for else "none"
        drops = (
            ", ".join(self.drop_ceiling_violations)
            if self.drop_ceiling_violations
            else "none"
        )
        delta_sigma = (
            "outcome-tier factory default"
            if self.delta_prior_sigma is None
            else f"{self.delta_prior_sigma:g}"
        )
        return (
            "Note: Generated from the validated block-exposure run plan; template "
            "drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Ability covariate: "
            f"{self.ability_covariate or 'none'}. Requested adjustment terms: "
            f"{adjusters}. Child random intercept: {self.use_child_re}. Likelihood: "
            f"`{self.likelihood}`. Ceiling-violation exceptions: {drops}. Focal "
            f"delta prior sigma: {delta_sigma}.\n\n"
            "## Uncertainty and checks\n\n"
            f"The observation node is `{self.observation_node}` and PSIS-LOO uses "
            f"the `{self.loo_unit}` unit. Interpret the posterior only after the "
            "zero-divergence convergence gate, posterior-predictive checks and "
            "power-scaling sensitivity diagnostics pass. The saved `config.json` "
            "contains the same resolved run plan in machine-readable form.\n"
        )


def declared_block_exposure_settings(
    spec: ModelSpec,
) -> tuple[BlockExposureModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: block-exposure settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, BlockExposureModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='block_exposure' requires "
                f"BlockExposureModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        BlockExposureModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
        ),
        "legacy_extra",
    )


def resolve_block_exposure_run_plan(spec: ModelSpec) -> BlockExposureRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "block_exposure":
        raise ValueError(
            f"{spec.model_id}: expected kind 'block_exposure', got {spec.kind!r}"
        )
    if spec.study_id != "rli":
        raise ValueError(
            f"{spec.model_id}: block_exposure requires study_id='rli', got "
            f"{spec.study_id!r}"
        )
    if spec.outcome_symbol not in _OUTCOMES:
        raise ValueError(
            f"{spec.model_id}: block_exposure outcome_symbol must be one of "
            f"{sorted(_OUTCOMES)!r}, got {spec.outcome_symbol!r}"
        )

    settings, source = declared_block_exposure_settings(spec)
    own = spec.outcome_symbol
    if settings.ability_covariate in settings.adjust_for:
        raise ValueError(
            f"{spec.model_id}: ability_covariate {settings.ability_covariate!r} "
            "must not also appear in adjust_for"
        )
    unexpected_drops = sorted(set(settings.drop_ceiling_violations) - {own})
    if unexpected_drops:
        raise ValueError(
            f"{spec.model_id}: drop_ceiling_violations may only name the fitted "
            f"outcome {own!r}, got {unexpected_drops!r}"
        )

    pre_adj, post_adj = split_covariates_by_wave(settings.adjust_for)
    baseline_adj, post_adj = split_confounders_by_timing(post_adj)
    baseline_covariates = (
        (settings.ability_covariate,) if settings.ability_covariate else ()
    ) + baseline_adj
    off_floor = settings.likelihood == "bernoulli_offfloor"

    return BlockExposureRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        outcome_symbol=own,
        ability_covariate=settings.ability_covariate,
        adjust_for=settings.adjust_for,
        use_child_re=settings.use_child_re,
        likelihood=settings.likelihood,
        off_floor=off_floor,
        drop_ceiling_violations=settings.drop_ceiling_violations,
        delta_prior_sigma=settings.delta_prior_sigma,
        phase_mode="levels",
        baseline_covariates=baseline_covariates,
        pre_covariates=pre_adj,
        post_covariates=post_adj,
        observation_node="y_offfloor" if off_floor else "y_post",
        compute_loo=True,
        loo_unit="observation_row",
        focal_term="delta",
        design=(
            "Staggered block-2 exposure model over the RLI per-wave levels panel. "
            "Immediate-arm children are block-2 active at wave 3; wait-list children "
            "remain block-1 active until both arms are block-2 active at wave 4. "
            "Wave intercepts absorb the shared secular trend and an optional child "
            "random intercept partially pools stable between-child differences."
        ),
        estimand=(
            "The pooled block-active coefficient delta: the adjusted change in the "
            "outcome's logit mean when block 2 is active rather than block 1, also "
            "translated to the outcome-item scale. Identification requires parallel "
            "untreated trajectories across arms."
        ),
        causal_status=(
            "Adjusted association, not a randomised treatment effect. The identifying "
            "wave-3 contrast compares block-2-active with block-1-active children; "
            "age at block 2 and arm-specific trajectory differences can still confound "
            "delta."
        ),
        analysis_population=(
            f"Available RLI person-wave observations with an observed {own} block-2 "
            "score and all retained requested covariates. Wave 1 self-excludes because "
            "block-2 outcomes have no baseline score."
        ),
        missing_data_assumption=(
            "Available-case analysis under ignorable missingness conditional on the "
            "modelled covariates. Constant missingness indicators may be removed by "
            "preprocessing and are excluded from the active adjustment set. Declared "
            "ceiling violations are set missing before fitting rather than repaired."
        ),
    )
