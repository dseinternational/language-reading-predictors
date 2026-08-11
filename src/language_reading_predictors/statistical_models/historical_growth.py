# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and run plan for historical-cohort growth models.

The registered ``kind="historical_growth"`` models fit one bounded Byrne
reading-language-memory measure at a time over a complete-case core window and
an optional available-case extension.  This module replaces the family's
free-form ``ModelSpec.extra`` boundary with immutable settings and a validated
plan resolved before an output transaction is opened or study data are loaded
(#394 pillar 4).

The migration is behaviour-preserving: selected rows, the Beta-Binomial
likelihood, priors, fitted equation, diagnostic variables, PSIS-LOO policy and
published tables remain unchanged for all nine registered models.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.datasets import resolve_dataset

__all__ = [
    "HistoricalGrowthModelSettings",
    "HistoricalGrowthRunPlan",
    "declared_historical_growth_settings",
    "resolve_historical_growth_run_plan",
]


_DEFAULT_MEASURE = "basread"
_DEFAULT_WAVES = (1, 2, 3)
_LEGACY_KEYS = frozenset(
    {
        "study_id",
        "measure",
        "waves",
        "extension_waves",
        "eta_prior_sigma",
        "sigma_subject_prior_sigma",
        "kappa_prior_sigma",
        # Global sampler setting resolved by ``make_context``, not this family.
        "target_accept",
    }
)
_GROWTH_DETERMINISTICS = (
    "growth_first_next_items",
    "growth_next_last_items",
    "growth_first_last_items",
)


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _wave_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of positive integers, got {value!r}")
    out = tuple(value)
    if any(isinstance(wave, bool) or not isinstance(wave, int) for wave in out):
        raise TypeError(f"{name} must contain positive integers, got {out!r}")
    if any(wave <= 0 for wave in out):
        raise ValueError(f"{name} must contain positive integers, got {out!r}")
    if tuple(sorted(set(out))) != out:
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    return out


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return out


@dataclass(frozen=True, slots=True)
class HistoricalGrowthModelSettings:
    """Immutable declaration for one historical-cohort growth model."""

    measure: str = _DEFAULT_MEASURE
    waves: tuple[int, ...] = _DEFAULT_WAVES
    extension_waves: tuple[int, ...] = ()
    eta_prior_sigma: float = 1.5
    sigma_subject_prior_sigma: float = 1.0
    kappa_prior_sigma: float = 50.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "measure", _non_empty_string(self.measure, name="measure"))
        object.__setattr__(self, "waves", _wave_tuple(self.waves, name="waves"))
        if len(self.waves) < 2:
            raise ValueError("historical_growth waves must contain at least two waves")
        object.__setattr__(
            self,
            "extension_waves",
            _wave_tuple(self.extension_waves, name="extension_waves"),
        )
        overlap = sorted(set(self.waves) & set(self.extension_waves))
        if overlap:
            raise ValueError(f"extension_waves overlap the complete-case core waves: {overlap}")
        for name in (
            "eta_prior_sigma",
            "sigma_subject_prior_sigma",
            "kappa_prior_sigma",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name=name))

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
        spec_study_id: str,
        outcome_symbol: str | None,
    ) -> HistoricalGrowthModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown historical_growth setting(s): "
                f"{', '.join(unknown)}. Declare HistoricalGrowthModelSettings so "
                "misspellings fail fast."
            )
        legacy_study_id = extra.get("study_id", spec_study_id)
        if not isinstance(legacy_study_id, str) or not legacy_study_id:
            raise TypeError("study_id must be a non-empty string")
        if legacy_study_id != spec_study_id:
            raise ValueError(
                f"{model_id}: extra study_id={legacy_study_id!r} contradicts "
                f"ModelSpec.study_id={spec_study_id!r}"
            )
        return cls(
            measure=extra.get("measure", outcome_symbol or _DEFAULT_MEASURE),
            waves=extra.get("waves", _DEFAULT_WAVES),
            extension_waves=extra.get("extension_waves", ()),
            eta_prior_sigma=extra.get("eta_prior_sigma", 1.5),
            sigma_subject_prior_sigma=extra.get("sigma_subject_prior_sigma", 1.0),
            kappa_prior_sigma=extra.get("kappa_prior_sigma", 50.0),
        )


@dataclass(frozen=True, slots=True)
class HistoricalGrowthRunPlan:
    """Concrete, validated instructions consumed by the complete family fit."""

    model_id: str
    settings_source: str
    study_id: str
    measure: str
    waves: tuple[int, ...]
    extension_waves: tuple[int, ...]
    complete_case: bool
    likelihood: str
    observation_node: str
    eta_prior_sigma: float
    sigma_subject_prior_sigma: float
    kappa_prior_sigma: float
    compute_loo: bool
    loo_unit: str
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``load_longitudinal_panel``."""
        return {
            "waves": self.waves,
            "complete_case": self.complete_case,
            "extension_waves": self.extension_waves,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_historical_growth_model``."""
        return {
            "measure": self.measure,
            "eta_prior_sigma": self.eta_prior_sigma,
            "sigma_subject_prior_sigma": self.sigma_subject_prior_sigma,
            "kappa_prior_sigma": self.kappa_prior_sigma,
        }

    def diagnostic_vars(self, available_vars: Collection[str]) -> list[str]:
        """Curated diagnostics, preserving the factory's conditional deterministics."""
        return [
            "eta_cell",
            "sigma_subject",
            "kappa",
            *(name for name in _GROWTH_DETERMINISTICS if name in available_vars),
        ]

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language account generated from the validated run plan."""
        core_waves = ", ".join(str(wave) for wave in self.waves)
        extension = ", ".join(str(wave) for wave in self.extension_waves) if self.extension_waves else "none"
        return (
            "Note: Generated from the validated historical-growth run plan; "
            "template drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Measure: `{self.measure}`. Complete-case core waves: {core_waves}. "
            f"Available-case extension waves: {extension}. Likelihood: "
            "Beta-Binomial bounded counts with group-by-wave means, "
            "group-specific child-level scales and group-specific "
            "overdispersion.\n\n"
            "## Uncertainty and checks\n\n"
            "Interpret the posterior only after the convergence gate, PSIS-LOO, "
            "posterior-predictive checks and prior-sensitivity diagnostics pass. "
            "The saved `config.json` contains the same resolved run plan in "
            "machine-readable form.\n"
        )


def declared_historical_growth_settings(
    spec: ModelSpec,
) -> tuple[HistoricalGrowthModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: historical_growth settings cannot be split "
                "between model_settings and extra"
            )
        if not isinstance(settings, HistoricalGrowthModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='historical_growth' requires "
                "HistoricalGrowthModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return (
        HistoricalGrowthModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
            spec_study_id=spec.study_id,
            outcome_symbol=spec.outcome_symbol,
        ),
        "legacy_extra",
    )


def resolve_historical_growth_run_plan(spec: ModelSpec) -> HistoricalGrowthRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "historical_growth":
        raise ValueError(f"{spec.model_id}: expected kind 'historical_growth', got {spec.kind!r}")
    if not isinstance(spec.study_id, str) or not spec.study_id:
        raise TypeError(f"{spec.model_id}: study_id must be a non-empty string")

    settings, source = declared_historical_growth_settings(spec)
    _dataset, catalogue = resolve_dataset(spec.study_id)
    if settings.measure not in catalogue:
        raise ValueError(
            f"{spec.model_id}: unregistered {spec.study_id!r} measure symbol: {settings.measure}"
        )
    if spec.outcome_symbol is not None and spec.outcome_symbol != settings.measure:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol={spec.outcome_symbol!r} contradicts "
            f"historical_growth measure={settings.measure!r}"
        )

    return HistoricalGrowthRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        measure=settings.measure,
        waves=settings.waves,
        extension_waves=settings.extension_waves,
        complete_case=True,
        likelihood="beta_binomial",
        observation_node="score",
        eta_prior_sigma=settings.eta_prior_sigma,
        sigma_subject_prior_sigma=settings.sigma_subject_prior_sigma,
        kappa_prior_sigma=settings.kappa_prior_sigma,
        compute_loo=True,
        loo_unit="observation_row",
        design=(
            "Descriptive Beta-Binomial group-by-wave growth model for one bounded "
            "measure in a historical cohort. Supported group-wave cells have separate "
            "means; child-level heterogeneity and overdispersion are group-specific."
        ),
        estimand=(
            "The headline quantities are within-group changes in expected item score "
            "over supported wave intervals. Group-by-wave expected levels and "
            "between-group contrasts over the common observation window are secondary "
            "descriptive summaries."
        ),
        causal_status=(
            "Descriptive only: cohort group is observational, no coefficient is a "
            "treatment effect, and between-group differences must not be read causally."
        ),
        analysis_population=(
            "Children observed on the selected measure at every complete-case core "
            "wave. Retained children contribute extension-wave rows when the measure "
            "is observed there."
        ),
        missing_data_assumption=(
            "Complete-case selection defines the core cohort; extension waves are "
            "available-case among that retained cohort. Later-wave summaries therefore "
            "describe an attrition-selected observed tail, not automatically all "
            "recruited children."
        ),
    )
