# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and run plan for joint historical-cohort growth models.

The registered ``kind="historical_joint"`` model jointly fits several bounded
measures from the Byrne reading-language-memory cohort and reports the
between-child correlation of their stable levels.  This module replaces the
family's free-form ``ModelSpec.extra`` boundary with immutable settings and a
validated plan resolved before an output transaction is opened or study data are
loaded (#394 pillar 4).

The migration is behaviour-preserving: the selected rows, likelihoods, priors,
fitted equation, diagnostic variables and output tables do not change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.datasets import resolve_dataset

__all__ = [
    "HistoricalJointModelSettings",
    "HistoricalJointRunPlan",
    "declared_historical_joint_settings",
    "resolve_historical_joint_run_plan",
]


_DEFAULT_MEASURES = ("basread", "bpvs", "basdig")
_DEFAULT_WAVES = (1, 2, 3)
_LEGACY_KEYS = frozenset(
    {
        "study_id",
        "measures",
        "waves",
        "extension_waves",
        "eta_prior_sigma",
        "sigma_subject_prior_sigma",
        "kappa_prior_sigma",
        "lkj_eta",
        # Global sampler setting resolved by ``make_context``, not this family.
        "target_accept",
    }
)


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
class HistoricalJointModelSettings:
    """Immutable declaration for one joint historical-cohort growth model."""

    measures: tuple[str, ...] = _DEFAULT_MEASURES
    waves: tuple[int, ...] = _DEFAULT_WAVES
    extension_waves: tuple[int, ...] = ()
    eta_prior_sigma: float = 1.5
    sigma_subject_prior_sigma: float = 1.0
    kappa_prior_sigma: float = 50.0
    lkj_eta: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "measures", _tuple_of_strings(self.measures, name="measures"))
        if len(self.measures) < 2:
            raise ValueError("historical_joint measures must contain at least two measures")
        object.__setattr__(self, "waves", _wave_tuple(self.waves, name="waves"))
        if len(self.waves) < 2:
            raise ValueError("historical_joint waves must contain at least two waves")
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
            "lkj_eta",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name=name))

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
        spec_study_id: str,
    ) -> HistoricalJointModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown historical_joint setting(s): "
                f"{', '.join(unknown)}. Declare HistoricalJointModelSettings so "
                "misspellings fail fast."
            )
        legacy_study_id = extra.get("study_id", spec_study_id)
        if not isinstance(legacy_study_id, str) or not legacy_study_id:
            raise TypeError("study_id must be a non-empty string")
        if legacy_study_id != spec_study_id:
            raise ValueError(
                f"{model_id}: extra study_id={legacy_study_id!r} contradicts ModelSpec.study_id={spec_study_id!r}"
            )
        return cls(
            measures=extra.get("measures", _DEFAULT_MEASURES),
            waves=extra.get("waves", _DEFAULT_WAVES),
            extension_waves=extra.get("extension_waves", ()),
            eta_prior_sigma=extra.get("eta_prior_sigma", 1.5),
            sigma_subject_prior_sigma=extra.get("sigma_subject_prior_sigma", 1.0),
            kappa_prior_sigma=extra.get("kappa_prior_sigma", 50.0),
            lkj_eta=extra.get("lkj_eta", 2.0),
        )


@dataclass(frozen=True, slots=True)
class HistoricalJointRunPlan:
    """Concrete, validated instructions consumed by the complete family fit."""

    model_id: str
    settings_source: str
    study_id: str
    measures: tuple[str, ...]
    waves: tuple[int, ...]
    extension_waves: tuple[int, ...]
    complete_case: bool
    likelihood: str
    observation_nodes: tuple[str, ...]
    eta_prior_sigma: float
    sigma_subject_prior_sigma: float
    kappa_prior_sigma: float
    lkj_eta: float
    compute_loo: bool
    loo_unit: str
    loo_reason: str
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
        """Arguments for ``build_rlm_joint_growth_model``."""
        return {
            "measures": self.measures,
            "eta_prior_sigma": self.eta_prior_sigma,
            "sigma_subject_prior_sigma": self.sigma_subject_prior_sigma,
            "kappa_prior_sigma": self.kappa_prior_sigma,
            "lkj_eta": self.lkj_eta,
        }

    def diagnostic_vars(self) -> list[str]:
        """Curated summary and power-sensitivity parameters."""
        return ["eta_cell", "sigma_subject", "kappa", "measure_corr_pairs"]

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language account generated from the validated run plan."""
        measures = ", ".join(self.measures)
        core_waves = ", ".join(str(wave) for wave in self.waves)
        extension = ", ".join(str(wave) for wave in self.extension_waves) if self.extension_waves else "none"
        return (
            "Note: Generated from the validated historical-joint run plan; "
            "template drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Measures: {measures}. Complete-case core waves: {core_waves}. "
            f"Available-case extension waves: {extension}. Likelihood: "
            f"{self.likelihood}, with one observation node per measure. The model "
            "uses measure-specific group-by-wave means, group-specific child-level "
            "scales and overdispersion, and one between-measure correlation matrix "
            "shared across groups.\n\n"
            "## Uncertainty and checks\n\n"
            "Interpret the posterior only after the convergence gate, "
            "posterior-predictive checks and prior-sensitivity diagnostics pass. "
            f"PSIS-LOO is not computed: {self.loo_reason}\n"
        )


def declared_historical_joint_settings(
    spec: ModelSpec,
) -> tuple[HistoricalJointModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: historical_joint settings cannot be split between model_settings and extra"
            )
        if not isinstance(settings, HistoricalJointModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='historical_joint' requires "
                "HistoricalJointModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return (
        HistoricalJointModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
            spec_study_id=spec.study_id,
        ),
        "legacy_extra",
    )


def resolve_historical_joint_run_plan(spec: ModelSpec) -> HistoricalJointRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "historical_joint":
        raise ValueError(f"{spec.model_id}: expected kind 'historical_joint', got {spec.kind!r}")
    if not isinstance(spec.study_id, str) or not spec.study_id:
        raise TypeError(f"{spec.model_id}: study_id must be a non-empty string")

    settings, source = declared_historical_joint_settings(spec)
    _dataset, catalogue = resolve_dataset(spec.study_id)
    unknown = sorted(set(settings.measures) - set(catalogue))
    if unknown:
        raise ValueError(f"{spec.model_id}: unregistered {spec.study_id!r} measure symbol(s): {', '.join(unknown)}")

    loo_reason = (
        "the model has one likelihood node per measure, so no single pooled pointwise predictive unit is defined"
    )
    design = (
        "Joint descriptive Beta-Binomial growth model for a historical cohort. "
        "Each measure has its own group-by-wave mean, group-specific child-level "
        "scale and group-specific overdispersion; stable child deviations are "
        "correlated across measures through a shared LKJ correlation matrix."
    )
    estimand = (
        "The headline is the between-child correlation matrix of stable measure "
        "levels. Per-measure group-by-wave levels and growth contrasts are "
        "secondary descriptive summaries. The correlation is shared across "
        "cohort groups and does not estimate within-child coupling or direction."
    )
    causal_status = (
        "Descriptive only: cohort group is observational, no coefficient is a "
        "treatment effect, and the cross-measure correlations must not be read "
        "causally."
    )
    analysis_population = (
        "Children observed on every selected measure at every complete-case core "
        "wave. Retained children contribute extension-wave rows only when every "
        "selected measure is observed at that extension wave."
    )
    missing_data_assumption = (
        "Complete-case selection is applied jointly across measures and core waves; "
        "extension waves are available-case among that retained cohort. The "
        "descriptive summaries therefore apply to the selected observed cohort, "
        "not automatically to all recruited children."
    )

    return HistoricalJointRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        measures=settings.measures,
        waves=settings.waves,
        extension_waves=settings.extension_waves,
        complete_case=True,
        likelihood="beta_binomial",
        observation_nodes=tuple(f"score_{measure}" for measure in settings.measures),
        eta_prior_sigma=settings.eta_prior_sigma,
        sigma_subject_prior_sigma=settings.sigma_subject_prior_sigma,
        kappa_prior_sigma=settings.kappa_prior_sigma,
        lkj_eta=settings.lkj_eta,
        compute_loo=False,
        loo_unit="not_defined_multiple_likelihood_nodes",
        loo_reason=loo_reason,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
