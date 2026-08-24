# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and run plan for joint historical-cohort growth models.

The registered ``kind="historical_joint"`` model jointly fits several bounded
measures from the Byrne reading-language-memory cohort and reports the
between-child correlation of their stable levels.  Its within-child companion
also estimates the correlation of wave-specific departures from those stable
levels.  This module replaces the family's free-form ``ModelSpec.extra``
boundary with immutable settings and a validated plan resolved before an output
transaction is opened or study data are loaded (#394 pillar 4).

The original ``lrp-rlm-jc-001`` path remains behaviour-preserving: its selected
rows, likelihoods, priors, fitted equation, diagnostic variables and output
tables do not change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.datasets import resolve_dataset
from language_reading_predictors.statistical_models.historical_growth import (
    check_declared_waves,
    check_extension_after_core,
)

__all__ = [
    "HISTORICAL_JOINT_PRIOR_COMPANIONS",
    "HistoricalJointModelSettings",
    "HistoricalJointRunPlan",
    "declared_historical_joint_settings",
    "resolve_historical_joint_run_plan",
]


#: Within-child fit -> its registered wider-``sigma_within``-prior sensitivity
#: companion (#588 finding 5). The *authority* is the companion module, which is
#: built from its parent's frozen settings with ``dataclasses.replace`` so only the
#: within-scale prior can differ; this constant restates the pairing so the release
#: decision can reach it without importing every model module, exactly as
#: ``JOINT_DEPENDENCE_COMPANIONS`` and ``BLENDING_LINK_MODELS`` do for their
#: policies. It exists because the parent's classification of which measures clear
#: the resolvability threshold — and therefore which correlations are interpretable
#: at all — turns on that prior, and power scaling measures ``sigma_within`` as the
#: most prior-sensitive quantity in the fit. ``test_historical_joint_run_plan``
#: fails if the constant and the module declarations drift apart.
HISTORICAL_JOINT_PRIOR_COMPANIONS: dict[str, str] = {
    "lrp-rlm-jc-002": "lrp-rlm-jc-102",
}


_DEFAULT_MEASURES = ("basread", "bpvs", "basdig")
_DEFAULT_WAVES = (1, 2, 3)
_DEFAULT_DISPERSION_SIGMA = 0.25
_LEGACY_KEYS = frozenset(
    {
        "study_id",
        "measures",
        "waves",
        "extension_waves",
        "eta_prior_sigma",
        "sigma_subject_prior_sigma",
        "dispersion_prior_sigma",
        "lkj_eta",
        "within_correlation",
        "sigma_within_prior_sigma",
        "within_lkj_eta",
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
    dispersion_prior_sigma: float = _DEFAULT_DISPERSION_SIGMA
    """Scale of the HalfNormal on ``1/sqrt(kappa)``; see
    :func:`priors.inv_sqrt_kappa_prior`."""
    lkj_eta: float = 2.0
    within_correlation: bool = False
    sigma_within_prior_sigma: float = 0.5
    within_lkj_eta: float = 2.0

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
        check_extension_after_core(self.waves, self.extension_waves)
        for name in (
            "eta_prior_sigma",
            "sigma_subject_prior_sigma",
            "dispersion_prior_sigma",
            "lkj_eta",
            "sigma_within_prior_sigma",
            "within_lkj_eta",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name=name))
        if not isinstance(self.within_correlation, bool):
            raise TypeError(
                "within_correlation must be a boolean, got "
                f"{self.within_correlation!r}"
            )
        if self.within_correlation and self.extension_waves:
            raise ValueError(
                "within_correlation requires a balanced complete-case window; "
                "extension_waves must be empty"
            )
        if (
            self.within_correlation
            and self.dispersion_prior_sigma != _DEFAULT_DISPERSION_SIGMA
        ):
            # The within-child branch has a Binomial likelihood with no
            # Beta-Binomial concentration term, so this setting has no effect
            # there. Silently discarding an explicitly-declared value is exactly
            # the incoherent cross-field combination #455 asks resolution to
            # reject (2026-08-21 review, finding 10).
            raise ValueError(
                "dispersion_prior_sigma has no effect when within_correlation is "
                "true: that branch fits a Binomial likelihood with no "
                "Beta-Binomial concentration term. Leave it at its default."
            )

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
        spec_study_id: str,
    ) -> HistoricalJointModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        if "kappa_prior_sigma" in extra:
            raise ValueError(
                f"{model_id}: 'kappa_prior_sigma' was replaced by "
                "'dispersion_prior_sigma' when the overdispersion prior moved "
                "onto the dispersion scale (1/sqrt of the concentration). The "
                "reviewed default is 0.25; see priors.inv_sqrt_kappa_prior."
            )
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
            dispersion_prior_sigma=extra.get(
                "dispersion_prior_sigma", _DEFAULT_DISPERSION_SIGMA
            ),
            lkj_eta=extra.get("lkj_eta", 2.0),
            within_correlation=extra.get("within_correlation", False),
            sigma_within_prior_sigma=extra.get("sigma_within_prior_sigma", 0.5),
            within_lkj_eta=extra.get("within_lkj_eta", 2.0),
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
    dispersion_prior_sigma: float | None
    lkj_eta: float
    within_correlation: bool
    sigma_within_prior_sigma: float | None
    within_lkj_eta: float | None
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
        kwargs: dict[str, Any] = {
            "measures": self.measures,
            "eta_prior_sigma": self.eta_prior_sigma,
            "sigma_subject_prior_sigma": self.sigma_subject_prior_sigma,
            "lkj_eta": self.lkj_eta,
        }
        if self.within_correlation:
            kwargs.update(
                {
                    "within_correlation": True,
                    "sigma_within_prior_sigma": self.sigma_within_prior_sigma,
                    "within_lkj_eta": self.within_lkj_eta,
                }
            )
        else:
            kwargs["dispersion_prior_sigma"] = self.dispersion_prior_sigma
        return kwargs

    def diagnostic_vars(self) -> list[str]:
        """Curated summary and power-sensitivity parameters."""
        names = ["eta_cell", "sigma_subject"]
        if self.within_correlation:
            names.extend(["sigma_within", "within_corr_pairs"])
        else:
            names.append("kappa")
        names.append("measure_corr_pairs")
        return names

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language account generated from the validated run plan."""
        measures = ", ".join(self.measures)
        core_waves = ", ".join(str(wave) for wave in self.waves)
        extension = ", ".join(str(wave) for wave in self.extension_waves) if self.extension_waves else "none"
        likelihood_terms = (
            "Binomial counts with a logistic-normal wave-specific residual; the "
            "residual supplies the extra-Binomial variance, so no Beta-Binomial "
            "concentration term is fitted"
            if self.within_correlation
            else (
                "Beta-Binomial counts with group-specific overdispersion, whose "
                "prior is placed on the dispersion scale (1/sqrt of the "
                "concentration) so that a measure showing no extra-Binomial "
                "dispersion is permitted rather than excluded"
            )
        )
        correlation_terms = (
            "one between-measure correlation matrix for stable child levels and "
            "one correlation matrix for wave-specific within-child departures, "
            "both shared across groups"
            if self.within_correlation
            else "one between-measure correlation matrix shared across groups"
        )
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
            f"{self.likelihood}, with one observation node per measure: "
            f"{likelihood_terms}. The model uses measure-specific group-by-wave "
            "means, group-specific child-level scales and "
            f"{correlation_terms}.\n\n"
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
    check_declared_waves(
        catalogue,
        settings.measures,
        model_id=spec.model_id,
        waves=settings.waves,
        extension_waves=settings.extension_waves,
    )

    # 2026-08-23 joint audit, finding 8. The previous reason — "multiple likelihood
    # nodes, so no pointwise unit is defined" — was wrong. The nodes share an
    # observation coordinate, so their conditional log-likelihood contributions can
    # be summed per child-wave row; multiple nodes are not the obstacle. What is
    # missing is a *defined and implemented prediction target*: row-level validation
    # predicts another occasion for an already observed child (and, when
    # ``within_correlation`` is on, must handle that occasion's own latent
    # departure), while child-level validation predicts a new child and must
    # integrate the stable and within-child latent effects under an explicit
    # treatment of the sample-dependent centring constraints. An exploratory
    # post-hoc probe of older stored traces also gave maximum Pareto-k of about
    # 0.92/1.21 per child-wave and 1.64/1.63 per child, so naive conditional PSIS
    # was unreliable there; those traces predate the current priors, so those values
    # are not current diagnostics and nothing from them is published.
    loo_reason = (
        "no out-of-sample prediction target has been defined and implemented for "
        "this family. Multiple likelihood nodes are not the obstacle: they share an "
        "observation coordinate, so their contributions can be summed per "
        "child-wave row. Choosing between a new occasion for a known child and a "
        "new child changes what has to be integrated"
        + (
            " — including each held-out occasion's own latent departure and the "
            "sample-dependent double-centring constraint"
            if settings.within_correlation
            else ""
        )
        + ". Until a target-specific implementation exists (grouped child-level "
        "K-fold or exact refits with held-out child effects integrated from their "
        "population distribution), production LOO is not computed, and an "
        "exploratory PSIS probe of older stored traces was unreliable and is not "
        "reported"
    )
    if settings.within_correlation:
        design = (
            "Joint descriptive logistic-normal Binomial growth model for a balanced "
            "historical cohort panel. Each measure has its own group-by-wave mean "
            "and group-specific child-level scale. Stable child deviations and "
            "wave-specific within-child deviations have separate cross-measure LKJ "
            "correlation matrices shared across groups; the wave-specific residual "
            "supplies the extra-Binomial variance. Its scale is pooled across "
            "groups while the stable child-level scale is group-indexed - a "
            "parsimony assumption on the parameter that decides the resolvability "
            "classification, so a group whose wave-to-wave departures differ from "
            "the others cannot show it (2026-08-24 historical-joint review)."
        )
        estimand = (
            "The headline is the within-child correlation matrix of wave-specific "
            "departures from each child's stable level and the group-by-wave mean, "
            "on the latent logit scale. The matched between-child stable-level "
            "correlation matrix is reported alongside it. Both are symmetric "
            "descriptive associations shared across cohort groups; neither estimates "
            "direction. Two limits are structural rather than incidental. The "
            "wave-specific residual carries ALL extra-Binomial variance, because "
            "this branch fits a Binomial likelihood with no separate concentration "
            "term: true within-child fluctuation and measurement noise are one term "
            "and cannot be separated, so an independent measurement-error share "
            "attenuates the reported correlation toward zero by an amount the model "
            "cannot estimate. A small or unresolvable residual scale therefore "
            "cannot distinguish 'these skills do not co-fluctuate' from 'they do, "
            "but the measurements are too noisy for this design to see it'. And the "
            "double sum-to-zero centring makes the realised departures smaller than "
            "sigma_within (the fitted scale parameter); the correlation itself is "
            "unaffected, because the same projection is applied to every measure."
        )
    else:
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
    if settings.within_correlation:
        analysis_population = (
            "Children observed on every selected measure at every complete-case core "
            "wave. Extension waves are excluded so every retained child contributes "
            "the same number of observations to the within-child estimand."
        )
        missing_data_assumption = (
            "Complete-case selection is applied jointly across measures and core "
            "waves. The descriptive summaries therefore apply to the balanced "
            "selected cohort, not automatically to all recruited children."
        )
    else:
        analysis_population = (
            "Children observed on every selected measure at every complete-case core "
            "wave. Retained children contribute extension-wave rows only when every "
            "selected measure is observed at that extension wave."
        )
        missing_data_assumption = (
            "Complete-case selection is applied jointly across measures and core "
            "waves; extension waves are available-case among that retained cohort. "
            "The descriptive summaries therefore apply to the selected observed "
            "cohort, not automatically to all recruited children."
        )

    return HistoricalJointRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        measures=settings.measures,
        waves=settings.waves,
        extension_waves=settings.extension_waves,
        complete_case=True,
        likelihood=(
            "logistic_normal_binomial"
            if settings.within_correlation
            else "beta_binomial"
        ),
        observation_nodes=tuple(f"score_{measure}" for measure in settings.measures),
        eta_prior_sigma=settings.eta_prior_sigma,
        sigma_subject_prior_sigma=settings.sigma_subject_prior_sigma,
        # Every prior scale the fitted model does not contain is recorded as
        # null, in both directions. config.json is the estimand of record; it
        # must not name a prior the posterior lacks, and before the 2026-08-21
        # review (finding 10) it nulled the unused kappa but kept live
        # within-child scales for the between-child model, which has neither.
        dispersion_prior_sigma=(
            None
            if settings.within_correlation
            else settings.dispersion_prior_sigma
        ),
        lkj_eta=settings.lkj_eta,
        within_correlation=settings.within_correlation,
        sigma_within_prior_sigma=(
            settings.sigma_within_prior_sigma if settings.within_correlation else None
        ),
        within_lkj_eta=(
            settings.within_lkj_eta if settings.within_correlation else None
        ),
        compute_loo=False,
        loo_unit="undeclared_prediction_target_not_implemented",
        loo_reason=loo_reason,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
