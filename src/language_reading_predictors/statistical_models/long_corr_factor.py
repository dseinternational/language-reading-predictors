# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved plan for longitudinal factor models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.settings_validation import (
    require_declared_booleans,
)

__all__ = [
    "LongCorrFactorModelSettings",
    "LongCorrFactorRunPlan",
    "declared_long_corr_factor_settings",
    "resolve_long_corr_factor_run_plan",
]


DomainItems = tuple[tuple[str, tuple[str, ...]], ...]

_DEFAULT_DOMAINS: DomainItems = (
    ("vocabulary", ("R", "E", "TR", "TE")),
    ("code", ("L", "B")),
    ("grammar", ("F", "T")),
)
_FAMILY_KEYS = frozenset(
    {
        "domains",
        "loading_prior",
        "comm_alpha",
        "comm_beta",
        "loading_sigma",
        "residual_sigma",
        "lkj_eta",
        "factor_mean_sigma",
        "trait_share_a",
        "trait_share_b",
    }
)
_GLOBAL_KEYS = frozenset({"target_accept"})
_LEGACY_KEYS = _FAMILY_KEYS | _GLOBAL_KEYS


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _optional_positive_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name=name)


def _domains(value: Any) -> DomainItems:
    if isinstance(value, Mapping):
        raw = tuple(value.items())
    elif isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("domains must be a mapping or sequence of (name, indicators)")
    else:
        raw = tuple(value)
    out: list[tuple[str, tuple[str, ...]]] = []
    for entry in raw:
        if not isinstance(entry, Sequence) or isinstance(entry, str) or len(entry) != 2:
            raise TypeError("each domain must be a (name, indicators) pair")
        name, indicators = entry
        if not isinstance(name, str) or not name:
            raise TypeError(f"domain names must be non-empty strings, got {name!r}")
        if isinstance(indicators, str) or not isinstance(indicators, Sequence):
            raise TypeError(f"domain {name!r} indicators must be a sequence of strings")
        symbols = tuple(indicators)
        if not symbols:
            raise ValueError(f"domain {name!r} must contain at least one indicator")
        for symbol in symbols:
            if not isinstance(symbol, str) or not symbol:
                raise TypeError(
                    f"domain {name!r} indicators must be non-empty strings"
                )
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"domain {name!r} contains duplicate indicators")
        out.append((name, symbols))
    if len(out) < 2:
        raise ValueError("domains must contain at least two latent domains")
    names = [name for name, _ in out]
    if len(names) != len(set(names)):
        raise ValueError("domains contains duplicate domain names")
    all_symbols = [symbol for _, symbols in out for symbol in symbols]
    if len(all_symbols) != len(set(all_symbols)):
        raise ValueError("an indicator may belong to only one domain")
    return tuple(out)


@dataclass(frozen=True, slots=True)
class LongCorrFactorModelSettings:
    """Immutable declaration for the longitudinal correlated-factor family."""

    domains: DomainItems = _DEFAULT_DOMAINS
    loading_prior: Literal["communality", "free"] = "communality"
    comm_alpha: float | None = None
    comm_beta: float | None = None
    loading_sigma: float | None = None
    residual_sigma: float | None = None
    lkj_eta: float = 2.0
    factor_mean_sigma: float = 1.0
    trait_share_a: float = 1.5
    trait_share_b: float = 1.5

    def __post_init__(self) -> None:
        require_declared_booleans(self)
        object.__setattr__(self, "domains", _domains(self.domains))
        if self.loading_prior not in {"communality", "free"}:
            raise ValueError(
                "loading_prior must be 'communality' or 'free', got "
                f"{self.loading_prior!r}"
            )
        for name in ("comm_alpha", "comm_beta", "loading_sigma", "residual_sigma"):
            object.__setattr__(
                self,
                name,
                _optional_positive_float(getattr(self, name), name=name),
            )
        for name in (
            "lkj_eta",
            "factor_mean_sigma",
            "trait_share_a",
            "trait_share_b",
        ):
            object.__setattr__(
                self,
                name,
                _positive_float(getattr(self, name), name=name),
            )

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
    ) -> LongCorrFactorModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown longitudinal-factor setting(s): "
                f"{', '.join(unknown)}. Declare LongCorrFactorModelSettings so "
                "misspellings fail fast."
            )
        return cls(
            domains=_domains(extra.get("domains") or _DEFAULT_DOMAINS),
            loading_prior=extra.get("loading_prior", "communality"),
            comm_alpha=extra.get("comm_alpha"),
            comm_beta=extra.get("comm_beta"),
            loading_sigma=extra.get("loading_sigma"),
            residual_sigma=extra.get("residual_sigma"),
            lkj_eta=extra.get("lkj_eta", 2.0),
            factor_mean_sigma=extra.get("factor_mean_sigma", 1.0),
            trait_share_a=extra.get("trait_share_a", 1.5),
            trait_share_b=extra.get("trait_share_b", 1.5),
        )


@dataclass(frozen=True, slots=True)
class LongCorrFactorRunPlan:
    """Concrete, validated instructions for the longitudinal factor fit."""

    model_id: str
    settings_source: str
    study_id: str
    domains: DomainItems
    indicators: tuple[str, ...]
    loading_prior: Literal["communality", "free"]
    comm_alpha: float
    comm_beta: float
    loading_sigma: float
    residual_sigma: float
    lkj_eta: float
    factor_mean_sigma: float
    trait_share_a: float
    trait_share_b: float
    observation_node: str
    compute_loo: bool
    custom_loo: bool
    loo_unit: str
    focal_term: str | None
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def domain_mapping(self) -> dict[str, tuple[str, ...]]:
        """Return the factory's ordered domain mapping."""
        return dict(self.domains)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_wave_panel``."""
        return {"outcomes": self.indicators}

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_longitudinal_corr_factor_model``."""
        return {
            "domains": self.domain_mapping(),
            "loading_prior": self.loading_prior,
            "comm_alpha": self.comm_alpha,
            "comm_beta": self.comm_beta,
            "loading_sigma": self.loading_sigma,
            "residual_sigma": self.residual_sigma,
            "lkj_eta": self.lkj_eta,
            "factor_mean_sigma": self.factor_mean_sigma,
            "trait_share_a": self.trait_share_a,
            "trait_share_b": self.trait_share_b,
        }

    def diagnostic_vars(self) -> list[str]:
        """Released quantities scanned by summaries and the convergence gate."""
        return [
            "lambda_load",
            "sigma_indicator",
            "communality",
            "within_share",
            "trait_share",
            "factor_corr_pairs",
        ]

    def prior_vars(
        self,
        *,
        free_rv_names: Sequence[str],
        observation_nodes: Sequence[str],
    ) -> list[str]:
        """Prior-predictive variables, deduplicated in fitted graph order."""
        return list(
            dict.fromkeys(
                [
                    *free_rv_names,
                    "communality",
                    "lambda_load",
                    "sigma_indicator",
                    "within_share",
                    "factor_corr_pairs",
                    *observation_nodes,
                ]
            )
        )

    def psense_vars(self) -> list[str]:
        """Reported parameters requiring power-scaling review."""
        return ["factor_corr_pairs", "trait_share", "communality"]

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language recipe generated from the validated plan."""
        domains = "; ".join(
            f"{name}: {', '.join(symbols)}" for name, symbols in self.domains
        )
        return (
            "Note: Generated from the validated longitudinal-factor run plan; "
            "template drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Domains: {domains}. Loading parameterisation: "
            f"`{self.loading_prior}`. LKJ eta: {self.lkj_eta:g}. Trait-share "
            f"Beta shape: ({self.trait_share_a:g}, {self.trait_share_b:g}).\n\n"
            "## Uncertainty and checks\n\n"
            "The primary sampler omits automatic observation-level LOO because the "
            "masked likelihood is split by missingness pattern; exact pattern "
            f"likelihoods are stitched to the `{self.loo_unit}` unit before PSIS-LOO. "
            "Interpret the released correlations only after the zero-divergence "
            "convergence gate, posterior-predictive checks, indicator prior checks "
            "and power-scaling sensitivity diagnostics pass. The saved "
            "`config.json` contains the same resolved run plan.\n"
        )


def declared_long_corr_factor_settings(
    spec: ModelSpec,
) -> tuple[LongCorrFactorModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: longitudinal-factor settings cannot be split "
                f"between model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, LongCorrFactorModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='long_corr_factor' requires "
                f"LongCorrFactorModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        LongCorrFactorModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
        ),
        "legacy_extra",
    )


def resolve_long_corr_factor_run_plan(spec: ModelSpec) -> LongCorrFactorRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "long_corr_factor":
        raise ValueError(
            f"{spec.model_id}: expected kind 'long_corr_factor', got {spec.kind!r}"
        )
    if spec.study_id != "rli":
        raise ValueError(
            f"{spec.model_id}: long_corr_factor requires study_id='rli', got "
            f"{spec.study_id!r}"
        )
    if spec.outcome_symbol is not None:
        raise ValueError(
            f"{spec.model_id}: long_corr_factor is a measurement model and requires "
            "outcome_symbol=None"
        )

    settings, source = declared_long_corr_factor_settings(spec)
    free_knobs = sorted(
        name
        for name in ("loading_sigma", "residual_sigma")
        if getattr(settings, name) is not None
    )
    comm_knobs = sorted(
        name
        for name in ("comm_alpha", "comm_beta")
        if getattr(settings, name) is not None
    )
    if settings.loading_prior == "communality" and free_knobs:
        raise ValueError(
            f"{spec.model_id}: {free_knobs} only apply to loading_prior='free'"
        )
    if settings.loading_prior == "free" and comm_knobs:
        raise ValueError(
            f"{spec.model_id}: {comm_knobs} only apply to loading_prior='communality'"
        )
    indicators = tuple(symbol for _, symbols in settings.domains for symbol in symbols)

    return LongCorrFactorRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        domains=settings.domains,
        indicators=indicators,
        loading_prior=settings.loading_prior,
        comm_alpha=2.0 if settings.comm_alpha is None else settings.comm_alpha,
        comm_beta=2.0 if settings.comm_beta is None else settings.comm_beta,
        loading_sigma=(
            1.0 if settings.loading_sigma is None else settings.loading_sigma
        ),
        residual_sigma=(
            1.0 if settings.residual_sigma is None else settings.residual_sigma
        ),
        lkj_eta=settings.lkj_eta,
        factor_mean_sigma=settings.factor_mean_sigma,
        trait_share_a=settings.trait_share_a,
        trait_share_b=settings.trait_share_b,
        observation_node="missingness_pattern_z_nodes",
        compute_loo=False,
        custom_loo=True,
        loo_unit="child",
        focal_term=None,
        design=(
            "Four-wave correlated-domain measurement model with invariant loadings "
            "and residuals, marginalised factor scores, and a trait/state "
            "decomposition of across-wave dependence."
        ),
        estimand=(
            "Per-wave latent domain correlations, conditional latent slopes, "
            "indicator loadings and trait shares after accounting for indicator "
            "measurement error."
        ),
        causal_status=(
            "Descriptive measurement associations only; no latent correlation or "
            "conditional slope is a causal effect."
        ),
        analysis_population=(
            "All 54 RLI children across four waves; observed cells contribute under "
            "their realised indicator-missingness pattern."
        ),
        missing_data_assumption=(
            "The likelihood masks missing indicator cells rather than dropping a "
            "child-wave row. Inference assumes the observed-cell patterns are "
            "ignorable conditional on the longitudinal factor model."
        ),
    )
