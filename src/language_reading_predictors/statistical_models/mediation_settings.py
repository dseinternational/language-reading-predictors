# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and resolved run plans for the mediation families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES
from language_reading_predictors.statistical_models.preprocessing import (
    split_confounders_by_timing,
    split_covariates_by_wave,
)

__all__ = [
    "MediationModelSettings",
    "MediationMultiModelSettings",
    "MediationMultiRunPlan",
    "MediationRunPlan",
    "NamedConfounderCalibration",
    "declared_mediation_multi_settings",
    "declared_mediation_settings",
    "resolve_mediation_multi_run_plan",
    "resolve_mediation_run_plan",
]

_GLOBAL_KEYS = frozenset({"target_accept"})
_SINGLE_KEYS = frozenset(
    {
        "outcomes",
        "drop_missing_pre",
        "outcome_time",
        "mediator_kind",
        "route_symbols",
        "outcome_kind",
        "estimand",
        "companion_of",
        "period_stacked",
    }
)
_MULTI_KEYS = frozenset(
    {
        "mediators",
        "order",
        "chain",
        "second_mediator_offfloor",
        "outcomes",
        "named_confounder_calibration",
    }
)


def _symbols(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    if any(not isinstance(item, str) or not item for item in out):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate symbols: {out!r}")
    return out


def _optional_symbols(value: Any, *, name: str) -> tuple[str, ...] | None:
    return None if value is None else _symbols(value, name=name)


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class NamedConfounderCalibration:
    """Observed benchmark loaded for calibration but excluded from the fit."""

    symbol: str = "attend"
    label: str = "IS"

    def __post_init__(self) -> None:
        if not self.symbol or not isinstance(self.symbol, str):
            raise TypeError("named calibration symbol must be a non-empty string")
        if not self.label or not isinstance(self.label, str):
            raise TypeError("named calibration label must be a non-empty string")

    @classmethod
    def from_value(cls, value: Any) -> NamedConfounderCalibration | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("named_confounder_calibration must be a mapping")
        unknown = sorted(set(value) - {"symbol", "label"})
        if unknown:
            raise ValueError(
                "unknown named-confounder calibration setting(s): "
                f"{', '.join(unknown)}"
            )
        return cls(
            symbol=value.get("symbol", "attend"),
            label=value.get("label", "IS"),
        )


@dataclass(frozen=True, slots=True)
class MediationModelSettings:
    """Immutable declaration for one single-mediator fit."""

    outcomes: tuple[str, ...] | None = None
    drop_missing_pre: bool = True
    outcome_time: int | None = None
    mediator_kind: Literal["beta_binomial", "gaussian_composite"] = "beta_binomial"
    route_symbols: tuple[str, ...] = ()
    outcome_kind: Literal["beta_binomial", "bernoulli_offfloor"] = "beta_binomial"
    estimand: Literal["natural", "interventional"] = "natural"
    companion_of: str | None = None
    period_stacked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcomes",
            _optional_symbols(self.outcomes, name="outcomes"),
        )
        object.__setattr__(
            self,
            "route_symbols",
            _symbols(self.route_symbols, name="route_symbols"),
        )
        _bool(self.drop_missing_pre, name="drop_missing_pre")
        _bool(self.period_stacked, name="period_stacked")
        if self.outcome_time is not None:
            if isinstance(self.outcome_time, bool) or not isinstance(self.outcome_time, int):
                raise TypeError("outcome_time must be an integer or None")
            if self.outcome_time not in {3, 4}:
                raise ValueError("outcome_time must be 3 or 4")
        if self.mediator_kind not in {"beta_binomial", "gaussian_composite"}:
            raise ValueError(f"unsupported mediator_kind {self.mediator_kind!r}")
        if self.outcome_kind not in {"beta_binomial", "bernoulli_offfloor"}:
            raise ValueError(f"unsupported outcome_kind {self.outcome_kind!r}")
        if self.estimand not in {"natural", "interventional"}:
            raise ValueError(f"unsupported estimand {self.estimand!r}")
        if self.companion_of is not None and (
            not isinstance(self.companion_of, str) or not self.companion_of
        ):
            raise TypeError("companion_of must be a non-empty string or None")

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
    ) -> MediationModelSettings:
        unknown = sorted(set(extra) - _SINGLE_KEYS - _GLOBAL_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown mediation setting(s): {', '.join(unknown)}"
            )
        return cls(
            outcomes=extra.get("outcomes"),
            drop_missing_pre=extra.get("drop_missing_pre", True),
            outcome_time=extra.get("outcome_time"),
            mediator_kind=extra.get("mediator_kind", "beta_binomial"),
            route_symbols=extra.get("route_symbols", ()),
            outcome_kind=extra.get("outcome_kind", "beta_binomial"),
            estimand=extra.get("estimand", "natural"),
            companion_of=extra.get("companion_of"),
            period_stacked=extra.get("period_stacked", False),
        )


@dataclass(frozen=True, slots=True)
class MediationMultiModelSettings:
    """Immutable declaration for one two-mediator fit."""

    mediators: tuple[str, ...] = ("L", "E")
    order: tuple[str, ...] = ("L", "E")
    chain: bool = False
    second_mediator_offfloor: bool = False
    outcomes: tuple[str, ...] | None = None
    named_confounder_calibration: NamedConfounderCalibration | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mediators",
            _symbols(self.mediators, name="mediators"),
        )
        object.__setattr__(self, "order", _symbols(self.order, name="order"))
        object.__setattr__(
            self,
            "outcomes",
            _optional_symbols(self.outcomes, name="outcomes"),
        )
        object.__setattr__(
            self,
            "named_confounder_calibration",
            NamedConfounderCalibration.from_value(self.named_confounder_calibration),
        )
        _bool(self.chain, name="chain")
        _bool(self.second_mediator_offfloor, name="second_mediator_offfloor")

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
    ) -> MediationMultiModelSettings:
        unknown = sorted(set(extra) - _MULTI_KEYS - _GLOBAL_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown multi-mediation setting(s): {', '.join(unknown)}"
            )
        mediators = _symbols(extra.get("mediators", ("L", "E")), name="mediators")
        return cls(
            mediators=mediators,
            order=extra.get("order", mediators),
            chain=extra.get("chain", False),
            second_mediator_offfloor=extra.get("second_mediator_offfloor", False),
            outcomes=extra.get("outcomes"),
            named_confounder_calibration=extra.get("named_confounder_calibration"),
        )


def _raw_covariates(confounders: tuple[str, ...]) -> tuple[str, ...]:
    from language_reading_predictors.statistical_models.measures import MEASURES

    return tuple(symbol for symbol in confounders if symbol not in MEASURES)


@dataclass(frozen=True, slots=True)
class MediationRunPlan:
    """Concrete instructions for one single-mediator pipeline entry point."""

    model_id: str
    settings_source: str
    entrypoint: Literal["single", "period_stacked"]
    outcome_symbol: str
    mediator_symbol: str
    outcomes: tuple[str, ...] | None
    drop_missing_pre: bool
    outcome_time: int | None
    mediator_kind: Literal["beta_binomial", "gaussian_composite"]
    route_symbols: tuple[str, ...]
    outcome_kind: Literal["beta_binomial", "bernoulli_offfloor"]
    estimand: Literal["natural", "interventional"]
    companion_of: str | None
    declared_confounders: tuple[str, ...]
    effective_confounders: tuple[str, ...]
    raw_covariates: tuple[str, ...]
    observation_nodes: tuple[str, ...]
    compute_loo: bool
    design: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_effective_confounders(
        self, confounders: tuple[str, ...]
    ) -> MediationRunPlan:
        unknown = sorted(set(confounders) - set(self.declared_confounders))
        if unknown:
            raise ValueError(f"effective confounders were not declared: {unknown}")
        return replace(self, effective_confounders=tuple(confounders))

    def prepare_kwargs(self) -> dict[str, Any]:
        if self.entrypoint != "single":
            raise ValueError("prepare_kwargs applies only to single-mediator fits")
        if self.outcome_time is not None:
            return {
                "outcome_symbol": self.outcome_symbol,
                "outcome_time": self.outcome_time,
                "outcomes": self.outcomes or ITT_OUTCOMES,
                "covariates": self.raw_covariates,
            }
        kwargs: dict[str, Any] = {
            "phase_mode": "itt",
            "covariates": self.raw_covariates,
        }
        if self.outcomes is not None:
            kwargs.update(
                outcomes=self.outcomes,
                drop_missing_pre=self.drop_missing_pre,
            )
        return kwargs

    def period_prepare_kwargs(self) -> dict[str, Any]:
        if self.entrypoint != "period_stacked":
            raise ValueError("period_prepare_kwargs requires a period-stacked plan")
        pre_covariates, post_covariates = split_covariates_by_wave(self.raw_covariates)
        baseline_covariates, post_covariates = split_confounders_by_timing(
            post_covariates
        )
        measures = tuple(
            symbol
            for symbol in self.declared_confounders
            if symbol not in self.raw_covariates
        )
        return {
            "phase_mode": "all",
            "outcomes": (self.outcome_symbol, self.mediator_symbol, *measures),
            "covariates": pre_covariates,
            "post_covariates": post_covariates,
            "baseline_covariates": baseline_covariates,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        return {
            "mediator_symbol": self.mediator_symbol,
            "outcome_symbol": self.outcome_symbol,
            "confounder_symbols": self.effective_confounders,
            "mediator_kind": self.mediator_kind,
            "route_symbols": self.route_symbols,
            "outcome_kind": self.outcome_kind,
        }

    def period_factory_kwargs(self) -> dict[str, Any]:
        return {
            "mediator_symbol": self.mediator_symbol,
            "outcome_symbol": self.outcome_symbol,
            "confounder_symbols": self.effective_confounders,
        }

    def recipe_markdown(self, *, title: str) -> str:
        return _recipe(
            title=title,
            model_id=self.model_id,
            design=self.design,
            estimand=(
                f"{self.estimand.capitalize()} direct and indirect effects through "
                f"{self.mediator_symbol}, expressed on the {self.outcome_symbol} scale."
            ),
            causal_status=self.causal_status,
            population=self.analysis_population,
            missing=self.missing_data_assumption,
            terms=(
                f"Declared confounders: {', '.join(self.declared_confounders) or 'none'}. "
                f"Active confounders: {', '.join(self.effective_confounders) or 'none'}. "
                f"Mediator likelihood: {self.mediator_kind}; outcome likelihood: "
                f"{self.outcome_kind}."
            ),
        )


@dataclass(frozen=True, slots=True)
class MediationMultiRunPlan:
    """Concrete instructions for one two-mediator decomposition."""

    model_id: str
    settings_source: str
    outcome_symbol: str
    mediators: tuple[str, str]
    order: tuple[str, str]
    chain: bool
    second_mediator_offfloor: bool
    outcomes: tuple[str, ...] | None
    named_confounder_calibration: NamedConfounderCalibration | None
    declared_confounders: tuple[str, ...]
    effective_confounders: tuple[str, ...]
    raw_covariates: tuple[str, ...]
    loaded_covariates: tuple[str, ...]
    observation_nodes: tuple[str, ...]
    compute_loo: bool
    design: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_effective_confounders(
        self, confounders: tuple[str, ...]
    ) -> MediationMultiRunPlan:
        unknown = sorted(set(confounders) - set(self.declared_confounders))
        if unknown:
            raise ValueError(f"effective confounders were not declared: {unknown}")
        return replace(self, effective_confounders=tuple(confounders))

    def prepare_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "phase_mode": "itt",
            "covariates": self.loaded_covariates,
        }
        if self.outcomes is not None:
            kwargs["outcomes"] = self.outcomes
        return kwargs

    def factory_kwargs(self) -> dict[str, Any]:
        return {
            "outcome_symbol": self.outcome_symbol,
            "mediator_symbols": self.mediators,
            "confounder_symbols": self.effective_confounders,
            "chain": self.chain,
            "second_mediator_offfloor": self.second_mediator_offfloor,
        }

    def recipe_markdown(self, *, title: str) -> str:
        return _recipe(
            title=title,
            model_id=self.model_id,
            design=self.design,
            estimand=(
                "Joint and ordering-dependent path-specific indirect effects through "
                f"{self.mediators[0]} and {self.mediators[1]}."
            ),
            causal_status=self.causal_status,
            population=self.analysis_population,
            missing=self.missing_data_assumption,
            terms=(
                f"Mediator order: {', '.join(self.order)}. Sequential chain: "
                f"{self.chain}. Declared confounders: "
                f"{', '.join(self.declared_confounders) or 'none'}. Active "
                f"confounders: {', '.join(self.effective_confounders) or 'none'}."
            ),
        )


def _recipe(
    *,
    title: str,
    model_id: str,
    design: str,
    estimand: str,
    causal_status: str,
    population: str,
    missing: str,
    terms: str,
) -> str:
    return (
        "Note: Generated from the validated mediation run plan; template drafted "
        "by a LLM-based AI tool (Codex/GPT-5).\n\n"
        f"# Model recipe: {title}\n\nModel ID: `{model_id}`.\n\n"
        f"## Design\n\n{design}\n\n## Estimand\n\n{estimand}\n\n"
        f"## Causal status\n\n{causal_status}\n\n"
        f"## Analysis population\n\n{population}\n\n"
        f"## Missing data\n\n{missing}\n\n## Terms\n\n{terms}\n\n"
        "## Uncertainty and checks\n\nThis family does not compute ordinary "
        "PSIS-LOO because the reported estimand is a counterfactual simulation "
        "rather than one pointwise predictive density. Release requires the "
        "convergence gate, posterior-predictive checks, power-scaling sensitivity "
        "and the registered temporal or confounding checks. The saved `config.json` "
        "contains the same resolved run plan.\n"
    )


def declared_mediation_settings(
    spec: ModelSpec,
) -> tuple[MediationModelSettings, str]:
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: mediation settings cannot be split between "
                f"model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, MediationModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='mediation' requires MediationModelSettings, "
                f"got {type(settings).__name__}"
            )
        return settings, "typed"
    return MediationModelSettings.from_legacy_extra(
        spec.extra, model_id=spec.model_id
    ), "legacy_extra"


def declared_mediation_multi_settings(
    spec: ModelSpec,
) -> tuple[MediationMultiModelSettings, str]:
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: multi-mediation settings cannot be split between "
                f"model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, MediationMultiModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='mediation_multi' requires "
                f"MediationMultiModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return MediationMultiModelSettings.from_legacy_extra(
        spec.extra, model_id=spec.model_id
    ), "legacy_extra"


def resolve_mediation_run_plan(spec: ModelSpec) -> MediationRunPlan:
    """Validate one single-mediator declaration before context or data I/O."""
    if spec.kind != "mediation":
        raise ValueError(f"{spec.model_id}: expected kind 'mediation', got {spec.kind!r}")
    if spec.study_id != "rli":
        raise ValueError(f"{spec.model_id}: mediation requires study_id='rli'")
    if not spec.outcome_symbol:
        raise ValueError(f"{spec.model_id}: mediation requires an outcome symbol")
    settings, source = declared_mediation_settings(spec)
    mediator_symbol = spec.mechanism_symbol or "L"
    if settings.mediator_kind == "gaussian_composite":
        if not settings.route_symbols:
            raise ValueError("gaussian_composite mediation requires route_symbols")
    elif settings.route_symbols:
        raise ValueError("route_symbols apply only to gaussian_composite mediation")
    if settings.companion_of and settings.estimand != "interventional":
        raise ValueError("companion_of requires estimand='interventional'")
    if settings.outcomes is not None:
        required = {spec.outcome_symbol, mediator_symbol, *settings.route_symbols}
        missing = sorted(required - set(settings.outcomes))
        if missing:
            raise ValueError(f"outcomes omits required measure(s): {', '.join(missing)}")
    if settings.period_stacked and any(
        (
            settings.outcome_time is not None,
            settings.mediator_kind != "beta_binomial",
            settings.outcome_kind != "beta_binomial",
            settings.estimand != "natural",
            bool(settings.route_symbols),
        )
    ):
        raise ValueError("period-stacked mediation requires the graded natural-effects path")
    markers = (
        ("T", "A", "W_pre", f"{mediator_symbol}_pre")
        if settings.period_stacked
        else ("G", "A", "W_pre", f"{mediator_symbol}_t1")
    )
    confounders = tuple(symbol for symbol in spec.adjustment if symbol not in markers)
    raw = _raw_covariates(confounders)
    mediator_node = (
        "M_post"
        if settings.mediator_kind == "gaussian_composite"
        else f"{mediator_symbol}_post"
    )
    outcome_node = (
        "y_offfloor"
        if settings.outcome_kind == "bernoulli_offfloor"
        else "y_post"
    )
    entrypoint: Literal["single", "period_stacked"] = (
        "period_stacked" if settings.period_stacked else "single"
    )
    design = (
        "Period-stacked all-transition mediation with per-period treatment exposure."
        if settings.period_stacked
        else "Available-case phase-1 treatment mediation with counterfactual simulation."
    )
    return MediationRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        entrypoint=entrypoint,
        outcome_symbol=spec.outcome_symbol,
        mediator_symbol=mediator_symbol,
        outcomes=settings.outcomes,
        drop_missing_pre=settings.drop_missing_pre,
        outcome_time=settings.outcome_time,
        mediator_kind=settings.mediator_kind,
        route_symbols=settings.route_symbols,
        outcome_kind=settings.outcome_kind,
        estimand=settings.estimand,
        companion_of=settings.companion_of,
        declared_confounders=confounders,
        effective_confounders=confounders,
        raw_covariates=raw,
        observation_nodes=(mediator_node, outcome_node),
        compute_loo=False,
        design=design,
        causal_status=(
            "Exploratory decomposition under measured and unmeasured-confounding "
            "assumptions; not an identified causal mediation effect."
        ),
        analysis_population=(
            "Available RLI children with every outcome, mediator and declared "
            "confounder needed by this fit."
        ),
        missing_data_assumption=(
            "Complete-case analysis assumes the fitted rows are conditionally "
            "representative for the declared decomposition."
        ),
    )


def resolve_mediation_multi_run_plan(spec: ModelSpec) -> MediationMultiRunPlan:
    """Validate one two-mediator declaration before context or data I/O."""
    if spec.kind != "mediation_multi":
        raise ValueError(
            f"{spec.model_id}: expected kind 'mediation_multi', got {spec.kind!r}"
        )
    if spec.study_id != "rli" or not spec.outcome_symbol:
        raise ValueError(f"{spec.model_id}: multi-mediation requires an RLI outcome")
    settings, source = declared_mediation_multi_settings(spec)
    if len(settings.mediators) != 2:
        raise ValueError("multi-mediation requires exactly two mediators")
    if settings.mediators[0] != "L":
        raise ValueError("the first multi-mediation mediator must be 'L'")
    if set(settings.order) != set(settings.mediators) or len(settings.order) != 2:
        raise ValueError("order must be a permutation of mediators")
    if settings.outcomes is not None:
        required = {spec.outcome_symbol, *settings.mediators}
        missing = sorted(required - set(settings.outcomes))
        if missing:
            raise ValueError(f"outcomes omits required measure(s): {', '.join(missing)}")
    baselines = tuple(f"{symbol}_t1" for symbol in settings.mediators)
    confounders = tuple(
        symbol
        for symbol in spec.adjustment
        if symbol not in ("G", "A", "W_pre", *baselines)
    )
    raw = _raw_covariates(confounders)
    calibration = settings.named_confounder_calibration
    loaded = tuple(dict.fromkeys((*raw, *((calibration.symbol,) if calibration else ()))))
    mediators = (settings.mediators[0], settings.mediators[1])
    order = (settings.order[0], settings.order[1])
    second_node = (
        f"{mediators[1]}_offfloor"
        if settings.second_mediator_offfloor
        else f"{mediators[1]}_post"
    )
    return MediationMultiRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        outcome_symbol=spec.outcome_symbol,
        mediators=mediators,
        order=order,
        chain=settings.chain,
        second_mediator_offfloor=settings.second_mediator_offfloor,
        outcomes=settings.outcomes,
        named_confounder_calibration=calibration,
        declared_confounders=confounders,
        effective_confounders=confounders,
        raw_covariates=raw,
        loaded_covariates=loaded,
        observation_nodes=(f"{mediators[0]}_post", second_node, "y_post"),
        compute_loo=False,
        design="Available-case phase-1 two-mediator counterfactual simulation.",
        causal_status=(
            "Exploratory decomposition under ordering and confounding assumptions; "
            "not an identified causal mediation effect."
        ),
        analysis_population=(
            "Available RLI children with every outcome, mediator and declared "
            "confounder needed by this fit."
        ),
        missing_data_assumption=(
            "Complete-case analysis assumes the fitted rows are conditionally "
            "representative for the declared decomposition."
        ),
    )
