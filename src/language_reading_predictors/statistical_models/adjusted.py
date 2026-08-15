# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and resolved plans for RLI and RLM adjusted associations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec

__all__ = [
    "AdjustedModelSettings",
    "AdjustedRunPlan",
    "declared_adjusted_settings",
    "resolve_adjusted_run_plan",
]

_GLOBAL_KEYS = frozenset({"target_accept"})
_FAMILY_KEYS = frozenset(
    {
        "design",
        "post_time",
        "predictor_symbols",
        "language_composite_symbols",
        "covariates",
        "ses_covariates",
        "predictor_measures",
        "use_age_predictor",
        "pre_wave",
        "post_wave",
        "group_codes",
        "require_confirmed_inputs",
        "predictor_slope_sigma",
        "prior_sensitivity_sigmas",
        "study_id",
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


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_positive_ints(
    value: Any, *, name: str
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integers or None")
    out = tuple(value)
    if not out:
        raise ValueError(f"{name} cannot be empty")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in out):
        raise TypeError(f"{name} must contain integers")
    if any(item < 1 for item in out):
        raise ValueError(f"{name} must contain positive integers")
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicates")
    return out


@dataclass(frozen=True, slots=True)
class AdjustedModelSettings:
    """Immutable declaration shared by the intervention and Byrne ports."""

    design: str | None = None
    post_time: int | None = None
    predictor_symbols: tuple[str, ...] | None = None
    language_composite_symbols: tuple[str, ...] | None = None
    covariates: tuple[str, ...] | None = None
    ses_covariates: tuple[str, ...] | None = None
    predictor_measures: tuple[str, ...] | None = None
    use_age_predictor: bool = True
    pre_wave: int | None = None
    post_wave: int | None = None
    group_codes: tuple[int, ...] | None = None
    require_confirmed_inputs: bool = False
    predictor_slope_sigma: float = 0.3
    prior_sensitivity_sigmas: tuple[float, ...] = (0.5, 0.7)

    def __post_init__(self) -> None:
        if self.design is not None and (
            not isinstance(self.design, str) or not self.design
        ):
            raise TypeError("design must be a non-empty string or None")
        for name in (
            "predictor_symbols",
            "language_composite_symbols",
            "covariates",
            "ses_covariates",
            "predictor_measures",
        ):
            object.__setattr__(
                self,
                name,
                _optional_symbols(getattr(self, name), name=name),
            )
        if not isinstance(self.use_age_predictor, bool):
            raise TypeError("use_age_predictor must be a boolean")
        if not isinstance(self.require_confirmed_inputs, bool):
            raise TypeError("require_confirmed_inputs must be a boolean")
        for name in ("post_time", "pre_wave", "post_wave"):
            object.__setattr__(
                self,
                name,
                _optional_positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "group_codes",
            _optional_positive_ints(self.group_codes, name="group_codes"),
        )
        object.__setattr__(
            self,
            "predictor_slope_sigma",
            _positive_float(
                self.predictor_slope_sigma, name="predictor_slope_sigma"
            ),
        )
        if isinstance(self.prior_sensitivity_sigmas, str) or not isinstance(
            self.prior_sensitivity_sigmas, Sequence
        ):
            raise TypeError("prior_sensitivity_sigmas must be a sequence")
        sigmas = tuple(
            _positive_float(value, name="prior_sensitivity_sigmas")
            for value in self.prior_sensitivity_sigmas
        )
        if len(sigmas) != len(set(sigmas)):
            raise ValueError("prior_sensitivity_sigmas contains duplicates")
        object.__setattr__(self, "prior_sensitivity_sigmas", sigmas)

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
    ) -> AdjustedModelSettings:
        unknown = sorted(set(extra) - _FAMILY_KEYS - _GLOBAL_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown adjusted setting(s): {', '.join(unknown)}"
            )
        return cls(
            design=extra.get("design"),
            post_time=extra.get("post_time"),
            predictor_symbols=extra.get("predictor_symbols"),
            language_composite_symbols=extra.get("language_composite_symbols"),
            covariates=extra.get("covariates"),
            ses_covariates=extra.get("ses_covariates"),
            predictor_measures=extra.get("predictor_measures"),
            use_age_predictor=extra.get("use_age_predictor", True),
            pre_wave=extra.get("pre_wave"),
            post_wave=extra.get("post_wave"),
            group_codes=extra.get("group_codes"),
            require_confirmed_inputs=extra.get("require_confirmed_inputs", False),
            predictor_slope_sigma=extra.get("predictor_slope_sigma", 0.3),
            prior_sensitivity_sigmas=extra.get(
                "prior_sensitivity_sigmas", (0.5, 0.7)
            ),
        )


@dataclass(frozen=True, slots=True)
class AdjustedRunPlan:
    """Concrete, validated instructions for one adjusted-association fit."""

    model_id: str
    settings_source: str
    study_id: Literal["rli", "rlm"]
    port: Literal["rli", "rlm"]
    outcome_symbol: str
    design: str
    post_time: int | None
    predictor_symbols: tuple[str, ...]
    language_composite_symbols: tuple[str, ...]
    declared_covariates: tuple[str, ...]
    active_covariates: tuple[str, ...]
    ses_covariates: tuple[str, ...]
    predictor_measures: tuple[str, ...]
    use_age_predictor: bool
    pre_wave: int | None
    post_wave: int | None
    group_codes: tuple[int, ...] | None
    require_confirmed_inputs: bool
    predictor_slope_sigma: float
    prior_sensitivity_sigmas: tuple[float, ...]
    observation_nodes: tuple[str, ...]
    compute_loo: bool
    loo_unit: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_active_covariates(
        self, covariates: tuple[str, ...]
    ) -> AdjustedRunPlan:
        if self.port != "rli":
            raise ValueError("active covariates apply only to the RLI port")
        unknown = sorted(set(covariates) - set(self.declared_covariates))
        if unknown:
            raise ValueError(f"active covariates were not declared: {unknown}")
        return replace(self, active_covariates=tuple(covariates))

    def headline_predictors(self) -> tuple[str, ...]:
        if self.port != "rli":
            raise ValueError("headline_predictors applies only to the RLI port")
        return (
            *self.predictor_symbols,
            "lang",
            *(("age",) if self.use_age_predictor else ()),
            *self.active_covariates,
        )

    def rli_prepare_kwargs(
        self, *, include_ses: bool = False
    ) -> dict[str, Any]:
        if self.port != "rli" or self.post_time is None:
            raise ValueError("rli_prepare_kwargs requires an RLI plan")
        covariates = self.active_covariates
        if include_ses:
            covariates = (*covariates, *self.ses_covariates)
        outcomes = tuple(
            dict.fromkeys(
                (
                    self.outcome_symbol,
                    *self.predictor_symbols,
                    *self.language_composite_symbols,
                )
            )
        )
        return {
            "phase_mode": "span",
            "post_time": self.post_time,
            "outcomes": outcomes,
            "covariates": covariates,
        }

    def rli_factory_kwargs(
        self, *, predictors: Sequence[str] | None = None
    ) -> dict[str, Any]:
        if self.port != "rli":
            raise ValueError("rli_factory_kwargs requires an RLI plan")
        return {
            "outcome_symbol": self.outcome_symbol,
            "predictors": tuple(predictors or self.headline_predictors()),
            "language_composite_symbols": self.language_composite_symbols,
            "predictor_slope_sigma": self.predictor_slope_sigma,
        }

    def rlm_prepare_kwargs(self) -> dict[str, Any]:
        if self.port != "rlm" or self.pre_wave is None or self.post_wave is None:
            raise ValueError("rlm_prepare_kwargs requires an RLM plan")
        return {
            "outcome": self.outcome_symbol,
            "predictor_measures": self.predictor_measures,
            "include_age": self.use_age_predictor,
            "pre_wave": self.pre_wave,
            "post_wave": self.post_wave,
            "group_codes": self.group_codes,
        }

    def rlm_factory_kwargs(self, predictors: Sequence[str]) -> dict[str, Any]:
        if self.port != "rlm":
            raise ValueError("rlm_factory_kwargs requires an RLM plan")
        return {
            "predictors": tuple(predictors),
            "predictor_slope_sigma": self.predictor_slope_sigma,
        }

    def diagnostic_vars(
        self, predictors: Sequence[str], nuisance: Sequence[str] = ()
    ) -> list[str]:
        return [
            "alpha",
            "gamma_own",
            "kappa",
            *(f"beta_{key}" for key in predictors),
            *nuisance,
        ]

    def recipe_markdown(self, *, title: str) -> str:
        if self.port == "rli":
            terms = (
                f"Baseline skills: {', '.join(self.predictor_symbols)}; language "
                f"composite: {', '.join(self.language_composite_symbols)}; age: "
                f"{self.use_age_predictor}; active tested covariates: "
                f"{', '.join(self.active_covariates) or 'none'}."
            )
            design = (
                f"One row per RLI child from t1 to t{self.post_time}; the final "
                "bounded score is conditioned on its own t1 score."
            )
            sensitivity_checks = (
                "Every bivariate, prior-width and SES sensitivity refit"
            )
        else:
            population = (
                "all observational reading groups"
                if self.group_codes is None
                else "group code(s) " + ", ".join(map(str, self.group_codes))
            )
            terms = (
                f"Wave-{self.pre_wave} predictors: "
                f"{', '.join(self.predictor_measures)}; age: "
                f"{self.use_age_predictor}; population: {population}. "
                f"Confirmed measurement inputs required: "
                f"{self.require_confirmed_inputs}. "
                + (
                    "Observational group indicators are nuisance terms."
                    if self.group_codes is None or len(self.group_codes) > 1
                    else "No group nuisance term is fitted within one selected group."
                )
            )
            design = (
                f"One row per Byrne child from wave {self.pre_wave} to wave "
                f"{self.post_wave}, restricted to {population}."
            )
            sensitivity_checks = "Every bivariate and prior-width sensitivity refit"
        return (
            "Note: Generated from the validated adjusted-association run plan; "
            "template drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\nModel ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{design}\n\n## Estimand\n\nMutually adjusted "
            "between-child associations and their baseline-only bivariate "
            "comparators, with +1 SD contrasts translated to outcome items.\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            f"## Terms\n\n{terms} Predictor-slope prior SD: "
            f"{self.predictor_slope_sigma:g}; sensitivity SDs: "
            f"{', '.join(f'{value:g}' for value in self.prior_sensitivity_sigmas)}.\n\n"
            "## Uncertainty and checks\n\nRelease requires the convergence gate, "
            "child-level PSIS-LOO and posterior-predictive checks. "
            f"{sensitivity_checks} records its own convergence and "
            "row provenance. The saved `config.json` contains the same resolved run "
            "plan.\n"
        )


def declared_adjusted_settings(
    spec: ModelSpec,
) -> tuple[AdjustedModelSettings, str]:
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: adjusted settings cannot be split between "
                f"model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, AdjustedModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='adjusted' requires AdjustedModelSettings, "
                f"got {type(settings).__name__}"
            )
        return settings, "typed"
    return AdjustedModelSettings.from_legacy_extra(
        spec.extra, model_id=spec.model_id
    ), "legacy_extra"


def resolve_adjusted_run_plan(spec: ModelSpec) -> AdjustedRunPlan:
    """Resolve either cohort port before context creation or data loading."""
    if spec.kind != "adjusted":
        raise ValueError(f"{spec.model_id}: expected kind 'adjusted', got {spec.kind!r}")
    if spec.study_id not in {"rli", "rlm"}:
        raise ValueError(f"{spec.model_id}: adjusted study_id must be 'rli' or 'rlm'")
    if not spec.outcome_symbol:
        raise ValueError(f"{spec.model_id}: adjusted requires an outcome symbol")
    settings, source = declared_adjusted_settings(spec)
    legacy_study = spec.extra.get("study_id")
    if legacy_study is not None and legacy_study != spec.study_id:
        raise ValueError(
            f"{spec.model_id}: legacy study_id conflicts with ModelSpec.study_id"
        )
    if settings.predictor_slope_sigma in settings.prior_sensitivity_sigmas:
        raise ValueError(
            "prior_sensitivity_sigmas must not repeat predictor_slope_sigma"
        )

    if spec.study_id == "rli":
        rlm_only = {
            "predictor_measures": settings.predictor_measures,
            "pre_wave": settings.pre_wave,
            "post_wave": settings.post_wave,
            "group_codes": settings.group_codes,
            "require_confirmed_inputs": (
                True if settings.require_confirmed_inputs else None
            ),
        }
        supplied = [name for name, value in rlm_only.items() if value is not None]
        if supplied:
            raise ValueError(f"RLM-only adjusted settings on RLI port: {supplied}")
        port: Literal["rli", "rlm"] = "rli"
        design = settings.design or "between_child"
        if design != "between_child":
            raise ValueError("RLI adjusted design must be 'between_child'")
        post_time = 4 if settings.post_time is None else settings.post_time
        if post_time < 2:
            raise ValueError("post_time must be at least 2")
        predictor_symbols = (
            ("L", "B")
            if settings.predictor_symbols is None
            else settings.predictor_symbols
        )
        language_symbols = (
            ("R", "E", "F")
            if settings.language_composite_symbols is None
            else settings.language_composite_symbols
        )
        if not predictor_symbols or not language_symbols:
            raise ValueError("RLI adjusted predictor and language sets cannot be empty")
        covariates = (
            ("blocks", "behav")
            if settings.covariates is None
            else settings.covariates
        )
        ses_covariates = (
            ("mumedupost16",)
            if settings.ses_covariates is None
            else settings.ses_covariates
        )
        predictor_measures: tuple[str, ...] = ()
        pre_wave = None
        post_wave = None
        group_codes = None
        population = (
            f"Available RLI children with t1 predictors and {spec.outcome_symbol} "
            f"observed through t{post_time}."
        )
        missing = (
            "The headline uses available rows after missing-indicator covariates; "
            "parental education is tested in a separate complete-case sensitivity."
        )
    else:
        rli_only = {
            "design": settings.design,
            "post_time": settings.post_time,
            "predictor_symbols": settings.predictor_symbols,
            "language_composite_symbols": settings.language_composite_symbols,
            "covariates": settings.covariates,
            "ses_covariates": settings.ses_covariates,
        }
        supplied = [name for name, value in rli_only.items() if value is not None]
        if supplied:
            raise ValueError(f"RLI-only adjusted settings on RLM port: {supplied}")
        port = "rlm"
        design = "historical_cohort"
        post_time = None
        predictor_symbols = ()
        language_symbols = ()
        covariates = ()
        ses_covariates = ()
        predictor_measures = (
            ("bpvs", "trog", "basdig", "bassim", "basnum")
            if settings.predictor_measures is None
            else settings.predictor_measures
        )
        if not predictor_measures:
            raise ValueError("RLM adjusted predictor_measures cannot be empty")
        pre_wave = 1 if settings.pre_wave is None else settings.pre_wave
        post_wave = 3 if settings.post_wave is None else settings.post_wave
        if post_wave <= pre_wave:
            raise ValueError("post_wave must be later than pre_wave")
        from language_reading_predictors.statistical_models.datasets import (
            resolve_dataset,
        )

        dataset, measures = resolve_dataset("rlm")
        requested = (spec.outcome_symbol, *predictor_measures)
        unknown_measures = sorted(set(requested) - set(measures))
        if unknown_measures:
            raise ValueError(
                "unknown RLM adjusted measure(s): "
                + ", ".join(unknown_measures)
            )
        if settings.require_confirmed_inputs:
            unresolved = [
                symbol
                for symbol in requested
                if not measures[symbol].n_trials_confirmed
                or not measures[symbol].instrument_identity_confirmed
            ]
            if unresolved:
                raise ValueError(
                    "RLM adjusted model requires confirmed denominators and "
                    "instrument identities; unresolved: "
                    + ", ".join(dict.fromkeys(unresolved))
                )
        group_codes = settings.group_codes
        if group_codes is not None:
            unknown_groups = sorted(set(group_codes) - set(dataset.group_labels))
            if unknown_groups:
                raise ValueError(
                    "unknown RLM group_codes: "
                    + ", ".join(map(str, unknown_groups))
                )
        group_description = (
            "all observational reading groups"
            if group_codes is None
            else ", ".join(
                dataset.group_labels[code] for code in group_codes
            )
        )
        population = (
            f"Complete-case Byrne children in {group_description}, observed at "
            f"waves {pre_wave} and {post_wave} with every declared baseline predictor."
        )
        missing = (
            "Complete-case analysis assumes included children are conditionally "
            "representative of the target historical cohort."
        )

    return AdjustedRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        port=port,
        outcome_symbol=spec.outcome_symbol,
        design=design,
        post_time=post_time,
        predictor_symbols=predictor_symbols,
        language_composite_symbols=language_symbols,
        declared_covariates=covariates,
        active_covariates=covariates,
        ses_covariates=ses_covariates,
        predictor_measures=predictor_measures,
        use_age_predictor=settings.use_age_predictor,
        pre_wave=pre_wave,
        post_wave=post_wave,
        group_codes=group_codes,
        require_confirmed_inputs=settings.require_confirmed_inputs,
        predictor_slope_sigma=settings.predictor_slope_sigma,
        prior_sensitivity_sigmas=settings.prior_sensitivity_sigmas,
        observation_nodes=("y_post",),
        compute_loo=True,
        loo_unit="child",
        causal_status=(
            "Adjusted between-child association, not an intervention effect."
        ),
        analysis_population=population,
        missing_data_assumption=missing,
    )
