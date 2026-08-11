# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and resolved plans for RLI and RLM horseshoe models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec

__all__ = [
    "HorseshoeModelSettings",
    "HorseshoeRunPlan",
    "declared_horseshoe_settings",
    "resolve_horseshoe_run_plan",
]


_FAMILY_KEYS = frozenset(
    {
        "gain",
        "predictors",
        "language_composite_symbols",
        "covariates",
        "delta",
        "tau0",
        "slab_scale",
        "slab_df",
        "post_time",
        "phase_mode",
        "gb_reference",
        "predictor_measures",
        "use_age_predictor",
        "pre_wave",
        "post_wave",
        # Redundant legacy declaration; the resolver validates it against the spec.
        "study_id",
    }
)
_GLOBAL_KEYS = frozenset({"target_accept"})
_LEGACY_KEYS = _FAMILY_KEYS | _GLOBAL_KEYS


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean or None, got {value!r}")
    return value


def _optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string or None, got {value!r}")
    return value


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
        raise TypeError(f"{name} must be an integer or None, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True, slots=True)
class HorseshoeModelSettings:
    """Immutable declaration shared by the RLI and historical RLM ports."""

    gain: bool | None = None
    predictors: tuple[str, ...] = ()
    language_composite_symbols: tuple[str, ...] = ()
    covariates: tuple[str, ...] = ()
    delta: float = 0.1
    tau0: float = 0.1
    slab_scale: float = 2.0
    slab_df: float = 4.0
    post_time: int | None = None
    phase_mode: str | None = None
    gb_reference: str | None = None
    predictor_measures: tuple[str, ...] = ()
    use_age_predictor: bool | None = None
    pre_wave: int | None = None
    post_wave: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gain", _optional_bool(self.gain, name="gain"))
        for name in (
            "predictors",
            "language_composite_symbols",
            "covariates",
            "predictor_measures",
        ):
            object.__setattr__(
                self,
                name,
                _tuple_of_strings(getattr(self, name), name=name),
            )
        for name in ("delta", "tau0", "slab_scale", "slab_df"):
            object.__setattr__(
                self,
                name,
                _positive_float(getattr(self, name), name=name),
            )
        for name in ("post_time", "pre_wave", "post_wave"):
            object.__setattr__(
                self,
                name,
                _optional_positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "phase_mode",
            _optional_string(self.phase_mode, name="phase_mode"),
        )
        object.__setattr__(
            self,
            "gb_reference",
            _optional_string(self.gb_reference, name="gb_reference"),
        )
        object.__setattr__(
            self,
            "use_age_predictor",
            _optional_bool(self.use_age_predictor, name="use_age_predictor"),
        )

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
    ) -> HorseshoeModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown horseshoe setting(s): {', '.join(unknown)}. "
                "Declare HorseshoeModelSettings so misspellings fail fast."
            )
        return cls(
            gain=extra.get("gain"),
            predictors=extra.get("predictors", ()),
            language_composite_symbols=extra.get(
                "language_composite_symbols",
                (),
            ),
            covariates=extra.get("covariates", ()),
            delta=extra.get("delta", 0.1),
            tau0=extra.get("tau0", 0.1),
            slab_scale=extra.get("slab_scale", 2.0),
            slab_df=extra.get("slab_df", 4.0),
            post_time=extra.get("post_time"),
            phase_mode=extra.get("phase_mode"),
            gb_reference=extra.get("gb_reference"),
            predictor_measures=extra.get("predictor_measures", ()),
            use_age_predictor=extra.get("use_age_predictor"),
            pre_wave=extra.get("pre_wave"),
            post_wave=extra.get("post_wave"),
        )


@dataclass(frozen=True, slots=True)
class HorseshoeRunPlan:
    """Concrete, validated instructions for one RLI or RLM horseshoe fit."""

    model_id: str
    settings_source: str
    study_id: Literal["rli", "rlm"]
    port: Literal["rli", "rlm"]
    outcome_symbol: str
    gain: bool
    predictors: tuple[str, ...]
    language_composite_symbols: tuple[str, ...]
    covariates: tuple[str, ...]
    measure_symbols: tuple[str, ...]
    phase_mode: str | None
    post_time: int | None
    predictor_measures: tuple[str, ...]
    use_age_predictor: bool | None
    pre_wave: int | None
    post_wave: int | None
    delta: float
    tau0: float
    slab_scale: float
    slab_df: float
    gb_reference: str | None
    observation_node: str
    compute_loo: bool
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

    def rli_prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for RLI ``load_and_prepare``."""
        if self.port != "rli":
            raise ValueError("rli_prepare_kwargs requires an RLI horseshoe plan")
        return {
            "phase_mode": self.phase_mode,
            "post_time": self.post_time,
            "outcomes": self.measure_symbols,
            "covariates": self.covariates,
        }

    def rli_factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_horseshoe_model``."""
        if self.port != "rli":
            raise ValueError("rli_factory_kwargs requires an RLI horseshoe plan")
        return {
            "outcome_symbol": self.outcome_symbol,
            "predictors": list(self.predictors),
            "gain": self.gain,
            "tau0": self.tau0,
            "slab_scale": self.slab_scale,
            "slab_df": self.slab_df,
            "language_composite_symbols": self.language_composite_symbols,
        }

    def rlm_prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_rlm_span_frame``."""
        if self.port != "rlm":
            raise ValueError("rlm_prepare_kwargs requires an RLM horseshoe plan")
        return {
            "outcome": self.outcome_symbol,
            "predictor_measures": self.predictor_measures,
            "include_age": self.use_age_predictor,
            "pre_wave": self.pre_wave,
            "post_wave": self.post_wave,
        }

    def rlm_factory_kwargs(self, *, predictors: list[str]) -> dict[str, Any]:
        """Arguments for ``build_rlm_horseshoe_model``."""
        if self.port != "rlm":
            raise ValueError("rlm_factory_kwargs requires an RLM horseshoe plan")
        return {
            "predictors": predictors,
            "tau0": self.tau0,
            "slab_scale": self.slab_scale,
            "slab_df": self.slab_df,
        }

    def diagnostic_vars(self, *, nuisance: tuple[str, ...] = ()) -> list[str]:
        """Variables scanned by summaries and the convergence gate."""
        if self.port == "rlm":
            return [
                "alpha",
                "gamma_own",
                "kappa",
                "hs_tau",
                "hs_c2",
                "beta",
                *nuisance,
            ]
        coupling = ["gamma_own"] if self.gain else []
        if not self.gain and "age" not in self.predictors:
            coupling = ["gamma_A"]
        return ["alpha", *coupling, "kappa", "hs_tau", "hs_c2", "beta"]

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language recipe generated from the validated plan."""
        predictors = ", ".join(
            self.predictors if self.port == "rli" else self.predictor_measures
        )
        return (
            "Note: Generated from the validated horseshoe run plan; template "
            "drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Study port: `{self.port}`. Outcome: `{self.outcome_symbol}`. "
            f"Framing: {'gain' if self.gain else 'level'}. Ranked predictors: "
            f"{predictors}. Ranking threshold: {self.delta:g}. Regularised "
            f"horseshoe hyperparameters: tau0={self.tau0:g}, slab scale="
            f"{self.slab_scale:g}, slab df={self.slab_df:g}.\n\n"
            "## Uncertainty and checks\n\n"
            f"The observation node is `{self.observation_node}` and PSIS-LOO uses "
            f"the `{self.loo_unit}` unit. Horseshoe rankings require a "
            "zero-divergence fit, adequate effective sample sizes, "
            "posterior-predictive checks and power-scaling sensitivity review. "
            "The saved `config.json` contains the same resolved run plan in "
            "machine-readable form.\n"
        )


def declared_horseshoe_settings(
    spec: ModelSpec,
) -> tuple[HorseshoeModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: horseshoe settings cannot be split between "
                f"model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, HorseshoeModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='horseshoe' requires "
                f"HorseshoeModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        HorseshoeModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
        ),
        "legacy_extra",
    )


def resolve_horseshoe_run_plan(spec: ModelSpec) -> HorseshoeRunPlan:
    """Resolve and validate either study port before context or data I/O."""
    if spec.kind != "horseshoe":
        raise ValueError(f"{spec.model_id}: expected kind 'horseshoe', got {spec.kind!r}")
    if spec.study_id not in {"rli", "rlm"}:
        raise ValueError(
            f"{spec.model_id}: horseshoe study_id must be 'rli' or 'rlm', got "
            f"{spec.study_id!r}"
        )
    if not spec.outcome_symbol:
        raise ValueError(f"{spec.model_id}: outcome_symbol is required for horseshoe")

    settings, source = declared_horseshoe_settings(spec)
    legacy_study = spec.extra.get("study_id")
    if legacy_study is not None and legacy_study != spec.study_id:
        raise ValueError(
            f"{spec.model_id}: legacy study_id {legacy_study!r} conflicts with "
            f"ModelSpec.study_id {spec.study_id!r}"
        )

    outcome = spec.outcome_symbol
    if spec.study_id == "rli":
        rlm_fields = {
            "predictor_measures": settings.predictor_measures,
            "use_age_predictor": settings.use_age_predictor,
            "pre_wave": settings.pre_wave,
            "post_wave": settings.post_wave,
        }
        supplied = [name for name, value in rlm_fields.items() if value not in ((), None)]
        if supplied:
            raise ValueError(
                f"{spec.model_id}: RLM-only settings are invalid for the RLI port: "
                f"{', '.join(supplied)}"
            )
        if not settings.predictors:
            raise ValueError(f"{spec.model_id}: RLI horseshoe predictors cannot be empty")
        gain = True if settings.gain is None else settings.gain
        language = settings.language_composite_symbols or ("R", "E", "F")
        phase_mode = settings.phase_mode or ("span" if gain else "levels")
        if phase_mode not in {"span", "levels"}:
            raise ValueError(
                f"{spec.model_id}: RLI phase_mode must be 'span' or 'levels', got "
                f"{phase_mode!r}"
            )
        post_time = 4 if settings.post_time is None else settings.post_time
        measures = tuple(
            dict.fromkeys(
                (outcome,)
                + tuple(
                    predictor
                    for predictor in settings.predictors
                    if predictor not in ("age", "lang", *settings.covariates)
                )
                + language
            )
        )
        port: Literal["rli", "rlm"] = "rli"
        predictor_measures: tuple[str, ...] = ()
        use_age: bool | None = None
        pre_wave: int | None = None
        rlm_post_wave: int | None = None
        design = (
            "RLI between-child span regression for gain models or repeated levels "
            "regression for level models, with a regularised horseshoe over the "
            "declared construct predictor set."
        )
        population = (
            f"Available RLI rows with observed {outcome} and every declared "
            "predictor, composite component and covariate."
        )
    else:
        rli_fields = {
            "gain": settings.gain,
            "predictors": settings.predictors,
            "language_composite_symbols": settings.language_composite_symbols,
            "covariates": settings.covariates,
            "phase_mode": settings.phase_mode,
            "post_time": settings.post_time,
            "gb_reference": settings.gb_reference,
        }
        supplied = [name for name, value in rli_fields.items() if value not in ((), None)]
        if supplied:
            raise ValueError(
                f"{spec.model_id}: RLI-only settings are invalid for the RLM port: "
                f"{', '.join(supplied)}"
            )
        predictor_measures = settings.predictor_measures or (
            "bpvs",
            "trog",
            "basdig",
            "bassim",
            "basnum",
        )
        use_age = (
            True if settings.use_age_predictor is None else settings.use_age_predictor
        )
        pre_wave = 1 if settings.pre_wave is None else settings.pre_wave
        rlm_post_wave = 3 if settings.post_wave is None else settings.post_wave
        if rlm_post_wave <= pre_wave:
            raise ValueError(
                f"{spec.model_id}: post_wave must be greater than pre_wave"
            )
        port = "rlm"
        gain = True
        language = ()
        phase_mode = None
        post_time = None
        measures = ()
        design = (
            "Historical Byrne-cohort span regression of the selected post-wave "
            "outcome on its pre-wave baseline and wave-1 predictors, with group "
            "nuisance terms outside the regularised horseshoe."
        )
        population = (
            f"Available RLM children observed at waves {pre_wave} and "
            f"{rlm_post_wave} with all declared predictor measures."
        )

    return HorseshoeRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        port=port,
        outcome_symbol=outcome,
        gain=gain,
        predictors=settings.predictors,
        language_composite_symbols=language,
        covariates=settings.covariates,
        measure_symbols=measures,
        phase_mode=phase_mode,
        post_time=post_time,
        predictor_measures=predictor_measures,
        use_age_predictor=use_age,
        pre_wave=pre_wave,
        post_wave=rlm_post_wave,
        delta=settings.delta,
        tau0=settings.tau0,
        slab_scale=settings.slab_scale,
        slab_df=settings.slab_df,
        gb_reference=settings.gb_reference,
        observation_node="y_post",
        compute_loo=True,
        loo_unit="observation_row",
        focal_term=None,
        design=design,
        estimand=(
            "A mutually adjusted association ranking: posterior probability that "
            f"each standardised predictor's absolute coefficient exceeds {settings.delta:g}."
        ),
        causal_status=(
            "Associational ranking only. Shrinkage identifies which measured "
            "predictors retain signal jointly; it does not identify intervention effects."
        ),
        analysis_population=population,
        missing_data_assumption=(
            "Complete-case analysis over the model's required outcome, baseline, "
            "predictors and covariates, under ignorable missingness conditional on "
            "the observed model variables."
        ),
    )
