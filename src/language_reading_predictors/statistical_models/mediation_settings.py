# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and resolved run plans for the mediation families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.likelihood import (
    SCORE_MEAN_LINKS,
    ScoreMeanLink,
)
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES
from language_reading_predictors.statistical_models.preprocessing import (
    split_confounders_by_timing,
    split_covariates_by_wave,
)

#: The registered phoneme-blending response-link pair for this family (#619, under
#: the #608 policy). ``lrp-rli-med-087`` fits the ordinary Beta-Binomial
#: inverse-logit outcome mean; ``lrp-rli-med-387`` fits the same decomposition with
#: that mean mapped onto [1/3, 1]. Neither may be released without the other.
MEDIATION_BLENDING_PRIMARY_MODEL_ID = "lrp-rli-med-087"
MEDIATION_BLENDING_COMPANION_MODEL_ID = "lrp-rli-med-387"

__all__ = [
    "BaselineTerm",
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
        "score_mean_link",
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
    #: Phoneme-blending response link for the **outcome** leg (#619, under the #608
    #: policy). ``"logit"`` is the ordinary Beta-Binomial inverse-logit score mean;
    #: ``"three_choice_guessing_floor"`` maps it onto [1/3, 1] for the ten
    #: three-alternative forced-choice blending items, whose expected score cannot
    #: fall below chance. It governs the outcome only -- a mediator is a separate leg
    #: with its own measure. B outcomes only, graded only, and released only beside
    #: the paired opposite-link fit.
    score_mean_link: ScoreMeanLink = "logit"

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
        if self.score_mean_link not in SCORE_MEAN_LINKS:
            raise ValueError(
                f"score_mean_link must be one of {SCORE_MEAN_LINKS}, "
                f"got {self.score_mean_link!r}"
            )
        # The off-floor outcome models a binary indicator, which has no score mean to
        # map and no chance floor to respect. Checked here so an incoherent pair
        # fails at declaration, before an output directory is reset; the B-only check
        # needs ``outcome_symbol`` and lives in the resolver.
        if (
            self.score_mean_link != "logit"
            and self.outcome_kind != "beta_binomial"
        ):
            raise ValueError(
                "score_mean_link applies to the graded Beta-Binomial outcome mean; "
                f"the {self.outcome_kind!r} outcome has no score mean to map"
            )
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
            score_mean_link=extra.get("score_mean_link", "logit"),
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


def _measure_confounders(confounders: tuple[str, ...]) -> tuple[str, ...]:
    from language_reading_predictors.statistical_models.measures import MEASURES

    return tuple(symbol for symbol in confounders if symbol in MEASURES)


@dataclass(frozen=True, slots=True)
class BaselineTerm:
    """One t1 baseline regressor in a mediation leg's design matrix (#585).

    The mediation g-formula integrates the outcome law over the mediator law
    within levels of a **common** pre-exposure covariate vector ``C``. Before
    #585 each leg received only its own measure's baseline: the mediator law
    never saw the outcome baseline (which the lagged DAG makes a mediator-outcome
    confounder via ``WR_t -> LS_t1``), and the outcome law never saw the mediator
    baseline, even though ``tests/test_lagged_dag_adjustment_sets.py`` certifies
    the *union*. ``BaselineTerm`` names the terms that restore the common vector
    on each leg so the resolved plan, the factory and the counterfactual
    simulator consume one list instead of reconstructing three.

    ``form`` is a property of the measure in this fit, not of the leg: a measure
    whose likelihood this model declares off-floor enters **every** leg as the
    binary off-floor-at-baseline indicator, because a floored measure's baseline
    logit is a near-degenerate spike wherever it appears (the project's
    ``gamma_own_offfloor`` convention).
    """

    symbol: str
    coefficient: str
    form: Literal["logit", "offfloor"]
    role: Literal["own", "cross"] = "cross"


def _baseline_terms(
    *,
    prefix: str,
    common: tuple[str, ...],
    own_symbols: tuple[str, ...],
    named: tuple[str, ...],
    floored: frozenset[str],
) -> tuple[BaselineTerm, ...]:
    """Cross-leg baseline terms this leg is missing from the common vector.

    ``own_symbols`` already carry a leg-specific own-baseline coefficient and
    ``named`` the legacy ``a_<symbol>`` / ``b_<symbol>`` confounder coefficients,
    so both are skipped: only genuinely absent members of ``common`` produce a
    term. New coefficients are prefixed ``<leg>_base_`` so they cannot collide
    with the legacy names (notably the hard-coded outcome own-baseline ``b_W``,
    which the reverse-direction models MED-176/276 would otherwise clash with).
    """
    skip = set(own_symbols) | set(named)
    return tuple(
        BaselineTerm(
            symbol=symbol,
            coefficient=(
                f"{prefix}_base_{symbol}_offfloor"
                if symbol in floored
                else f"{prefix}_base_{symbol}"
            ),
            form="offfloor" if symbol in floored else "logit",
        )
        for symbol in common
        if symbol not in skip
    )


def _validate_load_set(
    *,
    model_id: str,
    required: tuple[str, ...],
    outcomes: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Fail before any I/O when a modelled measure would not be loaded (#585).

    Returns the resolved load set. A declared bounded-measure confounder that is
    absent from ``outcomes`` used to be filtered out silently after preparation
    (MED-060 lost ``E`` and ``R`` this way, and ``dropped_confounders`` compared
    only raw covariates, so nothing recorded the loss).
    """
    load = tuple(outcomes) if outcomes is not None else ITT_OUTCOMES
    missing = [symbol for symbol in required if symbol not in load]
    if missing:
        raise ValueError(
            f"{model_id}: modelled measure(s) {', '.join(missing)} are declared "
            f"but not loaded; add them to outcomes={load!r}"
        )
    return load


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
    # Phoneme-blending response link for the outcome leg and its release pairing
    # (#619). ``required_link_companion_model_id`` names the opposite-link fit that
    # must be released beside this one.
    score_mean_link: str
    required_link_companion_model_id: str | None
    link_sensitivity_required_for_release: bool
    declared_confounders: tuple[str, ...]
    effective_confounders: tuple[str, ...]
    raw_covariates: tuple[str, ...]
    #: Measures whose t1 value must enter BOTH legs (the g-formula's ``C``; #585).
    common_baselines: tuple[str, ...]
    #: Common-vector terms the mediator leg was missing before #585.
    mediator_cross_baselines: tuple[BaselineTerm, ...]
    #: Common-vector terms the outcome leg was missing before #585.
    outcome_cross_baselines: tuple[BaselineTerm, ...]
    #: Functional form of the outcome leg's OWN baseline. ``"offfloor"`` restores
    #: the binary off-floor-at-baseline indicator the off-floor outcome leg used
    #: to drop entirely (#585 finding 4), so the sample rule and the likelihood
    #: finally require the same measurements.
    outcome_own_baseline_form: Literal["logit", "offfloor"]
    #: Measures whose baseline the fitted legs actually use — the resolved
    #: complete-case rule, so an unused loaded measure cannot silently exclude a
    #: child (#585 finding 4).
    pre_required: tuple[str, ...]
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
                "pre_required": self.pre_required,
            }
        # ``pre_required`` is always passed now (#585 finding 4): the loader's
        # default requires every LOADED outcome's baseline, so a measure the legs
        # never model used to shrink the fitted sample.
        kwargs: dict[str, Any] = {
            "phase_mode": "itt",
            "covariates": self.raw_covariates,
            "pre_required": self.pre_required,
            "drop_missing_pre": self.drop_missing_pre,
        }
        if self.outcomes is not None:
            kwargs.update(outcomes=self.outcomes)
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
        outcomes = tuple(
            dict.fromkeys((self.outcome_symbol, self.mediator_symbol, *measures))
        )
        return {
            "phase_mode": "all",
            "outcomes": outcomes,
            "covariates": pre_covariates,
            "post_covariates": post_covariates,
            "baseline_covariates": baseline_covariates,
            "pre_required": self.pre_required,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        return {
            "mediator_symbol": self.mediator_symbol,
            "outcome_symbol": self.outcome_symbol,
            "confounder_symbols": self.effective_confounders,
            "mediator_kind": self.mediator_kind,
            "route_symbols": self.route_symbols,
            "outcome_kind": self.outcome_kind,
            "mediator_cross_baselines": self.mediator_cross_baselines,
            "outcome_cross_baselines": self.outcome_cross_baselines,
            "score_mean_link": self.score_mean_link,
        }

    def period_factory_kwargs(self) -> dict[str, Any]:
        return {
            "mediator_symbol": self.mediator_symbol,
            "outcome_symbol": self.outcome_symbol,
            "confounder_symbols": self.effective_confounders,
            "mediator_cross_baselines": self.mediator_cross_baselines,
            "outcome_cross_baselines": self.outcome_cross_baselines,
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
                f"Common baseline vector C (both legs): "
                f"{', '.join(self.common_baselines) or 'none'}. Mediator-leg "
                f"cross baselines: "
                f"{', '.join(t.coefficient for t in self.mediator_cross_baselines) or 'none'}. "
                f"Outcome-leg cross baselines: "
                f"{', '.join(t.coefficient for t in self.outcome_cross_baselines) or 'none'}. "
                f"Baselines required complete-case: {', '.join(self.pre_required)}. "
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
    #: Measures whose t1 value must enter EVERY leg (the g-formula's ``C``; #585).
    common_baselines: tuple[str, ...]
    #: Per-mediator-leg common-vector terms missing before #585, keyed by mediator.
    mediator_cross_baselines: dict[str, tuple[BaselineTerm, ...]]
    #: Common-vector terms the outcome leg was missing before #585.
    outcome_cross_baselines: tuple[BaselineTerm, ...]
    #: Functional form of the second mediator's OWN baseline (``"offfloor"``
    #: restores the indicator the off-floor mediator leg used to drop; #585).
    second_mediator_own_baseline_form: Literal["logit", "offfloor"]
    #: Resolved complete-case rule — the measures whose baseline the legs use.
    pre_required: tuple[str, ...]
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
        # ``pre_required`` is the resolved complete-case rule (#585 finding 4):
        # without it the loader requires every LOADED outcome's baseline, so a
        # measure no leg models could shrink the fitted sample.
        kwargs: dict[str, Any] = {
            "phase_mode": "itt",
            "covariates": self.loaded_covariates,
            "pre_required": self.pre_required,
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
            "mediator_cross_baselines": self.mediator_cross_baselines,
            "outcome_cross_baselines": self.outcome_cross_baselines,
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
                f"Common baseline vector C (every leg): "
                f"{', '.join(self.common_baselines) or 'none'}. Baselines required "
                f"complete-case: {', '.join(self.pre_required)}. "
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
    if (
        settings.score_mean_link == "three_choice_guessing_floor"
        and spec.outcome_symbol != "B"
    ):
        raise ValueError(
            f"{spec.model_id}: three_choice_guessing_floor is only valid for "
            f"phoneme blending (B) as the modelled OUTCOME, got "
            f"{spec.outcome_symbol!r}. A mediator is a separate leg with its own "
            "measure."
        )

    # The mandatory phoneme-blending link pairing (#619, under the #608 policy).
    # Scope is the model of record: an ``interventional`` relabelling declares
    # ``companion_of`` and is, by this family's own contract, a companion of the
    # natural-effects fit whose numbers it reproduces exactly -- so it is exempt on
    # the boundary the level window comparator, the gain variants and the aligned
    # dose sensitivity already draw, and its prose names the paired headline. The
    # off-floor outcome has no score mean.
    graded_outcome = settings.outcome_kind == "beta_binomial"
    model_of_record = settings.companion_of is None
    link_pair_required = (
        spec.outcome_symbol == "B" and graded_outcome and model_of_record
    )
    link_companion = (
        (
            MEDIATION_BLENDING_PRIMARY_MODEL_ID
            if settings.score_mean_link == "three_choice_guessing_floor"
            else MEDIATION_BLENDING_COMPANION_MODEL_ID
        )
        if link_pair_required
        else None
    )
    if settings.companion_of and settings.estimand != "interventional":
        raise ValueError("companion_of requires estimand='interventional'")
    if settings.mediator_kind == "gaussian_composite" and (
        settings.outcome_kind != "beta_binomial"
    ):
        # The composite factory branches before ``outcome_kind`` is consulted, so
        # the combination used to resolve, silently fit a graded outcome and then
        # ask the PPC writer for a ``y_offfloor`` node that was never built (#585).
        raise ValueError(
            "gaussian_composite mediation supports only outcome_kind="
            "'beta_binomial'; the composite factory has no off-floor outcome leg"
        )
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
    measure_confounders = _measure_confounders(confounders)
    # The g-formula's common pre-exposure vector C: the outcome baseline, the
    # mediator baseline(s) and every bounded-measure confounder, conditioned on by
    # BOTH legs (#585 finding 1). ``dict.fromkeys`` keeps declaration order and
    # de-duplicates a measure that is both a baseline and a declared confounder.
    mediator_own = (
        settings.route_symbols
        if settings.mediator_kind == "gaussian_composite"
        else (mediator_symbol,)
    )
    common = tuple(
        dict.fromkeys((spec.outcome_symbol, *mediator_own, *measure_confounders))
    )
    floored = (
        frozenset({spec.outcome_symbol})
        if settings.outcome_kind == "bernoulli_offfloor"
        else frozenset()
    )
    # A composite mediator conditions the outcome leg on the composite baseline
    # (one term matching the mediator leg's ``a_comp``), not on its route symbols
    # one by one.
    outcome_cross = (
        (BaselineTerm(symbol="M", coefficient="b_base_M", form="logit"),)
        if settings.mediator_kind == "gaussian_composite"
        else _baseline_terms(
            prefix="b",
            common=common,
            own_symbols=(spec.outcome_symbol,),
            named=measure_confounders,
            floored=floored,
        )
    )
    mediator_cross = _baseline_terms(
        prefix="a",
        common=common,
        own_symbols=mediator_own,
        named=measure_confounders,
        floored=floored,
    )
    pre_required = common
    _validate_load_set(
        model_id=spec.model_id,
        required=pre_required,
        outcomes=settings.outcomes if not settings.period_stacked else None,
    )
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
        score_mean_link=settings.score_mean_link,
        required_link_companion_model_id=link_companion,
        link_sensitivity_required_for_release=link_pair_required,
        declared_confounders=confounders,
        effective_confounders=confounders,
        raw_covariates=raw,
        common_baselines=common,
        mediator_cross_baselines=mediator_cross,
        outcome_cross_baselines=outcome_cross,
        outcome_own_baseline_form=(
            "offfloor" if settings.outcome_kind == "bernoulli_offfloor" else "logit"
        ),
        pre_required=pre_required,
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
    baselines = tuple(f"{symbol}_t1" for symbol in settings.mediators)
    confounders = tuple(
        symbol
        for symbol in spec.adjustment
        if symbol not in ("G", "A", "W_pre", *baselines)
    )
    raw = _raw_covariates(confounders)
    measure_confounders = _measure_confounders(confounders)
    calibration = settings.named_confounder_calibration
    loaded = tuple(dict.fromkeys((*raw, *((calibration.symbol,) if calibration else ()))))
    mediators = (settings.mediators[0], settings.mediators[1])
    # Common pre-exposure vector, conditioned on by every leg (#585 finding 1):
    # before this each mediator law saw only its own baseline and the outcome law
    # saw neither mediator's.
    common = tuple(
        dict.fromkeys((spec.outcome_symbol, *mediators, *measure_confounders))
    )
    floored = (
        frozenset({mediators[1]}) if settings.second_mediator_offfloor else frozenset()
    )
    mediator_cross = {
        symbol: _baseline_terms(
            prefix=f"a{symbol}",
            common=common,
            own_symbols=(symbol,),
            named=measure_confounders,
            floored=floored,
        )
        for symbol in mediators
    }
    outcome_cross = _baseline_terms(
        prefix="b",
        common=common,
        own_symbols=(spec.outcome_symbol,),
        named=measure_confounders,
        floored=floored,
    )
    pre_required = common
    _validate_load_set(
        model_id=spec.model_id, required=pre_required, outcomes=settings.outcomes
    )
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
        common_baselines=common,
        mediator_cross_baselines=mediator_cross,
        outcome_cross_baselines=outcome_cross,
        second_mediator_own_baseline_form=(
            "offfloor" if settings.second_mediator_offfloor else "logit"
        ),
        pre_required=pre_required,
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
