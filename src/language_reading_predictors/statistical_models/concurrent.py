# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for the concurrent-associations family (#394 pillar 4).

Mirrors the ITT / gain-factor / level-factor / DiD run-plan pattern (:mod:`itt`,
:mod:`gain_factors`, :mod:`level_factors`, :mod:`did`) for the per-wave concurrent
conditional-associations (``kind="concurrent"``) models. A model module declares its
settings; the plan is resolved and **validated before any data are loaded or an
output directory is reset**, then drives data preparation and the ``config.json`` /
``model_recipe.md`` audit trail. This removes the untyped ``spec.extra`` boundary
(where a misspelled key silently defaulted) and records the resolved design,
estimand, causal status, analysis population and missing-data assumption alongside
every fit.

The concurrent design fits, **at each wave separately**, a between-child
Beta-Binomial regression of the focal outcome's level on the standardised same-wave
logits of a predictor skill set (plus age and a group nuisance term), reported side
by side with matched single-skill refits that retain the trait covariates. Every coefficient is an **adjusted
association**; the family makes no causal claim, so conditioning on contemporaneous
(post-treatment) skill levels is intentional and the Table-2 fallacy applies.

Because the model is fit once per wave with a wave-specific usable-predictor subset
(and the single-skill refits vary ``include_age`` / ``include_group``), the factory is
called many times by the pipeline; this plan therefore owns the *settings*, the
single ``load_and_prepare`` call and the recorded metadata, while the per-wave
factory calls stay in ``fit_concurrent`` using the resolved plan attributes.
``predictor_slope_sigma`` defaults to ``None`` so the pipeline can fill the factory
default through ``_default_of`` reflection, keeping the anti-drift single source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.preprocessing import (
    MISSINGNESS_INDICATOR_PAIRS,
)

# The complete, closed set of legacy ``spec.extra`` keys the concurrent family
# understands. Anything else is a typo and must fail before a fit starts.
_LEGACY_KEYS = frozenset(
    {
        "predictor_symbols",
        "covariates",
        "require_observed",
        "include_age",
        "include_group",
        "predictor_slope_sigma",
        "waves",
        # Sampler knob, not a model setting: ``target_accept`` is resolved centrally by
        # ``context.make_context`` (CLI override > spec default > preset) and is never
        # read by this family's settings. Listed so a legitimate per-model declaration
        # is not rejected as a misspelling by the strict unknown-key check.
        "target_accept",
    }
)

_DEFAULT_PREDICTORS = ("L", "B", "TR", "TE", "R", "E")


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} must contain non-empty strings, got {item!r}")
    return out


def _optional_positive_ints(
    value: Any, *, name: str
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
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
    if tuple(sorted(out)) != out:
        raise ValueError(f"{name} must be in increasing order")
    return out


@dataclass(frozen=True, slots=True)
class ConcurrentModelSettings:
    """Immutable settings declared by a single concurrent-associations model module.

    Defaults encode the primary six-skill concurrent read with age and a group
    nuisance term. ``predictor_slope_sigma`` is ``None`` by default so the pipeline
    fills the ``build_concurrent_model`` default via ``_default_of`` reflection
    rather than duplicating the numeric literal here.
    """

    predictor_symbols: tuple[str, ...] = _DEFAULT_PREDICTORS
    covariates: tuple[str, ...] = ()
    require_observed: tuple[str, ...] = ()
    include_age: bool = True
    include_group: bool = True
    predictor_slope_sigma: float | None = None
    waves: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predictor_symbols",
            _tuple_of_strings(self.predictor_symbols, name="predictor_symbols"),
        )
        object.__setattr__(
            self, "covariates", _tuple_of_strings(self.covariates, name="covariates")
        )
        object.__setattr__(
            self,
            "require_observed",
            _tuple_of_strings(self.require_observed, name="require_observed"),
        )
        for flag in ("include_age", "include_group"):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be bool")
        sigma = self.predictor_slope_sigma
        if sigma is not None:
            # bool is an int subclass but is never a valid slope scale.
            if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
                raise TypeError("predictor_slope_sigma must be a number or None")
            if sigma <= 0:
                raise ValueError("predictor_slope_sigma must be positive")
            object.__setattr__(self, "predictor_slope_sigma", float(sigma))
        object.__setattr__(
            self, "waves", _optional_positive_ints(self.waves, name="waves")
        )

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> ConcurrentModelSettings:
        """Strictly translate the former ``spec.extra`` dictionary boundary.

        Rejects unknown keys so a misspelling fails before data loading rather than
        silently taking a default."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown concurrent setting(s): {', '.join(unknown)}. "
                "Declare ConcurrentModelSettings so misspellings fail fast."
            )
        # Pass raw values through so __post_init__ is the single validation/coercion
        # point. ``predictor_slope_sigma`` absent -> None (pipeline uses the factory
        # default via _default_of), matching the former .get(key, _default_of(...)).
        return cls(
            predictor_symbols=extra.get("predictor_symbols", _DEFAULT_PREDICTORS),
            covariates=extra.get("covariates", ()),
            require_observed=extra.get("require_observed", ()),
            include_age=extra.get("include_age", True),
            include_group=extra.get("include_group", True),
            predictor_slope_sigma=extra.get("predictor_slope_sigma"),
            waves=extra.get("waves"),
        )


@dataclass(frozen=True, slots=True)
class ConcurrentRunPlan:
    """Concrete, validated instructions consumed by preparation and modelling."""

    model_id: str
    study_id: Literal["rli", "rlm"]
    port: Literal["rli", "rlm"]
    outcome_symbol: str
    settings_source: str
    predictor_symbols: tuple[str, ...]
    covariates: tuple[str, ...]
    require_observed: tuple[str, ...]
    include_age: bool
    include_group: bool
    # ``None`` -> the pipeline fills the build_concurrent_model default via _default_of.
    predictor_slope_sigma: float | None
    waves: tuple[int, ...]
    # Recorded audit metadata (#394 pillar 4).
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    @property
    def measure_outcomes(self) -> tuple[str, ...]:
        """The outcome plus its predictor skills, de-duplicated and order-preserving."""
        return tuple(dict.fromkeys((self.outcome_symbol, *self.predictor_symbols)))

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan.

        The concurrent family loads the per-wave ``levels`` panel with the outcome and
        its predictor skills as bounded-count measures; the trait covariates are
        t1-measured and enter as baseline covariates broadcast across the waves."""
        if self.port != "rli":
            raise ValueError("prepare_kwargs applies only to the RLI concurrent port")
        filter_only_indicators = tuple(
            MISSINGNESS_INDICATOR_PAIRS[name]
            for name in self.require_observed
        )
        covariates_to_load = tuple(
            dict.fromkeys((*self.covariates, *filter_only_indicators))
        )
        return {
            "phase_mode": "levels",
            "outcomes": self.measure_outcomes,
            "baseline_covariates": covariates_to_load,
            "require_observed": self.require_observed,
        }

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        preds = ", ".join(self.predictor_symbols) if self.predictor_symbols else "none"
        covs = ", ".join(self.covariates) if self.covariates else "none"
        complete = (
            ", ".join(self.require_observed)
            if self.require_observed
            else "none"
        )
        sigma = (
            "build default"
            if self.predictor_slope_sigma is None
            else f"{self.predictor_slope_sigma:g}"
        )
        return (
            "Note: Generated from the validated concurrent-associations run plan; "
            "template drafted by an LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Predictor skills: {preds}. Trait "
            f"covariates: {covs}. Age term: {self.include_age}. Group nuisance term: "
            f"{self.include_group}. Complete-case covariates: {complete}. "
            f"Predictor-slope prior sigma: {sigma}. Missingness indicators are "
            "nuisance subgroup offsets, not skill effects. Fitted waves: "
            f"{', '.join(map(str, self.waves))}.\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate and posterior-predictive checks pass. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_concurrent_settings(
    spec: ModelSpec,
) -> tuple[ConcurrentModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: concurrent settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, ConcurrentModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='concurrent' requires "
                f"ConcurrentModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        ConcurrentModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def _validate_missing_covariate_policy(
    *,
    model_id: str,
    covariates: tuple[str, ...],
    require_observed: tuple[str, ...],
) -> None:
    """Require an explicit missing-data policy for every filled trait covariate."""
    covariate_set = set(covariates)
    required_set = set(require_observed)
    supported = set(MISSINGNESS_INDICATOR_PAIRS)
    supported_indicators = set(MISSINGNESS_INDICATOR_PAIRS.values())
    unsupported = sorted(required_set - supported)
    if unsupported:
        raise ValueError(
            f"{model_id}: require_observed supports only "
            f"{', '.join(sorted(supported))}; got {', '.join(unsupported)}"
        )
    undeclared = sorted(required_set - covariate_set)
    if undeclared:
        raise ValueError(
            f"{model_id}: require_observed covariate(s) must also be declared in "
            f"covariates: {', '.join(undeclared)}"
        )
    unknown_indicators = sorted(
        name
        for name in covariate_set
        if name.endswith("_missing") and name not in supported_indicators
    )
    if unknown_indicators:
        raise ValueError(
            f"{model_id}: unsupported missingness indicator(s): "
            f"{', '.join(unknown_indicators)}"
        )

    for parent, indicator in MISSINGNESS_INDICATOR_PAIRS.items():
        has_parent = parent in covariate_set
        has_indicator = indicator in covariate_set
        complete_case = parent in required_set
        if has_indicator and not has_parent:
            raise ValueError(
                f"{model_id}: orphan missingness indicator {indicator!r}; declare "
                f"its parent {parent!r}"
            )
        if not has_parent:
            continue
        if has_indicator and complete_case:
            raise ValueError(
                f"{model_id}: {parent!r} cannot use both {indicator!r} and "
                "require_observed"
            )
        if not has_indicator and not complete_case:
            raise ValueError(
                f"{model_id}: filled covariate {parent!r} requires companion "
                f"{indicator!r} or require_observed=({parent!r},)"
            )


def resolve_concurrent_run_plan(spec: ModelSpec) -> ConcurrentRunPlan:
    """Resolve and validate a concurrent-associations spec before any data are loaded."""
    if spec.kind != "concurrent":
        raise ValueError(
            f"{spec.model_id}: expected kind 'concurrent', got {spec.kind!r}"
        )
    if not spec.outcome_symbol:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol is required for a concurrent model"
        )

    settings, source = declared_concurrent_settings(spec)
    if spec.study_id not in {"rli", "rlm"}:
        raise ValueError(
            f"{spec.model_id}: concurrent supports study_id 'rli' or 'rlm', "
            f"got {spec.study_id!r}"
        )
    port: Literal["rli", "rlm"] = spec.study_id
    waves: tuple[int, ...]
    if port == "rli":
        _validate_missing_covariate_policy(
            model_id=spec.model_id,
            covariates=settings.covariates,
            require_observed=settings.require_observed,
        )
        if settings.waves is not None:
            raise ValueError(
                f"{spec.model_id}: waves is an RLM-only concurrent setting"
            )
        # Validate the measure symbols against the registry *before* make_context
        # can reset an output directory — the RLM branch already did; the RLI
        # branch previously failed only inside the loader, after the reset
        # (2026-08-21 concurrent review, finding 5).
        from language_reading_predictors.statistical_models.measures import (
            MEASURES as _rli_measures,
        )

        requested_rli = (spec.outcome_symbol, *settings.predictor_symbols)
        unknown_rli = sorted(set(requested_rli) - set(_rli_measures))
        if unknown_rli:
            raise ValueError(
                f"{spec.model_id}: unknown RLI measure(s): {', '.join(unknown_rli)}"
            )
        waves = (1, 2, 3, 4)
    else:
        if settings.covariates or settings.require_observed:
            raise ValueError(
                f"{spec.model_id}: the RLM concurrent port does not support RLI "
                "trait covariates or require_observed"
            )
        waves = settings.waves or (1, 2, 3, 4)
    own = spec.outcome_symbol

    if not settings.predictor_symbols:
        raise ValueError(f"{spec.model_id}: predictor_symbols cannot be empty")
    if len(settings.predictor_symbols) != len(set(settings.predictor_symbols)):
        raise ValueError(f"{spec.model_id}: predictor_symbols contains duplicates")

    if port == "rlm":
        from language_reading_predictors.statistical_models.datasets import (
            resolve_dataset,
        )

        _dataset, measures = resolve_dataset("rlm")
        if own in settings.predictor_symbols:
            raise ValueError(
                f"{spec.model_id}: the outcome cannot also be a predictor"
            )
        requested = (own, *settings.predictor_symbols)
        unknown = sorted(set(requested) - set(measures))
        if unknown:
            raise ValueError(
                f"{spec.model_id}: unknown RLM measure(s): {', '.join(unknown)}"
            )
        provisional = [
            sym
            for sym in requested
            if not measures[sym].n_trials_confirmed
            or not measures[sym].instrument_identity_confirmed
        ]
        if provisional:
            raise ValueError(
                f"{spec.model_id}: RLM concurrent models require confirmed "
                "denominators and instrument identities; unresolved: "
                f"{', '.join(provisional)}"
            )
        unavailable = {
            symbol: tuple(wave for wave in waves if wave not in measures[symbol].available_waves)
            for symbol in requested
            if measures[symbol].available_waves
            and any(wave not in measures[symbol].available_waves for wave in waves)
        }
        if unavailable:
            detail = "; ".join(
                f"{symbol} is not available at requested wave(s) "
                f"{', '.join(map(str, missing_waves))}"
                for symbol, missing_waves in unavailable.items()
            )
            raise ValueError(f"{spec.model_id}: {detail}")

    cohort = "RLI intervention cohort" if port == "rli" else "Byrne observational cohort"
    design = (
        f"Per-wave concurrent conditional associations in the {cohort}: at each "
        "timepoint a between-child Beta-Binomial regression of the outcome level on "
        "the standardised same-wave logits of the predictor skills, plus optional "
        f"age and a group nuisance term. Waves {', '.join(map(str, waves))} are "
        "reported side by side, each with a matched single-skill refit retaining "
        "the same declared covariates."
    )
    estimand = (
        "Adjusted per-wave conditional associations (per +1 SD of each predictor's "
        "same-wave logit), reported alongside a reduced-skill comparator that keeps "
        "the same other declared covariates but omits age, group and the other skills. "
        "No causal quantity is estimated: conditioning on contemporaneous skill "
        "levels is intentional and the Table-2 fallacy applies."
    )
    causal_status = (
        "Associational only. The family makes no causal claim; every coefficient is a "
        "potentially residual-confounded adjusted association at a wave, never a cause."
    )
    analysis_population = (
        "Available-case children with the focal outcome observed at each requested "
        "wave. The reported associations average over the fitted rows at each wave -- "
        "a descriptive averaging population."
    )
    if port == "rli":
        missing_data_assumption = (
            "Rows missing the focal outcome are dropped (an outcome cannot be "
            "imputed); missing skill-predictor values are mean-imputed for this "
            "descriptive read. Filled trait covariates must either carry their paired "
            "missingness indicator as a nuisance subgroup offset or be declared "
            "complete-case through require_observed. Neither policy guarantees "
            "unbiased associations."
        )
    else:
        missing_data_assumption = (
            "Rows missing the focal outcome are dropped separately at each wave; "
            "missing skill-predictor values are mean-imputed to zero after within-wave "
            "standardisation. Observed and imputed counts are reported. This pragmatic "
            "available-case policy does not guarantee unbiased associations."
        )

    return ConcurrentRunPlan(
        model_id=spec.model_id,
        study_id=port,
        port=port,
        outcome_symbol=own,
        settings_source=source,
        predictor_symbols=settings.predictor_symbols,
        covariates=settings.covariates,
        require_observed=settings.require_observed,
        include_age=settings.include_age,
        include_group=settings.include_group,
        predictor_slope_sigma=settings.predictor_slope_sigma,
        waves=waves,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
