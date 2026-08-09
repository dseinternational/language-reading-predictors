# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and construction plans for the ``mechanism`` family.

The mechanism family already had one prepared-data construction path shared by the
primary fit and exact leave-one-out refits (#438). This module now adds the earlier,
pure boundary required by #394 pillar 4: :class:`MechanismModelSettings` and
:class:`MechanismRunPlan` validate the declared design **before data loading or an
output-directory reset**, generate the human-readable model recipe, and provide the
machine-readable ``resolved_run_plan`` recorded in ``config.json``.

The existing :class:`MechanismPlan` remains the post-load plan used by the fit and
``reloo``. It carries the effective adjustment set after preprocessing has removed
constant covariates, while its ``run_plan`` retains the complete declared contract.
This tranche is behaviour-preserving: every existing mechanism model keeps its
Beta-Binomial likelihood, exposure alignment and transform, priors, rows and factory
arguments. New likelihood or exposure-transform capabilities for #433 belong in a
separate scientific change after this boundary is reviewed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from language_reading_predictors.statistical_models import factories as _factories
from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES, MEASURES
from language_reading_predictors.statistical_models.preprocessing import (
    MISSINGNESS_INDICATOR_PAIRS,
    PreparedData,
    load_and_prepare,
    split_covariates_by_wave,
)

__all__ = [
    "MechanismModelSettings",
    "MechanismRunPlan",
    "MechanismPlan",
    "declared_mechanism_settings",
    "resolve_mechanism_run_plan",
    "validate_mechanism_run_plan",
    "resolve_mechanism_plan",
    "build_mechanism_for_plan",
    "mechanism_diagnostic_vars",
]


# The complete, closed set of legacy ``spec.extra`` keys consumed by the mechanism
# family. ``target_accept`` is a sampler option resolved centrally by ``make_context``;
# it is accepted here so the family validator does not mistake a legitimate
# model-specific sampler default for a misspelled model setting.
_LEGACY_KEYS = frozenset(
    {
        "outcomes",
        "adjust_baseline_symbol",
        "adjust_for",
        "require_observed",
        "use_age_gp",
        "phase_specific_mechanism",
        "use_subject_random_intercept",
        "moderator_symbol",
        "moderator_is_covariate",
        "include_interaction",
        "linear_mechanism",
        "mechanism_is_covariate",
        "mechanism_at_pre",
        "mech_hsgp_m",
        "mech_lengthscale_tight",
        "items_ref_quantiles",
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


def _reference_quantiles(value: Any) -> tuple[float, float]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError("items_ref_quantiles must contain exactly two probabilities")
    raw = tuple(value)
    if len(raw) != 2:
        raise ValueError("items_ref_quantiles must contain exactly two probabilities")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in raw):
        raise TypeError("items_ref_quantiles must contain numeric probabilities")
    lo, hi = (float(raw[0]), float(raw[1]))
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(
            "items_ref_quantiles must satisfy 0 <= lower < upper <= 1, "
            f"got {(lo, hi)!r}"
        )
    return lo, hi


def _validate_missing_covariate_policy(
    *,
    model_id: str,
    adjust_for: tuple[str, ...],
    require_observed: tuple[str, ...],
    exposure: str | None,
    moderator: str | None,
) -> None:
    """Require an explicit policy for every reference- or mean-filled covariate."""
    adjusters = set(adjust_for)
    role_parents = {name for name in (exposure, moderator) if name is not None}
    declared = adjusters | role_parents
    required = set(require_observed)
    supported_parents = set(MISSINGNESS_INDICATOR_PAIRS)
    supported_indicators = set(MISSINGNESS_INDICATOR_PAIRS.values())

    unsupported_required = sorted(required - supported_parents)
    if unsupported_required:
        raise ValueError(
            f"{model_id}: require_observed supports only "
            f"{', '.join(sorted(supported_parents))}; got "
            f"{', '.join(unsupported_required)}"
        )
    undeclared_required = sorted(required - declared)
    if undeclared_required:
        raise ValueError(
            f"{model_id}: require_observed covariate(s) are not loaded by the "
            f"mechanism plan: {', '.join(undeclared_required)}"
        )
    unknown_indicators = sorted(
        name
        for name in adjusters
        if name.endswith("_missing") and name not in supported_indicators
    )
    if unknown_indicators:
        raise ValueError(
            f"{model_id}: unsupported missingness indicator(s): "
            f"{', '.join(unknown_indicators)}"
        )

    for parent, indicator in MISSINGNESS_INDICATOR_PAIRS.items():
        has_parent = parent in declared
        has_indicator = indicator in adjusters
        complete_case = parent in required
        if has_indicator and not has_parent:
            raise ValueError(
                f"{model_id}: orphan missingness indicator {indicator!r}; declare "
                f"its parent {parent!r}"
            )
        if parent in role_parents and not complete_case:
            role = "exposure" if parent == exposure else "moderator"
            raise ValueError(
                f"{model_id}: filled covariate {role} {parent!r} must be declared "
                "in require_observed"
            )
        if parent in adjusters and not has_indicator and not complete_case:
            raise ValueError(
                f"{model_id}: filled covariate {parent!r} requires companion "
                f"{indicator!r} or require_observed=({parent!r},)"
            )


@dataclass(frozen=True, slots=True)
class MechanismModelSettings:
    """Immutable settings declared by a mechanism-model module.

    Defaults reproduce the historical family: all bounded outcomes are loaded, the
    outcome's own period-start score is the autoregressive baseline, the exposure is
    the same-period post-score, and an HSGP curve plus child random intercept is fit.
    ``target_accept`` is deliberately absent because it is a run option, not a model
    setting.
    """

    outcomes: tuple[str, ...] | None = None
    adjust_baseline_symbol: str = "W"
    adjust_for: tuple[str, ...] = ()
    require_observed: tuple[str, ...] = ()
    use_age_gp: bool = False
    phase_specific_mechanism: bool = False
    use_subject_random_intercept: bool = True
    moderator_symbol: str | None = None
    moderator_is_covariate: bool = False
    include_interaction: bool = True
    linear_mechanism: bool = False
    mechanism_is_covariate: bool = False
    mechanism_at_pre: bool = False
    mech_hsgp_m: int | None = None
    mech_lengthscale_tight: bool = False
    items_ref_quantiles: tuple[float, float] = (0.25, 0.75)

    def __post_init__(self) -> None:
        if self.outcomes is not None:
            object.__setattr__(
                self, "outcomes", _tuple_of_strings(self.outcomes, name="outcomes")
            )
        if not isinstance(self.adjust_baseline_symbol, str) or not self.adjust_baseline_symbol:
            raise TypeError("adjust_baseline_symbol must be a non-empty string")
        object.__setattr__(
            self,
            "adjust_for",
            _tuple_of_strings(self.adjust_for, name="adjust_for"),
        )
        object.__setattr__(
            self,
            "require_observed",
            _tuple_of_strings(self.require_observed, name="require_observed"),
        )
        if self.moderator_symbol is not None and (
            not isinstance(self.moderator_symbol, str) or not self.moderator_symbol
        ):
            raise TypeError("moderator_symbol must be a non-empty string or None")
        for flag in (
            "use_age_gp",
            "phase_specific_mechanism",
            "use_subject_random_intercept",
            "moderator_is_covariate",
            "include_interaction",
            "linear_mechanism",
            "mechanism_is_covariate",
            "mechanism_at_pre",
            "mech_lengthscale_tight",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be bool")
        if self.moderator_is_covariate and self.moderator_symbol is None:
            raise ValueError("moderator_is_covariate requires moderator_symbol")
        basis = self.mech_hsgp_m
        if basis is not None:
            if isinstance(basis, bool) or not isinstance(basis, int):
                raise TypeError("mech_hsgp_m must be a positive integer or None")
            if basis < 1:
                raise ValueError("mech_hsgp_m must be a positive integer or None")
        if self.linear_mechanism and (
            self.mech_hsgp_m is not None or self.mech_lengthscale_tight
        ):
            raise ValueError(
                "linear_mechanism cannot declare HSGP basis or lengthscale settings"
            )
        if self.linear_mechanism and self.phase_specific_mechanism:
            raise ValueError(
                "linear_mechanism cannot be combined with "
                "phase_specific_mechanism; the factory's linear branch would "
                "silently ignore the phase-specific declaration"
            )
        object.__setattr__(
            self,
            "items_ref_quantiles",
            _reference_quantiles(self.items_ref_quantiles),
        )

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> MechanismModelSettings:
        """Strictly translate the former untyped ``spec.extra`` boundary."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown mechanism setting(s): {', '.join(unknown)}. "
                "Declare MechanismModelSettings so misspellings fail fast."
            )
        return cls(
            outcomes=extra.get("outcomes"),
            adjust_baseline_symbol=extra.get("adjust_baseline_symbol", "W"),
            adjust_for=extra.get("adjust_for", ()),
            require_observed=extra.get("require_observed", ()),
            use_age_gp=extra.get("use_age_gp", False),
            phase_specific_mechanism=extra.get("phase_specific_mechanism", False),
            use_subject_random_intercept=extra.get(
                "use_subject_random_intercept", True
            ),
            moderator_symbol=extra.get("moderator_symbol"),
            moderator_is_covariate=extra.get("moderator_is_covariate", False),
            include_interaction=extra.get("include_interaction", True),
            linear_mechanism=extra.get("linear_mechanism", False),
            mechanism_is_covariate=extra.get("mechanism_is_covariate", False),
            mechanism_at_pre=extra.get("mechanism_at_pre", False),
            mech_hsgp_m=extra.get("mech_hsgp_m"),
            mech_lengthscale_tight=extra.get("mech_lengthscale_tight", False),
            items_ref_quantiles=extra.get("items_ref_quantiles", (0.25, 0.75)),
        )


@dataclass(frozen=True, slots=True)
class MechanismRunPlan:
    """Concrete, validated instructions resolved before data are loaded."""

    model_id: str
    outcome_symbol: str
    mechanism_symbol: str
    settings_source: str
    outcomes: tuple[str, ...] | None
    pre_required: tuple[str, ...]
    adjust_baseline_symbol: str
    adjust_for: tuple[str, ...]
    require_observed: tuple[str, ...]
    use_age_gp: bool
    phase_specific_mechanism: bool
    use_subject_random_intercept: bool
    moderator_symbol: str | None
    moderator_is_covariate: bool
    include_interaction: bool
    linear_mechanism: bool
    mechanism_is_covariate: bool
    mechanism_at_pre: bool
    mech_hsgp_m: int | None
    mech_lengthscale_tight: bool
    items_ref_quantiles: tuple[float, float]
    confounders: tuple[str, ...]
    likelihood: str
    observation_node: str
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    @property
    def measure_confounders(self) -> tuple[str, ...]:
        return tuple(s for s in self.confounders if s in ("G", "A") or s in MEASURES)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def _load_covariates(self) -> tuple[str, ...]:
        load_covariates = self.adjust_for
        # ``load_and_prepare(require_observed=...)`` filters on each parent's
        # generated ``*_missing`` flag. Load the flag even when it is not an
        # adjustment coefficient: after filtering it is constant and the loader
        # drops it, while the parent remains in the effective factory adjustment.
        for parent in self.require_observed:
            load_covariates = tuple(
                dict.fromkeys(
                    (
                        *load_covariates,
                        parent,
                        MISSINGNESS_INDICATOR_PAIRS[parent],
                    )
                )
            )
        if self.mechanism_is_covariate:
            extra_load: tuple[str, ...] = (self.mechanism_symbol,)
            load_covariates = tuple(dict.fromkeys((*load_covariates, *extra_load)))

        moderator = self.moderator_symbol
        if self.moderator_is_covariate and moderator not in (None, "A"):
            mod_load: tuple[str, ...] = (moderator,)
            load_covariates = tuple(dict.fromkeys((*load_covariates, *mod_load)))
        return load_covariates

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the validated declaration."""
        pre_adj, post_adj = split_covariates_by_wave(self._load_covariates())
        kwargs: dict[str, Any] = {
            "phase_mode": "all",
            "covariates": pre_adj,
            "post_covariates": post_adj,
            "require_observed": self.require_observed,
            "pre_required": self.pre_required,
        }
        if self.outcomes is not None:
            kwargs["outcomes"] = self.outcomes
        return kwargs

    def factory_kwargs(
        self, *, effective_adjust_for: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """Arguments for ``build_mechanism_model`` after preprocessing."""
        adjusters = self.adjust_for if effective_adjust_for is None else effective_adjust_for
        return {
            "mechanism_symbol": self.mechanism_symbol,
            "outcome_symbol": self.outcome_symbol,
            "adjust_baseline_symbol": self.adjust_baseline_symbol,
            "confounder_symbols": self.measure_confounders,
            "use_age_gp": self.use_age_gp,
            "phase_specific_mechanism": self.phase_specific_mechanism,
            "use_subject_random_intercept": self.use_subject_random_intercept,
            "moderator_symbol": self.moderator_symbol,
            "moderator_is_covariate": self.moderator_is_covariate,
            "include_interaction": self.include_interaction,
            "linear_mechanism": self.linear_mechanism,
            "adjust_for": adjusters,
            "mechanism_is_covariate": self.mechanism_is_covariate,
            "mechanism_at_pre": self.mechanism_at_pre,
            "mech_hsgp_m": self.mech_hsgp_m,
            "mech_lengthscale_prior": (
                _priors.ell_prior_mech_tight() if self.mech_lengthscale_tight else None
            ),
        }

    def diagnostic_vars(
        self, *, effective_adjust_for: tuple[str, ...] | None = None
    ) -> list[str]:
        """Curated variables used by summaries, sensitivity and the gate."""
        adjusters = self.adjust_for if effective_adjust_for is None else effective_adjust_for
        names = ["alpha", "beta_G", "gamma_own", "kappa"]
        names += [f"gamma_{s}" for s in self.confounders if s in MEASURES]
        names += [f"gamma_{c}" for c in adjusters]
        if "A" in self.confounders and not self.use_age_gp:
            names.append("gamma_A")
        if self.use_subject_random_intercept:
            names.append("sigma_child")
        if self.linear_mechanism:
            names.append("beta_mech")
        if self.moderator_symbol is not None:
            names.append("gamma_mod")
            if self.include_interaction:
                names.append("gamma_int")
        return names

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        exposure = (
            "standardised raw covariate"
            if self.mechanism_is_covariate
            else "period-start bounded score"
            if self.mechanism_at_pre
            else "same-period post bounded score"
        )
        form = "linear slope" if self.linear_mechanism else "HSGP curve"
        outcomes = ", ".join(self.outcomes) if self.outcomes else "family default set"
        pre_required = ", ".join(self.pre_required)
        confounders = ", ".join(self.confounders) if self.confounders else "none"
        adjusters = ", ".join(self.adjust_for) if self.adjust_for else "none"
        complete = (
            ", ".join(self.require_observed) if self.require_observed else "none"
        )
        return (
            "Note: Generated from the validated mechanism run plan; template drafted "
            "by an LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Exposure: `{self.mechanism_symbol}` "
            f"as a {exposure}, fitted with a {form}. Loaded outcomes: {outcomes}. "
            f"Required period-start scores: {pre_required}. "
            f"Autoregressive baseline: `{self.adjust_baseline_symbol}`. Measure "
            f"confounders: {confounders}. Raw covariate adjusters: {adjusters}. "
            f"Complete-case covariates: {complete}. Moderator: "
            f"{self.moderator_symbol or 'none'}. Child random intercept: "
            f"{self.use_subject_random_intercept}. Likelihood: {self.likelihood} "
            f"(`{self.observation_node}`).\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate, posterior-predictive checks and PSIS-LOO reliability "
            "checks pass. The saved `config.json` contains the same resolved run plan "
            "in machine-readable form.\n"
        )


def declared_mechanism_settings(
    spec: ModelSpec,
) -> tuple[MechanismModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: mechanism settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, MechanismModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='mechanism' requires "
                f"MechanismModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        MechanismModelSettings.from_legacy_extra(
            spec.extra, model_id=spec.model_id
        ),
        "legacy_extra",
    )


def resolve_mechanism_run_plan(spec: ModelSpec) -> MechanismRunPlan:
    """Resolve and validate a mechanism spec before data or output are touched."""
    if spec.kind != "mechanism":
        raise ValueError(
            f"{spec.model_id}: expected kind 'mechanism', got {spec.kind!r}"
        )
    if not spec.outcome_symbol:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol is required for a mechanism model"
        )
    if not spec.mechanism_symbol:
        raise ValueError(
            f"{spec.model_id}: mechanism_symbol is required for a mechanism model"
        )

    settings, source = declared_mechanism_settings(spec)
    bounded_symbols = {
        spec.outcome_symbol,
        settings.adjust_baseline_symbol,
        *(settings.outcomes or ()),
    }
    if not settings.mechanism_is_covariate:
        bounded_symbols.add(spec.mechanism_symbol)
    if settings.moderator_symbol is not None and not settings.moderator_is_covariate:
        bounded_symbols.add(settings.moderator_symbol)
    unknown_bounded = sorted(bounded_symbols - set(MEASURES))
    if unknown_bounded:
        raise ValueError(
            f"{spec.model_id}: unrecognised bounded measure symbol(s): "
            f"{', '.join(unknown_bounded)}"
        )
    if "G" not in spec.adjustment:
        raise ValueError(
            f"{spec.model_id}: adjustment must declare 'G' because beta_G is "
            "always fitted"
        )
    expected_baseline = f"{settings.adjust_baseline_symbol}_pre"
    declared_baselines = tuple(
        symbol for symbol in spec.adjustment if symbol.endswith("_pre")
    )
    if declared_baselines != (expected_baseline,):
        raise ValueError(
            f"{spec.model_id}: adjustment must declare exactly the fitted "
            f"autoregressive baseline {expected_baseline!r}; got "
            f"{declared_baselines!r}"
        )
    if settings.mechanism_at_pre and settings.mechanism_is_covariate:
        raise ValueError(
            f"{spec.model_id}: mechanism_at_pre is incompatible with "
            "mechanism_is_covariate"
        )
    if settings.mechanism_is_covariate and spec.mechanism_symbol in settings.adjust_for:
        raise ValueError(
            f"{spec.model_id}: covariate exposure {spec.mechanism_symbol!r} must not "
            "also appear in adjust_for"
        )
    moderator = settings.moderator_symbol
    if (
        settings.moderator_is_covariate
        and moderator is not None
        and moderator in settings.adjust_for
    ):
        raise ValueError(
            f"{spec.model_id}: covariate moderator {moderator!r} must not also "
            "appear in adjust_for"
        )
    if settings.mechanism_is_covariate and spec.mechanism_symbol in MEASURES:
        raise ValueError(
            f"{spec.model_id}: bounded measure exposure {spec.mechanism_symbol!r} "
            "cannot be declared as a raw covariate"
        )
    if (
        settings.moderator_is_covariate
        and moderator is not None
        and moderator in MEASURES
    ):
        raise ValueError(
            f"{spec.model_id}: bounded measure moderator {moderator!r} cannot be "
            "declared as a raw covariate"
        )
    bounded_adjusters = sorted(set(settings.adjust_for) & set(MEASURES))
    if bounded_adjusters:
        raise ValueError(
            f"{spec.model_id}: bounded measure adjuster(s) must be declared in "
            f"ModelSpec.adjustment, not raw adjust_for: {', '.join(bounded_adjusters)}"
        )

    covariate_exposure = spec.mechanism_symbol if settings.mechanism_is_covariate else None
    covariate_moderator = (
        moderator
        if settings.moderator_is_covariate and moderator not in (None, "A")
        else None
    )
    _validate_missing_covariate_policy(
        model_id=spec.model_id,
        adjust_for=settings.adjust_for,
        require_observed=settings.require_observed,
        exposure=covariate_exposure,
        moderator=covariate_moderator,
    )

    confounders = tuple(
        symbol
        for symbol in spec.adjustment
        if not symbol.endswith("_pre") and symbol != moderator
    )
    if len(confounders) != len(set(confounders)):
        raise ValueError(
            f"{spec.model_id}: adjustment contains duplicate non-baseline symbols"
        )
    unknown_confounders = sorted(
        symbol
        for symbol in confounders
        if symbol not in {"G", "A"} and symbol not in MEASURES
    )
    if unknown_confounders:
        raise ValueError(
            f"{spec.model_id}: unrecognised mechanism confounder(s): "
            f"{', '.join(unknown_confounders)}"
        )

    if settings.outcomes is not None:
        required_measures = {
            spec.outcome_symbol,
            settings.adjust_baseline_symbol,
            *(s for s in confounders if s in MEASURES),
        }
        if not settings.mechanism_is_covariate:
            required_measures.add(spec.mechanism_symbol)
        if moderator is not None and not settings.moderator_is_covariate:
            required_measures.add(moderator)
        missing_measures = sorted(required_measures - set(settings.outcomes))
        if missing_measures:
            raise ValueError(
                f"{spec.model_id}: outcomes omits required mechanism measure(s): "
                f"{', '.join(missing_measures)}"
            )

    alignment = (
        "standardised raw covariate exposure"
        if settings.mechanism_is_covariate
        else "period-start exposure"
        if settings.mechanism_at_pre
        else "same-period post exposure"
    )
    form = "linear slope" if settings.linear_mechanism else "HSGP curve"
    design = (
        "Stacked available-case period-transition Beta-Binomial regression with "
        f"phase intercepts, an autoregressive baseline, {alignment}, a {form}, "
        "declared adjustment terms and optional linear moderation. Repeated rows "
        "from a child are handled by the declared child random-intercept setting."
    )
    estimand = (
        "The adjusted association between the declared mechanism exposure and the "
        "post-period outcome, conditional on the fitted baseline and adjustment "
        "terms. A linear model reports a slope; an HSGP model reports an adjusted "
        "curve and its items-scale contrast. Neither is a randomised effect."
    )
    causal_status = (
        "Associational only. The exposure is not randomised, the child random "
        "intercept does not remove time-varying or latent confounding, and the "
        "mechanism term must not be described as causing the outcome."
    )
    analysis_population = (
        "Available-case period transitions that first have group, age, every loaded "
        "outcome's period-start score and at least one loaded post score observed, "
        "with raw covariates available after their declared filling or complete-case "
        "policy. The factory then requires the focal outcome, exposure, "
        "autoregressive baseline, bounded confounders and moderator. Children may "
        "contribute multiple transitions; fitted counts are recorded after the "
        "factory keep-mask."
    )
    missing_data_assumption = (
        "The loader drops rows lacking its required period-start scores or raw "
        "covariates; the factory then drops rows lacking the focal model terms. Raw "
        "covariates follow the declared adjustment and require_observed policy; "
        "filled values and missingness-indicator offsets do not by themselves make "
        "missingness ignorable."
    )

    pre_required = settings.outcomes or ITT_OUTCOMES

    return MechanismRunPlan(
        model_id=spec.model_id,
        outcome_symbol=spec.outcome_symbol,
        mechanism_symbol=spec.mechanism_symbol,
        settings_source=source,
        outcomes=settings.outcomes,
        pre_required=pre_required,
        adjust_baseline_symbol=settings.adjust_baseline_symbol,
        adjust_for=settings.adjust_for,
        require_observed=settings.require_observed,
        use_age_gp=settings.use_age_gp,
        phase_specific_mechanism=settings.phase_specific_mechanism,
        use_subject_random_intercept=settings.use_subject_random_intercept,
        moderator_symbol=settings.moderator_symbol,
        moderator_is_covariate=settings.moderator_is_covariate,
        include_interaction=settings.include_interaction,
        linear_mechanism=settings.linear_mechanism,
        mechanism_is_covariate=settings.mechanism_is_covariate,
        mechanism_at_pre=settings.mechanism_at_pre,
        mech_hsgp_m=settings.mech_hsgp_m,
        mech_lengthscale_tight=settings.mech_lengthscale_tight,
        items_ref_quantiles=settings.items_ref_quantiles,
        confounders=confounders,
        likelihood="beta_binomial",
        observation_node="y_post",
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )


def validate_mechanism_run_plan(
    spec: ModelSpec, run_plan: MechanismRunPlan
) -> MechanismRunPlan:
    """Require an attached plan to equal the plan implied by ``spec`` exactly.

    Matching only the model ID is insufficient: a stale caller could otherwise pair
    one spec's title/metadata with another spec's outcome, rows or factory arguments.
    Pure resolution performs no I/O, so reconstructing the expected value preserves
    the pre-load validation guarantee.
    """
    expected = resolve_mechanism_run_plan(spec)
    if run_plan != expected:
        raise ValueError(
            f"{spec.model_id}: supplied mechanism run plan does not match the "
            "current model specification"
        )
    return run_plan


@dataclass(frozen=True)
class MechanismPlan:
    """Everything needed to construct a mechanism model, resolved from a spec.

    ``prepared`` is the full analysis frame the spec implies. ``factory_kwargs`` are
    the keyword arguments for :func:`factories.build_mechanism_model` *other than*
    the prepared data, so a refit can rebuild on a subset by passing a different
    ``PreparedData`` with identical keywords. ``confounders`` and ``adjust_for`` are
    retained because the diagnostics variable list is derived from them.
    """

    spec: ModelSpec
    prepared: PreparedData
    factory_kwargs: dict[str, Any]
    confounders: tuple[str, ...]
    adjust_for: tuple[str, ...]
    run_plan: MechanismRunPlan | None = None


def resolve_mechanism_plan(
    spec: ModelSpec, *, run_plan: MechanismRunPlan | None = None
) -> MechanismPlan:
    """Load the analysis frame and resolve the factory keywords for ``spec``."""
    resolved = (
        resolve_mechanism_run_plan(spec)
        if run_plan is None
        else validate_mechanism_run_plan(spec, run_plan)
    )
    prepared = load_and_prepare(**resolved.prepare_kwargs())

    # A constant covariate (e.g. an all-zero ``_missing`` indicator on the fitted
    # rows) is dropped by the loader and receives no coefficient, so it must not be
    # built into the model nor reported as adjusted-for.
    adjust_for = tuple(
        covariate for covariate in resolved.adjust_for if covariate in prepared.covariates
    )
    if (
        resolved.mechanism_is_covariate
        and resolved.mechanism_symbol not in prepared.covariates
    ):
        # The drop-constant policy is fine for an adjuster but fatal for the
        # exposure itself — there is no model without it.
        raise ValueError(
            f"{spec.model_id}: covariate exposure {resolved.mechanism_symbol!r} was "
            "dropped by the loader (constant on the fitted rows); cannot fit."
        )

    return MechanismPlan(
        spec=spec,
        prepared=prepared,
        factory_kwargs=resolved.factory_kwargs(effective_adjust_for=adjust_for),
        confounders=resolved.confounders,
        adjust_for=adjust_for,
        run_plan=resolved,
    )


def build_mechanism_for_plan(
    plan: MechanismPlan,
    prepared: PreparedData | None = None,
    *,
    frozen_design: _factories.MechanismDesign | None = None,
) -> _factories.BuiltModel:
    """Build the mechanism model for ``plan``, optionally on a row subset.

    ``prepared`` defaults to the plan's full analysis frame. A refit passes a
    :func:`factories._subset` view so the construction is identical apart from the
    rows. The factory keywords are shared by reference, which is the point: a refit
    cannot silently differ in likelihood, priors or adjustment set.
    """
    return _factories.build_mechanism_model(
        plan.prepared if prepared is None else prepared,
        **plan.factory_kwargs,
        frozen_design=frozen_design,
    )


def mechanism_diagnostic_vars(plan: MechanismPlan) -> list[str]:
    """Curated diagnostic/summary variables for a mechanism fit."""
    if plan.run_plan is not None:
        return plan.run_plan.diagnostic_vars(effective_adjust_for=plan.adjust_for)

    # Compatibility for hand-constructed plans in external/tests code predating the
    # typed boundary. Production plans always carry ``run_plan`` and never take this
    # legacy path.
    spec = plan.spec
    settings, _ = declared_mechanism_settings(spec)
    names = ["alpha", "beta_G", "gamma_own", "kappa"]
    names += [f"gamma_{s}" for s in plan.confounders if s in MEASURES]
    names += [f"gamma_{c}" for c in plan.adjust_for]
    if "A" in plan.confounders and not settings.use_age_gp:
        names.append("gamma_A")
    if settings.use_subject_random_intercept:
        names.append("sigma_child")
    if settings.linear_mechanism:
        names.append("beta_mech")
    if settings.moderator_symbol is not None:
        names.append("gamma_mod")
        if settings.include_interaction:
            names.append("gamma_int")
    return names


def holdout_is_safe(prepared: PreparedData, idx: int) -> tuple[bool, str]:
    """Whether row ``idx`` can be held out without changing the parameter vector.

    ``factories._subset`` re-indexes children densely, so dropping the *only* row
    for a child removes an element of ``u_child_raw`` and shifts every later child's
    index. The refit posterior would then be incompatible with the full model used
    to evaluate the held-out point, and — worse — the shift would silently misalign
    child effects rather than raise. Refuse that case explicitly.
    """
    child = int(prepared.child_idx[idx])
    n_rows_for_child = int((prepared.child_idx == child).sum())
    if n_rows_for_child <= 1:
        return False, (
            f"row {idx} is the only observation for child index {child}; holding it "
            "out changes the child random-effect dimension"
        )
    return True, ""


def holdout_mask(prepared: PreparedData, idx: int) -> np.ndarray:
    """Boolean keep-mask excluding row ``idx``."""
    keep = np.ones(prepared.n_obs, dtype=bool)
    keep[idx] = False
    return keep
