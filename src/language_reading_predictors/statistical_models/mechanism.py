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
from language_reading_predictors.statistical_models.fitted_payloads import (
    MechanismPayload,
)
from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism_design import (
    validate_mechanism_design,
)
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES, MEASURES
from language_reading_predictors.statistical_models.preprocessing import (
    MISSINGNESS_INDICATOR_PAIRS,
    PreparedData,
    _subset_prepared,
    load_and_prepare,
    split_covariates_by_wave,
)
from language_reading_predictors.statistical_models.settings_validation import (
    require_declared_booleans,
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
        "ability_covariate",
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
        "exposure_positive_only",
        "mech_hsgp_m",
        "mech_lengthscale_tight",
        "items_ref_quantiles",
        "decompose_between_within",
        "phase_varying_slope",
        "kappa_prior_family",
        "kappa_sigma",
        "target_accept",
    }
)


#: Time-invariant t1 baseline columns a mechanism model may declare as its
#: ``ability_covariate``. The loader broadcasts them from t1 via
#: ``baseline_covariates``; anything else is a KeyError deep inside pandas, long
#: after an output directory has been reset and the data loaded (#586).
SUPPORTED_ABILITY_COVARIATES: frozenset[str] = frozenset({"blocks", "behav"})


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
    ability_covariate: str | None = None
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
    exposure_positive_only: bool = False
    mech_hsgp_m: int | None = None
    mech_lengthscale_tight: bool = False
    items_ref_quantiles: tuple[float, float] = (0.25, 0.75)
    #: Mundlak between/within split of the exposure (#603). Default off, so no
    #: registered fit changes; linear designs only.
    decompose_between_within: bool = False
    #: Partially-pooled per-period exposure slopes (#604). Default off; linear
    #: designs only, and never alongside ``phase_specific_mechanism``.
    phase_varying_slope: bool = False
    #: Beta-Binomial concentration prior family and scale (#605). The registered
    #: default enforces a floor on overdispersion at high denominators;
    #: ``"halfnormal_inverse_sqrt"`` reaches the near-Binomial limit.
    kappa_prior_family: str = "halfnormal_concentration"
    kappa_sigma: float | None = None

    def __post_init__(self) -> None:
        require_declared_booleans(self)
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
        if self.kappa_sigma is not None:
            if isinstance(self.kappa_sigma, bool) or not isinstance(
                self.kappa_sigma, (int, float)
            ):
                raise TypeError("kappa_sigma must be a positive number or None")
            if not self.kappa_sigma > 0:
                raise ValueError("kappa_sigma must be a positive number or None")
            object.__setattr__(self, "kappa_sigma", float(self.kappa_sigma))
        # The cross-field design rules live in one place shared with the factory
        # (#637 stage 1). They had drifted in both directions: the settings alone
        # rejected linear+phase-specific, the factory alone rejected
        # ``mechanism_at_pre`` beside a covariate exposure.
        validate_mechanism_design(
            linear_mechanism=self.linear_mechanism,
            phase_specific_mechanism=self.phase_specific_mechanism,
            phase_varying_slope=self.phase_varying_slope,
            decompose_between_within=self.decompose_between_within,
            mechanism_is_covariate=self.mechanism_is_covariate,
            mechanism_at_pre=self.mechanism_at_pre,
            moderator_symbol=self.moderator_symbol,
            moderator_is_covariate=self.moderator_is_covariate,
            mech_hsgp_m=self.mech_hsgp_m,
            hsgp_lengthscale_declared=self.mech_lengthscale_tight,
            kappa_prior_family=self.kappa_prior_family,
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
            ability_covariate=extra.get("ability_covariate"),
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
            exposure_positive_only=extra.get("exposure_positive_only", False),
            mech_hsgp_m=extra.get("mech_hsgp_m"),
            mech_lengthscale_tight=extra.get("mech_lengthscale_tight", False),
            items_ref_quantiles=extra.get("items_ref_quantiles", (0.25, 0.75)),
            decompose_between_within=extra.get("decompose_between_within", False),
            phase_varying_slope=extra.get("phase_varying_slope", False),
            kappa_prior_family=extra.get(
                "kappa_prior_family", "halfnormal_concentration"
            ),
            kappa_sigma=extra.get("kappa_sigma"),
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
    ability_covariate: str | None
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
    exposure_positive_only: bool
    mech_hsgp_m: int | None
    mech_lengthscale_tight: bool
    items_ref_quantiles: tuple[float, float]
    decompose_between_within: bool
    phase_varying_slope: bool
    kappa_prior_family: str
    kappa_sigma: float | None
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
            # Cognitive ability (block design) is recorded once at t1, so a per-row
            # pull is NaN for every phase after the first. It must be broadcast from
            # t1 via ``baseline_covariates`` — the same route the gain-/level-factor,
            # block-exposure and aligned families use for this adjuster.
            "baseline_covariates": (
                (self.ability_covariate,) if self.ability_covariate else ()
            ),
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
            "decompose_between_within": self.decompose_between_within,
            "phase_varying_slope": self.phase_varying_slope,
            "kappa_prior_family": self.kappa_prior_family,
            "kappa_sigma": self.kappa_sigma,
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
        if self.kappa_prior_family == "halfnormal_inverse_sqrt":
            # ``kappa`` is a Deterministic under this family (#605), so name the
            # sampled parameter too: the diagnostics table and the
            # prior-vs-posterior overlay should show what NUTS actually explored,
            # not only its reciprocal-square transform.
            names.append("inv_sqrt_kappa")
        names += [f"gamma_{s}" for s in self.confounders if s in MEASURES]
        names += [f"gamma_{c}" for c in adjusters]
        if "A" in self.confounders and not self.use_age_gp:
            names.append("gamma_A")
        if self.use_subject_random_intercept:
            names.append("sigma_child")
        if self.linear_mechanism:
            names += self.linear_slope_vars
        if self.moderator_symbol is not None:
            names.append("gamma_mod")
            if self.include_interaction:
                names.append("gamma_int")
        return names

    @property
    def linear_slope_vars(self) -> list[str]:
        """Fitted exposure-slope parameter names on the linear branch.

        The pooled design fits one ``beta_mech``; the Mundlak split (#603) replaces
        it with ``beta_between`` plus ``beta_within``; the period-varying
        sensitivity (#604) replaces the single within/pooled slope with the shared
        mean ``mu_mech``, the between-period scale ``sigma_mech_phase`` and the
        per-period vector ``beta_mech_phase``. Deriving the list here keeps the
        diagnostics, the convergence scan and the summaries from drifting apart.
        """
        if not self.linear_mechanism:
            return []
        names: list[str] = []
        if self.decompose_between_within:
            names.append("beta_between")
        if self.phase_varying_slope:
            names += ["mu_mech", "sigma_mech_phase", "beta_mech_phase"]
        elif self.decompose_between_within:
            names.append("beta_within")
        else:
            names.append("beta_mech")
        return names

    @property
    def exposure_form_label(self) -> str:
        """Plain-language name of the fitted exposure term."""
        if not self.linear_mechanism:
            return "HSGP curve"
        if self.decompose_between_within and self.phase_varying_slope:
            return (
                "between/within split with partially-pooled per-period within-child "
                "slopes"
            )
        if self.decompose_between_within:
            return "between/within (Mundlak) split of the linear slope"
        if self.phase_varying_slope:
            return "partially-pooled per-period linear slopes"
        return "linear slope"

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        exposure = (
            "standardised raw covariate"
            if self.mechanism_is_covariate
            else "period-start bounded score"
            if self.mechanism_at_pre
            else "same-period post bounded score"
        )
        form = self.exposure_form_label
        dispersion = (
            "1/sqrt(kappa) ~ HalfNormal("
            f"{0.25 if self.kappa_sigma is None else self.kappa_sigma:g}), which "
            "reaches the near-Binomial limit"
            if self.kappa_prior_family == "halfnormal_inverse_sqrt"
            else "kappa ~ HalfNormal("
            f"{50.0 if self.kappa_sigma is None else self.kappa_sigma:g}) on the "
            "concentration"
        )
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
            f"(`{self.observation_node}`) with {dispersion}.\n\n"
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


def _reject_unsupported_mechanism_design(
    spec: ModelSpec, settings: MechanismModelSettings
) -> None:
    """Fail closed on configurations the family cannot honestly report (#586).

    None of these is reachable from a registered model today, which is exactly why
    they went unnoticed: each resolved cleanly, and would have failed — if at all —
    only after an output directory had been reset and the data loaded, or not at all,
    silently fitting something other than the declared design. They are rejected here,
    in the pure run-plan stage, before any I/O.
    """
    model_id = spec.model_id
    exposure, outcome = spec.mechanism_symbol, spec.outcome_symbol
    moderator = settings.moderator_symbol

    # 1. Overlapping focal roles. Resolution accepted outcome == exposure (feeding
    #    the outcome back as its own predictor) and exposure == moderator (whose
    #    interaction column is then the exposure squared).
    if exposure == outcome:
        raise ValueError(
            f"{model_id}: the exposure and the outcome are both {outcome!r}; a "
            "mechanism model cannot regress a measure on itself"
        )
    if moderator is not None and moderator == exposure:
        raise ValueError(
            f"{model_id}: the moderator and the exposure are both {exposure!r}; the "
            "interaction column would be the exposure squared, not a moderation"
        )
    if moderator is not None and moderator == outcome:
        raise ValueError(
            f"{model_id}: the moderator and the outcome are both {outcome!r}; the "
            "outcome cannot moderate its own predictor"
        )
    # 2. A bounded exposure declared in the adjustment set too: the factory would
    #    fit it once as the curve and again as a confounder coefficient.
    if not settings.mechanism_is_covariate and exposure in spec.adjustment:
        raise ValueError(
            f"{model_id}: exposure {exposure!r} is also declared in "
            "ModelSpec.adjustment; it would be fitted twice, as the mechanism term "
            "and as a confounder"
        )

    # 3. Phase-specific curves. The factory builds per-phase ``f_mech``, but the
    #    pipeline mixes them into one endpoint curve, one items curve and one
    #    steepest-interval summary, and power scaling asks for global hyperparameter
    #    names that a per-phase build never registers.
    if settings.phase_specific_mechanism:
        raise ValueError(
            f"{model_id}: phase_specific_mechanism is not supported. The factory can "
            "build per-phase curves, but the pipeline's curve, items and "
            "steepest-interval artefacts and its power-scaling variables are all "
            "single-curve; enable it only once those emit per-phase output."
        )

    # 4. An age GP alongside age moderation fits a nonparametric age effect and a
    #    separate linear age main effect on the same covariate, and the age-GP exact
    #    LOO refit does not freeze its boundary, so a refit reinterprets its basis.
    if settings.use_age_gp and moderator == "A":
        raise ValueError(
            f"{model_id}: use_age_gp cannot be combined with age moderation "
            "(moderator_symbol='A'). The GP already models age nonparametrically "
            "while gamma_mod adds a linear age main effect on the same covariate, "
            "and the age-GP leave-one-out refit does not freeze its boundary."
        )

    # 6. Positive-exposure restriction. It is a covariate-exposure design choice:
    #    a bounded measure's exposure is a count whose zero is a real score, not an
    #    "unexposed" state, and the fitted regressor is its logit.
    if settings.exposure_positive_only and not settings.mechanism_is_covariate:
        raise ValueError(
            f"{model_id}: exposure_positive_only applies to a continuous covariate "
            "exposure only; for a bounded measure a zero count is an observed score, "
            "not an unexposed row"
        )

    # 5. The ability covariate reached pandas before its type or name was checked.
    ability = settings.ability_covariate
    if ability is not None:
        if not isinstance(ability, str) or not ability:
            raise TypeError(
                f"{model_id}: ability_covariate must be a non-empty column name, "
                f"got {ability!r}"
            )
        if ability not in SUPPORTED_ABILITY_COVARIATES:
            raise ValueError(
                f"{model_id}: unsupported ability_covariate {ability!r}; expected one "
                f"of {', '.join(sorted(SUPPORTED_ABILITY_COVARIATES))}"
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
    _reject_unsupported_mechanism_design(spec, settings)
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

    # Required-measure coverage is checked against the **effective** outcome set,
    # whether declared or defaulted (#586). Guarding this on ``outcomes is not None``
    # meant a model leaving the default in place could name a bounded exposure,
    # baseline or moderator outside ``ITT_OUTCOMES`` (N, TR, TE, ...) and resolve
    # cleanly, only to fail in the factory after the output directory had been reset
    # and the data loaded.
    effective_outcomes = settings.outcomes or ITT_OUTCOMES
    required_measures = {
        spec.outcome_symbol,
        settings.adjust_baseline_symbol,
        *(s for s in confounders if s in MEASURES),
    }
    if not settings.mechanism_is_covariate:
        required_measures.add(spec.mechanism_symbol)
    if moderator is not None and not settings.moderator_is_covariate:
        required_measures.add(moderator)
    missing_measures = sorted(required_measures - set(effective_outcomes))
    if missing_measures:
        raise ValueError(
            f"{spec.model_id}: the loaded outcomes {effective_outcomes!r} omit "
            f"required mechanism measure(s): {', '.join(missing_measures)}"
        )

    alignment = (
        "standardised raw covariate exposure"
        if settings.mechanism_is_covariate
        else "period-start exposure"
        if settings.mechanism_at_pre
        else "same-period post exposure"
    )
    form_settings = settings
    form = (
        "HSGP curve"
        if not form_settings.linear_mechanism
        else "between/within split with partially-pooled per-period within-child "
        "slopes"
        if form_settings.decompose_between_within and form_settings.phase_varying_slope
        else "between/within (Mundlak) split of the linear slope"
        if form_settings.decompose_between_within
        else "partially-pooled per-period linear slopes"
        if form_settings.phase_varying_slope
        else "linear slope"
    )
    design = (
        "Stacked available-case period-transition Beta-Binomial regression with "
        f"phase intercepts, an autoregressive baseline, {alignment}, a {form}, "
        "declared adjustment terms and optional linear moderation. Repeated rows "
        "from a child are handled by the declared child random-intercept setting."
    )
    if settings.kappa_prior_family == "halfnormal_inverse_sqrt":
        design += (
            " The Beta-Binomial concentration is put on the dispersion scale, "
            "1/sqrt(kappa) ~ HalfNormal("
            f"{0.25 if settings.kappa_sigma is None else settings.kappa_sigma:g}), "
            "so the near-Binomial limit is reachable (#605)."
        )
    estimand = (
        "The adjusted association between the declared mechanism exposure and the "
        "post-period outcome, conditional on the fitted baseline and adjustment "
        "terms. The family's headline natural-scale contrast is the "
        f"{int(round(100 * settings.items_ref_quantiles[0]))}th-to-"
        f"{int(round(100 * settings.items_ref_quantiles[1]))}th percentile exposure "
        "difference in outcome items, standardised over the fitted rows (each row "
        "keeping its own phase, covariates, baseline and fitted child intercept); "
        "the full observed exposure range is reported beside it as a labelled "
        "secondary contrast. A linear model also reports its slope; an HSGP model "
        "also reports the adjusted curve. None is a randomised effect."
    )
    causal_status = (
        "Associational only. The exposure is not randomised, the child random "
        "intercept does not remove time-varying or latent confounding, and the "
        "mechanism term must not be described as causing the outcome."
    )
    if settings.decompose_between_within:
        causal_status += (
            " The between/within split removes stable between-child confounding "
            "from the within-child coefficient, including the stable part of latent "
            "general ability, but exposure and outcome are still measured at the "
            "same wave, so neither coefficient is temporally ordered and neither "
            "rules out time-varying confounding or reverse causation."
        )
    if settings.phase_varying_slope:
        causal_status += (
            " A difference between the per-period slopes is evidence against "
            "pooling, not evidence that the mechanism changed over time: a child's "
            "third transition differs from their first in age, treatment history "
            "and measurement position at once, and only the first period is "
            "randomised-arm-clean."
        )
    analysis_population = (
        "Available-case period transitions that first have group, age, the "
        "autoregressive baseline's period-start score and at least one loaded post "
        "score observed, with raw covariates available after their declared filling "
        "or complete-case policy. The factory then requires the focal outcome, "
        "exposure, autoregressive baseline, bounded confounders and moderator. "
        "Children may contribute multiple transitions; fitted counts are recorded "
        "after the factory keep-mask."
    )
    if settings.exposure_positive_only:
        analysis_population += (
            " Rows whose exposure is zero are then excluded, so the population is "
            "the periods that actually carried a positive exposure. This is an "
            "estimand restriction: the result is an association among exposed "
            "periods and does not rest on an unexposed comparison group."
        )
    missing_data_assumption = (
        "The loader drops rows lacking its required period-start scores or raw "
        "covariates; the factory then drops rows lacking the focal model terms. Raw "
        "covariates follow the declared adjustment and require_observed policy; "
        "filled values and missingness-indicator offsets do not by themselves make "
        "missingness ignorable."
    )

    # Complete-case only on the pre-scores the fitted model actually consumes
    # (#586 finding 4). This used to be *every* loaded outcome, but the factory's
    # linear predictor reads exactly one period-start score — the autoregressive
    # baseline — while the exposure, measure confounders and moderator are all
    # contemporaneous post measurements. Requiring their baselines dropped rows for
    # a measurement absent from the model: mech-063/163 lost four otherwise
    # eligible transitions apiece to a missing ``N_pre`` whose ``N_post`` and every
    # fitted term were observed. The exposure's own pre-score joins the requirement
    # only under ``mechanism_at_pre``, where it *is* the regressor.
    pre_required: tuple[str, ...] = (settings.adjust_baseline_symbol,)
    if settings.mechanism_at_pre:
        pre_required = tuple(dict.fromkeys(pre_required + (spec.mechanism_symbol,)))
    # Both symbols are already covered by the required-measure check above, which
    # runs first, so ``load_and_prepare``'s "pre_required must be a subset of
    # outcomes" contract cannot be violated from here.

    return MechanismRunPlan(
        model_id=spec.model_id,
        outcome_symbol=spec.outcome_symbol,
        mechanism_symbol=spec.mechanism_symbol,
        settings_source=source,
        outcomes=settings.outcomes,
        pre_required=pre_required,
        adjust_baseline_symbol=settings.adjust_baseline_symbol,
        adjust_for=settings.adjust_for,
        ability_covariate=settings.ability_covariate,
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
        exposure_positive_only=settings.exposure_positive_only,
        mech_hsgp_m=settings.mech_hsgp_m,
        mech_lengthscale_tight=settings.mech_lengthscale_tight,
        items_ref_quantiles=settings.items_ref_quantiles,
        decompose_between_within=settings.decompose_between_within,
        phase_varying_slope=settings.phase_varying_slope,
        kappa_prior_family=settings.kappa_prior_family,
        kappa_sigma=settings.kappa_sigma,
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

    if resolved.exposure_positive_only:
        # Restrict to rows that actually carry a positive exposure (#586 finding 2,
        # decided 2026-08-23). ``attend`` is an interval covariate read from the
        # transition's pre row, and the loader treats an absent session count as a
        # recorded zero — so mech-191's frame kept 28 zero-session rows while its
        # module and report both stated it held exactly the on-intervention rows.
        # Those zeros were not spread across the design: in period 1 *all* 25 fitted
        # waitlist rows sat at zero and no immediate-arm row did, so the bottom of
        # the exposure range was an arm-and-period contrast rather than a dose one.
        # This is an estimand restriction, not a data-quality drop: what remains is
        # an association among treated periods, and it no longer borrows the
        # randomised zero-dose anchor.
        symbol = resolved.mechanism_symbol
        scaler = prepared.covariate_scalers.get(symbol)
        z = np.asarray(prepared.covariates[symbol], dtype=float)
        raw = scaler.inverse(z) if scaler is not None else z
        keep = raw > 0.0
        if not keep.any():
            raise ValueError(
                f"{spec.model_id}: exposure_positive_only leaves no rows with a "
                f"positive {symbol!r}."
            )
        prepared = _subset_prepared(prepared, keep)

    # A constant covariate (e.g. an all-zero ``_missing`` indicator on the fitted
    # rows) is dropped by the loader and receives no coefficient, so it must not be
    # built into the model nor reported as adjusted-for.
    # The ability adjuster is declared separately (it loads from t1 via
    # ``baseline_covariates``) but is an ordinary standardised linear adjustment
    # coefficient in the fitted model, so it joins the effective adjustment set here.
    declared_adjust_for = resolved.adjust_for + (
        (resolved.ability_covariate,) if resolved.ability_covariate else ()
    )
    adjust_for = tuple(
        covariate for covariate in declared_adjust_for if covariate in prepared.covariates
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
) -> _factories.BuiltModel[MechanismPayload]:
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
    if settings.kappa_prior_family == "halfnormal_inverse_sqrt":
        names.append("inv_sqrt_kappa")
    names += [f"gamma_{s}" for s in plan.confounders if s in MEASURES]
    names += [f"gamma_{c}" for c in plan.adjust_for]
    if "A" in plan.confounders and not settings.use_age_gp:
        names.append("gamma_A")
    if settings.use_subject_random_intercept:
        names.append("sigma_child")
    if settings.linear_mechanism:
        if settings.decompose_between_within:
            names.append("beta_between")
        if settings.phase_varying_slope:
            names += ["mu_mech", "sigma_mech_phase", "beta_mech_phase"]
        elif settings.decompose_between_within:
            names.append("beta_within")
        else:
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
