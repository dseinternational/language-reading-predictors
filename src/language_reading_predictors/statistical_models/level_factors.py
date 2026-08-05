# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for the level-factor family (#389 finding 6).

Mirrors the ITT / gain-factor run-plan pattern (:mod:`itt`, :mod:`gain_factors`) for
the level-factor (``kind="level_factors"``) models. A model module declares its
settings; the plan is resolved and **validated before any data are loaded or an
output directory is reset**, then a single object drives data preparation, factory
construction and the ``config.json`` / ``model_recipe.md`` audit trail. This removes
the untyped ``spec.extra`` boundary (where a misspelled key silently defaulted) and
records the resolved design, estimand, causal status, analysis population and
missing-data assumption alongside every fit -- the level family previously persisted
null ``family`` / ``design`` / ``estimand_type`` / ``causal_status`` metadata while
its report published an unqualified cause-and-effect statement (#389 finding 4).

The level design is a per-wave levels model: each wave's score is regressed on the
randomised group (entered as a per-timepoint vector when ``group_by_time``), the
ability covariate (optionally wave-varying) and, when ``group_ability``, a
group x ability effect-modification term, with a non-centred child random intercept.
The single randomised quantity is the **t2 group contrast** ``b_grp_time[1]`` (an
items- or risk-difference average marginal effect read at the t2 rows); the other
waves are post-crossover and every ability / interaction term is a
latent-ability-confounded **adjusted association**, never a causal effect. The
precise t2 estimand -- population-standardised average vs conditional-at-a-profile,
and the treatment of the currently time-invariant ``group x ability`` term -- is the
open methodological decision recorded in #389 finding 1; the prose below states the
quantity as it is presently implemented and flags that review.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.preprocessing import (
    split_confounders_by_timing,
    split_covariates_by_wave,
)

# The complete, closed set of legacy ``spec.extra`` keys the level-factor family
# understands. Anything else is a typo and must fail before a fit starts.
_LEGACY_KEYS = frozenset(
    {
        "ability_covariate",
        "adjust_for",
        "group_by_time",
        "ability_by_time",
        "group_ability",
        "likelihood",
        # Sampler knob, not a model setting: ``target_accept`` is resolved centrally by
        # ``context.make_context`` (CLI override > spec default > preset) and is never
        # read by this family's settings. Listed so a legitimate per-model declaration
        # is not rejected as a misspelling by the strict unknown-key check.
        "target_accept",
    }
)

_LIKELIHOODS = frozenset({"beta_binomial", "bernoulli_offfloor"})


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} must contain non-empty strings, got {item!r}")
    return out


@dataclass(frozen=True, slots=True)
class LevelFactorsModelSettings:
    """Immutable settings declared by a single level-factor model module.

    Defaults encode the primary per-wave levels model: no extra adjusters, group and
    ability both entered as per-timepoint vectors with a group x ability
    effect-modification term, and the Beta-Binomial working likelihood.

    ``ability_covariate`` has no default because there is no coherent one: every
    registered model sets it, and the default ``group_ability=True`` requires it, so
    the settings object is deliberately not constructible with no arguments at all
    (:func:`resolve_level_factors_run_plan` rejects that pairing).
    """

    ability_covariate: str | None = None
    adjust_for: tuple[str, ...] = ()
    group_by_time: bool = True
    ability_by_time: bool = True
    group_ability: bool = True
    likelihood: str = "beta_binomial"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "adjust_for", _tuple_of_strings(self.adjust_for, name="adjust_for")
        )
        if self.ability_covariate is not None and (
            not isinstance(self.ability_covariate, str) or not self.ability_covariate
        ):
            raise TypeError("ability_covariate must be a non-empty string or None")
        for flag in ("group_by_time", "ability_by_time", "group_ability"):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be bool")
        if self.likelihood not in _LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(_LIKELIHOODS)}, got {self.likelihood!r}"
            )

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> LevelFactorsModelSettings:
        """Strictly translate the former ``spec.extra`` dictionary boundary.

        Rejects unknown keys so a misspelling fails before data loading rather than
        silently taking a default."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown level-factor setting(s): {', '.join(unknown)}. "
                "Declare LevelFactorsModelSettings so misspellings fail fast."
            )
        # Pass raw values through so __post_init__ is the single validation/coercion
        # point; pre-coercing here (tuple(...)/bool(...)) would silently reshape
        # misshaped legacy settings ("hs" -> ('h', 's'), 1 -> True) instead of failing
        # fast against the strict checks in __post_init__. The bool flags default True.
        return cls(
            ability_covariate=extra.get("ability_covariate"),
            adjust_for=extra.get("adjust_for", ()),
            group_by_time=extra.get("group_by_time", True),
            ability_by_time=extra.get("ability_by_time", True),
            group_ability=extra.get("group_ability", True),
            likelihood=extra.get("likelihood", "beta_binomial"),
        )


@dataclass(frozen=True, slots=True)
class LevelFactorsRunPlan:
    """Concrete, validated instructions consumed by preparation and modelling."""

    model_id: str
    outcome_symbol: str
    settings_source: str
    ability_covariate: str | None
    adjust_for: tuple[str, ...]
    group_by_time: bool
    ability_by_time: bool
    group_ability: bool
    likelihood: str
    off_floor: bool
    # Covariate loading split by measurement wave (resolved from adjust_for).
    baseline_covariates: tuple[str, ...]
    pre_covariates: tuple[str, ...]
    post_covariates: tuple[str, ...]
    # Recorded audit metadata (#389 findings 4 & 6 acceptance criteria).
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    @property
    def obs_node(self) -> str:
        return "y_offfloor" if self.off_floor else "y_post"

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan.

        The level family loads the per-wave ``levels`` panel with only its own
        outcome (no upstream skill baselines); the ability covariate and any
        baseline-timed confounders load at t1, interval covariates at the pre row and
        contemporaneous confounders (e.g. hearing) at the post row (#247 timing)."""
        return {
            "phase_mode": "levels",
            "outcomes": (self.outcome_symbol,),
            "baseline_covariates": self.baseline_covariates,
            "covariates": self.pre_covariates,
            "post_covariates": self.post_covariates,
        }

    def factory_kwargs(
        self, *, effective_adjustment: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """Arguments for ``build_level_factors_model`` for this plan."""
        return {
            "outcome_symbol": self.outcome_symbol,
            "ability_covariate": self.ability_covariate,
            "adjust_for": self.adjust_for
            if effective_adjustment is None
            else effective_adjustment,
            "group_by_time": self.group_by_time,
            "ability_by_time": self.ability_by_time,
            "group_ability": self.group_ability,
            "likelihood": self.likelihood,
        }

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        adjust = ", ".join(self.adjust_for) if self.adjust_for else "none"
        return (
            "Note: Generated from the validated level-factor run plan; template "
            "drafted by an LLM-based AI tool (Claude Code/Opus 4.8).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Ability covariate: "
            f"{self.ability_covariate or 'none'}. Group entered per timepoint: "
            f"{self.group_by_time}. Ability entered per timepoint: "
            f"{self.ability_by_time}. Group x ability effect modification: "
            f"{self.group_ability}. Requested adjustment terms: {adjust}.\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate and posterior-predictive checks pass. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_level_factors_settings(
    spec: ModelSpec,
) -> tuple[LevelFactorsModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: level-factor settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, LevelFactorsModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='level_factors' requires "
                f"LevelFactorsModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        LevelFactorsModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def resolve_level_factors_run_plan(spec: ModelSpec) -> LevelFactorsRunPlan:
    """Resolve and validate a level-factor specification before any data are loaded."""
    if spec.kind != "level_factors":
        raise ValueError(
            f"{spec.model_id}: expected kind 'level_factors', got {spec.kind!r}"
        )
    if not spec.outcome_symbol:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol is required for a level-factor model"
        )

    settings, source = declared_level_factors_settings(spec)
    own = spec.outcome_symbol
    if settings.group_ability and settings.ability_covariate is None:
        # build_level_factors_model raises this too, but only after make_context has
        # reset the output directory and the loader has read the panel. Lifting it
        # here is the point of the plan: an incoherent contract fails before either
        # (cf. the did family's period_varying_dose => dose check).
        raise ValueError(
            f"{spec.model_id}: group_ability requires an ability_covariate"
        )
    off_floor = settings.likelihood == "bernoulli_offfloor"

    # Covariate loading split by measurement wave -- identical to the former inline
    # logic in fit_level_factors: the ability covariate and any baseline-timed
    # confounders load at t1 (the language-proximal SP/RW confounders are read at the
    # pre-randomisation baseline so the t2 causal contrast is not conditioned on a
    # treatment-affected descendant), interval covariates at the pre row, hearing
    # contemporaneous at the post row (#247 timing; review finding A1).
    pre_adj, post_adj = split_covariates_by_wave(settings.adjust_for)
    baseline_adj, post_adj = split_confounders_by_timing(post_adj)
    baseline_covariates = (
        (settings.ability_covariate,) if settings.ability_covariate else ()
    ) + baseline_adj

    if off_floor:
        design = (
            "Per-wave off-floor levels model: a Bernoulli likelihood for whether the "
            "child is above the outcome floor at each wave, with the randomised group "
            "entered per timepoint, the ability covariate, an optional group x ability "
            "term, and a non-centred child random intercept."
        )
        estimand = (
            "The t2 randomised group contrast on the probability of being off the "
            "floor (a risk difference read at the t2 rows). The other waves are "
            "post-crossover; ability and interaction terms are adjusted associations."
        )
    else:
        design = (
            "Per-wave levels model: each wave's score is regressed (a Beta-Binomial "
            "working likelihood) on the randomised group entered per timepoint, the "
            "ability covariate, an optional group x ability term, and a non-centred "
            "child random intercept for the repeated observations."
        )
        estimand = (
            "The t2 randomised group contrast b_grp_time[1] (an items-scale average "
            "marginal effect read at the t2 rows). The other waves are post-crossover; "
            "ability and interaction terms are adjusted associations. The precise t2 "
            "estimand -- population-standardised average vs conditional-at-a-profile, "
            "and the treatment of the currently time-invariant group x ability term -- "
            "is under methodological review (#389 finding 1)."
        )
    causal_status = (
        "Only the t2 group term is randomised (a contrast on the available-case t2 "
        "population); the other timepoints are post-crossover and every ability and "
        "group x ability term is a latent-ability-confounded adjusted association, "
        "never a causal effect."
    )
    analysis_population = (
        "Available-case children observed across the level waves (about 53-54 "
        "depending on outcome). The randomised interpretation applies to the t2 "
        "contrast on this available-case population, not automatically the complete "
        "randomised cohort."
    )
    missing_data_assumption = (
        "Available-case analysis under ignorable missingness: missing outcomes and "
        "covariates are assumed ignorable given the modelled covariates."
    )

    return LevelFactorsRunPlan(
        model_id=spec.model_id,
        outcome_symbol=own,
        settings_source=source,
        ability_covariate=settings.ability_covariate,
        adjust_for=settings.adjust_for,
        group_by_time=settings.group_by_time,
        ability_by_time=settings.ability_by_time,
        group_ability=settings.group_ability,
        likelihood=settings.likelihood,
        off_floor=off_floor,
        baseline_covariates=baseline_covariates,
        pre_covariates=pre_adj,
        post_covariates=post_adj,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
