# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for the gain-factor family (#391 finding 6).

Mirrors the ITT family's :mod:`itt` run-plan pattern for the gain-factor
(``kind="gain_factors"``) models. A model module declares its settings; the plan
is resolved and **validated before any data are loaded or an output directory is
reset**, then a single object drives data preparation, factory construction and
the ``config.json`` / ``model_recipe.md`` audit trail. This removes the untyped
``spec.extra`` boundary (where a misspelled key silently defaulted) and records the
resolved design, estimand, causal status, analysis population and missing-data
assumption alongside every fit.

The gain-factor design is a period-stacked ANCOVA: the post-score is regressed on
the child's own pre-score with a non-centred child random intercept. The headline
randomised quantity is the interaction-aware **period-1 average marginal effect**
of random assignment (``beta_trt``); every skill / ability / interaction term is a
latent-ability-confounded **adjusted association**, never a causal effect.
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

# The complete, closed set of legacy ``spec.extra`` keys the gain-factor family
# understands. Anything else is a typo and must fail before a fit starts.
_LEGACY_KEYS = frozenset(
    {
        "skill_symbols",
        "ability_covariate",
        "adjust_for",
        "interactions",
        "treated_only",
        "likelihood",
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


def _tuple_of_pairs(value: Any, *, name: str) -> tuple[tuple[str, str], ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of (a, b) pairs, got {value!r}")
    pairs: list[tuple[str, str]] = []
    for pair in value:
        p = tuple(pair)
        if len(p) != 2 or not all(isinstance(x, str) and x for x in p):
            raise TypeError(f"{name} entries must be (str, str) pairs, got {pair!r}")
        pairs.append((p[0], p[1]))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class GainFactorsModelSettings:
    """Immutable settings declared by a single gain-factor model module.

    Defaults encode the primary graded ANCOVA: no upstream skill baselines, no
    ability covariate, no interactions, a randomised (not treated-only) contrast and
    the Beta-Binomial working likelihood.
    """

    skill_symbols: tuple[str, ...] = ()
    ability_covariate: str | None = None
    adjust_for: tuple[str, ...] = ()
    interactions: tuple[tuple[str, str], ...] = ()
    treated_only: bool = False
    likelihood: str = "beta_binomial"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "skill_symbols", _tuple_of_strings(self.skill_symbols, name="skill_symbols")
        )
        object.__setattr__(
            self, "adjust_for", _tuple_of_strings(self.adjust_for, name="adjust_for")
        )
        object.__setattr__(
            self, "interactions", _tuple_of_pairs(self.interactions, name="interactions")
        )
        if self.ability_covariate is not None and (
            not isinstance(self.ability_covariate, str) or not self.ability_covariate
        ):
            raise TypeError("ability_covariate must be a non-empty string or None")
        if not isinstance(self.treated_only, bool):
            raise TypeError("treated_only must be bool")
        if self.likelihood not in _LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(_LIKELIHOODS)}, got {self.likelihood!r}"
            )

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> GainFactorsModelSettings:
        """Strictly translate the former ``spec.extra`` dictionary boundary.

        Rejects unknown keys so a misspelling fails before data loading rather than
        silently taking a default."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown gain-factor setting(s): {', '.join(unknown)}. "
                "Declare GainFactorsModelSettings so misspellings fail fast."
            )
        return cls(
            skill_symbols=tuple(extra.get("skill_symbols", ())),
            ability_covariate=extra.get("ability_covariate"),
            adjust_for=tuple(extra.get("adjust_for", ())),
            interactions=tuple(tuple(p) for p in extra.get("interactions", ())),
            treated_only=bool(extra.get("treated_only", False)),
            likelihood=extra.get("likelihood", "beta_binomial"),
        )


@dataclass(frozen=True, slots=True)
class GainFactorsRunPlan:
    """Concrete, validated instructions consumed by preparation and modelling."""

    model_id: str
    outcome_symbol: str
    settings_source: str
    skill_symbols: tuple[str, ...]
    ability_covariate: str | None
    adjust_for: tuple[str, ...]
    interactions: tuple[tuple[str, str], ...]
    treated_only: bool
    likelihood: str
    off_floor: bool
    # Covariate loading split by measurement wave (resolved from adjust_for).
    baseline_covariates: tuple[str, ...]
    pre_covariates: tuple[str, ...]
    post_covariates: tuple[str, ...]
    # Recorded audit metadata (#391 finding 6 acceptance criterion).
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
        d = asdict(self)
        # asdict turns the interaction pairs into lists; keep them as [a, b] lists
        # (JSON has no tuples) — round-trips fine and reads cleanly.
        d["interactions"] = [list(p) for p in self.interactions]
        return d

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan."""
        return {
            "phase_mode": "all",
            "outcomes": (self.outcome_symbol, *self.skill_symbols),
            "baseline_covariates": self.baseline_covariates,
            "covariates": self.pre_covariates,
            "post_covariates": self.post_covariates,
        }

    def factory_kwargs(
        self, *, effective_adjustment: tuple[str, ...] | None = None
    ) -> dict[str, Any]:
        """Arguments for ``build_gain_factors_model`` for this plan."""
        return {
            "outcome_symbol": self.outcome_symbol,
            "skill_symbols": self.skill_symbols,
            "ability_covariate": self.ability_covariate,
            "adjust_for": self.adjust_for
            if effective_adjustment is None
            else effective_adjustment,
            "interactions": self.interactions,
            "treated_only": self.treated_only,
            "likelihood": self.likelihood,
        }

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        skills = ", ".join(self.skill_symbols) if self.skill_symbols else "none"
        adjust = ", ".join(self.adjust_for) if self.adjust_for else "none"
        inter = (
            "; ".join(f"{a} x {b}" for a, b in self.interactions)
            if self.interactions
            else "none"
        )
        return (
            "Note: Generated from the validated gain-factor run plan; template "
            "drafted by an LLM-based AI tool (Claude Code/Opus 4.8).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Upstream skill baselines: {skills}. "
            f"Ability covariate: {self.ability_covariate or 'none'}. Requested "
            f"adjustment terms: {adjust}. Interactions: {inter}.\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate and posterior-predictive checks pass. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_gain_factors_settings(
    spec: ModelSpec,
) -> tuple[GainFactorsModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: gain-factor settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, GainFactorsModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='gain_factors' requires "
                f"GainFactorsModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        GainFactorsModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def resolve_gain_factors_run_plan(spec: ModelSpec) -> GainFactorsRunPlan:
    """Resolve and validate a gain-factor specification before any data are loaded."""
    if spec.kind != "gain_factors":
        raise ValueError(
            f"{spec.model_id}: expected kind 'gain_factors', got {spec.kind!r}"
        )
    if not spec.outcome_symbol:
        raise ValueError(f"{spec.model_id}: outcome_symbol is required for a gain-factor model")

    settings, source = declared_gain_factors_settings(spec)
    own = spec.outcome_symbol
    if own in settings.skill_symbols:
        raise ValueError(
            f"{spec.model_id}: the outcome {own!r} cannot also be an upstream "
            "skill baseline"
        )
    if len(settings.skill_symbols) != len(set(settings.skill_symbols)):
        raise ValueError(
            f"{spec.model_id}: skill_symbols contains duplicates: {settings.skill_symbols!r}"
        )
    off_floor = settings.likelihood == "bernoulli_offfloor"

    # Covariate loading split by measurement wave — identical to the former inline
    # logic in fit_gain_factors: the ability covariate and any baseline-timed
    # confounders load at t1, interval covariates at the pre row, contemporaneous
    # confounders (e.g. hearing) at the post row (#247 timing).
    pre_adj, post_adj = split_covariates_by_wave(settings.adjust_for)
    baseline_adj, post_adj = split_confounders_by_timing(post_adj)
    baseline_covariates = (
        (settings.ability_covariate,) if settings.ability_covariate else ()
    ) + baseline_adj

    if off_floor:
        design = (
            "Period-stacked off-floor transition model: a Bernoulli likelihood for "
            "whether the child moves above their own baseline floor, on the "
            "randomised period-1 window."
        )
        estimand = (
            "Interaction-aware period-1 average marginal effect of random assignment "
            "on the probability of moving off the floor (a risk difference), on the "
            "fitted available-case sample."
        )
    else:
        design = (
            "Period-stacked ANCOVA: the post-score is regressed on the child's own "
            "pre-score (a Beta-Binomial working likelihood) with a non-centred child "
            "random intercept for repeated observations."
        )
        estimand = (
            "Interaction-aware period-1 average marginal effect of random assignment "
            "(beta_trt), on the fitted available-case sample."
        )
    if settings.treated_only:
        estimand = (
            "Adjusted skill / ability associations on the outcome gain only; no "
            "randomised treatment contrast is estimated (treated-only fit)."
        )
        causal_status = (
            "Associational: no randomised contrast. Every coefficient is a "
            "latent-ability-confounded adjusted association."
        )
    else:
        causal_status = (
            "The treatment term is randomised (a period-1 average marginal effect); "
            "every skill, ability and interaction term is a latent-ability-confounded "
            "adjusted association, never a causal effect."
        )
    analysis_population = (
        "Available-case children observed at the period-1 randomised transition "
        "(about 53-54 children). The causal target is this available-case period-1 "
        "population, not automatically the complete randomised cohort."
    )
    missing_data_assumption = (
        "Available-case analysis under ignorable missingness: missing outcomes and "
        "covariates are assumed ignorable given the modelled covariates."
    )

    return GainFactorsRunPlan(
        model_id=spec.model_id,
        outcome_symbol=own,
        settings_source=source,
        skill_symbols=settings.skill_symbols,
        ability_covariate=settings.ability_covariate,
        adjust_for=settings.adjust_for,
        interactions=settings.interactions,
        treated_only=settings.treated_only,
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
