# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for the latent growth-curve family (#394 pillar 4).

Mirrors the ITT / gain-factor / level-factor / DiD / concurrent / aligned run-plan
pattern for the joint multivariate latent growth-curve (``kind="growth"``) models. A
model module declares its settings; the plan is resolved and **validated before any
data are loaded or an output directory is reset**, then drives data preparation,
factory construction and the ``config.json`` / ``model_recipe.md`` audit trail. This
removes the untyped ``spec.extra`` boundary (where a misspelled key silently
defaulted) and records the resolved design, estimand, causal status, analysis
population and missing-data assumption alongside every fit.

The growth design characterises each measure's within-child trajectory across the
four RLI waves (linear in standardised age) and asks whether a **baseline** covariate
(non-verbal ability) predicts trajectory shape: ``gamma`` on the growth *rate* (the
headline Q5 estimand) and ``delta`` on the baseline *level*. It is a multi-outcome
family (no single ``outcome_symbol``); every non-randomised term is an **adjusted,
latent-general-ability-confounded association**, never causal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec

# The complete, closed set of legacy ``spec.extra`` keys the growth family
# understands. Anything else is a typo and must fail before a fit starts.
_LEGACY_KEYS = frozenset(
    {
        "outcomes",
        "baseline_covariate",
        "use_shared_factor",
        "age_ability_interaction",
        # Sampler knob, not a model setting: ``target_accept`` is resolved centrally by
        # ``context.make_context`` (CLI override > spec default > preset) and is never
        # read by this family's settings. Listed so a legitimate per-model declaration
        # is not rejected as a misspelling by the strict unknown-key check.
        "target_accept",
    }
)

_DEFAULT_OUTCOMES = ("R", "E", "T", "W", "L")


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} must contain non-empty strings, got {item!r}")
    return out


@dataclass(frozen=True, slots=True)
class GrowthModelSettings:
    """Immutable settings declared by a single latent growth-curve model module.

    Defaults encode the primary five-measure growth-curve read on non-verbal ability
    (``blocks``), without the shared growth-tempo factor or the age x ability term.
    """

    outcomes: tuple[str, ...] = _DEFAULT_OUTCOMES
    baseline_covariate: str = "blocks"
    use_shared_factor: bool = False
    age_ability_interaction: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcomes", _tuple_of_strings(self.outcomes, name="outcomes")
        )
        if not self.outcomes:
            raise ValueError("outcomes must list at least one measure")
        if not isinstance(self.baseline_covariate, str) or not self.baseline_covariate:
            raise TypeError("baseline_covariate must be a non-empty string")
        for flag in ("use_shared_factor", "age_ability_interaction"):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be bool")

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> GrowthModelSettings:
        """Strictly translate the former ``spec.extra`` dictionary boundary.

        Rejects unknown keys so a misspelling fails before data loading rather than
        silently taking a default."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown growth setting(s): {', '.join(unknown)}. "
                "Declare GrowthModelSettings so misspellings fail fast."
            )
        # Pass raw values through so __post_init__ is the single validation/coercion
        # point; pre-coercing here would silently reshape misshaped legacy settings.
        return cls(
            outcomes=extra.get("outcomes", _DEFAULT_OUTCOMES),
            baseline_covariate=extra.get("baseline_covariate", "blocks"),
            use_shared_factor=extra.get("use_shared_factor", False),
            age_ability_interaction=extra.get("age_ability_interaction", False),
        )


@dataclass(frozen=True, slots=True)
class GrowthRunPlan:
    """Concrete, validated instructions consumed by preparation and modelling."""

    model_id: str
    settings_source: str
    outcomes: tuple[str, ...]
    baseline_covariate: str
    use_shared_factor: bool
    age_ability_interaction: bool
    # Recorded audit metadata (#394 pillar 4).
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_wave_panel`` from the resolved plan.

        The growth family loads the wave panel of its measures with the single
        t1-only baseline covariate (non-verbal ability)."""
        return {
            "outcomes": self.outcomes,
            "baseline_covariates": (self.baseline_covariate,),
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_growth_model`` for this plan."""
        return {
            "baseline_covariate": self.baseline_covariate,
            "use_shared_factor": self.use_shared_factor,
            "age_ability_interaction": self.age_ability_interaction,
        }

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        outcomes = ", ".join(self.outcomes)
        return (
            "Note: Generated from the validated latent growth-curve run plan; "
            "template drafted by an LLM-based AI tool (Claude Code/Opus 4.8).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Measures: {outcomes}. Baseline covariate: `{self.baseline_covariate}`. "
            f"Shared growth-tempo factor: {self.use_shared_factor}. Age x ability "
            f"interaction: {self.age_ability_interaction}.\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate and posterior-predictive checks pass. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_growth_settings(spec: ModelSpec) -> tuple[GrowthModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: growth settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, GrowthModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='growth' requires GrowthModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return (
        GrowthModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def resolve_growth_run_plan(spec: ModelSpec) -> GrowthRunPlan:
    """Resolve and validate a latent growth-curve spec before any data are loaded."""
    if spec.kind != "growth":
        raise ValueError(f"{spec.model_id}: expected kind 'growth', got {spec.kind!r}")

    settings, source = declared_growth_settings(spec)

    design = (
        "Joint multivariate latent growth-curve model on the logit scale: each "
        "measure's within-child trajectory across the four waves (linear in "
        "standardised age), with a per-measure child-level random intercept and slope; "
        "optionally a rank-1 shared growth-tempo factor coupling the slopes."
    )
    estimand = (
        "gamma (baseline non-verbal ability -> growth RATE) is the headline Q5 "
        "association; delta is the association with baseline LEVEL. Both are adjusted, "
        "latent-general-ability-confounded associations, never causal (block design is "
        "an off-DAG ability proxy)."
    )
    causal_status = (
        "Associational only: every non-randomised term is a latent-general-ability-"
        "confounded adjusted association under the locked DAG, never a causal effect."
    )
    analysis_population = (
        "Available-case children with wave-panel trajectories (about 54). The "
        "associations describe this observed cohort, not a randomised contrast."
    )
    missing_data_assumption = (
        "Available-case analysis under ignorable missingness: missing wave scores are "
        "assumed ignorable given the modelled trajectory and baseline covariate."
    )

    return GrowthRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        outcomes=settings.outcomes,
        baseline_covariate=settings.baseline_covariate,
        use_shared_factor=settings.use_shared_factor,
        age_ability_interaction=settings.age_ability_interaction,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
