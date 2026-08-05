# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for the difference-in-differences family (#394 pillar 4).

Mirrors the ITT / gain-factor / level-factor run-plan pattern (:mod:`itt`,
:mod:`gain_factors`, :mod:`level_factors`) for the waitlist-crossover
difference-in-differences (``kind="did"``) models. A model module declares its
settings; the plan is resolved and **validated before any data are loaded or an
output directory is reset**, then a single object drives data preparation, factory
construction and the ``config.json`` / ``model_recipe.md`` audit trail. This removes
the untyped ``spec.extra`` boundary (where a misspelled key silently defaulted) and
records the resolved design, estimand, causal status, analysis population and
missing-data assumption alongside every fit.

Two designs share the family. **Binary** models fit the t1-t3 *levels* frame and
estimate the arm gap at each wave separately: ``tau_t2`` is the clean randomised t2
difference-in-differences contrast, ``arm_gap_t3`` the post-crossover
40-vs-20-week association, and ``delta_crossover = tau_t2 - arm_gap_t3`` the
waitlist catch-up. **Dose** variants keep the P1/P2 *transition* frame because
sessions are interval exposures, carry an explicit treatment-presence term with the
``attend`` session covariate, and their session coefficient is observational, never
randomised.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec

# The complete, closed set of legacy ``spec.extra`` keys the DiD family understands.
# Anything else is a typo and must fail before a fit starts.
_LEGACY_KEYS = frozenset(
    {
        "dose",
        "period_varying_dose",
        "likelihood",
        "outcomes",
        "waves",
        "periods",
        "use_child_re",
        "use_age",
        "use_varying_delta",
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


def _tuple_of_ints(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of integers, got {value!r}")
    out = tuple(value)
    for item in out:
        # bool is an int subclass but is never a valid wave/period index.
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must contain integers, got {item!r}")
    return out


@dataclass(frozen=True, slots=True)
class DiDModelSettings:
    """Immutable settings declared by a single difference-in-differences model module.

    Defaults encode the primary binary DiD: the t1-t3 levels frame with a child
    random intercept and age, no dose term, and the Beta-Binomial working likelihood.
    """

    dose: bool = False
    period_varying_dose: bool = False
    likelihood: str = "beta_binomial"
    outcomes: tuple[str, ...] = ()
    waves: tuple[int, ...] = (0, 1, 2)
    periods: tuple[int, ...] = (0, 1)
    use_child_re: bool = True
    use_age: bool = True
    use_varying_delta: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcomes", _tuple_of_strings(self.outcomes, name="outcomes")
        )
        object.__setattr__(self, "waves", _tuple_of_ints(self.waves, name="waves"))
        object.__setattr__(self, "periods", _tuple_of_ints(self.periods, name="periods"))
        for flag in ("dose", "period_varying_dose", "use_child_re", "use_age", "use_varying_delta"):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be bool")
        if self.period_varying_dose and not self.dose:
            raise ValueError("period_varying_dose requires dose=True")
        if self.likelihood not in _LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(_LIKELIHOODS)}, got {self.likelihood!r}"
            )
        # The remaining cross-field constraints build_did_model enforces (#455). They
        # depend on nothing but these settings, so the factory would only reject them
        # after make_context had reset an output directory and the loader had read the
        # panel. Checked here, incoherent settings fail at settings construction time
        # (often at model-module import time, otherwise at resolve time). The factory keeps its own copies as belt-and-braces for direct callers.
        if self.dose and self.likelihood == "bernoulli_offfloor":
            raise ValueError(
                "bernoulli_offfloor is the binary prevalence estimand; use dose=False"
            )
        if self.use_varying_delta and self.dose:
            raise ValueError("use_varying_delta is unavailable for dose models")
        if self.use_varying_delta and not self.use_child_re:
            raise ValueError("use_varying_delta=True requires use_child_re=True")
        if self.dose and self.periods != (0, 1):
            raise ValueError(
                f"DiD dose variants require periods=(0, 1); got {self.periods}."
            )

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> DiDModelSettings:
        """Strictly translate the former ``spec.extra`` dictionary boundary.

        Rejects unknown keys so a misspelling fails before data loading rather than
        silently taking a default."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown DiD setting(s): {', '.join(unknown)}. "
                "Declare DiDModelSettings so misspellings fail fast."
            )
        # Pass raw values through so __post_init__ is the single validation/coercion
        # point; pre-coercing here (tuple(...)/bool(...)) would silently reshape
        # misshaped legacy settings instead of failing fast against the strict checks.
        return cls(
            dose=extra.get("dose", False),
            period_varying_dose=extra.get("period_varying_dose", False),
            likelihood=extra.get("likelihood", "beta_binomial"),
            outcomes=extra.get("outcomes", ()),
            waves=extra.get("waves", (0, 1, 2)),
            periods=extra.get("periods", (0, 1)),
            use_child_re=extra.get("use_child_re", True),
            use_age=extra.get("use_age", True),
            use_varying_delta=extra.get("use_varying_delta", False),
        )


@dataclass(frozen=True, slots=True)
class DiDRunPlan:
    """Concrete, validated instructions consumed by preparation and modelling."""

    model_id: str
    outcome_symbol: str
    settings_source: str
    dose: bool
    period_varying: bool
    likelihood: str
    off_floor: bool
    outcomes: tuple[str, ...]
    waves: tuple[int, ...]
    periods: tuple[int, ...]
    use_child_re: bool
    use_age: bool
    use_varying_delta: bool
    # Recorded audit metadata (#394 pillar 4).
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    @property
    def obs_node(self) -> str:
        return "y_offfloor" if self.off_floor else "y_post"

    @property
    def effect_term(self) -> str:
        """The focal coefficient: the randomised t2 DiD contrast, or the dose term."""
        if self.period_varying:
            return "mu_dose"
        return "beta_dose" if self.dose else "tau_t2"

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        d = asdict(self)
        # JSON has no tuples; keep the integer wave/period vectors as lists.
        d["waves"] = list(self.waves)
        d["periods"] = list(self.periods)
        return d

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan.

        Binary models load the t1-t3 ``levels`` panel; dose models keep the P1/P2
        transition frame (``phase_mode="all"``) and add the ``attend`` session
        covariate. Both keep ``require_any_post=False`` (a child observed at only one
        wave still contributes to the arm-by-wave contrasts)."""
        if self.dose:
            return {
                "phase_mode": "all",
                "outcomes": self.outcomes,
                "covariates": ("attend",),
                "pre_required": (),
                "require_any_post": False,
            }
        return {
            "phase_mode": "levels",
            "outcomes": self.outcomes,
            "require_any_post": False,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_did_model`` for this plan.

        ``period_varying_dose`` receives the *resolved* ``period_varying`` (dose AND
        the flag), matching the former inline behaviour."""
        return {
            "outcome_symbol": self.outcome_symbol,
            "waves": self.waves,
            "periods": self.periods,
            "use_child_re": self.use_child_re,
            "use_age": self.use_age,
            "dose": self.dose,
            "period_varying_dose": self.period_varying,
            "use_varying_delta": self.use_varying_delta,
            "likelihood": self.likelihood,
        }

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        waves = ", ".join(str(w) for w in self.waves)
        return (
            "Note: Generated from the validated difference-in-differences run plan; "
            "template drafted by an LLM-based AI tool (Claude Code/Opus 4.8).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Waves: {waves}. Dose term: "
            f"{self.dose} (period-varying: {self.period_varying}). Child random "
            f"intercept: {self.use_child_re}. Age adjustment: {self.use_age}.\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate and posterior-predictive checks pass. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_did_settings(spec: ModelSpec) -> tuple[DiDModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: DiD settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, DiDModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='did' requires DiDModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return (
        DiDModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def resolve_did_run_plan(spec: ModelSpec) -> DiDRunPlan:
    """Resolve and validate a difference-in-differences spec before any data are loaded."""
    if spec.kind != "did":
        raise ValueError(f"{spec.model_id}: expected kind 'did', got {spec.kind!r}")
    if not spec.outcome_symbol:
        raise ValueError(f"{spec.model_id}: outcome_symbol is required for a DiD model")

    settings, source = declared_did_settings(spec)
    own = spec.outcome_symbol
    off_floor = settings.likelihood == "bernoulli_offfloor"
    period_varying = settings.dose and settings.period_varying_dose
    # The outcome loads as its own outcome when a spec does not list an explicit set.
    outcomes = settings.outcomes if settings.outcomes else (own,)

    if settings.dose:
        design = (
            "Waitlist-crossover dose model on the P1/P2 transition rows (sessions are "
            "interval exposures): an explicit treatment-presence term with the "
            "randomised arm, the shared t1 baseline and age, and sessions entered "
            "centred and scaled among treated rows."
        )
        estimand = (
            "The session-dose association with the outcome among treated rows "
            "(period-varying if requested). Observational, not a randomised contrast."
        )
        causal_status = (
            "Associational: the session-dose coefficient is observational (sessions "
            "are not randomised); the randomised arm enters only as an adjuster."
        )
    else:
        design = (
            "Waitlist-crossover difference-in-differences on the t1-t3 levels frame: "
            "the arm gap is estimated separately at each wave (arm_gap_t1 the "
            "pre-treatment balance quantity, tau_t2 the randomised t2 contrast, "
            "arm_gap_t3 the post-crossover 40-vs-20-week association, "
            "delta_crossover = tau_t2 - arm_gap_t3 the waitlist catch-up), with a "
            "non-centred child random intercept."
        )
        estimand = (
            "The t2 arm gap tau_t2 is the clean randomised difference-in-differences "
            "contrast on the fitted available-case sample; arm_gap_t3 is a "
            "post-crossover association and delta_crossover is descriptive catch-up."
        )
        causal_status = (
            "tau_t2 is randomised (a t2 difference-in-differences contrast on the "
            "available-case sample); arm_gap_t3, delta_crossover and any dose term "
            "are latent-ability-confounded associations, never randomised effects."
        )
    analysis_population = (
        "Available-case children observed across the difference-in-differences waves "
        "(about 53-54 depending on outcome). The randomised interpretation applies to "
        "the t2 contrast on this available-case population, not automatically the "
        "complete randomised cohort."
    )
    missing_data_assumption = (
        "Available-case analysis under ignorable missingness: missing outcomes and "
        "covariates are assumed ignorable given the modelled covariates."
    )

    return DiDRunPlan(
        model_id=spec.model_id,
        outcome_symbol=own,
        settings_source=source,
        dose=settings.dose,
        period_varying=period_varying,
        likelihood=settings.likelihood,
        off_floor=off_floor,
        outcomes=outcomes,
        waves=settings.waves,
        periods=settings.periods,
        use_child_re=settings.use_child_re,
        use_age=settings.use_age,
        use_varying_delta=settings.use_varying_delta,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
