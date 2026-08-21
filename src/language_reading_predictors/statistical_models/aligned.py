# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for the onset-aligned family (#394 pillar 4).

Mirrors the ITT / gain-factor / level-factor / DiD / concurrent run-plan pattern for
the per-protocol onset-aligned (``kind="aligned"``) models. A model module declares
its settings; the plan is resolved and **validated before any data are loaded or an
output directory is reset**, then drives data preparation, factory construction and
the ``config.json`` / ``model_recipe.md`` audit trail. This removes the untyped
``spec.extra`` boundary (where a misspelled key silently defaulted) and records the
resolved design, estimand, causal status, analysis population and missing-data
assumption alongside every fit.

The aligned design is a per-protocol onset-aligned single-gain ANCOVA (LRPAL): a
cross-sectional Beta-Binomial regression of the aligned post-score on its own onset
baseline, age-at-onset and cognitive ability, optionally with a cohort indicator and
the cumulative session dose. One row per child, so there is **no child random
intercept**. The cohort contrast is a **per-protocol association**, not the
available-case modified ITT estimate (it is confounded by age-at-onset and cohort/timing), and the
dose term is a collider descendant of group and ability -- a sensitivity variant.
The heavily-floored outcome P takes the suite floor rule
(``likelihood="bernoulli_offfloor"``): a Bernoulli on the off-floor indicator with
the **binary off-floor-at-onset indicator** as the own-baseline main effect (#391
finding 2, adopted here by the 2026-08-21 aligned review) and the cohort marginal
an off-floor risk difference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec

# The complete, closed set of legacy ``spec.extra`` keys the aligned family
# understands. Anything else is a typo and must fail before a fit starts.
_LEGACY_KEYS = frozenset(
    {
        "ability_covariate",
        "use_cohort",
        "use_dose",
        "likelihood",
        # Sampler knob, not a model setting: ``target_accept`` is resolved centrally by
        # ``context.make_context`` (CLI override > spec default > preset) and is never
        # read by this family's settings. Listed so a legitimate per-model declaration
        # is not rejected as a misspelling by the strict unknown-key check.
        "target_accept",
    }
)

_LIKELIHOODS = frozenset({"beta_binomial", "bernoulli_offfloor"})


@dataclass(frozen=True, slots=True)
class AlignedModelSettings:
    """Immutable settings declared by a single onset-aligned model module.

    Defaults encode the primary per-protocol ANCOVA: the cohort contrast on, without
    the collider-descendant session-dose covariate, and the Beta-Binomial working
    likelihood.
    """

    ability_covariate: str | None = None
    use_cohort: bool = True
    use_dose: bool = False
    likelihood: str = "beta_binomial"

    def __post_init__(self) -> None:
        if self.ability_covariate is not None and (
            not isinstance(self.ability_covariate, str) or not self.ability_covariate
        ):
            raise TypeError("ability_covariate must be a non-empty string or None")
        for flag in ("use_cohort", "use_dose"):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be bool")
        if self.likelihood not in _LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(_LIKELIHOODS)}, got {self.likelihood!r}"
            )

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> AlignedModelSettings:
        """Strictly translate the former ``spec.extra`` dictionary boundary.

        Rejects unknown keys so a misspelling fails before data loading rather than
        silently taking a default."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown aligned setting(s): {', '.join(unknown)}. "
                "Declare AlignedModelSettings so misspellings fail fast."
            )
        # Pass raw values through so __post_init__ is the single validation/coercion
        # point; pre-coercing here would silently reshape misshaped legacy settings.
        return cls(
            ability_covariate=extra.get("ability_covariate"),
            use_cohort=extra.get("use_cohort", True),
            use_dose=extra.get("use_dose", False),
            likelihood=extra.get("likelihood", "beta_binomial"),
        )


@dataclass(frozen=True, slots=True)
class AlignedRunPlan:
    """Concrete, validated instructions consumed by preparation and modelling."""

    model_id: str
    outcome_symbol: str
    settings_source: str
    ability_covariate: str | None
    use_cohort: bool
    use_dose: bool
    likelihood: str
    off_floor: bool
    # Recorded audit metadata (#394 pillar 4).
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
        """Arguments for ``load_and_prepare_aligned`` from the resolved plan.

        The aligned family uses its own onset-aligned loader (one row per child at the
        per-protocol onset window); ``include_dose`` requests the cumulative-session
        covariate only when the dose sensitivity variant is fit."""
        return {
            "outcomes": (self.outcome_symbol,),
            "ability_covariate": self.ability_covariate,
            "include_dose": self.use_dose,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_aligned_model`` for this plan."""
        return {
            "outcome_symbol": self.outcome_symbol,
            "ability_covariate": self.ability_covariate,
            "use_cohort": self.use_cohort,
            "use_dose": self.use_dose,
            "likelihood": self.likelihood,
        }

    def coefficient_names(self) -> list[str]:
        """Interpretable coefficients written to the aligned factor table.

        The off-floor likelihood replaces the graded ``gamma_own`` onset-logit
        coupling with the binary off-floor-at-onset contrast
        ``gamma_own_offfloor`` (#391 finding 2, adopted for this family by the
        2026-08-21 aligned review, finding 2)."""
        names: list[str] = []
        if self.use_cohort:
            names.append("beta_cohort")
        names += ["gamma_own_offfloor" if self.off_floor else "gamma_own", "gamma_A"]
        if self.ability_covariate:
            names.append("gamma_ability")
        if self.use_dose:
            names.append("gamma_dose")
        return names

    def diagnostic_vars(self) -> list[str]:
        """Variables scanned by summaries and the convergence gate."""
        tail = [] if self.off_floor else ["kappa"]
        return ["alpha", *self.coefficient_names(), *tail]

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        return (
            "Note: Generated from the validated onset-aligned run plan; template "
            "drafted by an LLM-based AI tool (Claude Code/Opus 4.8).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Likelihood: `{self.likelihood}`"
            + (
                " (a Bernoulli on the off-floor indicator, aligned post-score > 0, "
                "with no dispersion parameter; the own-baseline term is the binary "
                "off-floor-at-onset indicator and the cohort marginal is an "
                "off-floor risk difference in percentage points)"
                if self.off_floor
                else " (a Beta-Binomial on the bounded post-score count)"
            )
            + ". Ability covariate: "
            f"{self.ability_covariate or 'none'}. Cohort contrast: {self.use_cohort}. "
            f"Cumulative session dose (sensitivity): {self.use_dose}.\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate and posterior-predictive checks pass. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_aligned_settings(spec: ModelSpec) -> tuple[AlignedModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: aligned settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, AlignedModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='aligned' requires AlignedModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return (
        AlignedModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def resolve_aligned_run_plan(spec: ModelSpec) -> AlignedRunPlan:
    """Resolve and validate an onset-aligned spec before any data are loaded."""
    if spec.kind != "aligned":
        raise ValueError(f"{spec.model_id}: expected kind 'aligned', got {spec.kind!r}")
    if not spec.outcome_symbol:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol is required for an aligned model"
        )
    # Validate the outcome against the measure registry *before* make_context can
    # reset an output directory (2026-08-21 aligned review, finding 5) — the
    # loader's KeyError otherwise fires only after the reset.
    from language_reading_predictors.statistical_models.measures import MEASURES

    if spec.outcome_symbol not in MEASURES:
        raise ValueError(
            f"{spec.model_id}: unknown aligned outcome_symbol "
            f"{spec.outcome_symbol!r}; not in the measure registry"
        )

    settings, source = declared_aligned_settings(spec)
    own = spec.outcome_symbol
    off_floor = settings.likelihood == "bernoulli_offfloor"

    # The design and estimand must describe the fitted likelihood: the off-floor
    # variant is a Bernoulli on the off-floor indicator, not a Beta-Binomial on
    # the post-score (2026-08-21 aligned review, finding 3).
    if off_floor:
        design = (
            "Per-protocol onset-aligned off-floor analysis: a cross-sectional "
            "Bernoulli (logit) regression of the off-floor indicator (aligned "
            "post-score > 0) on the binary off-floor-at-onset indicator, "
            "age-at-onset and cognitive ability, optionally with a cohort "
            "indicator and the cumulative session dose. One row per child, so no "
            "child random intercept and no dispersion parameter."
        )
        estimand = (
            "The cohort contrast in the probability of being off the floor at the "
            "two arms' own onset-aligned endpoints (an off-floor risk difference) "
            "-- a per-protocol association, NOT an available-case modified ITT "
            "estimate (it is confounded by age-at-onset and cohort/timing). With "
            "the dose variant, the cumulative-session covariate is a collider "
            "descendant of group and ability, a sensitivity variant."
        )
    else:
        design = (
            "Per-protocol onset-aligned single-gain ANCOVA: a cross-sectional "
            "Beta-Binomial regression of the aligned post-score on its own onset baseline, "
            "age-at-onset and cognitive ability, optionally with a cohort indicator and "
            "the cumulative session dose. One row per child, so no child random intercept."
        )
        estimand = (
            "The cohort contrast at the two arms' own onset-aligned endpoints -- a "
            "per-protocol association, NOT an available-case modified ITT estimate (it is confounded by "
            "age-at-onset and cohort/timing). With the dose variant, the cumulative-session "
            "covariate is a collider descendant of group and ability, a sensitivity variant."
        )
    causal_status = (
        "Associational / per-protocol: no randomised contrast is estimated. The cohort "
        "term is confounded by age-at-onset and cohort/timing and the dose term is a "
        "collider descendant; neither is an available-case modified ITT estimate."
    )
    analysis_population = (
        "Available-case children with onset-aligned pre and post scores (the "
        "per-protocol onset window)."
    )
    missing_data_assumption = (
        "Available-case analysis under ignorable missingness: children without an "
        "onset-aligned pre+post pair are dropped and assumed ignorable given the "
        "modelled covariates."
    )

    return AlignedRunPlan(
        model_id=spec.model_id,
        outcome_symbol=own,
        settings_source=source,
        ability_covariate=settings.ability_covariate,
        use_cohort=settings.use_cohort,
        use_dose=settings.use_dose,
        likelihood=settings.likelihood,
        off_floor=off_floor,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
