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
randomised quantity is the **period-1 average marginal effect** of random
assignment. Since the #391 finding 3 decision (2026-07-22) the causal headline is
**interaction-free**: treatment-by-covariate interactions are estimated on all
stacked periods — including post-crossover rows with no untreated comparison — so
a headline that nets them out is partly model-dependent extrapolation. Headline
specifications therefore may not declare a ``trt`` interaction; the pre-specified
moderation questions live on in explicitly associational **moderation variants**
(``moderation_variant=True``), whose interaction-aware marginal keeps the #391
finding 1 netting and is labelled model-dependent rather than causal. Every skill
/ ability / interaction term is a latent-ability-confounded **adjusted
association**, never a causal effect.
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
        "moderation_variant",
        # Sampler knob, not a model setting: ``target_accept`` is resolved centrally by
        # ``context.make_context`` (CLI override > spec default > preset) and is never
        # read by this family's settings. Listed so a legitimate per-model declaration
        # is not rejected as a misspelling by the strict unknown-key check.
        "target_accept",
    }
)

_LIKELIHOODS = frozenset({"beta_binomial", "bernoulli_offfloor"})


def resolve_active_interactions(
    interactions: Any, *, treated_only: bool
) -> tuple[tuple[str, str], ...]:
    """The interactions a fit actually contains, given ``treated_only``.

    In a treated-only fit every kept row is on intervention, so the treatment
    indicator is constant and unidentified; ``build_gain_factors_model`` drops it and
    every interaction naming it. The ``b`` companions still *declare* those pairs, on
    purpose — it keeps each companion a one-line diff from its parent so the two are
    directly comparable — so the declared and effective sets legitimately differ and
    both are worth recording.

    This is the single definition of that rule for everything downstream of the
    factory: the run plan, the reported coefficient names and the covariate
    marginals. The factory keeps its own copy, since it accepts raw keyword
    arguments from any caller; ``test_gain_factors.py`` pins the two together.
    """
    pairs = tuple(tuple(p) for p in interactions)
    if not treated_only:
        return pairs
    return tuple(p for p in pairs if "trt" not in p)


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
    #: Explicitly associational treatment-moderation variant (#391 finding 3
    #: decision, 2026-07-22): the only kind of specification allowed to declare a
    #: ``trt`` interaction. Its interaction-aware marginal is model-dependent
    #: (partly informed by post-crossover data) and is never the causal headline —
    #: that lives in the interaction-free primary it varies.
    moderation_variant: bool = False

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
        if not isinstance(self.moderation_variant, bool):
            raise TypeError("moderation_variant must be bool")
        if self.likelihood not in _LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(_LIKELIHOODS)}, got {self.likelihood!r}"
            )
        # Interaction terms must name something the model actually builds (#455).
        # build_gain_factors_model raises the same way, but only once make_context has
        # reset an output directory and the loader has read the panel; the vocabulary
        # is fixed by skill_symbols and ability_covariate, both settings fields, so it
        # can be checked at declaration. This deliberately mirrors the factory's set
        # exactly — including "trt" for treated_only fits, which the factory also
        # allows — so the two cannot disagree about what is buildable.
        valid_terms = self.interaction_vocabulary()
        for pair in self.interactions:
            for term in pair:
                if term not in valid_terms:
                    raise ValueError(
                        f"interaction term {term!r} not available; "
                        f"have {sorted(valid_terms)}"
                    )
        # #391 finding 3 decision (2026-07-22): the causal headline is interaction-free.
        # Treatment interactions are estimated on all stacked periods — including
        # post-crossover rows with no untreated comparison — so a headline that nets
        # them out is partly model-dependent extrapolation, and one that ignores them
        # is the pre-#395 bug. Only an explicitly associational moderation variant may
        # declare a trt pair; everything else (headline primaries AND their treated-only
        # companions, which stay a one-line diff from their parents) must not. Checked
        # after the vocabulary loop so a typo'd term reads as a typo, not as a
        # finding-3 violation.
        trt_pairs = tuple(p for p in self.interactions if "trt" in p)
        if self.moderation_variant:
            if self.treated_only:
                raise ValueError(
                    "moderation_variant is incoherent with treated_only: the "
                    "treatment indicator is constant in a treated-only fit, so no "
                    "treatment interaction can be estimated"
                )
            if not trt_pairs:
                raise ValueError(
                    "moderation_variant requires at least one trt interaction — a "
                    "moderation variant without treatment moderation is a headline "
                    "specification, so declare it as one"
                )
        elif trt_pairs:
            listed = ", ".join(f"({a}, {b})" for a, b in trt_pairs)
            raise ValueError(
                "headline gain-factor specifications are interaction-free in trt "
                "(#391 finding 3 decision): declare moderation_variant=True to fit "
                f"treatment interactions as an explicitly associational variant "
                f"(got {listed})"
            )

    def interaction_vocabulary(self) -> frozenset[str]:
        """Terms an interaction pair may name, given the declared skills / ability."""
        terms = {"trt", "age", "own", *self.skill_symbols}
        if self.ability_covariate is not None:
            terms.add("ability")
        return frozenset(terms)

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
        # Pass raw values through so __post_init__ is the single validation/coercion
        # point; pre-coercing here (tuple(...)/bool(...)) would silently reshape
        # misshaped legacy settings ("TR" -> ('T', 'R'), 1 -> True) instead of failing
        # fast against the strict checks in __post_init__.
        return cls(
            skill_symbols=extra.get("skill_symbols", ()),
            ability_covariate=extra.get("ability_covariate"),
            adjust_for=extra.get("adjust_for", ()),
            interactions=extra.get("interactions", ()),
            treated_only=extra.get("treated_only", False),
            likelihood=extra.get("likelihood", "beta_binomial"),
            moderation_variant=extra.get("moderation_variant", False),
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
    moderation_variant: bool
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

    @property
    def active_interactions(self) -> tuple[tuple[str, str], ...]:
        """The interactions the fitted model actually contains.

        ``interactions`` is what the module *declared*; this is what survives
        resolution. The two differ only for a treated-only fit — see
        :func:`resolve_active_interactions`.
        """
        return resolve_active_interactions(
            self.interactions, treated_only=self.treated_only
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        d = asdict(self)
        # asdict turns the interaction pairs into lists; keep them as [a, b] lists
        # (JSON has no tuples) — round-trips fine and reads cleanly.
        d["interactions"] = [list(p) for p in self.interactions]
        # Record what was actually fitted alongside what was declared. A treated-only
        # fit drops its trt interactions, so recording only the declared list would
        # describe a model with terms the posterior does not contain (#455 follow-up).
        d["active_interactions"] = [list(p) for p in self.active_interactions]
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
            # The effective set, not the declared one. Byte-identical in the built
            # model — the factory applies the same filter — but it means the arguments
            # the plan hands over are the arguments the fit actually uses.
            "interactions": self.active_interactions,
            "treated_only": self.treated_only,
            "likelihood": self.likelihood,
        }

    def coefficient_names(
        self, *, effective_adjustment: tuple[str, ...] | None = None
    ) -> list[str]:
        """Interpretable coefficients written to the gain-factor table."""
        adjust_for = (
            self.adjust_for
            if effective_adjustment is None
            else effective_adjustment
        )
        names: list[str] = []
        if not self.treated_only:
            names.append("beta_trt")
        names.append("gamma_own_offfloor" if self.off_floor else "gamma_own")
        names.append("gamma_A")
        if self.ability_covariate:
            names.append("gamma_ability")
        names += [f"gamma_{symbol}" for symbol in self.skill_symbols]
        names += [f"gamma_{covariate}" for covariate in adjust_for]
        names += [f"gamma_int_{left}_{right}" for left, right in self.active_interactions]
        return names

    def diagnostic_vars(
        self, *, effective_adjustment: tuple[str, ...] | None = None
    ) -> list[str]:
        """Variables scanned by summaries and the convergence gate."""
        tail = ["sigma_child"] if self.off_floor else ["kappa", "sigma_child"]
        return [
            "alpha",
            "alpha_phase",
            *self.coefficient_names(effective_adjustment=effective_adjustment),
            *tail,
        ]

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        skills = ", ".join(self.skill_symbols) if self.skill_symbols else "none"
        adjust = ", ".join(self.adjust_for) if self.adjust_for else "none"
        # Report the interactions the fit contains, and say plainly which declared
        # ones were dropped — a treated-only recipe that listed trt x ability would
        # describe a coefficient the posterior does not have.
        active = self.active_interactions
        inter = "; ".join(f"{a} x {b}" for a, b in active) if active else "none"
        dropped = tuple(p for p in self.interactions if p not in active)
        if dropped:
            inter += (
                " (declared but not fitted, because the treatment indicator is "
                "constant in a treated-only fit and so is dropped along with its "
                "interactions: "
                + "; ".join(f"{a} x {b}" for a, b in dropped)
                + ")"
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
            "Period-stacked off-floor model: a Bernoulli likelihood on the child's "
            "off-floor status at the period end (post > 0) — pooling moving off the "
            "floor, staying above it and returning to it, not a move-off-the-floor "
            "transition — with the binary off-floor-at-pre indicator as the baseline "
            "main effect (#391 finding 2 decision — the graded pre logit of a "
            "heavily-floored measure is a near-degenerate spike, so the indicator is "
            "the honest functional form)."
        )
        estimand = (
            "Period-1 average marginal effect of random assignment on the "
            "probability of being off the floor at the period end (a risk "
            "difference), on the fitted available-case sample."
        )
    else:
        design = (
            "Period-stacked ANCOVA: the post-score is regressed on the child's own "
            "pre-score (a Beta-Binomial working likelihood) with a non-centred child "
            "random intercept for repeated observations."
        )
        estimand = (
            "Period-1 average marginal effect of random assignment on the fitted "
            "available-case sample."
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
    elif settings.moderation_variant:
        estimand = (
            "Interaction-aware period-1 average marginal effect of the "
            "on-intervention term (beta_trt with every fitted treatment interaction "
            "netted out), on the fitted available-case sample. The treatment "
            "interactions are estimated on ALL stacked periods — including "
            "post-crossover rows with no untreated comparison — so this marginal is "
            "model-dependent and partly informed by post-crossover data, not "
            "exclusively randomised period-1 evidence (#391 finding 3)."
        )
        causal_status = (
            "Explicitly associational moderation variant: no term here is reported "
            "as a standalone causal effect. The randomised causal headline for this "
            "outcome lives in the interaction-free primary this model varies; the "
            "treatment-by-covariate interactions and the netted marginal are "
            "model-dependent adjusted associations."
        )
    else:
        # Headline: interaction-free in trt by validated invariant, so the period-1
        # marginal direction coincides with the beta_trt coefficient draw-by-draw.
        causal_status = (
            "The treatment term is randomised (a period-1 average marginal effect; "
            "the headline specification carries no treatment interactions, #391 "
            "finding 3 decision); every skill, ability and interaction term is a "
            "latent-ability-confounded adjusted association, never a causal effect."
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
        moderation_variant=settings.moderation_variant,
        baseline_covariates=baseline_covariates,
        pre_covariates=pre_adj,
        post_covariates=post_adj,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
