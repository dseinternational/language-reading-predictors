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
Under the default ``arm_gap_reference="t1"`` parameterisation (#552) the arm-by-time
vector is **centred on the timepoint-1 arm gap**: ``arm_gap_t1`` is the
covariate-adjusted pre-randomisation balance quantity and ``d_grp_time[t]`` the
change in the arm gap from t1 to each later wave, with the per-wave levels view
``b_grp_time = (arm_gap_t1, arm_gap_t1 + d_grp_time)`` retained as a Deterministic.
The single randomised quantity is then the **t2 change** ``d_grp_time[t2]`` -- a
difference-in-differences of adjusted levels, read as an items- or
risk-difference average marginal effect at the t2 rows. ``arm_gap_reference="free"``
keeps the former parameterisation (a free per-timepoint vector whose t2 element
``b_grp_time[1]`` is the focal raw gap) as an explicit comparator. Either way the
other waves are post-crossover and every ability / interaction term is a
latent-ability-confounded **adjusted association**, never a causal effect.

The natural-scale target -- open through #389 finding 1 and #584 finding 1 -- was
settled on 2026-08-23 (``notes/202608231800-level-factors-584-decisions.md``): the
card is the **arm-free standardised** marginal effect, the average over the fitted
t2 rows, each evaluated at its own arm-free profile, of adding the focal contrast
alone. The standardisation population is the fitted t2 children, the random-effect
convention is each child's own posterior intercept, and the time-invariant
``group x ability`` increment is held at centred ability and reported separately.
:meth:`LevelFactorsRunPlan.natural_scale_estimand` states it in the stored
``config.json`` so no reader has to infer it from the reporting code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import KAPPA_PRIOR_FAMILIES
from language_reading_predictors.statistical_models.likelihood import (
    SCORE_MEAN_LINKS,
    ScoreMeanLink,
)
from language_reading_predictors.statistical_models.preprocessing import (
    MISSINGNESS_INDICATOR_PAIRS,
    PreparedData,
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
        "arm_gap_reference",
        "score_mean_link",
        "waves",
        "kappa_prior_family",
        "kappa_prior_sigma",
        "sigma_child_prior_sigma",
        # Sampler knob, not a model setting: ``target_accept`` is resolved centrally by
        # ``context.make_context`` (CLI override > spec default > preset) and is never
        # read by this family's settings. Listed so a legitimate per-model declaration
        # is not rejected as a misspelling by the strict unknown-key check.
        "target_accept",
    }
)

_LIKELIHOODS = frozenset({"beta_binomial", "bernoulli_offfloor"})

# How the per-timepoint arm coefficients are parameterised (#552). ``"t1"`` centres
# them on the pre-randomisation t1 gap (balance term + changes, the default);
# ``"free"`` keeps one free coefficient per timepoint (the pre-#552 comparator).
ARM_GAP_REFERENCES = frozenset({"t1", "free"})

#: The registered phoneme-blending link pair (#584 decision 2): the ordinary-logit
#: primary and its one-in-three guessing-floor companion. Neither releases without
#: the other, so the ids are resolved here — the one place that already knows which
#: outcome a plan fits — rather than being restated by each consumer.
LEVEL_BLENDING_PRIMARY_MODEL_ID = "lrp-rli-lf-006"
LEVEL_BLENDING_COMPANION_MODEL_ID = "lrp-rli-lf-106"

#: Labels of the post-t1 waves whose arm-gap *changes* ``d_grp_time`` carries
#: (the ``post_phase`` coordinate): t2 is the randomised contrast, t3 / t4 are
#: post-crossover associations. A two-wave comparator (#584 decision 3) carries
#: only ``("t2",)``; :meth:`LevelFactorsRunPlan.post_phase_labels` derives the
#: right subset from the declared analysis window.
POST_PHASE_LABELS: tuple[str, ...] = ("t2", "t3", "t4")

#: Every wave of the levels panel, in order. The declared analysis window is a
#: contiguous **prefix** of this: the t1-centred parameterisation measures every
#: arm-gap change from t1, so t1 is always present and a gap in the middle would
#: leave a ``d_grp_time`` coordinate with no data behind it.
WAVE_LABELS: tuple[str, ...] = ("t1", *POST_PHASE_LABELS)


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
    effect-modification term, the arm-by-time vector centred on the t1 gap
    (``arm_gap_reference="t1"``, #552) and the Beta-Binomial working likelihood.

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
    arm_gap_reference: str = "t1"
    #: Phoneme-blending response link (#584 decision 2). ``"logit"`` is the ordinary
    #: Beta-Binomial inverse-logit mean; ``"three_choice_guessing_floor"`` maps it
    #: onto [1/3, 1] for the ten three-alternative forced-choice blending items.
    #: B only, graded only, and released only beside its paired opposite-link fit.
    score_mean_link: ScoreMeanLink = "logit"
    #: Dispersion parameterisation (#584 decision 4). The default puts the prior on
    #: the **dispersion scale** ``1/sqrt(kappa)``, where "no extra-Binomial
    #: dispersion beyond the child random intercept" is simply zero and therefore
    #: reachable; ``"halfnormal_concentration"`` is the pre-decision prior, retained
    #: as the comparator and for the sensitivity sweep. Inert on an off-floor fit,
    #: which has no score mean and so no concentration -- the resolved plan records
    #: ``None`` there rather than a setting that does nothing.
    kappa_prior_family: str = "halfnormal_inverse_sqrt"
    #: Scale override for whichever dispersion prior is in force (the sweep axis).
    #: ``None`` keeps the registered default for the declared family.
    kappa_prior_sigma: float | None = None
    #: Child random-intercept SD prior (#584 decision 4). A levels model has **no
    #: own-baseline term**, so this intercept carries the entire between-child
    #: spread in level, where a gain model conditions that spread away -- and the
    #: gain-model scale of 0.5 asserts a middle-95% child range of 0.18 to 0.45 on a
    #: mid-difficulty measure, narrower than the tests were built to resolve.
    sigma_child_prior_sigma: float = 1.0
    #: The waves this fit analyses (#584 decision 3). The default is the whole
    #: four-wave panel, the model of record. ``("t1", "t2")`` is the **randomised
    #: window comparator**: the same model restricted to the pre-randomisation
    #: baseline and the one wave at which the arms differ by randomisation alone,
    #: so its t2 contrast borrows nothing from the post-crossover waves through
    #: the shared balance term, child intercept, dispersion or group x ability.
    waves: tuple[str, ...] = WAVE_LABELS

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
        if self.arm_gap_reference not in ARM_GAP_REFERENCES:
            raise ValueError(
                "arm_gap_reference must be one of "
                f"{sorted(ARM_GAP_REFERENCES)}, got {self.arm_gap_reference!r}"
            )
        object.__setattr__(self, "waves", _tuple_of_strings(self.waves, name="waves"))
        if self.waves != WAVE_LABELS[: len(self.waves)] or len(self.waves) < 2:
            raise ValueError(
                "waves must be a contiguous prefix of "
                f"{WAVE_LABELS} with at least two entries (the t1 reference and "
                f"one later wave), got {self.waves}"
            )
        if self.kappa_prior_family not in KAPPA_PRIOR_FAMILIES:
            raise ValueError(
                "kappa_prior_family must be one of "
                f"{sorted(KAPPA_PRIOR_FAMILIES)}, got {self.kappa_prior_family!r}"
            )
        for name in ("kappa_prior_sigma", "sigma_child_prior_sigma"):
            value = getattr(self, name)
            if value is None and name == "kappa_prior_sigma":
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number, got {value!r}")
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if self.score_mean_link not in SCORE_MEAN_LINKS:
            raise ValueError(
                f"score_mean_link must be one of {SCORE_MEAN_LINKS}, "
                f"got {self.score_mean_link!r}"
            )
        if (
            self.score_mean_link != "logit"
            and self.likelihood != "beta_binomial"
        ):
            raise ValueError(
                "score_mean_link applies to the graded Beta-Binomial mean; the "
                f"{self.likelihood!r} branch has no score mean to map"
            )
        # Adjustment-set hygiene (#584 lower-severity 4). A repeated adjuster used
        # to survive resolution and fail only later inside PyMC, when the factory
        # tried to create a second ``gamma_<c>`` with the same name; an indicator
        # declared without its base term used to fit a missingness flag with no
        # covariate for it to flag, which is not the two-term missing-indicator
        # idiom the reports describe.
        duplicates = sorted({c for c in self.adjust_for if self.adjust_for.count(c) > 1})
        if duplicates:
            raise ValueError(
                f"adjust_for repeats {', '.join(duplicates)}; each adjuster enters "
                "the linear predictor once"
            )
        indicator_bases = {v: k for k, v in MISSINGNESS_INDICATOR_PAIRS.items()}
        unpaired = sorted(
            c
            for c in self.adjust_for
            if c in indicator_bases and indicator_bases[c] not in self.adjust_for
        )
        if unpaired:
            raise ValueError(
                f"adjust_for declares missing-indicator(s) {', '.join(unpaired)} "
                "without the covariate they flag; declare the base term alongside "
                "each indicator"
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
            arm_gap_reference=extra.get("arm_gap_reference", "t1"),
            score_mean_link=extra.get("score_mean_link", "logit"),
            waves=extra.get("waves", WAVE_LABELS),
            kappa_prior_family=extra.get(
                "kappa_prior_family", "halfnormal_inverse_sqrt"
            ),
            kappa_prior_sigma=extra.get("kappa_prior_sigma"),
            sigma_child_prior_sigma=extra.get("sigma_child_prior_sigma", 1.0),
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
    # Phoneme-blending response link and its release pairing (#584 decision 2).
    # ``required_link_companion_model_id`` names the opposite-link fit that must be
    # released beside this one; ``link_sensitivity_required_for_release`` is the
    # policy flag the release gate reads, so a future B level fit outside the
    # registered pair fails closed rather than publishing unpaired.
    score_mean_link: str
    required_link_companion_model_id: str | None
    link_sensitivity_required_for_release: bool
    # The declared analysis window (#584 decision 3): the four-wave panel (the
    # model of record) or the two-wave randomised-window comparator.
    waves: tuple[str, ...]
    # Dispersion and child-heterogeneity priors (#584 decision 4).
    # ``kappa_prior_family`` is ``None`` on an off-floor fit, which has no
    # concentration at all -- recorded as absent rather than as an inert setting.
    kappa_prior_family: str | None
    kappa_prior_sigma: float | None
    sigma_child_prior_sigma: float
    # Arm-by-time parameterisation (#552): ``"t1"`` (balance term + changes) or
    # ``"free"`` (one free coefficient per timepoint).
    arm_gap_reference: str
    # The focal randomised quantity, resolved once here so the factory, the AME,
    # the key-findings builder, the release gate and the prior-sensitivity runner
    # all read the same term. ``focal_vector`` is the posterior variable,
    # ``focal_index`` the element position within it and ``focal_term`` the
    # labelled element as it appears in ``factor_summary.csv`` /
    # ``psense_summary.csv`` (``d_grp_time[t2]`` or ``b_grp_time[1]``). All three
    # are ``None`` for a pooled group term, which has no randomised element.
    focal_vector: str | None
    focal_index: int | None
    focal_term: str | None
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
        """Return the JSON-ready run-plan contract for ``config.json``.

        The derived natural-scale target rides along (#584 decision 1): it is a
        property rather than a field because it is a statement *about* the settings,
        but a stored fit must record which quantity its card is, not leave a reader
        to infer it from whichever reporting code happens to be current."""
        recorded = asdict(self)
        recorded["natural_scale_estimand"] = self.natural_scale_estimand
        recorded["standardisation_balance_term"] = self.standardisation_balance_term
        return recorded

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
            "arm_gap_reference": self.arm_gap_reference,
            "score_mean_link": self.score_mean_link,
            "post_phase_labels": self.post_phase_labels,
            "sigma_child_prior_sigma": self.sigma_child_prior_sigma,
            **(
                {}
                if self.kappa_prior_family is None
                else {
                    "kappa_prior_family": self.kappa_prior_family,
                    "kappa_prior_sigma": self.kappa_prior_sigma,
                }
            ),
        }

    # -- Single source of truth for names, roles and diagnostics (#389 finding 6:
    # the review found coefficient names and diagnostic variables separately
    # reconstructed by ``_lf_coef_names``, ``_lf_diag_vars``, the factory and the
    # reporting code; they now all derive from the resolved plan).

    def coefficient_names(
        self, *, effective_adjustment: tuple[str, ...] | None = None
    ) -> list[str]:
        """The reported structural coefficients, in report order.

        ``effective_adjustment`` mirrors :meth:`factory_kwargs`: the loader drops a
        constant covariate (e.g. an all-zero ``_missing`` indicator), and the
        reported set must match what was actually built."""
        adj = self.adjust_for if effective_adjustment is None else effective_adjustment
        if not self.group_by_time:
            names = ["beta_grp"]
        elif self.t1_referenced:
            # Balance term, the arm-gap changes (t2 randomised) and the derived
            # per-wave levels view, in that order (#552).
            names = ["arm_gap_t1", "d_grp_time", "b_grp_time"]
        else:
            names = ["b_grp_time"]
        names.append("gamma_A")
        if self.ability_covariate:
            names.append(
                "gamma_ability_time" if self.ability_by_time else "gamma_ability"
            )
            if self.group_ability:
                names.append("gamma_grp_ability")
        names += [f"gamma_{c}" for c in adj]
        return names

    def diag_vars(
        self, *, effective_adjustment: tuple[str, ...] | None = None
    ) -> list[str]:
        """Variables named in the summary/gate diagnostics for this plan's model.

        ``alpha`` is a Deterministic (the t1-anchored level) and ``alpha_offset``
        its free empirical-Bayes offset (#389 finding 2); both are reported."""
        # The gate scans free RVs; the reports read ``kappa``. Under the dispersion
        # parameterisation those are different variables, so both are named (#584
        # decision 4).
        tail = (
            ["sigma_child"]
            if self.off_floor
            else [
                *dict.fromkeys([self.dispersion_free_term, "kappa"]),
                "sigma_child",
            ]
        )
        return [
            "alpha",
            "alpha_offset",
            "alpha_time",
            *self.coefficient_names(effective_adjustment=effective_adjustment),
            *tail,
        ]

    @property
    def post_phase_labels(self) -> tuple[str, ...]:
        """The post-t1 waves this fit carries, i.e. ``d_grp_time``'s coordinates."""
        return tuple(self.waves[1:])

    @property
    def two_wave_window(self) -> bool:
        """True when the fit is restricted to the randomised t1 -> t2 window.

        The distinguishing property of this comparator is not that it is smaller:
        it is that **no post-crossover observation informs the reported contrast**.
        In the four-wave model of record the t3/t4 likelihood reaches the t2 change
        through the shared ``arm_gap_t1``, the child intercept, the dispersion and
        the time-invariant group x ability term (posterior correlations between
        ``arm_gap_t1`` and ``d_grp_time[t2]`` run from -0.07 to -0.44 across the
        stored suite). Here there is nothing for it to reach through."""
        return self.waves == ("t1", "t2")

    @property
    def standardisation_balance_term(self) -> str | None:
        """The group term the natural-scale standardisation nets out beside the focal
        contrast (#584 decision 1).

        Under the t1-centred parameterisation the fitted group contribution at t2 is
        ``(arm_gap_t1 + d_grp_time[t2] + gamma_grp_ability x ability) * G``, and the
        card removes all of it before adding the focal contrast back, so both arms
        are evaluated at the same arm-free operating point. Under the free
        comparator the focal ``b_grp_time[1]`` *is* the whole t2 gap, so there is no
        separate balance term to remove and this is ``None``."""
        return "arm_gap_t1" if self.t1_referenced else None

    @property
    def natural_scale_estimand(self) -> str:
        """The published natural-scale target, in words (#584 decision 1).

        Recorded in ``config.json`` through :meth:`as_dict` so a stored fit states
        the standardisation population, the random-effect convention and the
        treatment of effect modification rather than leaving them to be inferred
        from the reporting code."""
        if self.focal_term is None:
            return (
                "none: a pooled group coefficient is not a randomised contrast, so "
                "no natural-scale treatment effect is published for this plan"
            )
        scale = (
            "off-floor risk difference"
            if self.off_floor
            else "items-scale average marginal effect"
        )
        return (
            f"Arm-free standardised {scale} of {self.focal_term}: the average, over "
            "the fitted timepoint-2 rows each evaluated at its own arm-free profile "
            "(the complete group contribution netted out, each row keeping its own "
            "age, ability, adjusters and fitted child intercept), of adding the "
            "focal contrast alone. Standardisation population: the fitted t2 "
            "children. Random-effect convention: each child's own posterior "
            "intercept. Effect modification: the group x ability increment held at "
            "centred ability and reported separately, never folded into the card."
        )

    @property
    def dispersion_free_term(self) -> str | None:
        """The **free** dispersion parameter, or ``None`` on an off-floor fit.

        Under the dispersion parameterisation (#584 decision 4) ``kappa`` is a
        Deterministic and ``inv_sqrt_kappa`` is the sampled quantity, so anything
        that must name a free random variable — power scaling, the all-free-RV
        convergence gate — has to ask for this rather than for ``kappa``. The
        reports keep speaking in ``kappa``, which is the unit the family
        documents."""
        if self.off_floor:
            return None
        return (
            "inv_sqrt_kappa"
            if self.kappa_prior_family == "halfnormal_inverse_sqrt"
            else "kappa"
        )

    @property
    def nuisance_terms(self) -> tuple[str, ...]:
        """Free nuisance parameters that carry their own prior-data conflict risk.

        The child random-intercept SD and (on a graded outcome) the free dispersion
        parameter. Both are scale parameters on half-normal priors, and the stored
        suite flagged both for prior-data conflict in the #584 finding-6 audit, so
        the family power-scaling audit reports them beside the arm terms rather than
        establishing focal-term behaviour alone."""
        dispersion = self.dispersion_free_term
        return ("sigma_child",) if dispersion is None else (dispersion, "sigma_child")

    @property
    def t1_referenced(self) -> bool:
        """True when the arm-by-time vector is centred on the t1 gap (#552)."""
        return self.group_by_time and self.arm_gap_reference == "t1"

    @property
    def causal_vector(self) -> str:
        """The group coefficient the extended diagnostics treat as focal: the
        arm-gap *changes* under the t1 reference, the free per-timepoint vector
        under ``"free"``, the pooled main effect otherwise."""
        if self.focal_vector is not None:
            return self.focal_vector
        return "beta_grp"

    @property
    def causal_terms(self) -> tuple[str, ...]:
        """The elements flagged causal in summaries: only the randomised t2
        contrast, and only when group is entered per timepoint — a pooled
        ``beta_grp`` mixes post-crossover waves and is never flagged."""
        return (self.focal_term,) if self.focal_term is not None else ()

    @property
    def balance_terms(self) -> tuple[str, ...]:
        """Terms reported as pre-randomisation *balance* quantities — never as
        effects (#552): the adjusted t1 arm gap under the t1 reference."""
        return ("arm_gap_t1",) if self.t1_referenced else ()

    @property
    def levels_view_terms(self) -> tuple[str, ...]:
        """Derived per-wave arm gaps retained for the levels view (#552): the
        Deterministic ``b_grp_time`` under the t1 reference. Its elements are
        neither the causal estimand (that is ``d_grp_time[t2]``) nor ordinary
        adjusted associations, so the summary labels them separately."""
        return ("b_grp_time",) if self.t1_referenced else ()

    def factor_summary_roles(self) -> dict[str, str]:
        """Role overrides for :func:`reporting.factor_summary` beyond the
        causal / association split: the balance term and the levels view.

        Under the free comparator the t1 element of the free vector,
        ``b_grp_time[0]``, is the covariate-adjusted pre-randomisation arm gap —
        a balance quantity, not an adjusted association — so it takes the
        ``balance`` role by element label (2026-08-20 review, finding 9).
        ``balance_terms`` itself stays t1-reference-only because its other
        consumers (the forest / psense variable lists) need whole variable
        names, not indexed elements."""
        roles = {t: "balance" for t in self.balance_terms}
        roles.update({t: "levels_view" for t in self.levels_view_terms})
        if self.group_by_time and not self.t1_referenced:
            roles["b_grp_time[0]"] = "balance"
        return roles

    def restrict_to_declared_waves(self, prepared: PreparedData) -> PreparedData:
        """Drop rows outside the declared analysis window (#584 decision 3).

        Returns ``prepared`` unchanged for the four-wave model of record. For the
        two-wave comparator it keeps the t1 and t2 rows, which leaves ``phase``
        already dense (0, 1) and ``n_phases`` recomputed by the shared subsetter —
        the window is a *prefix*, so no renumbering is needed. The drop is
        attributed to its own reason rather than folded into the missing-data
        counts, because these rows are excluded by design, not by absence.
        """
        if len(self.waves) == len(WAVE_LABELS):
            return prepared
        from language_reading_predictors.statistical_models.factories import _subset

        keep = np.asarray(prepared.phase) < len(self.waves)
        return _subset(prepared, keep, reason="outside_declared_analysis_window")

    def validate_prepared(self, prepared: PreparedData) -> None:
        """Fail before model construction if the loaded panel cannot identify the
        declared quantities (#389 acceptance criterion: fail before fitting if t2
        lacks either randomised arm or required ability values are non-finite).

        Every wave carries a *published* arm coefficient — the t1 balance term (the
        reference the t1-centred changes are measured from), the randomised t2
        change, and the post-crossover t3/t4 changes — so each wave must contain
        both randomised arms among its fitted rows, not t2 alone (#584 finding 8).
        A wave with one arm leaves its coefficient determined by its prior and by
        the other waves through the shared reference, while the report still labels
        the t2 change a t1-to-t2 randomised difference-in-differences.

        Mirrors the factory's row filter: rows with a missing outcome (or a
        missing requested adjuster) never enter the likelihood, so they are
        excluded from the checks too."""
        own = self.outcome_symbol
        fitted = ~np.isnan(prepared.post_counts[own])
        for c in self.adjust_for:
            if c in prepared.covariates:
                fitted = fitted & ~np.isnan(prepared.covariates[c])
        if self.group_by_time:
            labels = self.waves
            for wave in range(int(prepared.n_phases)):
                label = labels[wave] if wave < len(labels) else f"phase {wave}"
                rows = fitted & (prepared.phase == wave)
                arms = {int(g) for g in np.unique(prepared.G[rows])}
                if {0, 1} <= arms:
                    continue
                role = {
                    "t1": (
                        "the balance term arm_gap_t1 the changes are measured from"
                        if self.t1_referenced
                        else "the t1 arm gap b_grp_time[0]"
                    ),
                    "t2": f"the declared randomised t2 contrast {self.focal_term}",
                }.get(label, f"the published {label} arm gap")
                raise ValueError(
                    f"{self.model_id}: the {label} rows with an observed {own} "
                    "outcome do not contain both randomised arms (present: "
                    f"{sorted(arms)}), so {role} is unidentified."
                )
        if self.ability_covariate is not None:
            ability = np.asarray(
                prepared.covariates[self.ability_covariate], dtype=float
            )
            bad = int(np.sum(fitted & ~np.isfinite(ability)))
            if bad:
                raise ValueError(
                    f"{self.model_id}: {bad} fitted row(s) carry a non-finite "
                    f"{self.ability_covariate!r} value, so the ability and "
                    "group x ability terms are not computable; a NaN here would "
                    "otherwise propagate silently into the likelihood."
                )

    def _arm_gap_reference_prose(self) -> str:
        if not self.group_by_time:
            return (
                "a single pooled group coefficient (`beta_grp`) across all waves -- "
                "not a randomised contrast, because it mixes post-crossover waves"
            )
        if self.t1_referenced:
            return (
                "centred on the timepoint-1 arm gap (#552): `arm_gap_t1` is the "
                "covariate-adjusted pre-randomisation balance quantity, `d_grp_time[t]` "
                "the change in the arm gap from t1 to t2 / t3 / t4, and the per-wave "
                "levels view `b_grp_time = (arm_gap_t1, arm_gap_t1 + d_grp_time)` is "
                "kept as a derived quantity; the randomised contrast is the t2 change "
                "`d_grp_time[t2]`, a difference-in-differences of adjusted levels"
            )
        return (
            "one free coefficient per timepoint (`b_grp_time[t]`, the pre-#552 "
            "comparator); the randomised contrast is the raw adjusted t2 gap "
            "`b_grp_time[1]`, which carries any covariate-adjusted chance imbalance "
            "present at t1"
        )

    def _prior_choices_markdown(self) -> str:
        """The two nuisance priors this family sets for itself (#584 decision 4)."""
        child = (
            f"Child heterogeneity: `sigma_child ~ HalfNormal("
            f"{self.sigma_child_prior_sigma:g})`"
            + (
                " -- wider than the shared 0.5 because a levels model has no "
                "own-baseline term, so this intercept carries the whole "
                "between-child spread in level rather than the residual a gain "
                "model leaves."
                if self.sigma_child_prior_sigma > 0.5
                else "."
            )
        )
        if self.kappa_prior_family is None:
            dispersion = (
                "Dispersion: none -- the off-floor branch models a binary "
                "indicator, which has no score mean and so no concentration."
            )
        elif self.kappa_prior_family == "halfnormal_inverse_sqrt":
            dispersion = (
                "Dispersion: the prior sits on `1/sqrt(kappa)`, where "
                "\"no extra-Binomial dispersion beyond the child random "
                "intercept\" is zero and therefore reachable; `kappa` is reported "
                "as a derived quantity. The scale is calibration-preserving -- it "
                "reproduces the previous prior's median variance inflation at every "
                "level denominator to within 3%."
            )
        else:
            dispersion = (
                "Dispersion: `kappa ~ HalfNormal` on the concentration scale (the "
                "pre-#584 comparator, which cannot reach the near-Binomial limit)."
            )
        return f"## Nuisance priors\n\n{child} {dispersion}\n\n"

    def _required_robustness_markdown(self) -> str:
        """The release pairing this fit cannot publish without (#584 decision 2).

        Mirrors the ITT plan's own required-robustness section: phoneme blending is
        response-link sensitive, so the ordinary-logit and guessing-floor estimates
        are a pair, and the recipe says so rather than leaving the requirement to
        the release gate alone."""
        if not self.link_sensitivity_required_for_release:
            return ""
        return (
            "## Required robustness\n\n"
            "Phoneme blending is response-link sensitive: each item has three "
            "response alternatives, so the ordinary inverse-logit mean permits "
            "expected scores below chance while the guessing-floor link does not. "
            "Release requires this fit to be reported beside the current "
            f"`{self.required_link_companion_model_id}` fit; neither the "
            "ordinary-logit nor the guessing-floor estimate is sufficient evidence "
            "on its own.\n\n"
        )

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
            f"{self.group_ability}. Arm-by-time parameterisation: "
            f"{self._arm_gap_reference_prose()}. Requested adjustment terms: "
            f"{adjust}. Score-mean link: {self.score_mean_link}.\n\n"
            f"{self._prior_choices_markdown()}"
            f"{self._required_robustness_markdown()}"
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
    if settings.ability_by_time and settings.ability_covariate is None:
        # ``ability_by_time`` silently did nothing without a covariate to vary, so a
        # declaration could claim a per-wave ability vector the fit never built and
        # the report never showed (#584 lower-severity 4). A model with no ability
        # covariate must say ability_by_time=False rather than leave the default.
        raise ValueError(
            f"{spec.model_id}: ability_by_time requires an ability_covariate; "
            "declare ability_by_time=False for a model that fits no ability term"
        )
    if settings.arm_gap_reference == "t1" and not settings.group_by_time:
        # A pooled group main effect has no per-wave gap to centre on the t1 gap:
        # the declaration is incoherent and must fail before the output directory
        # is reset or data are loaded (#552). Declare arm_gap_reference="free"
        # explicitly for a pooled comparator.
        raise ValueError(
            f"{spec.model_id}: arm_gap_reference='t1' requires group_by_time=True "
            "(a pooled group coefficient has no t1 gap to centre on); declare "
            "arm_gap_reference='free' for a pooled group term"
        )
    off_floor = settings.likelihood == "bernoulli_offfloor"
    # Phoneme blending is response-link sensitive, so a graded B level fit is
    # released only beside its opposite-link twin (#584 decision 2, mirroring the
    # registered ITT-008 / ITT-108 pair). The off-floor branch is exempt: it models
    # a binary indicator, which has no chance floor to respect.
    #
    # So is a **window comparator** (#584 decision 3), for the reason the gain
    # family's moderation variants are outside the robustness gate: the pairing
    # governs the fit whose B card is published as the headline, and that is the
    # four-wave model of record. A two-wave comparator is a diagnostic reported
    # beside that headline, and requiring it to be link-paired would demand a
    # two-wave floor-link twin that does not exist -- fail-closed doing damage
    # rather than work. Its prose says where the headline lives instead.
    model_of_record_window = settings.waves == WAVE_LABELS
    link_pair_required = own == "B" and not off_floor and model_of_record_window
    link_companion = (
        (
            LEVEL_BLENDING_PRIMARY_MODEL_ID
            if settings.score_mean_link == "three_choice_guessing_floor"
            else LEVEL_BLENDING_COMPANION_MODEL_ID
        )
        if link_pair_required
        else None
    )
    if not settings.group_by_time:
        focal_vector: str | None = None
        focal_index: int | None = None
        focal_term: str | None = None
    elif settings.arm_gap_reference == "t1":
        focal_vector, focal_index = "d_grp_time", POST_PHASE_LABELS.index("t2")
        focal_term = f"d_grp_time[{POST_PHASE_LABELS[focal_index]}]"
    else:
        focal_vector, focal_index = "b_grp_time", 1
        focal_term = "b_grp_time[1]"

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

    t1_referenced = settings.group_by_time and settings.arm_gap_reference == "t1"
    if t1_referenced:
        group_clause = (
            "the randomised group entered per timepoint and centred on the "
            "timepoint-1 arm gap (a pre-randomisation balance term plus the change "
            "in the arm gap at each later wave, #552)"
        )
        contrast_clause = (
            "the t2 change in the adjusted arm gap d_grp_time[t2], a "
            "difference-in-differences of levels that removes the covariate-adjusted "
            "chance imbalance carried by the t1 gap arm_gap_t1"
        )
    elif settings.group_by_time:
        group_clause = "the randomised group entered per timepoint"
        contrast_clause = (
            "the raw adjusted t2 arm gap b_grp_time[1] (the pre-#552 free "
            "parameterisation, retained as a comparator; it carries any "
            "covariate-adjusted chance imbalance present at t1)"
        )
    else:
        group_clause = "a single pooled group coefficient across all waves"
        contrast_clause = (
            "the pooled group coefficient beta_grp mixes post-crossover waves and "
            "is not a randomised contrast"
        )
    # A pooled group term has no randomised element at all (``focal_term is None``),
    # so the estimand and causal-status prose must not name a t2 contrast the fit
    # does not carry: the generated text used to read "The t2 randomised group
    # contrast -- none: the pooled ... -- as an items-scale average marginal effect"
    # (#584 lower-severity 6). Branch on the resolved focal term, the same switch the
    # summaries and the release gate use.
    scale_clause = (
        "on the probability of being off the floor (a risk difference read at the "
        "t2 rows)"
        if off_floor
        else "as an items-scale average marginal effect read at the t2 rows"
    )
    blending_window_clause = (
        " This outcome's published estimate is the link-paired four-wave headline "
        f"({LEVEL_BLENDING_PRIMARY_MODEL_ID} + {LEVEL_BLENDING_COMPANION_MODEL_ID}): "
        "phoneme blending is response-link sensitive, and this comparator carries "
        "the ordinary inverse-logit score mean alone, so it answers the analysis-"
        "window question and not the response-link one."
        if own == "B" and not off_floor and not model_of_record_window
        else ""
    )
    window_clause = (
        " Analysis window: the randomised t1 -> t2 window only, so no "
        "post-crossover observation enters the likelihood and the t2 change is "
        "identified without the longitudinal working model the four-wave fit uses "
        "(#584 decision 3). This is a comparator; the four-wave fit is the model of "
        f"record.{blending_window_clause}"
        if not model_of_record_window
        else ""
    )
    if off_floor:
        design = (
            "Per-wave off-floor levels model: a Bernoulli likelihood for whether the "
            f"child is above the outcome floor at each wave, with {group_clause}, the "
            "ability covariate, an optional group x ability term, and a non-centred "
            f"child random intercept.{window_clause}"
        )
    else:
        link_clause = (
            " The score mean is mapped onto [1/3, 1] by the three-choice guessing "
            "floor, because each phoneme-blending item has three response "
            "alternatives and an expected score cannot fall below chance (#584 "
            "decision 2)."
            if settings.score_mean_link == "three_choice_guessing_floor"
            else ""
        )
        design = (
            "Per-wave levels model: each wave's score is regressed (a Beta-Binomial "
            f"working likelihood) on {group_clause}, the ability covariate, an "
            "optional group x ability term, and a non-centred child random intercept "
            f"for the repeated observations.{link_clause}{window_clause}"
        )
    # The natural-scale target was open through #389 finding 1 and #584 finding 1;
    # it was settled on 2026-08-23 (notes/202608231800-level-factors-584-decisions.md)
    # and the plan now states it rather than flagging a review.
    review_clause = (
        " The card is the arm-free standardised marginal effect: every fitted t2 "
        "row is evaluated at its own arm-free profile -- the complete group "
        "contribution netted out, each row keeping its own age, ability, adjusters "
        "and fitted child intercept -- and only the focal contrast is added back, "
        "with the group x ability increment held at centred ability and reported "
        "separately (#584 decision 1)."
    )
    if focal_term is None:
        estimand = (
            f"No randomised contrast: {contrast_clause}. The pooled group "
            "coefficient is reported as an adjusted association, as are the "
            "ability and interaction terms, so no treatment-effect marginal is "
            "published for this plan."
        )
        causal_status = (
            "No coefficient in this fit is causal: the pooled group term averages "
            "the randomised t2 window with the post-crossover waves, and every "
            "ability and group x ability term is a latent-ability-confounded "
            "adjusted association."
        )
    else:
        estimand = (
            f"The t2 randomised group contrast -- {contrast_clause} -- "
            f"{scale_clause}. The other waves are post-crossover; ability and "
            f"interaction terms are adjusted associations.{review_clause}"
        )
        causal_status = (
            "Only the t2 group term is randomised (a contrast on the available-case "
            "t2 population); the other timepoints are post-crossover and every "
            "ability and group x ability term is a latent-ability-confounded "
            "adjusted association, never a causal effect."
            + (
                " The t1 arm gap is a pre-randomisation balance quantity, reported "
                "with its own prior and never as an effect."
                if t1_referenced
                else ""
            )
            + (
                " Because only the randomised window is fitted, no post-crossover "
                "observation informs the contrast through the shared balance term, "
                "child intercept, dispersion or group x ability term -- the "
                "dependence this comparator exists to quantify."
                if settings.waves != WAVE_LABELS
                else ""
            )
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
        score_mean_link=settings.score_mean_link,
        required_link_companion_model_id=link_companion,
        link_sensitivity_required_for_release=link_pair_required,
        waves=settings.waves,
        kappa_prior_family=None if off_floor else settings.kappa_prior_family,
        kappa_prior_sigma=None if off_floor else settings.kappa_prior_sigma,
        sigma_child_prior_sigma=settings.sigma_child_prior_sigma,
        arm_gap_reference=settings.arm_gap_reference,
        focal_vector=focal_vector,
        focal_index=focal_index,
        focal_term=focal_term,
        baseline_covariates=baseline_covariates,
        pre_covariates=pre_adj,
        post_covariates=post_adj,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
