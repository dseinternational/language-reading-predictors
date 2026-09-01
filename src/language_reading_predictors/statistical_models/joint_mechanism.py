# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for joint-mechanism models.

The family has two deliberately different designs: per-wave bivariate levels
fits with an observation-row residual correlation, and a phase-stacked ANCOVA
with a bivariate child intercept.  This module replaces the free-form
``ModelSpec.extra`` boundary for both designs and resolves their data, factory,
diagnostic and reporting contracts before a fit context is created or RLI data
are loaded (#394 pillar 4).

The migration does not change the fitted equations or scientific warrant.  The
letter-sound slopes, decoding-specificity contrast and conditional-slope ratio
remain adjusted associations rather than causal mechanism effects.

Both designs report ``rho_outcome``: it is the off-diagonal of whichever
dependence block the design carries. Only the *levels* design also reports the
conditional slope and its ratio to the marginal slope, because partialling the
held-fixed outcome is a same-row operation and the transition design's block
sits between children (2026-08-23 follow-up review, documentation gap 4).

The 2026-08-23 follow-up review (#591) also settled three interpretation
contracts this module now encodes in the generated recipe: the contrast is a
measurement-scale-dependent association rather than a decoding-mechanism test;
the transition design's estimand is an ANCOVA post-level association, not a
within-child change effect; and the matched comparators are related estimands
whose difference from the joint fit is not attributable to dependence alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.preprocessing import (
    split_covariates_by_wave,
)
from language_reading_predictors.statistical_models.new_child_kfold import KFoldPlan
from language_reading_predictors.statistical_models.new_child_predictive import (
    PREDICTION_TARGET_NEW_CHILD,
    PREDICTION_TARGETS,
    NewChildPlan,
)
from language_reading_predictors.statistical_models.settings_validation import (
    require_declared_booleans,
)
from language_reading_predictors.statistical_models.invariants import (
    require_value,
)

__all__ = [
    "JointMechanismModelSettings",
    "JointMechanismRunPlan",
    "declared_joint_mechanism_settings",
    "resolve_joint_mechanism_run_plan",
]


_DESIGNS = frozenset({"levels", "transition"})
_GLOBAL_KEYS = frozenset({"target_accept"})
_FAMILY_KEYS = frozenset(
    {
        "design",
        "outcome_symbols",
        "contrast",
        "confounder_symbols",
        "include_group",
        "covariates",
        "adjust_for",
        "predictor_slope_sigma",
        "prediction_target",
        "kfold_folds",
    }
)
_LEGACY_KEYS = _FAMILY_KEYS | _GLOBAL_KEYS


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    if any(not isinstance(item, str) or not item for item in out):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate symbols: {out!r}")
    return out


def _optional_positive_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number or None")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number or None")
    return out


@dataclass(frozen=True, slots=True)
class JointMechanismModelSettings:
    """Immutable declaration for one bivariate joint-mechanism model."""

    design: Literal["levels", "transition"] = "levels"
    outcome_symbols: tuple[str, ...] = ("W", "N")
    contrast: tuple[str, ...] = ("N", "W")
    confounder_symbols: tuple[str, ...] = ("G", "A")
    include_group: bool = True
    covariates: tuple[str, ...] = ()
    adjust_for: tuple[str, ...] = ()
    predictor_slope_sigma: float | None = None
    prediction_target: str = PREDICTION_TARGET_NEW_CHILD
    """Out-of-sample target the fit's cross-validation answers (#626).

    Both designs carry a child-level dependence block, so the child-aggregated
    PSIS-LOO alone is conditional on the held-out child's own residual; the declared
    target is what the matching validation integrates that residual out for."""
    kfold_folds: int = 5
    """Folds in the grouped child-level K-fold that backs up the integrated estimator.

    Both designs need it. Integrating the child's dependence block out leaves
    importance ratios this family cannot smooth — the levels design worst, since its
    residual *is* the child effect — so the ELPD was withheld on both registered fits
    until the refit route existed. Each fold is a refit, so this is the cost knob."""

    def __post_init__(self) -> None:
        require_declared_booleans(self)
        if self.prediction_target not in PREDICTION_TARGETS:
            raise ValueError(
                "joint-mechanism prediction_target must be one of "
                f"{', '.join(PREDICTION_TARGETS)}; got {self.prediction_target!r}"
            )
        if not isinstance(self.kfold_folds, int) or isinstance(self.kfold_folds, bool):
            raise TypeError("kfold_folds must be an int")
        if self.kfold_folds < 2:
            raise ValueError("kfold_folds must be at least 2")
        if self.design not in _DESIGNS:
            raise ValueError(
                f"design must be 'levels' or 'transition', got {self.design!r}"
            )
        for name in (
            "outcome_symbols",
            "contrast",
            "confounder_symbols",
            "covariates",
            "adjust_for",
        ):
            object.__setattr__(
                self,
                name,
                _tuple_of_strings(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "predictor_slope_sigma",
            _optional_positive_float(
                self.predictor_slope_sigma,
                name="predictor_slope_sigma",
            ),
        )

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> JointMechanismModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown joint-mechanism setting(s): "
                f"{', '.join(unknown)}. Declare JointMechanismModelSettings so "
                "misspellings fail fast."
            )
        return cls(
            design=extra.get("design", "levels"),
            outcome_symbols=extra.get("outcome_symbols", ("W", "N")),
            contrast=extra.get("contrast", ("N", "W")),
            confounder_symbols=extra.get("confounder_symbols", ("G", "A")),
            include_group=extra.get("include_group", True),
            covariates=extra.get("covariates", ()),
            adjust_for=extra.get("adjust_for", ()),
            predictor_slope_sigma=extra.get("predictor_slope_sigma"),
        )


@dataclass(frozen=True, slots=True)
class JointMechanismRunPlan:
    """Concrete, validated instructions for either joint-mechanism design."""

    model_id: str
    settings_source: str
    study_id: str
    design: Literal["levels", "transition"]
    mechanism_symbol: str
    outcome_symbols: tuple[str, str]
    contrast: tuple[str, str]
    confounder_symbols: tuple[str, ...]
    include_group: bool
    declared_adjustment: tuple[str, ...]
    active_adjustment: tuple[str, ...]
    predictor_slope_sigma: float | None
    phase_mode: Literal["levels", "all"]
    pre_covariates: tuple[str, ...]
    post_covariates: tuple[str, ...]
    likelihood: Literal["binomial", "beta_binomial"]
    joint_dependence: Literal["lkj_residual_within_wave", "lkj_child_intercept"]
    observation_node: str
    compute_loo: bool
    loo_unit: str
    prediction_target: str
    kfold_folds: int
    min_wave_rows: int | None
    min_wave_outcome_rows: int | None
    min_wave_overlap_rows: int | None
    matched_comparators: tuple[str, str]
    comparator_equivalence: str
    design_description: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def with_active_adjustment(
        self, active_adjustment: tuple[str, ...]
    ) -> JointMechanismRunPlan:
        """Record constant requested terms removed on the fitted rows."""
        unknown = sorted(set(active_adjustment) - set(self.declared_adjustment))
        if unknown:
            raise ValueError(
                f"active joint-mechanism adjustment was not declared: {unknown!r}"
            )
        return replace(self, active_adjustment=active_adjustment)

    @property
    def fits_group_nuisance(self) -> bool:
        """Whether the model includes its design-specific group term."""
        return self.include_group or "G" in self.confounder_symbols

    def new_child_plan(self) -> NewChildPlan:
        """The declared out-of-sample target, as the validation engine consumes it.

        The dependence block sits on the observation row in the ``levels`` design
        (one row per child, so the residual *is* the child effect) and on the child
        in ``transition``; either way the non-centred ``*_z`` offsets are the free
        variable a new child gets a fresh draw of.
        """
        latent = "u_resid_z" if self.design == "levels" else "u_child_z"
        # In ``levels`` one row *is* one child. In ``transition`` the three rows sit
        # inside the child, so both dimensions are declared: an effect on either would
        # have to be re-drawn for an unseen child, and naming them is what makes
        # leaving one undeclared fail rather than pass silently.
        dims = ("obs_id",) if self.design == "levels" else ("child", "obs_id")
        return NewChildPlan(
            prediction_target=self.prediction_target,
            child_dims=dims,
            latent_vars=(latent,),
            observed_nodes=(self.observation_node,),
        )

    def kfold_plan(self) -> KFoldPlan:
        """The grouped child-level K-fold that estimates the declared target."""
        return KFoldPlan(n_folds=self.kfold_folds, stratify=True)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan."""
        kwargs: dict[str, Any] = {
            "phase_mode": self.phase_mode,
            "outcomes": (*self.outcome_symbols, self.mechanism_symbol),
        }
        if self.design == "levels":
            kwargs["baseline_covariates"] = self.declared_adjustment
        else:
            kwargs["covariates"] = self.pre_covariates
            kwargs["post_covariates"] = self.post_covariates
            # Only the two outcome baselines are model terms; the default
            # ``pre_required`` would also demand the mechanism's period-start
            # score, which the model never uses — an undeclared row filter
            # (2026-08-21 joint-mechanism review).
            kwargs["pre_required"] = self.outcome_symbols
        return kwargs

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_joint_mechanism_model`` from the same plan."""
        kwargs: dict[str, Any] = {
            "design": self.design,
            "mechanism_symbol": self.mechanism_symbol,
            "outcome_symbols": self.outcome_symbols,
            "contrast": self.contrast,
            "adjust_for": self.active_adjustment,
            "confounder_symbols": self.confounder_symbols,
            "include_group": self.include_group,
        }
        if self.design == "levels":
            kwargs["predictor_slope_sigma"] = require_value(
                self.predictor_slope_sigma,
                "predictor_slope_sigma (the levels design's slope prior)",
            )
        return kwargs

    def diagnostic_vars(self, available_names: set[str]) -> list[str]:
        """Reported parameters present in a built model, in stable gate order."""
        names = ["alpha", "beta_mech", "delta_ls_decoding"]
        if self.fits_group_nuisance:
            names.append(
                "beta_group_nuisance" if self.design == "levels" else "beta_G"
            )
        if "A" in self.confounder_symbols:
            names.append("gamma_A")
        names.extend(f"gamma_{name}" for name in self.active_adjustment)
        if self.design == "levels":
            names.extend(
                [
                    "sigma_u_resid",
                    "rho_outcome",
                    "beta_mech_focal_given_held",
                    "share_retained",
                ]
            )
        else:
            names.extend(
                [
                    "gamma_own",
                    "alpha_phase",
                    "kappa",
                    "sigma_u_child",
                    "rho_outcome",
                ]
            )
        return [name for name in names if name in available_names]

    def psense_vars(self, available_names: set[str]) -> list[str]:
        """Stable parameters suitable for power-scaling sensitivity."""
        candidates = ["beta_mech", "delta_ls_decoding", "rho_outcome"]
        if self.design == "levels":
            candidates.append("beta_mech_focal_given_held")
        return [name for name in candidates if name in available_names]

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language account generated from the validated run plan."""
        adjusters = ", ".join(self.declared_adjustment) or "none"
        return (
            "Note: Generated from the validated joint-mechanism run plan; template "
            "drafted by a LLM-based AI tool (Codex/GPT-5) and substantially edited "
            "by a LLM-based AI tool (Claude Code/Opus 5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design_description}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"The standardised `{self.mechanism_symbol}` exposure is fitted jointly "
            f"against `{self.outcome_symbols[0]}` and "
            f"`{self.outcome_symbols[1]}`. The reported contrast is "
            f"`beta({self.contrast[0]}) - beta({self.contrast[1]})`. Declared "
            f"adjustment terms: {adjusters}. Confounder flags: "
            f"{', '.join(self.confounder_symbols) or 'none'}. Likelihood: "
            f"`{self.likelihood}`; dependence block: `{self.joint_dependence}`.\n\n"
            "## Matched comparators\n\n"
            f"{', '.join(f'`{m}`' for m in self.matched_comparators)}. "
            f"{self.comparator_equivalence}\n\n"
            "## Uncertainty and checks\n\n"
            f"{self._loo_sentence()} Interpret the posterior only after every "
            "published fit passes the zero-divergence convergence gate, predictive "
            "checks and power-scaling sensitivity diagnostics. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )

    def _loo_sentence(self) -> str:
        """The recipe's LOO sentence, honest about the levels design's saturation."""
        if self.compute_loo:
            return (
                f"The observation node is `{self.observation_node}` and PSIS-LOO "
                f"uses the `{self.loo_unit}` unit."
            )
        return (
            f"The observation node is `{self.observation_node}`. PSIS-LOO is not "
            "computed for this design: with one bivariate latent residual per "
            "child over at most two observed cells, importance-sampled "
            "leave-one-child-out is conditional on a saturated per-child latent "
            "and fails its Pareto-k diagnostics en masse, so no `elpd` is "
            "reported. Predictive assessment uses the conditional and new-child "
            "(marginal) posterior-predictive checks instead."
        )


def declared_joint_mechanism_settings(
    spec: ModelSpec,
) -> tuple[JointMechanismModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: joint-mechanism settings cannot be split between "
                f"model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, JointMechanismModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='joint_mechanism' requires "
                f"JointMechanismModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        JointMechanismModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
        ),
        "legacy_extra",
    )


def resolve_joint_mechanism_run_plan(spec: ModelSpec) -> JointMechanismRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "joint_mechanism":
        raise ValueError(
            f"{spec.model_id}: expected kind 'joint_mechanism', got {spec.kind!r}"
        )
    if spec.study_id != "rli":
        raise ValueError(
            f"{spec.model_id}: joint_mechanism requires study_id='rli', got "
            f"{spec.study_id!r}"
        )
    mechanism_symbol = spec.mechanism_symbol or "L"
    if not isinstance(mechanism_symbol, str) or not mechanism_symbol:
        raise TypeError("joint_mechanism mechanism_symbol must be a non-empty string")

    settings, source = declared_joint_mechanism_settings(spec)
    if len(settings.outcome_symbols) != 2:
        raise ValueError("joint_mechanism requires exactly two outcome_symbols")
    outcome_symbols = settings.outcome_symbols
    if mechanism_symbol in outcome_symbols:
        raise ValueError("mechanism_symbol must differ from both outcome_symbols")
    if len(settings.contrast) != 2 or set(settings.contrast) != set(outcome_symbols):
        raise ValueError(
            "contrast must contain the two outcome_symbols exactly once"
        )
    unknown_confounders = sorted(set(settings.confounder_symbols) - {"G", "A"})
    if unknown_confounders:
        raise ValueError(
            "confounder_symbols supports only the group and age flags G/A; got "
            f"{unknown_confounders!r}"
        )

    if settings.design == "levels":
        if settings.adjust_for:
            raise ValueError("adjust_for is transition-only; use covariates for levels")
        adjustment = settings.covariates
        slope_sigma = settings.predictor_slope_sigma or 0.3
        pre_covariates: tuple[str, ...] = ()
        post_covariates: tuple[str, ...] = ()
        phase_mode: Literal["levels", "all"] = "levels"
        likelihood: Literal["binomial", "beta_binomial"] = "binomial"
        dependence: Literal[
            "lkj_residual_within_wave", "lkj_child_intercept"
        ] = "lkj_residual_within_wave"
        min_wave_rows = 10
        # A wave needs enough rows on *each* outcome, and enough jointly observed
        # pairs, before its residual correlation and conditional slope mean
        # anything: the earlier rule only counted rows with the exposure and *at
        # least one* outcome, so a wave observing one outcome twice could in
        # principle be fitted and publish a prior-dominated rho (2026-08-23
        # follow-up review, robustness gap 1). Prespecified, not data-adaptive.
        min_wave_outcome_rows = 10
        min_wave_overlap_rows = 10
        # One bivariate latent residual per child over at most two observed cells:
        # PSIS-LOO is conditional on a saturated per-child latent (the fitted
        # reporting run showed p_loo > n and 48/53 Pareto-k above 0.7), so it is
        # not computed at all rather than published as a check that cannot work
        # (2026-08-21 joint-mechanism review, finding 2). The same saturation is
        # why the pipeline publishes the new-child *marginal* predictive check.
        compute_loo = False
        comparators = ("lrp-rli-ca-010", "lrp-rli-ca-011")
        comparator_equivalence = (
            "Related sensitivity estimands, NOT like-for-like replacements. This "
            "design is a bivariate logistic-normal Binomial model conditioning on "
            "the *latent* held-fixed outcome; `lrp-rli-ca-010` / `lrp-rli-ca-011` "
            "are separate Beta-Binomial fits and `lrp-rli-ca-011` conditions on an "
            "*observed* transformed nonword count with missing predictor values "
            "mean-imputed. Common rows and a common exposure scale would not make "
            "them nested, so the difference between the two conditional slopes is "
            "not attributable to cross-outcome dependence alone (2026-08-23 "
            "follow-up review, finding 2)."
        )
        design_description = (
            "Separate cross-sectional bivariate fits at each RLI wave. Both bounded "
            "outcomes share the same standardised mechanism exposure and an LKJ "
            "observation-row residual block. Every published wave is fitted, "
            "convergence-scanned over its reported deterministics, given the "
            "informative new-child predictive check and power-scaling sensitivity, "
            "and persisted as a named trace; one wave additionally carries the "
            "fit-level artefacts, chosen by row count purely as an operational "
            "artefact-hosting rule."
        )
        estimand = (
            "Per-wave mechanism slopes for both outcomes, their within-model "
            "decoding-specificity difference, the residual outcome correlation, and "
            "the conditional-to-marginal slope ratio obtained after partialling the "
            "other latent outcome. The ratio is a slope ratio, not a bounded "
            "pathway share, and no wave is selected as a headline after seeing its "
            "posterior."
        )
        population = (
            "At each of four waves, archived RLI children with the mechanism score "
            "and at least one outcome observed, plus all retained baseline trait "
            "covariates. A wave is named and skipped rather than fitted unless it "
            "has at least ten usable children, at least ten observations on each "
            "outcome and at least ten children observing both outcomes. The "
            "exposure is standardised within each wave, so one standard deviation "
            "denotes a wave-specific raw letter-sound increment."
        )
    else:
        if settings.covariates:
            raise ValueError("covariates is levels-only; use adjust_for for transition")
        if settings.predictor_slope_sigma is not None:
            raise ValueError("predictor_slope_sigma is levels-only")
        adjustment = settings.adjust_for
        slope_sigma = None
        pre_covariates, post_covariates = split_covariates_by_wave(adjustment)
        phase_mode = "all"
        likelihood = "beta_binomial"
        dependence = "lkj_child_intercept"
        min_wave_rows = None
        min_wave_outcome_rows = None
        min_wave_overlap_rows = None
        # Genuine leave-one-child-out: the factory registers ``loo_child_idx``
        # (cells -> child), so the shared aggregation sums each child's cells
        # across all three transitions before importance sampling.
        compute_loo = True
        comparators = ("lrp-rli-mech-096", "lrp-rli-mech-101")
        comparator_equivalence = (
            "Same parameterisation, but NOT the same fitted sample or exposure "
            "scale. This model requires both outcome baselines on every retained "
            "transition and standardises the letter-sound logit once over that "
            "joint union; each matched mechanism model requires only its own "
            "baseline, filters to its own outcome's rows and re-standardises the "
            "exposure on those rows. One standard deviation therefore denotes a "
            "different raw letter-sound increment in each fit, and the word-reading "
            "marginal retains rows this model excludes, so the gap between the "
            "joint contrast and the paired-marginal sensitivity cannot be "
            "attributed to cross-outcome covariance alone (2026-08-23 follow-up "
            "review, finding 2). Each fit records its own fitted rows and exposure "
            "scaler, and `scripts/compare_statistical_models.py` publishes that "
            "reconciliation beside the two contrast rows with an explicit "
            "`comparable` verdict."
        )
        design_description = (
            "One phase-stacked bivariate ANCOVA over the three RLI transitions. Each "
            "outcome retains its own baseline and phase intercepts, while an LKJ "
            "bivariate child intercept represents between-child outcome dependence."
        )
        estimand = (
            "The two adjusted mechanism slopes for each outcome's post-level "
            "conditional on that outcome's own baseline — an ANCOVA-parameterised "
            "association, not a within-child change effect — and their within-model "
            "decoding-specificity difference on the same parameterisation as the "
            "two matched mechanism models. The common slope pools between-child and "
            "within-child information; the child random intercept does not remove "
            "stable general-ability confounding."
        )
        population = (
            "Available RLI transition rows with the mechanism exposure, at least one "
            "outcome, both outcome baselines and all retained requested covariates."
        )

    return JointMechanismRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        design=settings.design,
        mechanism_symbol=mechanism_symbol,
        outcome_symbols=(outcome_symbols[0], outcome_symbols[1]),
        contrast=(settings.contrast[0], settings.contrast[1]),
        confounder_symbols=settings.confounder_symbols,
        include_group=settings.include_group,
        declared_adjustment=adjustment,
        active_adjustment=adjustment,
        predictor_slope_sigma=slope_sigma,
        phase_mode=phase_mode,
        pre_covariates=pre_covariates,
        post_covariates=post_covariates,
        likelihood=likelihood,
        joint_dependence=dependence,
        observation_node="y_post",
        compute_loo=compute_loo,
        loo_unit="child",
        prediction_target=settings.prediction_target,
        kfold_folds=settings.kfold_folds,
        min_wave_rows=min_wave_rows,
        min_wave_outcome_rows=min_wave_outcome_rows,
        min_wave_overlap_rows=min_wave_overlap_rows,
        matched_comparators=comparators,
        comparator_equivalence=comparator_equivalence,
        design_description=design_description,
        estimand=estimand,
        causal_status=(
            "Adjusted association only. Randomised group is a nuisance term here; "
            "unobserved general ability can still confound the mechanism-outcome "
            "slopes, and the dependence block does not repair that backdoor path. "
            "The decoding-specificity contrast is measurement-scale dependent: with "
            "both outcomes and the exposure loading on one latent general ability, "
            "the two latent-scale slopes remain proportional to their loadings, so "
            "the difference is proportional to the loading difference even with no "
            "causal letter-sound route. Different item counts, link discrimination, "
            "floor compression and non-classical measurement error can each produce "
            "a non-zero contrast, and the model imposes no cross-instrument "
            "measurement invariance. A positive contrast is consistent with the "
            "proposed decoding account; it neither rejects a common-factor "
            "explanation nor identifies a mechanism (2026-08-23 follow-up review, "
            "finding 3)."
        ),
        analysis_population=population,
        missing_data_assumption=(
            "The focal mechanism is never imputed. A row enters when that exposure "
            "and at least one outcome are observed, subject to the design-specific "
            "baseline and covariate requirements. Outcome-specific missing cells are "
            "masked, under ignorable missingness conditional on fitted terms — a "
            "conditional missing-at-random assumption, alongside filled covariates "
            "carrying explicit missingness indicators. No missing-not-at-random or "
            "complete-case joint-model sensitivity is registered for this family: "
            "the deferral is deliberate and its consequence is that the published "
            "slopes, contrast and ratio are conditional on that assumption "
            "(2026-08-23 follow-up review, robustness gap 7)."
        ),
    )
