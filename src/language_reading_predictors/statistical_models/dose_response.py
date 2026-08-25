# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for dose-response models.

The family estimates observational associations between intervention-session dose
and bounded skill outcomes.  Resolution happens before an output transaction is
opened or RLI data are loaded, while preserving the existing fitted equations and
artefacts (#394 pillar 4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.likelihood import (
    SCORE_MEAN_LINKS,
    ScoreMeanLink,
)
from language_reading_predictors.statistical_models.measures import MEASURES

#: The registered phoneme-blending response-link pair for this family (#619, under
#: the #608 policy). ``lrp-rli-dose-084`` fits the ordinary Beta-Binomial
#: inverse-logit score mean; ``lrp-rli-dose-384`` fits the same model with the mean
#: mapped onto [1/3, 1]. Neither may be released without the other.
DOSE_BLENDING_PRIMARY_MODEL_ID = "lrp-rli-dose-084"
DOSE_BLENDING_COMPANION_MODEL_ID = "lrp-rli-dose-384"

__all__ = [
    "ABILITY_BASELINE_WAVES",
    "DoseResponseModelSettings",
    "DoseResponseRunPlan",
    "declared_dose_response_settings",
    "resolve_dose_response_run_plan",
]


#: Which wave an ``ability_adjust_symbols`` covariate is read from.
#:
#: ``"t1"`` broadcasts each child's verified pre-randomisation value across all three
#: transitions — the only reading under which the fit is the *baseline*-ability
#: sensitivity it is published as. ``"transition_start"`` is the pre-#587 behaviour,
#: which silently used t2 skills in period 2 and t3 skills in period 3: values that are
#: themselves downstream of earlier intervention and dose, so conditioning on them
#: adjusts a treatment-affected time-varying covariate (Robins, Hernán & Brumback 2000)
#: rather than blocking the latent-ability back door. It is retained only as an
#: explicitly labelled comparator and must never be presented as a baseline sensitivity
#: (#587 finding 1).
ABILITY_BASELINE_WAVES = ("t1", "transition_start")


_FAMILY_KEYS = frozenset(
    {
        "adjust_baseline_symbol",
        "dose_covariate",
        "dose_stage_covariate",
        "period_varying_dose",
        "use_subject_random_intercept",
        "ability_adjust_symbols",
        "ability_baseline_wave",
        "decompose_between_within",
        "outcomes",
        "adjust_group",
        "adjust_age",
        "score_mean_link",
    }
)
_GLOBAL_KEYS = frozenset({"target_accept"})
_LEGACY_KEYS = _FAMILY_KEYS | _GLOBAL_KEYS


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name=name)


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        _string(item, name=name)
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate symbols: {out!r}")
    return out


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class DoseResponseModelSettings:
    """Immutable declaration for one dose-response model."""

    adjust_baseline_symbol: str = "W"
    dose_covariate: str = "attend"
    dose_stage_covariate: str | None = None
    period_varying_dose: bool = True
    use_subject_random_intercept: bool = True
    ability_adjust_symbols: tuple[str, ...] = ()
    ability_baseline_wave: str = "t1"
    decompose_between_within: bool = True
    outcomes: tuple[str, ...] = ()
    adjust_group: bool = True
    adjust_age: bool = True
    #: Phoneme-blending response link (#619, under the #608 policy). ``"logit"`` is
    #: the ordinary Beta-Binomial inverse-logit score mean;
    #: ``"three_choice_guessing_floor"`` maps it onto [1/3, 1] for the ten
    #: three-alternative forced-choice blending items, whose expected score cannot
    #: fall below chance. B only, and released only beside its paired opposite-link
    #: fit. The family's focal estimand is the natural-scale treated-row dose
    #: marginal, so it is link-dependent in exactly the way a treatment contrast is.
    score_mean_link: ScoreMeanLink = "logit"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adjust_baseline_symbol",
            _string(self.adjust_baseline_symbol, name="adjust_baseline_symbol"),
        )
        object.__setattr__(
            self,
            "dose_covariate",
            _string(self.dose_covariate, name="dose_covariate"),
        )
        object.__setattr__(
            self,
            "dose_stage_covariate",
            _optional_string(
                self.dose_stage_covariate,
                name="dose_stage_covariate",
            ),
        )
        object.__setattr__(
            self,
            "ability_adjust_symbols",
            _tuple_of_strings(
                self.ability_adjust_symbols,
                name="ability_adjust_symbols",
            ),
        )
        object.__setattr__(
            self,
            "outcomes",
            _tuple_of_strings(self.outcomes, name="outcomes"),
        )
        wave = _string(self.ability_baseline_wave, name="ability_baseline_wave")
        if wave not in ABILITY_BASELINE_WAVES:
            raise ValueError(
                "ability_baseline_wave must be one of "
                f"{ABILITY_BASELINE_WAVES!r}, got {wave!r}"
            )
        if self.score_mean_link not in SCORE_MEAN_LINKS:
            raise ValueError(
                f"score_mean_link must be one of {SCORE_MEAN_LINKS}, "
                f"got {self.score_mean_link!r}"
            )
        object.__setattr__(self, "ability_baseline_wave", wave)
        for name in (
            "period_varying_dose",
            "use_subject_random_intercept",
            "decompose_between_within",
            "adjust_group",
            "adjust_age",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name=name))

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
    ) -> DoseResponseModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown dose-response setting(s): "
                f"{', '.join(unknown)}. Declare DoseResponseModelSettings so "
                "misspellings fail fast."
            )
        return cls(
            adjust_baseline_symbol=extra.get("adjust_baseline_symbol", "W"),
            dose_covariate=extra.get("dose_covariate", "attend"),
            dose_stage_covariate=extra.get("dose_stage_covariate"),
            period_varying_dose=extra.get("period_varying_dose", True),
            use_subject_random_intercept=extra.get(
                "use_subject_random_intercept",
                True,
            ),
            ability_adjust_symbols=extra.get("ability_adjust_symbols", ()),
            ability_baseline_wave=extra.get("ability_baseline_wave", "t1"),
            decompose_between_within=extra.get("decompose_between_within", True),
            score_mean_link=extra.get("score_mean_link", "logit"),
            outcomes=extra.get("outcomes", ()),
            adjust_group=extra.get("adjust_group", True),
            adjust_age=extra.get("adjust_age", True),
        )


@dataclass(frozen=True, slots=True)
class DoseResponseRunPlan:
    """Concrete, validated instructions for a complete dose-response fit."""

    model_id: str
    settings_source: str
    study_id: str
    outcome_symbol: str
    adjust_baseline_symbol: str
    dose_covariate: str
    dose_stage_covariate: str | None
    period_varying_dose: bool
    use_subject_random_intercept: bool
    ability_adjust_symbols: tuple[str, ...]
    ability_baseline_wave: str
    decompose_between_within: bool
    outcomes: tuple[str, ...]
    adjust_group: bool
    adjust_age: bool
    # Phoneme-blending response link and its release pairing (#619).
    score_mean_link: str
    required_link_companion_model_id: str | None
    link_sensitivity_required_for_release: bool
    phase_mode: str
    loader_covariates: tuple[str, ...]
    observation_node: str
    compute_loo: bool
    loo_unit: str
    loo_note: str
    focal_term: str
    exposure: str
    dose_margin: str
    dose_contrast: str
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan."""
        return {
            "phase_mode": self.phase_mode,
            "outcomes": self.outcomes,
            "covariates": self.loader_covariates,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_dose_response_model``."""
        return {
            "outcome_symbol": self.outcome_symbol,
            "adjust_baseline_symbol": self.adjust_baseline_symbol,
            "dose_covariate": self.dose_covariate,
            "dose_stage_covariate": self.dose_stage_covariate,
            "period_varying_dose": self.period_varying_dose,
            "use_subject_random_intercept": self.use_subject_random_intercept,
            "adjust_group": self.adjust_group,
            "adjust_age": self.adjust_age,
            "ability_adjust_symbols": self.ability_adjust_symbols,
            "ability_baseline_wave": self.ability_baseline_wave,
            "decompose_between_within": self.decompose_between_within,
            "score_mean_link": self.score_mean_link,
        }

    def coefficient_meanings(self) -> dict[str, str]:
        """One unambiguous sentence per fitted coefficient (#587 finding 2).

        The audit's acceptance criterion is that treatment presence, intensity,
        assigned arm/history and the between/within dose split each have a meaning a
        reader cannot confuse with another. They are recorded here, resolved from the
        declared settings, so ``config.json`` and the report carry the same statement
        and neither can drift from the fitted equation.
        """
        meanings: dict[str, str] = {
            "alpha": "Reference-phase intercept (period 1, untreated, at every covariate mean).",
            "alpha_phase": (
                "Period intercept deviations from period 1, reference-coded so period "
                "1 is exactly zero and the four-column intercept design has full rank."
            ),
            "theta_treated": (
                "Extensive margin: being on the intervention during a period versus "
                "not, at the treated-mean session count. In period 1 this contrast is "
                "randomised (every immediate-arm child attended, every waitlist child "
                "attended zero sessions), so it is the only term here identified by "
                "randomisation; it is otherwise informed only by the few later "
                "zero-session rows."
            ),
            "gamma_own": "Autoregression / regression-to-the-mean control on the period's own baseline logit.",
        }
        if self.adjust_group:
            meanings["beta_arm_late"] = (
                "Assigned-arm contrast in the post-crossover periods only (periods 2 "
                "and 3, where both arms are on the intervention), so it reads as "
                "intervention order and treatment history, never as a treatment "
                "effect. Period 1's arm difference is carried by `theta_treated`, "
                "with which it would otherwise be exactly collinear."
            )
        if self.adjust_age:
            meanings["gamma_A"] = "Linear age at the period start — a precision / maturation covariate."
        if self.use_subject_random_intercept:
            meanings["sigma_child"] = (
                "Between-child SD of the random intercept. It partially pools stable "
                "child differences; it does not make the dose slope a within-child "
                "quantity, which is why the exposure is split explicitly."
            )
        intensity = (
            "Intensive margin among on-intervention rows only, per 1 SD of "
            "treated-row sessions; untreated rows contribute exactly zero to every "
            "dose term, so no dose coefficient absorbs the extensive margin."
        )
        if self.decompose_between_within:
            meanings["beta_dose_between"] = (
                "Between-child intensity: a child whose study-average attendance is 1 "
                "SD higher than another child's. " + intensity
            )
            slope_role = (
                "the **within-child** component — the same child in a period when "
                "they attended more than their own study average"
            )
        else:
            slope_role = (
                "a precision-weighted **blend** of the between-child and within-child "
                "associations, because the exposure is not split"
            )
        if self.period_varying_dose:
            meanings["mu_dose"] = (
                f"Overall (partially pooled) dose slope across the three periods; "
                f"{slope_role}. " + intensity
            )
            meanings["sigma_dose"] = (
                "Between-period SD of the period-specific dose slopes. Its prior "
                "partially pools only three slopes, so read it beside the pooled "
                "comparator rather than as evidence about period variation on its own."
            )
            meanings["beta_dose_phase"] = (
                f"Period-specific dose slopes; each is {slope_role}. " + intensity
            )
        else:
            meanings["beta_dose"] = (
                f"Pooled dose slope; {slope_role}. " + intensity
            )
        for symbol in self.ability_adjust_symbols:
            wave = "verified pre-randomisation t1" if self.ability_baseline_wave == "t1" else "transition-start"
            meanings[f"gamma_{symbol}_pre"] = (
                f"Adjusted association with the {wave} logit of {symbol} — a baseline "
                "ability proxy, never an effect."
            )
        if self.dose_stage_covariate is not None:
            meanings["gamma_dose_stage"] = (
                "Flagged collider sensitivity on cumulative prior dose; reopens the "
                "latent-ability back door by construction (#269)."
            )
        meanings["kappa"] = "Beta-Binomial concentration (overdispersion) of the post-count."
        return meanings

    def diagnostic_vars(self) -> list[str]:
        """Variables scanned by summaries and the convergence gate."""
        names = ["alpha", "gamma_own", "kappa", "theta_treated"]
        if self.use_subject_random_intercept:
            names.append("sigma_child")
        if self.adjust_group:
            names.append("beta_arm_late")
        if self.adjust_age:
            names.append("gamma_A")
        if self.decompose_between_within:
            names.append("beta_dose_between")
        if self.period_varying_dose:
            names.extend(["mu_dose", "sigma_dose", "beta_dose_phase"])
        else:
            names.append("beta_dose")
        if self.dose_stage_covariate is not None:
            names.append("gamma_dose_stage")
        names.extend(f"gamma_{symbol}_pre" for symbol in self.ability_adjust_symbols)
        return names

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language recipe generated from the validated plan."""
        outcomes = ", ".join(self.outcomes)
        ability = (
            ", ".join(self.ability_adjust_symbols)
            if self.ability_adjust_symbols
            else "none"
        )
        return (
            "Note: Generated from the validated dose-response run plan; template "
            "drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Loaded outcomes: {outcomes}. "
            f"Own baseline: `{self.adjust_baseline_symbol}`. Exposure: "
            f"`{self.dose_covariate}` (period-varying: "
            f"{self.period_varying_dose}; between/within split: "
            f"{self.decompose_between_within}). Stage-dose covariate: "
            f"{self.dose_stage_covariate or 'none'}. Ability adjustments: "
            f"{ability} (read at `{self.ability_baseline_wave}`). Group adjustment: "
            f"{self.adjust_group}. Age adjustment: {self.adjust_age}. Child random "
            f"intercept: {self.use_subject_random_intercept}.\n\n"
            "Every fitted coefficient and its meaning:\n\n"
            + "".join(
                f"- `{name}` — {meaning}\n"
                for name, meaning in self.coefficient_meanings().items()
            )
            + "\n## Uncertainty and checks\n\n"
            f"The observation node is `{self.observation_node}` and PSIS-LOO uses "
            f"the `{self.loo_unit}` unit. {self.loo_note}\n\n"
            "Interpret the posterior only after the zero-divergence convergence gate "
            "and the posterior-predictive checks pass; both are enforced "
            "automatically, and a failed gate suppresses the scientific tables. "
            "Power-scaling prior sensitivity is **reported, not enforced**: this is an "
            "observational family, so `release.py` does not gate publication on it. "
            "Read `psense_summary.csv` directly — the between-period slope scale is "
            "prior-informed by construction, and a flagged focal slope means the "
            "reported association is substantially prior-driven. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_dose_response_settings(
    spec: ModelSpec,
) -> tuple[DoseResponseModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: dose-response settings cannot be split between "
                f"model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, DoseResponseModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='dose_response' requires "
                f"DoseResponseModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        DoseResponseModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
        ),
        "legacy_extra",
    )


def resolve_dose_response_run_plan(spec: ModelSpec) -> DoseResponseRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "dose_response":
        raise ValueError(
            f"{spec.model_id}: expected kind 'dose_response', got {spec.kind!r}"
        )
    if spec.study_id != "rli":
        raise ValueError(
            f"{spec.model_id}: dose_response requires study_id='rli', got "
            f"{spec.study_id!r}"
        )
    if not spec.outcome_symbol:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol is required for dose_response"
        )

    settings, source = declared_dose_response_settings(spec)
    outcome = spec.outcome_symbol
    outcomes = settings.outcomes or (outcome,)
    if settings.score_mean_link == "three_choice_guessing_floor" and outcome != "B":
        raise ValueError(
            f"{spec.model_id}: three_choice_guessing_floor is only valid for "
            f"phoneme blending (B), got {outcome!r}"
        )

    # The mandatory phoneme-blending link pairing (#619, under the #608 policy).
    # This family has no variant role, so every registered B dose fit is a model of
    # record and the pairing binds whenever the outcome is B. The family's focal
    # estimand is the natural-scale treated-row dose marginal -- a quantity published
    # in items -- so it inherits the link exactly as a treatment contrast does, which
    # is the case #608 used to reject exempting observational families.
    link_pair_required = outcome == "B"
    link_companion = (
        (
            DOSE_BLENDING_PRIMARY_MODEL_ID
            if settings.score_mean_link == "three_choice_guessing_floor"
            else DOSE_BLENDING_COMPANION_MODEL_ID
        )
        if link_pair_required
        else None
    )
    # Reject symbols no measure defines *before* any I/O (#587 finding 12). The
    # previous resolver accepted an arbitrary string and only failed inside the
    # loader, after the output directory had been reset and the data read — exactly
    # the ordering the family contract says it prevents.
    unknown = sorted(
        {
            symbol
            for symbol in (
                outcome,
                settings.adjust_baseline_symbol,
                *settings.ability_adjust_symbols,
                *outcomes,
            )
            if symbol not in MEASURES
        }
    )
    if unknown:
        raise ValueError(
            f"{spec.model_id}: unknown measure symbol(s) {unknown!r}; valid symbols "
            f"are {sorted(MEASURES)!r}"
        )
    # An ability adjuster that repeats the own baseline would put two coefficients on
    # one identical column — an exact collinearity the proper priors would hide rather
    # than surface (#587 finding 12).
    duplicated = sorted(
        set(settings.ability_adjust_symbols) & {settings.adjust_baseline_symbol}
    )
    if duplicated:
        raise ValueError(
            f"{spec.model_id}: ability_adjust_symbols {duplicated!r} duplicate "
            f"adjust_baseline_symbol {settings.adjust_baseline_symbol!r}; the factory "
            "would fit gamma_own and gamma_<symbol>_pre on the identical predictor"
        )
    required = {
        outcome,
        settings.adjust_baseline_symbol,
        *settings.ability_adjust_symbols,
    }
    missing = sorted(required - set(outcomes))
    if missing:
        raise ValueError(
            f"{spec.model_id}: outcomes must load every fitted outcome/baseline/"
            f"ability symbol; missing {missing!r} from {outcomes!r}"
        )
    if settings.dose_stage_covariate == settings.dose_covariate:
        raise ValueError(
            f"{spec.model_id}: dose_stage_covariate must differ from dose_covariate"
        )
    if settings.ability_baseline_wave == "transition_start" and not settings.ability_adjust_symbols:
        raise ValueError(
            f"{spec.model_id}: ability_baseline_wave='transition_start' is a labelled "
            "comparator for a fit that has ability adjusters; it means nothing without "
            "ability_adjust_symbols"
        )
    loader_covariates = tuple(
        value
        for value in (settings.dose_covariate, settings.dose_stage_covariate)
        if value is not None
    )
    focal = "mu_dose" if settings.period_varying_dose else "beta_dose"

    return DoseResponseRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        outcome_symbol=outcome,
        adjust_baseline_symbol=settings.adjust_baseline_symbol,
        dose_covariate=settings.dose_covariate,
        dose_stage_covariate=settings.dose_stage_covariate,
        period_varying_dose=settings.period_varying_dose,
        use_subject_random_intercept=settings.use_subject_random_intercept,
        ability_adjust_symbols=settings.ability_adjust_symbols,
        ability_baseline_wave=settings.ability_baseline_wave,
        decompose_between_within=settings.decompose_between_within,
        outcomes=outcomes,
        adjust_group=settings.adjust_group,
        adjust_age=settings.adjust_age,
        score_mean_link=settings.score_mean_link,
        required_link_companion_model_id=link_companion,
        link_sensitivity_required_for_release=link_pair_required,
        phase_mode="all",
        loader_covariates=loader_covariates,
        observation_node="y_post",
        compute_loo=True,
        # Whole-child, not row-level (#587 finding 4). A transition row's own baseline
        # IS the previous transition's fitted outcome — every period-2 row and all but
        # one period-3 row — so dropping a single row's likelihood factor leaves that
        # held-out score in the next row's design matrix. Holding out the whole child
        # removes it, and the family's small child random-intercept SD keeps the PSIS
        # approximation reliable.
        loo_unit="child",
        loo_note=(
            "Leave-one-child-out PSIS over the child-summed pointwise log likelihood. "
            "The unit is a new child, not a future row: the score answers 'how well "
            "does this model predict a child it has not seen', and it is not a "
            "forecast of a later wave for a known child. Row-level LOO is not used "
            "here because a held-out t2/t3 score is retained as the next transition's "
            "own baseline, so it is not out-of-sample."
        ),
        focal_term=focal,
        exposure=settings.dose_covariate,
        dose_margin="intensive_treated_rows",
        dose_contrast="treated_row_interquartile_within_phase",
        design=(
            "Period-resolved conditional-change model over all RLI transitions. "
            "Intervention presence and intervention intensity are separated: an "
            "on-intervention indicator carries the extensive margin, and session dose "
            "is centred and standardised over the fitted on-intervention rows only, "
            "entered as partially pooled period slopes or one pooled slope, with the "
            "child's study-average attendance split from their within-child deviation "
            "where declared. Assigned arm enters only in the post-crossover periods, "
            "where it is not collinear with intervention presence."
        ),
        estimand=(
            "The adjusted association between higher session attendance and the "
            "post-score among on-intervention rows (the intensive margin), reported "
            "on the items scale as a within-period interquartile contrast of observed "
            "treated attendance. The separate on-intervention indicator carries the "
            "extensive margin; in period 1 that contrast is randomised."
        ),
        causal_status=(
            "Observational association, not a randomised treatment effect. Session "
            "attendance is post-randomisation and may be confounded by attendance and "
            "engagement processes, and the authoritative DAG carries edges into "
            "intervention sessions from age, latent general ability and assigned "
            "group, so conditioning on measured baselines does not close that door. "
            "The one exception is the period-1 on-intervention indicator, which is a "
            "randomised contrast; it is reported as such and is not the family's "
            "headline."
        ),
        analysis_population=(
            f"Available RLI transition rows with observed {outcome}, "
            f"{settings.adjust_baseline_symbol} baseline and dose covariates. The "
            "items-scale dose contrast is averaged over the on-intervention rows "
            "only, since a session step on a row with no intervention is not a "
            "supported counterfactual of this design."
        ),
        missing_data_assumption=(
            "Available-case analysis under ignorable missingness conditional on the "
            "modelled variables; rows missing a required score, dose or group value "
            "are excluded before fitting."
        ),
    )
