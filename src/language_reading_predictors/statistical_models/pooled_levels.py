# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Wave-pooled between-child level associations (``kind="pooled_levels"``).

The suite already answers "does a higher exposure level go with a higher outcome
level?" **at each wave separately** (``concurrent``), and "does a higher exposure
level go with ending a period higher, given where the child started?"
(``mechanism``). It has no model for the third question a reader naturally asks:
the same level association **pooled over all four waves**, as one coefficient.

That gap is not filled by either neighbour. ``concurrent`` is per-wave by
construction — it fits a separate model at each timepoint and every row it writes
is keyed by ``timepoint`` — so pooling is not a flag on it but a different
likelihood. ``mechanism`` conditions each outcome on its own period-start score,
which partials out exactly the stable between-child variation a levels question is
asking about; that is why its slopes are so much smaller than the per-wave level
associations, and why the two must not be read as estimates of one quantity.

The only stacked-levels skill-to-skill estimate that existed before this family was
the ``horseshoe`` ranking (``hs-002`` / ``hs-004``), which is unsuitable as an
association estimate on three counts: it is shrinkage-regularised, it carries **no
child random intercept** despite stacking ~4 rows per child, and it is framed as a
ranking cross-check rather than an effect.

**Model.** One Beta-Binomial likelihood over every (child, wave) row:

    eta = alpha_wave[t] + beta_G G + beta_mech z(logit exposure_t)
          + gamma_A z(A) + sum_c gamma_c z(c) + u_child

with no own-baseline term — its absence *is* the levels estimand — and a child
random intercept carrying the repeated measures.

**Wave intercepts.** ``use_wave_intercepts`` (default true) gives each wave its own
intercept, so ``beta_mech`` is the within-wave association averaged over waves
rather than a quantity part-driven by both measures rising together over the study.
The unpooled alternative is a registered comparator, not a hidden default: on the
RLI letter-sound / word-reading rows the pooled correlation of the two logit-scale
scores is 0.68 against 0.62 once each wave is centred (0.64 against 0.60 on the raw
counts), so secular co-movement is a real but modest tenth or less of the pooled
association, and reporting both is cheaper than arguing about which is meant.

Nothing here is causal. Exposure and outcome are measured at the same wave, so this
family has *less* temporal structure than ``mechanism``, not more, and its
coefficient absorbs every stable between-child difference the two skills share.

**Covariate exposures and same-wave skill adjusters (#553).** Two extensions let the
split reach the other predictors of word reading. ``mechanism_is_covariate`` takes
the ``mechanism`` family's route for a raw-score exposure (``erbto``, ``deapp_c``)
whose documented maximum is recorded nowhere: the exposure is the standardised raw
score (the raw-units SD is recorded beside the fit), ``require_observed`` must name
it so the mean-imputed rows are dropped — imputation plus an indicator is an
*adjuster* policy, never acceptable for the exposure itself — and the Mundlak split
is unchanged. ``skill_symbols`` adds other bounded-count measures at the **same
wave** as standardised logits (the ``concurrent`` family's idiom), each a
``gamma_{symbol}`` adjusted association; a row is kept only when the outcome, the
exposure and every skill adjuster are observed at that wave, and the dropped count
is reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm

from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.factories import (
    BuiltModel,
    _add_child_random_intercept,
)
from language_reading_predictors.statistical_models.fitted_payloads import FittedPayload
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.mechanism import (
    _validate_missing_covariate_policy,
)
from language_reading_predictors.statistical_models.preprocessing import (
    MISSINGNESS_INDICATOR_PAIRS,
    standardise,
)

_ALLOWED_KEYS = frozenset(
    {
        "adjust_for",
        "ability_covariate",
        "use_wave_intercepts",
        "decompose_between_within",
        "use_subject_random_intercept",
        "include_group",
        "waves",
        "mechanism_is_covariate",
        "require_observed",
        "skill_symbols",
        "target_accept",
    }
)


#: Display labels for the raw-score covariate exposures this family accepts (#553);
#: bounded-count exposures take their registered measure label.
COVARIATE_EXPOSURE_LABELS: dict[str, str] = {
    "erbto": "phonological memory (word/nonword repetition)",
    "deapp_c": "speech production accuracy",
    "hs": "hearing status",
}


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} must contain non-empty strings, got {item!r}")
    return out


@dataclass(frozen=True)
class PooledLevelsPayload(FittedPayload):
    """Fitted-row accounting for the pooled-levels design.

    The levels loader keeps a row when *any* requested outcome is observed, so a
    child-wave row can carry the exposure without the outcome (or, with same-wave
    skill adjusters, the outcome without a skill). Those rows cannot enter a
    single-outcome likelihood; they are dropped here and counted, rather than
    silently, because the count is part of the analysis population. A covariate
    exposure's raw-units anchor (the SD and mean of the fitted exposure in the
    raw score's units) is recorded so a per-SD slope can be read back in points.
    """

    n_fitted_rows: int
    n_dropped_incomplete: int
    n_children: int
    exposure_kind: str = "bounded_count"
    exposure_sd_raw: float | None = None
    exposure_mean_raw: float | None = None


@dataclass(frozen=True, slots=True)
class PooledLevelsModelSettings:
    """Immutable settings declared by a pooled-levels model module."""

    adjust_for: tuple[str, ...] = ()
    ability_covariate: str | None = None
    use_wave_intercepts: bool = True
    decompose_between_within: bool = True
    use_subject_random_intercept: bool = True
    include_group: bool = True
    waves: tuple[int, ...] = (1, 2, 3, 4)
    # #553: raw-score exposure (standardised covariate, complete-case on the
    # exposure) and same-wave bounded-count skill adjusters.
    mechanism_is_covariate: bool = False
    require_observed: tuple[str, ...] = ()
    skill_symbols: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "adjust_for", _tuple_of_strings(self.adjust_for, name="adjust_for"))
        object.__setattr__(
            self, "require_observed", _tuple_of_strings(self.require_observed, name="require_observed")
        )
        object.__setattr__(
            self, "skill_symbols", _tuple_of_strings(self.skill_symbols, name="skill_symbols")
        )
        if not isinstance(self.mechanism_is_covariate, bool):
            raise TypeError("mechanism_is_covariate must be bool")
        if len(set(self.skill_symbols)) != len(self.skill_symbols):
            raise ValueError("skill_symbols must not repeat")
        object.__setattr__(self, "waves", tuple(int(w) for w in self.waves))
        if not self.waves:
            raise ValueError("waves must not be empty")
        if sorted(set(self.waves)) != sorted(self.waves):
            raise ValueError("waves must not repeat")
        if not set(self.waves) <= {1, 2, 3, 4}:
            raise ValueError("waves must be a subset of (1, 2, 3, 4)")
        if len(self.waves) < 2 and self.use_wave_intercepts:
            raise ValueError(
                "use_wave_intercepts requires at least two waves; a single-wave fit "
                "is a concurrent model, not a pooled one."
            )
        if not self.use_subject_random_intercept and len(self.waves) > 1:
            raise ValueError(
                "pooling several waves without a child random intercept treats "
                "repeated measures on one child as independent and understates "
                "uncertainty; that is the defect this family exists to avoid."
            )
        if self.ability_covariate is not None and self.ability_covariate in self.adjust_for:
            raise ValueError(
                f"{self.ability_covariate!r} is declared both as ability_covariate and "
                "in adjust_for; name it once."
            )

    @classmethod
    def from_extra(cls, extra: dict[str, Any], *, model_id: str) -> PooledLevelsModelSettings:
        unknown = sorted(set(extra) - _ALLOWED_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown pooled_levels setting(s): {', '.join(unknown)}."
            )
        return cls(
            adjust_for=extra.get("adjust_for", ()),
            ability_covariate=extra.get("ability_covariate"),
            use_wave_intercepts=extra.get("use_wave_intercepts", True),
            decompose_between_within=extra.get("decompose_between_within", True),
            use_subject_random_intercept=extra.get("use_subject_random_intercept", True),
            include_group=extra.get("include_group", True),
            waves=extra.get("waves", (1, 2, 3, 4)),
            mechanism_is_covariate=extra.get("mechanism_is_covariate", False),
            require_observed=extra.get("require_observed", ()),
            skill_symbols=extra.get("skill_symbols", ()),
        )


@dataclass(frozen=True, slots=True)
class PooledLevelsRunPlan:
    """Validated instructions resolved before any data are loaded."""

    model_id: str
    outcome_symbol: str
    mechanism_symbol: str
    settings_source: str
    adjust_for: tuple[str, ...]
    ability_covariate: str | None
    use_wave_intercepts: bool
    decompose_between_within: bool
    use_subject_random_intercept: bool
    include_group: bool
    waves: tuple[int, ...]
    # #553: exposure kind ("bounded_count" — the standardised logit of a measure's
    # proportion — or "raw_covariate" — the standardised raw score, complete-case
    # on the exposure), the complete-case covariates and the same-wave skill
    # adjusters.
    mechanism_is_covariate: bool
    exposure_kind: str
    require_observed: tuple[str, ...]
    skill_symbols: tuple[str, ...]
    likelihood: str
    observation_node: str
    focal_term: str
    compute_loo: bool
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare``.

        The ability adjuster is recorded once at t1, so it broadcasts from there via
        ``baseline_covariates`` rather than being pulled per row (where it is NaN
        after t1). Same routing as the mechanism, gain-factor and aligned families.
        """
        outcomes: tuple[str, ...] = (self.outcome_symbol,)
        if not self.mechanism_is_covariate:
            outcomes += (self.mechanism_symbol,)
        outcomes += tuple(s for s in self.skill_symbols if s not in outcomes)
        post_covariates: tuple[str, ...] = self.adjust_for
        if self.mechanism_is_covariate:
            # The raw-score exposure and its missingness flag load as same-wave
            # covariates; ``require_observed`` then drops the imputed rows and the
            # constant flag is dropped downstream (the mechanism family's route).
            post_covariates = tuple(
                dict.fromkeys(
                    (
                        *post_covariates,
                        self.mechanism_symbol,
                        MISSINGNESS_INDICATOR_PAIRS[self.mechanism_symbol],
                    )
                )
            )
        for parent in self.require_observed:
            post_covariates = tuple(
                dict.fromkeys(
                    (*post_covariates, parent, MISSINGNESS_INDICATOR_PAIRS[parent])
                )
            )
        return {
            "phase_mode": "levels",
            "outcomes": outcomes,
            "post_covariates": post_covariates,
            "baseline_covariates": (
                (self.ability_covariate,) if self.ability_covariate else ()
            ),
            "require_observed": self.require_observed,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        return {
            "outcome_symbol": self.outcome_symbol,
            "mechanism_symbol": self.mechanism_symbol,
            "use_wave_intercepts": self.use_wave_intercepts,
            "decompose_between_within": self.decompose_between_within,
            "use_subject_random_intercept": self.use_subject_random_intercept,
            "include_group": self.include_group,
            "waves": self.waves,
            "mechanism_is_covariate": self.mechanism_is_covariate,
            "skill_symbols": self.skill_symbols,
        }

    @property
    def exposure_label(self) -> str:
        """Display label for the exposure: the measure's registered label, or the
        covariate's documented name."""
        if self.mechanism_is_covariate:
            return COVARIATE_EXPOSURE_LABELS.get(
                self.mechanism_symbol, self.mechanism_symbol
            )
        return MEASURES[self.mechanism_symbol].label

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        waves = ", ".join(f"t{w}" for w in self.waves)
        adjusters = ", ".join(self.adjust_for) if self.adjust_for else "none"
        ability = self.ability_covariate or "none"
        slopes = (
            "`beta_between` (child study-average exposure) and `beta_within` "
            "(wave deviation from the child's own average)"
            if self.decompose_between_within
            else "`beta_mech` (a single blended slope; comparator only)"
        )
        intercepts = (
            "one intercept per wave (`alpha_wave`)"
            if self.use_wave_intercepts
            else "a single intercept (`alpha`), so the slopes also carry the secular "
            "co-movement of the two measures across waves"
        )
        exposure = (
            f"`{self.mechanism_symbol}` at the same wave, as the standardised raw "
            "score (a covariate exposure, #553: its documented maximum is recorded "
            "nowhere, so a bounded-count logit would fabricate a denominator; rows "
            "whose exposure was imputed are dropped via `require_observed`)"
            if self.mechanism_is_covariate
            else f"`{self.mechanism_symbol}` at the same wave, as the standardised "
            "logit of the observed proportion"
        )
        skills = (
            ", ".join(f"`{s}`" for s in self.skill_symbols)
            if self.skill_symbols
            else "none"
        )
        complete_case = (
            ", ".join(f"`{c}`" for c in self.require_observed)
            if self.require_observed
            else "none"
        )
        return (
            "Note: Generated from the validated pooled-levels run plan; template "
            "drafted by an LLM-based AI tool (Claude Code/Opus 5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}` at each of waves {waves}. Exposure: "
            f"{exposure}. Exposure slopes: {slopes}. Intercepts: "
            f"{intercepts}. Arm main effect (`beta_G`, an adjusted association pooled "
            f"over post-crossover waves): {self.include_group}. Linear age at the "
            "wave (`gamma_A`). Same-wave covariate adjusters via `adjust_for`: "
            f"{adjusters}. Same-wave skill adjusters (standardised logits of other "
            f"measures, each a `gamma_<symbol>` adjusted association): {skills}. "
            f"Complete-case covariates (`require_observed`): {complete_case}. t1 "
            f"ability baseline broadcast across waves: {ability}. "
            f"Child random intercept: {self.use_subject_random_intercept}. No "
            f"own-baseline term. Likelihood: {self.likelihood} "
            f"(`{self.observation_node}`).\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate, posterior-predictive checks and PSIS-LOO reliability "
            "checks pass. Every coefficient is an adjusted association: exposure and "
            "outcome are contemporaneous, so nothing here orders them in time. The "
            "saved `config.json` contains the same resolved run plan in "
            "machine-readable form.\n"
        )

    def diagnostic_vars(self, covariates: tuple[str, ...]) -> tuple[str, ...]:
        """Summary variables; ``covariates`` is the loaded covariate set, from which
        a covariate exposure is excluded (it carries the focal slopes, not a
        ``gamma_`` adjuster coefficient)."""
        names = (
            ["beta_between", "beta_within", "kappa"]
            if self.decompose_between_within
            else ["beta_mech", "kappa"]
        )
        names.append("alpha_wave" if self.use_wave_intercepts else "alpha")
        if self.include_group:
            names.append("beta_G")
        names.append("gamma_A")
        names += [f"gamma_{s}" for s in self.skill_symbols]
        names += [
            f"gamma_{c}"
            for c in covariates
            if not (self.mechanism_is_covariate and c == self.mechanism_symbol)
        ]
        if self.use_subject_random_intercept:
            names.append("sigma_child")
        return tuple(names)


def resolve_pooled_levels_run_plan(spec: ModelSpec) -> PooledLevelsRunPlan:
    """Resolve and validate the family contract before any I/O."""
    if spec.kind != "pooled_levels":
        raise ValueError(f"{spec.model_id}: expected kind='pooled_levels'")
    if not spec.outcome_symbol or not spec.mechanism_symbol:
        raise ValueError(
            f"{spec.model_id}: pooled_levels needs both outcome_symbol and "
            "mechanism_symbol"
        )
    if spec.outcome_symbol == spec.mechanism_symbol:
        raise ValueError(
            f"{spec.model_id}: outcome and exposure are the same measure "
            f"({spec.outcome_symbol!r}); the coefficient would be trivially 1."
        )
    if spec.outcome_symbol not in MEASURES:
        raise ValueError(
            f"{spec.model_id}: unknown measure symbol {spec.outcome_symbol!r}"
        )

    declared = getattr(spec, "model_settings", None)
    if declared is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: pooled_levels settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(declared, PooledLevelsModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='pooled_levels' requires "
                f"PooledLevelsModelSettings, got {type(declared).__name__}"
            )
        settings, source = declared, "typed_settings"
    else:
        settings = PooledLevelsModelSettings.from_extra(
            dict(spec.extra or {}), model_id=spec.model_id
        )
        source = "legacy_extra"

    # --- exposure kind (#553). A bounded-count measure enters as its standardised
    # logit; a raw-score covariate (erbto, deapp_c) enters standardised, and must
    # be complete-case on the exposure. Each declaration must be coherent before
    # any output directory is reset or data are loaded.
    if settings.mechanism_is_covariate:
        if spec.mechanism_symbol in MEASURES:
            raise ValueError(
                f"{spec.model_id}: bounded measure exposure {spec.mechanism_symbol!r} "
                "cannot be declared as a raw covariate (mechanism_is_covariate)"
            )
        if spec.mechanism_symbol not in MISSINGNESS_INDICATOR_PAIRS:
            raise ValueError(
                f"{spec.model_id}: covariate exposure {spec.mechanism_symbol!r} is "
                "not a supported filled covariate; the loader can only enforce "
                "require_observed for "
                f"{', '.join(sorted(MISSINGNESS_INDICATOR_PAIRS))}"
            )
        if spec.mechanism_symbol not in settings.require_observed:
            raise ValueError(
                f"{spec.model_id}: covariate exposure {spec.mechanism_symbol!r} must "
                "be declared in require_observed — mean-imputation plus an indicator "
                "is an adjuster policy, never acceptable for the exposure itself"
            )
        if spec.mechanism_symbol in settings.adjust_for:
            raise ValueError(
                f"{spec.model_id}: covariate exposure {spec.mechanism_symbol!r} must "
                "not also appear in adjust_for"
            )
        if settings.ability_covariate == spec.mechanism_symbol:
            raise ValueError(
                f"{spec.model_id}: covariate exposure {spec.mechanism_symbol!r} must "
                "not also be the ability_covariate"
            )
    elif spec.mechanism_symbol not in MEASURES:
        raise ValueError(
            f"{spec.model_id}: unknown measure symbol {spec.mechanism_symbol!r} "
            "(declare mechanism_is_covariate=True for a raw-score exposure)"
        )
    _validate_missing_covariate_policy(
        model_id=spec.model_id,
        adjust_for=settings.adjust_for,
        require_observed=settings.require_observed,
        exposure=spec.mechanism_symbol if settings.mechanism_is_covariate else None,
        moderator=None,
    )
    # --- same-wave skill adjusters (#553): other bounded-count measures, never the
    # outcome or the exposure.
    for sym in settings.skill_symbols:
        if sym not in MEASURES:
            raise ValueError(
                f"{spec.model_id}: unknown skill adjuster symbol {sym!r}"
            )
        if sym == spec.outcome_symbol:
            raise ValueError(
                f"{spec.model_id}: skill adjuster {sym!r} is the outcome"
            )
        if sym == spec.mechanism_symbol:
            raise ValueError(
                f"{spec.model_id}: skill adjuster {sym!r} is the exposure"
            )
    bounded_adjusters = sorted(set(settings.adjust_for) & set(MEASURES))
    if bounded_adjusters:
        raise ValueError(
            f"{spec.model_id}: bounded measure adjuster(s) must be declared in "
            f"skill_symbols, not adjust_for: {', '.join(bounded_adjusters)}"
        )

    out_label = MEASURES[spec.outcome_symbol].label
    mech_label = (
        COVARIATE_EXPOSURE_LABELS.get(spec.mechanism_symbol, spec.mechanism_symbol)
        if settings.mechanism_is_covariate
        else MEASURES[spec.mechanism_symbol].label
    )
    exposure_scale = (
        "standardised raw score" if settings.mechanism_is_covariate else "logit"
    )
    skill_clause = (
        " and the same-wave standardised logits of "
        + ", ".join(MEASURES[s].label for s in settings.skill_symbols)
        + " as skill adjusters"
        if settings.skill_symbols
        else ""
    )
    waves = ", ".join(f"t{w}" for w in settings.waves)
    intercepts = (
        "per-wave intercepts, so the slope is the within-wave association averaged "
        "over waves"
        if settings.use_wave_intercepts
        else "a single intercept, so the slope also carries the secular co-movement "
        "of the two measures across waves"
    )
    return PooledLevelsRunPlan(
        model_id=spec.model_id,
        outcome_symbol=spec.outcome_symbol,
        mechanism_symbol=spec.mechanism_symbol,
        settings_source=source,
        adjust_for=settings.adjust_for,
        ability_covariate=settings.ability_covariate,
        use_wave_intercepts=settings.use_wave_intercepts,
        decompose_between_within=settings.decompose_between_within,
        use_subject_random_intercept=settings.use_subject_random_intercept,
        include_group=settings.include_group,
        waves=settings.waves,
        mechanism_is_covariate=settings.mechanism_is_covariate,
        exposure_kind=(
            "raw_covariate" if settings.mechanism_is_covariate else "bounded_count"
        ),
        require_observed=settings.require_observed,
        skill_symbols=settings.skill_symbols,
        likelihood="beta_binomial",
        observation_node="y_post",
        focal_term=("beta_between" if settings.decompose_between_within else "beta_mech"),
        compute_loo=True,
        design=(
            f"Wave-pooled between-child level model: one Beta-Binomial likelihood over "
            f"every ({waves}) child-wave row for {out_label}, with {mech_label} at the "
            f"same wave as the standardised exposure ({exposure_scale}){skill_clause}, "
            f"{intercepts}, and a child random "
            "intercept carrying the repeated measures. No own-baseline term: its "
            "absence is what makes this a levels rather than a transition estimand."
        ),
        estimand=(
            (
                f"beta_between and beta_within: the between-child association (a child "
                f"whose {mech_label} {exposure_scale} sits 1 SD higher across the study) "
                f"and the within-child association (a wave where a child sits 1 SD above "
                f"their own {mech_label} average) with the {out_label} logit at the same "
                "wave, holding the declared adjusters fixed; the SD is that of the pooled "
                f"row-level exposure {exposure_scale}."
            )
            if settings.decompose_between_within
            else (
                f"beta_mech, the pooled association between a 1 SD higher {mech_label} "
                f"{exposure_scale} and the {out_label} logit at the same wave, holding "
                "the declared adjusters fixed — a precision-weighted blend of the "
                "between-child and within-child associations."
            )
        ),
        causal_status=(
            "Association only. Exposure and outcome are contemporaneous, so this "
            "family has less temporal structure than the mechanism family, not more; "
            "the coefficient absorbs every stable between-child difference the two "
            "skills share, and latent general ability is not blocked."
            + (
                " Same-wave skill adjusters condition on contemporaneous, possibly "
                "post-treatment skill levels; their coefficients are adjusted "
                "associations and the Table-2 fallacy applies."
                if settings.skill_symbols
                else ""
            )
        ),
        analysis_population=(
            "Children with an observed outcome, exposure"
            + (", every same-wave skill adjuster" if settings.skill_symbols else "")
            + " and adjuster set at a given wave; a child contributes as many rows as "
            "they have complete waves."
            + (
                " Rows whose exposure was mean-imputed are dropped "
                "(require_observed), and the count is reported."
                if settings.mechanism_is_covariate
                else ""
            )
        ),
        missing_data_assumption=(
            "Complete-case at the row level, with the loader's mean-imputed covariate "
            "indicators carrying the unknown groups for the adjusters"
            + (
                "; the exposure itself is complete-case, never imputed."
                if settings.mechanism_is_covariate
                else "."
            )
        ),
    )


def build_pooled_levels_model(
    prepared,
    *,
    outcome_symbol: str,
    mechanism_symbol: str,
    use_wave_intercepts: bool = True,
    decompose_between_within: bool = True,
    use_subject_random_intercept: bool = True,
    include_group: bool = True,
    waves: tuple[int, ...] = (1, 2, 3, 4),
    mechanism_is_covariate: bool = False,
    skill_symbols: tuple[str, ...] = (),
) -> BuiltModel[PooledLevelsPayload]:
    """Wave-pooled Beta-Binomial level model on a ``phase_mode='levels'`` frame.

    ``mechanism_is_covariate`` (#553) reads the exposure from
    ``prepared.covariates[mechanism_symbol]`` — the loader's standardised raw score,
    already complete-case through ``require_observed`` — instead of the logit of a
    bounded count, and re-standardises it on the fitted rows (the mechanism family's
    route); the raw-units SD and mean of the fitted exposure are recorded in the
    payload. ``skill_symbols`` adds the same-wave standardised logits of other
    bounded-count measures as ``gamma_{symbol}`` adjusters on the cross-coupling
    prior; a row is kept only when the outcome, the exposure and every skill are
    observed at that wave.
    """
    skill_symbols = tuple(skill_symbols)
    y_all = np.asarray(prepared.post_counts[outcome_symbol], dtype=float)
    if mechanism_is_covariate:
        if mechanism_symbol not in prepared.covariates:
            raise KeyError(
                f"pooled_levels: covariate exposure {mechanism_symbol!r} is not in "
                "prepared.covariates"
            )
        x_all = np.asarray(prepared.covariates[mechanism_symbol], dtype=float)
    else:
        x_all = np.asarray(prepared.post_counts[mechanism_symbol], dtype=float)
    skill_all = {}
    for sym in skill_symbols:
        if sym not in prepared.post_counts:
            raise KeyError(
                f"pooled_levels: skill adjuster {sym!r} is not in prepared.post_counts"
            )
        skill_all[sym] = np.asarray(prepared.post_counts[sym], dtype=float)
    in_wave = np.isin(np.asarray(prepared.phase) + 1, np.asarray(waves))
    complete = np.isfinite(y_all) & np.isfinite(x_all)
    for values in skill_all.values():
        complete = complete & np.isfinite(values)
    keep = in_wave & complete
    n_dropped = int((in_wave & ~complete).sum())
    if not keep.any():
        raise ValueError(
            "pooled_levels: no child-wave row has the outcome, the exposure and "
            "every skill adjuster observed in the requested waves."
        )
    y = y_all[keep]
    exposure = x_all[keep]

    n_out = MEASURES[outcome_symbol].n_trials
    exposure_sd_raw: float | None = None
    exposure_mean_raw: float | None = None
    if mechanism_is_covariate:
        # Re-standardise the loader's z on the fitted rows so +1 SD is one SD of
        # the exposure actually fitted; the loader scaler maps it back to raw units.
        mech_std, _ = standardise(exposure)
        scaler = prepared.covariate_scalers.get(mechanism_symbol)
        if scaler is not None:
            exposure_sd_raw = float(scaler.sd * np.std(exposure, ddof=1))
            exposure_mean_raw = float(scaler.mean + scaler.sd * np.mean(exposure))
    else:
        n_mech = MEASURES[mechanism_symbol].n_trials
        # Standardised logit of the exposure proportion — the same per-SD scale the
        # mechanism family reports on, so the two are directly comparable.
        p = np.clip(exposure / n_mech, 1e-3, 1 - 1e-3)
        mech_std, _ = standardise(np.log(p / (1 - p)))
    # Same-wave skill adjusters: the standardised logit of each skill's observed
    # proportion on the fitted rows (the same transform as a bounded exposure).
    skill_std: dict[str, np.ndarray] = {}
    for sym, values in skill_all.items():
        p_s = np.clip(values[keep] / MEASURES[sym].n_trials, 1e-3, 1 - 1e-3)
        skill_std[sym], _ = standardise(np.log(p_s / (1 - p_s)))

    child_idx = np.asarray(prepared.child_idx)[keep]
    _, child_idx = np.unique(child_idx, return_inverse=True)
    wave_idx_raw = np.asarray(prepared.phase)[keep]
    wave_labels = sorted(set(int(w) + 1 for w in wave_idx_raw))
    remap = {w - 1: i for i, w in enumerate(wave_labels)}
    wave_idx = np.array([remap[int(w)] for w in wave_idx_raw])

    # Child mean and within-child deviation of the standardised exposure.
    mech_bar = np.zeros_like(mech_std)
    for c in np.unique(child_idx):
        m = child_idx == c
        mech_bar[m] = mech_std[m].mean()
    mech_dev = mech_std - mech_bar

    coords = {
        "obs_id": np.arange(len(y)),
        "child": np.arange(int(child_idx.max()) + 1),
        "wave": [f"t{w}" for w in wave_labels],
    }
    with pm.Model(coords=coords) as model:
        mech_d = pm.Data("mech_post_logit_std", mech_std, dims="obs_id")
        mech_bar_d = pm.Data("mech_child_mean", mech_bar, dims="obs_id")
        mech_dev_d = pm.Data("mech_within_dev", mech_dev, dims="obs_id")
        wave_d = pm.Data("wave_idx", wave_idx, dims="obs_id")

        if use_wave_intercepts:
            alpha = _priors.alpha_prior().to_pymc("alpha_wave", dims="wave")
            eta = alpha[wave_d]
        else:
            eta = _priors.alpha_prior().to_pymc("alpha")

        if decompose_between_within:
            # Mundlak / within-between split. A random-intercept model with one
            # exposure coefficient returns a precision-weighted BLEND of the
            # between-child and within-child associations, which correspond to
            # different questions and, on these data, to very different values
            # (r = 0.81 between against 0.45 within for the logit-scale letter-sound
            # and word-reading scores; 0.70 against 0.51 on the raw counts).
            # Splitting the exposure into the child mean and the deviation from it
            # estimates each cleanly and leaves nothing blended.
            beta_between = _priors.beta_mech_prior().to_pymc("beta_between")
            beta_within = _priors.beta_mech_prior().to_pymc("beta_within")
            eta = eta + beta_between * mech_bar_d + beta_within * mech_dev_d
        else:
            beta_mech = _priors.beta_mech_prior().to_pymc("beta_mech")
            eta = eta + beta_mech * mech_d

        if include_group:
            g = pm.Data("G", np.asarray(prepared.G, dtype=float)[keep], dims="obs_id")
            eta = eta + _priors.tau_prior().to_pymc("beta_G") * g

        age = pm.Data("A_std", np.asarray(prepared.A_std, dtype=float)[keep], dims="obs_id")
        eta = eta + _priors.gamma_age_prior().to_pymc("gamma_A") * age

        # Same-wave skill adjusters (#553): cross-coupling prior, like every other
        # family's measure adjusters; adjusted associations, never effects.
        for sym, z in skill_std.items():
            sk = pm.Data(f"{sym}_post_logit_std", z, dims="obs_id")
            eta = eta + _priors.gamma_cross_prior().to_pymc(f"gamma_{sym}") * sk

        for name, values in prepared.covariates.items():
            if mechanism_is_covariate and name == mechanism_symbol:
                continue  # the exposure carries the focal slopes, not a gamma_
            cov = pm.Data(f"{name}_std", np.asarray(values, dtype=float)[keep], dims="obs_id")
            eta = eta + _priors.predictor_slope_prior().to_pymc(f"gamma_{name}") * cov

        if use_subject_random_intercept:
            eta = _add_child_random_intercept(eta, child_idx)

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        mu = pm.Deterministic("mu", pm.math.sigmoid(eta), dims="obs_id")
        kappa = _priors.kappa_prior().to_pymc("kappa")
        pm.BetaBinomial(
            "y_post",
            alpha=mu * kappa,
            beta=(1.0 - mu) * kappa,
            n=n_out,
            observed=y.astype(int),
            dims="obs_id",
        )

    return BuiltModel(
        model=model,
        prepared=prepared,
        payload=PooledLevelsPayload(
            n_fitted_rows=int(len(y)),
            n_dropped_incomplete=n_dropped,
            n_children=int(child_idx.max()) + 1,
            exposure_kind="raw_covariate" if mechanism_is_covariate else "bounded_count",
            exposure_sd_raw=exposure_sd_raw,
            exposure_mean_raw=exposure_mean_raw,
        ),
    )


def pooled_levels_summary(trace, *, ci_prob: float, n_trials: int) -> pd.DataFrame:
    """Coefficient table plus the items-scale reading of the focal slope."""
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    post = trace.posterior
    rows: list[dict] = []
    for name in ["beta_between", "beta_within", "beta_mech", "beta_G", "gamma_A"] + sorted(
        str(v) for v in post.data_vars if str(v).startswith("gamma_") and str(v) != "gamma_A"
    ):
        if name not in post:
            continue
        d = np.asarray(post[name]).reshape(-1)
        rows.append(
            {
                "term": name,
                "role": "association",
                "median": float(np.median(d)),
                "lo": float(np.quantile(d, lo_q)),
                "hi": float(np.quantile(d, hi_q)),
                "prob_positive": float(np.mean(d > 0)),
            }
        )
    # Items-scale reading of a 1 SD exposure shift at the fitted mean linear predictor.
    eta = np.asarray(post["eta"]).reshape(np.asarray(post["eta"]).shape[0] * np.asarray(post["eta"]).shape[1], -1)
    focal = "beta_between" if "beta_between" in post else "beta_mech"
    beta = np.asarray(post[focal]).reshape(-1)
    base = eta.mean(axis=1)
    shifted = base + beta
    items = n_trials * (1 / (1 + np.exp(-shifted)) - 1 / (1 + np.exp(-base)))
    rows.append(
        {
            "term": f"{focal} (items per +1 SD)",
            "role": "association",
            "median": float(np.median(items)),
            "lo": float(np.quantile(items, lo_q)),
            "hi": float(np.quantile(items, hi_q)),
            "prob_positive": float(np.mean(items > 0)),
        }
    )
    return pd.DataFrame(rows)
