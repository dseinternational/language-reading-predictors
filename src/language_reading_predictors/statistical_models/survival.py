# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discrete-time survival family: time-to-off-floor for phonics/nonword (#230 §5).

The floored outcomes phonetic spelling (``P``) and nonword reading (``N``) are
modelled elsewhere by a single-transition off-floor estimand (the ``lrp-rli-itt-009``
/ ``lrp-rli-itt-011`` floor rule: a logistic ``tau`` on ``Pr(post > 0 | pre == 0)``
over the t1->t2 window). This family **generalises that single crossing to the full
four-wave sequence**: a discrete-time survival model for the *time* to first come off
the floor, recovering the information the fixed-timepoint rule discards.

Design (fixed in ``notes/…-persistent-floor-sitters-nonword-spelling.md``):

- **At-risk set.** A child enters at t1 iff they are at the floor at t1 (score == 0);
  children already off the floor at baseline were never floor-sitters and contribute
  no rows.
- **Person-period expansion.** One row per still-at-floor interval. The intervals are
  the three transitions (1: t1->t2, 2: t2->t3, 3: t3->t4). A child contributes rows
  from t1 until the first interval whose post-wave score is above zero (the **event**),
  or until an unobserved post-wave (**censored**). The ``"first"`` event rule (any
  crossing above zero) is the PRIMARY, mirroring the existing off-floor estimand; a
  sustained-off-floor sensitivity is deferred (it needs a look-ahead risk set — the
  flicker caveat is documented in the descriptive note).
- **Discrete-time hazard.** ``link(h_ik) = alpha_k + tau * treated_ik + beta_L * L0 +
  beta_W * W0 + beta_A * A0``, with a per-interval baseline hazard ``alpha_k``. The
  default link is complementary-log-log (grouped proportional hazards, the direct
  survival generalisation of the off-floor logit); a logistic-hazard variant is the
  documented sensitivity.
- **Treatment as a hazard contrast.** ``treated_ik`` is the intervention-aligned
  (treatment-on) indicator: the immediate arm (``G == 1``) is treated in every interval
  (session records confirm delivery continues through t3->t4); the waitlist arm is
  treated from interval 2 (its crossover), mirroring the DiD ``treated`` term.
  ``G = 2 - group`` (positive = benefit), so a positive ``tau`` raises the hazard of
  coming off the floor. Because every person-period row outside interval 1 is
  treatment-on, the likelihood carries **no arm contrast after the first interval**:
  under the legacy pooled parameterisation the split of the post-crossover hazard
  between ``tau`` and ``alpha_2``/``alpha_3`` was decided by the zero-centred alpha
  priors, which (centring the per-interval off-floor probability at 63% against
  observed 8-29%) dragged ``tau`` negative (2026-08-21 survival review, finding 1).
  The default ``treatment_window="randomised"`` therefore enters ``tau`` **only in the
  randomised first interval**, making it the immediate-vs-waitlist off-floor hazard
  contrast among children at the floor at t1, with the post-crossover intervals
  fitting their own (both-arms-treated) baseline hazards. The legacy pooled shift is
  retained as the explicit comparator ``treatment_window="pooled"``.
- **Covariates** are the *baseline* (t1) letter-sound knowledge (``L0``), word reading
  (``W0``) and age (``A0``) — prognostic, pre-intervention quantities, each entering as a
  weakly-regularised ``beta_*`` slope (concurrent letter sounds would be a
  treatment-affected mediator, so they are deliberately not used).
- **No child frailty.** The repeated person-period rows per child could carry a shrunken
  child random intercept (as the ``gain_factors`` family does), but it is deliberately
  omitted: at n≈36 at-risk children with ≤3 rows and ~one event each, a frailty term is
  weakly identified, and the discrete-time hazard likelihood already factorises over
  person-periods. ``child_idx`` / ``n_children`` are carried on the panel for reporting,
  not consumed by the model.

**Prognostic, not causal.** By t4 both arms are treated, so only the immediate arm's
first interval is randomised; the treatment hazard shift is read as a prognostic
association anchored on that window, not a licence to gate-keep (see the note's causal
caveat, and ``METHODS.md``).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from language_reading_predictors import paths as _paths
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.factories import (
    BuiltModel,
)
from language_reading_predictors.statistical_models.fitted_payloads import EmptyPayload
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.preprocessing import (
    Standardiser,
    standardise,
)

# The family-owned settings formerly read directly from ``ModelSpec.extra`` in
# ``pipelines/survival.py``.  ``target_accept`` remains a centrally resolved sampler
# option rather than a scientific model setting.
_LEGACY_KEYS = frozenset(
    {"hazard_link", "use_treatment", "treatment_window", "target_accept"}
)
_SURVIVAL_OUTCOMES = frozenset({"P", "N"})
_HAZARD_LINKS = frozenset({"cloglog", "logit"})
_TREATMENT_WINDOWS = frozenset({"randomised", "pooled"})


@dataclass(frozen=True, slots=True)
class SurvivalModelSettings:
    """Immutable declaration for one discrete-time off-floor survival model.

    ``treatment_window`` decides where the treatment term enters the hazard:
    ``"randomised"`` (the default since the 2026-08-21 survival review, finding 1)
    fits ``tau`` only in the randomised first interval — the immediate-vs-waitlist
    off-floor hazard contrast among children at the floor at t1 — while the
    post-crossover intervals (whose person-period rows are all treatment-on and so
    carry no arm contrast) fit their own baseline hazards. ``"pooled"`` is the
    legacy proportional-hazards shift across all intervals, retained as an explicit
    comparator: its split between ``tau`` and the post-crossover baseline hazards
    is identified only through the alpha priors.
    """

    hazard_link: Literal["cloglog", "logit"] = "cloglog"
    use_treatment: bool = True
    treatment_window: Literal["randomised", "pooled"] = "randomised"

    def __post_init__(self) -> None:
        if self.hazard_link not in _HAZARD_LINKS:
            raise ValueError("hazard_link must be 'cloglog' or 'logit'")
        if not isinstance(self.use_treatment, bool):
            raise TypeError("use_treatment must be a boolean")
        if self.treatment_window not in _TREATMENT_WINDOWS:
            raise ValueError("treatment_window must be 'randomised' or 'pooled'")

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> SurvivalModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown survival setting(s): {', '.join(unknown)}. "
                "Declare SurvivalModelSettings so misspellings fail fast."
            )
        return cls(
            hazard_link=extra.get("hazard_link", "cloglog"),
            use_treatment=extra.get("use_treatment", True),
            treatment_window=extra.get("treatment_window", "randomised"),
        )


@dataclass(frozen=True, slots=True)
class SurvivalRunPlan:
    """Concrete, validated instructions for a complete survival-family fit."""

    model_id: str
    settings_source: str
    study_id: str
    outcome_symbol: str
    hazard_link: Literal["cloglog", "logit"]
    use_treatment: bool
    treatment_window: Literal["randomised", "pooled"]
    likelihood: str
    observation_node: str
    compute_loo: bool
    loo_unit: str
    focal_term: str | None
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, str]:
        """Arguments for :func:`prepare_survival`."""
        return {"symbol": self.outcome_symbol}

    def factory_kwargs(self) -> dict[str, str | bool]:
        """Arguments for :func:`build_survival_model`."""
        return {
            "hazard_link": self.hazard_link,
            "use_treatment": self.use_treatment,
            "treatment_window": self.treatment_window,
        }

    def diagnostic_vars(self, covariates: Collection[str]) -> tuple[str, ...]:
        """Curated diagnostics in the same order as the prepared covariates."""
        return (
            "alpha",
            *(f"beta_{name}" for name in covariates),
            *(("tau",) if self.use_treatment else ()),
        )

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language recipe generated from the validated plan."""
        if not self.use_treatment:
            treatment = "This comparator omits the intervention-aligned treatment term."
        elif self.treatment_window == "randomised":
            treatment = (
                "The treatment term `tau` enters only the randomised first interval "
                "(t1 to t2): it is the immediate-versus-waitlist off-floor hazard "
                "contrast among children at the floor at wave 1, and the "
                "post-crossover intervals fit their own (both-arms-treated) baseline "
                "hazards. It is reported as a prognostic association, not a causal "
                "effect of record."
            )
        else:
            treatment = (
                "The intervention-aligned treatment-on indicator enters as a pooled "
                "`tau` across all intervals (the legacy comparator). Because no "
                "person-period row outside the first interval is untreated, the "
                "likelihood identifies this pooled coefficient only through the "
                "first-interval arm contrast; its split from the post-crossover "
                "baseline hazards is set by the alpha priors, so it is prognostic "
                "and prior-mediated, never a clean randomised treatment effect."
            )
        return (
            "Note: Generated from the validated survival run plan; template drafted "
            "by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Hazard link: `{self.hazard_link}`. "
            "Baseline predictors: letter-sound knowledge, word reading and age, "
            f"standardised across children. {treatment}\n\n"
            "## Uncertainty and checks\n\n"
            f"The observation node is `{self.observation_node}` and PSIS-LOO uses "
            f"the `{self.loo_unit}` unit. Interpret the posterior only after the "
            "zero-divergence convergence gate, posterior-predictive checks and "
            "power-scaling sensitivity diagnostics pass. The saved `config.json` "
            "contains the same resolved run plan in machine-readable form.\n"
        )


def declared_survival_settings(
    spec: ModelSpec,
) -> tuple[SurvivalModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: survival settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, SurvivalModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='survival' requires "
                f"SurvivalModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        SurvivalModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
        ),
        "legacy_extra",
    )


def resolve_survival_run_plan(spec: ModelSpec) -> SurvivalRunPlan:
    """Resolve and validate the survival contract before context or data I/O."""
    if spec.kind != "survival":
        raise ValueError(
            f"{spec.model_id}: expected kind 'survival', got {spec.kind!r}"
        )
    if spec.study_id != "rli":
        raise ValueError(
            f"{spec.model_id}: survival currently requires study_id='rli', got "
            f"{spec.study_id!r}"
        )
    if spec.outcome_symbol not in _SURVIVAL_OUTCOMES:
        raise ValueError(
            f"{spec.model_id}: survival outcome_symbol must be one of "
            f"{sorted(_SURVIVAL_OUTCOMES)!r}, got {spec.outcome_symbol!r}"
        )

    settings, source = declared_survival_settings(spec)
    if not settings.use_treatment:
        estimand = (
            "The interval-specific probability of first moving above the floor, "
            "without an intervention-aligned treatment coefficient."
        )
        causal_status = (
            "Descriptive baseline-hazard comparator: no treatment term is fitted, "
            "so no arm quantity of any kind is estimated."
        )
    elif settings.treatment_window == "randomised":
        estimand = (
            "The interval-specific probability of first moving above the floor. The "
            "headline tau is the intervention hazard contrast in the randomised "
            "first interval (t1 to t2) among children at the floor at wave 1; the "
            "post-crossover intervals fit their own baseline hazards and contain no "
            "arm contrast."
        )
        causal_status = (
            "Randomisation-anchored association: tau contrasts the randomised arms "
            "within the pre-randomisation at-floor subgroup over the first interval "
            "only, adjusted for baseline covariates. It is reported as a prognostic "
            "association rather than a causal effect of record — available-case "
            "censoring and the covariate adjustment are untested assumptions, and "
            "this family releases no causal headline."
        )
    else:
        estimand = (
            "The interval-specific probability of first moving above the floor. The "
            "headline tau is the pooled intervention-aligned log-hazard shift (the "
            "legacy comparator). Every person-period row outside the randomised "
            "first interval is treatment-on, so the likelihood identifies this "
            "pooled coefficient only through the first-interval arm contrast; its "
            "split from the post-crossover baseline hazards is prior-mediated "
            "(2026-08-21 survival review, finding 1)."
        )
        causal_status = (
            "Prognostic, prior-mediated association, not a causal treatment effect: "
            "only the first interval is randomisation-anchored, both arms are "
            "treated after the wait-list crossover, and the post-crossover share of "
            "the pooled coefficient is set by the baseline-hazard priors rather "
            "than by any observed arm comparison."
        )
    return SurvivalRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        outcome_symbol=spec.outcome_symbol,
        hazard_link=settings.hazard_link,
        use_treatment=settings.use_treatment,
        treatment_window=settings.treatment_window,
        likelihood="bernoulli_discrete_time_hazard",
        observation_node="y_event",
        compute_loo=True,
        loo_unit="person_period_row",
        focal_term="tau" if settings.use_treatment else None,
        design=(
            "Discrete-time first-off-floor survival model. Children at the outcome "
            "floor at wave 1 contribute one Bernoulli person-period row for each "
            "observed interval while they remain at risk, with an interval-specific "
            f"baseline hazard under a {settings.hazard_link} link."
        ),
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=(
            f"RLI children at the {spec.outcome_symbol} floor at wave 1 who have an "
            "observed wave-2 outcome and can therefore contribute at least one "
            "person-period row."
        ),
        missing_data_assumption=(
            "An unobserved next-wave outcome censors later follow-up. Missing baseline "
            "letter-sound, word-reading or age values are mean-imputed after "
            "child-level standardisation; interpretation assumes those operations do "
            "not create informative selection beyond the fitted predictors."
        ),
    )


#: The three wave-to-wave intervals (start wave, end wave); interval label = index + 1.
_INTERVALS: tuple[tuple[int, int], ...] = ((1, 2), (2, 3), (3, 4))

#: Baseline (t1) covariate columns entering the hazard, in report order.
_COVARIATES: tuple[tuple[str, str], ...] = (
    ("L0", V.YARCLET),  # letter-sound knowledge (prerequisite)
    ("W0", V.EWRSWR),  # word reading (sight-word reading without decoding)
    ("A0", V.AGE),  # baseline age (older children may come off the floor sooner)
)


@dataclass
class SurvivalPanel:
    """Person-period at-risk table for a discrete-time off-floor survival model.

    Exposes the ``n_obs`` / ``n_children`` / ``n_phases`` / ``dropped_rows`` accessors
    the shared pipeline header and ``reporting.write_run_metadata`` expect, so it is a
    drop-in container alongside :class:`preprocessing.PreparedData`.
    """

    symbol: str
    """Floored outcome symbol (``"P"`` or ``"N"``)."""
    subject_ids: np.ndarray
    """Subject id for each person-period row. shape (n_obs,)."""
    child_idx: np.ndarray
    """Integer child index in ``0..n_children-1`` per row. shape (n_obs,)."""
    interval_idx: np.ndarray
    """0-based interval index (0 = t1->t2, 1 = t2->t3, 2 = t3->t4). shape (n_obs,)."""
    event: np.ndarray
    """1 if the child came off the floor in this interval, else 0. shape (n_obs,)."""
    treated: np.ndarray
    """Intervention-aligned treatment-on indicator (1 = on). shape (n_obs,)."""
    G: np.ndarray
    """Arm indicator, positive-benefit coded (1 = immediate, 0 = waitlist). shape (n_obs,)."""
    covariates: dict[str, np.ndarray]
    """Standardised baseline covariate -> per-row value (broadcast within child)."""
    covariate_scalers: dict[str, Standardiser]
    n_children: int
    n_at_risk_children: int
    """Children at floor at t1 who entered the at-risk set."""
    n_events: int
    dropped_rows: int = 0
    """At-risk children (at floor at t1) who contributed no person-period row because
    their t2 post-score was unobserved (no interval could be placed). Named ``dropped_rows``
    for the shared pipeline-header / ``write_run_metadata`` interface."""
    dropped_by_reason: dict[str, int] = field(default_factory=dict)
    """Attribution of ``dropped_rows``, following the shared ledger convention that
    the values sum to ``dropped_rows`` (2026-08-21 survival review, finding 4)."""
    imputed_covariate_rows: dict[str, int] = field(default_factory=dict)
    """Rows whose (missing) baseline covariate was mean-imputed (z = 0), by name."""

    @property
    def n_obs(self) -> int:
        return int(self.event.shape[0])

    @property
    def n_phases(self) -> int:
        return len(_INTERVALS)


def _first_off_floor_wave(scores: dict[int, float]) -> int | None:
    """First wave with an observed score above zero (the PRIMARY event), or ``None``."""
    for t in (1, 2, 3, 4):
        s = scores.get(t)
        if s is not None and np.isfinite(s) and s > 0:
            return t
    return None


def prepare_survival(symbol: str, df: pd.DataFrame | None = None) -> SurvivalPanel:
    """Build the person-period at-risk table for a floored outcome (``"P"`` or ``"N"``).

    ``df`` is the long-format frame; when ``None`` it is read from
    ``data/rli_data_long.csv`` (the fit path). Passing a small frame directly makes the
    person-period expansion unit-testable without a data file.
    """
    if symbol not in MEASURES:
        raise ValueError(f"Unknown outcome symbol {symbol!r}.")
    col = MEASURES[symbol].column
    if df is None:
        df = pd.read_csv(_paths.DATA_DIR / "rli_data_long.csv")

    # Fail-loud source integrity, matching the shared loaders (2026-08-21 survival
    # review, finding 5): a duplicated (subject, time) key would otherwise surface
    # as a cryptic Series truth-value error inside the expansion loop, and an
    # invalid or within-child-unstable group code would silently miscode ``G``.
    dup = df.duplicated(subset=[V.SUBJECT_ID, V.TIME], keep=False)
    if bool(dup.any()):
        pairs = sorted(
            {
                (str(s), int(t))
                for s, t in df.loc[dup, [V.SUBJECT_ID, V.TIME]].itertuples(index=False)
            }
        )
        raise ValueError(f"Duplicate (subject, time) rows in survival source: {pairs}")
    raw_group = pd.to_numeric(df[V.GROUP], errors="coerce").to_numpy(dtype=float)
    valid_group = np.isfinite(raw_group) & np.isin(raw_group, (1.0, 2.0))
    if not valid_group.all():
        invalid = np.unique(raw_group[~valid_group])
        raise ValueError(
            "Group codes must be exactly 1 (immediate intervention) or 2 "
            f"(wait-list control); found invalid raw value(s) {invalid.tolist()}"
        )
    unstable = df.groupby(V.SUBJECT_ID)[V.GROUP].nunique(dropna=False)
    if bool((unstable > 1).any()):
        children = sorted(str(c) for c in unstable[unstable > 1].index)
        raise ValueError(f"Group code changes within child: {children}")

    subject_ids: list = []
    interval_idx: list[int] = []
    event: list[int] = []
    treated: list[int] = []
    G_rows: list[int] = []
    # Pandas may surface a missing scalar as ``None`` or ``NaN``; NumPy performs
    # the existing float coercion after row construction.
    cov_rows: dict[str, list[Any]] = {name: [] for name, _ in _COVARIATES}

    n_at_risk = 0
    for sid, g in df.groupby(V.SUBJECT_ID):
        by_time = g.set_index(V.TIME)
        scores = {t: (by_time[col].get(t) if t in by_time.index else np.nan) for t in (1, 2, 3, 4)}
        # Enter the at-risk set only if at the floor at t1.
        if not (np.isfinite(scores[1]) and scores[1] == 0):
            continue
        n_at_risk += 1
        group = int(by_time[V.GROUP].iloc[0])
        G = 2 - group  # positive-benefit coding: 1 = immediate, 0 = waitlist
        base = {name: (by_time[c].get(1) if 1 in by_time.index else np.nan) for name, c in _COVARIATES}
        off_wave = _first_off_floor_wave(scores)
        for k, (t_start, t_end) in enumerate(_INTERVALS, start=1):
            # Still at risk requires being at the floor at the interval's start wave.
            if not (np.isfinite(scores[t_start]) and scores[t_start] == 0):
                break
            if not np.isfinite(scores[t_end]):
                break  # censored: post-wave unobserved
            ev = 1 if (off_wave is not None and t_end == off_wave) else 0
            subject_ids.append(sid)
            interval_idx.append(k - 1)
            event.append(ev)
            treated.append(1 if (G == 1 or k >= 2) else 0)
            G_rows.append(G)
            for name in cov_rows:
                cov_rows[name].append(base[name])
            if ev == 1:
                break  # exits the risk set

    subject_arr = np.asarray(subject_ids)
    _, child_idx = np.unique(subject_arr, return_inverse=True)
    child_idx = child_idx.astype(np.int64)
    n_children = int(np.unique(child_idx).size)

    # Standardise each baseline covariate on the UNIQUE-CHILD baseline (one value per
    # child), then broadcast the scaler to the person-period rows — so the "per SD" scale
    # is the child-level SD and does not depend on how many intervals a child contributes
    # (#293 review). Missing baselines are mean-imputed to z = 0 so an at-risk child is
    # never dropped for a missing prerequisite.
    covariates: dict[str, np.ndarray] = {}
    scalers: dict[str, Standardiser] = {}
    imputed: dict[str, int] = {}
    for name in cov_rows:
        row_vals = np.asarray(cov_rows[name], dtype=float)
        child_vals = np.full(n_children, np.nan)
        for ci in range(n_children):
            rows_ci = np.flatnonzero(child_idx == ci)
            if rows_ci.size:
                child_vals[ci] = row_vals[rows_ci[0]]  # baseline is constant within child
        _, scaler = standardise(child_vals)  # mean / SD over children, not rows
        z = scaler(row_vals)
        missing = ~np.isfinite(z)
        imputed[name] = int(missing.sum())
        z[missing] = 0.0
        covariates[name] = z
        scalers[name] = scaler

    # At-risk children (at the floor at t1) who contributed no person-period row — the
    # t2 post-score was unobserved, so no interval could be placed. Surfaced rather than
    # silently dropped (#293 review); the fit prints it.
    dropped = int(n_at_risk - n_children)

    event_arr = np.asarray(event, dtype=np.int64)
    return SurvivalPanel(
        symbol=symbol,
        subject_ids=subject_arr,
        child_idx=child_idx,
        interval_idx=np.asarray(interval_idx, dtype=np.int64),
        event=event_arr,
        treated=np.asarray(treated, dtype=np.int64),
        G=np.asarray(G_rows, dtype=np.int64),
        covariates=covariates,
        covariate_scalers=scalers,
        n_children=n_children,
        n_at_risk_children=n_at_risk,
        n_events=int(event_arr.sum()),
        dropped_rows=dropped,
        dropped_by_reason=(
            {"no_observed_wave2_outcome": dropped} if dropped else {}
        ),
        imputed_covariate_rows=imputed,
    )


def build_survival_model(
    panel: SurvivalPanel,
    *,
    hazard_link: str = "cloglog",
    use_treatment: bool = True,
    treatment_window: str = "randomised",
) -> BuiltModel[EmptyPayload]:
    """Discrete-time off-floor hazard model on a :class:`SurvivalPanel`.

    ``hazard_link`` is ``"cloglog"`` (grouped proportional hazards, the default /
    primary) or ``"logit"`` (logistic-hazard sensitivity). ``tau`` is the treatment
    hazard term on the intervention-aligned ``treated`` indicator; set
    ``use_treatment=False`` for a covariate-only baseline-hazard comparator.

    ``treatment_window="randomised"`` (the default since the 2026-08-21 survival
    review, finding 1) enters ``tau`` only in the randomised first interval, where
    the panel's ``treated`` indicator equals the arm indicator ``G`` — so ``tau``
    is the immediate-vs-waitlist off-floor hazard contrast among children at the
    floor at t1, and ``alpha_2`` / ``alpha_3`` are the (both-arms-treated)
    post-crossover interval hazards. ``"pooled"`` retains the legacy shift across
    all intervals as an explicit comparator; because no row outside interval 1 is
    untreated, its split between ``tau`` and the post-crossover baselines is
    identified only through the alpha priors.
    """
    if hazard_link not in ("cloglog", "logit"):
        raise ValueError("hazard_link must be 'cloglog' or 'logit'.")
    if treatment_window not in ("randomised", "pooled"):
        raise ValueError("treatment_window must be 'randomised' or 'pooled'.")

    coords = {
        "obs_id": np.arange(panel.n_obs),
        "interval": [f"t{s}->t{e}" for s, e in _INTERVALS],
    }
    with pm.Model(coords=coords) as model:
        interval_d = pm.Data("interval_idx", panel.interval_idx, dims="obs_id")
        treated_d = pm.Data("treated", panel.treated.astype(float), dims="obs_id")
        cov_d = {
            name: pm.Data(f"{name}_std", panel.covariates[name], dims="obs_id")
            for name in panel.covariates
        }

        # Per-interval baseline hazard (the discrete-time nuisance trajectory).
        alpha = _priors.alpha_prior().to_pymc("alpha", dims="interval")
        eta = alpha[interval_d]

        # Baseline (prognostic) covariate slopes — associations, weakly regularised.
        for name in panel.covariates:
            beta = _priors.predictor_slope_prior().to_pymc(f"beta_{name}")
            eta = eta + beta * cov_d[name]

        # Treatment hazard term (the randomised-anchored, prognostic quantity).
        # Under the "randomised" window tau multiplies treated only in the first
        # interval (where treated == G); the pooled comparator keeps the legacy
        # all-interval shift.
        if use_treatment:
            tau = _priors.tau_prior().to_pymc("tau")
            trt_term = treated_d
            if treatment_window == "randomised":
                trt_term = treated_d * pt.eq(interval_d, 0)
            eta = eta + tau * trt_term

        eta = pm.Deterministic("eta", eta, dims="obs_id")

        if hazard_link == "cloglog":
            # h = 1 - exp(-exp(eta)); -expm1(-exp(eta)) is the stable form.
            h = pm.Deterministic(
                "hazard", pt.clip(-pt.expm1(-pt.exp(eta)), 1e-9, 1 - 1e-9), dims="obs_id"
            )
            pm.Bernoulli("y_event", p=h, observed=panel.event, dims="obs_id")
        else:
            pm.Deterministic("hazard", pm.math.sigmoid(eta), dims="obs_id")
            pm.Bernoulli("y_event", logit_p=eta, observed=panel.event, dims="obs_id")

    return BuiltModel(
        model=model,
        prepared=panel,
        payload=EmptyPayload(),
    )
