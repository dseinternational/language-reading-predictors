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
- **Treatment as a hazard shift.** ``treated_ik`` is the intervention-aligned (treatment-on)
  indicator: the immediate arm (``G == 1``) is treated in every interval; the waitlist
  arm is treated from interval 2 (its crossover), mirroring the DiD ``treated`` term.
  ``G = 2 - group`` (positive = benefit), so a positive ``tau`` raises the hazard of
  coming off the floor.
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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from language_reading_predictors import paths as _paths
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.factories import (
    BuiltModel,
    _variables_dict,
)
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.preprocessing import (
    Standardiser,
    standardise,
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

    subject_ids: list = []
    interval_idx: list[int] = []
    event: list[int] = []
    treated: list[int] = []
    G_rows: list[int] = []
    cov_rows: dict[str, list[float]] = {name: [] for name, _ in _COVARIATES}

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
        imputed_covariate_rows=imputed,
    )


def build_survival_model(
    panel: SurvivalPanel,
    *,
    hazard_link: str = "cloglog",
    use_treatment: bool = True,
) -> BuiltModel:
    """Discrete-time off-floor hazard model on a :class:`SurvivalPanel`.

    ``hazard_link`` is ``"cloglog"`` (grouped proportional hazards, the default /
    primary) or ``"logit"`` (logistic-hazard sensitivity). ``tau`` is the treatment
    hazard shift on the intervention-aligned ``treated`` indicator; set
    ``use_treatment=False`` for a covariate-only baseline-hazard comparator.
    """
    if hazard_link not in ("cloglog", "logit"):
        raise ValueError("hazard_link must be 'cloglog' or 'logit'.")

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

        # Treatment as a hazard shift (the randomised-anchored, prognostic term).
        if use_treatment:
            tau = _priors.tau_prior().to_pymc("tau")
            eta = eta + tau * treated_d

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
        variables=_variables_dict(model),
        prepared=panel,
        extras={},
    )
