# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Counterfactual (g-formula) mediation decomposition for LRP59.

Given the joint mediator + outcome posterior from
:func:`factories.build_mediation_model`, decompose the intervention's effect on
word reading into the natural indirect effect (NIE, the part flowing *through*
letter-sound knowledge) and the natural direct effect (NDE, everything else).

NDE/NIE are **not** products of coefficients — that is only valid on a linear
identity scale, and these are logit Beta-Binomial models. They are computed by
simulation from the posterior: for each posterior draw, simulate the mediator
under each treatment counterfactual from the mediator model, push it through the
outcome model, and average the resulting outcome probability over the observed
covariate distribution. This yields full posteriors for every quantity.

Repo sign convention: ``G = group - 1`` with ``G = 0`` = initial-intervention arm
and ``G = 1`` = wait-list control in phase 0. Effects are reported in the
**intervention-helps** direction (intervention minus control), so positive =
intervention raises reading. With ``treat = 0`` (intervention) and ``ctrl = 1``
(control), and ``M(g)`` the mediator simulated under arm ``g``:

    Total = E[Y(treat, M(treat))] - E[Y(ctrl, M(ctrl))]
    NDE   = E[Y(treat, M(ctrl))]  - E[Y(ctrl, M(ctrl))]
    NIE   = E[Y(treat, M(treat))] - E[Y(treat, M(ctrl))]
    proportion mediated = NIE / Total

reported on the response scale (probability and word-count out of N_W), which is
the natural and interpretable scale for the g-formula decomposition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.math.constants import EPSILON

from language_reading_predictors.statistical_models.factories import (
    MediationData,
    TwoMediatorData,
)
from language_reading_predictors.statistical_models.preprocessing import logit_safe

_TREAT = 0.0  # initial-intervention arm (G = 0)
_CTRL = 1.0  # wait-list control arm (G = 1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decompose(
    trace: xr.DataTree,
    med: MediationData,
    *,
    hdi_prob: float = 0.95,
    n_replicates: int = 50,
    seed: int = 47,
) -> pd.DataFrame:
    """Return the NDE / NIE / Total / proportion-mediated posterior summary.

    ``med.mediator_kind`` selects the mediator counterfactual: ``"beta_binomial"``
    (LRP59, a single count mediator) or ``"gaussian_composite"`` (LRP62, a
    continuous standardised phonics-route composite). The outcome model and the
    NDE/NIE/proportion decomposition are identical in both cases. The mediator is
    re-simulated ``n_replicates`` times per posterior draw and averaged, to
    control Monte-Carlo noise from the mediator draw; the posterior itself
    supplies the inferential uncertainty.
    """
    post = trace.posterior
    # Confounders come from the fitted model (via MediationData), so the
    # g-formula adjusts for exactly the set the outcome/mediator legs included —
    # the symbols cannot drift between the fit and the counterfactual simulation.
    confounder_symbols = med.confounder_symbols

    def d(name: str) -> np.ndarray:
        return post[name].stack(_s=("chain", "draw")).values  # (S,)

    # --- Outcome model (shared across mediator kinds) ---
    b0, b_G, b_M, b_GM, b_W, b_A = (
        d("b0"), d("b_G"), d("b_M"), d("b_GM"), d("b_W"), d("b_A")
    )
    b_conf = {s: d(f"b_{s}") for s in confounder_symbols}

    # Covariates as (1, n) row vectors for broadcasting against (S, 1) draws.
    W1 = med.W1_logit[None, :]
    A = med.A_std[None, :]
    conf = {s: med.conf_logit[s][None, :] for s in confounder_symbols}
    N_W = med.n_trials_W
    S = b0.shape[0]
    rng = np.random.default_rng(seed)

    def outcome_p(g: float, z_m: np.ndarray) -> np.ndarray:
        eta = (
            b0[:, None]
            + b_G[:, None] * g
            + b_M[:, None] * z_m
            + b_GM[:, None] * (g * z_m)
            + b_W[:, None] * W1
            + b_A[:, None] * A
        )
        for s in confounder_symbols:
            eta = eta + b_conf[s][:, None] * conf[s]
        return _sigmoid(eta)

    # E[Y(g_out, M(g'))] averaged over units, accumulated over mediator replicates.
    y_treat_Mtreat = np.zeros(S)
    y_ctrl_Mctrl = np.zeros(S)
    y_treat_Mctrl = np.zeros(S)

    if med.mediator_kind == "gaussian_composite":
        # LRP62: continuous standardised route composite ~ Normal. The mediator
        # is already on the standardised scale the outcome model consumes, so the
        # simulated value IS z_m (no logit/Beta-Binomial round-trip).
        a0, a_G, a_comp, a_A = d("a0"), d("a_G"), d("a_comp"), d("a_A")
        a_conf = {s: d(f"a_{s}") for s in confounder_symbols}
        sigma_M = d("sigma_M")
        Mpre = med.M_pre_std[None, :]

        def mediator_mu(g: float) -> np.ndarray:
            mu = a0[:, None] + a_G[:, None] * g + a_comp[:, None] * Mpre + a_A[:, None] * A
            for s in confounder_symbols:
                mu = mu + a_conf[s][:, None] * conf[s]
            return mu

        mu_treat, mu_ctrl = mediator_mu(_TREAT), mediator_mu(_CTRL)
        for _ in range(n_replicates):
            zm_treat = rng.normal(mu_treat, sigma_M[:, None])
            zm_ctrl = rng.normal(mu_ctrl, sigma_M[:, None])
            y_treat_Mtreat += outcome_p(_TREAT, zm_treat).mean(axis=1)
            y_ctrl_Mctrl += outcome_p(_CTRL, zm_ctrl).mean(axis=1)
            y_treat_Mctrl += outcome_p(_TREAT, zm_ctrl).mean(axis=1)
    else:
        # LRP59: single count mediator ~ Beta-Binomial, standardised logit -> z_m.
        a0, a_G, a_L, a_A = d("a0"), d("a_G"), d("a_L"), d("a_A")
        a_conf = {s: d(f"a_{s}") for s in confounder_symbols}
        kappa_M = d("kappa_M")
        L1 = med.L1_logit[None, :]
        N_L = med.n_trials_L

        def mediator_p(g: float) -> np.ndarray:
            mu = a0[:, None] + a_G[:, None] * g + a_L[:, None] * L1 + a_A[:, None] * A
            for s in confounder_symbols:
                mu = mu + a_conf[s][:, None] * conf[s]
            return np.clip(_sigmoid(mu), EPSILON, 1 - EPSILON)

        p_treat = mediator_p(_TREAT)  # (S, n)
        p_ctrl = mediator_p(_CTRL)
        a_beta_t, b_beta_t = p_treat * kappa_M[:, None], (1 - p_treat) * kappa_M[:, None]
        a_beta_c, b_beta_c = p_ctrl * kappa_M[:, None], (1 - p_ctrl) * kappa_M[:, None]
        for _ in range(n_replicates):
            k_treat = rng.binomial(N_L, rng.beta(a_beta_t, b_beta_t))
            k_ctrl = rng.binomial(N_L, rng.beta(a_beta_c, b_beta_c))
            zm_treat = (logit_safe(k_treat, N_L) - med.med_mean) / med.med_sd
            zm_ctrl = (logit_safe(k_ctrl, N_L) - med.med_mean) / med.med_sd
            y_treat_Mtreat += outcome_p(_TREAT, zm_treat).mean(axis=1)
            y_ctrl_Mctrl += outcome_p(_CTRL, zm_ctrl).mean(axis=1)
            y_treat_Mctrl += outcome_p(_TREAT, zm_ctrl).mean(axis=1)

    y_treat_Mtreat /= n_replicates
    y_ctrl_Mctrl /= n_replicates
    y_treat_Mctrl /= n_replicates

    total = y_treat_Mtreat - y_ctrl_Mctrl  # probability scale, intervention-helps
    nde = y_treat_Mctrl - y_ctrl_Mctrl
    nie = y_treat_Mtreat - y_treat_Mctrl

    lo_q, hi_q = (1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2

    def row(name: str, draws: np.ndarray) -> dict:
        return {
            "quantity": name,
            "prob_mean": float(np.mean(draws)),
            "prob_lo": float(np.quantile(draws, lo_q)),
            "prob_hi": float(np.quantile(draws, hi_q)),
            "words_mean": float(np.mean(draws) * N_W),
            "words_lo": float(np.quantile(draws, lo_q) * N_W),
            "words_hi": float(np.quantile(draws, hi_q) * N_W),
            "prob_pos": float(np.mean(draws > 0)),
        }

    rows = [row("total", total), row("NDE", nde), row("NIE", nie)]

    # Proportion mediated = NIE / Total. The ratio is unstable when Total can
    # cross zero, so report the posterior median + interval and P(Total>0); the
    # decomposition itself (NDE/NIE) is the robust reading.
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = nie / total
    prop = prop[np.isfinite(prop)]
    rows.append(
        {
            "quantity": "proportion_mediated",
            "prob_mean": float(np.median(prop)),
            "prob_lo": float(np.quantile(prop, lo_q)),
            "prob_hi": float(np.quantile(prop, hi_q)),
            "words_mean": np.nan,
            "words_lo": np.nan,
            "words_hi": np.nan,
            "prob_pos": float(np.mean(total > 0)),  # P(Total > 0) for context
        }
    )
    return pd.DataFrame(rows)


def decompose_two_mediator(
    trace: xr.DataTree,
    med: TwoMediatorData,
    *,
    hdi_prob: float = 0.95,
    n_replicates: int = 50,
    seed: int = 47,
    order: tuple[str, str] = ("L", "E"),
) -> pd.DataFrame:
    """Two-mediator g-formula decomposition for LRP64 (letter-sound L + vocab E -> W).

    Returns posteriors for, on the response (word-count / probability) scale:

    - ``total`` — the intervention's total effect on word reading.
    - ``NDE`` — the direct / residual path (neither mediator moves).
    - ``NIE_joint`` — the **headline**: the joint indirect effect through the
      ``{L, E}`` block (both mediators allowed to respond). This is the robust,
      assumption-light quantity (the same philosophy as LRP62's route composite).
    - ``NIE_L`` / ``NIE_E`` — the **exploratory** path-specific indirect effects,
      under the mediator ``order`` (default L before E): ``NIE_L`` moves L from its
      control to its treated draw with E held at control, then ``NIE_E`` moves E
      with L held at treated. They sum exactly to ``NIE_joint`` but the split
      depends on the order and on the cross-world / conditional-independence
      assumptions — read as ordering-dependent, not a unique attribution.
    - ``proportion_mediated`` — ``NIE_joint / total`` (median + interval).

    Each mediator is re-simulated from its Beta-Binomial leg under each treatment
    arm, with common random draws reused across counterfactual cells to reduce
    Monte-Carlo noise. Sign convention as :func:`decompose` (intervention-helps;
    ``G=0`` intervention, ``G=1`` control). The two mediators are simulated as
    conditionally independent given the covariates (no residual L-E correlation is
    modelled) — a simplifying assumption stated in the report.
    """
    post = trace.posterior

    def d(name: str) -> np.ndarray:
        return post[name].stack(_s=("chain", "draw")).values

    confs = med.confounder_symbols

    # Outcome model.
    b0, b_G, b_L, b_E, b_GL, b_GE, b_W, b_A = (
        d("b0"), d("b_G"), d("b_L"), d("b_E"), d("b_GL"), d("b_GE"), d("b_W"), d("b_A")
    )
    b_conf = {s: d(f"b_{s}") for s in confs}
    # Mediator legs.
    aL0, aL_G, aL_L, aL_A = d("aL0"), d("aL_G"), d("aL_L"), d("aL_A")
    aL_conf = {s: d(f"aL_{s}") for s in confs}
    kappa_L = d("kappa_L")
    aE0, aE_G, aE_E, aE_A = d("aE0"), d("aE_G"), d("aE_E"), d("aE_A")
    aE_conf = {s: d(f"aE_{s}") for s in confs}
    kappa_E = d("kappa_E")

    A = med.A_std[None, :]
    W1 = med.W1_logit[None, :]
    L1 = med.L1_logit[None, :]
    E1 = med.E1_logit[None, :]
    conf = {s: med.conf1_logit[s][None, :] for s in confs}
    N_W = med.n_trials_W
    N_L = med.n_trials_L
    N_E = med.n_trials_E
    S = b0.shape[0]
    rng = np.random.default_rng(seed)

    def outcome_p(g: float, zL: np.ndarray, zE: np.ndarray) -> np.ndarray:
        eta = (
            b0[:, None]
            + b_G[:, None] * g
            + b_L[:, None] * zL
            + b_E[:, None] * zE
            + b_GL[:, None] * (g * zL)
            + b_GE[:, None] * (g * zE)
            + b_W[:, None] * W1
            + b_A[:, None] * A
        )
        for s in confs:
            eta = eta + b_conf[s][:, None] * conf[s]
        return _sigmoid(eta)

    def mediator_p(a0, a_G, a_base, a_A, a_conf, base, g):
        mu = a0[:, None] + a_G[:, None] * g + a_base[:, None] * base + a_A[:, None] * A
        for s in confs:
            mu = mu + a_conf[s][:, None] * conf[s]
        return np.clip(_sigmoid(mu), EPSILON, 1 - EPSILON)

    pL_t = mediator_p(aL0, aL_G, aL_L, aL_A, aL_conf, L1, _TREAT)
    pL_c = mediator_p(aL0, aL_G, aL_L, aL_A, aL_conf, L1, _CTRL)
    pE_t = mediator_p(aE0, aE_G, aE_E, aE_A, aE_conf, E1, _TREAT)
    pE_c = mediator_p(aE0, aE_G, aE_E, aE_A, aE_conf, E1, _CTRL)

    def zdraw(p, kappa, N, mean, sd):
        k = rng.binomial(N, rng.beta(p * kappa[:, None], (1 - p) * kappa[:, None]))
        return (logit_safe(k, N) - mean) / sd

    # Counterfactual cells (g_out, L-arm, E-arm); common draws across cells.
    y_TT_TT = np.zeros(S)  # treat outcome, both mediators treated
    y_TT_CC = np.zeros(S)  # treat outcome, both mediators control
    y_CC_CC = np.zeros(S)  # control outcome, both mediators control
    y_T_Lt_Ec = np.zeros(S)  # treat outcome, L treated, E control
    y_T_Lc_Et = np.zeros(S)  # treat outcome, L control, E treated
    for _ in range(n_replicates):
        zL_t = zdraw(pL_t, kappa_L, N_L, med.zL_mean, med.zL_sd)
        zL_c = zdraw(pL_c, kappa_L, N_L, med.zL_mean, med.zL_sd)
        zE_t = zdraw(pE_t, kappa_E, N_E, med.zE_mean, med.zE_sd)
        zE_c = zdraw(pE_c, kappa_E, N_E, med.zE_mean, med.zE_sd)
        y_TT_TT += outcome_p(_TREAT, zL_t, zE_t).mean(axis=1)
        y_TT_CC += outcome_p(_TREAT, zL_c, zE_c).mean(axis=1)
        y_CC_CC += outcome_p(_CTRL, zL_c, zE_c).mean(axis=1)
        y_T_Lt_Ec += outcome_p(_TREAT, zL_t, zE_c).mean(axis=1)
        y_T_Lc_Et += outcome_p(_TREAT, zL_c, zE_t).mean(axis=1)
    for arr in (y_TT_TT, y_TT_CC, y_CC_CC, y_T_Lt_Ec, y_T_Lc_Et):
        arr /= n_replicates

    total = y_TT_TT - y_CC_CC
    nde = y_TT_CC - y_CC_CC
    nie_joint = y_TT_TT - y_TT_CC
    if order == ("L", "E"):
        nie_L = y_T_Lt_Ec - y_TT_CC  # move L first (E at control)
        nie_E = y_TT_TT - y_T_Lt_Ec  # then move E (L at treated)
    else:
        nie_E = y_T_Lc_Et - y_TT_CC  # move E first (L at control)
        nie_L = y_TT_TT - y_T_Lc_Et  # then move L (E at treated)

    lo_q, hi_q = (1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2

    def row(name: str, draws: np.ndarray) -> dict:
        return {
            "quantity": name,
            "prob_mean": float(np.mean(draws)),
            "prob_lo": float(np.quantile(draws, lo_q)),
            "prob_hi": float(np.quantile(draws, hi_q)),
            "words_mean": float(np.mean(draws) * N_W),
            "words_lo": float(np.quantile(draws, lo_q) * N_W),
            "words_hi": float(np.quantile(draws, hi_q) * N_W),
            "prob_pos": float(np.mean(draws > 0)),
        }

    rows = [
        row("total", total),
        row("NDE", nde),
        row("NIE_joint", nie_joint),
        row("NIE_L", nie_L),
        row("NIE_E", nie_E),
    ]

    with np.errstate(divide="ignore", invalid="ignore"):
        prop = nie_joint / total
    prop = prop[np.isfinite(prop)]
    rows.append(
        {
            "quantity": "proportion_mediated",
            "prob_mean": float(np.median(prop)),
            "prob_lo": float(np.quantile(prop, lo_q)),
            "prob_hi": float(np.quantile(prop, hi_q)),
            "words_mean": np.nan,
            "words_lo": np.nan,
            "words_hi": np.nan,
            "prob_pos": float(np.mean(total > 0)),  # P(Total > 0) for context
        }
    )
    return pd.DataFrame(rows)
