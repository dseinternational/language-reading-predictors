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

Repo sign convention: ``G = 2 - group`` with ``G = 1`` = immediate-intervention
arm and ``G = 0`` = wait-list control in phase 0. Effects are reported in the
**intervention-helps** direction (intervention minus control), so positive =
intervention raises reading. With ``treat = 1`` (intervention) and ``ctrl = 0``
(control), and ``M(g)`` the mediator simulated under arm ``g``:

    Total = E[Y(treat, M(treat))] - E[Y(ctrl, M(ctrl))]
    NDE   = E[Y(treat, M(ctrl))]  - E[Y(ctrl, M(ctrl))]
    NIE   = E[Y(treat, M(treat))] - E[Y(treat, M(ctrl))]
    proportion mediated = NIE / Total

reported on the response scale (probability and word-count out of N_W), which is
the natural and interpretable scale for the g-formula decomposition.

These are **not identified natural effects**. Two structural obstacles block a
natural NDE/NIE interpretation, and neither is removed by more data: (1) every
mediator->outcome path is confounded by latent general ability (and, at a single
wave, by shared timing); (2) independently, intervention dose ``IS`` (sessions
attended) is a *treatment-induced (exposure-induced) mediator-outcome confounder*
-- ``IG -> IS`` and ``IS`` points into both the candidate mediators and word
reading -- so natural direct/indirect effects are not identified even by
randomising ``IG`` and are **not** repaired by measuring and adjusting ``IS``,
because ``IS`` is itself a descendant of the exposure (VanderWeele, Vansteelandt
& Robins 2014, Epidemiology 25(2):300-306, doi:10.1097/EDE.0000000000000034).
The outputs are therefore model-based g-formula decompositions under the stated
(cross-world) assumptions, not identified natural effects.

An *interventional* (rather than natural) mediation estimand -- available here via
``decompose(..., interventional=True)`` and fitted in MED-078/186/187 -- addresses
the *second* obstacle only: it invokes no cross-world quantity, so the
exposure-induced confounder does not defeat it. It does **not** address the first.
Identification of
stochastic interventional (in)direct effects still requires no unmeasured
mediator-outcome confounding (Hejazi, Rudolph, van der Laan & Diaz 2022,
Biostatistics 24(3):686-707, assumption A5, doi:10.1093/biostatistics/kxac002),
which latent general ability violates here; their positivity requirement (A3) and
the absence of temporal ordering between mediator and outcome also remain binding.
Treat IDE/IIE as a *weaker-assumption* target, not an identified one.

The exposure-induced structure is encoded in the base DAG
(``dag/dag-language-reading.dagitty``: ``IG -> IS`` and
``IS -> { TR TE PA LS WR ... }``); see also the model reports' assumptions
sections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.math.constants import EPSILON

from language_reading_predictors.statistical_models.factories import (
    MediationData,
    PeriodStackedMediationData,
    TwoMediatorData,
)
from language_reading_predictors.statistical_models.preprocessing import (
    PreparedData,
    logit_safe,
    standardise,
)

_TREAT = 1.0  # immediate-intervention arm (G = 1)
_CTRL = 0.0  # wait-list control arm (G = 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _effect_row(
    name: str,
    draws: np.ndarray,
    n_words: int,
    lo_q: float,
    hi_q: float,
    off_floor: bool,
    *,
    n_chains: int | None = None,
    n_draws: int | None = None,
) -> dict:
    """One decomposition-quantity row (shared by the single-mediator g-formulas)."""
    row = {
        "quantity": name,
        # Median-first (house convention; #268). Mean kept as a secondary column.
        "prob_median": float(np.median(draws)),
        "prob_mean": float(np.mean(draws)),
        "prob_lo": float(np.quantile(draws, lo_q)),
        "prob_hi": float(np.quantile(draws, hi_q)),
        "prob_lo50": float(np.quantile(draws, 0.25)),
        "prob_hi50": float(np.quantile(draws, 0.75)),
        "words_median": float(np.median(draws) * n_words),
        "words_mean": float(np.mean(draws) * n_words),
        "words_lo": float(np.quantile(draws, lo_q) * n_words),
        "words_hi": float(np.quantile(draws, hi_q) * n_words),
        "words_lo50": float(np.quantile(draws, 0.25) * n_words),
        "words_hi50": float(np.quantile(draws, 0.75) * n_words),
        "prob_pos": float(np.mean(draws > 0)),
        # For an off-floor outcome n_trials_W = 1, so the words_* columns equal
        # prob_* and both are the off-floor RISK DIFFERENCE; the flag lets the
        # report label the scale ("off-floor risk difference", not items).
        "off_floor": bool(off_floor),
    }
    # Monte-Carlo precision of this derived g-formula effect. Because the NDE/NIE
    # are simulated (mediator re-draw noise on top of posterior autocorrelation),
    # their tail ESS can be worse than the parent coefficients the gate checks —
    # so report it per row (Kruschke 2021 BARG step 2.C).
    if n_chains is not None and n_draws is not None:
        from language_reading_predictors.statistical_models.reporting import (
            derived_mc_diagnostics,
        )

        row.update(
            derived_mc_diagnostics(draws, n_chains=n_chains, n_draws=n_draws)
        )
    return row


def _proportion_row(nie: np.ndarray, total: np.ndarray, lo_q: float, hi_q: float) -> dict:
    """The proportion-mediated row (NIE / Total; unstable when Total crosses 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = nie / total
    prop = prop[np.isfinite(prop)]
    # Guard the degenerate all-non-finite case (e.g. Total == 0 on every draw):
    # np.quantile / np.median on an empty array raise, so report NaN intervals
    # instead. The decomposition itself (NDE/NIE/Total) is the robust reading;
    # the proportion is a fragile ratio at the best of times.
    if prop.size == 0:
        prop_stats = dict.fromkeys(
            ("prob_median", "prob_mean", "prob_lo", "prob_hi", "prob_lo50", "prob_hi50"),
            float("nan"),
        )
    else:
        prop_stats = {
            # A ratio: report its median (the mean is unstable when Total crosses
            # zero); no longer overloaded onto prob_mean (#268).
            "prob_median": float(np.median(prop)),
            "prob_mean": float(np.mean(prop)),
            "prob_lo": float(np.quantile(prop, lo_q)),
            "prob_hi": float(np.quantile(prop, hi_q)),
            "prob_lo50": float(np.quantile(prop, 0.25)),
            "prob_hi50": float(np.quantile(prop, 0.75)),
        }
    return {
        "quantity": "proportion_mediated",
        **prop_stats,
        "words_median": np.nan,
        "words_mean": np.nan,
        "words_lo": np.nan,
        "words_hi": np.nan,
        "words_lo50": np.nan,
        "words_hi50": np.nan,
        "prob_pos": float(np.mean(total > 0)),  # P(Total > 0) for context
    }


def decompose(
    trace: xr.DataTree,
    med: MediationData,
    *,
    ci_prob: float = 0.95,
    n_replicates: int = 50,
    seed: int = 47,
    interventional: bool = False,
    b_m_shift: float = 0.0,
) -> pd.DataFrame:
    """Return the NDE / NIE / Total / proportion-mediated posterior summary.

    ``med.mediator_kind`` selects the mediator counterfactual: ``"beta_binomial"``
    (LRP59, a single count mediator) or ``"gaussian_composite"`` (LRP62, a
    continuous standardised code-based-route composite). The outcome model and the
    NDE/NIE/proportion decomposition are identical in both cases. The mediator is
    re-simulated ``n_replicates`` times per posterior draw and averaged, to
    control Monte-Carlo noise from the mediator draw; the posterior itself
    supplies the inferential uncertainty.

    ``interventional=True`` (MED-078/186/187) labels the decomposition as the
    **randomised interventional analogue** (IDE / IIE) rather than the natural
    NDE / NIE. The
    mediator is drawn from its fitted **covariate-conditional** law ``P(M | C, g)``
    within strata (VanderWeele, Vansteelandt & Robins 2014) — which is exactly the
    distribution the g-formula already simulates. In this fully parametric model
    with no unit-level latent terms the interventional draw therefore **coincides
    numerically** with the natural-branch computation: the ``interventional`` flag
    changes the estimand's *interpretation* (an interventional effect requiring no
    cross-world quantity, only no-unmeasured-mediator-outcome-confounding; #260),
    not the number. It is **not** a marginal population permutation, and it does
    **not** turn a treatment-induced dose (``IS``) confounder into an identified
    quantity — ``IS`` never enters the fitted model, so the interventional
    functional cannot repair, or diagnose, dose confounding. Rows are labelled
    ``IDE``/``IIE`` and the proportion is ``IIE/Total``.
    """
    post = trace.posterior
    # Confounders come from the fitted model (via MediationData), so the
    # g-formula adjusts for exactly the set the outcome/mediator legs included —
    # the symbols cannot drift between the fit and the counterfactual simulation.
    confounder_symbols = med.confounder_symbols

    def d(name: str) -> np.ndarray:
        return post[name].stack(_s=("chain", "draw")).values  # (S,)

    # --- Outcome model (shared across mediator kinds) ---
    # An off-floor (Bernoulli) outcome (#228 item 12, e.g. nonword N) drops the
    # own-baseline term b_W, so it is absent from the trace; graded outcomes keep it.
    off_floor = getattr(med, "off_floor", False)
    b0, b_G, b_M, b_GM, b_A = d("b0"), d("b_G"), d("b_M"), d("b_GM"), d("b_A")
    b_W = None if off_floor else d("b_W")
    # Sensitivity lever (#230): subtract a bias delta from the mediator->outcome
    # coefficient — the portion of the fitted b_M one attributes to an unmeasured
    # mediator-outcome confounder. b_m_shift=0 is the primary (identified) analysis.
    b_M = b_M - b_m_shift
    b_conf = {s: d(f"b_{s}") for s in confounder_symbols}

    # Covariates as (1, n) row vectors for broadcasting against (S, 1) draws.
    W1 = med.W1_logit[None, :]
    A = med.A_std[None, :]
    conf = {s: med.conf_logit[s][None, :] for s in confounder_symbols}
    N_W = med.n_trials_W
    S = b0.shape[0]
    rng = np.random.default_rng(seed)

    def outcome_p(g: float, z_m: np.ndarray) -> np.ndarray:
        if off_floor:
            # No own-baseline term for the off-floor (Bernoulli) outcome.
            eta = (
                b0[:, None]
                + b_G[:, None] * g
                + b_M[:, None] * z_m
                + b_GM[:, None] * (g * z_m)
                + b_A[:, None] * A
            )
        else:
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

    def draw_m(zm: np.ndarray) -> np.ndarray:
        # Both the natural and the interventional analogue draw the mediator from
        # its fitted COVARIATE-CONDITIONAL law P(M | C, g) — exactly what
        # ``mediator_p`` / ``mediator_mu`` simulate above (``zm``). In this fully
        # parametric g-formula with no unit-level latent terms, the within-stratum
        # interventional draw therefore coincides *numerically* with the
        # natural-branch computation, so ``interventional`` is an interpretive
        # relabelling (an interventional IDE/IIE under weaker cross-world
        # assumptions; #260, #268), NOT a different number.
        #
        # It is deliberately NOT the marginal population permutation an earlier
        # version used (``zm[:, rng.permutation(...)]``): that draws M from the
        # marginal arm-g distribution, pairing mediator values with off-support
        # covariate profiles through the nonlinear G×M logit, which targets a
        # cruder estimand — and, crucially, its divergence from the natural
        # decomposition reflects that covariate decoupling, NOT the treatment-
        # induced dose (IS) confounding (IS never enters the fitted model), so it
        # cannot support an IIE-vs-NIE dose-distortion diagnostic.
        return zm

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
            y_treat_Mtreat += outcome_p(_TREAT, draw_m(zm_treat)).mean(axis=1)
            y_ctrl_Mctrl += outcome_p(_CTRL, draw_m(zm_ctrl)).mean(axis=1)
            y_treat_Mctrl += outcome_p(_TREAT, draw_m(zm_ctrl)).mean(axis=1)
    else:
        # LRP59: single count mediator ~ Beta-Binomial, standardised logit -> z_m.
        # The own-baseline coef is a_{mediator_symbol} (a_L for L, a_TE for TE, ...).
        a0, a_G, a_L, a_A = d("a0"), d("a_G"), d(f"a_{med.mediator_symbol}"), d("a_A")
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
            y_treat_Mtreat += outcome_p(_TREAT, draw_m(zm_treat)).mean(axis=1)
            y_ctrl_Mctrl += outcome_p(_CTRL, draw_m(zm_ctrl)).mean(axis=1)
            y_treat_Mctrl += outcome_p(_TREAT, draw_m(zm_ctrl)).mean(axis=1)

    y_treat_Mtreat /= n_replicates
    y_ctrl_Mctrl /= n_replicates
    y_treat_Mctrl /= n_replicates

    total = y_treat_Mtreat - y_ctrl_Mctrl  # probability scale, intervention-helps
    nde = y_treat_Mctrl - y_ctrl_Mctrl
    nie = y_treat_Mtreat - y_treat_Mctrl

    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2

    _nc, _nd = int(post.sizes["chain"]), int(post.sizes["draw"])

    def row(name: str, draws: np.ndarray) -> dict:
        return _effect_row(
            name, draws, N_W, lo_q, hi_q, off_floor, n_chains=_nc, n_draws=_nd
        )

    direct_label, indirect_label = ("IDE", "IIE") if interventional else ("NDE", "NIE")
    rows = [row("total", total), row(direct_label, nde), row(indirect_label, nie)]

    # Proportion mediated = NIE / Total. The ratio is unstable when Total can
    # cross zero, so report the posterior median + interval and P(Total>0); the
    # decomposition itself (NDE/NIE) is the robust reading.
    rows.append(_proportion_row(nie, total, lo_q, hi_q))
    return pd.DataFrame(rows)


def sensitivity_sweep(
    trace: xr.DataTree,
    med: MediationData | PeriodStackedMediationData,
    *,
    ci_prob: float = 0.95,
    n_deltas: int = 21,
    delta_max: float | None = None,
    decompose_fn=None,
    interaction_name: str = "b_GM",
    **decompose_kw,
) -> tuple[pd.DataFrame, dict]:
    """Unmeasured mediator-outcome confounding sensitivity for the NIE (#230).

    Sweeps a **non-negative** bias magnitude ``delta`` applied to the mediator->outcome
    coefficient ``b_M`` in the direction that attenuates the fitted effective slope
    toward zero (the share of the fitted mediator-outcome association one is willing to
    attribute to an unmeasured mediator-outcome confounder) and re-runs the g-formula
    (:func:`decompose`) at each. Returns the NIE across ``delta`` plus a
    summary whose headline is the **tipping point** ``delta*``: the confounding
    strength at which the NIE's ``ci_prob`` credible interval first includes 0.

    As a Bayesian E-value analogue, ``delta*`` is also expressed as a fraction of
    the fitted effective mediator-outcome slope (``b_M + b_GM`` at treatment) — how
    much of the mediator->reading association would have to be spurious confounding
    to null the indirect effect. Larger (or "robust across the full sweep") => more
    robust. This quantifies the no-unmeasured-mediator-outcome-confounding
    assumption the decomposition otherwise only states.

    ``decompose_fn`` (default :func:`decompose`) selects the decomposition the
    sweep re-runs — pass :func:`decompose_period_stacked` with
    ``interaction_name="b_trtM"`` for the period-stacked design (MED-092), whose
    exposure x mediator coefficient carries the on-intervention name.
    """
    if decompose_fn is None:
        decompose_fn = decompose
    post = trace.posterior

    def d(name: str) -> np.ndarray:
        return post[name].stack(_s=("chain", "draw")).values

    # Effective M->Y slope at treatment/exposure.
    ref = float(np.mean(d("b_M") + d(interaction_name)))
    ref_mag = abs(ref)
    ref_eps = 1e-6
    # ``delta`` is a NON-NEGATIVE magnitude of confounding, applied in the direction that
    # shrinks the fitted effective slope toward 0 (``b_m_shift = sign(ref) * delta``), so
    # the sweep attenuates the NIE toward the null whether the fitted slope is positive or
    # negative (#289 review — a fixed positive subtraction pushed a negative slope *away*
    # from 0 and silently reported "robust"). Fractions use ``abs(ref)`` so they stay a
    # positive "share of the fitted slope", with an epsilon guarding a near-zero slope
    # where they would otherwise explode.
    shrink_sign = 1.0 if ref >= 0 else -1.0
    if delta_max is None:
        delta_max = max(ref_mag * 1.5, 0.5)
    deltas = np.linspace(0.0, delta_max, n_deltas)

    indirect = {"NIE", "IIE"}
    rows = []
    for dlt in deltas:
        df = decompose_fn(
            trace, med, ci_prob=ci_prob, b_m_shift=float(shrink_sign * dlt), **decompose_kw
        )
        nie = df[df["quantity"].isin(indirect)].iloc[0]
        rows.append(
            {
                "delta": float(dlt),
                "delta_frac_of_bM": (
                    float(dlt / ref_mag) if ref_mag > ref_eps else float("nan")
                ),
                "nie_median": float(nie["prob_median"]),
                "nie_lo": float(nie["prob_lo"]),
                "nie_hi": float(nie["prob_hi"]),
                "nie_prob_pos": float(nie["prob_pos"]),
            }
        )
    sweep = pd.DataFrame(rows)

    # Tipping point: first delta at which the NIE interval includes 0, from the sign the
    # NIE takes at delta=0. The sweep shrinks the effective slope toward 0, so the NIE
    # moves toward 0 as delta grows regardless of the fitted slope's sign; if the interval
    # already includes 0 at delta=0 the indirect effect is not credibly nonzero, so the
    # sensitivity question ("how much confounding would null it?") does not apply.
    nie0 = sweep.iloc[0]
    already_null = bool(nie0["nie_lo"] <= 0 <= nie0["nie_hi"])
    positive = nie0["nie_median"] >= 0
    crossed = sweep[sweep["nie_lo"] <= 0] if positive else sweep[sweep["nie_hi"] >= 0]
    tip = float(crossed.iloc[0]["delta"]) if len(crossed) else float("nan")
    summary = {
        "b_M_effective_mean": ref,
        "already_null_at_zero": already_null,
        "tipping_delta": (float("nan") if already_null else tip),
        "tipping_frac_of_bM": (
            float(tip / ref_mag)
            if (ref_mag > ref_eps and not already_null and np.isfinite(tip))
            else float("nan")
        ),
        "nie_median_at_zero": float(nie0["nie_median"]),
        "robust_over_full_sweep": bool(not already_null and not np.isfinite(tip)),
    }
    return sweep, summary


def decompose_period_stacked(
    trace: xr.DataTree,
    med: PeriodStackedMediationData,
    *,
    ci_prob: float = 0.95,
    n_replicates: int = 50,
    seed: int = 47,
    b_m_shift: float = 0.0,
    row_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """NDE / NIE / Total for the period-stacked design (MED-092, #229).

    The :func:`decompose` g-formula transplanted onto the gain-factor scaffold
    (:func:`factories.build_period_stacked_mediation_model`): the exposure is
    the per-period **on-intervention** indicator, and the linear predictors add
    the per-phase intercepts and per-leg child random intercepts, all read at
    each row's own phase / child. Effects are averaged over the stacked rows'
    observed state (period-start baselines, phase, age, child intercept), so
    the estimand is the **per-period** switch-on-vs-off decomposition given
    where a child starts the period — under the gain-factor family's
    ignorability assumption, *not* randomisation. ``row_mask`` restricts the
    averaging rows (e.g. ``phase == 0`` for the period-1, ITT-anchored readout
    comparable with LRP59); the posterior itself is always the all-period fit.

    Everything :mod:`mediation`'s docstring says about non-identification
    carries over unchanged — model-based decomposition under stated
    assumptions, not identified natural effects. Sign convention: exposure = 1
    is on-intervention, so positive = being on the programme raises reading.
    """
    post = trace.posterior

    def d(name: str) -> np.ndarray:
        return post[name].stack(_s=("chain", "draw")).values  # (S,)

    def dvec(name: str, dim: str) -> np.ndarray:
        # Vector parameters as (S, len(dim)), row-indexable by phase/child idx.
        return post[name].stack(_s=("chain", "draw")).transpose("_s", dim).values

    mask = (
        np.ones(med.trt.shape[0], dtype=bool)
        if row_mask is None
        else np.asarray(row_mask, dtype=bool)
    )
    phase_idx = med.phase_idx[mask]
    child_idx = med.child_idx[mask]
    confs = med.confounder_symbols

    # --- Outcome leg ---
    b0, b_trt, b_M, b_trtM, b_W, b_A = (
        d("b0"), d("b_trt"), d("b_M"), d("b_trtM"), d("b_W"), d("b_A")
    )
    # Sensitivity lever (#230), as in :func:`decompose`.
    b_M = b_M - b_m_shift
    b_conf = {s: d(f"b_{s}") for s in confs}
    b_phase_rows = dvec("b_phase", "phase")[:, phase_idx]  # (S, n)
    uY_rows = dvec("u_child_Y", "child")[:, child_idx]

    # --- Mediator leg ---
    a0, a_trt, a_M, a_A = d("a0"), d("a_trt"), d(f"a_{med.mediator_symbol}"), d("a_A")
    a_conf = {s: d(f"a_{s}") for s in confs}
    a_phase_rows = dvec("a_phase", "phase")[:, phase_idx]
    uM_rows = dvec("u_child_M", "child")[:, child_idx]
    kappa_M = d("kappa_M")

    W1 = med.W1_logit[mask][None, :]
    L1 = med.L1_logit[mask][None, :]
    A = med.A_std[mask][None, :]
    conf = {s: med.conf_values[s][mask][None, :] for s in confs}
    N_W = med.n_trials_W
    N_L = med.n_trials_L
    S = b0.shape[0]
    rng = np.random.default_rng(seed)

    def outcome_p(t: float, z_m: np.ndarray) -> np.ndarray:
        eta = (
            b0[:, None]
            + b_phase_rows
            + b_trt[:, None] * t
            + b_M[:, None] * z_m
            + b_trtM[:, None] * (t * z_m)
            + b_W[:, None] * W1
            + b_A[:, None] * A
            + uY_rows
        )
        for s in confs:
            eta = eta + b_conf[s][:, None] * conf[s]
        return _sigmoid(eta)

    def mediator_p(t: float) -> np.ndarray:
        mu = (
            a0[:, None]
            + a_phase_rows
            + a_trt[:, None] * t
            + a_M[:, None] * L1
            + a_A[:, None] * A
            + uM_rows
        )
        for s in confs:
            mu = mu + a_conf[s][:, None] * conf[s]
        return np.clip(_sigmoid(mu), EPSILON, 1 - EPSILON)

    p_treat, p_ctrl = mediator_p(_TREAT), mediator_p(_CTRL)
    a_beta_t, b_beta_t = p_treat * kappa_M[:, None], (1 - p_treat) * kappa_M[:, None]
    a_beta_c, b_beta_c = p_ctrl * kappa_M[:, None], (1 - p_ctrl) * kappa_M[:, None]

    y_treat_Mtreat = np.zeros(S)
    y_ctrl_Mctrl = np.zeros(S)
    y_treat_Mctrl = np.zeros(S)
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

    total = y_treat_Mtreat - y_ctrl_Mctrl
    nde = y_treat_Mctrl - y_ctrl_Mctrl
    nie = y_treat_Mtreat - y_treat_Mctrl

    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    _nc, _nd = int(post.sizes["chain"]), int(post.sizes["draw"])
    rows = [
        _effect_row("total", total, N_W, lo_q, hi_q, False, n_chains=_nc, n_draws=_nd),
        _effect_row("NDE", nde, N_W, lo_q, hi_q, False, n_chains=_nc, n_draws=_nd),
        _effect_row("NIE", nie, N_W, lo_q, hi_q, False, n_chains=_nc, n_draws=_nd),
        _proportion_row(nie, total, lo_q, hi_q),
    ]
    return pd.DataFrame(rows)


def decompose_two_mediator(
    trace: xr.DataTree,
    med: TwoMediatorData,
    *,
    hdi_prob: float = 0.95,
    n_replicates: int = 50,
    seed: int = 47,
    order: tuple[str, str] = ("L", "E"),
    b_m_shifts: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Two-mediator g-formula decomposition for LRP64 (letter-sound L + vocab E -> W).

    Returns posteriors for, on the response (word-count / probability) scale:

    - ``total`` — the intervention's total effect on word reading.
    - ``NDE`` — the direct / residual path (neither mediator moves).
    - ``NIE_joint`` — the **preferred descriptive decomposition**: the joint
      indirect effect through the ``{L, E}`` block (both mediators allowed to
      respond). Its *only* advantage is that it is **less sensitive to mediator
      ordering** than the path split (the same philosophy as LRP62's route
      composite). It is **not** robust or identified in any stronger sense: like
      every quantity here it remains subject to latent-general-ability confounding
      and to exposure-induced ``IS`` confounding (see the module docstring).
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
    ``G=1`` intervention, ``G=0`` control, per ``G = 2 - group``). The two mediators are simulated as
    conditionally independent given the covariates (no residual L-E correlation is
    modelled) — a simplifying assumption stated in the report. ``b_m_shifts`` is
    the #335 sensitivity lever: a mapping from mediator symbol to the signed bias
    subtracted from that mediator's outcome-leg main slope before decomposition.
    """
    post = trace.posterior

    def d(name: str) -> np.ndarray:
        return post[name].stack(_s=("chain", "draw")).values

    confs = med.confounder_symbols
    # The first mediator leg is always L; the second is parameterised by its
    # symbol ``mE`` so the trace-variable names match the factory (LRP64 ``E``,
    # LRP66 ``B``). When mE == 'E' this reads exactly the original node names.
    mL, mE = med.mediator_symbols

    # Outcome model.
    b0, b_G, b_L, b_E, b_GL, b_GE, b_W, b_A = (
        d("b0"), d("b_G"), d("b_L"), d(f"b_{mE}"), d("b_GL"), d(f"b_G{mE}"), d("b_W"), d("b_A")
    )
    b_m_shifts = b_m_shifts or {}
    unknown_shifts = set(b_m_shifts) - {mL, mE}
    if unknown_shifts:
        raise ValueError(
            f"b_m_shifts contains unknown mediators {sorted(unknown_shifts)!r}; "
            f"expected a subset of {(mL, mE)!r}"
        )
    # Per-leg sensitivity lever (#335): subtract the portion of each fitted
    # mediator->outcome slope attributed to unmeasured confounding. The treatment
    # interactions stay fixed, matching the single-mediator ``b_m_shift`` lever.
    b_L = b_L - float(b_m_shifts.get(mL, 0.0))
    b_E = b_E - float(b_m_shifts.get(mE, 0.0))
    b_conf = {s: d(f"b_{s}") for s in confs}
    # Mediator legs.
    aL0, aL_G, aL_L, aL_A = d("aL0"), d("aL_G"), d("aL_L"), d("aL_A")
    aL_conf = {s: d(f"aL_{s}") for s in confs}
    kappa_L = d("kappa_L")
    aE0, aE_G, aE_E, aE_A = d(f"a{mE}0"), d(f"a{mE}_G"), d(f"a{mE}_{mE}"), d(f"a{mE}_A")
    aE_conf = {s: d(f"a{mE}_{s}") for s in confs}
    kappa_E = d(f"kappa_{mE}")

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

    # Sequential code route (LRP75): the second mediator regresses on post-L via
    # a{mE}_L, so its probability depends on the *simulated* first mediator (zL).
    aE_L = d(f"a{mE}_{mL}") if med.chain else None

    def mediator2_p(g, zL_val):
        """Second-mediator success prob under exposure ``g``; chained on ``zL_val``
        (the simulated first mediator on its standardised scale) when med.chain."""
        mu = aE0[:, None] + aE_G[:, None] * g + aE_E[:, None] * E1 + aE_A[:, None] * A
        for s in confs:
            mu = mu + aE_conf[s][:, None] * conf[s]
        if med.chain:
            mu = mu + aE_L[:, None] * zL_val
        return np.clip(_sigmoid(mu), EPSILON, 1 - EPSILON)

    pL_t = mediator_p(aL0, aL_G, aL_L, aL_A, aL_conf, L1, _TREAT)
    pL_c = mediator_p(aL0, aL_G, aL_L, aL_A, aL_conf, L1, _CTRL)
    # Parallel model: the second mediator is independent of L, so precompute once.
    if not med.chain:
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
        if med.chain:
            # Draw the second mediator conditional on the simulated L: treated B
            # under treated L, control B under control L (carrying L -> B -> W into
            # the joint indirect effect).
            zE_t = zdraw(mediator2_p(_TREAT, zL_t), kappa_E, N_E, med.zE_mean, med.zE_sd)
            zE_c = zdraw(mediator2_p(_CTRL, zL_c), kappa_E, N_E, med.zE_mean, med.zE_sd)
        else:
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
    if order not in ((mL, mE), (mE, mL)):
        raise ValueError(
            f"order must be {(mL, mE)!r} or {(mE, mL)!r}; got {order!r}"
        )
    if order == (mL, mE):
        nie_L = y_T_Lt_Ec - y_TT_CC  # move L first (mE at control)
        nie_E = y_TT_TT - y_T_Lt_Ec  # then move mE (L at treated)
    else:
        nie_E = y_T_Lc_Et - y_TT_CC  # move mE first (L at control)
        nie_L = y_TT_TT - y_T_Lc_Et  # then move L (mE at treated)

    lo_q, hi_q = (1 - hdi_prob) / 2, 1 - (1 - hdi_prob) / 2
    _nc, _nd = int(post.sizes["chain"]), int(post.sizes["draw"])

    def row(name: str, draws: np.ndarray) -> dict:
        from language_reading_predictors.statistical_models.reporting import (
            derived_mc_diagnostics,
        )

        return {
            "quantity": name,
            # Median-first (house convention; #268). Mean kept as a secondary column.
            "prob_median": float(np.median(draws)),
            "prob_mean": float(np.mean(draws)),
            "prob_lo": float(np.quantile(draws, lo_q)),
            "prob_hi": float(np.quantile(draws, hi_q)),
            "prob_lo50": float(np.quantile(draws, 0.25)),
            "prob_hi50": float(np.quantile(draws, 0.75)),
            "words_median": float(np.median(draws) * N_W),
            "words_mean": float(np.mean(draws) * N_W),
            "words_lo": float(np.quantile(draws, lo_q) * N_W),
            "words_hi": float(np.quantile(draws, hi_q) * N_W),
            "words_lo50": float(np.quantile(draws, 0.25) * N_W),
            "words_hi50": float(np.quantile(draws, 0.75) * N_W),
            "prob_pos": float(np.mean(draws > 0)),
            # Derived g-formula effect: report its own MC precision (see _effect_row).
            **derived_mc_diagnostics(draws, n_chains=_nc, n_draws=_nd),
        }

    rows = [
        row("total", total),
        row("NDE", nde),
        row("NIE_joint", nie_joint),
        row(f"NIE_{mL}", nie_L),
        row(f"NIE_{mE}", nie_E),
    ]

    with np.errstate(divide="ignore", invalid="ignore"):
        prop = nie_joint / total
    prop = prop[np.isfinite(prop)]
    rows.append(
        {
            "quantity": "proportion_mediated",
            # A ratio: report its median (the mean is unstable when Total crosses
            # zero); no longer overloaded onto prob_mean (#268).
            "prob_median": float(np.median(prop)),
            "prob_mean": float(np.mean(prop)),
            "prob_lo": float(np.quantile(prop, lo_q)),
            "prob_hi": float(np.quantile(prop, hi_q)),
            "prob_lo50": float(np.quantile(prop, 0.25)),
            "prob_hi50": float(np.quantile(prop, 0.75)),
            "words_median": np.nan,
            "words_mean": np.nan,
            "words_lo": np.nan,
            "words_hi": np.nan,
            "words_lo50": np.nan,
            "words_hi50": np.nan,
            "prob_pos": float(np.mean(total > 0)),  # P(Total > 0) for context
        }
    )
    return pd.DataFrame(rows)


def _tipping_summary(
    sweep: pd.DataFrame,
    *,
    median_col: str,
    lo_col: str,
    hi_col: str,
) -> dict[str, float | bool]:
    """Summarise when one swept effect's credible interval first reaches zero."""
    effect0 = sweep.iloc[0]
    already_null = bool(effect0[lo_col] <= 0 <= effect0[hi_col])
    positive = effect0[median_col] >= 0
    crossed = sweep[sweep[lo_col] <= 0] if positive else sweep[sweep[hi_col] >= 0]
    tip = float(crossed.iloc[0]["delta"]) if len(crossed) else float("nan")
    return {
        "already_null_at_zero": already_null,
        "tipping_delta": float("nan") if already_null else tip,
        "effect_median_at_zero": float(effect0[median_col]),
        "robust_over_full_sweep": bool(not already_null and not np.isfinite(tip)),
    }


def sensitivity_sweep_two_mediator(
    trace: xr.DataTree,
    med: TwoMediatorData,
    *,
    ci_prob: float = 0.95,
    n_deltas: int = 21,
    delta_max: float | dict[str, float] | None = None,
    **decompose_kw,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-leg unmeasured-confounding sweeps for a two-mediator decomposition.

    Each one-dimensional sweep attenuates one mediator->outcome leg toward zero
    while leaving the other leg unchanged. It records both the path-specific NIE
    and ``NIE_joint`` so the robustness of the preferred block-level indirect
    effect is visible under confounding of either leg. A two-dimensional surface is
    deliberately not generated: it is substantially more expensive and, at this
    sample size, would add a large grid without improving the leg-specific tipping
    decisions (#335).
    """
    post = trace.posterior

    def d(name: str) -> np.ndarray:
        return post[name].stack(_s=("chain", "draw")).values

    mL, mE = med.mediator_symbols
    coefficient_names = {
        mL: ("b_L", "b_GL"),
        mE: (f"b_{mE}", f"b_G{mE}"),
    }
    rows: list[dict] = []
    summaries: list[dict] = []
    ref_eps = 1e-6

    for mediator in (mL, mE):
        main_name, interaction_name = coefficient_names[mediator]
        ref = float(np.mean(d(main_name) + d(interaction_name)))
        ref_mag = abs(ref)
        shrink_sign = 1.0 if ref >= 0 else -1.0
        if isinstance(delta_max, dict):
            leg_delta_max = float(delta_max[mediator])
        elif delta_max is None:
            leg_delta_max = max(ref_mag * 1.5, 0.5)
        else:
            leg_delta_max = float(delta_max)

        leg_rows = []
        for dlt in np.linspace(0.0, leg_delta_max, n_deltas):
            decomposition = decompose_two_mediator(
                trace,
                med,
                hdi_prob=ci_prob,
                b_m_shifts={mediator: float(shrink_sign * dlt)},
                **decompose_kw,
            ).set_index("quantity")
            leg = decomposition.loc[f"NIE_{mediator}"]
            joint = decomposition.loc["NIE_joint"]
            row = {
                "mediator": mediator,
                "quantity": f"NIE_{mediator}",
                "delta": float(dlt),
                "delta_frac_of_effective_slope": (
                    float(dlt / ref_mag) if ref_mag > ref_eps else float("nan")
                ),
                "nie_median": float(leg["prob_median"]),
                "nie_lo": float(leg["prob_lo"]),
                "nie_hi": float(leg["prob_hi"]),
                "nie_prob_pos": float(leg["prob_pos"]),
                "nie_joint_median": float(joint["prob_median"]),
                "nie_joint_lo": float(joint["prob_lo"]),
                "nie_joint_hi": float(joint["prob_hi"]),
                "nie_joint_prob_pos": float(joint["prob_pos"]),
            }
            leg_rows.append(row)
            rows.append(row)

        leg_sweep = pd.DataFrame(leg_rows)
        leg_tip = _tipping_summary(
            leg_sweep,
            median_col="nie_median",
            lo_col="nie_lo",
            hi_col="nie_hi",
        )
        joint_tip = _tipping_summary(
            leg_sweep,
            median_col="nie_joint_median",
            lo_col="nie_joint_lo",
            hi_col="nie_joint_hi",
        )
        summaries.append(
            {
                "mediator": mediator,
                "quantity": f"NIE_{mediator}",
                "effective_slope_mean": ref,
                "already_null_at_zero": leg_tip["already_null_at_zero"],
                "tipping_delta": leg_tip["tipping_delta"],
                "tipping_frac_of_effective_slope": (
                    float(leg_tip["tipping_delta"] / ref_mag)
                    if (
                        ref_mag > ref_eps
                        and np.isfinite(float(leg_tip["tipping_delta"]))
                    )
                    else float("nan")
                ),
                "nie_median_at_zero": leg_tip["effect_median_at_zero"],
                "robust_over_full_sweep": leg_tip["robust_over_full_sweep"],
                "joint_already_null_at_zero": joint_tip["already_null_at_zero"],
                "joint_tipping_delta": joint_tip["tipping_delta"],
                "joint_tipping_frac_of_effective_slope": (
                    float(joint_tip["tipping_delta"] / ref_mag)
                    if (
                        ref_mag > ref_eps
                        and np.isfinite(float(joint_tip["tipping_delta"]))
                    )
                    else float("nan")
                ),
                "joint_nie_median_at_zero": joint_tip["effect_median_at_zero"],
                "joint_robust_over_full_sweep": joint_tip["robust_over_full_sweep"],
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _ols_coefficients(y: np.ndarray, columns: list[np.ndarray]) -> np.ndarray:
    """Least-squares coefficients after dropping non-informative adjustment columns."""
    kept = [np.ones_like(y, dtype=float)]
    for col in columns:
        arr = np.asarray(col, dtype=float)
        if np.isfinite(arr).all() and float(np.std(arr)) > 1e-10:
            kept.append(arr)
    return np.linalg.lstsq(np.column_stack(kept), np.asarray(y, dtype=float), rcond=None)[0]


def calibrate_session_confounding(
    prepared: PreparedData,
    med: TwoMediatorData,
    sensitivity_summary: pd.DataFrame,
    *,
    session_symbol: str = "attend",
    n_bootstrap: int = 2000,
    seed: int = 335,
) -> pd.DataFrame:
    """Benchmark each two-mediator sensitivity delta against observed sessions.

    The calibration is a treated-arm, phase-0 omitted-variable-bias diagnostic on
    the outcome model's working logit scale. For each mediator, it compares the
    mediator->reading coefficient before and after adding standardised session
    attendance to the same adjusted linear projection. The attenuation-aligned
    part of that coefficient change is ``delta_IS``. A child bootstrap supplies a
    deliberately wide uncertainty band at ``n ~= 25``.

    This is descriptive calibration, not adjustment that identifies a natural
    effect. Restricting to the treated arm avoids using the wait-list arm's
    structural zero sessions as if they were low voluntary attendance. The direct
    adjusted ``attend -> mediator`` association is also reported for each leg; for
    expressive vocabulary this empirically represents the DAG's indirect
    ``IS -> TE -> EV`` ancestry without inventing a product from an unfitted
    taught-vocabulary dose model (#335).
    """
    if session_symbol not in prepared.covariates:
        raise KeyError(
            f"Session calibration requires {session_symbol!r} in prepared.covariates"
        )
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")

    treated = np.asarray(prepared.G) == 1
    n_treated = int(treated.sum())
    if n_treated < 8:
        raise ValueError(
            f"Session calibration needs at least 8 treated observations; got {n_treated}"
        )

    mL, mE = med.mediator_symbols
    z_mediators = {
        mL: (
            logit_safe(prepared.post_counts[mL][treated], med.n_trials_L) - med.zL_mean
        )
        / med.zL_sd,
        mE: (
            logit_safe(prepared.post_counts[mE][treated], med.n_trials_E) - med.zE_mean
        )
        / med.zE_sd,
    }
    y = logit_safe(prepared.post_counts["W"][treated], med.n_trials_W)
    raw_sessions = (
        prepared.covariates[session_symbol]
        * prepared.covariate_scalers[session_symbol].sd
        + prepared.covariate_scalers[session_symbol].mean
    )[treated]
    sessions, _ = standardise(raw_sessions)
    outcome_adjusters = [
        med.W1_logit[treated],
        med.A_std[treated],
        *(med.conf1_logit[s][treated] for s in med.confounder_symbols),
    ]
    mediator_baselines = {
        mL: med.L1_logit[treated],
        mE: med.E1_logit[treated],
    }

    def estimates(
        idx: np.ndarray,
    ) -> dict[str, tuple[float, float, float, float, float]]:
        zL = z_mediators[mL][idx]
        zE = z_mediators[mE][idx]
        sess = sessions[idx]
        if min(float(np.std(zL)), float(np.std(zE)), float(np.std(sess))) <= 1e-10:
            raise ValueError("Bootstrap resample has a constant mediator or session dose")
        adjust = [a[idx] for a in outcome_adjusters]
        base = [zL, zE, *adjust]
        no_is = _ols_coefficients(y[idx], base)
        with_is = _ols_coefficients(y[idx], [*base, sess])
        # zL/zE are guaranteed non-constant in the fitted sample and therefore
        # occupy coefficient positions 1 and 2 after the intercept.
        out: dict[str, tuple[float, float, float, float, float]] = {}
        for position, mediator in enumerate((mL, mE), start=1):
            med_adjust = [
                mediator_baselines[mediator][idx],
                med.A_std[treated][idx],
                *(med.conf1_logit[s][treated][idx] for s in med.confounder_symbols),
                sess,
            ]
            med_fit = _ols_coefficients(z_mediators[mediator][idx], med_adjust)
            out[mediator] = (
                float(no_is[position]),
                float(with_is[position]),
                float(no_is[position] - with_is[position]),
                float(med_fit[-1]),
                float(with_is[-1]),
            )
        return out

    full = estimates(np.arange(n_treated))
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in (mL, mE)}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_treated, size=n_treated)
        try:
            result = estimates(idx)
        except (ValueError, np.linalg.LinAlgError):
            continue
        for mediator in (mL, mE):
            if np.isfinite(result[mediator]).all():
                boot[mediator].append(result[mediator])

    sens = sensitivity_summary.set_index("mediator")
    rows = []
    for mediator in (mL, mE):
        if mediator not in sens.index:
            raise KeyError(f"Sensitivity summary has no row for mediator {mediator!r}")
        ref = float(sens.loc[mediator, "effective_slope_mean"])
        shrink_sign = 1.0 if ref >= 0 else -1.0
        slope_without, slope_with, shift_signed, mediator_is, outcome_is = full[mediator]
        delta_is = max(0.0, shrink_sign * shift_signed)
        draws = np.asarray(boot[mediator], dtype=float)
        if draws.shape[0] < max(20, n_bootstrap // 5):
            raise RuntimeError(
                f"Only {draws.shape[0]} valid session-calibration bootstrap draws "
                f"for {mediator}; the design is too unstable"
            )
        delta_draws = np.maximum(0.0, shrink_sign * draws[:, 2])
        delta_lo, delta_hi = np.quantile(delta_draws, [0.055, 0.945])
        med_is_lo, med_is_hi = np.quantile(draws[:, 3], [0.055, 0.945])
        outcome_is_lo, outcome_is_hi = np.quantile(draws[:, 4], [0.055, 0.945])
        already_null = bool(sens.loc[mediator, "already_null_at_zero"])
        robust = bool(sens.loc[mediator, "robust_over_full_sweep"])
        tip = float(sens.loc[mediator, "tipping_delta"])
        reaches = bool(np.isfinite(tip) and delta_hi >= tip)

        if already_null:
            conclusion = (
                f"NIE_{mediator} is already inconclusive at delta=0, so there is "
                "no non-zero indirect effect for IS-strength confounding to explain away."
            )
        elif robust:
            conclusion = (
                f"NIE_{mediator} remains non-zero across the full one-leg sweep; "
                f"the phase-0 IS anchor is delta_IS={delta_is:.2f} "
                f"(89% bootstrap interval {delta_lo:.2f} to {delta_hi:.2f})."
            )
        elif reaches:
            conclusion = (
                f"The phase-0 IS anchor for NIE_{mediator} is delta_IS={delta_is:.2f} "
                f"(89% bootstrap interval {delta_lo:.2f} to {delta_hi:.2f}); its "
                f"interval reaches the delta*={tip:.2f} tipping point, so IS-strength "
                f"confounding could plausibly account for this path, with wide "
                f"uncertainty at n={n_treated}."
            )
        else:
            conclusion = (
                f"The phase-0 IS anchor for NIE_{mediator} is delta_IS={delta_is:.2f} "
                f"(89% bootstrap interval {delta_lo:.2f} to {delta_hi:.2f}); even "
                f"its upper bound is below the delta*={tip:.2f} tipping point, so "
                f"observed IS-strength confounding alone does not reach it, although "
                f"the calibration remains uncertain at n={n_treated}."
            )

        rows.append(
            {
                "mediator": mediator,
                "quantity": f"NIE_{mediator}",
                "construction": "phase-0 treated-arm adjusted slope change",
                "n_treated": n_treated,
                "effective_slope_mean": ref,
                "slope_without_is": slope_without,
                "slope_with_is": slope_with,
                "slope_shift_signed": shift_signed,
                "delta_is": delta_is,
                "delta_is_lo": float(delta_lo),
                "delta_is_hi": float(delta_hi),
                "mediator_is_slope": mediator_is,
                "mediator_is_slope_lo": float(med_is_lo),
                "mediator_is_slope_hi": float(med_is_hi),
                "outcome_is_slope": outcome_is,
                "outcome_is_slope_lo": float(outcome_is_lo),
                "outcome_is_slope_hi": float(outcome_is_hi),
                "tipping_delta": tip,
                "is_point_reaches_tipping": bool(np.isfinite(tip) and delta_is >= tip),
                "is_band_reaches_tipping": reaches,
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)
