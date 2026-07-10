# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-fit reporting helpers shared across the statistical models."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.statistics.evidence import evidence_label, odds_string
from scipy.special import expit

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)


def _hdi_bounds(draws: np.ndarray, prob: float) -> tuple[float, float]:
    """Highest-density interval (HPDI) bounds for 1-D posterior ``draws``.

    A *sensitivity companion* to the equal-tailed interval reported elsewhere in
    this module: for a skewed or bounded-scale posterior the HPDI is the narrowest
    interval covering ``prob`` mass, so it can differ materially from the
    equal-tailed quantiles. The HPDI is **not** transformation-invariant, so
    callers report it per scale (logit / probability / items) rather than mapping
    it between scales. Uses :func:`arviz.hdi` (``prob=`` keyword in the installed
    ArviZ 1.x).
    """
    lo, hi = np.asarray(az.hdi(np.asarray(draws, dtype=float), prob=prob)).ravel()[:2]
    return float(lo), float(hi)


def _eti_bands(
    draws: np.ndarray, *, probs: tuple[float, ...] = (0.5, 0.9, 0.95)
) -> dict[str, float]:
    """Equal-tailed interval bounds at each coverage in ``probs``, keyed
    ``lo{pct}`` / ``hi{pct}`` (e.g. ``lo50`` / ``hi50``).

    The suite's fixed posterior-band convention (#177): the central **50%**
    interval (where the middle half of the posterior mass sits — a visual aid, not
    a decision threshold), the equal-tailed **90%** sensitivity / compatibility
    band, and the equal-tailed **95%** headline interval. Equal-tailed throughout
    (see ``METHODS.md``); the HDI/HPDI is a separate per-scale sensitivity
    companion (:func:`_hdi_bounds`) and is never labelled with an ``eti`` band key.
    """
    draws = np.asarray(draws, dtype=float)
    out: dict[str, float] = {}
    for p in probs:
        lo, hi = np.quantile(draws, [(1 - p) / 2, 1 - (1 - p) / 2])
        pct = int(round(p * 100))
        out[f"lo{pct}"] = float(lo)
        out[f"hi{pct}"] = float(hi)
    return out


def favoured_direction(prob_positive: float) -> dict[str, float | str]:
    """Evidence for the *favoured* direction of a signed effect (#179).

    :func:`evidence_label` needs a probability already oriented to a **named**
    claim. For a sign claim the favoured direction is ``"positive"`` when
    ``P(effect > 0) >= 0.5`` else ``"negative"``; the claim probability is
    ``max(P>0, P<0)`` and the label qualifies the evidence for THAT direction — so
    a clearly negative effect reads as strong evidence of harm / a negative
    association rather than the "inconclusive" that ``evidence_label(P>0)`` returns
    for the *positive* claim. The raw ``P(effect > 0)`` is still reported
    separately (the benefit claim); callers supply the direction words (benefit /
    harm for treatment effects, positive / negative for associations).
    """
    p_pos = float(prob_positive)
    prob = max(p_pos, 1.0 - p_pos)
    return {
        "favoured_direction": "positive" if p_pos >= 0.5 else "negative",
        "favoured_direction_prob": prob,
        "favoured_direction_label": evidence_label(prob),
    }


def _itt_ame_draws(
    trace: xr.DataTree,
    *,
    G: np.ndarray,
    term: str = "tau",
    varying_term: str = "tau_i",
    eta_name: str = "eta",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    group: str = "posterior",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-draw treatment effect and its probability-scale average marginal effect.

    Shared counterfactual-AME core for the ITT report helpers. For every posterior
    draw and observation ``i`` it forms the untreated baseline linear predictor
    ``η0_i = η_i − δ_i·G_i`` (the treatment contribution removed from the model's
    stored ``eta``) and averages ``expit(η0_i + δ_i) − expit(η0_i)`` over
    observations. ``δ_i`` is the constant ``term`` (``tau``) broadcast over
    observations, or the per-observation ``varying_term`` (``tau_i``) when the model
    carries an age-varying effect.

    ``moderators`` handles treatment×covariate interactions: a sequence of
    ``(coefficient_name, moderator_vector)`` pairs whose contributions are *added*
    to ``δ_i`` per observation, so ``δ_i = base_i + Σ_k c_k · m_{k,i}``. This makes
    the counterfactual net out (and toggle) the *full* per-row treatment
    contribution — the treatment main effect plus every fitted treatment
    interaction — rather than the main effect alone. The gain family passes its
    ``gamma_int_trt_*`` coefficients with the standardised moderator vectors the
    factory used (via ``BuiltModel.extras``); the ITT Part-B moderator passes
    ``gamma_tau_int``. Interaction terms that do **not** involve treatment
    (e.g. ``age×ability``) are unchanged between the treated and untreated
    counterfactual, so they stay inside ``η`` and correctly cancel — they must
    *not* be listed here. Each moderator vector must align with ``eta``'s
    ``obs_id`` axis.

    Returns ``(term_draws, ame_prob)`` — the logit-scale effect draws ``(S,)`` and
    the probability-scale average marginal effect per draw ``(S,)``. Both
    :func:`tau_summary_itt` and :func:`rope_summary` build on this so the two cannot
    drift; it is the same quantity as ``treatment_marginal_effect`` (#128,
    parameterised by ``term``/``trt``), which should fold onto this helper at merge.

    ``group`` selects the inference group: ``"posterior"`` (default) for the
    estimate, or ``"prior"`` to push the *prior* through the same transform for the
    prior-predictive estimand check (issue #125 Area 1/2). The prior group must
    carry ``term`` and ``eta_name`` — it does, since :func:`run_prior_predictive`
    now samples all free RVs + deterministics.
    """
    posterior = getattr(trace, group)
    term_draws = posterior[term].stack(sample=("chain", "draw")).values  # (S,)
    eta = (
        posterior[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    G = np.asarray(G, dtype=float)
    if G.shape[0] != eta.shape[0]:
        raise ValueError(
            f"G has {G.shape[0]} rows but eta has {eta.shape[0]} observations; "
            "pass built.prepared.G (aligned with the fitted subset)."
        )
    # Per-observation treatment contribution δ_i: age-varying ``varying_term`` if the
    # model has it, otherwise the constant ``term`` broadcast over observations.
    if varying_term and varying_term in posterior:
        delta = (
            posterior[varying_term]
            .stack(sample=("chain", "draw"))
            .transpose("obs_id", "sample")
            .values
        )  # (n_obs, S)
    else:
        delta = term_draws[None, :]  # (1, S)
    # Add each treatment interaction's per-row contribution ``c_k · m_{k,i}``, which
    # promotes ``delta`` to (n_obs, S) on the first addition.
    for coef_name, mod_vec in moderators or ():
        if coef_name not in posterior:
            raise KeyError(
                f"moderator coefficient {coef_name!r} not in the {group} group; "
                "the model must register it for the interaction-aware AME."
            )
        coef_draws = posterior[coef_name].stack(sample=("chain", "draw")).values.ravel()  # (S,)
        mod = np.asarray(mod_vec, dtype=float)
        if mod.shape[0] != eta.shape[0]:
            raise ValueError(
                f"moderator {coef_name!r} has {mod.shape[0]} rows but eta has "
                f"{eta.shape[0]} observations; pass the fitted-subset vector."
            )
        delta = delta + np.outer(mod, coef_draws)  # (n_obs, S)
    eta0 = eta - delta * G[:, None]  # untreated baseline (G=0 = control) per obs/draw
    ame_prob = (expit(eta0 + delta) - expit(eta0)).mean(axis=0)  # (S,)
    return term_draws, ame_prob


def tau_summary_itt(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    G: np.ndarray,
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
) -> dict[str, float]:
    """Summarise the treatment effect ``tau`` on both scales for an ITT model.

    The central estimate on each scale is the posterior **median** (``*_median``) —
    the house convention shared with :func:`rope_summary`, so the treatment-effect
    card and the ROPE card lead with the same statistic (see
    ``notes/202606261304-evidence-strength-and-rope-reporting.md``). The median is
    also the more honest lead here: at this sample size the point estimate is
    magnitude-inflated (a Type-M / winner's-curse effect), and the median discounts
    the right tail the mean chases.

    Logit scale: the posterior summary of ``tau`` directly.

    Probability scale: the **average marginal effect** of randomised
    assignment over the fitted sample. For every posterior draw and every
    observation ``i`` we form the counterfactual baseline linear predictor
    ``η0_i = η_i − δ_i · G_i`` from the model's stored per-observation ``eta``
    (the treatment contribution removed; ``δ_i`` is ``tau`` for a constant
    effect, or ``tau_i`` when the effect varies with age), then average
    ``expit(η0_i + δ_i) − expit(η0_i)`` over observations. Each observation's
    effect is therefore evaluated at its *actual* covariate profile —
    including the cross-baseline, adjuster and GP terms carried in ``eta`` —
    rather than at a single constructed baseline point, and the average is
    taken per draw so the posterior uncertainty of the marginal effect is
    preserved.

    ``G`` is the per-observation treatment indicator from the *fitted* prepared
    data (``built.prepared.G``), aligned with ``eta``'s ``obs_id`` axis.

    ``ci_prob`` names the *coverage* probability of the headline interval. The
    ``*_lo`` / ``*_hi`` values are the equal-tailed headline credible interval
    (95% by default); ``*_lo50`` / ``*_hi50`` (the central 50% interval, a visual
    aid) and ``*_lo90`` / ``*_hi90`` (the equal-tailed 90% sensitivity band) follow
    the fixed band convention (#177, see :func:`_eti_bands`). The ``*_hpdi_lo`` /
    ``*_hpdi_hi`` values are the highest-density interval (HPDI) at ``ci_prob`` — a
    per-scale sensitivity companion (see :func:`_hdi_bounds`), not a replacement,
    since the HPDI is not transformation-invariant across the logit and
    probability scales.
    """
    tau_draws, marginal = _itt_ame_draws(trace, G=G, moderators=moderators)

    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    tau_median = float(np.median(tau_draws))
    lower, upper = np.quantile(tau_draws, [lo_q, hi_q])
    tau_hpdi_lo, tau_hpdi_hi = _hdi_bounds(tau_draws, ci_prob)
    tau_b = _eti_bands(tau_draws, probs=(0.5, 0.9))
    marg_median = float(np.median(marginal))
    marg_lo, marg_hi = np.quantile(marginal, [lo_q, hi_q])
    marg_hpdi_lo, marg_hpdi_hi = _hdi_bounds(marginal, ci_prob)
    marg_b = _eti_bands(marginal, probs=(0.5, 0.9))
    prob_pos = float(np.mean(tau_draws > 0))
    # Posterior mean retained as a *secondary* field on each scale (issue #144):
    # the median leads (transformation-invariant, and it discounts the
    # winner's-curse right tail), but the mean is kept available for reference.
    tau_mean = float(np.mean(tau_draws))
    marg_mean = float(np.mean(marginal))

    # Fixed band convention (#177): central 50% (visual aid) + equal-tailed 90%
    # (sensitivity) alongside the equal-tailed 95% headline (``*_lo`` / ``*_hi``).
    return {
        "tau_logit_median": tau_median,
        "tau_logit_mean": tau_mean,
        "tau_logit_lo50": tau_b["lo50"],
        "tau_logit_hi50": tau_b["hi50"],
        "tau_logit_lo90": tau_b["lo90"],
        "tau_logit_hi90": tau_b["hi90"],
        "tau_logit_lo": float(lower),
        "tau_logit_hi": float(upper),
        "tau_logit_hpdi_lo": tau_hpdi_lo,
        "tau_logit_hpdi_hi": tau_hpdi_hi,
        "tau_prob_median": marg_median,
        "tau_prob_mean": marg_mean,
        "tau_prob_lo50": marg_b["lo50"],
        "tau_prob_hi50": marg_b["hi50"],
        "tau_prob_lo90": marg_b["lo90"],
        "tau_prob_hi90": marg_b["hi90"],
        "tau_prob_lo": float(marg_lo),
        "tau_prob_hi": float(marg_hi),
        "tau_prob_hpdi_lo": marg_hpdi_lo,
        "tau_prob_hpdi_hi": marg_hpdi_hi,
        "prob_tau_pos": prob_pos,
        "direction_label": evidence_label(prob_pos),
        **favoured_direction(prob_pos),
    }


def tau_summary_offfloor(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    G: np.ndarray,
) -> dict[str, float]:
    """Summarise the binary off-floor treatment effect (floor-rule PRIMARY, #119).

    For the ``bernoulli_offfloor`` model, ``expit(eta)`` is ``Pr(post > 0 at t2)``
    (the probability of coming off the floor), so the marginal-effect machinery of
    :func:`tau_summary_itt` returns exactly the off-floor quantities: the logit
    scale is the log-odds of coming off the floor, and the probability scale is
    the average **risk difference** in off-floor probability between the
    intervention and control arms. The keys match :func:`tau_summary_itt` (so the
    report and CSV share a schema); the off-floor interpretation is documented in
    the floored-outcome report.
    """
    return tau_summary_itt(trace, ci_prob=ci_prob, G=G)


# --- ROPE-anchored continuous report -----------------------------------------
# notes/202606261304-evidence-strength-and-rope-reporting.md
# evidence_label / odds_string now live in dse_research_utils.statistics.evidence
# (imported above) so the evidence ladder is shared across DSE reports.


def rope_markdown(rope: pd.DataFrame, outcome_label: str, *, with_title: bool = True) -> str:
    """Render the ROPE report card as report markdown (issue #125 Area 4).

    Shared by the ITT and factor result partials so the direction-vs-magnitude
    prose cannot drift between archetypes. Reads the single-row ``rope_summary``
    frame and handles both the items scale and the floored-outcome risk-difference
    scale (``delta_scale == "risk_difference"``, reported in percentage points and
    flagged provisional when ``provisional_delta`` is set).
    """
    r = rope.iloc[0]
    cols = set(rope.columns)
    is_rd = "delta_scale" in cols and str(r.get("delta_scale")) == "risk_difference"
    unit = "percentage points (risk difference)" if is_rd else "items"
    scale = 100.0 if is_rd else 1.0
    prov = ""
    if "provisional_delta" in cols and bool(r.get("provisional_delta")):
        prov = " *(provisional δ, pending education-lead sign-off)*"
    parts: list[str] = []
    if with_title:
        parts.append("## Reporting: direction, magnitude, and practical significance\n")
    parts.append(
        "Following `notes/202606261304-evidence-strength-and-rope-reporting.md` and the "
        '`METHODS.md` "Interpret" rule: report the **median** effect with intervals, and '
        "separate **direction** (is there a benefit?) from **magnitude** (is it big enough "
        f"to matter?), judged against a minimally-important difference δ on the {unit} scale.\n"
    )
    # Direction claim, harm-aware (#179): lead with P(helps) + odds, then state the
    # favoured-direction evidence so a negative effect reads as evidence of harm,
    # not the "inconclusive" that a benefit-only label would give. Guarded so an
    # older rope_summary.csv without the favoured fields still renders.
    _fav = str(r.get("favoured_direction", "positive"))
    _fav_prob = float(r.get("favoured_direction_prob", r["pd"]))
    _fav_label = str(r.get("favoured_direction_label", r["direction_label"]))
    if is_rd:
        _fav_claim = (
            "the intervention raises the off-floor probability"
            if _fav == "positive"
            else "the intervention lowers the off-floor probability"
        )
    else:
        _fav_claim = (
            "the intervention helps"
            if _fav == "positive"
            else "the intervention is harmful"
        )
    direction_clause = (
        f"**Direction** — P(intervention helps) = {r['pd']:.3f} "
        f"({odds_string(r['pd'])}); favoured direction: {_fav_claim} — "
        f"*{_fav_label} evidence* (P = {_fav_prob:.3f})."
    )
    parts.append(
        f"The intervention changed {outcome_label} by a median of "
        f"**{r['items_median'] * scale:+.1f} {unit}**{prov} "
        f"(central 50% interval {r['items_lo50'] * scale:+.1f} to "
        f"{r['items_hi50'] * scale:+.1f}; "
        f"equal-tailed 95% credible interval {r['items_lo'] * scale:+.1f} to "
        f"{r['items_hi'] * scale:+.1f}). "
        f"{direction_clause} "
        f"**Magnitude** — evidence the benefit is at least δ = {r['delta_items'] * scale:g} "
        f"{unit}: P = {r['prob_benefit_ge_delta']:.3f} "
        f"({odds_string(r['prob_benefit_ge_delta'])}, *{r['benefit_label']} evidence*); "
        f"probability inside the ROPE (practically negligible): {r['prob_in_rope']:.3f}.\n"
    )
    if "items_hpdi_lo" in cols:
        parts.append(
            f"_Sensitivity — the 95% highest posterior density interval (HPDI) on the "
            f"{unit} scale is {r['items_hpdi_lo'] * scale:+.1f} to "
            f"{r['items_hpdi_hi'] * scale:+.1f}. HPDI is not transformation-invariant, "
            f"so it is a scale-specific check, not a replacement for the equal-tailed "
            f"interval above._\n"
        )
    return "\n".join(parts)


def _rope_card(
    effect_draws: np.ndarray,
    items: np.ndarray,
    *,
    delta: float,
    ci_prob: float,
) -> dict[str, float | str]:
    """Assemble the ROPE report card from logit-effect and items-effect draws.

    The formatting core shared by every ROPE report so they emit an identical
    ``rope_summary.csv`` schema: :func:`rope_summary` (ITT ``tau`` and the gain
    family's ``beta_trt``) and :func:`level_t2_marginal_effect`'s consumer (the
    level family's t2 randomised contrast). ``effect_draws`` are the logit-scale
    effect draws ``(S,)`` (used for the ``pd`` direction probability), ``items`` the
    matching items-scale average marginal effect per draw ``(S,)``, ``delta`` the
    items-scale ROPE half-width. The point estimate is the **median** because it is
    transformation-invariant across the logit and items scales. The ``tau_logit_*``
    keys are retained verbatim across families (as :func:`tau_summary_offfloor`
    already reuses the ``tau`` schema) so one CSV layout serves the whole suite.

    ``*_hpdi_lo`` / ``*_hpdi_hi`` add the highest-density interval at ``ci_prob``
    for the logit effect and the items scale — a per-scale sensitivity companion
    to the equal-tailed ``*_lo`` / ``*_hi`` fields, kept alongside them (the HPDI
    is not transformation-invariant, so it is reported per scale).
    """
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    pd_ = float(np.mean(effect_draws > 0))
    p_benefit = float(np.mean(items >= delta))
    tau_hpdi_lo, tau_hpdi_hi = _hdi_bounds(effect_draws, ci_prob)
    items_hpdi_lo, items_hpdi_hi = _hdi_bounds(items, ci_prob)
    tau_b90 = _eti_bands(effect_draws, probs=(0.9,))
    items_b90 = _eti_bands(items, probs=(0.9,))
    return {
        "tau_logit_median": float(np.median(effect_draws)),
        "tau_logit_lo50": float(np.quantile(effect_draws, 0.25)),
        "tau_logit_hi50": float(np.quantile(effect_draws, 0.75)),
        "tau_logit_lo90": tau_b90["lo90"],
        "tau_logit_hi90": tau_b90["hi90"],
        "tau_logit_lo": float(np.quantile(effect_draws, lo_q)),
        "tau_logit_hi": float(np.quantile(effect_draws, hi_q)),
        "tau_logit_hpdi_lo": tau_hpdi_lo,
        "tau_logit_hpdi_hi": tau_hpdi_hi,
        "items_median": float(np.median(items)),
        "items_lo50": float(np.quantile(items, 0.25)),
        "items_hi50": float(np.quantile(items, 0.75)),
        "items_lo90": items_b90["lo90"],
        "items_hi90": items_b90["hi90"],
        "items_lo": float(np.quantile(items, lo_q)),
        "items_hi": float(np.quantile(items, hi_q)),
        "items_hpdi_lo": items_hpdi_lo,
        "items_hpdi_hi": items_hpdi_hi,
        "delta_items": float(delta),
        "pd": pd_,
        "prob_benefit_ge_delta": p_benefit,
        "prob_in_rope": float(np.mean(np.abs(items) <= delta)),
        "prob_harm_ge_delta": float(np.mean(items <= -delta)),
        "direction_label": evidence_label(pd_),
        "benefit_label": evidence_label(p_benefit),
        **favoured_direction(pd_),
    }


def rope_summary(
    trace: xr.DataTree,
    *,
    G: np.ndarray,
    n_trials: int,
    delta: float,
    ci_prob: float = 0.95,
    term: str = "tau",
    varying_term: str = "tau_i",
    eta_name: str = "eta",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
) -> dict[str, float | str]:
    """ROPE-anchored continuous report card for a randomised treatment effect.

    Built on :func:`_itt_ame_draws`, so it shares the average-marginal-effect core
    with :func:`tau_summary_itt`. Reports the effect on the logit scale (``term``)
    and the items scale (the average marginal effect × ``n_trials``) as a **median**
    with a 50 % and a ``ci_prob`` (default 95 %) equal-tailed interval, plus:

    - ``pd`` — ``P(effect > 0)``, the probability of direction;
    - ``prob_benefit_ge_delta`` — ``P(items effect > δ)``, the probability of a
      *meaningful* benefit, where ``delta`` is the minimally-important difference
      (the ROPE half-width) on the items scale;
    - ``prob_in_rope`` — ``P(|items effect| < δ)``, practically negligible;
    - ``prob_harm_ge_delta`` — ``P(items effect < −δ)``;
    - ``direction_label`` / ``benefit_label`` — the round-odds evidence labels
      (:func:`evidence_label`) for the direction and meaningful-benefit claims.

    ``term`` / ``varying_term`` / ``G`` select the randomised effect: the ITT suite
    uses the defaults (``tau`` with the age-varying ``tau_i``, ``G`` the arm
    indicator); the gain-factor family passes ``term="beta_trt"``, ``varying_term=""``
    and ``G`` the on-intervention indicator. See
    ``notes/202606261304-evidence-strength-and-rope-reporting.md`` for the rationale
    (sign-vs-size, the median convention, the δ choice).
    """
    effect_draws, ame_prob = _itt_ame_draws(
        trace, G=G, term=term, varying_term=varying_term, eta_name=eta_name,
        moderators=moderators,
    )
    items = ame_prob * float(n_trials)
    return _rope_card(effect_draws, items, delta=delta, ci_prob=ci_prob)


def rope_sensitivity(
    trace: xr.DataTree,
    *,
    G: np.ndarray,
    n_trials: int,
    deltas: Sequence[float],
    term: str = "tau",
    varying_term: str = "tau_i",
    eta_name: str = "eta",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """How the meaningful-benefit claim moves as the threshold δ varies (issue #144).

    A δ-sensitivity view of :func:`rope_summary`: the ``P(benefit ≥ δ)`` headline is
    only as robust as the δ choice, so this sweeps a grid of δ and returns one row per
    δ — ``prob_benefit_ge_delta``, ``prob_in_rope``, ``prob_harm_ge_delta`` and the
    round-odds ``benefit_label``. The education lead's decision (2026-07-01, issue
    #144) is to show this for **all** outcomes, with word reading at δ = 1 and δ = 2;
    the floored outcomes sweep the risk-difference scale (10/15/20 pp).

    Built on the single :func:`_itt_ame_draws` pass (``items = AME × n_trials``), so
    the whole table is one forward computation and cannot drift from the headline
    :func:`rope_summary` card. ``term`` / ``varying_term`` / ``eta_name`` select the
    randomised effect exactly as :func:`rope_summary` does (the floored path passes
    ``n_trials=1`` and ``varying_term=""`` so ``items`` is the risk difference).
    """
    _effect_draws, ame_prob = _itt_ame_draws(
        trace, G=G, term=term, varying_term=varying_term, eta_name=eta_name,
        moderators=moderators,
    )
    items = ame_prob * float(n_trials)
    rows: list[dict[str, float | str]] = []
    for d in deltas:
        d = float(d)
        p_benefit = float(np.mean(items >= d))
        rows.append(
            {
                "delta_items": d,
                "prob_benefit_ge_delta": p_benefit,
                "prob_in_rope": float(np.mean(np.abs(items) <= d)),
                "prob_harm_ge_delta": float(np.mean(items <= -d)),
                "benefit_label": evidence_label(p_benefit),
            }
        )
    return pd.DataFrame(rows)


def rope_sensitivity_markdown(
    sens: pd.DataFrame, *, is_risk_difference: bool = False
) -> str:
    """Render the δ-sensitivity sweep (:func:`rope_sensitivity`) as a markdown table.

    Shared by the ITT and floored result partials so the δ-robustness view cannot
    drift between archetypes. ``is_risk_difference`` reports δ and the effect on the
    percentage-point (risk-difference) scale for the floored outcomes; otherwise the
    items scale.
    """
    scale = 100.0 if is_risk_difference else 1.0
    unit = "pp" if is_risk_difference else "items"
    # Render by ascending δ so the row order (and the prose below) can't drift from
    # the caller's ``deltas`` order or a future grid refactor: the adopted δ is the
    # smallest in the sweep, stricter δ follow.
    sens = sens.sort_values("delta_items")
    lines = [
        "**δ-sensitivity** — how the meaningful-benefit claim moves as the "
        f"minimally-important difference δ rises (δ on the {unit} scale). Rows are in "
        "ascending δ: the top row is the adopted (smallest) δ, stricter δ below it:\n",
        f"| δ ({unit}) | P(benefit ≥ δ) | P(inside ROPE) | P(harm ≥ δ) | evidence |",
        "| ---: | ---: | ---: | ---: | :--- |",
    ]
    for _, r in sens.iterrows():
        lines.append(
            f"| {r['delta_items'] * scale:g} | {r['prob_benefit_ge_delta']:.3f} | "
            f"{r['prob_in_rope']:.3f} | {r['prob_harm_ge_delta']:.3f} | "
            f"{r['benefit_label']} |"
        )
    return "\n".join(lines) + "\n"


def prior_pushforward(
    trace: xr.DataTree,
    *,
    G: np.ndarray,
    n_trials: int,
    term: str = "tau",
    varying_term: str = "tau_i",
    eta_name: str = "eta",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    ci_prob: float = 0.95,
) -> dict[str, float]:
    """Push the **prior** on the effect through the items-scale AME (issue #125 Area 1/2).

    The estimand-scale prior-predictive check: before seeing data, what does the
    prior on the treatment term (``Normal(0, 0.5)`` on the logit) imply for the
    items-scale average marginal effect? A well-calibrated prior should be wide but
    not absurd (it should not put substantial mass on, say, +40 words). Reuses the
    shared :func:`_itt_ame_draws` core on the persisted ``prior`` group, so the
    prior is pushed through the *same* transform as the posterior estimate.
    Requires the prior group to carry ``term`` and ``eta_name`` (it does — see
    :func:`diagnostics.run_prior_predictive`).
    """
    effect_draws, ame_prob = _itt_ame_draws(
        trace,
        G=G,
        term=term,
        varying_term=varying_term,
        eta_name=eta_name,
        moderators=moderators,
        group="prior",
    )
    items = ame_prob * float(n_trials)
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    return {
        "prior_logit_median": float(np.median(effect_draws)),
        "prior_logit_lo": float(np.quantile(effect_draws, lo_q)),
        "prior_logit_hi": float(np.quantile(effect_draws, hi_q)),
        "prior_items_median": float(np.median(items)),
        "prior_items_lo50": float(np.quantile(items, 0.25)),
        "prior_items_hi50": float(np.quantile(items, 0.75)),
        "prior_items_lo": float(np.quantile(items, lo_q)),
        "prior_items_hi": float(np.quantile(items, hi_q)),
        "n_trials": int(n_trials),
    }


def offfloor_mover_table(prepared, symbol: str) -> pd.DataFrame:
    """Per-arm off-floor "mover" counts for a floored outcome (floor-rule, #119).

    Returns, for each randomised arm, the number of children with a non-missing
    post-score, how many came **off the floor** (``post > 0`` at t2), how many
    stayed at the floor, and the off-floor proportion. ``prepared.G`` uses the
    positive-benefit coding (1 = intervention, 0 = wait-list control).
    """
    post = np.asarray(prepared.post_counts[symbol], dtype=float)
    G = np.asarray(prepared.G, dtype=int)
    rows = []
    for g, label in ((1, "intervention"), (0, "control")):
        mask = (G == g) & np.isfinite(post)
        n = int(mask.sum())
        off = int(np.sum(post[mask] > 0))
        rows.append(
            {
                "arm": label,
                "n": n,
                "off_floor": off,
                "at_floor": n - off,
                "prop_off_floor": (off / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def tau_moderation_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
) -> dict[str, float]:
    """Summarise the ITT tau-moderator coefficients ``gamma_tau_int`` / ``gamma_tau_mod``.

    Part B (HTE) analogue of :func:`gamma_interaction_summary`, but for the
    treatment-moderator path of :func:`factories.build_itt_model`: ``gamma_tau_int``
    is the effect modification (how the treatment effect ``tau`` changes per 1 SD
    of the pre-randomisation moderator), and ``gamma_tau_mod`` is the moderator's
    main effect. Equal-tailed central interval at coverage ``ci_prob`` and
    ``P(coef > 0)`` for each coefficient present in the trace.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    out: dict[str, float] = {}
    for name in ("gamma_tau_int", "gamma_tau_mod"):
        if name not in posterior:
            continue
        d = posterior[name].stack(sample=("chain", "draw")).values
        out[f"{name}_mean"] = float(np.mean(d))
        out[f"{name}_lo"] = float(np.quantile(d, lo_q))
        out[f"{name}_hi"] = float(np.quantile(d, hi_q))
        out[f"prob_{name}_pos"] = float(np.mean(d > 0))
    return out


def proportion_at_zero_ppc(
    prepared,
    symbol: str,
    trace: xr.DataTree,
    *,
    node: str = "y_post",
) -> dict[str, float]:
    """Posterior-predictive check on the proportion-at-zero (floor-rule diagnostic).

    Compares the observed fraction of zero post-scores to the posterior-predictive
    distribution of that fraction under the graded Beta-Binomial model — the check
    that reveals whether the graded model reproduces the floor (it typically does
    not for ``P``/``N``, which is the motivation for the binary primary estimand).
    Returns the observed proportion, the predictive mean, and the
    posterior-predictive p-value ``P(rep >= observed)``; the per-draw replicated
    proportions are returned under ``"rep"`` for plotting.
    """
    post = np.asarray(prepared.post_counts[symbol], dtype=float)
    finite = post[np.isfinite(post)]
    obs_p0 = float(np.mean(finite == 0.0)) if finite.size else float("nan")
    pp = trace.posterior_predictive[node]
    yrep = (
        pp.stack(sample=("chain", "draw")).transpose("sample", "obs_id").values
    )  # (S, n_obs)
    rep_p0 = np.mean(yrep == 0.0, axis=1)  # (S,)
    return {
        "obs_prop_at_zero": obs_p0,
        "ppc_mean_prop_at_zero": float(np.mean(rep_p0)),
        "ppc_p_value": float(np.mean(rep_p0 >= obs_p0)),
        "rep": rep_p0,
    }


def did_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    n_trials: int,
    dose: bool = False,
    off_floor: bool = False,
) -> dict[str, float]:
    """Summarise the waitlist-crossover / difference-in-differences effect (kind="did").

    For the binary model ``delta`` is the treatment effect on the logit scale, and
    ``delta_items_*`` is the average marginal effect of toggling ``Treated`` 0 -> 1
    across the fitted rows (per draw), times ``n_trials`` — directly comparable to
    the ITT ``tau_summary_itt`` items figures. ``beta_period`` is the period
    (time / maturation) anchor estimated from the immediate arm. Equal-tailed
    central intervals at coverage ``ci_prob``. With ``dose=True`` the key
    coefficient is ``beta_dose`` (effect per 1 SD of intervention sessions) and no
    items translation is produced.

    With ``off_floor=True`` (the floor-rule DiD for heavily-floored P / N, fitted
    as a Bernoulli on the off-floor indicator) the caller passes ``n_trials=1`` so
    ``delta_items_*`` is the off-floor RISK DIFFERENCE — the change in the
    probability of coming off the floor, not an item count. The returned
    ``off_floor`` flag lets the report partial label the scale accordingly.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    def _summ(name: str) -> dict[str, float]:
        d = posterior[name].stack(sample=("chain", "draw")).values
        prob_pos = float(np.mean(d > 0))
        return {
            f"{name}_mean": float(np.mean(d)),
            f"{name}_lo": float(np.quantile(d, lo_q)),
            f"{name}_hi": float(np.quantile(d, hi_q)),
            f"prob_{name}_pos": prob_pos,
            f"{name}_direction_label": evidence_label(prob_pos),
            f"{name}_favoured_direction": "positive" if prob_pos >= 0.5 else "negative",
            f"{name}_favoured_label": evidence_label(max(prob_pos, 1.0 - prob_pos)),
        }

    out: dict[str, float] = {}
    out.update(_summ("beta_period"))
    if dose:
        out.update(_summ("beta_dose"))
        return out

    out.update(_summ("delta"))
    # Items-scale average marginal effect: toggle Treated 0 -> 1 per fitted row.
    delta = posterior["delta"].stack(sample=("chain", "draw")).values  # (S,)
    eta_base = (
        posterior["eta_base"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    eff = (expit(eta_base + delta[None, :]) - expit(eta_base)).mean(axis=0) * n_trials
    out["delta_items_mean"] = float(np.mean(eff))
    out["delta_items_lo"] = float(np.quantile(eff, lo_q))
    out["delta_items_hi"] = float(np.quantile(eff, hi_q))
    out["off_floor"] = bool(off_floor)
    return out


def tau_summary_joint(
    trace: xr.DataTree,
    outcomes: list[str],
    ci_prob: float,
) -> pd.DataFrame:
    """Return a DataFrame summarising tau_k for each outcome (logit scale).

    ``tau_median`` is the posterior median (the house convention — see
    :func:`tau_summary_itt`); ``tau_lo`` / ``tau_hi`` are equal-tailed central
    quantiles at coverage ``ci_prob``.
    """
    draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values  # (K, n_sample)
    out = []
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    for k, s in enumerate(outcomes):
        d = draws[k]
        out.append(
            {
                "outcome": s,
                "tau_median": float(np.median(d)),
                "tau_lo": float(np.quantile(d, lo_q)),
                "tau_hi": float(np.quantile(d, hi_q)),
                "prob_pos": float(np.mean(d > 0)),
            }
        )
    return pd.DataFrame(out)


def gamma_interaction_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
) -> dict[str, float]:
    """Summarise the linear-moderation coefficients ``gamma_int`` / ``gamma_mod``.

    Reports the posterior mean, equal-tailed central interval at coverage
    ``ci_prob`` (same convention as :func:`tau_summary_itt`), and ``P(coef > 0)``
    for each coefficient present in the trace. ``gamma_int`` is the moderation
    (>0: the standardised mechanism effect strengthens with the moderator);
    ``gamma_mod`` is the moderator main effect at the mean of the mechanism.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    out: dict[str, float] = {}
    for name in ("gamma_int", "gamma_mod"):
        if name not in posterior:
            continue
        d = posterior[name].stack(sample=("chain", "draw")).values
        out[f"{name}_mean"] = float(np.mean(d))
        out[f"{name}_lo"] = float(np.quantile(d, lo_q))
        out[f"{name}_hi"] = float(np.quantile(d, hi_q))
        out[f"prob_{name}_pos"] = float(np.mean(d > 0))
    return out


def tau_contrast_matrix(
    trace: xr.DataTree,
    outcomes: list[str],
) -> pd.DataFrame:
    """Compute P(tau_k > tau_j) for every outcome pair."""
    draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values  # (K, n_sample)
    K = draws.shape[0]
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                M[i, j] = np.nan
            else:
                M[i, j] = float(np.mean(draws[i] > draws[j]))
    return pd.DataFrame(M, index=outcomes, columns=outcomes)


def tau_difference_summary(
    trace: xr.DataTree,
    outcomes: list[str],
    pair: tuple[str, str],
    *,
    ci_prob: float,
) -> dict[str, float | str]:
    """Summarise the difference ``tau[a] - tau[b]`` between two joint outcomes.

    The difference is computed per posterior draw and then summarised, so the
    reported interval propagates the full joint posterior (including any residual
    correlation between the two outcomes) rather than combining two marginal
    summaries. ``pair = (a, b)`` names the contrast ``tau[a] - tau[b]``.

    Sign convention: ``tau`` is the coefficient on ``G = 2 - group``, and group 1
    receives the intervention from t1, so a *positive* ``tau`` means the
    intervention raised that outcome (see the "Sign convention" section of
    METHODS.md). For the LRPITT15/15b generalisation contrast the pair is therefore
    ``("TE", "UE")``: ``tau_TE - tau_UE`` equals the intervention benefit on
    taught words minus the benefit on not-taught words, so a *positive* difference
    means the directly-taught words moved *more* than the not-taught comparison
    words - i.e. limited generalisation.

    ``_lo`` / ``_hi`` are equal-tailed central quantiles at coverage ``ci_prob``
    (same convention as :func:`tau_summary_itt`).
    """
    a, b = pair
    draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values  # (K, n_sample)
    ia, ib = outcomes.index(a), outcomes.index(b)
    diff = draws[ia] - draws[ib]
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    return {
        "contrast": f"{a}_minus_{b}",
        "diff_logit_mean": float(np.mean(diff)),
        "diff_logit_lo": float(np.quantile(diff, lo_q)),
        "diff_logit_hi": float(np.quantile(diff, hi_q)),
        "prob_diff_pos": float(np.mean(diff > 0)),
    }


def write_run_metadata(context: StatisticalFitContext, extra: dict | None = None) -> None:
    """Persist a ``config.json`` and basic metrics for the report."""
    out = context.output_dir
    os.makedirs(out, exist_ok=True)
    spec = context.spec
    cfg = {
        "model_id": spec.model_id,
        # Canonical model-ID scheme (#168 Phase 1); legacy id stays primary.
        "canonical_model_id": spec.canonical_model_id,
        "legacy_model_id": spec.legacy_model_id,
        "family_code": spec.family_code,
        "study_code": spec.study_code,
        "variant_role": spec.variant_role,
        "parent_model_id": spec.parent_model_id,
        "kind": spec.kind,
        "title": spec.title,
        "outcome_symbol": spec.outcome_symbol,
        "mechanism_symbol": spec.mechanism_symbol,
        "adjustment": spec.adjustment,
        # Dataset / estimand metadata (#165) - default to the RLI intervention
        # study for the existing models; historical/cross-study models set them.
        "study_id": spec.study_id,
        "family": spec.family,
        "design": spec.design,
        "estimand_type": spec.estimand_type,
        "causal_status": spec.causal_status,
        "dataset_ref": spec.dataset_ref,
        "audit_baseline": spec.audit_baseline,
        "n_obs": context.prepared.n_obs if context.prepared else None,
        "n_children": context.prepared.n_children if context.prepared else None,
        "n_phases": context.prepared.n_phases if context.prepared else None,
        "dropped_rows": context.prepared.dropped_rows if context.prepared else None,
        "ci_prob": context.reporting.hdi,
        "sampling": {
            "draws": context.sampling.draws,
            "tune": context.sampling.tune,
            "chains": context.sampling.chains,
            "target_accept": context.sampling.target_accept,
            "random_seed": context.sampling.random_seed,
        },
        "output_root": str(_paths.output_root()),
        "extra": extra or {},
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)


def write_loo_summary(context: StatisticalFitContext) -> None:
    if context.loo is None:
        return
    out = context.output_dir
    path = os.path.join(out, "loo.txt")
    with open(path, "w") as f:
        f.write(str(context.loo))


def loo_delta(loo_a: az.ELPDData, loo_b: az.ELPDData) -> dict[str, float]:
    """Delta-ELPD between two models using ArviZ compare.

    arviz 1.x ``az.compare`` reports the ELPD in an ``elpd`` column (the 0.x
    ``elpd_loo`` was renamed); ``dse`` is unchanged.
    """
    df = az.compare({"a": loo_a, "b": loo_b})
    # ``az.compare`` reports ``dse`` relative to the top-ranked (reference) model,
    # whose own ``dse`` is 0; the SE of the ELPD difference sits on the *other*
    # row. Reading ``df.loc["a", "dse"]`` returns 0 whenever "a" ranks first
    # (misleadingly certain). The pairwise difference SE is the single non-zero
    # ``dse`` across the two rows, so take the max (the reference's is exactly 0).
    if "dse" in df.columns:
        d_se = float(max(df.loc["a", "dse"], df.loc["b", "dse"]))
    else:
        d_se = float("nan")
    return {
        "d_elpd": float(df.loc["a", "elpd"] - df.loc["b", "elpd"]),
        "d_se": d_se,
    }


def factor_summary(
    trace: xr.DataTree,
    coef_names: list[str],
    *,
    ci_prob: float,
    causal_terms: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Per-coefficient posterior summary for a factor model (LRPGF / LRPLF, #127).

    One row per coefficient in ``coef_names`` present in the trace: posterior
    ``median`` (the house headline statistic), posterior ``mean`` (secondary),
    equal-tailed central interval at coverage ``ci_prob`` (``lo``/``hi``, same
    convention as :func:`tau_summary_itt`), and ``prob_positive`` =
    ``P(coef > 0)``. The ``role`` column labels each term **causal** (the
    randomised treatment terms named in ``causal_terms``) or **association** —
    under the locked DAG every non-randomised coefficient is an adjusted
    association confounded by latent general ability and must never be read as
    "drives".
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    def _row(term: str, base: str, d: np.ndarray) -> dict[str, object]:
        causal = term in causal_terms or base in causal_terms
        prob_pos = float(np.mean(d > 0))
        return {
            "term": term,
            "role": "causal" if causal else "association",
            "median": float(np.median(d)),
            "mean": float(np.mean(d)),
            "lo": float(np.quantile(d, lo_q)),
            "hi": float(np.quantile(d, hi_q)),
            "prob_positive": prob_pos,
            "direction_label": evidence_label(prob_pos),
            **favoured_direction(prob_pos),
        }

    rows: list[dict[str, object]] = []
    for name in coef_names:
        if name not in posterior:
            continue
        da = posterior[name]
        extra_dims = [dd for dd in da.dims if dd not in ("chain", "draw")]
        if not extra_dims:
            d = da.stack(sample=("chain", "draw")).values.ravel()
            rows.append(_row(name, name, d))
        else:
            # Vector coefficient (e.g. the level model's per-timepoint b_grp_time):
            # one row per element, so an element can be labelled causal on its own
            # (e.g. only the t2 group contrast is the clean randomised effect).
            dim = extra_dims[0]
            for i in range(int(da.sizes[dim])):
                d = da.isel({dim: i}).stack(sample=("chain", "draw")).values.ravel()
                rows.append(_row(f"{name}[{i}]", name, d))
    return pd.DataFrame(rows)


def growth_association_summary(
    trace: xr.DataTree,
    *,
    coefs: tuple[str, ...] = ("gamma", "delta", "beta", "loading"),
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Per-(coefficient, outcome) posterior summary for the growth models (LRP69/70).

    One row per element of each vector coefficient in ``coefs`` (each carries the
    ``outcome`` dim): the posterior **median** (the house lead statistic, robust to
    the Type-M inflation at this n), the fixed 50 / 90 / 95 equal-tailed bands
    (:func:`_eti_bands`, #177), ``prob_positive`` = ``P(coef > 0)`` and the
    evidence-language fields (:func:`favoured_direction`, #179).

    ``gamma`` (baseline non-verbal ability -> growth *rate*) is the headline Q5
    estimand; ``delta`` is the effect on baseline *level*; ``beta`` is the mean
    slope (trajectory characterisation); ``loading`` is the shared growth-tempo
    loading present only in the factor model (LRP70) and skipped otherwise. Every
    row is an **adjusted association** (``role`` fixed to ``"association"``): under
    the locked DAG these non-randomised, latent-GA-confounded terms are never read
    as "drives". ``ci_prob`` is retained for signature parity with
    :func:`factor_summary`; the reported bands are the fixed 50/90/95 set.
    """
    posterior = trace.posterior
    rows: list[dict[str, object]] = []
    for coef in coefs:
        if coef not in posterior:
            continue
        da = posterior[coef]
        outcome_dim = "outcome" if "outcome" in da.dims else None
        labels = list(da[outcome_dim].values) if outcome_dim else [coef]
        for lab in labels:
            sub = da.sel({outcome_dim: lab}) if outcome_dim else da
            d = sub.stack(sample=("chain", "draw")).values.ravel()
            prob_pos = float(np.mean(d > 0))
            rows.append(
                {
                    "coefficient": coef,
                    "outcome": str(lab),
                    "role": "association",
                    "median": float(np.median(d)),
                    "prob_positive": prob_pos,
                    "direction_label": evidence_label(prob_pos),
                    **_eti_bands(d),
                    **favoured_direction(prob_pos),
                }
            )
    return pd.DataFrame(rows)


def treatment_marginal_effect(
    trace: xr.DataTree,
    *,
    trt: np.ndarray,
    n_trials: int,
    term: str = "beta_trt",
    eta_name: str = "eta",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    ci_prob: float = 0.95,
) -> dict[str, float]:
    """Items-scale average marginal effect of the treatment term (LRPGF, #127).

    A thin wrapper over the shared counterfactual-AME core :func:`_itt_ame_draws`
    (#130): the gain model's treatment term ``term`` (``beta_trt``) plays the role of
    the ITT ``tau`` and the on-intervention indicator ``trt`` the role of ``G``, with
    no age-varying term. Per draw the core forms the untreated baseline by removing
    the *full* per-row treatment contribution and toggles it back on: with
    ``moderators`` giving the fitted treatment interactions
    ``(gamma_int_trt_k, z_k)``, the effect is ``beta_trt + Σ_k gamma_int_trt_k·z_{k,i}``
    per row, so the reported AME reflects the treatment main effect *and* its
    interactions rather than ``beta_trt`` alone. This folds onto that core so the
    two parameterisations of the same quantity cannot drift.

    Reported on the probability and items scales (``n_trials`` × probability) with an
    equal-tailed ``ci_prob`` interval. Point estimates are the **median** —
    transformation-invariant across the logit and items scales, matching the ROPE
    convention adopted in #130 (notes/202606261304-evidence-strength-and-rope-
    reporting.md). ``prob_trt_pos`` is ``P(term > 0)`` on the logit scale.
    """
    b, ame_prob = _itt_ame_draws(
        trace, G=trt, term=term, varying_term="", eta_name=eta_name,
        moderators=moderators,
    )
    ame_items = float(n_trials) * ame_prob
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    return {
        "trt_prob_median": float(np.median(ame_prob)),
        "trt_prob_lo": float(np.quantile(ame_prob, lo_q)),
        "trt_prob_hi": float(np.quantile(ame_prob, hi_q)),
        "trt_items_median": float(np.median(ame_items)),
        "trt_items_lo": float(np.quantile(ame_items, lo_q)),
        "trt_items_hi": float(np.quantile(ame_items, hi_q)),
        "prob_trt_pos": float(np.mean(b > 0)),
    }


def level_t2_marginal_effect(
    trace: xr.DataTree,
    *,
    phase: np.ndarray,
    G: np.ndarray,
    t2_phase: int = 1,
    contrast_term: str = "b_grp_time",
    interaction_term: str = "gamma_grp_ability",
    ability: np.ndarray | None = None,
    eta_name: str = "eta",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-draw t2 randomised contrast and its items-scale AME (LRPLF, #127).

    The level model enters group as a per-timepoint vector ``b_grp_time[t]`` because
    the trial is a waitlist crossover; **only the t2 element is a clean randomised
    contrast** (later timepoints are post-crossover associations). This isolates that
    one causal effect on the items scale.

    Unlike the gain family's single ``beta_trt * trt`` term, the level model's group
    contribution at t2 is ``(b_grp_time[t2] + gamma_grp_ability * ability) * G`` — it
    also carries the group×ability interaction — so the plain ``eta - term*G`` removal
    of :func:`_itt_ame_draws` does not apply. Restricting to the t2 rows, per draw we
    net out the *full* group contribution to recover the untreated baseline
    ``eta0 = eta - (b_grp_time[t2] + gamma_grp_ability*ability)*G``, add it back at
    ``G=1``, and average ``expit(eta1) - expit(eta0)`` over the t2 rows.

    Returns ``(contrast_draws, ame_prob)`` — the logit-scale ``b_grp_time[t2]`` draws
    ``(S,)`` (the term flagged causal in the report) and the probability-scale average
    marginal effect per draw ``(S,)``, ready for :func:`_rope_card`. ``ability`` is the
    standardised ability covariate aligned with ``eta``'s ``obs_id`` axis (pass
    ``None`` when the model has no group×ability term).
    """
    posterior = trace.posterior
    phase = np.asarray(phase)
    G = np.asarray(G, dtype=float)
    mask = phase == t2_phase
    if not mask.any():
        raise ValueError(f"No rows at t2_phase={t2_phase}; phases present: {np.unique(phase)}")

    eta = (
        posterior[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    if eta.shape[0] != phase.shape[0]:
        raise ValueError(
            f"phase has {phase.shape[0]} rows but eta has {eta.shape[0]} observations; "
            "pass built.prepared.phase (aligned with the fitted subset)."
        )

    bgt = posterior[contrast_term]
    extra = [d for d in bgt.dims if d not in ("chain", "draw")]
    if not extra:
        raise ValueError(f"{contrast_term!r} is not a per-timepoint vector; t2 contrast undefined")
    contrast_draws = (
        bgt.isel({extra[0]: t2_phase}).stack(sample=("chain", "draw")).values
    )  # (S,)

    # δ_i per t2 row and draw: the constant t2 contrast, plus the group×ability slope
    # times each row's ability if the interaction is in the model.
    delta_rows = contrast_draws[None, :]  # (1, S)
    if interaction_term in posterior and ability is not None:
        g_ab = posterior[interaction_term].stack(sample=("chain", "draw")).values.ravel()  # (S,)
        ab_t2 = np.asarray(ability, dtype=float)[mask]  # (m,)
        delta_rows = contrast_draws[None, :] + np.outer(ab_t2, g_ab)  # (m, S)

    eta_t2 = eta[mask]  # (m, S)
    G_t2 = G[mask]  # (m,)
    eta0 = eta_t2 - delta_rows * G_t2[:, None]  # untreated baseline at the t2 profile
    ame_prob = (expit(eta0 + delta_rows) - expit(eta0)).mean(axis=0)  # (S,)
    return contrast_draws, ame_prob


def horseshoe_ranking(trace: xr.DataTree, *, delta: float = 0.1) -> pd.DataFrame:
    """Per-predictor ranking from a horseshoe fit (LRPHS, #116 Phase E).

    One row per predictor: ``p_abs_gt_delta`` = posterior ``P(|beta_k| > delta)``
    (the ranking key), the posterior mean/sd and 94% HDI (``beta_hdi_3`` /
    ``beta_hdi_97``, an actual highest-density interval via :func:`arviz.hdi`, not
    equal-tailed percentiles) of the standardised coefficient, its ``sign``, and
    ``lambda_mean`` (mean local shrinkage — small ⇒ shrunk toward zero). ``delta``
    is on the logit / per-SD scale (the minimally-interesting coefficient). Ranked
    by ``p_abs_gt_delta`` descending — the horseshoe analogue of the GB
    permutation-importance order.
    """
    posterior = trace.posterior
    beta = posterior["beta"]  # (chain, draw, predictor)
    predictors = [str(p) for p in beta.coords["predictor"].values]
    lam = posterior["hs_lambda"] if "hs_lambda" in posterior else None
    rows = []
    for i, name in enumerate(predictors):
        b = beta.isel(predictor=i).stack(sample=("chain", "draw")).values  # (S,)
        mean = float(np.mean(b))
        hdi = np.asarray(az.hdi(b, prob=0.94))  # 94% highest-density interval
        row = {
            "predictor": name,
            "p_abs_gt_delta": float(np.mean(np.abs(b) > delta)),
            "beta_mean": mean,
            "beta_sd": float(np.std(b)),
            "beta_hdi_3": float(hdi[0]),
            "beta_hdi_97": float(hdi[1]),
            "sign": "+" if mean > 0 else ("-" if mean < 0 else "0"),
        }
        if lam is not None:
            row["lambda_mean"] = float(
                lam.isel(predictor=i).stack(sample=("chain", "draw")).values.mean()
            )
        rows.append(row)
    df = (
        pd.DataFrame(rows)
        .sort_values("p_abs_gt_delta", ascending=False)
        .reset_index(drop=True)
    )
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
