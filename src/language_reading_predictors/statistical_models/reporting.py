# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-fit reporting helpers shared across the statistical models."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.statistics.evidence import (
    evidence_label,
    favoured_direction,
    odds_string,
)
from dse_research_utils.statistics.intervals import eti_bands, hdi_1d
from dse_research_utils.statistics.rope import rope_card
from scipy.special import expit, logit

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)


def band90(draws: np.ndarray) -> tuple[float, float]:
    """Equal-tailed 90 % band ``(lo05, hi95)`` reported alongside the 95 % headline.

    A single sensitivity band so the summary builders that report only a headline
    ``ci_prob`` (95 %) interval can also carry the narrower 90 % equal-tailed
    interval without re-deriving quantiles at each call site. The wider ITT / growth
    summaries use the shared ``eti_bands`` helper (50 / 90 / 95); this covers the
    families that emit a single headline interval.
    """
    return float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))


def _itt_ame_draws(
    trace: xr.DataTree,
    *,
    G: np.ndarray,
    term: str = "tau",
    varying_term: str = "tau_i",
    eta_name: str = "eta",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    group: str = "posterior",
    row_mask: np.ndarray | None = None,
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

    ``row_mask`` (default None = all rows): restrict the observation average to a
    subset of ``obs_id`` rows. The gain-factor family passes the **period-1** mask
    (``phase == 0``) so the marginal treatment effect is averaged only over the
    genuinely randomised, all-untreated-baseline transition — not the post-crossover
    transitions that carry no untreated observations and baselines that may already
    be treatment-affected (#247 P2). A boolean or integer-index array aligned with
    ``eta``'s ``obs_id`` axis; ITT/level paths leave it None (unchanged behaviour).
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
    contrib = expit(eta0 + delta) - expit(eta0)  # (n_obs, S)
    if row_mask is not None:
        m = np.asarray(row_mask)
        # Validate dtype + dimensionality so a 2-D or float mask fails loudly rather
        # than silently changing the indexing semantics of ``contrib[m]`` (which would
        # yield a wrong AME). Only a 1-D boolean mask (length n_obs) or a 1-D integer
        # index array (in range) is accepted.
        if m.ndim != 1:
            raise ValueError(f"row_mask must be 1-D, got a {m.ndim}-D array.")
        if m.dtype == bool:
            if m.shape[0] != eta.shape[0]:
                raise ValueError(
                    f"boolean row_mask has {m.shape[0]} entries but eta has "
                    f"{eta.shape[0]} observations; pass the fitted-subset mask."
                )
        elif np.issubdtype(m.dtype, np.integer):
            if m.size and (int(m.min()) < 0 or int(m.max()) >= eta.shape[0]):
                raise ValueError(
                    f"integer row_mask has indices outside [0, {eta.shape[0]})."
                )
        else:
            raise ValueError(
                "row_mask must be a boolean mask or integer index array, got dtype "
                f"{m.dtype}."
            )
        contrib = contrib[m]
        if contrib.shape[0] == 0:
            raise ValueError("row_mask selects no observations for the marginal effect.")
    ame_prob = contrib.mean(axis=0)  # (S,)
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
    the fixed band convention (#177, see :func:`eti_bands`). The ``*_hpdi_lo`` /
    ``*_hpdi_hi`` values are the highest-density interval (HPDI) at ``ci_prob`` — a
    per-scale sensitivity companion (see :func:`hdi_1d`), not a replacement,
    since the HPDI is not transformation-invariant across the logit and
    probability scales.
    """
    tau_draws, marginal = _itt_ame_draws(trace, G=G, moderators=moderators)

    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    tau_median = float(np.median(tau_draws))
    lower, upper = np.quantile(tau_draws, [lo_q, hi_q])
    tau_hpdi_lo, tau_hpdi_hi = hdi_1d(tau_draws, ci_prob)
    tau_b = eti_bands(tau_draws, probs=(0.5, 0.9))
    marg_median = float(np.median(marginal))
    marg_lo, marg_hi = np.quantile(marginal, [lo_q, hi_q])
    marg_hpdi_lo, marg_hpdi_hi = hdi_1d(marginal, ci_prob)
    marg_b = eti_bands(marginal, probs=(0.5, 0.9))
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
    row_mask: np.ndarray | None = None,
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
        moderators=moderators, row_mask=row_mask,
    )
    items = ame_prob * float(n_trials)
    return rope_card(effect_draws, items, delta=delta, ci_prob=ci_prob)


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
    row_mask: np.ndarray | None = None,
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
        moderators=moderators, row_mask=row_mask,
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
    row_mask: np.ndarray | None = None,
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
        row_mask=row_mask,
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
        out[f"{name}_median"] = float(np.median(d))  # median-first (#271)
        out[f"{name}_mean"] = float(np.mean(d))
        out[f"{name}_lo"] = float(np.quantile(d, lo_q))
        out[f"{name}_hi"] = float(np.quantile(d, hi_q))
        out[f"{name}_lo90"], out[f"{name}_hi90"] = band90(d)
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


def _readiness_knee(
    f: np.ndarray,
    ell: np.ndarray,
    *,
    n_trials: int,
    ci_prob: float = 0.95,
    n_bins: int = 6,
) -> dict[str, float]:
    """Locate the readiness knee from a per-observation ``f_mech`` posterior + logit input.

    Pure-numpy core of :func:`readiness_threshold` (split out so the knee logic is
    unit-testable without a trace, #293 review). ``f`` is ``(n_obs, n_draws)`` HSGP
    curve draws; ``ell`` is the ``(n_obs,)`` Haldane-corrected mechanism logit.

    The knee is the *steepest rise* of the binned curve — the predictor level at
    which the outcome rises fastest per additional item — not the onset of the
    rise; ``slope_below/above_knee_median`` support the "flat below, rising above"
    read. ``half_rise_count_*`` is a complementary mid-rise summary (where the
    curve first reaches the midpoint of its binned range). Both are summarised
    over the *increasing* draws only (net end-to-end rise on the binned curve;
    the share is ``increasing_frac``) — on a flat or falling draw the estimands
    are undefined, and a low ``increasing_frac`` flags an ill-defined knee
    (#297 review).
    """
    # Inverse Haldane-corrected logit -> approximate predictor count, clipped to range.
    # ell = log((y+0.5)/(n-y+0.5)) => expit(ell) = (y+0.5)/(n+1), so y = (n+1)*expit(ell) - 0.5
    # (the denominator is n+1, not n; #293 review).
    L = np.clip((n_trials + 1.0) / (1.0 + np.exp(-ell)) - 0.5, 0.0, float(n_trials))

    edges = np.unique(np.quantile(L, np.linspace(0.0, 1.0, n_bins + 1)))
    nb = len(edges) - 1
    if nb < 2:
        raise ValueError("Too few distinct predictor bins to locate a knee.")
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(L, edges[1:-1]), 0, nb - 1)
    binmean = np.full((nb, f.shape[1]), np.nan)
    for b in range(nb):
        m = idx == b
        if m.any():
            binmean[b] = f[m].mean(axis=0)
    slope = np.diff(binmean, axis=0) / np.diff(centers)[:, None]  # (nb-1, S)
    knee_bin = np.nanargmax(slope, axis=0)  # steepest-rise interval per draw
    knee_L = 0.5 * (centers[knee_bin] + centers[knee_bin + 1])  # (S,)

    # Net end-to-end rise per draw; the estimand summaries pool these draws only.
    increasing = binmean[-1] > binmean[0]  # (S,) — NaN endpoints compare False

    # Per-draw half-rise: where the binned curve first reaches the midpoint of its
    # range, linearly interpolated between the straddling bin centres.
    lo_f = np.nanmin(binmean, axis=0)  # (S,)
    hi_f = np.nanmax(binmean, axis=0)
    target = 0.5 * (lo_f + hi_f)
    first = np.argmax(binmean >= target[None, :], axis=0)  # first bin at/above midpoint
    half_L = np.full(f.shape[1], centers[0])  # first==0: starts at/above the midpoint
    interior = first > 0
    if interior.any():
        s = np.flatnonzero(interior)
        j = first[s]
        f_lo, f_hi = binmean[j - 1, s], binmean[j, s]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(f_hi > f_lo, (target[s] - f_lo) / (f_hi - f_lo), 0.0)
        half_L[s] = centers[j - 1] + t * (centers[j] - centers[j - 1])

    lo, hi = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2

    def _q(a: np.ndarray) -> tuple[float, float, float]:
        a = a[np.isfinite(a)]
        if not a.size:
            return (float("nan"),) * 3
        return (
            float(np.median(a)),
            float(np.quantile(a, lo)),
            float(np.quantile(a, hi)),
        )

    kmed, k_lo, k_hi = _q(knee_L[increasing])
    hmed, h_lo, h_hi = _q(half_L[increasing])

    # Classify each between-bin interval by its midpoint relative to the median knee, so
    # the knee interval itself counts as "above" and the "above" set is never empty when
    # the steepest rise is the top interval (a late-accelerating curve).
    if np.isfinite(kmed):
        interval_mid = 0.5 * (centers[:-1] + centers[1:])
        below = interval_mid < kmed
        slope_below = np.nanmean(np.where(below[:, None], slope, np.nan), axis=0)
        slope_above = np.nanmean(np.where(~below[:, None], slope, np.nan), axis=0)
    else:  # no increasing draws — the below/above split is undefined
        slope_below = slope_above = np.full(f.shape[1], np.nan)

    def _med(a: np.ndarray) -> float:
        a = a[np.isfinite(a) & increasing]
        return float(np.median(a)) if a.size else float("nan")

    return {
        "knee_count_median": kmed,
        "knee_count_ci_low": k_lo,
        "knee_count_ci_high": k_hi,
        "half_rise_count_median": hmed,
        "half_rise_count_ci_low": h_lo,
        "half_rise_count_ci_high": h_hi,
        "slope_below_knee_median": _med(slope_below),
        "slope_above_knee_median": _med(slope_above),
        "increasing_frac": float(np.mean(increasing)),
        "obs_count_min": float(L.min()),
        "obs_count_max": float(L.max()),
        "ci_prob": float(ci_prob),
        "n_draws": int(f.shape[1]),
        "n_obs": int(f.shape[0]),
        "n_bins": int(nb),
    }


def readiness_threshold(
    trace: xr.DataTree,
    *,
    n_trials: int,
    ci_prob: float = 0.95,
    n_bins: int = 6,
) -> dict[str, float]:
    """Readiness-threshold estimand: the steepest rise of a mechanism curve (#230 §2/§5).

    Post-processes an HSGP mechanism model's adjusted curve ``f_mech`` to locate its
    steepest rise — the "knee", in the predictor's raw count units. For each posterior
    draw the per-observation ``f_mech`` is binned over the observed predictor range
    (quantile bins) and the steepest between-bin rise is found; the knee is that
    interval's midpoint, giving a posterior over the knee location. Reports its median
    + equal-tailed CI, a complementary half-rise summary, and the mean marginal slope
    below vs above the knee (a "flat below, rising above" read). Note the knee is where
    the outcome rises *fastest* — the middle of the rise, not its onset; for the
    letter-sound -> word-reading mechanism (``lrp-rli-mech-058``) read it as "reading
    rises fastest around ~k letter sounds", and let the below-knee slope say whether it
    is near-flat before that.

    Pure post-processing (no re-fit): needs the ``f_mech`` posterior and the
    ``mech_post_logit`` constant-data node of a standard HSGP mechanism fit (e.g.
    ``lrp-rli-mech-058``). ``n_trials`` is the mechanism predictor's item ceiling (letter
    sounds = 32) used to back-transform the logit input to an approximate count.
    """
    post = trace.posterior
    if "f_mech" not in post:
        raise KeyError(
            "trace has no 'f_mech' posterior — the readiness threshold needs an HSGP "
            "mechanism fit (not the linear-mechanism or phase-specific variant)."
        )
    # The HSGP ``f_mech`` carries an auto-named obs dimension (e.g. ``f_mech_dim_0``),
    # not ``obs_id``; take whichever non-sample dim it has. Its rows are in the model's
    # observation order, aligned to the ``mech_post_logit`` constant-data node below.
    f_stacked = post["f_mech"].stack(sample=("chain", "draw"))
    obs_dim = next(d for d in f_stacked.dims if d != "sample")
    f = f_stacked.transpose(obs_dim, "sample").values  # (n_obs, S)
    ell = np.asarray(trace.constant_data["mech_post_logit"].values).reshape(-1)  # (n_obs,)
    return _readiness_knee(f, ell, n_trials=n_trials, ci_prob=ci_prob, n_bins=n_bins)


def did_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    n_trials: int,
    dose: bool = False,
    off_floor: bool = False,
) -> dict[str, float | bool | str]:
    """Summarise the waitlist-crossover / difference-in-differences effect (kind="did").

    For the binary model ``delta`` is the treatment effect on the logit scale, and
    ``delta_items_*`` is the average marginal effect of toggling ``Treated`` 0 -> 1
    across the fitted rows (per draw), times ``n_trials`` — directly comparable to
    the ITT ``tau_summary_itt`` items figures. ``beta_period`` is the period
    (time / maturation) anchor estimated from the immediate arm. Equal-tailed
    central intervals at coverage ``ci_prob``. With ``dose=True`` the key
    coefficient is ``beta_dose`` (effect per 1 SD of intervention sessions) and no
    items translation is produced.

    With ``off_floor=True`` (the off-floor prevalence DiD for heavily-floored P / N,
    fitted as a Bernoulli on the off-floor indicator) the caller passes
    ``n_trials=1`` so ``delta_items_*`` is a model-implied off-floor RISK DIFFERENCE
    — the difference in the probability of *being* off the floor at period end
    obtained by toggling ``Treated``, not an item count and not a probability-scale
    DiD cross-difference. The returned ``off_floor`` flag lets the report partial
    label the scale accordingly.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    def _summ(name: str) -> dict[str, float | str]:
        d = posterior[name].stack(sample=("chain", "draw")).values
        prob_pos = float(np.mean(d > 0))
        lo90, hi90 = band90(d)
        return {
            # Median-first to match the ITT tau_summary_itt convention (#144 / #271);
            # the mean is kept as a secondary column.
            f"{name}_median": float(np.median(d)),
            f"{name}_mean": float(np.mean(d)),
            f"{name}_lo": float(np.quantile(d, lo_q)),
            f"{name}_hi": float(np.quantile(d, hi_q)),
            f"{name}_lo90": lo90,
            f"{name}_hi90": hi90,
            f"prob_{name}_pos": prob_pos,
            f"{name}_direction_label": evidence_label(prob_pos),
            f"{name}_favoured_direction": "positive" if prob_pos >= 0.5 else "negative",
            f"{name}_favoured_label": evidence_label(max(prob_pos, 1.0 - prob_pos)),
        }

    out: dict[str, float | bool] = {}
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
    out["delta_items_median"] = float(np.median(eff))
    out["delta_items_mean"] = float(np.mean(eff))
    out["delta_items_lo"] = float(np.quantile(eff, lo_q))
    out["delta_items_hi"] = float(np.quantile(eff, hi_q))
    out["delta_items_lo90"], out["delta_items_hi90"] = band90(eff)
    out["off_floor"] = bool(off_floor)
    return out


def block_exposure_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    n_trials: int,
) -> dict[str, float | bool | str]:
    """Summarise the block-2 block-active exposure effect (kind="block_exposure").

    ``delta`` is the exposure effect on the logit scale; ``delta_items_*`` is the
    average marginal effect of toggling ``exposed`` 0 -> 1 across the fitted rows
    (per draw), times ``n_trials`` — the block-2 taught-word count attributable to
    block-2 being actively taught. This is an ASSOCIATION (parallel-trends), not a
    randomised effect (see :func:`factories.build_block_exposure_model`). Equal-tailed
    central intervals at coverage ``ci_prob``. Mirrors the ``delta`` block of
    :func:`did_summary` (the DiD sibling) but carries no ``beta_period`` — the
    per-timepoint ``alpha_time`` vector is the secular-trend anchor here.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    d = posterior["delta"].stack(sample=("chain", "draw")).values  # (S,)
    prob_pos = float(np.mean(d > 0))
    lo90, hi90 = band90(d)
    out: dict[str, float | bool | str] = {
        "delta_median": float(np.median(d)),
        "delta_mean": float(np.mean(d)),
        "delta_lo": float(np.quantile(d, lo_q)),
        "delta_hi": float(np.quantile(d, hi_q)),
        "delta_lo90": lo90,
        "delta_hi90": hi90,
        "prob_delta_pos": prob_pos,
        "delta_direction_label": evidence_label(prob_pos),
        "delta_favoured_direction": "positive" if prob_pos >= 0.5 else "negative",
        "delta_favoured_label": evidence_label(max(prob_pos, 1.0 - prob_pos)),
    }
    # Items-scale average marginal effect: toggle exposed 0 -> 1 per fitted row.
    eta_base = (
        posterior["eta_base"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    eff = (expit(eta_base + d[None, :]) - expit(eta_base)).mean(axis=0) * n_trials
    out["delta_items_median"] = float(np.median(eff))
    out["delta_items_mean"] = float(np.mean(eff))
    out["delta_items_lo"] = float(np.quantile(eff, lo_q))
    out["delta_items_hi"] = float(np.quantile(eff, hi_q))
    out["delta_items_lo90"], out["delta_items_hi90"] = band90(eff)
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
                "tau_lo90": band90(d)[0],
                "tau_hi90": band90(d)[1],
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
        out[f"{name}_median"] = float(np.median(d))  # median-first (#271)
        out[f"{name}_mean"] = float(np.mean(d))
        out[f"{name}_lo"] = float(np.quantile(d, lo_q))
        out[f"{name}_hi"] = float(np.quantile(d, hi_q))
        out[f"{name}_lo90"], out[f"{name}_hi90"] = band90(d)
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
        "diff_logit_median": float(np.median(diff)),  # median-first (#271)
        "diff_logit_mean": float(np.mean(diff)),
        "diff_logit_lo": float(np.quantile(diff, lo_q)),
        "diff_logit_hi": float(np.quantile(diff, hi_q)),
        "diff_logit_lo90": band90(diff)[0],
        "diff_logit_hi90": band90(diff)[1],
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
        "ci_prob": context.reporting.ci_prob,
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
        lo90, hi90 = band90(d)
        return {
            "term": term,
            "role": "causal" if causal else "association",
            "median": float(np.median(d)),
            "mean": float(np.mean(d)),
            "lo": float(np.quantile(d, lo_q)),
            "hi": float(np.quantile(d, hi_q)),
            "lo90": lo90,
            "hi90": hi90,
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
    (:func:`eti_bands`, #177), ``prob_positive`` = ``P(coef > 0)`` and the
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
                    **eti_bands(d),
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
    row_mask: np.ndarray | None = None,
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

    ``row_mask`` (default None = all fitted rows): restrict the observation average to
    a row subset. The gain-factor family passes the **period-1** mask (``phase == 0``)
    so the marginal is averaged only over the genuinely randomised transition, not the
    post-crossover ones that carry no untreated observations (#247 P2). The logit-scale
    ``prob_trt_pos`` is unaffected — it summarises the ``term`` draws directly.
    """
    b, ame_prob = _itt_ame_draws(
        trace, G=trt, term=term, varying_term="", eta_name=eta_name,
        moderators=moderators, row_mask=row_mask,
    )
    ame_items = float(n_trials) * ame_prob
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    prob_lo90, prob_hi90 = band90(ame_prob)
    items_lo90, items_hi90 = band90(ame_items)
    return {
        "trt_prob_median": float(np.median(ame_prob)),
        "trt_prob_lo": float(np.quantile(ame_prob, lo_q)),
        "trt_prob_hi": float(np.quantile(ame_prob, hi_q)),
        "trt_prob_lo90": prob_lo90,
        "trt_prob_hi90": prob_hi90,
        "trt_items_median": float(np.median(ame_items)),
        "trt_items_lo": float(np.quantile(ame_items, lo_q)),
        "trt_items_hi": float(np.quantile(ame_items, hi_q)),
        "trt_items_lo90": items_lo90,
        "trt_items_hi90": items_hi90,
        "prob_trt_pos": float(np.mean(b > 0)),
    }


@dataclass(frozen=True)
class AssociationTerm:
    """One adjusted-association covariate for the gain-factor items-scale marginals (#310).

    Describes how a single covariate enters the gain-factor linear predictor, so
    :func:`association_marginals` can push a ``+1 SD`` (and, for bounded-count
    covariates, a ``+k items``) perturbation of it through the fitted posterior onto
    the probability / items scales — the covariate analogue of the treatment
    marginal. The pipeline (which holds the prepared design) builds these; the
    reporting helper stays agnostic about the gain-factor internals.

    Attributes
    ----------
    label
        Human covariate name for the report row (e.g. ``"own"``, ``"L"``, ``"age"``).
    coef
        Posterior variable name of the covariate's main-effect coefficient
        (e.g. ``"gamma_own"``, ``"gamma_A"``, ``"gamma_ability"``, ``"gamma_L"``).
    main_scale
        Data-scale shift of the covariate's *main-effect* input per ``+1`` standardised
        unit (``b_t``). For age / cognitive ability the main effect already enters on
        the standardised scale, so ``main_scale = 1``. For the own baseline and skill
        baselines the main effect enters on the **raw logit** scale while the fitted
        interactions use the standardised vector, so ``main_scale`` is the SD of that
        raw logit — ``+1 SD`` shifts the raw-logit input by ``main_scale``.
    interactions
        ``(gamma_int_name, z_partner)`` pairs for every fitted interaction this term
        participates in. Because the interaction inputs are plain elementwise products
        of standardised vectors (``z_a · z_b``), a ``+1`` shift in this term's
        standardised value changes the product by exactly the partner's standardised
        vector — so the per-row interaction contribution to ``Δη`` is
        ``gamma_int · z_partner``. Treatment interactions are included: the covariate
        marginal holds the treatment indicator fixed and perturbs the covariate, so a
        ``trt × covariate`` term does move with the covariate.
    n_items
        Denominator of the covariate when it is a bounded-count measure (own / skill
        baselines); enables the ``+k items`` variant. ``None`` for age / ability /
        continuous adjusters.
    mean_prop
        Mean baseline proportion of a bounded-count covariate on the fitted rows — the
        operating point at which the ``+k items`` perturbation is evaluated (the logit
        shift for ``+k items`` is level-dependent, so it is anchored at the mean).
    sd_items
        Informational: how many items ``+1 SD`` of a bounded-count covariate is,
        evaluated at ``mean_prop`` — so a reader can translate the opaque ``+1 SD``.
    """

    label: str
    coef: str
    main_scale: float
    interactions: tuple[tuple[str, np.ndarray], ...] = ()
    n_items: int | None = None
    mean_prop: float | None = None
    sd_items: float | None = None


def association_marginals(
    trace: xr.DataTree,
    *,
    terms: Sequence[AssociationTerm],
    n_trials: int,
    off_floor: bool = False,
    k_items: int = 5,
    eta_name: str = "eta",
    ci_prob: float = 0.95,
    row_mask: np.ndarray | None = None,
    group: str = "posterior",
) -> pd.DataFrame:
    """Per-covariate items-scale association marginals for the gain family (#310).

    The adjusted-association analogue of :func:`treatment_marginal_effect`: for each
    covariate in ``terms`` it forms the per-draw change in the linear predictor from a
    ``+1 SD`` perturbation of that covariate, holding everything else at its observed
    value, and averages the probability-scale change ``expit(η + Δη) − expit(η)`` over
    observations. Reported on the probability and items scales (``n_trials`` ×
    probability), with an equal-tailed ``ci_prob`` interval and a 90 % band.

    Per draw ``s`` and observation ``i`` the perturbation's linear-predictor shift is

        Δη_{i,s} = γ_{c,s} · (main_scale) + Σ_k γ^{int}_{k,s} · z^{partner}_{k,i},

    i.e. the covariate's main-effect coefficient scaled to the ``+1 SD`` data shift,
    plus each fitted interaction's contribution (the interaction inputs are elementwise
    products of standardised vectors, so a ``+1`` standardised shift moves the product
    by the partner's standardised vector). This mirrors the treatment marginal's
    "net out and toggle" idiom (:func:`_itt_ame_draws`), specialised to a covariate
    increment rather than a 0/1 switch.

    For **bounded-count** covariates (``n_items`` set) a second ``+{k_items} items`` row
    is emitted, evaluated at the covariate's mean baseline proportion (``mean_prop``):
    the raw-logit shift ``Δraw = logit(p̄ + k/N) − logit(p̄)`` replaces the ``+1 SD``
    shift, and the interaction contribution scales by ``Δz = Δraw / main_scale`` (the
    same shift in standardised units). ``+1 SD`` is opaque to readers; ``+k items`` is
    the interpretable companion.

    For ``off_floor`` outcomes (``n_trials`` should be passed as ``1``) the items scale
    collapses to the off-floor probability delta, mirroring the treatment marginal's
    floor-rule handling.

    ``row_mask`` (default ``None`` = **all** stacked rows): the covariate associations
    are descriptive, so the natural averaging population is every fitted observation —
    unlike the treatment marginal, which restricts to the randomised period-1 rows. The
    choice is pre-specified in the design note and recorded in ``config.json``.

    Every row carries ``role = "association"`` — none of these terms is causal, per the
    gain family's documented estimand structure.
    """
    posterior = getattr(trace, group)
    eta = (
        posterior[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    n_obs = eta.shape[0]

    mask: np.ndarray | None = None
    if row_mask is not None:
        m = np.asarray(row_mask)
        if m.ndim != 1:
            raise ValueError(f"row_mask must be 1-D, got a {m.ndim}-D array.")
        if m.dtype == bool:
            if m.shape[0] != n_obs:
                raise ValueError(
                    f"boolean row_mask has {m.shape[0]} entries but eta has "
                    f"{n_obs} observations."
                )
        elif np.issubdtype(m.dtype, np.integer):
            if m.size and (int(m.min()) < 0 or int(m.max()) >= n_obs):
                raise ValueError(f"integer row_mask has indices outside [0, {n_obs}).")
        else:
            raise ValueError(
                "row_mask must be a boolean mask or integer index array, got dtype "
                f"{m.dtype}."
            )
        mask = m

    eta_sel = eta if mask is None else eta[mask]
    if eta_sel.shape[0] == 0:
        raise ValueError("row_mask selects no observations for the marginal effect.")

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    rows: list[dict[str, float | str]] = []

    for term in terms:
        coef = posterior[term.coef].stack(sample=("chain", "draw")).values.ravel()  # (S,)

        # (scale label, standardised shift Δz). +1 SD is Δz = 1; +k items maps the
        # bounded-count increment to standardised units at the mean operating point.
        perturbations: list[tuple[str, float]] = [("+1 SD", 1.0)]
        if term.n_items and term.mean_prop is not None and term.main_scale > 0:
            eps = 1e-6
            p = float(np.clip(term.mean_prop, eps, 1 - eps))
            p_k = float(np.clip(p + k_items / term.n_items, eps, 1 - eps))
            dz = (logit(p_k) - logit(p)) / term.main_scale
            perturbations.append((f"+{k_items} items", dz))

        for scale_label, dz in perturbations:
            # Main-effect shift: γ_c scaled to the requested data increment. Broadcast
            # over observations (shape (1, S)); promoted to (n_obs, S) by interactions.
            delta_eta = (coef * (dz * term.main_scale))[None, :]
            for gi_name, z_partner in term.interactions:
                gi = posterior[gi_name].stack(sample=("chain", "draw")).values.ravel()  # (S,)
                zp = np.asarray(z_partner, dtype=float)
                if zp.shape[0] != n_obs:
                    raise ValueError(
                        f"interaction partner for {term.label!r}/{gi_name!r} has "
                        f"{zp.shape[0]} rows but eta has {n_obs} observations."
                    )
                delta_eta = delta_eta + np.outer(zp, gi) * dz  # (n_obs, S)

            de_sel = delta_eta if delta_eta.shape[0] == 1 else (
                delta_eta if mask is None else delta_eta[mask]
            )
            ame_prob = (expit(eta_sel + de_sel) - expit(eta_sel)).mean(axis=0)  # (S,)
            ame_items = float(n_trials) * ame_prob
            prob_lo90, prob_hi90 = band90(ame_prob)
            items_lo90, items_hi90 = band90(ame_items)
            rows.append(
                {
                    "term": term.label,
                    "role": "association",
                    "scale": scale_label,
                    "prob_median": float(np.median(ame_prob)),
                    "prob_lo": float(np.quantile(ame_prob, lo_q)),
                    "prob_hi": float(np.quantile(ame_prob, hi_q)),
                    "prob_lo90": prob_lo90,
                    "prob_hi90": prob_hi90,
                    "items_median": float(np.median(ame_items)),
                    "items_lo": float(np.quantile(ame_items, lo_q)),
                    "items_hi": float(np.quantile(ame_items, hi_q)),
                    "items_lo90": items_lo90,
                    "items_hi90": items_hi90,
                    "prob_pos": float(np.mean(ame_items > 0)),
                    "off_floor": bool(off_floor),
                    "sd_items": (
                        float(term.sd_items) if term.sd_items is not None else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class ConcurrentTerm:
    """One standardised predictor for the concurrent-associations items-scale marginals (#312).

    The concurrent family (``kind="concurrent"``) fits, per wave, a between-child
    Beta-Binomial regression of the focal outcome's *level* on the standardised
    same-wave logits of a set of predictor skills (main effects only — no
    interactions, unlike the gain family). Each predictor's coefficient
    ``beta_{label}`` is therefore per-SD-of-the-raw-logit, and a ``+1 SD`` (or, for
    a bounded-count predictor, a ``+k items``) perturbation maps to a *scalar*
    linear-predictor shift per posterior draw — so :func:`concurrent_marginals`
    needs none of the per-observation interaction machinery of
    :func:`association_marginals`.

    Attributes
    ----------
    label
        Predictor name for the report row (e.g. ``"L"``, ``"TR"``, ``"age"``).
    coef
        Posterior variable name of the predictor's standardised main-effect
        coefficient (``"beta_L"`` etc.).
    sd_logit
        SD of the predictor's raw same-wave logit on the fitted rows — the data-scale
        size of ``+1 SD``. A ``+k items`` increment at the mean operating point is
        ``Δz = (logit(p̄ + k/N) − logit(p̄)) / sd_logit`` standardised units.
    n_items
        Denominator of the predictor when it is a bounded-count measure; enables the
        ``+k items`` row. ``None`` for age / continuous predictors.
    mean_prop
        Mean baseline proportion of a bounded-count predictor on the fitted rows — the
        operating point at which the ``+k items`` perturbation is evaluated.
    k_items
        The per-predictor items increment for the ``+k items`` row (the pipeline sets
        it per measure, e.g. ``max(1, round(n_items / 10))``, so a fixed ``+5`` does
        not span 3 %–50 % of scales that differ tenfold — the #310/#325 caveat).
    """

    label: str
    coef: str
    sd_logit: float
    n_items: int | None = None
    mean_prop: float | None = None
    k_items: int | None = None


def concurrent_marginals(
    trace: xr.DataTree,
    *,
    terms: Sequence[ConcurrentTerm],
    n_trials: int,
    eta_name: str = "eta",
    ci_prob: float = 0.95,
    group: str = "posterior",
) -> pd.DataFrame:
    """Per-predictor items-scale marginals for the concurrent family (#312).

    For each predictor in ``terms`` it forms the per-draw change in the linear
    predictor from a ``+1 SD`` perturbation of that predictor (and, for a
    bounded-count predictor, a ``+k items`` perturbation at the mean operating
    point), holding every other predictor at its observed value, and averages the
    probability-scale change ``expit(η + Δη) − expit(η)`` over the fitted rows.
    Reported on the probability and items scales (``n_trials`` = the *focal
    outcome's* denominator × probability), with an equal-tailed ``ci_prob`` interval
    and a 90 % band.

    Because the concurrent model has **no interaction terms**, the shift is a scalar
    per draw: ``Δη_s = β_s · Δz`` where ``Δz = 1`` for ``+1 SD`` and
    ``Δz = (logit(p̄ + k/N) − logit(p̄)) / sd_logit`` for ``+k items``. This is the
    mutually-adjusted association's marginal — every row carries
    ``role = "association"``; no term here is causal (post-treatment conditioning is
    intentional, per the family's documented estimand).
    """
    posterior = getattr(trace, group)
    eta = (
        posterior[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    eps = 1e-6
    rows: list[dict[str, float | str]] = []

    for term in terms:
        beta = posterior[term.coef].stack(sample=("chain", "draw")).values.ravel()  # (S,)

        perturbations: list[tuple[str, float]] = [("+1 SD", 1.0)]
        if (
            term.n_items
            and term.mean_prop is not None
            and term.k_items
            and term.sd_logit > 0
        ):
            p = float(np.clip(term.mean_prop, eps, 1 - eps))
            # Cap the increment to what actually fits below the ceiling at the mean
            # operating point: a fixed +k with ``p + k/N > 1`` would be silently
            # clamped, reporting a smaller-than-labelled (or near-zero) shift. Use the
            # largest whole-item ``k_eff <= k`` with ``p + k_eff/N <= 1 - eps`` and
            # label that; if even +1 item exceeds the ceiling, no positive increment is
            # representable, so skip the row.
            max_k = int(np.floor((1.0 - eps - p) * term.n_items))
            k_eff = min(int(term.k_items), max_k)
            if k_eff >= 1:
                p_k = p + k_eff / term.n_items
                dz = (logit(p_k) - logit(p)) / term.sd_logit
                perturbations.append((f"+{k_eff} items", dz))

        for scale_label, dz in perturbations:
            delta_eta = beta * dz  # (S,), scalar shift per draw (no interactions)
            ame_prob = (
                expit(eta + delta_eta[None, :]) - expit(eta)
            ).mean(axis=0)  # (S,)
            ame_items = float(n_trials) * ame_prob
            prob_lo90, prob_hi90 = band90(ame_prob)
            items_lo90, items_hi90 = band90(ame_items)
            rows.append(
                {
                    "term": term.label,
                    "role": "association",
                    "scale": scale_label,
                    "prob_median": float(np.median(ame_prob)),
                    "prob_lo": float(np.quantile(ame_prob, lo_q)),
                    "prob_hi": float(np.quantile(ame_prob, hi_q)),
                    "prob_lo90": prob_lo90,
                    "prob_hi90": prob_hi90,
                    "items_median": float(np.median(ame_items)),
                    "items_lo": float(np.quantile(ame_items, lo_q)),
                    "items_hi": float(np.quantile(ame_items, hi_q)),
                    "items_lo90": items_lo90,
                    "items_hi90": items_hi90,
                    "prob_pos": float(np.mean(ame_items > 0)),
                }
            )
    return pd.DataFrame(rows)


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
    ``eta0 = eta - (b_grp_time[t2] + gamma_grp_ability*ability)*G``, then add back
    **only** ``b_grp_time[t2]`` at ``G=1`` and average ``expit(eta1) - expit(eta0)``
    over the t2 rows. ``gamma_grp_ability`` is a single *time-invariant* coefficient
    (identified mostly from the non-randomised t1/t3/t4 rows), so it is deliberately
    excluded from this causal AME — the card is the clean randomised t2 effect **at
    mean ability**; the group×ability moderation is reported separately, not folded
    into the causal claim (issue #271 item 5).

    Returns ``(contrast_draws, ame_prob)`` — the logit-scale ``b_grp_time[t2]`` draws
    ``(S,)`` (the term flagged causal in the report) and the probability-scale average
    marginal effect per draw ``(S,)``, ready for :func:`rope_card`. ``ability`` is the
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
    # Restrict the causal card to the clean randomised main contrast at MEAN
    # ability: ``gamma_grp_ability`` is a single time-invariant coefficient
    # (identified mostly from the non-randomised t1/t3/t4 rows), so folding it into
    # the t2 AME would borrow a non-randomised component and ~4×-attenuate any real
    # t2 moderation. Net the *full* group contribution out to recover the untreated
    # baseline, but add back only ``b_grp_time[t2]`` (ability is standardised, so
    # "mean ability" simply drops the interaction). The interaction is reported
    # separately, not in the causal card (issue #271 item 5).
    ame_prob = (expit(eta0 + contrast_draws[None, :]) - expit(eta0)).mean(axis=0)  # (S,)
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


# ---------------------------------------------------------------------------
# Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313)
# ---------------------------------------------------------------------------


def _factor_corr_draws(trace: xr.DataTree, group: str = "posterior") -> tuple:
    """Return ``(corr, waves, domains)`` from a longitudinal-CFA ``factor_corr`` node.

    ``corr`` is a numpy array of shape ``(S, T, D, D)`` (sample × wave × domain ×
    domain), ``waves`` the wave labels, ``domains`` the domain names.
    """
    post = getattr(trace, group)
    fc = post["factor_corr"].stack(sample=("chain", "draw"))
    fc = fc.transpose("sample", "wave", "domain", "domain_b")
    corr = np.asarray(fc.values)  # (S, T, D, D)
    waves = [w.item() if hasattr(w, "item") else w for w in fc.coords["wave"].values]
    domains = [str(d) for d in fc.coords["domain"].values]
    return corr, waves, domains


def longitudinal_factor_correlations(
    trace: xr.DataTree, *, ci_prob: float = 0.95, group: str = "posterior"
) -> pd.DataFrame:
    """Per-wave latent factor correlations (the #313 headline).

    One row per (wave, unique off-diagonal domain pair): the posterior mean and
    equal-tailed ``ci_prob`` interval (plus a 90 % band) of the within-wave latent
    correlation, and ``prob_pos`` = ``P(rho > 0)``. These are **disattenuated
    descriptive associations** — the latent skill coupling with binomial counting
    noise modelled out — never causal.
    """
    corr, waves, domains = _factor_corr_draws(trace, group)
    D = len(domains)
    lo_q = (1 - ci_prob) / 2
    rows: list[dict] = []
    for w_i, w in enumerate(waves):
        for i in range(D):
            for j in range(i + 1, D):
                d = corr[:, w_i, i, j]
                lo90, hi90 = band90(d)
                rows.append(
                    {
                        "wave": w,
                        "domain_i": domains[i],
                        "domain_j": domains[j],
                        "mean": float(np.mean(d)),
                        "sd": float(np.std(d)),
                        "lo": float(np.quantile(d, lo_q)),
                        "hi": float(np.quantile(d, 1 - lo_q)),
                        "lo90": lo90,
                        "hi90": hi90,
                        "prob_pos": float(np.mean(d > 0)),
                    }
                )
    return pd.DataFrame(rows)


def longitudinal_conditional_slopes(
    trace: xr.DataTree, *, ci_prob: float = 0.95, group: str = "posterior"
) -> pd.DataFrame:
    """Per-wave conditional (partial) latent slopes among the domain factors.

    For each wave and each ordered pair ``(target, predictor)`` the partial
    regression coefficient of the (unit-variance) target factor on the predictor
    factor **controlling for every other factor**, derived per draw from the
    within-wave latent correlation matrix (the multiple-regression coefficient
    ``beta = R[pred, pred]^-1 R[pred, target]``). This is the disattenuated
    analogue of the concurrent family's mutually-adjusted regression slopes (#312):
    an **adjusted association**, not a causal effect. With two predictors the
    coefficient is a partial slope; with one it coincides with the pairwise
    correlation.
    """
    corr, waves, domains = _factor_corr_draws(trace, group)
    S, T, D, _ = corr.shape
    lo_q = (1 - ci_prob) / 2
    rows: list[dict] = []
    for w_i, w in enumerate(waves):
        R = corr[:, w_i]  # (S, D, D)
        for a in range(D):
            preds = [k for k in range(D) if k != a]
            R_pp = R[:, preds][:, :, preds]  # (S, P, P)
            r_pa = R[:, preds, a]  # (S, P)
            beta = np.linalg.solve(R_pp, r_pa[..., None])[..., 0]  # (S, P)
            for bi, b in enumerate(preds):
                d = beta[:, bi]
                lo90, hi90 = band90(d)
                rows.append(
                    {
                        "wave": w,
                        "target": domains[a],
                        "predictor": domains[b],
                        "mean": float(np.mean(d)),
                        "sd": float(np.std(d)),
                        "lo": float(np.quantile(d, lo_q)),
                        "hi": float(np.quantile(d, 1 - lo_q)),
                        "lo90": lo90,
                        "hi90": hi90,
                        "prob_pos": float(np.mean(d > 0)),
                    }
                )
    return pd.DataFrame(rows)


def disattenuation_crosscheck(
    latent_df: pd.DataFrame, observed_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge latent factor correlations with observed indicator correlations.

    ``latent_df`` is :func:`longitudinal_factor_correlations` output; ``observed_df``
    carries the raw same-wave observed correlation (``observed_corr``) for each
    ``(wave, domain_i, domain_j)`` — the mean pairwise correlation between the two
    domains' standardised indicators. The disattenuation check: measurement-error
    correction can only **inflate** a correlation, so the latent ``|mean|`` should be
    ``>=`` the observed ``|observed_corr|``; ``latent_ge_observed = False`` (a
    reversal) is a red flag for the latent model at this wave/pair. ``gap`` is
    ``|latent| - |observed|``.
    """
    merged = latent_df.merge(
        observed_df, on=["wave", "domain_i", "domain_j"], how="left"
    )
    lat = merged["mean"].abs()
    obs = merged["observed_corr"].abs()
    merged["gap"] = lat - obs
    # A small tolerance absorbs Monte-Carlo noise so a trivially-negative gap does
    # not read as a genuine reversal.
    merged["latent_ge_observed"] = (lat + 1e-3) >= obs
    return merged
