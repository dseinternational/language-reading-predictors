# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Posterior estimands and their summaries, shared across the statistical models.

The quantities a fit reports: the average-marginal-effect cores, the treatment
and factor summaries, the ROPE cards, the DiD and joint contrast tables, the
association and concurrent marginals, the readiness knee, the horseshoe ranking
and the correlated-factor summaries.

Carved out of ``reporting.py`` by #637 stage 3, which is why every name here is
still re-exported from that module. This group depends on nothing else that was
in it.
"""


from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from scipy.special import expit

from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    apply_score_mean_link,
)

# House reporting coverage: median + inner 50% + outer 89% equal-tailed
# (notes/202607172359-credible-interval-standard.md). The single source of truth for
# the *outer* coverage — standalone producers (the prior-sensitivity sweeps) import
# this rather than hard-coding a number, so a report label can never drift from the
# coverage its numbers were computed at (PR #359 review). ``ReportingConfiguration``
# / ``make_context`` default to the same 0.89 for the pipeline path.
REPORTING_CI_PROB = 0.89


def band50(draws: np.ndarray) -> tuple[float, float]:
    """Inner 50 % equal-tailed band ``(lo25, hi75)`` reported alongside the headline.

    A single inner band so the summary builders that report only a headline
    ``ci_prob`` interval can also carry the inner 50 % equal-tailed interval
    without re-deriving quantiles at each call site. The wider ITT / growth
    summaries use the shared ``eti_bands`` helper; this covers the families that
    emit a single headline interval.
    """
    return float(np.quantile(draws, 0.25)), float(np.quantile(draws, 0.75))


def derived_mc_diagnostics(
    draws: np.ndarray,
    *,
    n_chains: int,
    n_draws: int,
    prefix: str = "",
) -> dict[str, float]:
    """Monte-Carlo precision (Bulk-/Tail-ESS + MCSE) of a *derived* estimand.

    Many report headlines are derived quantities computed from posterior draws in
    post-processing rather than as PyMC deterministics — the probability-scale
    average marginal effect / off-floor risk difference (:func:`_itt_ame_draws`),
    the g-formula NDE / NIE (:mod:`...mediation`), and the readiness knee
    (:func:`_readiness_knee`). ``az.summary`` and the convergence gate therefore
    never see them, yet a derived quantity can have materially worse tail effective
    sample size than its parent parameters — the g-formula effects also carry
    mediator re-simulation noise and the knee is a non-smooth argmax — so its MC
    precision must be reported in its own right (Kruschke 2021 BARG step 2.C;
    Vehtari et al. 2021, doi:10.1214/20-BA1221).

    ``draws`` is the sample-stacked ``(chain*draw,)`` array produced by
    ``DataArray.stack(sample=("chain","draw"))`` (chain-major, ``draw`` varying
    fastest), so it is reshaped back to ``(chain, draw)`` for ``az.ess`` / ``az.mcse``
    to recover the between-chain information both need. **Bulk-ESS** governs the
    median / mean; **Tail-ESS** governs the 89 % equal-tailed interval limits — and
    because Tail-ESS is calibrated to the 5 %/95 % quantiles it is the near-exact
    diagnostic for our reported 5.5 %/94.5 % ETI limits. ``mcse_median`` is the
    Monte-Carlo standard error of the reported point estimate. When the layout
    cannot be recovered (e.g. a masked or partly-undefined estimand) the finite
    draws fall back to a single-chain lower bound.
    """
    arr = np.asarray(draws, dtype=float).ravel()
    if arr.size == n_chains * n_draws and np.all(np.isfinite(arr)):
        da = xr.DataArray(arr.reshape(n_chains, n_draws), dims=("chain", "draw"))
    else:
        finite = arr[np.isfinite(arr)]
        da = xr.DataArray(finite[None, :], dims=("chain", "draw"))
    return {
        f"{prefix}ess_bulk": float(az.ess(da, method="bulk")),
        f"{prefix}ess_tail": float(az.ess(da, method="tail")),
        f"{prefix}mcse_median": float(az.mcse(da, method="median")),
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
    row_mask: np.ndarray | None = None,
    score_mean_link: ScoreMeanLink = "logit",
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
    factory used (via its typed fitted payload); the ITT Part-B moderator passes
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

    ``score_mean_link`` is the inverse link used by the fitted score model.  The
    standard ITT models use ``"logit"``.  The registered phoneme-blending
    sensitivity uses ``"three_choice_guessing_floor"`` and therefore maps the
    latent inverse-logit probability onto ``[1/3, 1]`` before differencing.  This
    argument is carried by every estimand and prediction consumer so a fit cannot
    use one link and report an effect from another.

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
    treated_mean = apply_score_mean_link(expit(eta0 + delta), score_mean_link)
    untreated_mean = apply_score_mean_link(expit(eta0), score_mean_link)
    contrib = treated_mean - untreated_mean  # (n_obs, S)
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
    row_mask: np.ndarray | None = None,
    score_mean_link: ScoreMeanLink = "logit",
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
    ``row_mask`` optionally restricts only the population over which the AME is
    averaged; the posterior and all linear predictors still come from the same
    fitted trace. This supports common-population case-deletion comparisons.

    ``ci_prob`` names the *coverage* probability of the headline interval. The
    ``*_lo`` / ``*_hi`` values are the equal-tailed headline credible interval
    (89% by default); ``*_lo50`` / ``*_hi50`` (the inner 50% interval, a visual
    aid) follow the fixed band convention (#177, see :func:`eti_bands`). The ``*_hpdi_lo`` /
    ``*_hpdi_hi`` values are the highest-density interval (HPDI) at ``ci_prob`` — a
    per-scale sensitivity companion (see :func:`hdi_1d`), not a replacement,
    since the HPDI is not transformation-invariant across the logit and
    probability scales. Direction is similarly scale-explicit:
    ``prob_ame_pos`` is the headline posterior probability that the
    probability-scale average marginal effect is positive, while
    ``prob_tau_logit_pos`` is the secondary posterior probability that the
    conditional logit coefficient is positive. ``prob_tau_pos`` is retained only
    as a backward-compatible alias of ``prob_ame_pos`` for existing artefacts and
    downstream readers.
    """
    tau_draws, marginal = _itt_ame_draws(
        trace,
        G=G,
        moderators=moderators,
        row_mask=row_mask,
        score_mean_link=score_mean_link,
    )
    # Monte-Carlo precision of the probability-scale AME — a *derived* estimand
    # (post-processed from draws, so the convergence gate never sees it). Reported
    # alongside the estimate per Kruschke 2021 BARG step 2.C.
    _post = trace.posterior
    _mc = derived_mc_diagnostics(
        marginal,
        n_chains=int(_post.sizes["chain"]),
        n_draws=int(_post.sizes["draw"]),
        prefix="tau_prob_",
    )

    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    tau_median = float(np.median(tau_draws))
    lower, upper = np.quantile(tau_draws, [lo_q, hi_q])
    tau_hpdi_lo, tau_hpdi_hi = hdi_1d(tau_draws, ci_prob)
    tau_b = eti_bands(tau_draws, probs=(0.5,))
    marg_median = float(np.median(marginal))
    marg_lo, marg_hi = np.quantile(marginal, [lo_q, hi_q])
    marg_hpdi_lo, marg_hpdi_hi = hdi_1d(marginal, ci_prob)
    marg_b = eti_bands(marginal, probs=(0.5,))
    # Direction is a statement about the reported sample-average estimand, not
    # necessarily the centred logit coefficient. They have the same sign for a
    # constant treatment effect because expit is monotone, but can disagree when
    # treatment varies by child or is moderated. Keep the coefficient probability
    # as a secondary diagnostic and use the per-draw AME for the headline claim.
    prob_logit_pos = float(np.mean(tau_draws > 0))
    prob_ame_pos = float(np.mean(marginal > 0))
    # Posterior mean retained as a *secondary* field on each scale (issue #144):
    # the median leads (transformation-invariant, and it discounts the
    # winner's-curse right tail), but the mean is kept available for reference.
    tau_mean = float(np.mean(tau_draws))
    marg_mean = float(np.mean(marginal))

    # Fixed band convention (#177): inner 50% (visual aid) alongside the
    # equal-tailed headline interval at ``ci_prob`` (``*_lo`` / ``*_hi``).
    return {
        "tau_logit_median": tau_median,
        "tau_logit_mean": tau_mean,
        "tau_logit_lo50": tau_b["lo50"],
        "tau_logit_hi50": tau_b["hi50"],
        "tau_logit_lo": float(lower),
        "tau_logit_hi": float(upper),
        "tau_logit_hpdi_lo": tau_hpdi_lo,
        "tau_logit_hpdi_hi": tau_hpdi_hi,
        "tau_prob_median": marg_median,
        "tau_prob_mean": marg_mean,
        "tau_prob_lo50": marg_b["lo50"],
        "tau_prob_hi50": marg_b["hi50"],
        "tau_prob_lo": float(marg_lo),
        "tau_prob_hi": float(marg_hi),
        "tau_prob_hpdi_lo": marg_hpdi_lo,
        "tau_prob_hpdi_hi": marg_hpdi_hi,
        "prob_ame_pos": prob_ame_pos,
        # Backward-compatible alias. New report code and callers should use the
        # scale-explicit ``prob_ame_pos`` field for the headline direction.
        "prob_tau_pos": prob_ame_pos,
        "prob_tau_logit_pos": prob_logit_pos,
        "direction_label": evidence_label(prob_ame_pos),
        **favoured_direction(prob_ame_pos),
        **_mc,
    }


def tau_summary_offfloor(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    G: np.ndarray,
) -> dict[str, float]:
    """Summarise the post-hoc binary off-floor exploratory effect (#119/#341).

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
        f"equal-tailed 89% credible interval {r['items_lo'] * scale:+.1f} to "
        f"{r['items_hi'] * scale:+.1f}). "
        f"{direction_clause} "
        f"**Magnitude** — evidence the benefit is at least δ = {r['delta_items'] * scale:g} "
        f"{unit}: P = {r['prob_benefit_ge_delta']:.3f} "
        f"({odds_string(r['prob_benefit_ge_delta'])}, *{r['benefit_label']} evidence*); "
        f"probability inside the ROPE (practically negligible): {r['prob_in_rope']:.3f}.\n"
    )
    if "items_hpdi_lo" in cols:
        parts.append(
            f"_Sensitivity — the 89% highest posterior density interval (HPDI) on the "
            f"{unit} scale is {r['items_hpdi_lo'] * scale:+.1f} to "
            f"{r['items_hpdi_hi'] * scale:+.1f}. HPDI is not transformation-invariant, "
            f"so it is a scale-specific check, not a replacement for the equal-tailed "
            f"interval above._\n"
        )
    return "\n".join(parts)


def drop_retired_90_band(card):
    """Remove the retired ``*_lo90``/``*_hi90`` fields from a ROPE card.

    The external ``rope_card`` still emits a 90% band; the suite retired it
    (2026-07-17 credible-interval standard) in favour of median + 50% + 89%.
    Shared so every family that builds a ROPE card — via :func:`rope_summary`
    or by calling ``rope_card`` directly, as the level pipeline does — strips
    the same columns (2026-08-20 level-factors review, finding 3). Accepts the
    plain dict or the DataFrame form ``rope_card`` returns.
    """

    def _is90(key) -> bool:
        k = str(key)
        return k.endswith("_lo90") or k.endswith("_hi90")

    if isinstance(card, dict):
        return {k: v for k, v in card.items() if not _is90(k)}
    return card.drop(columns=[c for c in card.columns if _is90(c)])


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
    direction_from_ame: bool = False,
    score_mean_link: ScoreMeanLink = "logit",
) -> dict[str, float | str]:
    """ROPE-anchored continuous report card for a randomised treatment effect.

    Built on :func:`_itt_ame_draws`, so it shares the average-marginal-effect core
    with :func:`tau_summary_itt`. Reports the effect on the logit scale (``term``)
    and the items scale (the average marginal effect × ``n_trials``) as a **median**
    with a 50 % and a ``ci_prob`` (default 89 %) equal-tailed interval, plus:

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

    ``direction_from_ame`` (default False → ITT behaviour unchanged): when True the
    direction fields (``pd`` / ``direction_label`` / ``favoured_direction*``) are taken
    from the probability-scale AME rather than the coefficient, and ``pd_coef`` records
    the coefficient direction. The gain-factor family sets this because its treatment
    interactions make the coefficient and the marginal effect diverge in sign (#391).
    """
    effect_draws, ame_prob = _itt_ame_draws(
        trace,
        G=G,
        term=term,
        varying_term=varying_term,
        eta_name=eta_name,
        moderators=moderators,
        row_mask=row_mask,
        score_mean_link=score_mean_link,
    )
    items = ame_prob * float(n_trials)
    card = rope_card(effect_draws, items, delta=delta, ci_prob=ci_prob)
    # The external rope_card still emits a 90% band (`*_lo90`/`*_hi90`); the suite
    # retired it (2026-07-17 credible-interval standard). Drop it here so the raw
    # rope table matches the median + 50% + 89% convention everywhere it surfaces.
    # rope_card returns a plain dict of scalars.
    # rope_card derives its direction fields (``pd`` / ``direction_label`` /
    # ``favoured_direction*``) from the first argument — the coefficient draws. With
    # active treatment interactions the coefficient and the marginal effect can differ
    # in sign per draw, so ``direction_from_ame`` re-derives the direction from the
    # probability-scale AME (the reported estimand), exactly as ``tau_summary_itt``,
    # and keeps the coefficient direction as ``pd_coef`` (#391). Benefit/harm/ROPE
    # already use the items (AME) draws, so they are unaffected.
    def _redirect_from_ame(mapping):
        prob_ame_pos = float(np.mean(ame_prob > 0))
        mapping["pd_coef"] = mapping["pd"]
        mapping["pd"] = prob_ame_pos
        mapping["direction_label"] = evidence_label(prob_ame_pos)
        mapping.update(favoured_direction(prob_ame_pos))
        return mapping

    if isinstance(card, dict):
        card = drop_retired_90_band(card)
        return _redirect_from_ame(card) if direction_from_ame else card
    card = drop_retired_90_band(card)
    if direction_from_ame:
        prob_ame_pos = float(np.mean(ame_prob > 0))
        card["pd_coef"] = float(card["pd"].iloc[0])
        card["pd"] = prob_ame_pos
        card["direction_label"] = evidence_label(prob_ame_pos)
        for _k, _v in favoured_direction(prob_ame_pos).items():
            card[_k] = _v
    return card


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
    score_mean_link: ScoreMeanLink = "logit",
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
        trace,
        G=G,
        term=term,
        varying_term=varying_term,
        eta_name=eta_name,
        moderators=moderators,
        row_mask=row_mask,
        score_mean_link=score_mean_link,
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
        out[f"{name}_lo50"], out[f"{name}_hi50"] = band50(d)
        out[f"prob_{name}_pos"] = float(np.mean(d > 0))
    return out


#: Qualification thresholds for calling a steepest latent-logit interval a "knee"
#: (#586 finding 1). ``_KNEE_MIN_INCREASING`` is the pre-existing net-rise share;
#: ``_KNEE_MIN_CURVATURE`` is the shared evidence ladder's "moderate" rung (10:1
#: odds, ``dse_research_utils.statistics.evidence``), applied to the local slope
#: contrast so a straight line — which sits near 0.5 whatever its net rise — can
#: never qualify.
_KNEE_MIN_INCREASING: float = 0.9


_KNEE_MIN_CURVATURE: float = 0.91


def _readiness_knee(
    f: np.ndarray,
    ell: np.ndarray | None,
    *,
    n_trials: int | None = None,
    count_values: np.ndarray | None = None,
    ci_prob: float = 0.89,
    n_bins: int = 6,
    n_chains: int | None = None,
    n_draws: int | None = None,
) -> dict[str, float]:
    """Locate the steepest latent-logit interval of a per-observation ``f_mech`` posterior.

    Pure-numpy core of :func:`readiness_threshold` (split out so the logic is
    unit-testable without a trace, #293 review). ``f`` is ``(n_obs, n_draws)`` HSGP
    curve draws; ``ell`` is the ``(n_obs,)`` Haldane-corrected mechanism logit.

    **What the statistic is.** The located quantity is the between-bin interval with
    the largest derivative of ``f_mech`` — the *steepest latent-logit interval*. The
    derivative is taken on the outcome-**logit** scale, because ``f_mech`` is a logit
    contribution; the expected-items derivative carries an extra ``p * (1 - p)``
    inverse-link factor and can peak at a different exposure value (#586 finding 1).
    ``scale`` records this so no downstream renderer can silently call it an items
    result. ``half_rise_count_*`` is a complementary mid-rise summary (where the
    curve first reaches the midpoint of its binned range). Both are summarised over
    the *increasing* draws only (net end-to-end rise on the binned curve; the share
    is ``increasing_frac``) — on a flat or falling draw the estimands are undefined.

    **When it may be called a knee.** A net rise is not a threshold: a perfectly
    linear increasing curve has ``increasing_frac == 1`` and still yields an
    ``argmax``, and a curve that accelerates all the way to the edge of its support
    pins that ``argmax`` on the last interval, where it is censored by the data
    rather than located by them. Both failure modes were live — the letter-sound
    fits put 73% of draws in the top interval with the knee median equal to its own
    upper credible limit (#586 finding 1). ``knee_well_defined`` is therefore a
    conjunction of three checks, each also reported on its own so a reader can see
    which one failed:

    - ``increasing_frac`` > ``_KNEE_MIN_INCREASING`` — the curve rises at all;
    - ``not boundary_pinned`` — the modal steepest interval is interior, so the
      location is identified rather than censored by the end of the observed range;
    - ``prob_slope_above_gt_below`` >= ``_KNEE_MIN_CURVATURE`` — the mean slope above
      the located interval genuinely exceeds the mean slope below it. For a straight
      line this probability sits near 0.5 whatever ``increasing_frac`` says, which is
      what separates a bend from a constant rise.

    ``steepest_interval_share`` is the share of increasing draws whose ``argmax``
    falls in the modal interval — selection stability, low when the intervals are
    effectively tied.
    """
    if count_values is not None:
        # Continuous-covariate exposure (e.g. intervention sessions, LRP92): the knee
        # is located in the exposure's own raw units directly — there is no bounded
        # count and no logit -> count back-transform. ``knee_count_*`` /
        # ``half_rise_count_*`` / ``obs_count_*`` then read in those raw units.
        L = np.asarray(count_values, dtype=float).reshape(-1)
    else:
        # Inverse Haldane-corrected logit -> approximate predictor count, clipped to range.
        # ell = log((y+0.5)/(n-y+0.5)) => expit(ell) = (y+0.5)/(n+1), so y = (n+1)*expit(ell) - 0.5
        # (the denominator is n+1, not n; #293 review).
        if ell is None or n_trials is None:
            raise ValueError("_readiness_knee needs ell + n_trials unless count_values is given.")
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
        # An all-NaN "below" set is a real outcome, not an error: it means the
        # steepest interval is the lowest one, so there is nothing below it to
        # average. It stays NaN and the renderer must say so rather than print it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            slope_below = np.nanmean(np.where(below[:, None], slope, np.nan), axis=0)
            slope_above = np.nanmean(np.where(~below[:, None], slope, np.nan), axis=0)
    else:  # no increasing draws — the below/above split is undefined
        slope_below = slope_above = np.full(f.shape[1], np.nan)

    def _med(a: np.ndarray) -> float:
        a = a[np.isfinite(a) & increasing]
        return float(np.median(a)) if a.size else float("nan")

    # --- qualification diagnostics (#586 finding 1) -------------------------
    # Selection stability and boundary censoring are properties of the *argmax*
    # over intervals, so they are computed on the increasing draws' winning bins.
    n_intervals = nb - 1
    kb_increasing = knee_bin[increasing]
    if kb_increasing.size:
        counts = np.bincount(kb_increasing, minlength=n_intervals)
        modal_interval = int(np.argmax(counts))
        interval_share = float(counts[modal_interval] / kb_increasing.size)
        # An argmax on the first or last interval is censored by the end of the
        # observed range: the curve may go on steepening where there are no data,
        # so the location is a bound, not an estimate.
        boundary_pinned = modal_interval in (0, n_intervals - 1)
    else:
        modal_interval, interval_share, boundary_pinned = -1, float("nan"), True

    # Local slope contrast: does the curve genuinely bend? For a straight line the
    # above/below means coincide and this sits near 0.5 however strongly the curve
    # rises, which is precisely the case ``increasing_frac`` cannot detect.
    contrast = slope_above - slope_below
    contrast = contrast[np.isfinite(contrast) & increasing]
    prob_curvature = float(np.mean(contrast > 0)) if contrast.size else float("nan")

    increasing_frac = float(np.mean(increasing))
    well_defined = bool(
        increasing_frac > _KNEE_MIN_INCREASING
        and not boundary_pinned
        and np.isfinite(prob_curvature)
        and prob_curvature >= _KNEE_MIN_CURVATURE
    )

    result = {
        "knee_count_median": kmed,
        "knee_count_ci_low": k_lo,
        "knee_count_ci_high": k_hi,
        "half_rise_count_median": hmed,
        "half_rise_count_ci_low": h_lo,
        "half_rise_count_ci_high": h_hi,
        "slope_below_knee_median": _med(slope_below),
        "slope_above_knee_median": _med(slope_above),
        "increasing_frac": increasing_frac,
        # The derivative is a logit-scale contribution, never expected items: the
        # items-scale maximum carries an extra p*(1-p) factor and can sit elsewhere.
        "scale": "latent_logit",
        "steepest_interval_index": modal_interval,
        "steepest_interval_share": interval_share,
        "boundary_pinned": boundary_pinned,
        "prob_slope_above_gt_below": prob_curvature,
        "knee_well_defined": well_defined,
        "obs_count_min": float(L.min()),
        "obs_count_max": float(L.max()),
        "ci_prob": float(ci_prob),
        "n_draws": int(f.shape[1]),
        "n_obs": int(f.shape[0]),
        "n_bins": int(nb),
    }
    # Monte-Carlo precision of the derived knee location (a non-smooth argmax over
    # binned draws, so it can mix worse than its parent GP weights). ESS is computed
    # over all draws to keep the chain layout; the reported median/CI pool the
    # ``increasing`` subset (share ``increasing_frac``).
    if n_chains is not None and n_draws is not None:
        result.update(
            derived_mc_diagnostics(
                knee_L, n_chains=n_chains, n_draws=n_draws, prefix="knee_"
            )
        )
    return result


def readiness_threshold(
    trace: xr.DataTree,
    *,
    n_trials: int | None = None,
    exposure_values: np.ndarray | None = None,
    ci_prob: float = 0.89,
    n_bins: int = 6,
    curve: np.ndarray | None = None,
    scale: str = "latent_logit",
) -> dict[str, float]:
    """Steepest latent-logit interval of a mechanism curve (#230 §2/§5, #586 finding 1).

    Post-processes an HSGP mechanism model's adjusted curve ``f_mech`` to locate the
    interval over which it rises fastest, in the predictor's raw count units. For each
    posterior draw the per-observation ``f_mech`` is binned over the observed predictor
    range (quantile bins) and the steepest between-bin rise is found; the reported
    location is that interval's midpoint, giving a posterior over it. Reports its
    median + equal-tailed CI, a complementary half-rise summary, and the mean marginal
    slope below vs above it.

    The derivative is on the outcome-**logit** scale (``f_mech`` is a logit
    contribution), not the items scale: the expected-items derivative carries an extra
    ``p * (1 - p)`` factor and its maximum can fall at a different exposure value. The
    returned ``scale`` field records this.

    A located interval is **not** by itself a threshold. ``knee_well_defined``
    combines the net-rise share with a boundary check (an ``argmax`` on the first or
    last interval is censored by the end of the observed range) and a local
    slope-contrast probability (near 0.5 for a straight line). Read
    ``increasing_frac``, ``boundary_pinned``, ``steepest_interval_share`` and
    ``prob_slope_above_gt_below`` alongside the location; only call it a knee when
    ``knee_well_defined`` is true.

    Pure post-processing (no re-fit): needs the ``f_mech`` posterior and the
    ``mech_post_logit`` constant-data node of a standard HSGP mechanism fit (e.g.
    ``lrp-rli-mech-058``). ``n_trials`` is the mechanism predictor's item ceiling (letter
    sounds = 32) used to back-transform the logit input to an approximate count.

    For a continuous-covariate exposure (``mechanism_is_covariate`` with the HSGP curve
    on, e.g. ``lrp-rli-mech-191`` sessions -> word reading), pass ``exposure_values``
    (the per-observation raw exposure, in the same order as ``f_mech``'s rows) instead
    of ``n_trials``; the knee/half-rise/``obs_count_*`` fields are then in the
    exposure's own raw units (e.g. sessions) rather than a bounded count.
    """
    post = trace.posterior
    if curve is None:
        if "f_mech" not in post:
            raise KeyError(
                "trace has no 'f_mech' posterior — the readiness threshold needs an "
                "HSGP mechanism fit (not the linear-mechanism or phase-specific "
                "variant)."
            )
        # The HSGP ``f_mech`` carries an auto-named obs dimension (e.g.
        # ``f_mech_dim_0``), not ``obs_id``; take whichever non-sample dim it has. Its
        # rows are in the model's observation order, aligned to the
        # ``mech_post_logit`` constant-data node below.
        f_stacked = post["f_mech"].stack(sample=("chain", "draw"))
        obs_dim = next(d for d in f_stacked.dims if d != "sample")
        f = f_stacked.transpose(obs_dim, "sample").values  # (n_obs, S)
    else:
        # Caller-supplied curve on another scale — the expected-items curve
        # standardised over the fitted rows (#602). The binning, argmax, boundary and
        # curvature logic is scale-free, so it is reused verbatim; only ``scale``
        # changes, and it is recorded so no renderer can confuse the two.
        f = np.asarray(curve, dtype=float)
        if f.ndim != 2:
            raise ValueError("curve must be a (n_obs, n_draws) array")
    n_chains, n_draws = int(post.sizes["chain"]), int(post.sizes["draw"])
    if exposure_values is not None:
        # Continuous-covariate exposure: the knee lives in the exposure's own units.
        # ``exposure_values`` must be in the same observation order as the curve rows.
        result = _readiness_knee(
            f, None,
            count_values=np.asarray(exposure_values, dtype=float).reshape(-1),
            ci_prob=ci_prob, n_bins=n_bins, n_chains=n_chains, n_draws=n_draws,
        )
    else:
        ell = np.asarray(
            trace.constant_data["mech_post_logit"].values
        ).reshape(-1)  # (n_obs,)
        result = _readiness_knee(
            f, ell, n_trials=n_trials, ci_prob=ci_prob, n_bins=n_bins,
            n_chains=n_chains, n_draws=n_draws,
        )
    result["scale"] = scale
    return result


def did_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    n_trials: int,
    dose: bool = False,
    off_floor: bool = False,
    child_idx: np.ndarray | None = None,
    standardization_cells: Mapping[str, np.ndarray] | None = None,
    wave: np.ndarray | None = None,
    score_mean_link: ScoreMeanLink = "logit",
    subject_ids: np.ndarray | None = None,
) -> dict[str, float | bool | str]:
    """Summarise a waitlist-crossover arm-by-wave model (kind="did").

    The current binary model exposes three immediate-minus-waitlist logit contrasts:
    ``arm_gap_t1`` (pre-randomisation balance), ``tau_t2`` (the randomised
    immediate-treatment-versus-no-treatment assignment contrast at t2) and
    ``arm_gap_t3`` (a **different randomised contrast** — assignment to the
    early-start rather than the delayed-start treatment schedule, both arms being
    treated by t3). Its derived ``delta_crossover = tau_t2 - arm_gap_t3`` is the
    change between those two randomised regime contrasts: positive means the gap
    between the arms is smaller at t3 than at t2. It is **not** an identified
    catch-up mechanism — duration, carryover, maturation, ceiling effects and
    different taught blocks are inseparable in it (#576 finding 3) — and it is not a
    second treated-versus-untreated effect.

    ``score_mean_link`` is the inverse link of the fitted score model. The
    phoneme-blending guessing-floor companion maps the mean onto ``[1/3, 1]``, so
    every outcome-scale quantity here must go through the same link the likelihood
    used; reading ``expit(eta)`` directly would understate the fitted score by up to
    a third of the test (#576 finding 2).

    ``wave`` must contain the fitted row's zero-based t1/t2/t3 code (0/1/2). For
    each wave, the function standardises both arms over that wave's fitted rows
    using ``eta_base``, which excludes the arm term. It reports the two standardised
    arm means and their immediate-minus-waitlist difference on the outcome scale.
    ``delta_crossover_items_*`` is the t2 standardised arm gap minus the t3
    standardised arm gap, not ``expit(delta_crossover)``. Because the logit link is
    nonlinear, this outcome-scale change-in-gap depends on the wave-specific
    operating points. These are fitted-sample standardisations and the t2 quantity
    is not numerically interchangeable with an ITT summary standardised over a
    different fitted sample or covariate distribution.

    For the exploratory varying-crossover model, ``delta_crossover_i`` is averaged
    over the fitted waitlist children per posterior draw and reported separately as
    ``delta_crossover_sample_average_*``. The outcome-scale change-in-gap is omitted
    for that model because a scalar arm-gap toggle would fail to integrate the
    fitted child-specific catch-up terms. For the same reason its t3 standardised
    quantities (``t3_waitlist_items_*``, ``t3_immediate_items_*``,
    ``arm_gap_t3_items_*``) are omitted: ``eta_base`` excludes the fitted
    ``v_delta`` deviations, so a population-mean t3 toggle would misstate the
    fitted waitlist t3 level. ``arm_gap_t3_items_available`` records the omission;
    the t1/t2 quantities are unaffected (``v_delta`` enters only waitlist t3 rows),
    and the logit-scale ``arm_gap_t3`` posterior is always reported.

    The legacy ``beta_period``/``delta`` branch remains readable so existing traces
    fail gracefully during the refit transition. Its ``delta_items_*`` quantity is
    a fitted-row model-implied treated-versus-untreated toggle, not a four-cell DiD
    cross-difference and not automatically comparable with the corresponding
    available-case modified ITT estimate.

    When the posterior contains child-specific ``delta_i`` draws, ``child_idx`` is
    required and must map each fitted row to the corresponding ``child`` position.
    The marginal effect then uses the fitted child's posterior slope rather than the
    population-mean ``delta``. This is conditional standardisation over the fitted
    children; it does not integrate a new child's random slope from the population
    distribution. For a constant-effect fit, ``child_idx`` is ignored.

    ``standardization_cells`` optionally maps short, identifier-like names (for
    example ``{"p1": phase == 0, "waitlist_p1": ...}``) to boolean masks aligned
    with the fitted rows. Each cell receives a companion
    ``delta_items_{name}_*`` summary. These remain model-implied treatment toggles
    at that cell's covariate distribution, rather than observed arm contrasts.

    With ``off_floor=True`` (the off-floor prevalence DiD for heavily-floored P / N,
    fitted as a Bernoulli on the off-floor indicator) the caller passes
    ``n_trials=1``. Every ``*_items_*`` field is then on the probability scale:
    arm-gap fields are off-floor risk differences and cell fields are probabilities
    of *being* off the floor at that wave, not item counts or transitions from the
    floor. The returned ``off_floor`` flag lets the report label the scale.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    def _summ_draws(name: str, draws: np.ndarray) -> dict[str, float | str]:
        d = np.asarray(draws).ravel()
        prob_pos = float(np.mean(d > 0))
        lo50, hi50 = band50(d)
        return {
            # Median-first to match the ITT tau_summary_itt convention (#144 / #271);
            # the mean is kept as a secondary column.
            f"{name}_median": float(np.median(d)),
            f"{name}_mean": float(np.mean(d)),
            f"{name}_lo": float(np.quantile(d, lo_q)),
            f"{name}_hi": float(np.quantile(d, hi_q)),
            f"{name}_lo50": lo50,
            f"{name}_hi50": hi50,
            f"prob_{name}_pos": prob_pos,
            f"{name}_direction_label": evidence_label(prob_pos),
            f"{name}_favoured_direction": "positive" if prob_pos >= 0.5 else "negative",
            f"{name}_favoured_label": evidence_label(max(prob_pos, 1.0 - prob_pos)),
        }

    def _summ(name: str) -> dict[str, float | str]:
        return _summ_draws(
            name, posterior[name].stack(sample=("chain", "draw")).values
        )

    out: dict[str, float | bool | str] = {}

    def _effect_summary(draws: np.ndarray, *, prefix: str) -> None:
        scaled = draws * n_trials
        out[f"{prefix}_median"] = float(np.median(scaled))
        out[f"{prefix}_mean"] = float(np.mean(scaled))
        out[f"{prefix}_lo"] = float(np.quantile(scaled, lo_q))
        out[f"{prefix}_hi"] = float(np.quantile(scaled, hi_q))
        out[f"{prefix}_lo50"], out[f"{prefix}_hi50"] = band50(scaled)

    if "tau_t2" in posterior:
        required = {"arm_gap_t1", "arm_gap_t3", "delta_crossover", "eta_base"}
        missing = sorted(required.difference(posterior.data_vars))
        if missing:
            raise KeyError(
                "arm-by-wave DiD trace is missing required posterior nodes: "
                + ", ".join(missing)
            )
        for name in ("arm_gap_t1", "tau_t2", "arm_gap_t3", "delta_crossover"):
            out.update(_summ(name))
        if "delta_crossover_i" in posterior:
            child_draws = (
                posterior["delta_crossover_i"]
                .stack(sample=("chain", "draw"))
                .transpose("waitlist_child", "sample")
                .values
            )
            out.update(
                _summ_draws(
                    "delta_crossover_sample_average", child_draws.mean(axis=0)
                )
            )
            out["delta_crossover_sample_n_children"] = int(child_draws.shape[0])
        if dose and "beta_dose" in posterior:
            out.update(_summ("beta_dose"))

        if wave is None:
            raise ValueError(
                "wave is required for arm-by-wave outcome-scale standardisation; "
                "pass the fitted prepared.phase array."
            )
        wave_arr = np.asarray(wave)
        if wave_arr.ndim != 1:
            raise ValueError(f"wave must be 1-D, got a {wave_arr.ndim}-D array.")
        if not np.issubdtype(wave_arr.dtype, np.integer):
            raise ValueError(f"wave must contain integer phase codes, got {wave_arr.dtype}.")
        eta_base = (
            posterior["eta_base"]
            .stack(sample=("chain", "draw"))
            .transpose("obs_id", "sample")
            .values
        )  # (n_obs, S)
        if wave_arr.shape[0] != eta_base.shape[0]:
            raise ValueError(
                f"wave has {wave_arr.shape[0]} rows but eta_base has "
                f"{eta_base.shape[0]} observations; pass the fitted-subset phases."
            )

        varying_catch_up = "delta_crossover_i" in posterior
        wave_effects: dict[str, np.ndarray] = {}
        wave_terms = (
            (0, "t1", "arm_gap_t1"),
            (1, "t2", "tau_t2"),
            (2, "t3", "arm_gap_t3"),
        )
        for wave_code, wave_name, term_name in wave_terms:
            rows = wave_arr == wave_code
            if not np.any(rows):
                raise ValueError(
                    f"wave contains no {wave_name} rows (expected phase code {wave_code})."
                )
            if varying_catch_up and wave_code == 2:
                # The fitted waitlist-child catch-up deviations (v_delta) enter
                # the waitlist t3 rows but are absent from eta_base, so a scalar
                # arm-gap toggle would misstate the fitted t3 levels — the same
                # partial-integration reason delta_crossover_items is withheld
                # below. Omit rather than publish a partially-integrated summary.
                continue
            gap = (
                posterior[term_name]
                .stack(sample=("chain", "draw"))
                .values.ravel()
            )
            waitlist = apply_score_mean_link(
                expit(eta_base[rows]), score_mean_link
            ).mean(axis=0)
            immediate = apply_score_mean_link(
                expit(eta_base[rows] + gap[None, :]), score_mean_link
            ).mean(axis=0)
            arm_gap = immediate - waitlist
            _effect_summary(waitlist, prefix=f"{wave_name}_waitlist_items")
            _effect_summary(immediate, prefix=f"{wave_name}_immediate_items")
            _effect_summary(arm_gap, prefix=f"{term_name}_items")
            out[f"{term_name}_items_n_rows"] = int(rows.sum())
            wave_effects[term_name] = arm_gap

        out["arm_gap_t3_items_available"] = not varying_catch_up
        if varying_catch_up:
            out["arm_gap_t3_items_omission_reason"] = (
                "the fitted waitlist-child catch-up deviations are not "
                "integrated by a scalar arm-gap toggle"
            )

        if not varying_catch_up:
            _effect_summary(
                wave_effects["tau_t2"] - wave_effects["arm_gap_t3"],
                prefix="delta_crossover_items",
            )
            out["delta_crossover_items_available"] = True
            out["delta_crossover_items_population"] = "wave_specific_fitted_rows"
            # Common-child gap change (#576 material qualification 6). The two legs
            # above are each standardised over their own wave's fitted rows. When a
            # child is observed at t2 but not t3 those row sets differ, and the
            # difference then mixes the change over time with a change in *who* is
            # being averaged over. Recomputing both legs on the children present at
            # both waves separates the two; the wave-specific quantity is retained
            # beside it, and the recorded flag says whether they can differ at all.
            if subject_ids is not None:
                ids = np.asarray(subject_ids).astype(str)
                if ids.shape[0] != eta_base.shape[0]:
                    raise ValueError(
                        f"subject_ids has {ids.shape[0]} rows but eta_base has "
                        f"{eta_base.shape[0]} observations; pass the fitted-subset ids."
                    )
                common = np.intersect1d(ids[wave_arr == 1], ids[wave_arr == 2])
                out["delta_crossover_items_common_n_children"] = int(common.size)
                out["delta_crossover_items_common_population_identical"] = bool(
                    common.size == np.unique(ids[wave_arr == 1]).size
                    and common.size == np.unique(ids[wave_arr == 2]).size
                )
                if common.size:
                    in_common = np.isin(ids, common)
                    common_effects: dict[str, np.ndarray] = {}
                    for wave_code, term_name in ((1, "tau_t2"), (2, "arm_gap_t3")):
                        rows = (wave_arr == wave_code) & in_common
                        gap = (
                            posterior[term_name]
                            .stack(sample=("chain", "draw"))
                            .values.ravel()
                        )
                        base = eta_base[rows]
                        common_effects[term_name] = (
                            apply_score_mean_link(
                                expit(base + gap[None, :]), score_mean_link
                            ).mean(axis=0)
                            - apply_score_mean_link(
                                expit(base), score_mean_link
                            ).mean(axis=0)
                        )
                    for term_name, prefix in (
                        ("tau_t2", "tau_t2_items_common"),
                        ("arm_gap_t3", "arm_gap_t3_items_common"),
                    ):
                        _effect_summary(common_effects[term_name], prefix=prefix)
                    _effect_summary(
                        common_effects["tau_t2"] - common_effects["arm_gap_t3"],
                        prefix="delta_crossover_items_common",
                    )
                    out["delta_crossover_items_common_available"] = True
                else:
                    out["delta_crossover_items_common_available"] = False
            else:
                out["delta_crossover_items_common_available"] = False
        else:
            out["delta_crossover_items_available"] = False
            out["delta_crossover_items_omission_reason"] = (
                "child-specific catch-up requires an explicitly integrated "
                "waitlist-child counterfactual"
            )
        out["arm_wave_marginal_estimand"] = (
            "wave-specific fitted-row standardized immediate-minus-waitlist arm gap"
        )
        out["arm_wave_marginal_effect_source"] = (
            "population-mean arm gaps; child-specific catch-up is not integrated"
            if "delta_crossover_i" in posterior
            else "fixed arm gaps"
        )
        out["score_mean_link"] = str(score_mean_link)
        out["tau_t2_interpretation"] = (
            "randomised assignment contrast: immediate treatment versus no treatment "
            "yet, read at t2"
        )
        out["arm_gap_t3_interpretation"] = (
            "randomised assignment contrast between treatment schedules: early-start "
            "versus delayed-start treatment history at t3; not a treated-versus-"
            "untreated effect"
        )
        out["delta_crossover_interpretation"] = (
            "change between two randomised regime contrasts (t2 gap minus t3 gap); "
            "not an identified catch-up mechanism"
        )
        out["off_floor"] = bool(off_floor)
        return out

    out.update(_summ("beta_period"))
    if dose:
        # The redesigned dose model separates the saturated arm-by-period cell
        # structure from intensive session variation. Report the arm and cell
        # coefficients whenever the trace carries them so the observational
        # beta_dose is not presented as though it were the randomised on/off
        # contrast. Under that saturated coding (treated = immediate arm OR
        # period 2) theta_treated at the mean treated dose is the crossover
        # *cell* contrast, not an isolated treatment-presence effect (#631
        # finding 12; the resolver in did.py says the same).
        for name in ("beta_group", "theta_treated", "gamma_t1", "beta_dose"):
            if name in posterior:
                out.update(_summ(name))
        out["dose_interpretation"] = (
            "beta_dose is an observational intensive-margin association; "
            "theta_treated is the crossover cell contrast at the mean treated "
            "dose, not an isolated treatment-presence effect"
        )
        return out

    out.update(_summ("delta"))
    # Model-implied treated-vs-untreated contrast, standardised over the fitted
    # rows. For the varying-slope fit, map each child's posterior delta_i to every
    # row belonging to that child; using the scalar population mean here would not
    # report the model that was actually fitted.
    delta = posterior["delta"].stack(sample=("chain", "draw")).values.ravel()  # (S,)
    eta_base = (
        posterior["eta_base"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    if "delta_i" in posterior:
        if child_idx is None:
            raise ValueError(
                "child_idx is required when the DiD posterior contains child-specific "
                "delta_i draws."
            )
        idx = np.asarray(child_idx)
        if idx.ndim != 1:
            raise ValueError(f"child_idx must be 1-D, got a {idx.ndim}-D array.")
        if not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"child_idx must contain integer positions, got {idx.dtype}.")
        if idx.shape[0] != eta_base.shape[0]:
            raise ValueError(
                f"child_idx has {idx.shape[0]} rows but eta_base has "
                f"{eta_base.shape[0]} observations; pass the fitted-subset mapping."
            )
        child_delta = (
            posterior["delta_i"]
            .stack(sample=("chain", "draw"))
            .transpose("child", "sample")
            .values
        )  # (n_child, S)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= child_delta.shape[0]):
            raise ValueError(
                f"child_idx contains positions outside [0, {child_delta.shape[0]})."
            )
        row_delta = child_delta[idx]  # (n_obs, S)
        effect_source = "child_specific_delta_i"
    else:
        row_delta = delta[None, :]  # (1, S), broadcast over observations
        effect_source = "population_mean_delta"

    row_effect = expit(eta_base + row_delta) - expit(eta_base)  # (n_obs, S)

    _effect_summary(row_effect.mean(axis=0), prefix="delta_items")
    out["delta_standardization_n_rows"] = int(eta_base.shape[0])
    cell_names: list[str] = []
    for name, raw_mask in (standardization_cells or {}).items():
        if not name.isascii() or not name.isidentifier():
            raise ValueError(
                "standardization cell names must be non-empty ASCII identifiers; "
                f"got {name!r}."
            )
        mask = np.asarray(raw_mask)
        if mask.ndim != 1:
            raise ValueError(
                f"standardization cell {name!r} must be 1-D, got {mask.ndim}-D."
            )
        if mask.dtype != bool:
            raise ValueError(
                f"standardization cell {name!r} must be a boolean mask, got "
                f"{mask.dtype}."
            )
        if mask.shape[0] != eta_base.shape[0]:
            raise ValueError(
                f"standardization cell {name!r} has {mask.shape[0]} rows but "
                f"eta_base has {eta_base.shape[0]} observations."
            )
        if not np.any(mask):
            raise ValueError(f"standardization cell {name!r} selects no observations.")
        _effect_summary(row_effect[mask].mean(axis=0), prefix=f"delta_items_{name}")
        out[f"delta_items_{name}_n_rows"] = int(mask.sum())
        cell_names.append(name)

    out["delta_marginal_estimand"] = (
        "fitted-row sample-average model-implied treated-versus-untreated contrast"
    )
    out["delta_marginal_effect_source"] = effect_source
    out["delta_standardization_cells"] = ",".join(cell_names)
    out["off_floor"] = bool(off_floor)
    return out


def did_cell_ppc(
    trace: xr.DataTree,
    *,
    phase: np.ndarray,
    G: np.ndarray,
    dose: bool = False,
    node: str = "y_post",
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Posterior-predictive checks for every fitted DiD arm-by-time cell.

    A pooled posterior-predictive plot can hide a cell-specific failure by letting
    well-fitted cells compensate for a badly fitted one. This helper therefore
    compares the observed cell mean and zero rate with their replicated posterior-
    predictive distributions for every wave/arm (binary model) or period/arm (dose
    model). The mean uses the upper-tail probability ``P(rep >= obs)``; the discrete
    zero rate uses a **mid-p** upper tail (``P(rep > obs) + 0.5 P(rep == obs)``) so a
    boundary cell — observed zero-rate exactly 0 or 1, where a plain ``>=`` tail is
    degenerate — is not falsely flagged. These are diagnostics, not hypothesis-test
    p-values. Values near zero or one flag an observed statistic in a predictive tail
    and should be investigated before interpreting contrasts. The ``*_tail_flag``
    columns use fixed 2.5% / 97.5% cutoffs (a 95% two-sided convention) regardless
    of ``ci_prob``, which shapes only the reported interval columns.
    """
    phase_arr = np.asarray(phase)
    group_arr = np.asarray(G)
    if phase_arr.ndim != 1 or group_arr.ndim != 1:
        raise ValueError("phase and G must both be one-dimensional")
    if phase_arr.shape != group_arr.shape:
        raise ValueError(
            f"phase and G must align; got {phase_arr.shape} and {group_arr.shape}"
        )
    if not np.issubdtype(phase_arr.dtype, np.integer):
        raise ValueError(f"phase must contain integer codes, got {phase_arr.dtype}")
    if not set(np.unique(group_arr)).issubset({0, 1}):
        raise ValueError("G must use 0=waitlist and 1=immediate coding")
    if not 0 < ci_prob < 1:
        raise ValueError(f"ci_prob must lie in (0, 1), got {ci_prob}")

    try:
        pp_da = trace.posterior_predictive[node]
        observed = np.asarray(trace.observed_data[node].values).reshape(-1)
    except (AttributeError, KeyError) as exc:
        raise KeyError(
            f"trace must contain posterior_predictive and observed_data for {node!r}"
        ) from exc

    sample_dims = {"chain", "draw"}
    obs_dims = [d for d in pp_da.dims if d not in sample_dims]
    if len(obs_dims) != 1:
        raise ValueError(
            f"{node!r} must have one observation dimension, got {pp_da.dims}"
        )
    replicated = (
        pp_da.stack(sample=("chain", "draw"))
        .transpose(obs_dims[0], "sample")
        .values
    )
    n_obs = phase_arr.shape[0]
    if replicated.shape[0] != n_obs or observed.shape[0] != n_obs:
        raise ValueError(
            f"fitted arrays are misaligned: phase={n_obs}, replicated="
            f"{replicated.shape[0]}, observed={observed.shape[0]}"
        )

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    rows: list[dict[str, float | int | str | bool]] = []
    prefix = "P" if dose else "t"
    for phase_code in sorted(np.unique(phase_arr)):
        for arm_code, arm_name in ((0, "waitlist"), (1, "immediate")):
            mask = (phase_arr == phase_code) & (group_arr == arm_code)
            if not np.any(mask):
                raise ValueError(
                    f"no rows for {prefix}{int(phase_code) + 1}/{arm_name}"
                )
            observed_cell = observed[mask]
            replicated_cell = replicated[mask]
            observed_mean = float(np.mean(observed_cell))
            observed_zero = float(np.mean(observed_cell == 0))
            replicated_mean = replicated_cell.mean(axis=0)
            replicated_zero = (replicated_cell == 0).mean(axis=0)
            p_mean = float(np.mean(replicated_mean >= observed_mean))
            # Mid-p upper tail for the discrete zero-rate statistic: split ties so a
            # boundary cell is not falsely flagged. A plain P(rep >= obs) is
            # necessarily 1.0 when the observed zero-rate is exactly 0 (and small when
            # it is exactly 1), which spuriously flagged well-fitting cells (#390 P2).
            zero_mid_p = float(
                np.mean(replicated_zero > observed_zero)
                + 0.5 * np.mean(replicated_zero == observed_zero)
            )
            rows.append(
                {
                    "cell": f"{prefix}{int(phase_code) + 1}_{arm_name}",
                    "time": f"{prefix}{int(phase_code) + 1}",
                    "phase_code": int(phase_code),
                    "arm": arm_name,
                    "n": int(mask.sum()),
                    "observed_mean": observed_mean,
                    "replicated_mean_median": float(np.median(replicated_mean)),
                    "replicated_mean_lo": float(np.quantile(replicated_mean, lo_q)),
                    "replicated_mean_hi": float(np.quantile(replicated_mean, hi_q)),
                    "p_rep_mean_ge_observed": p_mean,
                    "mean_tail_flag": bool(p_mean <= 0.025 or p_mean >= 0.975),
                    "observed_zero_rate": observed_zero,
                    "replicated_zero_rate_median": float(
                        np.median(replicated_zero)
                    ),
                    "replicated_zero_rate_lo": float(
                        np.quantile(replicated_zero, lo_q)
                    ),
                    "replicated_zero_rate_hi": float(
                        np.quantile(replicated_zero, hi_q)
                    ),
                    "zero_rate_ppc_mid_p": zero_mid_p,
                    "zero_tail_flag": bool(
                        zero_mid_p <= 0.025 or zero_mid_p >= 0.975
                    ),
                }
            )
    return pd.DataFrame(rows)


def did_within_child_ppc(
    trace: xr.DataTree,
    *,
    phase: np.ndarray,
    subject_ids: np.ndarray,
    G: np.ndarray,
    node: str = "y_post",
    ci_prob: float = 0.89,
) -> pd.DataFrame:
    """Posterior-predictive checks of the model's **within-child** structure (#576 MQ3).

    A single stable child random intercept plus conditionally independent
    Beta-Binomial rows imposes a restrictive repeated-measures covariance: it says
    every pair of a child's waves is equicorrelated, with the correlation set by one
    variance ratio, and it fixes how much a child can move between consecutive waves.
    The family's existing checks cannot see a failure of that assumption. The
    arm-by-time cell PPC compares *marginal* cell means and zero rates, which a model
    with badly wrong within-child dependence can still reproduce; the pooled score
    density likewise.

    This cross-checks the structure directly. For every child observed at both waves
    of a pair, it compares the observed **within-child change** (its mean and SD, per
    arm where the pair spans the randomised window) and the observed **across-child
    correlation** of the paired scores with the same statistics recomputed on each
    posterior-predictive replicate. A replicate distribution that systematically
    understates the spread of within-child changes, or overstates the wave-to-wave
    correlation, is the signature of an over-restrictive covariance — invisible in
    the marginal checks.

    Tail probabilities are the usual ``P(replicated >= observed)`` upper tails and the
    flag uses the family's fixed 2.5 % / 97.5 % convention (matching
    :func:`did_cell_ppc`), while the interval columns render at the house ``ci_prob``.
    These are predictive diagnostics, not hypothesis tests.
    """
    posterior_predictive = getattr(trace, "posterior_predictive", None)
    if posterior_predictive is None or node not in posterior_predictive:
        raise KeyError(f"posterior predictive group has no node {node!r}")
    replicated = (
        posterior_predictive[node]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values.astype(float)
    )
    observed = np.asarray(trace.observed_data[node].values, dtype=float)
    phase_arr = np.asarray(phase, dtype=int)
    ids = np.asarray(subject_ids).astype(str)
    arm_arr = np.asarray(G, dtype=int)
    n_obs = phase_arr.shape[0]
    for name, array in (
        ("subject_ids", ids), ("G", arm_arr), ("observed", observed),
        ("replicated", replicated),
    ):
        if array.shape[0] != n_obs:
            raise ValueError(
                f"fitted arrays are misaligned: phase={n_obs}, {name}={array.shape[0]}"
            )

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    rows: list[dict[str, float | int | str | bool]] = []

    def _record(
        statistic: str, pair: str, arm: str, obs_value: float, rep_values: np.ndarray
    ) -> None:
        finite = np.isfinite(rep_values)
        if not np.isfinite(obs_value) or not finite.any():
            return
        rep = rep_values[finite]
        tail = float(np.mean(rep >= obs_value))
        rows.append(
            {
                "statistic": statistic,
                "wave_pair": pair,
                "arm": arm,
                # Filled in by the wave-pair loop below, which knows the pairing.
                "n_children": 0,
                "observed": float(obs_value),
                "replicated_median": float(np.median(rep)),
                "replicated_lo": float(np.quantile(rep, lo_q)),
                "replicated_hi": float(np.quantile(rep, hi_q)),
                "p_rep_ge_observed": tail,
                "tail_flag": bool(tail <= 0.025 or tail >= 0.975),
            }
        )

    def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Across-child Pearson correlation of two row blocks, per draw/column."""
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        a_c = a - a.mean(axis=0, keepdims=True)
        b_c = b - b.mean(axis=0, keepdims=True)
        denominator = np.sqrt((a_c**2).sum(axis=0) * (b_c**2).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denominator > 0, (a_c * b_c).sum(axis=0) / denominator, np.nan)

    wave_pairs = ((0, 1, "t1_t2"), (1, 2, "t2_t3"), (0, 2, "t1_t3"))
    for first, second, pair in wave_pairs:
        first_rows = {i: r for r, i in enumerate(ids) if phase_arr[r] == first}
        second_rows = {i: r for r, i in enumerate(ids) if phase_arr[r] == second}
        common = sorted(set(first_rows) & set(second_rows))
        if len(common) < 3:
            continue
        idx_a = np.asarray([first_rows[i] for i in common])
        idx_b = np.asarray([second_rows[i] for i in common])
        change_obs = observed[idx_b] - observed[idx_a]
        change_rep = replicated[idx_b] - replicated[idx_a]
        arms_here = np.asarray([arm_arr[r] for r in idx_a])
        started = len(rows)
        _record(
            "within_child_change_sd", pair, "both",
            float(np.std(change_obs, ddof=1)), change_rep.std(axis=0, ddof=1),
        )
        _record(
            "across_child_correlation", pair, "both",
            float(_corr(observed[idx_a][:, None], observed[idx_b][:, None])[0]),
            _corr(replicated[idx_a], replicated[idx_b]),
        )
        for arm_code, arm_name in ((0, "waitlist"), (1, "immediate")):
            arm_mask = arms_here == arm_code
            if arm_mask.sum() < 2:
                continue
            _record(
                "within_child_change_mean", pair, arm_name,
                float(np.mean(change_obs[arm_mask])),
                change_rep[arm_mask].mean(axis=0),
            )
        for row in rows[started:]:
            row["n_children"] = (
                int(len(common))
                if row["arm"] == "both"
                else int((arms_here == (1 if row["arm"] == "immediate" else 0)).sum())
            )
    return pd.DataFrame(rows)


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
    lo50, hi50 = band50(d)
    out: dict[str, float | bool | str] = {
        "delta_median": float(np.median(d)),
        "delta_mean": float(np.mean(d)),
        "delta_lo": float(np.quantile(d, lo_q)),
        "delta_hi": float(np.quantile(d, hi_q)),
        "delta_lo50": lo50,
        "delta_hi50": hi50,
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
    out["delta_items_lo50"], out["delta_items_hi50"] = band50(eff)
    return out


def _joint_observed_row_masks(
    trace: xr.DataTree,
    *,
    n_outcomes: int,
    n_obs: int,
) -> np.ndarray:
    """Return the observed-row mask for each flattened joint outcome.

    New traces carry both flattened-cell mappings in ``constant_data``. Older
    traces do not; for those, standardise over every fitted row rather than fail.
    The fallback never mixes outcome counts. It only changes the covariate
    distribution over which an outcome's AME is averaged when that outcome has
    missing post-scores.
    """
    masks = np.ones((n_outcomes, n_obs), dtype=bool)
    constant = getattr(trace, "constant_data", None)
    if constant is None:
        return masks
    if not {"y_post_cell_row", "y_post_cell_outcome"}.issubset(constant):
        return masks
    rows = np.asarray(constant["y_post_cell_row"].values, dtype=int).ravel()
    cols = np.asarray(constant["y_post_cell_outcome"].values, dtype=int).ravel()
    if rows.size != cols.size:
        raise ValueError("joint flattened-cell row and outcome maps differ in length")
    if rows.size and (
        rows.min() < 0
        or rows.max() >= n_obs
        or cols.min() < 0
        or cols.max() >= n_outcomes
    ):
        raise ValueError("joint flattened-cell map contains an out-of-range index")
    masks[:] = False
    masks[cols, rows] = True
    if np.any(masks.sum(axis=1) == 0):
        raise ValueError("joint flattened-cell map leaves an outcome with no observations")
    return masks


def _joint_ame_draws(
    trace: xr.DataTree,
    outcomes: Sequence[str],
    *,
    G: np.ndarray | None = None,
    group: str = "posterior",
    row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return logit coefficients and probability-scale AMEs by outcome and draw.

    Both returned arrays have shape ``(outcome, sample)``. For outcome ``k`` and
    draw ``s`` the average marginal effect is the mean, over rows observed for
    that outcome, of ``expit(eta0 + tau_k) - expit(eta0)``. It is therefore a
    common proportion-correct risk-difference scale even when tests have different
    item denominators. This is the multi-outcome analogue of
    :func:`_itt_ame_draws`. ``row_mask`` optionally restricts the averaging
    population and is intersected with each outcome's observed-row mask.

    **Estimand, when the LKJ residual block is on** (2026-08-22 ITT audit,
    finding 3). ``eta`` as stored already contains the fitted per-child residual
    ``u_i``, and only the treatment term is netted out, so the AME is
    *observed-child, conditional on the fitted residuals* — not a new-child
    population marginal integrating ``u_new ~ MVN(0, Sigma)``. That is the
    intended target: a dependence companion exists to be read beside a parent
    that has no random effect at all, and marginalising would introduce
    attenuation the parent does not have, moving the estimand away from the one
    being compared. Measured on the three registered companions, integrating
    fresh residuals instead changes the medians by less than 0.00012, so the
    choice is about which quantity is named rather than about the number. Fits
    without the block have no ``u_i`` and the distinction does not arise.
    """
    posterior = getattr(trace, group)
    outcome_names = [str(o) for o in outcomes]
    tau_da = posterior["tau"]
    eta_da = posterior["eta"]
    if "outcome" not in tau_da.dims or "outcome" not in eta_da.dims:
        raise ValueError("joint tau and eta must carry a labelled outcome dimension")
    available = [str(o) for o in tau_da.coords["outcome"].values]
    missing = [o for o in outcome_names if o not in available]
    if missing:
        raise KeyError(f"joint outcomes absent from posterior: {missing}")
    outcome_indices = [available.index(outcome) for outcome in outcome_names]
    tau = (
        tau_da.sel(outcome=outcome_names)
        .stack(sample=("chain", "draw"))
        .transpose("outcome", "sample")
        .values
    )
    eta = (
        eta_da.sel(outcome=outcome_names)
        .stack(sample=("chain", "draw"))
        .transpose("outcome", "obs_id", "sample")
        .values
    )
    if G is None:
        constant = getattr(trace, "constant_data", None)
        if constant is None or "G" not in constant:
            raise ValueError("G is required when the trace has no constant_data['G']")
        G = np.asarray(constant["G"].values, dtype=float)
    else:
        G = np.asarray(G, dtype=float)
    if G.ndim != 1 or G.size != eta.shape[1]:
        raise ValueError(f"G must have one entry per fitted row ({eta.shape[1]}), got {G.shape}")
    all_masks = _joint_observed_row_masks(trace, n_outcomes=len(available), n_obs=eta.shape[1])
    masks = all_masks[outcome_indices]
    if row_mask is not None:
        selected = np.asarray(row_mask)
        if selected.ndim != 1:
            raise ValueError(f"row_mask must be 1-D, got a {selected.ndim}-D array.")
        if selected.dtype == bool:
            if selected.shape[0] != eta.shape[1]:
                raise ValueError(
                    f"boolean row_mask has {selected.shape[0]} entries but eta has "
                    f"{eta.shape[1]} observations; pass the fitted-subset mask."
                )
        elif np.issubdtype(selected.dtype, np.integer):
            if selected.size and (
                int(selected.min()) < 0 or int(selected.max()) >= eta.shape[1]
            ):
                raise ValueError(
                    f"integer row_mask has indices outside [0, {eta.shape[1]})."
                )
            selector = np.zeros(eta.shape[1], dtype=bool)
            selector[selected] = True
            selected = selector
        else:
            raise ValueError(
                "row_mask must be a boolean mask or integer index array, "
                f"got dtype {selected.dtype}."
            )
        masks = masks & selected[None, :]
        if np.any(masks.sum(axis=1) == 0):
            raise ValueError("row_mask leaves a joint outcome with no observations")
    ame = np.empty_like(tau, dtype=float)
    for k in range(len(outcome_names)):
        eta0 = eta[k] - tau[k][None, :] * G[:, None]
        contribution = expit(eta0 + tau[k][None, :]) - expit(eta0)
        ame[k] = contribution[masks[k]].mean(axis=0)
    return tau, ame


#: Posterior-to-prior SD ratios at or above this leave the block's correlation
#: indistinguishable from its prior. A convention for reading the table, not a
#: gate threshold: the ratio itself is reported so a reader can judge it.
DEPENDENCE_PRIOR_DOMINATED_RATIO = 0.95


#: Below this the block has genuinely narrowed the parameter.
DEPENDENCE_INFORMED_RATIO = 0.75


def _dependence_verdict(ratio: float | None) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "not assessable"
    if ratio >= DEPENDENCE_PRIOR_DOMINATED_RATIO:
        return "prior-dominated"
    if ratio >= DEPENDENCE_INFORMED_RATIO:
        return "weakly informed"
    return "informed"


def dependence_identification_summary(
    trace: xr.DataTree, *, ci_prob: float
) -> pd.DataFrame | None:
    """How far the LKJ residual-dependence block is informed by the data.

    Returns ``None`` for a fit without the block. One row per free parameter of
    the block — each outcome pair's residual correlation (``u_corr_pair``) and
    each outcome's residual SD (``sigma_outcome``) — comparing the posterior SD
    against the prior SD **measured from this fit's own prior group**.

    Measuring the prior empirically rather than from the LKJ closed form is
    deliberate. The marginal SD of an off-diagonal under ``LKJ(eta)`` is
    ``1 / sqrt(2a + 1)`` with ``a = eta + (d - 2) / 2``
    (:func:`priors.residual_correlation_prior_sd`), and for the registered
    two-outcome companions that closed form and the fitted prior agree to three
    decimals. For ``d > 2`` they do **not**: drawing from this environment's
    ``pm.LKJCorr(n=3, eta=4)`` gives off-diagonal SDs of 0.316, 0.302 and 0.301,
    where a true LKJ is exchangeable and all three must match the closed form's
    0.316. Reading the yardstick off the fit's own ``prior`` group sidesteps that
    entirely and is correct whatever the sampler produces. The closed form is used
    only as a fallback when no prior group was persisted, and the ``prior_source``
    column records which was used.

    Why this exists (2026-08-22 ITT audit, finding 3). The three registered
    companions publish a dependence-corrected interval for a paired contrast, and
    their prose invites the reader to treat that interval as the data's verdict on
    within-child covariance. It is not: at n = 53 the correlation posteriors have
    SD 0.334, 0.337 and 0.334 against a prior SD of 0.334 — ratios of 1.002, 1.008
    and 1.001. The posterior *is* the prior, so the "correction" is the LKJ
    prior's, not the data's. The residual SDs are a different story and the table
    shows it: those posteriors sit well inside their prior, which is why the block
    is reported per parameter rather than with one verdict for the whole thing.
    """
    posterior = getattr(trace, "posterior", None)
    if posterior is None or "u_corr_pair" not in posterior:
        return None
    prior = None
    try:
        groups = {str(g).strip("/") for g in getattr(trace, "groups", ())}
        if "prior" in groups:
            prior = trace["prior"]
    except Exception:  # pragma: no cover - defensive
        prior = None

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    n_outcomes = int(posterior.sizes.get("outcome", 0))
    rows: list[dict] = []

    for name, role, dim in (
        ("u_corr_pair", "residual correlation", "outcome_pair"),
        ("sigma_outcome", "residual SD", "outcome"),
    ):
        if name not in posterior:
            continue
        labels = [str(v) for v in posterior[name].coords[dim].values]
        for index, label in enumerate(labels):
            post = np.asarray(
                posterior[name].isel({dim: index}).values, dtype=float
            ).ravel()
            prior_sd: float | None = None
            prior_source = "unavailable"
            if prior is not None and name in prior:
                draws = np.asarray(
                    prior[name].isel({dim: index}).values, dtype=float
                ).ravel()
                if draws.size > 1:
                    prior_sd = float(draws.std(ddof=1))
                    prior_source = "fitted prior draws"
            if prior_sd is None and name == "u_corr_pair" and n_outcomes >= 2:
                from language_reading_predictors.statistical_models.priors import (
                    residual_correlation_prior_sd,
                )

                prior_sd = residual_correlation_prior_sd(n_outcomes)
                prior_source = "LKJ closed form"
            post_sd = float(post.std(ddof=1))
            ratio = (
                post_sd / prior_sd
                if prior_sd is not None and prior_sd > 0
                else None
            )
            rows.append(
                {
                    "parameter": f"{name}[{label}]",
                    "role": role,
                    "posterior_median": float(np.median(post)),
                    "lo": float(np.quantile(post, lo_q)),
                    "hi": float(np.quantile(post, hi_q)),
                    "posterior_sd": post_sd,
                    "prior_sd": prior_sd,
                    "prior_source": prior_source,
                    "posterior_prior_sd_ratio": ratio,
                    "information_gain": None if ratio is None else 1.0 - ratio,
                    "verdict": _dependence_verdict(ratio),
                    "ci_prob": ci_prob,
                }
            )
    return pd.DataFrame(rows)


def tau_summary_joint(
    trace: xr.DataTree,
    outcomes: list[str],
    ci_prob: float,
    *,
    G: np.ndarray | None = None,
    row_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarise each outcome on probability and logit scales.

    The headline ``ame_prob_*`` columns are average treatment risk differences
    in proportion correct, a common scale across outcome denominators. The
    ``tau_logit_*`` columns retain the conditional model coefficients as secondary
    summaries. Legacy ``tau_*`` aliases remain for existing comparison scripts
    and explicitly refer to the logit coefficient. ``row_mask`` optionally
    restricts every outcome to a common subset of fitted children, after
    intersection with that outcome's observed-score rows.
    """
    draws, ame = _joint_ame_draws(trace, outcomes, G=G, row_mask=row_mask)
    out = []
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    for k, s in enumerate(outcomes):
        d = draws[k]
        a = ame[k]
        a50 = band50(a)
        d50 = band50(d)
        out.append(
            {
                "outcome": s,
                "ame_prob_median": float(np.median(a)),
                "ame_prob_mean": float(np.mean(a)),
                "ame_prob_lo": float(np.quantile(a, lo_q)),
                "ame_prob_hi": float(np.quantile(a, hi_q)),
                "ame_prob_lo50": a50[0],
                "ame_prob_hi50": a50[1],
                "prob_ame_pos": float(np.mean(a > 0)),
                "tau_logit_median": float(np.median(d)),
                "tau_logit_lo": float(np.quantile(d, lo_q)),
                "tau_logit_hi": float(np.quantile(d, hi_q)),
                "tau_median": float(np.median(d)),
                "tau_lo": float(np.quantile(d, lo_q)),
                "tau_hi": float(np.quantile(d, hi_q)),
                "tau_lo50": d50[0],
                "tau_hi50": d50[1],
                "prob_pos": float(np.mean(d > 0)),
            }
        )
    return pd.DataFrame(out)


def joint_treatment_marginals(
    trace: xr.DataTree,
    *,
    outcomes: Sequence[str],
    G: np.ndarray,
    n_trials: Mapping[str, int],
    deltas: Mapping[str, float],
    ci_prob: float = 0.95,
    row_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Items-scale treatment marginals for every outcome in a joint ITT fit.

    The joint model stores ``eta`` on ``(obs_id, outcome)`` and one ``tau`` per
    outcome.  This is the items-scale companion to :func:`tau_summary_joint`: it
    takes that function's probability-scale average marginal effect and multiplies
    by each outcome's item denominator, so the two summaries of a single fit report
    the *same* quantity on two scales.

    **Averaging population (#392):** the AME is computed by
    :func:`_joint_ame_draws`, which averages each outcome over the rows where that
    outcome is *observed* (its flattened-cell mask), not over every fitted row.
    Under outcome-specific post-score missingness the observed populations differ
    per outcome — this function reports each outcome on its own observed population,
    matching :func:`tau_summary_joint`. Passing ``row_mask`` (a boolean/int mask over
    fitted rows) restricts every outcome to a *common* subset, intersected with each
    outcome's observed rows, for a common-population cross-outcome comparison. (On the
    current registered joint datasets every outcome is complete, so the mask is all
    rows and the estimates are unchanged.)

    ``deltas`` contains the project-agreed minimally-important item difference
    where one exists.  Rows without an agreed delta retain the items-scale
    estimate but leave the ROPE fields missing.
    """
    _, ame = _joint_ame_draws(trace, outcomes, G=G, row_mask=row_mask)
    lo_q = (1 - ci_prob) / 2
    rows: list[dict[str, float | str]] = []
    for k, outcome in enumerate(outcomes):
        item_draws = ame[k] * float(n_trials[outcome])
        delta = deltas.get(outcome)
        row: dict[str, float | str] = {
            "outcome": outcome,
            "items_median": float(np.median(item_draws)),
            "items_lo": float(np.quantile(item_draws, lo_q)),
            "items_hi": float(np.quantile(item_draws, 1 - lo_q)),
            "items_lo50": float(np.quantile(item_draws, 0.25)),
            "items_hi50": float(np.quantile(item_draws, 0.75)),
            "prob_pos": float(np.mean(item_draws > 0)),
        }
        if delta is not None:
            d = float(delta)
            row.update(
                {
                    "delta_items": d,
                    "prob_benefit_ge_delta": float(np.mean(item_draws >= d)),
                    "prob_in_rope": float(np.mean(np.abs(item_draws) <= d)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


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
        out[f"{name}_lo50"], out[f"{name}_hi50"] = band50(d)
        out[f"prob_{name}_pos"] = float(np.mean(d > 0))
    return out


def tau_contrast_matrix(
    trace: xr.DataTree,
    outcomes: list[str],
    *,
    G: np.ndarray | None = None,
    scale: str = "probability",
) -> pd.DataFrame:
    """Compute pairwise effect probabilities on the requested scale.

    ``scale='probability'`` (default) compares proportion-correct average
    marginal effects and is the reportable cross-outcome contrast. ``'logit'``
    retains the conditional-coefficient comparison as a secondary diagnostic.
    """
    logit_draws, probability_draws = _joint_ame_draws(trace, outcomes, G=G)
    if scale == "probability":
        draws = probability_draws
    elif scale == "logit":
        draws = logit_draws
    else:
        raise ValueError("scale must be 'probability' or 'logit'")
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
    G: np.ndarray | None = None,
    metadata: dict[str, str] | None = None,
    row_mask: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Summarise an outcome-effect difference on probability and logit scales.

    The headline contrast subtracts per-draw proportion-correct average marginal
    effects, giving a common risk-difference scale despite different test
    denominators. The logit-coefficient difference is retained as secondary.
    Both are computed per draw. For registered factorised models those draws do
    not estimate within-child residual covariance, so a paired contrast requires
    the documented dependence sensitivity.

    Human-readable semantics come from ``metadata`` rather than being inferred
    from symbols. This keeps LRPITT16's expressive-versus-receptive contrast
    distinct from LRPITT15/115's taught-versus-untaught contrasts.

    ``row_mask`` restricts the standardisation population exactly as it does in
    :func:`tau_summary_joint`, and is intersected with each outcome's observed
    rows. The influence audit needs it: per-outcome movement is not sufficient to
    determine contrast movement, because both magnitude *and* posterior covariance
    matter, so the declared contrast has to be recomputed over the retained
    children rather than reconstructed from its marginal components (2026-08-23
    joint audit, finding 9).
    """
    a, b = pair
    draws, ame = _joint_ame_draws(trace, outcomes, G=G, row_mask=row_mask)
    ia, ib = outcomes.index(a), outcomes.index(b)
    diff = draws[ia] - draws[ib]
    diff_prob = ame[ia] - ame[ib]
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    result: dict[str, float | str] = {
        "contrast": f"{a}_minus_{b}",
        "headline_scale": "proportion_correct_risk_difference",
        "diff_prob_median": float(np.median(diff_prob)),
        "diff_prob_mean": float(np.mean(diff_prob)),
        "diff_prob_lo": float(np.quantile(diff_prob, lo_q)),
        "diff_prob_hi": float(np.quantile(diff_prob, hi_q)),
        "diff_prob_lo50": band50(diff_prob)[0],
        "diff_prob_hi50": band50(diff_prob)[1],
        "prob_diff_pos": float(np.mean(diff_prob > 0)),
        "diff_logit_median": float(np.median(diff)),  # median-first (#271)
        "diff_logit_mean": float(np.mean(diff)),
        "diff_logit_lo": float(np.quantile(diff, lo_q)),
        "diff_logit_hi": float(np.quantile(diff, hi_q)),
        "diff_logit_lo50": band50(diff)[0],
        "diff_logit_hi50": band50(diff)[1],
        "prob_diff_logit_pos": float(np.mean(diff > 0)),
    }
    for key in (
        "contrast_kind",
        "contrast_label",
        "positive_interpretation",
        "negative_interpretation",
        "transfer_outcome",
        "transfer_interpretation",
        "dependence_note",
    ):
        if metadata and key in metadata:
            result[key] = str(metadata[key])
    return result


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
    role_overrides: Mapping[str, str] | None = None,
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
    "drives". ``role_overrides`` (term or base name -> role) names the further
    roles the level family carries under its t1-referenced arm-gap
    parameterisation (#552): ``balance`` for the pre-randomisation t1 arm gap,
    ``levels_view`` for the derived per-wave arm gaps, and ``regime`` for the
    t3/t4 arm-gap changes — randomised early-start-versus-delayed-start
    treatment-schedule contrasts, not treated-versus-untreated effects and not
    ordinary adjusted associations (#631 finding 13; the DiD arm_gap_t3 idiom).

    A vector coefficient is expanded to one row per element, labelled by the
    element's coordinate value (``b_grp_time[1]`` for the integer ``phase``
    coordinate, ``d_grp_time[t2]`` for the labelled ``post_phase`` one) — the
    same label ArviZ's summaries and ``psense_summary.csv`` use, so a focal term
    resolves to the same string everywhere.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    overrides = dict(role_overrides or {})

    def _row(term: str, base: str, d: np.ndarray) -> dict[str, object]:
        causal = term in causal_terms or base in causal_terms
        if causal:
            role = "causal"
        else:
            role = overrides.get(term, overrides.get(base, "association"))
        prob_pos = float(np.mean(d > 0))
        lo50, hi50 = band50(d)
        return {
            "term": term,
            "role": role,
            "median": float(np.median(d)),
            "mean": float(np.mean(d)),
            "lo": float(np.quantile(d, lo_q)),
            "hi": float(np.quantile(d, hi_q)),
            "lo50": lo50,
            "hi50": hi50,
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
            labels = (
                [str(v) for v in da.coords[dim].values]
                if dim in da.coords
                else [str(i) for i in range(int(da.sizes[dim]))]
            )
            for i, label in enumerate(labels):
                d = da.isel({dim: i}).stack(sample=("chain", "draw")).values.ravel()
                rows.append(_row(f"{name}[{label}]", name, d))
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
    the Type-M inflation at this n), the fixed 50 / 89 equal-tailed bands
    (:func:`eti_bands`, #177), ``prob_positive`` = ``P(coef > 0)`` and the
    evidence-language fields (:func:`favoured_direction`, #179).

    ``gamma`` (baseline non-verbal ability -> growth *rate*) is the headline Q5
    estimand; ``delta`` is the association with the level at the pooled-mean
    (mid-study) age — ``age_std`` is standardised over all child-wave cells, so
    the entry-level association is ``delta + gamma * E[age_std at t1]``, not
    ``delta``; ``beta`` is the mean slope (trajectory characterisation);
    ``loading`` is the shared growth-tempo loading present only in the factor
    model (LRP70) and skipped otherwise. The interaction model (LRP85) passes
    ``gamma_age``/``gamma_int`` in ``coefs`` so its registered headline reaches
    this summary (2026-08-21 review, finding 1). Every
    row is an **adjusted association** (``role`` fixed to ``"association"``): under
    the locked DAG these non-randomised, latent-GA-confounded terms are never read
    as "drives". ``ci_prob`` is retained for signature parity with
    :func:`factor_summary`; the reported bands are the fixed 50/89 set.
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
            group_dim = next(
                (name for name in ("reading_group", "group") if name in sub.dims),
                None,
            )
            groups: list[object | None] = (
                list(sub.coords[group_dim].values) if group_dim is not None else [None]
            )
            for group in groups:
                cell = (
                    sub.sel({group_dim: group})
                    if group is not None and group_dim is not None
                    else sub
                )
                d = cell.stack(sample=("chain", "draw")).values.ravel()
                prob_pos = float(np.mean(d > 0))
                rows.append(
                    {
                        "coefficient": coef,
                        "outcome": str(lab),
                        "group": group,
                        "role": "association",
                        "median": float(np.median(d)),
                        "prob_positive": prob_pos,
                        "direction_label": evidence_label(prob_pos),
                        **eti_bands(d, probs=(0.5, 0.89)),
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
    score_mean_link: ScoreMeanLink = "logit",
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
    reporting.md). ``prob_trt_pos`` is the probability of direction of the **marginal
    effect** (``P(AME > 0)``); ``prob_trt_logit_pos`` keeps ``P(term > 0)`` as a
    coefficient-scale diagnostic.

    ``row_mask`` (default None = all fitted rows): restrict the observation average to
    a row subset. The gain-factor family passes the **period-1** mask (``phase == 0``)
    so the marginal is averaged only over the genuinely randomised transition, not the
    post-crossover ones that carry no untreated observations (#247 P2). The direction
    probability follows that same marginal effect: with active treatment interactions
    the coefficient and the AME can differ in sign per draw, so ``prob_trt_pos`` is
    ``P(AME > 0)``, not ``P(term > 0)`` (#391) — mirroring ``tau_summary_itt``.

    ``score_mean_link`` is the inverse link of the fitted score model, forwarded to
    the shared core so both counterfactual arms are mapped onto the response scale
    the likelihood actually used. It must be the link the model was *built* with: a
    guessing-floor fit summarised at the default ``"logit"`` would publish an
    ordinary-link items number from a floor-link posterior (#596).
    """
    b, ame_prob = _itt_ame_draws(
        trace,
        G=trt,
        term=term,
        varying_term="",
        eta_name=eta_name,
        moderators=moderators,
        row_mask=row_mask,
        score_mean_link=score_mean_link,
    )
    ame_items = float(n_trials) * ame_prob
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    prob_lo50, prob_hi50 = band50(ame_prob)
    items_lo50, items_hi50 = band50(ame_items)
    # Monte-Carlo precision of the probability-scale AME — a *derived* estimand
    # the convergence gate never sees, so its own ESS/MCSE are reported beside
    # the estimate exactly as ``tau_summary_itt`` does (Kruschke 2021 BARG step
    # 2.C; #575 finding 10c).
    _post = trace.posterior
    _mc = derived_mc_diagnostics(
        ame_prob,
        n_chains=int(_post.sizes["chain"]),
        n_draws=int(_post.sizes["draw"]),
        prefix="trt_prob_",
    )
    return {
        **_mc,
        "trt_prob_median": float(np.median(ame_prob)),
        "trt_prob_lo": float(np.quantile(ame_prob, lo_q)),
        "trt_prob_hi": float(np.quantile(ame_prob, hi_q)),
        "trt_prob_lo50": prob_lo50,
        "trt_prob_hi50": prob_hi50,
        "trt_items_median": float(np.median(ame_items)),
        "trt_items_lo": float(np.quantile(ame_items, lo_q)),
        "trt_items_hi": float(np.quantile(ame_items, hi_q)),
        "trt_items_lo50": items_lo50,
        "trt_items_hi50": items_hi50,
        # Direction of the *marginal effect* (the reported estimand). With active
        # treatment interactions the coefficient ``b`` and the per-draw AME differ in
        # sign, so the probability of direction must summarise ``ame_prob``, not ``b``
        # (#391); the coefficient direction is kept as an explicit diagnostic. Mirrors
        # ``tau_summary_itt``'s ``prob_ame_pos`` / ``prob_tau_logit_pos`` convention.
        "prob_trt_pos": float(np.mean(ame_prob > 0)),
        "prob_trt_logit_pos": float(np.mean(b > 0)),
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
    perturbation_label
        Optional override for the ``scale`` column of the unit perturbation row
        (default ``"+1 SD"``). Used when the term is not a standardised continuous
        covariate — e.g. the gain family's off-floor binary off-floor-at-pre
        indicator, whose ``+1`` perturbation is the at-floor -> off-floor switch
        (#391 finding 2 decision) and would be mislabelled as a ``+1 SD`` shift.
    toggle_vector
        The observed 0/1 indicator vector (aligned with ``eta``'s ``obs_id`` axis)
        for a **binary** covariate entered raw — the off-floor path's
        off-floor-at-pre indicator. When set, the marginal uses the
        net-out-and-toggle idiom of :func:`_itt_ame_draws`: per row the observed
        contribution ``x_i·Δη_i`` is removed (``η0 = η − x_i·Δη_i``, exact because
        the main effect and any interaction product are linear in the indicator)
        and the full 0 -> 1 switch is contrasted at that baseline for every row.
        The default forward shift ``expit(η + Δη) − expit(η)`` would instead
        evaluate an out-of-support 1 -> 2 move on rows whose indicator is already
        1, understating the switch the label promises (gain-factors code review
        2026-08-20, finding 2). ``None`` (default) keeps the forward-shift
        convention, which IS the documented estimand for standardised continuous
        covariates. Incoherent with ``n_items`` (a ``+k items`` increment of a
        0/1 indicator has no meaning), and rejected loudly if both are set.
    """

    label: str
    coef: str
    main_scale: float
    interactions: tuple[tuple[str, np.ndarray], ...] = ()
    n_items: int | None = None
    mean_prop: float | None = None
    sd_items: float | None = None
    perturbation_label: str | None = None
    toggle_vector: np.ndarray | None = None
    #: Per-measure items increment for the bounded-count companion row (#575
    #: finding 3): ``None`` falls back to :func:`association_marginals`'
    #: ``k_items`` argument. A fixed suite-wide ``+5`` was a third of a 6-item
    #: scale and half a 10-item one; per-measure ``max(1, round(n/10))`` matches
    #: the concurrent family's convention.
    k_items: int | None = None


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
    score_mean_link: ScoreMeanLink = "logit",
) -> pd.DataFrame:
    """Per-covariate items-scale association marginals for the gain family (#310).

    The adjusted-association analogue of :func:`treatment_marginal_effect`: for each
    covariate in ``terms`` it forms the per-draw change in the linear predictor from a
    ``+1 SD`` perturbation of that covariate, holding everything else at its observed
    value, and averages the response-scale change ``m(η + Δη) − m(η)`` over
    observations, where ``m`` is the fitted score mean ``score_mean_link ∘ expit``.
    Reported on the probability and items scales (``n_trials`` ×
    probability), with an equal-tailed ``ci_prob`` interval and an inner 50 % band.

    ``score_mean_link`` must be the link the model was **built** with: under the
    phoneme-blending guessing floor the same ``Δη`` maps to a smaller response-scale
    change than under the ordinary logit, so summarising a floor-link posterior at
    the default would overstate every association in items (#596).

    Per draw ``s`` and observation ``i`` the perturbation's linear-predictor shift is

        Δη_{i,s} = γ_{c,s} · (main_scale) + Σ_k γ^{int}_{k,s} · z^{partner}_{k,i},

    i.e. the covariate's main-effect coefficient scaled to the ``+1 SD`` data shift,
    plus each fitted interaction's contribution (the interaction inputs are elementwise
    products of standardised vectors, so a ``+1`` standardised shift moves the product
    by the partner's standardised vector). For a continuous covariate the contrast is
    the **forward shift** from each row's observed ``η`` — the documented estimand.
    A **binary 0/1 indicator** term (``toggle_vector`` set, e.g. the off-floor path's
    off-floor-at-pre indicator) instead uses the treatment marginal's full
    "net out and toggle" idiom (:func:`_itt_ame_draws`): the observed contribution is
    removed per row and the 0 -> 1 switch contrasted at that baseline, since the
    forward shift would evaluate an out-of-support 1 -> 2 move on rows already at 1
    (gain-factors code review 2026-08-20, finding 2).

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
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
    )

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

        if term.toggle_vector is not None and term.n_items:
            raise ValueError(
                f"{term.label!r}: toggle_vector marks a binary 0/1 indicator; a "
                "+k items perturbation is incoherent with it — set one or the other."
            )

        # (scale label, standardised shift Δz). +1 SD is Δz = 1; +k items maps the
        # bounded-count increment to standardised units at the mean operating point.
        # A term may override the unit label (e.g. a 0/1 indicator switch).
        perturbations: list[tuple[str, float]] = [
            (term.perturbation_label or "+1 SD", 1.0)
        ]
        if term.n_items and term.mean_prop is not None and term.main_scale > 0:
            # The fitted baselines are Haldane logits, log((y+0.5)/(n-y+0.5)),
            # whose proportion p* = (y+0.5)/(n+1) is affine in the count — so the
            # mean fitted proportion inverts EXACTLY to the mean baseline count.
            # The former arithmetic added k/N to p* and clipped at 1, which is
            # wrong for this transform and, at the ceiling, silently manufactured
            # a huge logit shift (a "+5 items" row of ~15 logits on the 6-item
            # nonword scale). The correct increment is the Haldane-logit
            # difference of feasible counts, with the increment capped at the
            # items the scale has left — the concurrent family's idiom
            # (#575 finding 3).
            n = float(term.n_items)
            y_mean = float(np.clip(term.mean_prop * (n + 1.0) - 0.5, 0.0, n))
            k_req = int(term.k_items if term.k_items is not None else k_items)
            k_eff = min(k_req, int(np.floor(n - y_mean)))
            if k_eff >= 1:
                dz = float(
                    logit_safe(np.asarray([y_mean + k_eff]), int(n))[0]
                    - logit_safe(np.asarray([y_mean]), int(n))[0]
                ) / term.main_scale
                perturbations.append((f"+{k_eff} items", dz))

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

            de_sel = (
                delta_eta
                if delta_eta.shape[0] == 1
                else (delta_eta if mask is None else delta_eta[mask])
            )
            if term.toggle_vector is not None:
                # Binary-indicator toggle (gain-factors code review 2026-08-20,
                # finding 2): net the observed contribution out per row — exact,
                # because the main effect and any interaction product are linear in
                # the indicator — then contrast the full 0 -> 1 switch at that
                # baseline, mirroring _itt_ame_draws. The forward shift below would
                # evaluate an out-of-support 1 -> 2 move on rows already at 1,
                # understating the switch on the flattened part of the expit curve.
                x = np.asarray(term.toggle_vector, dtype=float)
                if x.shape[0] != n_obs:
                    raise ValueError(
                        f"toggle_vector for {term.label!r} has {x.shape[0]} rows "
                        f"but eta has {n_obs} observations."
                    )
                x_sel = x if mask is None else x[mask]
                eta_base = eta_sel - de_sel * x_sel[:, None]
            else:
                eta_base = eta_sel
            # Map both arms through the fitted score mean before differencing: under
            # a non-identity link the response-scale change is not the logit-scale
            # one rescaled, so differencing raw inverse-logits would report a
            # quantity the likelihood never modelled (#596).
            ame_prob = (
                apply_score_mean_link(expit(eta_base + de_sel), score_mean_link)
                - apply_score_mean_link(expit(eta_base), score_mean_link)
            ).mean(axis=0)  # (S,)
            ame_items = float(n_trials) * ame_prob
            prob_lo50, prob_hi50 = band50(ame_prob)
            items_lo50, items_hi50 = band50(ame_items)
            rows.append(
                {
                    "term": term.label,
                    "role": "association",
                    "scale": scale_label,
                    "prob_median": float(np.median(ame_prob)),
                    "prob_lo": float(np.quantile(ame_prob, lo_q)),
                    "prob_hi": float(np.quantile(ame_prob, hi_q)),
                    "prob_lo50": prob_lo50,
                    "prob_hi50": prob_hi50,
                    "items_median": float(np.median(ame_items)),
                    "items_lo": float(np.quantile(ame_items, lo_q)),
                    "items_hi": float(np.quantile(ame_items, hi_q)),
                    "items_lo50": items_lo50,
                    "items_hi50": items_hi50,
                    "prob_pos": float(np.mean(ame_items > 0)),
                    "off_floor": bool(off_floor),
                    "sd_items": (
                        float(term.sd_items)
                        if term.sd_items is not None
                        else float("nan")
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
        ``Δz = (logit_safe(ȳ + k, N) − logit_safe(ȳ, N)) / sd_logit``
        standardised units, using the fitted Haldane-corrected transformation.
    n_items
        Denominator of the predictor when it is a bounded-count measure; enables the
        ``+k items`` row. ``None`` for age / continuous predictors.
    mean_items
        Mean bounded-count predictor score on the fitted rows — the operating point at
        which the ``+k items`` perturbation is evaluated with the same
        Haldane-corrected logit used to fit the model.
    k_items
        The per-predictor items increment for the ``+k items`` row (the pipeline sets
        it per measure, e.g. ``max(1, round(n_items / 10))``, so a fixed ``+5`` does
        not span 3 %–50 % of scales that differ tenfold — the #310/#325 caveat).
    """

    label: str
    coef: str
    sd_logit: float
    n_items: int | None = None
    mean_items: float | None = None
    k_items: int | None = None


def concurrent_marginals(
    trace: xr.DataTree,
    *,
    terms: Sequence[ConcurrentTerm],
    n_trials: int,
    eta_name: str = "eta",
    ci_prob: float = 0.95,
    group: str = "posterior",
    score_mean_link: ScoreMeanLink = "logit",
) -> pd.DataFrame:
    """Per-predictor items-scale marginals for the concurrent family (#312).

    For each predictor in ``terms`` it forms the per-draw change in the linear
    predictor from a ``+1 SD`` perturbation of that predictor (and, for a
    bounded-count predictor, a ``+k items`` perturbation at the mean operating
    point), holding every other predictor at its observed value, and averages the
    response-scale change ``m(η + Δη) − m(η)`` over the fitted rows, where ``m`` is
    the fitted score mean ``score_mean_link ∘ expit``.
    Reported on the probability and items scales (``n_trials`` = the *focal
    outcome's* denominator × probability), with an equal-tailed ``ci_prob`` interval
    and an inner 50 % band.

    ``score_mean_link`` must be the link the model was **built** with: under the
    phoneme-blending guessing floor the same ``Δη`` maps to a smaller response-scale
    change, so summarising a floor-link posterior at the default would overstate
    every association in items (#619).

    Because the concurrent model has **no interaction terms**, the shift is a scalar
    per draw: ``Δη_s = β_s · Δz`` where ``Δz = 1`` for ``+1 SD`` and
    ``Δz = (logit_safe(ȳ + k, N) − logit_safe(ȳ, N)) / sd_logit`` for ``+k
    items``, where ``logit_safe`` is the Haldane-corrected transformation used in the
    factory. This helper applies equally to adjusted and bivariate traces; callers
    label that fit distinction in the output. Every row carries
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
    rows: list[dict[str, float | str]] = []

    for term in terms:
        beta = posterior[term.coef].stack(sample=("chain", "draw")).values.ravel()  # (S,)

        perturbations: list[tuple[str, float]] = [("+1 SD", 1.0)]
        if (
            term.n_items
            and term.mean_items is not None
            and np.isfinite(term.mean_items)
            and term.k_items
            and term.sd_logit > 0
            and np.isfinite(term.sd_logit)
        ):
            from language_reading_predictors.statistical_models.preprocessing import (
                logit_safe,
            )

            y = float(np.clip(term.mean_items, 0.0, term.n_items))
            # Cap the increment to the largest whole-item shift that reaches no farther
            # than the instrument ceiling. The Haldane correction is finite at both
            # boundaries, so a shift that lands exactly on the ceiling is valid.
            max_k = int(np.floor(term.n_items - y))
            k_eff = min(int(term.k_items), max_k)
            if k_eff >= 1:
                raw = logit_safe(np.asarray([y]), term.n_items)[0]
                raw_k = logit_safe(np.asarray([y + k_eff]), term.n_items)[0]
                dz = (raw_k - raw) / term.sd_logit
                perturbations.append((f"+{k_eff} items", dz))

        for scale_label, dz in perturbations:
            delta_eta = beta * dz  # (S,), scalar shift per draw (no interactions)
            # Map both operating points through the fitted score mean before
            # differencing: under a non-identity link the response-scale change is
            # not the logit-scale one rescaled (#619).
            ame_prob = (
                apply_score_mean_link(
                    expit(eta + delta_eta[None, :]), score_mean_link
                )
                - apply_score_mean_link(expit(eta), score_mean_link)
            ).mean(axis=0)  # (S,)
            ame_items = float(n_trials) * ame_prob
            prob_lo50, prob_hi50 = band50(ame_prob)
            items_lo50, items_hi50 = band50(ame_items)
            rows.append(
                {
                    "term": term.label,
                    "role": "association",
                    "scale": scale_label,
                    "prob_median": float(np.median(ame_prob)),
                    "prob_lo": float(np.quantile(ame_prob, lo_q)),
                    "prob_hi": float(np.quantile(ame_prob, hi_q)),
                    "prob_lo50": prob_lo50,
                    "prob_hi50": prob_hi50,
                    "items_median": float(np.median(ame_items)),
                    "items_lo": float(np.quantile(ame_items, lo_q)),
                    "items_hi": float(np.quantile(ame_items, hi_q)),
                    "items_lo50": items_lo50,
                    "items_hi50": items_hi50,
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
    contrast_index: int | None = None,
    interaction_term: str = "gamma_grp_ability",
    balance_term: str | None = None,
    score_mean_link: ScoreMeanLink = "logit",
    ability: np.ndarray | None = None,
    eta_name: str = "eta",
    group: str = "posterior",
) -> tuple[np.ndarray, np.ndarray]:
    """The t2 randomised contrast and its **arm-free standardised** AME (LRPLF, #127).

    The level model enters group as a per-timepoint vector because the trial is a
    waitlist crossover; **only the t2 element is the randomised
    treated-versus-untreated contrast** (the later timepoints are randomised
    early-start-versus-delayed-start schedule contrasts, #631 finding 13). This
    isolates that one causal effect on the items scale.

    **The estimand** (#584 finding 1, decided 2026-08-23 —
    ``notes/202608231800-level-factors-584-decisions.md``): the average, over the
    fitted t2 rows **each evaluated at its own arm-free profile**, of the effect of
    the randomised t2 change in the adjusted arm gap. Per draw, the *whole* group
    contribution is netted out of every t2 row to recover an arm-free baseline

    ``eta0 = eta - (balance + contrast + gamma_grp_ability*ability) * G``

    and only the focal contrast is added back:
    ``mean_i [ expit(eta0_i + contrast) - expit(eta0_i) ]``. Each row keeps its own
    age, ability main effect, adjusters and fitted child intercept, so the
    standardisation population is the fitted t2 children and the random-effect
    convention is each child's own posterior intercept — not an average child.

    Until #584 the balance term was neither netted out nor added back, so the
    immediate arm's rows were evaluated around ``z + arm_gap_t1`` while the waiting
    arm's were evaluated around ``z``. That is a hybrid over *observed-arm*
    operating points rather than a named estimand. Netting it out costs nothing
    numerically (no stored fit moved by more than 0.04 items, and no direction
    probability moved at all, because ``expit`` is near-linear over a shift that
    small) and it makes the population one the report can state.

    ``contrast_term`` names the posterior vector carrying the randomised t2 element
    and ``contrast_index`` that element's position in it (default: ``t2_phase``,
    the position in a ``phase``-indexed vector). Under the t1-referenced arm-gap
    parameterisation (#552) the caller passes ``contrast_term="d_grp_time"``,
    ``contrast_index=0`` (the ``t2`` entry of the ``post_phase``-indexed change
    vector) and ``balance_term="arm_gap_t1"``. Under the free comparator the focal
    ``b_grp_time[1]`` *is* the whole t2 arm gap, so there is no separate balance
    term to remove: the caller passes ``balance_term=None`` and the default
    ``b_grp_time`` / ``t2_phase`` reproduces the raw t2 gap, unchanged by this
    decision.

    ``gamma_grp_ability`` is a single *time-invariant* coefficient (identified mostly
    from the non-randomised t1/t3/t4 rows), so the moderation increment is held at
    centred ability rather than folded into the causal card — the group×ability
    moderation is reported separately (issue #271 item 5). Because the same focal
    draw is added to every row, the card is a per-draw monotone transform of the
    contrast: ``P(card > 0)`` equals ``P(contrast > 0)``, so the items median, the
    direction probability and the ROPE cannot disagree with the coefficient the
    report flags causal. A marginal response-scale difference-in-differences would
    not have that property (#584 decision 1).

    Returns ``(contrast_draws, ame_prob)`` — the logit-scale focal-contrast draws
    ``(S,)`` (the term flagged causal in the report) and the probability-scale average
    marginal effect per draw ``(S,)``, ready for :func:`rope_card`. ``ability`` is the
    standardised ability covariate aligned with ``eta``'s ``obs_id`` axis (pass
    ``None`` when the model has no group×ability term).
    """
    # ``group`` selects the posterior (the estimate) or the prior (the estimand-scale
    # prior-predictive pushforward, #389 finding 3); both carry eta / contrast /
    # interaction, so the same net-out transform applies to either.
    posterior = getattr(trace, group)
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
    idx = t2_phase if contrast_index is None else int(contrast_index)
    if not 0 <= idx < int(bgt.sizes[extra[0]]):
        raise ValueError(
            f"contrast_index {idx} is outside {contrast_term!r}'s "
            f"{extra[0]} dimension of size {int(bgt.sizes[extra[0]])}"
        )
    contrast_draws = bgt.isel({extra[0]: idx}).stack(sample=("chain", "draw")).values  # (S,)

    # δ_i per t2 row and draw: the WHOLE group contribution — the balance term (when
    # the parameterisation carries one), the focal t2 contrast, and the group×ability
    # slope times each row's ability if the interaction is in the model.
    delta_rows = contrast_draws[None, :]  # (1, S)
    if balance_term is not None:
        if balance_term not in posterior:
            raise ValueError(
                f"balance_term {balance_term!r} is not in the {group} group; pass "
                "the term the plan records (None under the free comparator)."
            )
        balance_draws = (
            posterior[balance_term].stack(sample=("chain", "draw")).values.ravel()
        )  # (S,)
        delta_rows = delta_rows + balance_draws[None, :]
    if interaction_term in posterior and ability is not None:
        g_ab = posterior[interaction_term].stack(sample=("chain", "draw")).values.ravel()  # (S,)
        ab_t2 = np.asarray(ability, dtype=float)[mask]  # (m,)
        delta_rows = delta_rows + np.outer(ab_t2, g_ab)  # (m, S)

    eta_t2 = eta[mask]  # (m, S)
    G_t2 = G[mask]  # (m,)
    # Arm-free baseline for every t2 row (#584 decision 1): remove the complete group
    # contribution, so a waiting-arm row and an immediate-arm row with the same
    # covariates are evaluated at the same operating point. Then add back ONLY the
    # focal contrast — the pre-randomisation balance term is a chance imbalance, not
    # part of the effect, and ``gamma_grp_ability`` is one time-invariant coefficient
    # identified mostly from the non-randomised waves, so the moderation increment is
    # held at centred ability (ability is standardised, so that simply drops it). The
    # interaction is reported separately, never folded into the causal card
    # (issue #271 item 5).
    eta0 = eta_t2 - delta_rows * G_t2[:, None]
    # The marginal is a difference of SCORE MEANS, so it goes through the fit's own
    # score-mean link (#584 decision 2). Under the blending guessing floor the mean
    # is 1/3 + 2/3 * expit(eta), which compresses the same logit contrast into a
    # smaller items difference — reading a floor-link fit through the ordinary expit
    # would publish an effect the model does not imply.
    treated = apply_score_mean_link(expit(eta0 + contrast_draws[None, :]), score_mean_link)
    untreated = apply_score_mean_link(expit(eta0), score_mean_link)
    ame_prob = (treated - untreated).mean(axis=0)  # (S,)
    return contrast_draws, ame_prob


def level_window_comparator_cards(
    output_dir, config: Mapping
) -> list[dict[str, Any]] | None:
    """The four-wave and t1/t2 cards side by side, when both fits are present.

    #584 decision 3 keeps the four-wave levels fit as the model of record and adds a
    randomised-window comparator, "reporting its difference". This finds the
    counterpart fit beside this one — ``lrp-rli-lf-0NN`` <-> ``lrp-rli-lf-2NN``,
    resolved through the id renumber table rather than by string arithmetic — and
    returns both cards, most-restricted last, or ``None`` when the counterpart has
    not been fitted.

    Deliberately **not** a gate. The comparator answers "how much did the
    longitudinal working model move the answer?", and a missing comparator leaves
    that question open rather than making the model of record unpublishable; the
    blending link pair is the case where absence *is* disqualifying, and it has its
    own check. Reads stored cards only.
    """
    from language_reading_predictors import model_ids

    plan = config.get("resolved_run_plan") or {}
    if str(config.get("kind")) != "level_factors" or not plan.get("waves"):
        return None
    model_id = str(config.get("model_id") or "")
    config_name = str(config.get("config_name") or "")
    if not model_id or not config_name:
        return None
    try:
        legacy = model_ids.to_legacy(model_id)
        counterpart_legacy = (
            legacy[:-1] if legacy.endswith("a") else f"{legacy}a"
        )
        counterpart = model_ids.to_canonical(counterpart_legacy, kind="level_factors")
    except Exception:  # noqa: BLE001 - an unmapped id simply has no counterpart
        return None

    def _card(directory, expected_id: str) -> dict[str, Any] | None:
        rope_path = os.path.join(str(directory), "rope_summary.csv")
        config_path = os.path.join(str(directory), "config.json")
        if not (os.path.exists(rope_path) and os.path.exists(config_path)):
            return None
        try:
            with open(config_path, encoding="utf-8") as handle:
                stored = json.load(handle)
            row = pd.read_csv(rope_path).iloc[0]
        except (OSError, ValueError, KeyError, IndexError):
            return None
        if str(stored.get("model_id")) != expected_id:
            return None
        waves = tuple((stored.get("resolved_run_plan") or {}).get("waves") or ())
        return {
            "model_id": expected_id,
            "waves": waves,
            "window": "t1-t2 only" if len(waves) == 2 else "all four waves",
            "items_median": float(row["items_median"]),
            "items_lo": float(row["items_lo"]),
            "items_hi": float(row["items_hi"]),
            "pd": float(row["pd"]),
        }

    here = Path(str(output_dir)).resolve()
    cards = [
        _card(here, model_id),
        _card(here.parent / f"{counterpart}-{config_name}", counterpart),
    ]
    if any(card is None for card in cards):
        return None
    return sorted(cards, key=lambda card: -len(card["waves"]))


def horseshoe_ranking(trace: xr.DataTree, *, delta: float = 0.1) -> pd.DataFrame:
    """Per-predictor ranking from a horseshoe fit (LRPHS, #116 Phase E).

    One row per predictor: ``p_abs_gt_delta`` = posterior ``P(|beta_k| > delta)``
    (the ranking key), the posterior median/mean/sd and 89% HDI (``beta_hdi_lo`` /
    ``beta_hdi_hi``, an actual highest-density interval via :func:`arviz.hdi`, not
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
        median = float(np.median(b))
        hdi = np.asarray(az.hdi(b, prob=0.89))  # 89% highest-density interval
        row = {
            "predictor": name,
            "p_abs_gt_delta": float(np.mean(np.abs(b) > delta)),
            "beta_median": median,
            "beta_mean": mean,
            "beta_sd": float(np.std(b)),
            "beta_hdi_lo": float(hdi[0]),
            "beta_hdi_hi": float(hdi[1]),
            # Direction from the median — the house lead statistic, and the same
            # statistic the key-findings box reads, so the CSV and the box cannot
            # disagree on a spike-and-slab posterior whose mean and median
            # straddle zero (2026-08-21 review, finding 10).
            "sign": "+" if median > 0 else ("-" if median < 0 else "0"),
        }
        if lam is not None:
            row["lambda_mean"] = float(
                lam.isel(predictor=i)
                .stack(sample=("chain", "draw"))
                .values.mean()
            )
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("p_abs_gt_delta", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


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

    One row per (wave, unique off-diagonal domain pair): the posterior median/mean and
    equal-tailed ``ci_prob`` interval (plus an inner 50 % band) of the within-wave latent
    correlation, and ``prob_pos`` = ``P(rho > 0)``. These are model-based latent-domain
    descriptive associations, with indicator-specific residual variation represented
    separately; they are never causal.
    """
    corr, waves, domains = _factor_corr_draws(trace, group)
    D = len(domains)
    lo_q = (1 - ci_prob) / 2
    rows: list[dict] = []
    for w_i, w in enumerate(waves):
        for i in range(D):
            for j in range(i + 1, D):
                d = corr[:, w_i, i, j]
                lo50, hi50 = band50(d)
                rows.append(
                    {
                        "wave": w,
                        "domain_i": domains[i],
                        "domain_j": domains[j],
                        "median": float(np.median(d)),
                        "mean": float(np.mean(d)),
                        "sd": float(np.std(d)),
                        "lo": float(np.quantile(d, lo_q)),
                        "hi": float(np.quantile(d, 1 - lo_q)),
                        "lo50": lo50,
                        "hi50": hi50,
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
    ``beta = R[pred, pred]^-1 R[pred, target]``). This is a latent-factor companion
    to the concurrent family's mutually-adjusted observed-score slopes (#312), not the
    same estimand or a guaranteed correction of it: an **adjusted association**, not a
    causal effect. With two predictors the coefficient is a partial slope; with one it
    coincides with the pairwise correlation.
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
                lo50, hi50 = band50(d)
                rows.append(
                    {
                        "wave": w,
                        "target": domains[a],
                        "predictor": domains[b],
                        "median": float(np.median(d)),
                        "mean": float(np.mean(d)),
                        "sd": float(np.std(d)),
                        "lo": float(np.quantile(d, lo_q)),
                        "hi": float(np.quantile(d, 1 - lo_q)),
                        "lo50": lo50,
                        "hi50": hi50,
                        "prob_pos": float(np.mean(d > 0)),
                    }
                )
    return pd.DataFrame(rows)


def disattenuation_crosscheck(latent_df: pd.DataFrame, observed_df: pd.DataFrame) -> pd.DataFrame:
    """Merge latent factor correlations with observed indicator correlations.

    ``latent_df`` is :func:`longitudinal_factor_correlations` output; ``observed_df``
    carries the raw same-wave observed correlation (``observed_corr``) for each
    ``(wave, domain_i, domain_j)`` — the mean pairwise correlation between the two
    domains' standardised indicators. ``gap`` is ``|latent| - |observed|`` and
    ``latent_ge_observed`` records its direction (with a small numerical tolerance).
    This is a descriptive model check, not an acceptance gate: the latent factor and
    the mean indicator-pair correlation are different estimands, so factor aggregation,
    the loading structure, residual structure and sampling uncertainty can all break a
    simple attenuation ordering even when measurement error is present.
    """
    merged = latent_df.merge(observed_df, on=["wave", "domain_i", "domain_j"], how="left")
    lat = merged["mean"].abs()
    obs = merged["observed_corr"].abs()
    merged["gap"] = lat - obs
    # A small tolerance absorbs Monte-Carlo noise around a zero gap. A missing
    # observed comparator (a wave/pair with too few pairwise-complete indicator
    # pairs, or a merge miss) must stay NA rather than comparing False and being
    # counted as a reversal (2026-08-21 review, finding 10).
    flags = pd.array((lat + 1e-3) >= obs, dtype="boolean")
    flags[obs.isna().to_numpy()] = pd.NA
    merged["latent_ge_observed"] = flags
    return merged


def beta_summary(trace, name: str, ci_prob: float) -> dict:
    """Posterior mean, equal-tailed ``ci_prob``-coverage interval, and P(>0) for ``name``.

    The interval is equal-tailed at ``ci_prob`` coverage, not an HDI — the parameter
    was previously named ``hdi``, which misdescribed it (the callers already pass
    ``ctx.reporting.ci_prob``).
    """
    draws = trace.posterior[name].stack(sample=("chain", "draw")).values
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    return {
        "median": float(np.median(draws)),
        "mean": float(np.mean(draws)),
        "lo": float(np.quantile(draws, lo_q)),
        "hi": float(np.quantile(draws, hi_q)),
        "lo50": float(np.quantile(draws, 0.25)),
        "hi50": float(np.quantile(draws, 0.75)),
        "prob_pos": float(np.mean(draws > 0)),
    }


def coef_row(label: str, draws, hdi_prob: float) -> dict:
    """Posterior mean, equal-tailed central interval and ``P(coef > 0)``.

    Equal-tailed quantiles at coverage ``hdi_prob`` — the same convention as
    :func:`reporting.tau_summary_itt` (not a highest-density interval).
    """
    d = np.asarray(draws).reshape(-1)
    lo_q = (1 - hdi_prob) / 2
    return {
        "coefficient": label,
        "median": float(np.median(d)),
        "mean": float(np.mean(d)),
        "lo": float(np.quantile(d, lo_q)),
        "hi": float(np.quantile(d, 1 - lo_q)),
        "lo50": float(np.quantile(d, 0.25)),
        "hi50": float(np.quantile(d, 0.75)),
        "prob_pos": float(np.mean(d > 0)),
    }
