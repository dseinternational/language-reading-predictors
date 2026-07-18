# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-fit reporting helpers shared across the statistical models."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass

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
from language_reading_predictors.statistical_models.itt import (
    IttRunPlan,
    declared_settings_dict,
    resolve_itt_run_plan,
)
from language_reading_predictors.statistical_models.provenance import run_provenance

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
    row_mask: np.ndarray | None = None,
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
    """
    effect_draws, ame_prob = _itt_ame_draws(
        trace,
        G=G,
        term=term,
        varying_term=varying_term,
        eta_name=eta_name,
        moderators=moderators,
        row_mask=row_mask,
    )
    items = ame_prob * float(n_trials)
    card = rope_card(effect_draws, items, delta=delta, ci_prob=ci_prob)
    # The external rope_card still emits a 90% band (`*_lo90`/`*_hi90`); the suite
    # retired it (2026-07-17 credible-interval standard). Drop it here so the raw
    # rope table matches the median + 50% + 89% convention everywhere it surfaces.
    # rope_card returns a plain dict of scalars.
    def _is90(key: str) -> bool:
        return key.endswith("_lo90") or key.endswith("_hi90")

    if isinstance(card, dict):
        return {k: v for k, v in card.items() if not _is90(k)}
    return card.drop(columns=[c for c in card.columns if _is90(c)])


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
        trace,
        G=G,
        term=term,
        varying_term=varying_term,
        eta_name=eta_name,
        moderators=moderators,
        row_mask=row_mask,
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
        out[f"{name}_lo50"], out[f"{name}_hi50"] = band50(d)
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
    distribution of that fraction under the graded Beta-Binomial model. Returns
    the observed proportion, the predictive mean, both inclusive predictive tails
    and their capped two-sided tail area. The inclusive definitions matter because
    this is a discrete statistic with frequent ties. ``ppc_p_value`` is retained as
    a compatibility alias for the upper tail. The per-draw replicated proportions
    are returned under ``"rep"`` for plotting.
    """
    post = np.asarray(prepared.post_counts[symbol], dtype=float)
    finite = post[np.isfinite(post)]
    obs_p0 = float(np.mean(finite == 0.0)) if finite.size else float("nan")
    pp = trace.posterior_predictive[node]
    yrep = (
        pp.stack(sample=("chain", "draw"))
        .transpose("sample", "obs_id")
        .values
    )  # (S, n_obs)
    rep_p0 = np.mean(yrep == 0.0, axis=1)  # (S,)
    upper_tail = float(np.mean(rep_p0 >= obs_p0))
    lower_tail = float(np.mean(rep_p0 <= obs_p0))
    two_sided_tail = min(1.0, 2.0 * min(upper_tail, lower_tail))
    return {
        "obs_prop_at_zero": obs_p0,
        "ppc_mean_prop_at_zero": float(np.mean(rep_p0)),
        "ppc_upper_tail": upper_tail,
        "ppc_lower_tail": lower_tail,
        "ppc_two_sided_tail": two_sided_tail,
        "ppc_p_value": upper_tail,
        "rep": rep_p0,
    }


# --- Posterior-predictive coverage (issue #318) ------------------------------
# The stock ArviZ overlay asked readers to judge band overlap by eye. These helpers
# turn "does the model fit?" into a decidable coverage number: the share of
# observations whose observed score falls inside the model's central prediction
# intervals. Everything reads the persisted ``posterior_predictive`` + ``observed_data``
# groups (no new sampling) and all sentence logic lives here (not in the .qmd) so the
# reported prose cannot drift from the computed quantity (#271 / #320).
#
# Convention (fixed across families, notes/202607151942-ppc-coverage-redesign.md):
# the interval for observation ``i`` at level ``p`` is the CLOSED interval between the
# ``(1-p)/2`` and ``(1+p)/2`` empirical quantiles of that observation's predictive
# draws; the observed count is inside iff ``q_lo <= y_obs <= q_hi``. These are
# same-children (conditional), in-sample ranges — how well the fitted model
# re-predicts the children it was fit on, not new-child prediction.


def _ppc_node_arrays(
    trace: xr.DataTree, node: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_rep, y_obs)`` for a likelihood ``node`` from the trace.

    ``y_rep`` is ``(n_obs, n_samples)`` (observation dims flattened, chain/draw
    stacked last) and ``y_obs`` is ``(n_obs,)``, taken from ``posterior_predictive``
    and ``observed_data`` respectively. The observed array is transposed into the
    predictive's observation-dim order before flattening so the two stay row-aligned
    for a multi-dim likelihood (e.g. the panel ``y_obs``). Non-finite observed rows
    are *kept* here — callers mask them — so both arrays share one row indexing.
    """
    try:
        pp = trace.posterior_predictive[node]
        obs_da = trace.observed_data[node]
    except (AttributeError, KeyError) as exc:
        raise KeyError(
            f"trace must contain posterior_predictive and observed_data for {node!r}"
        ) from exc
    sample_dims = [d for d in pp.dims if d in ("chain", "draw")]
    obs_dims = [d for d in pp.dims if d not in ("chain", "draw")]
    if not sample_dims or not obs_dims:
        raise ValueError(f"{node!r} predictive has unexpected dims {pp.dims}")
    y_rep = (
        pp.stack(__sample__=sample_dims)
        .transpose(*obs_dims, "__sample__")
        .values.reshape(-1, int(np.prod([pp.sizes[d] for d in sample_dims])))
    )
    y_obs = obs_da.transpose(*obs_dims).values.reshape(-1).astype(float)
    if y_obs.shape[0] != y_rep.shape[0]:
        raise ValueError(
            f"{node!r} observed ({y_obs.shape[0]}) and replicated "
            f"({y_rep.shape[0]}) rows are misaligned"
        )
    return y_rep, y_obs


def ppc_interval_coverage(
    trace: xr.DataTree,
    *,
    node: str = "y_post",
    ci_levels: Sequence[float] = (0.5, 0.9),
) -> pd.DataFrame:
    """Per-observation central prediction-interval coverage for a count outcome.

    For each level ``p`` in ``ci_levels``, computes the share of observations whose
    observed count falls inside the closed central ``p``-interval of that
    observation's posterior-predictive draws (see the module convention above).
    Returns a long-format frame — one row per level — with the uniform coverage
    schema (``mode``/``node``/``unit``/``quantity``/``level``/``level_pct``/
    ``n_total``/``n_inside``/``coverage``) consumed by :func:`ppc_coverage_markdown`.
    """
    y_rep, y_obs = _ppc_node_arrays(trace, node)
    finite = np.isfinite(y_obs)
    y_rep, y_obs = y_rep[finite], y_obs[finite]
    n = int(y_obs.shape[0])
    rows: list[dict[str, object]] = []
    for p in ci_levels:
        lo = np.quantile(y_rep, (1.0 - p) / 2.0, axis=1)
        hi = np.quantile(y_rep, (1.0 + p) / 2.0, axis=1)
        inside = (y_obs >= lo) & (y_obs <= hi)  # closed interval convention
        n_in = int(np.count_nonzero(inside))
        rows.append(
            {
                "mode": "count_interval",
                "node": node,
                "unit": "observations",
                "quantity": "observed score",
                "level": float(p),
                "level_pct": int(round(p * 100)),
                "n_total": n,
                "n_inside": n_in,
                "coverage": float(n_in / n) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def ppc_calibration_table(
    trace: xr.DataTree,
    *,
    node: str = "y_post",
    ci_prob: float = 0.9,
) -> pd.DataFrame:
    """Per-observation observed-vs-predicted table for the calibration panel.

    One row per observation: the observed count, the posterior-predictive median,
    and the closed central ``ci_prob``-interval endpoints (``pp_lo``/``pp_hi``), plus
    an ``inside`` flag. Feeds the ``ppc_calibration.png`` figure and its data CSV.
    """
    y_rep, y_obs = _ppc_node_arrays(trace, node)
    finite = np.isfinite(y_obs)
    y_rep, y_obs = y_rep[finite], y_obs[finite]
    lo_q, hi_q = (1.0 - ci_prob) / 2.0, (1.0 + ci_prob) / 2.0
    lo = np.quantile(y_rep, lo_q, axis=1)
    hi = np.quantile(y_rep, hi_q, axis=1)
    return pd.DataFrame(
        {
            "observed": y_obs,
            "pp_median": np.median(y_rep, axis=1),
            "pp_lo": lo,
            "pp_hi": hi,
            "inside": (y_obs >= lo) & (y_obs <= hi),
        }
    )


def _offfloor_cell_rates(
    trace: xr.DataTree, node: str, group: np.ndarray | None
) -> tuple[list[object], np.ndarray, np.ndarray, np.ndarray]:
    """Observed off-floor rate and per-draw replicated rate for each group cell.

    Returns ``(cell_labels, obs_rate, rep_rate, cell_n)`` where ``obs_rate`` is
    ``(n_cells,)``, ``rep_rate`` is ``(n_cells, n_samples)`` and ``cell_n`` is the
    per-cell observation count. Both the observed and replicated indicators are
    reduced to the 0/1 off-floor event so a raw-count node (defensively) and the
    ``y_offfloor`` Bernoulli node behave identically. A single ``"all"`` cell is used
    when ``group`` is absent or misaligned.
    """
    y_rep, y_obs = _ppc_node_arrays(trace, node)
    finite = np.isfinite(y_obs)
    y_rep, y_obs = y_rep[finite], y_obs[finite]
    y_obs = (y_obs > 0).astype(float)
    y_rep = (y_rep > 0).astype(float)
    n = int(y_obs.shape[0])
    g = None if group is None else np.asarray(group)
    if g is not None and g.shape[0] == finite.shape[0]:
        g = g[finite]  # align group to the finite-observed rows
    if g is None or g.shape[0] != n:
        labels: list[object] = ["all"]
        masks = [np.ones(n, dtype=bool)]
    else:
        labels = list(dict.fromkeys(g.tolist()))  # stable order
        masks = [g == u for u in labels]
    obs_rate = np.array([float(y_obs[m].mean()) for m in masks])
    rep_rate = np.stack([y_rep[m].mean(axis=0) for m in masks])  # (n_cells, S)
    cell_n = np.array([int(np.count_nonzero(m)) for m in masks])
    return labels, obs_rate, rep_rate, cell_n


def ppc_offfloor_cell_table(
    trace: xr.DataTree,
    *,
    node: str = "y_offfloor",
    group: np.ndarray | None = None,
    ci_prob: float = 0.9,
) -> pd.DataFrame:
    """Per-cell observed off-floor rate vs its posterior-predictive rate distribution.

    One row per group cell: the observed rate, the replicated-rate median and closed
    central ``ci_prob``-interval, an ``inside`` flag, and cell ``n``. Feeds the
    floor-rule PPC figure and its data CSV.
    """
    labels, obs_rate, rep_rate, cell_n = _offfloor_cell_rates(trace, node, group)
    lo_q, hi_q = (1.0 - ci_prob) / 2.0, (1.0 + ci_prob) / 2.0
    lo = np.quantile(rep_rate, lo_q, axis=1)
    hi = np.quantile(rep_rate, hi_q, axis=1)
    return pd.DataFrame(
        {
            "cell": [str(lbl) for lbl in labels],
            "n": cell_n,
            "observed_rate": obs_rate,
            "pp_rate_median": np.median(rep_rate, axis=1),
            "pp_rate_lo": lo,
            "pp_rate_hi": hi,
            "inside": (obs_rate >= lo) & (obs_rate <= hi),
        }
    )


def ppc_offfloor_rate_coverage(
    trace: xr.DataTree,
    *,
    node: str = "y_offfloor",
    group: np.ndarray | None = None,
    ci_levels: Sequence[float] = (0.5, 0.9),
) -> pd.DataFrame:
    """Group-cell off-floor RATE coverage for a floor-rule / binary outcome.

    Per-observation interval coverage of a 0/1 indicator is degenerate, so the
    floor-rule check asks instead whether the model reproduces the observed off-floor
    *rate*: for each group cell (arm × wave where available, else one overall cell)
    and each level ``p``, the cell is covered iff its observed rate falls in the
    closed central ``p``-interval of the replicated-rate distribution. Returns the
    same long-format schema as :func:`ppc_interval_coverage` (``unit`` = "group
    cells", ``quantity`` = "observed off-floor rate", ``mode`` = "offfloor_rate").
    """
    labels, obs_rate, rep_rate, _cell_n = _offfloor_cell_rates(trace, node, group)
    n_cells = len(labels)
    rows: list[dict[str, object]] = []
    for p in ci_levels:
        lo = np.quantile(rep_rate, (1.0 - p) / 2.0, axis=1)
        hi = np.quantile(rep_rate, (1.0 + p) / 2.0, axis=1)
        inside = (obs_rate >= lo) & (obs_rate <= hi)  # closed interval convention
        n_in = int(np.count_nonzero(inside))
        rows.append(
            {
                "mode": "offfloor_rate",
                "node": node,
                "unit": "group cells" if n_cells > 1 else "off-floor rate",
                "quantity": "observed off-floor rate",
                "level": float(p),
                "level_pct": int(round(p * 100)),
                "n_total": n_cells,
                "n_inside": n_in,
                "coverage": float(n_in / n_cells) if n_cells else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def ppc_coverage_markdown(cov: pd.DataFrame) -> str:
    """Render the posterior-predictive coverage sentence + verdict as report markdown.

    Consumes the uniform long-format frame produced by :func:`ppc_interval_coverage`
    or :func:`ppc_offfloor_rate_coverage` (``ppc_summary.csv``). All numbers and the
    plain-language verdict are derived from the frame — nothing is hand-stated in the
    report (#271). Returns an empty string for an empty/None frame, or one carrying no
    usable level (``n_total == 0`` or non-finite coverage — a degenerate fit whose
    coverage would render as ``nan%``), so the partial can ``print`` it unconditionally.
    """
    if cov is None or len(cov) == 0:
        return ""
    # Drop degenerate rows (no observations / non-finite coverage) so a NaN never
    # propagates into the rendered sentence (review: coverage is NaN when n_total==0).
    usable = cov[
        (cov["n_total"].astype(float) > 0) & np.isfinite(cov["coverage"].astype(float))
    ]
    if usable.empty:
        return ""
    d = usable.drop_duplicates("level_pct").set_index("level_pct")
    unit = str(usable["unit"].iloc[0])
    quantity = str(usable["quantity"].iloc[0])
    clauses: list[str] = []
    if 90 in d.index:
        r90 = d.loc[90]
        clauses.append(
            f"the model's 90% prediction ranges contained the {quantity} for "
            f"**{int(r90['n_inside'])} of {int(r90['n_total'])}** {unit} "
            f"({r90['coverage'] * 100:.0f}%, expected ≈ 90%)"
        )
    if 50 in d.index:
        r50 = d.loc[50]
        clauses.append(
            f"the 50% ranges contained **{int(r50['n_inside'])} of "
            f"{int(r50['n_total'])}** ({r50['coverage'] * 100:.0f}%, expected ≈ 50%)"
        )
    if not clauses:
        return ""
    # Verdict, derived from the 90% coverage (or the 50% if only that is present).
    # Two-sided: coverage can miss nominal by being too LOW (ranges too narrow, the
    # model over-confident) or too HIGH (ranges wider than the data need, the model
    # under-confident / the check under-powered at this n) — flag both (review).
    cov90 = float(d.loc[90, "coverage"]) if 90 in d.index else None
    ref = cov90 if cov90 is not None else float(d.loc[50, "coverage"])
    target = 0.90 if cov90 is not None else 0.50
    if abs(ref - target) <= 0.05:
        verdict = (
            "This is close to the nominal level: the fitted model reproduces the "
            "spread of these children's scores."
        )
    elif ref > target + 0.05:
        verdict = (
            "This is above the nominal level, so the model's prediction ranges are "
            "wider than the data need (mildly under-confident, or the check is "
            "under-powered at this sample size) rather than too narrow."
        )
    elif ref >= target - 0.15:
        verdict = (
            "This is a little below the nominal level, so the model is mildly "
            "over-confident (its prediction ranges are slightly too narrow) for some "
            "observations."
        )
    else:
        verdict = (
            "This is well below the nominal level, so the model's prediction ranges "
            "are too narrow — treat the fit with caution."
        )
    return (
        "**Coverage.** " + "; ".join(clauses) + ". " + verdict
        + " (These are same-children, in-sample ranges — how well the fitted model "
        "re-predicts the children it was fit on, not new-child prediction.)"
    )


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
        slope_below = np.nanmean(np.where(below[:, None], slope, np.nan), axis=0)
        slope_above = np.nanmean(np.where(~below[:, None], slope, np.nan), axis=0)
    else:  # no increasing draws — the below/above split is undefined
        slope_below = slope_above = np.full(f.shape[1], np.nan)

    def _med(a: np.ndarray) -> float:
        a = a[np.isfinite(a) & increasing]
        return float(np.median(a)) if a.size else float("nan")

    result = {
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

    For a continuous-covariate exposure (``mechanism_is_covariate`` with the HSGP curve
    on, e.g. ``lrp-rli-mech-191`` sessions -> word reading), pass ``exposure_values``
    (the per-observation raw exposure, in the same order as ``f_mech``'s rows) instead
    of ``n_trials``; the knee/half-rise/``obs_count_*`` fields are then in the
    exposure's own raw units (e.g. sessions) rather than a bounded count.
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
    n_chains, n_draws = int(post.sizes["chain"]), int(post.sizes["draw"])
    if exposure_values is not None:
        # Continuous-covariate exposure: the knee lives in the exposure's own units.
        # ``exposure_values`` must be in the same observation order as ``f_mech``'s rows.
        return _readiness_knee(
            f, None,
            count_values=np.asarray(exposure_values, dtype=float).reshape(-1),
            ci_prob=ci_prob, n_bins=n_bins, n_chains=n_chains, n_draws=n_draws,
        )
    ell = np.asarray(trace.constant_data["mech_post_logit"].values).reshape(-1)  # (n_obs,)
    return _readiness_knee(
        f, ell, n_trials=n_trials, ci_prob=ci_prob, n_bins=n_bins,
        n_chains=n_chains, n_draws=n_draws,
    )


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
) -> dict[str, float | bool | str]:
    """Summarise a waitlist-crossover arm-by-wave model (kind="did").

    The current binary model exposes three immediate-minus-waitlist logit contrasts:
    ``arm_gap_t1`` (pre-randomisation balance), ``tau_t2`` (the clean randomised t2
    arm contrast) and ``arm_gap_t3`` (a post-crossover association). Its derived
    ``delta_crossover = tau_t2 - arm_gap_t3`` is the reduction in the arm gap after
    crossover: positive means that the waitlist arm caught up. It is descriptive,
    not a second randomised treatment effect.

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
    fitted child-specific catch-up terms.

    The legacy ``beta_period``/``delta`` branch remains readable so existing traces
    fail gracefully during the refit transition. Its ``delta_items_*`` quantity is
    a fitted-row model-implied treated-versus-untreated toggle, not a four-cell DiD
    cross-difference and not automatically comparable with the randomised ITT.

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
            gap = (
                posterior[term_name]
                .stack(sample=("chain", "draw"))
                .values.ravel()
            )
            waitlist = expit(eta_base[rows]).mean(axis=0)
            immediate = expit(eta_base[rows] + gap[None, :]).mean(axis=0)
            arm_gap = immediate - waitlist
            _effect_summary(waitlist, prefix=f"{wave_name}_waitlist_items")
            _effect_summary(immediate, prefix=f"{wave_name}_immediate_items")
            _effect_summary(arm_gap, prefix=f"{term_name}_items")
            out[f"{term_name}_items_n_rows"] = int(rows.sum())
            wave_effects[term_name] = arm_gap

        if "delta_crossover_i" not in posterior:
            _effect_summary(
                wave_effects["tau_t2"] - wave_effects["arm_gap_t3"],
                prefix="delta_crossover_items",
            )
            out["delta_crossover_items_available"] = True
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
        out["delta_crossover_interpretation"] = (
            "t2 arm gap minus t3 arm gap; post-crossover association"
        )
        out["off_floor"] = bool(off_floor)
        return out

    out.update(_summ("beta_period"))
    if dose:
        # The redesigned dose model separates treatment presence from intensive
        # session variation. Report the arm and treatment-presence coefficients
        # whenever the trace carries them so the observational beta_dose is not
        # presented as though it were the randomized on/off contrast.
        for name in ("beta_group", "theta_treated", "gamma_t1", "beta_dose"):
            if name in posterior:
                out.update(_summ(name))
        out["dose_interpretation"] = (
            "beta_dose is an observational intensive-margin association; "
            "theta_treated is the model's treatment-presence term"
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
    model). The upper-tail probabilities are diagnostics, not hypothesis-test
    p-values. Values near zero or one flag an observed statistic in a predictive
    tail and should be investigated before interpreting contrasts.
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
            p_zero = float(np.mean(replicated_zero >= observed_zero))
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
                    "p_rep_zero_ge_observed": p_zero,
                    "zero_tail_flag": bool(p_zero <= 0.025 or p_zero >= 0.975),
                }
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
) -> pd.DataFrame:
    """Items-scale treatment marginals for every outcome in a joint ITT fit.

    The joint model stores ``eta`` on ``(obs_id, outcome)`` and one ``tau`` per
    outcome.  For each posterior draw and outcome this removes the fitted group
    contribution to recover the untreated linear predictor, toggles treatment
    on, averages the probability difference over fitted rows, and multiplies by
    that outcome's item denominator.  This is the joint analogue of
    :func:`treatment_marginal_effect`; keeping it as a fitted artefact lets the
    key-findings builder report the project-agreed range-plus-count headline
    without approximating item effects from logit coefficients.

    ``deltas`` contains the project-agreed minimally-important item difference
    where one exists.  Rows without an agreed delta retain the items-scale
    estimate but leave the ROPE fields missing.
    """
    posterior = trace.posterior
    tau = (
        posterior["tau"]
        .stack(sample=("chain", "draw"))
        .transpose("outcome", "sample")
    )
    eta = (
        posterior["eta"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "outcome", "sample")
    )
    groups = np.asarray(G, dtype=float)
    if groups.shape[0] != eta.sizes["obs_id"]:
        raise ValueError(
            f"G has {groups.shape[0]} rows but eta has {eta.sizes['obs_id']} "
            "observations"
        )

    lo_q = (1 - ci_prob) / 2
    rows: list[dict[str, float | str]] = []
    for outcome in outcomes:
        effect = np.asarray(tau.sel(outcome=outcome).values).reshape(-1)
        eta_k = np.asarray(eta.sel(outcome=outcome).values)
        eta_zero = eta_k - groups[:, None] * effect[None, :]
        item_draws = (
            expit(eta_zero + effect[None, :]) - expit(eta_zero)
        ).mean(axis=0) * float(n_trials[outcome])
        delta = deltas.get(outcome)
        row: dict[str, float | str] = {
            "outcome": outcome,
            "items_median": float(np.median(item_draws)),
            "items_lo": float(np.quantile(item_draws, lo_q)),
            "items_hi": float(np.quantile(item_draws, 1 - lo_q)),
            "items_lo50": float(np.quantile(item_draws, 0.25)),
            "items_hi50": float(np.quantile(item_draws, 0.75)),
            "prob_pos": float(np.mean(effect > 0)),
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
    """
    a, b = pair
    draws, ame = _joint_ame_draws(trace, outcomes, G=G)
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


def _json_safe(value):
    """Return a reconstructable JSON representation of model settings.

    ``ModelSpec.extra`` is intentionally free-form. Most registered settings are
    primitives, tuples or mappings, but a few families use NumPy scalars,
    dataclasses or callables. Serialising those with ``default=str`` alone loses
    structure and can make an old fit impossible to reconstruct.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
        return f"{module}.{name}" if module else name
    return str(value)


def _effective_model_settings(context: StatisticalFitContext) -> dict:
    """Resolve the spec and prepared-data choices that actually reached a fit."""

    spec = context.spec
    prepared = context.prepared
    spec_extra = _json_safe(spec.extra)
    settings = dict(spec_extra)

    if spec.kind == "itt":
        plan = _itt_run_plan(context)
        settings = plan.as_dict()
        if plan.floor_rule:
            likelihood = "bernoulli_offfloor_exploratory_with_beta_binomial_secondaries"
        else:
            likelihood = plan.headline_likelihood
        settings.update(
            {
                "likelihood": likelihood,
                "floor_rule": plan.floor_rule,
                "outcomes": list(plan.outcomes),
                "baseline_terms": {
                    "use_own_baseline": plan.use_own_baseline,
                    "use_own_baseline_gp": plan.use_own_baseline_gp,
                    "cross_symbols": list(plan.cross_symbols),
                    "pre_required": _json_safe(plan.pre_required),
                },
                "age_effect": plan.age_effect,
                "use_age_gp": plan.use_age_gp,
                "use_age_linear": plan.use_age_linear,
                "use_residual_correlation": False,
            }
        )
    elif spec.kind == "joint":
        outcomes = list(spec_extra.get("outcomes") or getattr(prepared, "pre_logit", {}))
        use_cross_baselines = bool(spec_extra.get("use_cross_baselines", True))
        use_age_gp = bool(spec_extra.get("use_age_gp", False))
        use_age_linear = bool(spec_extra.get("use_age_linear", False))
        settings.update(
            {
                "likelihood": "beta_binomial",
                "floor_rule": False,
                "outcomes": outcomes,
                "baseline_terms": {
                    "use_own_baseline": True,
                    "use_cross_baselines": use_cross_baselines,
                    "cross_symbols": outcomes if use_cross_baselines else [],
                },
                "age_effect": (
                    "gp" if use_age_gp else "linear" if use_age_linear else "none"
                ),
                "use_age_gp": use_age_gp,
                "partial_pool_age_gp": bool(spec_extra.get("partial_pool_age_gp", True)),
                "use_age_linear": use_age_linear,
                "use_residual_correlation": bool(spec_extra.get("use_residual_correlation", False)),
            }
        )

    post_counts = getattr(prepared, "post_counts", {}) if prepared is not None else {}
    covariates = getattr(prepared, "covariates", {}) if prepared is not None else {}
    effective_adjustment = list(covariates)
    if spec.kind == "itt":
        effective_adjustment = [
            name for name in plan.adjust_for if name in covariates
        ]
    settings.update(
        {
            "prepared_outcomes": list(post_counts),
            "effective_adjustment": effective_adjustment,
            "prepared_covariates": list(covariates),
            "covariate_time": _json_safe(
                getattr(prepared, "covariate_time", {})
                if prepared is not None
                else {}
            ),
            "dropped_covariates": list(
                getattr(prepared, "dropped_covariates", ())
                if prepared is not None
                else ()
            ),
            "phase_mode": getattr(prepared, "phase_mode", None),
        }
    )
    return settings


def _itt_analysis_set_metadata(context: StatisticalFitContext) -> dict:
    """Return arm-specific analysis-set counts for an ITT-family fit."""

    if context.spec.kind not in {"itt", "joint"}:
        return {}
    prepared = context.prepared
    if prepared is None or not hasattr(prepared, "G") or not hasattr(prepared, "post_counts"):
        return {}

    from language_reading_predictors.statistical_models.itt_audit import (
        analysis_set_table,
    )

    if context.spec.kind == "itt":
        symbol = context.spec.outcome_symbol
        return {
            "analysis_set_by_arm": _json_safe(
                analysis_set_table(prepared, outcome_symbol=symbol).to_dict(orient="records")
            )
        }

    records = []
    outcomes = tuple(context.spec.extra.get("outcomes") or prepared.post_counts)
    for symbol in outcomes:
        table = analysis_set_table(prepared, outcome_symbol=symbol)
        table.insert(0, "outcome", symbol)
        records.extend(table.to_dict(orient="records"))
    return {"analysis_set_by_outcome_and_arm": _json_safe(records)}


def _itt_run_plan(context: StatisticalFitContext) -> IttRunPlan:
    """Return the plan resolved before loading, or reconstruct it for old callers."""
    resolved_plan = getattr(context, "resolved_plan", None)
    if isinstance(resolved_plan, IttRunPlan):
        return resolved_plan
    return resolve_itt_run_plan(context.spec)


def write_model_recipe(context: StatisticalFitContext) -> str | None:
    """Write the human-readable recipe generated from a typed ITT run plan."""
    if context.spec.kind != "itt":
        return None
    plan = _itt_run_plan(context)
    os.makedirs(context.output_dir, exist_ok=True)
    path = os.path.join(context.output_dir, "model_recipe.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(plan.recipe_markdown(title=context.spec.title))
    return path


def write_run_metadata(context: StatisticalFitContext, extra: dict | None = None) -> None:
    """Persist a reconstructable ``config.json`` and basic report metrics."""
    out = context.output_dir
    os.makedirs(out, exist_ok=True)
    spec = context.spec
    recipe_path = write_model_recipe(context)
    resolved_plan = (
        _itt_run_plan(context).as_dict() if spec.kind == "itt" else None
    )
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
        # Preserve both what the module requested and what preprocessing/factory
        # resolution actually used. This is deliberately separate from ``extra``
        # below, which contains post-fit summaries supplied by the pipeline.
        "spec_extra": _json_safe(spec.extra),
        "model_settings": (
            _json_safe(declared_settings_dict(spec))
            if spec.kind == "itt" or spec.model_settings is not None
            else None
        ),
        "resolved_run_plan": _json_safe(resolved_plan),
        "model_recipe_file": os.path.basename(recipe_path) if recipe_path else None,
        "effective_model_settings": _effective_model_settings(context),
        "n_obs": context.prepared.n_obs if context.prepared else None,
        "n_children": context.prepared.n_children if context.prepared else None,
        "n_phases": context.prepared.n_phases if context.prepared else None,
        "n_waves": getattr(context.prepared, "n_waves", None) if context.prepared else None,
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
        "data_path": getattr(context.prepared, "data_path", None),
        "data_sha256": getattr(context.prepared, "data_sha256", None),
        "provenance": run_provenance(),
        "extra": _json_safe(extra or {}),
        **_itt_analysis_set_metadata(context),
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


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
        lo50, hi50 = band50(d)
        return {
            "term": term,
            "role": "causal" if causal else "association",
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
    the Type-M inflation at this n), the fixed 50 / 89 equal-tailed bands
    (:func:`eti_bands`, #177), ``prob_positive`` = ``P(coef > 0)`` and the
    evidence-language fields (:func:`favoured_direction`, #179).

    ``gamma`` (baseline non-verbal ability -> growth *rate*) is the headline Q5
    estimand; ``delta`` is the effect on baseline *level*; ``beta`` is the mean
    slope (trajectory characterisation); ``loading`` is the shared growth-tempo
    loading present only in the factor model (LRP70) and skipped otherwise. Every
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
        trace,
        G=trt,
        term=term,
        varying_term="",
        eta_name=eta_name,
        moderators=moderators,
        row_mask=row_mask,
    )
    ame_items = float(n_trials) * ame_prob
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    prob_lo50, prob_hi50 = band50(ame_prob)
    items_lo50, items_hi50 = band50(ame_items)
    return {
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
    probability), with an equal-tailed ``ci_prob`` interval and an inner 50 % band.

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

            de_sel = (
                delta_eta
                if delta_eta.shape[0] == 1
                else (delta_eta if mask is None else delta_eta[mask])
            )
            ame_prob = (expit(eta_sel + de_sel) - expit(eta_sel)).mean(axis=0)  # (S,)
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
) -> pd.DataFrame:
    """Per-predictor items-scale marginals for the concurrent family (#312).

    For each predictor in ``terms`` it forms the per-draw change in the linear
    predictor from a ``+1 SD`` perturbation of that predictor (and, for a
    bounded-count predictor, a ``+k items`` perturbation at the mean operating
    point), holding every other predictor at its observed value, and averages the
    probability-scale change ``expit(η + Δη) − expit(η)`` over the fitted rows.
    Reported on the probability and items scales (``n_trials`` = the *focal
    outcome's* denominator × probability), with an equal-tailed ``ci_prob`` interval
    and an inner 50 % band.

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
            ame_prob = (expit(eta + delta_eta[None, :]) - expit(eta)).mean(axis=0)  # (S,)
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
    contrast_draws = bgt.isel({extra[0]: t2_phase}).stack(sample=("chain", "draw")).values  # (S,)

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
            "sign": "+" if mean > 0 else ("-" if mean < 0 else "0"),
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
    # A small tolerance absorbs Monte-Carlo noise around a zero gap.
    merged["latent_ge_observed"] = (lat + 1e-3) >= obs
    return merged


# --- Key-findings generation (issue #320) -------------------------------------
# A fit-time distillation of each report's headline result into 3-5 template
# sentences a science undergraduate can read before any machinery. The generator
# reads ONLY the fit's own artefacts (family CSVs + diagnostics_summary.json +
# config.json) and writes ``key_findings.json``; the report partial
# ``_partials/_key_findings.qmd`` is a dumb renderer of that file. Generating at
# fit time (not render time) keeps the sentences tied to the fit that produced
# the numbers (#271); wording fixes and backfill go through
# ``scripts/regenerate_key_findings.py`` — no refit needed.

KEY_FINDINGS_FILENAME = "key_findings.json"
KEY_FINDINGS_SCHEMA_VERSION = 1
KEY_FINDINGS_MAX_SENTENCES = 5

# Plain-language labels for the factor-model coefficients (the family-highlight
# sentence). Terms not listed here are skipped rather than surfaced raw — a
# key-findings box must never ask the reader to decode a coefficient name.
_KF_FACTOR_LABELS: dict[str, str] = {
    "gamma_own": "the child's own starting point on this measure",
    "gamma_A": "the child's age",
    "gamma_ability": "general cognitive ability (block design)",
    "gamma_R": "receptive vocabulary at the start of the period",
    "gamma_E": "expressive vocabulary at the start of the period",
    "gamma_L": "letter-sound knowledge at the start of the period",
    "gamma_W": "word reading at the start of the period",
    "gamma_B": "sound blending at the start of the period",
    "gamma_hs": "hearing",
    "gamma_deapp_c": "speech accuracy",
    "gamma_erbto": "phonological memory (nonword repetition)",
}


class _KeyFindingsUnavailable(Exception):
    """Raised by a builder when the CSVs it needs are missing or unusable."""


def _kf_float(value) -> float:
    """Return ``value`` as a finite float, else raise (the no-``nan`` guard)."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise _KeyFindingsUnavailable(f"non-numeric value {value!r}") from exc
    if not np.isfinite(v):
        raise _KeyFindingsUnavailable(f"non-finite value {value!r}")
    return v


def _kf_pct(prob) -> str:
    """A probability as a plain percentage string, never rounding to a false
    certainty (``0.998`` renders as ``99.8``, not ``100``)."""
    p = _kf_float(prob)
    if not 0.0 <= p <= 1.0:
        raise _KeyFindingsUnavailable(f"probability out of range: {p!r}")
    v = 100.0 * p
    # Never display a false certainty: an empirical posterior probability of 1
    # (or 0) just means every retained draw agreed, so cap the display at 99.9
    # (or floor it at 0.1) rather than claiming 100% / 0%.
    if round(v) >= 100:
        return f"{min(v, 99.9):.1f}"
    if round(v) <= 0:
        return f"{max(v, 0.1):.1f}"
    return f"{v:.0f}"


def _kf_sentence(text: str, kind: str) -> dict[str, str]:
    return {"text": text, "kind": kind}


def _kf_csv_row(output_dir, name: str) -> dict | None:
    """First row of ``{output_dir}/{name}`` as a plain dict, or None if absent."""
    path = os.path.join(str(output_dir), name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def _kf_csv(output_dir, name: str) -> pd.DataFrame | None:
    """Read one fit CSV, returning ``None`` when it is absent or empty."""
    path = os.path.join(str(output_dir), name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def _kf_most_resolved_row(
    df: pd.DataFrame,
    *,
    prob_col: str,
) -> dict:
    """Return the row whose direction is clearest, never the largest estimate.

    The ranking is distance of ``P(positive)`` from 0.5.  This avoids presenting
    differently-scaled coefficients as though their raw magnitudes were
    comparable, and it keeps the selection rule tied to uncertainty.
    """
    if prob_col not in df.columns:
        raise _KeyFindingsUnavailable(f"{prob_col} is missing")
    probabilities = pd.to_numeric(df[prob_col], errors="coerce")
    usable = df[np.isfinite(probabilities)].copy()
    if usable.empty:
        raise _KeyFindingsUnavailable(f"{prob_col} has no finite values")
    usable["_kf_resolution"] = (
        pd.to_numeric(usable[prob_col], errors="coerce") - 0.5
    ).abs()
    return usable.sort_values("_kf_resolution", ascending=False).iloc[0].to_dict()


def _kf_plain_label(value) -> str:
    """Make an artefact identifier readable without inventing a construct name."""
    return str(value).replace("_", " ").strip()


def _kf_measure_label(symbol) -> str:
    """Display label for a registered measure symbol, else the symbol itself."""
    from language_reading_predictors.statistical_models.measures import MEASURES

    measure = MEASURES.get(str(symbol))
    return measure.label if measure is not None else _kf_plain_label(symbol)


def _kf_association_direction(
    prob_pos,
    *,
    positive_claim: str,
    negative_claim: str,
) -> str:
    """Harm-aware direction/strength sentence for a non-causal quantity."""
    p = _kf_float(prob_pos)
    fav = favoured_direction(p)
    positive = fav["favoured_direction"] == "positive"
    sign = "positive" if positive else "negative"
    claim = positive_claim if positive else negative_claim
    return (
        f"The posterior probability of a {sign} association is "
        f"{_kf_pct(fav['favoured_direction_prob'])}% — "
        f"{fav['favoured_direction_label']} evidence that {claim}."
    )


def _kf_outcome_label(config: Mapping) -> str:
    """Outcome display label, mirroring the ``_setup.qmd`` derivation."""
    from language_reading_predictors.statistical_models.measures import MEASURES

    symbol = config.get("outcome_symbol")
    measure = MEASURES.get(symbol) if symbol else None
    if measure is not None:
        return measure.label
    return config.get("title") or symbol or "the outcome"


def _kf_direction_words(prob_pos, *, is_rd: bool) -> str:
    """The harm-aware confidence sentence body (#179): evidence for the
    *favoured* direction, so a clearly negative effect reads as evidence of harm
    rather than 'inconclusive'."""
    p = _kf_float(prob_pos)
    fav = favoured_direction(p)
    label = fav["favoured_direction_label"]
    if fav["favoured_direction"] == "positive":
        sign_word = "positive"
        claim = (
            "the intervention raises the chance of coming off the floor"
            if is_rd
            else "the intervention helps"
        )
    else:
        sign_word = "negative"
        claim = (
            "the intervention lowers the chance of coming off the floor"
            if is_rd
            else "the intervention is harmful"
        )
    # State the probability for the FAVOURED direction so the number and the
    # evidence label qualify the same claim (harm-aware, #179): a clearly
    # negative effect reads "97% probability ... negative — strong evidence of
    # harm", not "3% probability ... positive — strong evidence of harm".
    return (
        f"There is a {_kf_pct(fav['favoured_direction_prob'])}% probability "
        f"that the true effect is {sign_word} — {label} evidence that {claim}."
    )


def _kf_headline_from_rope(rope: Mapping, outcome_label: str, scope: str) -> tuple[str, bool]:
    """Headline sentence from a ``rope_summary.csv`` row.

    Returns ``(sentence, is_risk_difference)``. ``scope`` is a clause naming the
    comparison (e.g. 'over the trial period'), so each family can state exactly
    which contrast the number is."""
    is_rd = str(rope.get("delta_scale", "")) == "risk_difference"
    scale = 100.0 if is_rd else 1.0
    med = _kf_float(rope["items_median"]) * scale
    lo = _kf_float(rope["items_lo"]) * scale
    hi = _kf_float(rope["items_hi"]) * scale
    if is_rd:
        text = (
            f"Best estimate: the intervention changed the chance of scoring above "
            f"zero on {outcome_label} by **{med:+.0f} percentage points** {scope} "
            f"(89% credible range {lo:+.0f} to {hi:+.0f})."
        )
    else:
        text = (
            f"Best estimate: the intervention changed {outcome_label} by "
            f"**{med:+.1f} items** {scope} "
            f"(89% credible range {lo:+.1f} to {hi:+.1f})."
        )
    return text, is_rd


def _kf_rope_sentence(rope: Mapping, *, is_rd: bool) -> str:
    """The magnitude (ROPE) verdict from a ``rope_summary.csv`` row."""
    delta = _kf_float(rope["delta_items"]) * (100.0 if is_rd else 1.0)
    if is_rd:
        unit = "percentage point" if delta == 1 else "percentage points"
    else:
        unit = "item" if delta == 1 else "items"
    p_benefit = _kf_pct(rope["prob_benefit_ge_delta"])
    p_rope = _kf_pct(rope["prob_in_rope"])
    return (
        f"The project agreed after its initial results review that a change of at "
        f"least {delta:g} {unit} would be the smallest difference that matters in "
        f"practice. The probability the benefit reaches that size is {p_benefit}%, "
        f"and the probability the effect is too small to matter either way is "
        f"{p_rope}%; because the threshold is post-hoc, read this beside the "
        f"threshold-sensitivity analysis."
    )


def _kf_strongest_factor(output_dir, *, exclude_roles: tuple[str, ...] = ("causal",)) -> str | None:
    """Family-highlight sentence: the most clearly resolved adjusted association
    in ``factor_summary.csv``, or None when nothing usable is present.

    Ranked by ``|prob_positive - 0.5|`` (how clearly the direction is resolved),
    NOT by ``|median|`` — the factor coefficients sit on different scales (the
    own baseline enters on the raw logit scale, other covariates per SD), so
    magnitudes are not comparable across terms. Interaction terms and
    unlabelled coefficients are skipped — the box must stay readable without a
    code key."""
    path = os.path.join(str(output_dir), "factor_summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    needed = {"term", "role", "prob_positive"}
    if df.empty or not needed.issubset(df.columns):
        return None
    rows = df[~df["role"].isin(exclude_roles) & df["term"].isin(_KF_FACTOR_LABELS)]
    probs = pd.to_numeric(rows["prob_positive"], errors="coerce")
    rows = rows[np.isfinite(probs)]
    if rows.empty:
        return None
    top = rows.loc[(pd.to_numeric(rows["prob_positive"]) - 0.5).abs().idxmax()]
    label = _KF_FACTOR_LABELS[str(top["term"])]
    p = float(top["prob_positive"])
    fav = favoured_direction(p)
    ends = (
        "also tended to score higher afterwards"
        if fav["favoured_direction"] == "positive"
        else "tended to score lower afterwards"
    )
    return (
        f"Of the other factors in the model, {label} had the most clearly "
        f"resolved link with the outcome: children higher on it {ends} "
        f"(a {_kf_pct(fav['favoured_direction_prob'])}% probability for that "
        f"direction; an adjusted association, not a cause)."
    )


def _kf_build_itt(output_dir, config: Mapping) -> list[dict[str, str]]:
    """ITT suite (incl. the floored off-floor primaries): rope card first, tau
    summary as the no-agreed-delta fallback (F / T)."""
    outcome_label = _kf_outcome_label(config)
    rope = _kf_csv_row(output_dir, "rope_summary.csv")
    sentences: list[dict[str, str]] = []
    if rope is not None:
        headline, is_rd = _kf_headline_from_rope(rope, outcome_label, "over the trial period")
        sentences.append(_kf_sentence(headline, "headline"))
        sentences.append(
            _kf_sentence(_kf_direction_words(rope["pd"], is_rd=is_rd), "confidence")
        )
        sentences.append(_kf_sentence(_kf_rope_sentence(rope, is_rd=is_rd), "rope"))
    else:
        tau = _kf_csv_row(output_dir, "tau_summary.csv")
        if tau is None:
            raise _KeyFindingsUnavailable(
                "neither rope_summary.csv nor tau_summary.csv is present"
            )
        from language_reading_predictors.statistical_models.measures import MEASURES

        measure = MEASURES.get(config.get("outcome_symbol"))
        if measure is not None:
            n = measure.n_trials
            med = _kf_float(tau["tau_prob_median"]) * n
            lo = _kf_float(tau["tau_prob_lo"]) * n
            hi = _kf_float(tau["tau_prob_hi"]) * n
            sentences.append(
                _kf_sentence(
                    f"Best estimate: the intervention changed {outcome_label} by "
                    f"**{med:+.1f} items** over the trial period "
                    f"(89% credible range {lo:+.1f} to {hi:+.1f}).",
                    "headline",
                )
            )
        sentences.append(
            _kf_sentence(_kf_direction_words(tau["prob_tau_pos"], is_rd=False), "confidence")
        )
        sentences.append(
            _kf_sentence(
                "No minimally-important difference has been agreed for this "
                "outcome, so no is-it-big-enough-to-matter verdict is reported.",
                "note",
            )
        )
    sentences.append(
        _kf_sentence(
            "Because children were randomly assigned to the intervention or the "
            "waiting list, this estimate supports a cause-and-effect reading.",
            "causal",
        )
    )
    return sentences


def _kf_build_gain_factors(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Gain (ANCOVA) family: the randomised on-intervention term is the only
    causal coefficient, averaged over the period-1 transition; treated-only
    companions have no causal term at all."""
    outcome_label = _kf_outcome_label(config)
    treated_only = bool((config.get("extra") or {}).get("treated_only", False))
    sentences: list[dict[str, str]] = []
    if treated_only:
        sentences.append(
            _kf_sentence(
                f"This companion model looks only at children while they were "
                f"receiving the intervention, so it estimates no treatment effect "
                f"on {outcome_label} — every result in it is an adjusted "
                f"association, not a cause.",
                "causal",
            )
        )
        highlight = _kf_strongest_factor(output_dir)
        if highlight:
            sentences.append(_kf_sentence(highlight, "highlight"))
        return sentences
    rope = _kf_csv_row(output_dir, "rope_summary.csv")
    scope = "during the randomised first period"
    if rope is not None:
        headline, is_rd = _kf_headline_from_rope(rope, outcome_label, scope)
        sentences.append(_kf_sentence(headline, "headline"))
        sentences.append(
            _kf_sentence(_kf_direction_words(rope["pd"], is_rd=is_rd), "confidence")
        )
        sentences.append(_kf_sentence(_kf_rope_sentence(rope, is_rd=is_rd), "rope"))
    else:
        tm = _kf_csv_row(output_dir, "treatment_marginal.csv")
        if tm is None:
            raise _KeyFindingsUnavailable(
                "neither rope_summary.csv nor treatment_marginal.csv is present"
            )
        med = _kf_float(tm["trt_items_median"])
        lo = _kf_float(tm["trt_items_lo"])
        hi = _kf_float(tm["trt_items_hi"])
        sentences.append(
            _kf_sentence(
                f"Best estimate: being on the intervention changed {outcome_label} "
                f"by **{med:+.1f} items** {scope} "
                f"(89% credible range {lo:+.1f} to {hi:+.1f}).",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(_kf_direction_words(tm["prob_trt_pos"], is_rd=False), "confidence")
        )
    sentences.append(
        _kf_sentence(
            "The on-intervention effect is the only cause-and-effect estimate in "
            "this report (it rests on the randomised first period); every other "
            "factor is an adjusted association.",
            "causal",
        )
    )
    highlight = _kf_strongest_factor(output_dir)
    if highlight:
        sentences.append(_kf_sentence(highlight, "highlight"))
    return sentences


def _kf_build_level_factors(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Level family: only the t2 group contrast is randomised; later timepoints
    are post-crossover associations."""
    outcome_label = _kf_outcome_label(config)
    rope = _kf_csv_row(output_dir, "rope_summary.csv")
    if rope is None:
        raise _KeyFindingsUnavailable(
            "rope_summary.csv (the t2 items-scale contrast) is not present"
        )
    sentences: list[dict[str, str]] = []
    headline, is_rd = _kf_headline_from_rope(
        rope, outcome_label, "at the end of the randomised period (t2)"
    )
    sentences.append(_kf_sentence(headline, "headline"))
    sentences.append(
        _kf_sentence(_kf_direction_words(rope["pd"], is_rd=is_rd), "confidence")
    )
    sentences.append(_kf_sentence(_kf_rope_sentence(rope, is_rd=is_rd), "rope"))
    sentences.append(
        _kf_sentence(
            "Only this t2 comparison is randomised and supports a cause-and-effect "
            "reading; group differences at later timepoints — after the "
            "waiting-list children had crossed over to the intervention — are "
            "associations.",
            "causal",
        )
    )
    return sentences


def _kf_build_did(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Waitlist-crossover arm-by-wave family: the t2 arm contrast is the clean
    randomised quantity; the crossover catch-up is descriptive. Dose companions
    (no ``tau_t2``) get the honest association wording."""
    outcome_label = _kf_outcome_label(config)
    did = _kf_csv_row(output_dir, "did_summary.csv")
    if did is None:
        raise _KeyFindingsUnavailable("did_summary.csv is not present")
    if "tau_t2_items_median" not in did:
        if any(str(k).startswith("beta_dose") for k in did):
            # Dose companion: no randomised t2 contrast to headline.
            return [
                _kf_sentence(
                    "This companion model estimates how outcomes vary with the "
                    "amount of intervention received; that dose relationship is "
                    "an observational association, not a randomised comparison, "
                    "so no causal treatment-effect headline is reported.",
                    "causal",
                ),
                _kf_sentence(
                    "See the results section below for the dose estimates and "
                    "their uncertainty.",
                    "note",
                ),
            ]
        raise _KeyFindingsUnavailable(
            "did_summary.csv predates the arm-by-wave schema (no t2 items-scale "
            "contrast); refit or regenerate after a refit"
        )
    off_floor = bool(did.get("off_floor", False))
    sentences: list[dict[str, str]] = []
    if off_floor:
        med = _kf_float(did["tau_t2_items_median"]) * 100.0
        lo = _kf_float(did["tau_t2_items_lo"]) * 100.0
        hi = _kf_float(did["tau_t2_items_hi"]) * 100.0
        sentences.append(
            _kf_sentence(
                f"Best estimate: at t2 — the randomised comparison — being in the "
                f"immediate-intervention group changed the chance of scoring above "
                f"zero on {outcome_label} by **{med:+.0f} percentage points** "
                f"compared with the waiting list "
                f"(89% credible range {lo:+.0f} to {hi:+.0f}).",
                "headline",
            )
        )
    else:
        med = _kf_float(did["tau_t2_items_median"])
        lo = _kf_float(did["tau_t2_items_lo"])
        hi = _kf_float(did["tau_t2_items_hi"])
        higher_lower = "higher" if med >= 0 else "lower"
        sentences.append(
            _kf_sentence(
                f"Best estimate: at t2 — the randomised comparison — children in "
                f"the immediate-intervention group scored **{abs(med):.1f} items "
                f"{higher_lower}** on {outcome_label} than the waiting-list "
                f"children (89% credible range {lo:+.1f} to {hi:+.1f}).",
                "headline",
            )
        )
    sentences.append(
        _kf_sentence(
            _kf_direction_words(did["prob_tau_t2_pos"], is_rd=off_floor), "confidence"
        )
    )
    sentences.append(
        _kf_sentence(
            "The t2 contrast is randomised and supports a cause-and-effect "
            "reading; the arm gaps at t1 and t3, and the catch-up quantity below, "
            "are descriptive associations.",
            "causal",
        )
    )
    if bool(did.get("delta_crossover_items_available", False)):
        try:
            catch = _kf_float(did["delta_crossover_items_median"])
        except _KeyFindingsUnavailable:
            catch = None
        if catch is not None:
            unit = "percentage points" if off_floor else "items"
            moved = "narrowed" if catch > 0 else "widened"
            scale = 100.0 if off_floor else 1.0
            sentences.append(
                _kf_sentence(
                    f"After the waiting-list children started the intervention, the "
                    f"gap between the arms {moved} by about "
                    f"{abs(catch) * scale:.1f} {unit} — a descriptive catch-up "
                    f"quantity, not a second randomised effect.",
                    "highlight",
                )
            )
    return sentences


def _kf_build_joint(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Joint ITT: range across outcomes, never a cherry-picked single result."""
    df = _kf_csv(output_dir, "joint_treatment_marginal.csv")
    if df is None:
        raise _KeyFindingsUnavailable(
            "joint_treatment_marginal.csv is not present; this fit predates the "
            "joint items-scale pushforward"
        )
    required = {"outcome", "items_median", "items_lo", "items_hi", "prob_pos"}
    if not required.issubset(df.columns):
        raise _KeyFindingsUnavailable(
            "joint_treatment_marginal.csv does not have the expected columns"
        )
    medians = [_kf_float(v) for v in df["items_median"]]
    lows = [_kf_float(v) for v in df["items_lo"]]
    highs = [_kf_float(v) for v in df["items_hi"]]
    sentences = [
        _kf_sentence(
            f"Across the {len(df)} outcomes, the best estimates ranged from "
            f"**{min(medians):+.1f} to {max(medians):+.1f} items**; the individual "
            f"89% credible ranges extended from {min(lows):+.1f} to "
            f"{max(highs):+.1f} items overall.",
            "headline",
        )
    ]

    clearest = _kf_most_resolved_row(df, prob_col="prob_pos")
    symbol = str(clearest["outcome"])
    label = _kf_measure_label(symbol)
    direction = _kf_direction_words(clearest["prob_pos"], is_rd=False)
    sentences.append(
        _kf_sentence(
            f"For {label}, the clearest directional result: "
            f"{direction[0].lower() + direction[1:]}",
            "confidence",
        )
    )

    if {"delta_items", "prob_benefit_ge_delta"}.issubset(df.columns):
        deltas = df[
            np.isfinite(pd.to_numeric(df["delta_items"], errors="coerce"))
            & np.isfinite(
                pd.to_numeric(df["prob_benefit_ge_delta"], errors="coerce")
            )
        ]
        if not deltas.empty:
            probabilities = [
                _kf_float(v) for v in deltas["prob_benefit_ge_delta"]
            ]
            more_likely_than_not = sum(p >= 0.5 for p in probabilities)
            sentences.append(
                _kf_sentence(
                    f"Among the {len(deltas)} outcomes with a post-hoc, "
                    f"project-agreed smallest-important difference, "
                    f"{more_likely_than_not} were "
                    f"more likely than not to reach it; the outcome-specific "
                    f"probabilities ranged from {_kf_pct(min(probabilities))}% to "
                    f"{_kf_pct(max(probabilities))}%.",
                    "rope",
                )
            )
    sentences.append(
        _kf_sentence(
            "These are intention-to-treat effects from random assignment, so the "
            "outcome-specific estimates support a cause-and-effect reading.",
            "causal",
        )
    )
    return sentences


def _kf_build_mechanism(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Adjusted mechanism curve, preferably using its items-scale end contrast."""
    outcome_label = _kf_outcome_label(config)
    summary = _kf_csv_row(output_dir, "mechanism_summary.csv")
    sentences: list[dict[str, str]] = []
    if summary is not None:
        med = _kf_float(summary["items_median"])
        lo = _kf_float(summary["items_lo"])
        hi = _kf_float(summary["items_hi"])
        low = _kf_float(summary["exposure_low"])
        high = _kf_float(summary["exposure_high"])
        unit = _kf_plain_label(summary.get("exposure_unit", "predictor units"))
        sentences.append(
            _kf_sentence(
                f"Across the fitted exposure range ({low:g} to {high:g} {unit}), "
                f"{outcome_label} differed by **{med:+.1f} items** on average "
                f"(89% credible range {lo:+.1f} to {hi:+.1f}).",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                _kf_association_direction(
                    summary["prob_pos"],
                    positive_claim="higher exposure accompanies a higher outcome",
                    negative_claim="higher exposure accompanies a lower outcome",
                ),
                "confidence",
            )
        )
    else:
        curve = _kf_csv(output_dir, "mechanism_curve.csv")
        if curve is None:
            raise _KeyFindingsUnavailable(
                "neither mechanism_summary.csv nor mechanism_curve.csv is present"
            )
        x_col = "mech_x" if "mech_x" in curve.columns else "mech_logit"
        required = {x_col, "f_mean", "f_lo", "f_hi"}
        if not required.issubset(curve.columns):
            raise _KeyFindingsUnavailable(
                "mechanism_curve.csv does not have the expected columns"
            )
        ordered = curve.sort_values(x_col)
        low, high = ordered.iloc[0], ordered.iloc[-1]
        sentences.append(
            _kf_sentence(
                f"Across the fitted predictor range, its model contribution changed "
                f"from {_kf_float(low['f_mean']):+.2f} logit units "
                f"(89% range {_kf_float(low['f_lo']):+.2f} to "
                f"{_kf_float(low['f_hi']):+.2f}) to "
                f"{_kf_float(high['f_mean']):+.2f} "
                f"({_kf_float(high['f_lo']):+.2f} to "
                f"{_kf_float(high['f_hi']):+.2f}).",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                "This older fit has pointwise curve intervals but no saved "
                "posterior end-to-end contrast, so a single direction probability "
                "is not available until it is refitted.",
                "note",
            )
        )
    sentences.append(
        _kf_sentence(
            "The curve is an adjusted association between measured skills, not "
            "evidence that changing one skill would cause the other to change.",
            "causal",
        )
    )
    return sentences


def _kf_build_mediation(output_dir, config: Mapping) -> list[dict[str, str]]:
    """One- or two-mediator g-formula decomposition."""
    df = _kf_csv(output_dir, "mediation_summary.csv")
    if df is None or "quantity" not in df.columns:
        raise _KeyFindingsUnavailable("mediation_summary.csv is not present")
    indexed = df.set_index("quantity")
    if "total" not in indexed.index:
        raise _KeyFindingsUnavailable(
            "mediation_summary.csv has no total-effect row"
        )
    total = indexed.loc["total"].to_dict()
    off_floor = str(total.get("off_floor", "false")).lower() in {"true", "1"}
    scale = 100.0 if off_floor else 1.0
    unit = "percentage points" if off_floor else "items"
    med = _kf_float(total["words_median"]) * scale
    lo = _kf_float(total["words_lo"]) * scale
    hi = _kf_float(total["words_hi"]) * scale
    fav = favoured_direction(_kf_float(total["prob_pos"]))
    positive = fav["favoured_direction"] == "positive"
    direction = "positive" if positive else "negative"
    claim = (
        "the intervention improves the outcome under the fitted model"
        if positive
        else "the intervention worsens the outcome under the fitted model"
    )
    sentences = [
        _kf_sentence(
            f"The model-based total intervention contrast was **{med:+.1f} "
            f"{unit}** (89% credible range {lo:+.1f} to {hi:+.1f}).",
            "headline",
        ),
        _kf_sentence(
            f"The posterior probability that this model-based total contrast is "
            f"{direction} is {_kf_pct(fav['favoured_direction_prob'])}% — "
            f"{fav['favoured_direction_label']} evidence that {claim}.",
            "confidence",
        ),
    ]
    indirect_name = next(
        (name for name in ("NIE_joint", "NIE", "IIE") if name in indexed.index),
        None,
    )
    if indirect_name is not None:
        indirect = indexed.loc[indirect_name].to_dict()
        i_med = _kf_float(indirect["words_median"]) * scale
        i_lo = _kf_float(indirect["words_lo"]) * scale
        i_hi = _kf_float(indirect["words_hi"]) * scale
        sentences.append(
            _kf_sentence(
                f"The estimated indirect component ({indirect_name}) was "
                f"{i_med:+.1f} {unit} (89% credible range {i_lo:+.1f} to "
                f"{i_hi:+.1f}).",
                "highlight",
            )
        )
    sentences.append(
        _kf_sentence(
            "The direct/indirect split is a model-based g-formula decomposition, "
            "not an identified natural mediation effect: unmeasured "
            "mediator-outcome confounding remains a binding assumption.",
            "causal",
        )
    )
    return sentences


def _kf_build_aligned(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Onset-aligned per-protocol cohort contrast; every term is associative."""
    outcome_label = _kf_outcome_label(config)
    marginal = _kf_csv_row(output_dir, "cohort_marginal.csv")
    if marginal is None:
        raise _KeyFindingsUnavailable("cohort_marginal.csv is not present")
    off_floor = (config.get("extra") or {}).get("likelihood") == "bernoulli_offfloor"
    scale = 100.0 if off_floor else 1.0
    unit = "percentage points" if off_floor else "items"
    med = _kf_float(marginal["trt_items_median"]) * scale
    lo = _kf_float(marginal["trt_items_lo"]) * scale
    hi = _kf_float(marginal["trt_items_hi"]) * scale
    sentences = [
        _kf_sentence(
            f"After aligning children by intervention onset, the immediate cohort "
            f"differed from the waiting-list cohort on {outcome_label} by "
            f"**{med:+.1f} {unit}** (89% credible range {lo:+.1f} to "
            f"{hi:+.1f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                marginal["prob_trt_pos"],
                positive_claim="the immediate cohort tends to score higher",
                negative_claim="the immediate cohort tends to score lower",
            ),
            "confidence",
        ),
        _kf_sentence(
            "This is a per-protocol cohort association, not a randomised treatment "
            "effect; age at onset and cohort timing can confound it.",
            "causal",
        ),
    ]
    highlight = _kf_strongest_factor(output_dir, exclude_roles=())
    if highlight:
        sentences.append(_kf_sentence(highlight, "highlight"))
    return sentences


def _kf_build_adjusted(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Between-child adjusted predictor associations on the items scale."""
    df = _kf_csv(output_dir, "predicted_gain_words.csv")
    if df is None:
        raise _KeyFindingsUnavailable("predicted_gain_words.csv is not present")
    row = _kf_most_resolved_row(df, prob_col="prob_pos")
    label = _kf_plain_label(row.get("label", row.get("predictor", "predictor")))
    med = _kf_float(row["delta_words_mean"])
    lo = _kf_float(row["delta_words_lo"])
    hi = _kf_float(row["delta_words_hi"])
    outcome_label = _kf_outcome_label(config)
    return [
        _kf_sentence(
            f"The clearest adjusted predictor was {label}: a 1-SD increase was "
            f"associated with **{med:+.1f} items** of difference in "
            f"{outcome_label} "
            f"(89% credible range {lo:+.1f} to {hi:+.1f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim="higher values accompany greater gain",
                negative_claim="higher values accompany less gain",
            ),
            "confidence",
        ),
        _kf_sentence(
            "This is a between-child adjusted association; it does not identify "
            "what would happen if the predictor were changed.",
            "causal",
        ),
    ]


def _kf_build_corr_factor(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Cross-sectional correlated-domain measurement model."""
    correlations = _kf_csv(output_dir, "factor_correlation_summary.csv")
    structural = _kf_csv(output_dir, "structural_summary.csv")
    if correlations is None and structural is None:
        raise _KeyFindingsUnavailable(
            "neither factor_correlation_summary.csv nor structural_summary.csv is present"
        )
    sentences: list[dict[str, str]] = []
    if correlations is not None:
        row = _kf_most_resolved_row(correlations, prob_col="prob_pos")
        pair = (
            f"{_kf_plain_label(row['domain_i'])} and "
            f"{_kf_plain_label(row['domain_j'])}"
        )
        sentences.extend(
            [
                _kf_sentence(
                    f"The clearest latent-domain correlation was between {pair}: "
                    f"**{_kf_float(row['median']):+.2f}** (89% credible range "
                    f"{_kf_float(row['lo']):+.2f} to "
                    f"{_kf_float(row['hi']):+.2f}).",
                    "headline",
                ),
                _kf_sentence(
                    _kf_association_direction(
                        row["prob_pos"],
                        positive_claim="the two latent skill areas tend to move together",
                        negative_claim="the two latent skill areas tend to move oppositely",
                    ),
                    "confidence",
                ),
            ]
        )
    if structural is not None:
        row = _kf_most_resolved_row(structural, prob_col="prob_pos")
        sentences.append(
            _kf_sentence(
                f"The clearest structural slope was "
                f"{_kf_plain_label(row['coefficient'])}: "
                f"{_kf_float(row['median']):+.2f} logit units (89% credible range "
                f"{_kf_float(row['lo']):+.2f} to "
                f"{_kf_float(row['hi']):+.2f}).",
                "highlight",
            )
        )
        if correlations is None:
            sentences.append(
                _kf_sentence(
                    _kf_association_direction(
                        row["prob_pos"],
                        positive_claim="the linked latent quantities tend to move together",
                        negative_claim="the linked latent quantities tend to move oppositely",
                    ),
                    "confidence",
                )
            )
    sentences.append(
        _kf_sentence(
            "This is a measurement and triangulation model; its factor "
            "correlations and structural slopes are associations, not causal effects.",
            "causal",
        )
    )
    return sentences


def _kf_build_dose_response(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Observational session-dose association."""
    marginal = _kf_csv_row(output_dir, "dose_marginal_summary.csv")
    outcome_label = _kf_outcome_label(config)
    sentences: list[dict[str, str]] = []
    if marginal is not None:
        sentences.append(
            _kf_sentence(
                f"A 1-SD increase in sessions was associated with "
                f"**{_kf_float(marginal['items_median']):+.1f} items** on "
                f"{outcome_label} "
                f"(89% credible range {_kf_float(marginal['items_lo']):+.1f} "
                f"to {_kf_float(marginal['items_hi']):+.1f}).",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                _kf_association_direction(
                    marginal["prob_pos"],
                    positive_claim="higher session dose accompanies a higher outcome",
                    negative_claim="higher session dose accompanies a lower outcome",
                ),
                "confidence",
            )
        )
    else:
        slopes = _kf_csv(output_dir, "dose_slope_summary.csv")
        if slopes is None:
            raise _KeyFindingsUnavailable(
                "neither dose_marginal_summary.csv nor dose_slope_summary.csv is present"
            )
        row = slopes.iloc[0].to_dict()
        sentences.append(
            _kf_sentence(
                f"The headline dose slope was "
                f"**{_kf_float(row['median']):+.2f} logit units per 1 SD of "
                f"sessions** (89% credible range {_kf_float(row['lo']):+.2f} "
                f"to {_kf_float(row['hi']):+.2f}).",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                _kf_association_direction(
                    row["p_pos"],
                    positive_claim="higher session dose accompanies a higher outcome",
                    negative_claim="higher session dose accompanies a lower outcome",
                ),
                "confidence",
            )
        )
    sentences.append(
        _kf_sentence(
            "Session dose was not randomised and may reflect ability, attendance "
            "or availability, so the slope is an observational association.",
            "causal",
        )
    )
    return sentences


def _kf_build_lcsm(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Latent change-score couplings, with an optional randomised-window check."""
    df = _kf_csv(output_dir, "coupling_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("coupling_summary.csv is not present")
    directed = df[df["coefficient"].astype(str).str.contains("->", regex=False)]
    if directed.empty:
        directed = df
    row = _kf_most_resolved_row(directed, prob_col="prob_pos")
    label = _kf_plain_label(row["coefficient"])
    if "(" in label and label.endswith(")"):
        label = label.split("(", 1)[1][:-1]
    sentences = [
        _kf_sentence(
            f"The clearest longitudinal coupling was {label}: "
            f"**{_kf_float(row['median']):+.2f} logit units** (89% credible range "
            f"{_kf_float(row['lo']):+.2f} to {_kf_float(row['hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim="a higher earlier level accompanies greater later change",
                negative_claim="a higher earlier level accompanies less later change",
            ),
            "confidence",
        ),
        _kf_sentence(
            "The couplings are conditional predictive associations among latent "
            "trajectories, not causal skill-to-skill effects.",
            "causal",
        ),
    ]
    itt = _kf_csv(output_dir, "itt_window1_contrast.csv")
    if itt is not None:
        check = _kf_most_resolved_row(itt, prob_col="prob_pos")
        sentences.append(
            _kf_sentence(
                f"The separate randomised window-1 consistency contrast was "
                f"{_kf_float(check['median']):+.2f} latent-logit units (89% credible "
                f"range {_kf_float(check['lo']):+.2f} to "
                f"{_kf_float(check['hi']):+.2f}); it is a check, not the coupling "
                f"headline.",
                "highlight",
            )
        )
    return sentences


def _kf_build_horseshoe(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Regularised-horseshoe predictor-ranking sensitivity analysis."""
    df = _kf_csv(output_dir, "predictor_ranking.csv")
    if df is None:
        raise _KeyFindingsUnavailable("predictor_ranking.csv is not present")
    row = df.sort_values("rank").iloc[0].to_dict()
    label = _kf_plain_label(row["predictor"])
    direction = "positive" if _kf_float(row["beta_median"]) >= 0 else "negative"
    return [
        _kf_sentence(
            f"The top-ranked predictor was {label}, with a standardised "
            f"{direction} association of **{_kf_float(row['beta_median']):+.2f} "
            f"logit units** (89% highest-density interval "
            f"{_kf_float(row['beta_hdi_lo']):+.2f} to "
            f"{_kf_float(row['beta_hdi_hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            f"Its probability of exceeding the model's worth-noticing "
            f"coefficient threshold was {_kf_pct(row['p_abs_gt_delta'])}%.",
            "confidence",
        ),
        _kf_sentence(
            "The ranking is an adjusted predictive sensitivity check, not a list "
            "of causal drivers; closely ranked predictors should not be treated as "
            "meaningfully ordered.",
            "causal",
        ),
    ]


def _kf_build_growth(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Multivariate growth: baseline ability association with growth rate."""
    df = _kf_csv(output_dir, "growth_association_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("growth_association_summary.csv is not present")
    gamma = df[df["coefficient"] == "gamma"]
    if gamma.empty:
        raise _KeyFindingsUnavailable("growth summary has no gamma rows")
    row = _kf_most_resolved_row(gamma, prob_col="prob_positive")
    outcome = _kf_measure_label(row["outcome"])
    return [
        _kf_sentence(
            f"For {outcome}, the clearest result, a 1-SD higher baseline "
            f"non-verbal ability score was associated with a growth-rate change of "
            f"**{_kf_float(row['median']):+.2f} logit units** (89% credible range "
            f"{_kf_float(row['lo89']):+.2f} to "
            f"{_kf_float(row['hi89']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_positive"],
                positive_claim="higher baseline ability accompanies faster growth",
                negative_claim="higher baseline ability accompanies slower growth",
            ),
            "confidence",
        ),
        _kf_sentence(
            "These trajectory coefficients are adjusted associations, not effects "
            "of changing non-verbal ability.",
            "causal",
        ),
    ]


def _kf_build_historical_growth(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Historical-cohort natural-history reproduction."""
    df = _kf_csv(output_dir, "posterior_growth_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("posterior_growth_summary.csv is not present")
    within = df[df["readgrp_label"].fillna("").astype(str).str.len() > 0]
    if within.empty:
        within = df
    row = _kf_most_resolved_row(within, prob_col="p_gt_0")
    group = _kf_plain_label(row.get("readgrp_label", "historical cohort"))
    fav = favoured_direction(_kf_float(row["p_gt_0"]))
    positive = fav["favoured_direction"] == "positive"
    direction = "positive" if positive else "negative"
    claim = (
        "scores tend to increase over that interval"
        if positive
        else "scores tend to decrease over that interval"
    )
    sentences = [
        _kf_sentence(
            f"For the {group} group, {_kf_plain_label(row['label'])} was "
            f"**{_kf_float(row['mean']):+.1f} items** (89% credible range "
            f"{_kf_float(row['q_lo']):+.1f} to "
            f"{_kf_float(row['q_hi']):+.1f}).",
            "headline",
        ),
        _kf_sentence(
            f"The posterior probability that this growth is {direction} is "
            f"{_kf_pct(fav['favoured_direction_prob'])}% — "
            f"{fav['favoured_direction_label']} evidence that {claim}.",
            "confidence",
        ),
        _kf_sentence(
            "This is descriptive natural-history growth in a historical cohort, "
            "not an intervention effect or an explanation of group differences.",
            "causal",
        ),
    ]
    cells = _kf_csv(output_dir, "posterior_cell_summary.csv")
    if cells is not None and "posterior_mean_minus_observed_mean" in cells.columns:
        gaps = [
            abs(_kf_float(v)) for v in cells["posterior_mean_minus_observed_mean"]
        ]
        sentences.append(
            _kf_sentence(
                f"As a reproduction check, the largest fitted-minus-observed cell "
                f"mean gap was {max(gaps):.1f} items.",
                "highlight",
            )
        )
    return sentences


def _kf_build_historical_joint(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Byrne joint correlated growth: cross-measure coupling headline (#338)."""
    df = _kf_csv(output_dir, "measure_correlation_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("measure_correlation_summary.csv is not present")
    row = _kf_most_resolved_row(df, prob_col="prob_pos")
    pair = (
        f"{_kf_plain_label(row.get('label_i', row['measure_i']))} and "
        f"{_kf_plain_label(row.get('label_j', row['measure_j']))}"
    )
    return [
        _kf_sentence(
            f"The clearest between-child coupling was between {pair}: a stable-"
            f"level correlation of **{_kf_float(row['median']):+.2f}** (89% credible "
            f"range {_kf_float(row['lo']):+.2f} to {_kf_float(row['hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim=(
                    "children who sit higher on one measure tend to sit higher "
                    "on the other"
                ),
                negative_claim=(
                    "children who sit higher on one measure tend to sit lower "
                    "on the other"
                ),
            ),
            "confidence",
        ),
        _kf_sentence(
            "This is a descriptive between-child correlation of stable levels in "
            "a historical cohort - it is not causal and does not say that "
            "changing one skill changes another.",
            "causal",
        ),
    ]


def _kf_build_survival(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Discrete-time off-floor hazard model."""
    df = _kf_csv(output_dir, "survival_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("survival_summary.csv is not present")
    effects = df[np.isfinite(pd.to_numeric(df["P(>0)"], errors="coerce"))]
    if effects.empty:
        raise _KeyFindingsUnavailable("survival summary has no directional effects")
    treatment = effects[effects["term"].astype(str).str.startswith("tau")]
    row = (treatment.iloc[0] if not treatment.empty else _kf_most_resolved_row(
        effects, prob_col="P(>0)"
    )).to_dict()
    ratio = np.exp(_kf_float(row["median"]))
    ratio_lo = np.exp(_kf_float(row["ci_low"]))
    ratio_hi = np.exp(_kf_float(row["ci_high"]))
    label = _kf_plain_label(row["term"])
    sentences = [
        _kf_sentence(
            f"The {label} corresponded to a hazard ratio of **{ratio:.2f}** "
            f"(89% credible range {ratio_lo:.2f} to {ratio_hi:.2f}) for coming "
            f"off the floor in an interval.",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["P(>0)"],
                positive_claim="the reported term accompanies earlier movement off the floor",
                negative_claim="the reported term accompanies later movement off the floor",
            ),
            "confidence",
        ),
        _kf_sentence(
            "This is a prognostic association over all waves; because both arms "
            "are treated by the final wave, it is not a randomised effect of record.",
            "causal",
        ),
    ]
    baseline = df[df["term"].astype(str).str.startswith("baseline off-floor prob")]
    if not baseline.empty:
        values = [_kf_float(v) for v in baseline["median"]]
        sentences.append(
            _kf_sentence(
                f"For an untreated child at mean covariates, the fitted baseline "
                f"off-floor probability ranged from {_kf_pct(min(values))}% to "
                f"{_kf_pct(max(values))}% across intervals.",
                "highlight",
            )
        )
    return sentences


def _kf_build_block_exposure(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Staggered block-2 active-exposure association."""
    row = _kf_csv_row(output_dir, "block_exposure_summary.csv")
    if row is None:
        raise _KeyFindingsUnavailable("block_exposure_summary.csv is not present")
    off_floor = (config.get("extra") or {}).get("likelihood") == "bernoulli_offfloor"
    scale = 100.0 if off_floor else 1.0
    unit = "percentage points" if off_floor else "items"
    outcome_label = _kf_outcome_label(config)
    return [
        _kf_sentence(
            f"When block-2 teaching was active, {outcome_label} differed by "
            f"**{_kf_float(row['delta_items_median']) * scale:+.1f} {unit}** "
            f"(89% credible range "
            f"{_kf_float(row['delta_items_lo']) * scale:+.1f} to "
            f"{_kf_float(row['delta_items_hi']) * scale:+.1f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_delta_pos"],
                positive_claim="active block-2 teaching accompanies a higher outcome",
                negative_claim="active block-2 teaching accompanies a lower outcome",
            ),
            "confidence",
        ),
        _kf_sentence(
            "Block-2 exposure was not randomised; this is a parallel-trends "
            "association comparing block-2-active with block-1-active periods.",
            "causal",
        ),
    ]


def _kf_build_concurrent(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Per-wave mutually-adjusted same-time associations."""
    df = _kf_csv(output_dir, "concurrent_marginals.csv")
    if df is None:
        raise _KeyFindingsUnavailable("concurrent_marginals.csv is not present")
    converged = df["converged"].astype(str).str.lower().isin({"true", "1"})
    rows = df[
        (df["adjustment"] == "adjusted")
        & (df["scale"] == "+1 SD")
        & converged
    ]
    if rows.empty:
        raise _KeyFindingsUnavailable(
            "no converged adjusted +1 SD concurrent marginals are present"
        )
    row = _kf_most_resolved_row(rows, prob_col="prob_pos")
    label = _kf_plain_label(row.get("label", row["term"]))
    return [
        _kf_sentence(
            f"At t{int(_kf_float(row['timepoint']))}, the clearest adjusted "
            f"same-wave predictor was {label}: +1 SD was associated with "
            f"**{_kf_float(row['items_median']):+.1f} outcome items** (89% "
            f"credible range {_kf_float(row['items_lo']):+.1f} to "
            f"{_kf_float(row['items_hi']):+.1f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim="the two same-wave skills tend to be higher together",
                negative_claim="the two same-wave skills tend to move oppositely",
            ),
            "confidence",
        ),
        _kf_sentence(
            "All concurrent coefficients condition on post-treatment skills and "
            "are descriptive associations, not causal pathways.",
            "causal",
        ),
    ]


def _kf_build_long_corr_factor(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Longitudinal latent-domain measurement model, using its items translation."""
    df = _kf_csv(output_dir, "latent_items_slopes.csv")
    if df is None:
        raise _KeyFindingsUnavailable("latent_items_slopes.csv is not present")
    row = _kf_most_resolved_row(df, prob_col="prob_pos")
    predictor = _kf_measure_label(row["predictor_indicator"])
    target = _kf_measure_label(row["target_indicator"])
    return [
        _kf_sentence(
            f"At wave {int(_kf_float(row['wave']))}, the clearest translated latent "
            f"coupling linked +1 {predictor} item with "
            f"**{_kf_float(row['items_per_item_mean']):+.2f} {target} items** "
            f"(89% credible range {_kf_float(row['items_per_item_lo']):+.2f} "
            f"to {_kf_float(row['items_per_item_hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim="the two latent domains tend to move together",
                negative_claim="the two latent domains tend to move oppositely",
            ),
            "confidence",
        ),
        _kf_sentence(
            "This items-scale slope is a linearised measurement-model "
            "association at the average operating point, not a caused gain.",
            "causal",
        ),
    ]


def _kf_build_fallback(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Unknown future family: an honest placeholder, never a wrong summary."""
    kind = config.get("kind") or "this"
    return [
        _kf_sentence(
            f"A plain-language key-findings summary has not yet been written for "
            f"the {kind} model family.",
            "note",
        ),
        _kf_sentence(
            "Unless a term is explicitly flagged as randomised in the results "
            "below, the estimates in this report are adjusted associations or "
            "descriptive quantities, not causal effects.",
            "causal",
        ),
        _kf_sentence(
            "See the results section below for the full estimates with their "
            "uncertainty.",
            "note",
        ),
    ]


_KF_BUILDERS = {
    "itt": _kf_build_itt,
    "joint": _kf_build_joint,
    "mechanism": _kf_build_mechanism,
    "mediation": _kf_build_mediation,
    "mediation_multi": _kf_build_mediation,
    "did": _kf_build_did,
    "gain_factors": _kf_build_gain_factors,
    "level_factors": _kf_build_level_factors,
    "aligned": _kf_build_aligned,
    "adjusted": _kf_build_adjusted,
    "corr_factor": _kf_build_corr_factor,
    "dose_response": _kf_build_dose_response,
    "lcsm": _kf_build_lcsm,
    "horseshoe": _kf_build_horseshoe,
    "growth": _kf_build_growth,
    "historical_growth": _kf_build_historical_growth,
    "historical_joint": _kf_build_historical_joint,
    "survival": _kf_build_survival,
    "block_exposure": _kf_build_block_exposure,
    "concurrent": _kf_build_concurrent,
    "long_corr_factor": _kf_build_long_corr_factor,
}

# Human-readable names for the convergence-gate checks (the gate-failed banner).
_KF_CHECK_LABELS = {
    "rhat": "R-hat",
    "ess": "effective sample size",
    "divergences": "divergent transitions",
    "bfmi": "sampling energy (BFMI)",
}


def _convergence_gate_failures(diag_summary: Mapping | None) -> list[str]:
    """Return readable failed checks from a diagnostics-gate payload.

    ``passed`` is the authoritative verdict written by
    ``dse_research_utils``.  The per-check mapping supplies the explanation; an
    absent or internally inconsistent payload fails closed as an incomplete
    convergence summary.  The same helper drives both the prominent report
    badge and the key-findings interlock so their failure labels cannot drift.
    """
    if not isinstance(diag_summary, Mapping) or diag_summary.get("passed") is not True:
        checks = diag_summary.get("checks") if isinstance(diag_summary, Mapping) else None
        if isinstance(checks, Mapping):
            failing = [
                _KF_CHECK_LABELS.get(str(name), str(name))
                for name, ok in checks.items()
                if ok is not True
            ]
            if failing:
                return failing
        return ["convergence summary incomplete"]
    return []


def convergence_gate_badge_markdown(diag_summary: Mapping | None) -> str:
    """Render the compact pass/fail badge shown before report findings (#321).

    A failed or unavailable gate is deliberately rendered as a red ``important``
    callout and explicitly withholds interpretation.  The full numerical banner
    remains available in the collapsed Technical checks section.
    """
    failing = _convergence_gate_failures(diag_summary)
    if not failing:
        return (
            '::: {.callout-tip title="Sampling-quality gate: passed"}\n\n'
            "**PASS** — All sampling-quality checks passed; details are under "
            "Technical checks.\n\n:::"
        )
    failed_text = ", ".join(failing)
    return (
        '::: {.callout-important title="Sampling-quality gate: failed"}\n\n'
        f"**FAIL** — Sampling-quality checks failed: {failed_text}. Findings are "
        "withheld; review Technical checks before interpreting any estimate.\n\n:::"
    )


def generate_key_findings(output_dir) -> dict:
    """Build and write ``key_findings.json`` for a fit output directory (#320).

    Reads only artefacts already in ``output_dir`` (``config.json``,
    ``diagnostics_summary.json`` and the family CSVs), so it can be re-run over
    an existing fit without refitting. The convergence gate is checked FIRST: a
    failed gate yields a ``gate_failed`` payload and no findings — students must
    not meet findings from an unconverged fit. Missing artefacts degrade to a
    ``not_available`` payload with a reason, never an exception; sentences are
    capped at :data:`KEY_FINDINGS_MAX_SENTENCES` and can never contain a
    non-finite number (:func:`_kf_float` raises, and the builder's whole payload
    then degrades). Returns the payload it wrote.
    """
    out = str(output_dir)
    config = {}
    config_path = os.path.join(out, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            # A malformed config.json must degrade to not_available (after the
            # gate check, which outranks it), not abort a fit's finalisation.
            config = None

    payload: dict = {
        "schema_version": KEY_FINDINGS_SCHEMA_VERSION,
        "model_id": (config or {}).get("model_id"),
        "kind": (config or {}).get("kind"),
        "sentences": [],
    }

    diag_path = os.path.join(out, "diagnostics_summary.json")
    if not os.path.exists(diag_path):
        payload["status"] = "not_available"
        payload["reason"] = (
            "diagnostics_summary.json is missing, so the convergence gate cannot "
            "be checked"
        )
        return _write_key_findings(out, payload)
    try:
        with open(diag_path) as f:
            diag = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        payload["status"] = "not_available"
        payload["reason"] = (
            "diagnostics_summary.json could not be parsed, so the convergence "
            "gate cannot be checked"
        )
        return _write_key_findings(out, payload)
    failing = _convergence_gate_failures(diag)
    if failing:
        payload["status"] = "gate_failed"
        payload["failing_checks"] = failing
        return _write_key_findings(out, payload)

    if not config:
        payload["status"] = "not_available"
        payload["reason"] = (
            "config.json could not be parsed"
            if config is None
            else "config.json is missing"
        )
        return _write_key_findings(out, payload)

    builder = _KF_BUILDERS.get(config.get("kind"), _kf_build_fallback)
    try:
        sentences = builder(out, config)
    except _KeyFindingsUnavailable as exc:
        payload["status"] = "not_available"
        payload["reason"] = str(exc)
        return _write_key_findings(out, payload)
    except (KeyError, ValueError, OSError) as exc:
        # A malformed CSV must degrade to an explicit note, never break a fit
        # or a render (#320 acceptance criteria).
        payload["status"] = "not_available"
        payload["reason"] = f"key-findings builder failed: {exc}"
        return _write_key_findings(out, payload)

    payload["status"] = "ok"
    payload["sentences"] = sentences[:KEY_FINDINGS_MAX_SENTENCES]
    return _write_key_findings(out, payload)


def _write_key_findings(output_dir: str, payload: dict) -> dict:
    with open(os.path.join(output_dir, KEY_FINDINGS_FILENAME), "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return payload
