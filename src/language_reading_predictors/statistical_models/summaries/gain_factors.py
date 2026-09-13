# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gain factors calculations and summaries."""

from __future__ import annotations

from language_reading_predictors.statistical_models.posteriors import REPORTING_CI_PROB

from collections.abc import Sequence
import numpy as np
import xarray as xr
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
)
from language_reading_predictors.statistical_models.posteriors import (
    band50,
    derived_mc_diagnostics,
)
from language_reading_predictors.statistical_models.summaries.itt import (
    _itt_ame_draws,
)


def treatment_marginal_effect(
    trace: xr.DataTree,
    *,
    trt: np.ndarray,
    n_trials: int,
    term: str = "beta_trt",
    eta_name: str = "eta",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    ci_prob: float = REPORTING_CI_PROB,
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
    equal-tailed ``ci_prob`` interval. Each effect is calculated from the joint coefficients within a posterior
    draw before taking its median and interval. Transforming median coefficients
    does not in general give the median marginal effect. ``prob_trt_pos`` is the probability of direction of the **marginal
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
