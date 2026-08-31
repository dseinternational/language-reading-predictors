# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prior-pushforward and posterior-predictive coverage helpers (#637 stage 3).

The estimand-scale prior checks (#381) and the predictive coverage statistics
(#318): what a prior implies on the reported scale before the data, and how much
of the observed data the fitted model's prediction intervals contain.

``prior_artifacts`` and ``ppc_artifacts`` write these; this module computes them.
"""


from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

from language_reading_predictors.statistical_models.estimands import (
    _itt_ame_draws,
    _joint_ame_draws,
    level_t2_marginal_effect,
)
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    apply_score_mean_link,
)

def pushforward_values(
    effect_draws: np.ndarray,
    items: np.ndarray,
    *,
    n_trials: int,
    ci_prob: float,
) -> dict[str, float]:
    """The numeric ``prior_pushforward.csv`` schema, shared by every family (#381).

    ``effect_draws`` is the estimand on the model's own linear-predictor scale
    (logit for every family that reports one) and ``items`` the same estimand
    already multiplied onto the items scale, both ``(S,)`` per-draw arrays. Kept
    separate from the family transforms so a new family cannot invent its own
    column names, and separate from the *labelling* columns (see
    :func:`labelled_pushforward`) so the numeric keys stay exactly what
    ``blending_sensitivity`` recomputes and key-matches against the released
    phoneme-blending bundle.
    """
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


def pushforward_scale_for(n_trials: float | int) -> str:
    """Name the scale of the ``prior_items_*`` columns from the denominator (#381).

    ``n_trials`` is the outcome's item ceiling, and the pipeline passes **1**
    exactly when the quantity is not an item count at all — a floor-rule model's
    off-floor risk difference, or a fit with no item denominator to put the
    marginal on. No real measure in the suite has a ceiling of one, so the
    denominator is a sound discriminator, and deriving the scale from it here
    means a call site cannot pass a denominator of 1 and label it "items".

    That mismatch is not hypothetical: eight fits across the four original
    families (``itt-009``/``011``, ``gf-005``/``011``, ``lf-005``/``011``,
    ``did-011``/``012``) published a probability difference described as items,
    and so a hundred times too small, before this rule existed.
    """
    return "percentage points" if int(n_trials) == 1 else "items"


def labelled_pushforward(
    values: Mapping[str, float],
    *,
    estimand: str,
    estimand_label: str,
    role: str,
    scale: str | None = None,
) -> dict[str, Any]:
    """Attach the estimand-naming columns to one numeric pushforward row (#381).

    The four label columns exist because the check generalised beyond the ITT
    families: ``_priors.qmd`` used to hard-code "the prior on the **treatment
    effect**", which is true of ``tau`` / ``beta_trt`` / ``tau_t2`` /
    ``b_grp_time[1]`` and false of every family added since — the aligned cohort
    contrast is explicitly *not* randomised, and the concurrent, dose, mechanism
    and horseshoe estimands are adjusted associations. A renderer that cannot
    read what the row is a pushforward *of* can only describe it wrongly, so the
    row carries its own name, role and scale.

    ``role`` is the same vocabulary the priors table uses (``causal``,
    ``association``, ``descriptive``). ``scale`` names the units of the
    ``prior_items_*`` columns and defaults to :func:`pushforward_scale_for` of
    the row's own denominator — pass it only to override that.
    """
    return {
        "estimand": estimand,
        "estimand_label": estimand_label,
        "role": role,
        "scale": scale
        if scale is not None
        else pushforward_scale_for(values.get("n_trials", 0)),
        "status": "ok",
        "reason": "",
        **{k: v for k, v in values.items()},
    }


def unavailable_pushforward(
    *,
    estimand: str,
    estimand_label: str,
    role: str,
    reason: str,
    scale: str | None = None,
) -> dict[str, Any]:
    """A pushforward row recording that the check could **not** be computed (#381).

    The finding that motivated #381 is that an absent artefact reads as a clean
    one: "no flags" and "not measured" are indistinguishable to a reader of the
    rendered report. So a family that cannot push its prior through its estimand
    writes this row rather than no file, and ``_priors.qmd`` prints the reason.
    """
    return {
        "estimand": estimand,
        "estimand_label": estimand_label,
        "role": role,
        # No numbers to scale, so the column is only a placeholder here; keep it
        # populated rather than blank so a consumer can read it unconditionally.
        "scale": scale if scale is not None else "items",
        "status": "unavailable",
        "reason": reason,
        "prior_logit_median": float("nan"),
        "prior_logit_lo": float("nan"),
        "prior_logit_hi": float("nan"),
        "prior_items_median": float("nan"),
        "prior_items_lo50": float("nan"),
        "prior_items_hi50": float("nan"),
        "prior_items_lo": float("nan"),
        "prior_items_hi": float("nan"),
        "n_trials": 0,
    }


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
    score_mean_link: ScoreMeanLink = "logit",
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

    Returns the bare numeric row. The caller labels it via
    :func:`labelled_pushforward`; the keys here are what ``blending_sensitivity``
    recomputes from the trace and compares against the saved CSV, so they must not
    grow non-numeric members.
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
        score_mean_link=score_mean_link,
    )
    return pushforward_values(
        effect_draws, ame_prob * float(n_trials), n_trials=n_trials, ci_prob=ci_prob
    )


def marginal_prior_pushforward(
    trace: xr.DataTree,
    *,
    term: str,
    n_trials: int,
    eta_name: str = "eta",
    ci_prob: float = 0.95,
    convention: Literal["net_out", "forward"] = "net_out",
    row_mask: np.ndarray | None = None,
    term_index: Mapping[str, Any] | None = None,
    score_mean_link: ScoreMeanLink = "logit",
) -> dict[str, float]:
    """Prior pushforward for a family whose estimand is a one-unit items-scale AME (#381).

    The generalisation of :func:`prior_pushforward` past the binary-treatment
    families. ``term`` is a scalar coefficient in the ``prior`` group and the
    estimand is the average, over fitted rows, of the probability-scale change
    from a ``+1`` shift of whatever that coefficient multiplies — a ``+1 SD``
    association for the concurrent / adjusted / horseshoe predictors, a ``+1 SD``
    session-dose step for the dose companions.

    For a **binary** exposure the estimand is the toggle-everyone contrast rather
    than a one-unit shift on the rows that already carry it, so those families
    (ITT, gain-factor, aligned, block-exposure) use :func:`prior_pushforward`
    instead — it nets the contribution out per row and adds it back on *every*
    row, which is not the same average and must not be approximated by this.

    ``convention`` must match the transform the *same family's posterior* summary
    uses, or the prior and the estimate would be pushed through different
    functions and the check would not be a check of the reported quantity:

    - ``"net_out"`` forms the baseline ``η₀ = η − β`` and adds the contribution
      back, i.e. the effect of the unit already in the linear predictor. This is
      the :func:`_itt_ame_draws` convention.
    - ``"forward"`` shifts from the observed operating point, ``expit(η + β) −
      expit(η)`` — the effect of *one more* unit. This is what
      :func:`concurrent_marginals` and the dose marginal summary do.

    ``term_index`` selects one element of a vector-valued coefficient, e.g.
    ``{"predictor": "age_t1"}``.
    """
    prior = trace.prior
    da = prior[term]
    if term_index:
        da = da.sel(term_index)
    beta = da.stack(sample=("chain", "draw")).values.ravel()  # (S,)
    eta = (
        prior[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    if row_mask is not None:
        eta = eta[np.asarray(row_mask)]
    delta = beta[None, :]  # (1, S) — a +1 shift on every retained row
    base = eta - delta if convention == "net_out" else eta
    # Map both operating points through the fitted score mean before differencing:
    # under a non-identity link the response-scale change is not the logit-scale one
    # rescaled, so the prior check must sit on the same scale as the posterior
    # marginal it is compared against (#619).
    ame_prob = (
        apply_score_mean_link(expit(base + delta), score_mean_link)
        - apply_score_mean_link(expit(base), score_mean_link)
    ).mean(axis=0)  # (S,)
    return pushforward_values(
        beta, ame_prob * float(n_trials), n_trials=n_trials, ci_prob=ci_prob
    )


def joint_prior_pushforward(
    trace: xr.DataTree,
    *,
    outcomes: Sequence[str],
    G: np.ndarray,
    n_trials: Mapping[str, int],
    ci_prob: float = 0.95,
    row_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """One labelled prior-pushforward row per outcome of a joint ITT fit (#381).

    The joint family stores ``tau`` and ``eta`` on a labelled ``outcome``
    dimension and each outcome has its own item denominator, so a single row
    cannot describe the fit. Runs :func:`_joint_ame_draws` — the same core the
    posterior :func:`joint_treatment_marginals` uses — on the ``prior`` group, and
    returns rows already labelled per outcome.
    """
    coefs, ames = _joint_ame_draws(
        trace, outcomes, G=G, group="prior", row_mask=row_mask
    )
    rows: list[dict[str, Any]] = []
    for k, name in enumerate(str(o) for o in outcomes):
        n = int(n_trials[name])
        rows.append(
            labelled_pushforward(
                pushforward_values(
                    coefs[k], ames[k] * float(n), n_trials=n, ci_prob=ci_prob
                ),
                estimand=f"tau[{name}]",
                estimand_label=f"the treatment effect on {name}",
                role="causal",
            )
        )
    return rows


def indicator_prior_check(
    trace: xr.DataTree,
    *,
    nodes: Sequence[str],
    ci_prob: float = 0.89,
) -> pd.DataFrame:
    """Indicator-scale prior-predictive check for the measurement families (#381).

    The CFA families report loadings, communalities and factor correlations —
    none of which is an outcome-scale quantity, so the estimand pushforward the
    other families use has nothing to push. #381 asks for this instead, so they
    are not silently exempt from the coverage guarantee. The check is on the
    scale the model actually observes: the standardised indicator matrix.

    Standardisation is what makes it sharp. The indicators are z-scored by
    construction, so the observed SD is ~1 by definition and ``sd_ratio`` —
    prior-predictive SD over observed SD — has a *known reference value of one*
    rather than a judgement call. A ratio near 1 means the prior generates
    indicator data of the right scale; well above 1 means it spends most of its
    mass on configurations the standardisation makes impossible; **below 1 is the
    one that matters**, because a prior narrower than the data cannot generate
    what was observed and will fight the likelihood.

    ``coverage_90`` is the complementary view: the share of observed values
    inside the prior-predictive 90% band, pooled over that indicator's rows. A
    well-scaled prior covers essentially all of them.

    One row per **indicator**, pooled across nodes. The longitudinal model splits
    its observations into missingness-pattern blocks — one node each, sharing
    indicator labels — and some of those blocks hold a single row, whose
    within-block SD is identically zero. Grouping by node would emit dozens of
    unassessable rows for what is really one indicator observed in several
    pieces, so the pieces are pooled by label. Nodes absent from the trace are
    skipped rather than raising: this runs after the fit and must not be able to
    take a report down.
    """
    seen_obs: dict[str, list[np.ndarray]] = {}
    seen_sim: dict[str, list[np.ndarray]] = {}
    inside_flags: dict[str, list[np.ndarray]] = {}
    order: list[str] = []
    for node in nodes:
        try:
            pp = trace.prior_predictive[node]
            observed = np.asarray(trace.observed_data[node].values, dtype=float)
        except (AttributeError, KeyError):
            continue
        stacked = pp.stack(sample=("chain", "draw"))
        other = [d for d in stacked.dims if d != "sample"]
        draws = stacked.transpose(*other, "sample").values
        labels = (
            [str(x) for x in stacked.coords[other[-1]].values]
            if len(other) > 1
            else [node]
        )
        if draws.ndim == 2:  # a single-column node
            draws = draws[:, None, :]
            observed = observed.reshape(observed.shape[0], 1)
        for k, label in enumerate(labels):
            sim = draws[:, k, :]  # (obs, sample)
            obs = observed[:, k]
            finite = np.isfinite(obs)
            if not finite.any():
                continue
            if label not in seen_obs:
                order.append(label)
                seen_obs[label], seen_sim[label], inside_flags[label] = [], [], []
            seen_obs[label].append(obs[finite])
            seen_sim[label].append(sim.ravel())
            # Per-row bands, so coverage asks "was this child's value reachable",
            # not "is it inside the pooled marginal" — the pooled version would
            # flatter a prior that is wide overall but wrong row by row.
            band_lo = np.quantile(sim, 0.05, axis=1)
            band_hi = np.quantile(sim, 0.95, axis=1)
            inside_flags[label].append(
                (obs[finite] >= band_lo[finite]) & (obs[finite] <= band_hi[finite])
            )

    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    rows: list[dict[str, Any]] = []
    for label in order:
        seen = np.concatenate(seen_obs[label])
        flat = np.concatenate(seen_sim[label])
        inside = np.concatenate(inside_flags[label])
        obs_sd = float(np.std(seen)) if seen.size > 1 else float("nan")
        prior_sd = float(np.std(flat))
        ratio = prior_sd / obs_sd if obs_sd else float("nan")
        coverage = float(np.mean(inside)) if inside.size else float("nan")
        rows.append(
            {
                "indicator": label,
                "n_observed": int(seen.size),
                "observed_sd": obs_sd,
                "observed_lo": float(np.quantile(seen, lo_q)),
                "observed_hi": float(np.quantile(seen, hi_q)),
                "prior_sd": prior_sd,
                "prior_lo": float(np.quantile(flat, lo_q)),
                "prior_hi": float(np.quantile(flat, hi_q)),
                "sd_ratio": ratio,
                "coverage_90": coverage,
                "verdict": _indicator_prior_verdict(ratio, coverage, int(seen.size)),
            }
        )
    return pd.DataFrame(rows)


def _indicator_prior_verdict(sd_ratio: float, coverage: float, n: int) -> str:
    """Label one indicator's prior scale (see :func:`indicator_prior_check`).

    Ordered by which failure actually threatens a conclusion. A prior narrower
    than the standardised data — or one that demonstrably fails to cover it — is
    a real problem, because it cannot generate what was observed and will fight
    the likelihood. A prior several times wider is wasteful and worth knowing
    about, but it invalidates nothing.

    The coverage arm is judged against **binomial sampling noise**, not against a
    bare 0.90. With 75 children the standard error on a 90% coverage rate is
    about 3.5 points, so a flat ``coverage < 0.9`` rule fires on ordinary noise:
    it labelled three perfectly scaled ``rlm-mm-001`` indicators (``sd_ratio``
    1.007) "too tight" at coverage 0.88. Two standard errors below nominal is the
    threshold, so the flag means a real shortfall rather than a coin-flip.
    """
    if not np.isfinite(sd_ratio):
        return "not assessable"
    tolerance = 2.0 * float(np.sqrt(0.9 * 0.1 / n)) if n > 0 else 0.0
    if sd_ratio < 0.9 or (np.isfinite(coverage) and coverage < 0.9 - tolerance):
        return "too tight"
    if sd_ratio > 3.0:
        return "very loose"
    if sd_ratio > 1.5:
        return "loose"
    return "well scaled"


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


def ppc_interval_coverage_by_group(
    trace: xr.DataTree,
    *,
    node: str,
    group_labels: Sequence[str],
    group_name: str = "outcome",
    ci_levels: Sequence[float] = (0.5, 0.9),
) -> pd.DataFrame:
    """:func:`ppc_interval_coverage`, split by a label per flattened observation.

    A multi-outcome family flattens its cells into one likelihood node, so the
    pooled coverage statistic mixes tests with different denominators, floors and
    ceilings and weights them by their observed cell counts. That aggregate is
    well-defined but it can conceal outcome-specific miscalibration -- a badly
    fitted 6-item floored outcome is invisible beside a well-fitted 79-item one
    (2026-08-23 joint audit, lower-priority reporting correction).

    Returns the same schema as the pooled frame with an extra label column, so the
    per-group rows concatenate with it and ``ppc_coverage_markdown`` continues to
    read the pooled row it always did.
    """
    y_rep, y_obs = _ppc_node_arrays(trace, node)
    labels = np.asarray(group_labels).astype(str)
    if labels.shape[0] != y_obs.shape[0]:
        raise ValueError(
            f"group_labels has {labels.shape[0]} entries but {node!r} has "
            f"{y_obs.shape[0]} observations"
        )
    finite = np.isfinite(y_obs)
    y_rep, y_obs, labels = y_rep[finite], y_obs[finite], labels[finite]
    rows: list[dict[str, object]] = []
    for label in dict.fromkeys(labels):
        keep = labels == label
        subset_rep, subset_obs = y_rep[keep], y_obs[keep]
        n = int(subset_obs.shape[0])
        for p in ci_levels:
            lo = np.quantile(subset_rep, (1.0 - p) / 2.0, axis=1)
            hi = np.quantile(subset_rep, (1.0 + p) / 2.0, axis=1)
            n_in = int(np.count_nonzero((subset_obs >= lo) & (subset_obs <= hi)))
            rows.append(
                {
                    "mode": "count_interval",
                    "node": node,
                    "unit": "observations",
                    "quantity": "observed score",
                    group_name: str(label),
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
    # Per-group breakdown rows (``ppc_interval_coverage_by_group``) are a *split* of
    # the pooled rows, not extra observations, so summing them in as well would
    # double every total. They carry a group label the pooled rows leave null; drop
    # them here and let the family's own results partial render them
    # (2026-08-23 joint audit, lower-priority reporting correction). Frames without
    # the column are untouched, so stored fits render identically.
    for group_column in ("outcome", "measure"):
        if group_column in usable.columns:
            usable = usable[usable[group_column].isna()]
    if usable.empty:
        return ""
    # Pool across likelihood nodes by summing counts, never by averaging rates:
    # a family with one node per measure writes one row per (node, level), and
    # ``drop_duplicates`` would have rendered the first measure's coverage as if
    # it covered all of them (2026-08-21 historical-families review, finding 4).
    # With a single node this is the identity on the one row per level.
    d = (
        usable.groupby("level_pct", as_index=True)[["n_total", "n_inside"]]
        .sum()
        .assign(coverage=lambda f: f["n_inside"] / f["n_total"])
    )
    n_nodes = int(usable["node"].nunique()) if "node" in usable.columns else 1
    unit = str(usable["unit"].iloc[0])
    if n_nodes > 1:
        unit = f"{unit} across {n_nodes} measures"
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


def level_prior_pushforward(
    trace: xr.DataTree,
    *,
    phase: np.ndarray,
    G: np.ndarray,
    n_trials: int,
    ability: np.ndarray | None = None,
    ci_prob: float = 0.95,
    contrast_term: str = "b_grp_time",
    contrast_index: int | None = None,
    balance_term: str | None = None,
    score_mean_link: ScoreMeanLink = "logit",
) -> dict[str, float]:
    """Push the **prior** on the t2 group contrast through the items-scale AME (#389 finding 3).

    The estimand-scale prior-predictive check for the level family, the counterpart
    of :func:`prior_pushforward` for the ITT/gain families: what does the prior on
    the t2 group term imply for the items-scale average marginal effect? The prior
    on the contrast is data-free, but the level it is pushed through is not — the
    family anchors ``alpha`` on the pooled observed t1 logit — so this is an
    **empirical-Bayes, data-conditioned** prior check rather than a wholly pre-data
    one, and the report discloses that beside the number (#584 lower-severity 2). It runs :func:`level_t2_marginal_effect` on the persisted ``prior`` group,
    so the prior is pushed through the *same* t2 net-out transform as the posterior
    estimate (the level model's per-timepoint group vector + group x ability term
    means the plain ``eta - term*G`` ITT pushforward does not apply). Returns the same
    schema as :func:`prior_pushforward` so ``config.json`` and any consumer treat the
    two families' prior checks uniformly.
    """
    contrast_draws, ame_prob = level_t2_marginal_effect(
        trace,
        phase=phase,
        G=G,
        ability=ability,
        group="prior",
        contrast_term=contrast_term,
        contrast_index=contrast_index,
        balance_term=balance_term,
        score_mean_link=score_mean_link,
    )
    return pushforward_values(
        contrast_draws, ame_prob * float(n_trials), n_trials=n_trials, ci_prob=ci_prob
    )
