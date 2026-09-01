# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Joint-mechanism orchestration (``kind="joint_mechanism"``, #421 Tier 3).

``fit_joint_mechanism`` fits letter-sound knowledge jointly against word reading
and nonword decoding in one of two designs: a *levels* fit carrying a bivariate
residual dependence block on the observation row, and a *transition* fit over
period transitions. **Both** designs report ``rho_outcome`` — it is the
off-diagonal of whichever block the design carries — while only the levels design
adds the conditional slope and its ratio, because partialling the held-fixed
outcome is a same-row operation (2026-08-23 follow-up review, documentation gap
4). A wave that fails the prespecified row, per-outcome and overlap minima is not
fitted at all: a residual correlation from a handful of jointly observed children
would be prior-dominated. Nothing here is randomised: every slope is an adjusted
association.

Publication lifecycle (2026-08-23 follow-up review, finding 1). Every wave the
levels design publishes is fitted, convergence-scanned over its reported
deterministics as well as its free random variables, given the informative
new-child predictive check and power-scaling sensitivity, and persisted as a
named trace. One wave additionally hosts the fit-level artefacts (``trace.nc``,
``diagnostics_summary.json``), chosen by row count — an operational
artefact-hosting rule, never a scientific primary. No reporting path selects a
wave after seeing its posterior.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import COLOUR_BLUE
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    joint_mechanism as _joint_mechanism,
    reporting as _report,
)
from language_reading_predictors.statistical_models.adjustment import (
    effective_adjustment,
)
from language_reading_predictors.statistical_models.artifacts import (
    guard_optional,
    record_artifact,
    save_table,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.plotting import save_styled_figure
from language_reading_predictors.statistical_models.new_child_predictive import (
    NewChildPlan,
    write_new_child_validation,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset_prepared,
    load_and_prepare,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
)
from language_reading_predictors.statistical_models.release import (
    JOINT_MECHANISM_WAVE_MARGINAL_PPC as _JM_WAVE_MARGINAL_PPC,
    JOINT_MECHANISM_WAVE_PSENSE as _JM_WAVE_PSENSE,
    JOINT_MECHANISM_WAVE_TRACE as _JM_WAVE_TRACE,
)
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import PrimaryFitPlan
from language_reading_predictors.statistical_models.subfits import run_subfit
from language_reading_predictors.statistical_models.invariants import (
    require_value,
)


#: Terms reported for both designs, in report order. ``rho_outcome`` is emitted by
#: **both** designs — it is the off-diagonal of whichever dependence block the design
#: carries — while the two conditional-slope terms exist only where that block sits on
#: the observation row (the ``levels`` design), so those are emitted conditionally.
#: ``share_retained`` keeps its machine key for continuity, but its published label is
#: the conditional-to-marginal slope *ratio*: it is unbounded, can be negative or
#: exceed one, and is not a mediated share (2026-08-23 follow-up review, finding 5).
_JM_TERM_LABELS: dict[str, str] = {
    # 2026-08-23 joint audit, finding 4, and the #591 follow-up review, finding 3.
    # "Decoding specificity" is a construct-level claim this model does not identify:
    # W and N differ in item count (79 vs 6), score distribution, discrimination,
    # reliability and floor/ceiling behaviour, and nothing here calibrates them to a
    # common latent outcome scale. A single common ability loading differently on the
    # two tests produces a non-zero slope contrast on its own. What IS identified is
    # an operational contrast between two adjusted test-score associations, so that
    # is what the label says.
    "delta_ls_decoding": (
        "Delta = beta(LS->N) - beta(LS->W): operational test-score slope contrast "
        "(logit per SD), not a construct-level decoding-specificity measure"
    ),
    "rho_outcome": "residual correlation between the two outcomes",
    "beta_held_on_focal": "implied coefficient of the held-fixed outcome",
    "beta_mech_focal_given_held": "beta(LS->W) holding latent nonword decoding fixed",
    # Likewise a ratio of two adjusted associations, not a mediation proportion or
    # a causal path fraction — and unbounded, so it is governed rather than read off
    # its median (#591 follow-up review, finding 5).
    "share_retained": (
        "ratio of adjusted associations: beta(LS->W) holding latent decoding fixed, "
        "over the unconditional beta(LS->W) (unbounded; not a mediated share)"
    ),
    "abs_slope_reduction": (
        "absolute reduction in beta(LS->W) when latent decoding is held fixed "
        "(logit per SD) - the denominator-free companion to the ratio"
    ),
}

#: The one reported term that is a ratio of posterior quantities. Its mean is never
#: published (a ratio's mean is dominated by draws where the denominator is small),
#: and it carries the governance table below rather than being read off its median.
_JM_RATIO_TERM = "share_retained"

#: Denominator of that ratio, and the residual scale that divides the conditional
#: slope through ``rho * sigma_focal / sigma_held``.
_JM_RATIO_DENOMINATOR = "beta_mech"

#: Deliberately small logit-scale identifiability threshold for the ``share_retained``
#: ratio's two instability routes (2026-08-23 joint audit, finding 10), matching the
#: historical-joint residual-scale rule's 0.05-logit convention. A ratio is reported
#: only when the posterior supports, with at least ``_JM_STABILITY_SUPPORT``
#: probability, BOTH that its denominator ``beta_mech[focal]`` is away from zero and
#: that the held-fixed outcome's residual scale ``sigma_u_resid[held]`` is away from
#: zero (it divides the conditional slope). Neither is a minimum-important-effect
#: threshold. A finite Monte Carlo mean over a heavy-tailed ratio looks reassuring
#: precisely when the quantity is least meaningful, which is why the mean is
#: withheld for this term regardless. This is the family's ONE stability rule: the
#: ``conditional_slope_ratio.csv`` governance table reports the same verdict rather
#: than applying a second, competing one (#591 follow-up review, finding 5).
_JM_RATIO_MIN_ABS = 0.05
_JM_STABILITY_SUPPORT = 0.95

#: Columns every ``joint_mechanism_slopes.csv`` row carries. The house standard is
#: median + inner 50% + outer 89% + tail probability (METHODS.md; #421 acceptance
#: criterion), so the inner interval is part of the contract, not an extra.
_JM_SLOPE_REQUIRED: frozenset[str] = frozenset(
    {
        "wave",
        "term",
        "label",
        "median",
        "mean",
        "lo50",
        "hi50",
        "lo",
        "hi",
        "prob_pos",
        "converged",
    }
)


def _jm_draws(trace, name: str, *, outcome: str | None = None) -> np.ndarray | None:
    """Flattened posterior draws for one term, or ``None`` when it is absent."""
    if name not in trace.posterior:
        return None
    var = trace.posterior[name]
    if outcome is not None:
        var = var.sel(outcome=outcome)
    return np.asarray(var.values).ravel()


def _jm_term_summary(
    trace,
    name: str,
    ci_prob: float,
    *,
    outcome: str | None = None,
) -> dict | None:
    """Median / mean / inner-50% / outer-``ci_prob`` / P(>0) for one posterior term.

    Returns ``None`` when the variable is absent, so a design that does not register
    a term (the transition design has no conditional slope) simply omits its rows
    instead of failing.

    The ratio term's mean is blanked by its caller rather than here, so there is one
    place that decides what a ratio may publish.
    """
    draws = _jm_draws(trace, name, outcome=outcome)
    if draws is None:
        return None
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    return {
        "median": float(np.median(draws)),
        "mean": float(np.mean(draws)),
        "lo50": float(np.quantile(draws, 0.25)),
        "hi50": float(np.quantile(draws, 0.75)),
        "lo": float(np.quantile(draws, lo_q)),
        "hi": float(np.quantile(draws, hi_q)),
        "prob_pos": float(np.mean(draws > 0)),
    }


def _jm_ratio_governance_row(
    trace,
    *,
    ci_prob: float,
    wave: str,
    contrast: tuple[str, str],
    converged: bool | None,
) -> dict | None:
    """One row of ``conditional_slope_ratio.csv``: the ratio and its stability verdict.

    The conditional-to-marginal slope ratio is unbounded. Under suppression or a sign
    reversal it is negative; under amplification it exceeds one; and as either its
    denominator or the held-fixed outcome's residual scale approaches zero it is not
    summarisable at all. Classifying its median against 0.5 — which the report did —
    reads a negative ratio as "most of the association runs through decoding", a
    mediation claim this observational model does not identify (#591 follow-up
    review, finding 5).

    So the ratio is published with the probability mass in the three regions that
    mean different things — ``prob_lt_0`` (suppression), ``prob_in_unit`` (ordinary
    attenuation) and ``prob_gt_1`` (amplification) — rather than as a median to be
    classified. The stability verdict is **not** a second rule: it is
    :func:`_jm_ratio_stability`'s, reproduced here with its two probabilities so the
    governance table and the slope table can never disagree about whether a wave's
    ratio is reportable.

    Returns ``None`` for a design that registers no ratio.
    """
    ratio = _jm_draws(trace, _JM_RATIO_TERM)
    focal = contrast[1]
    denominator = _jm_draws(trace, _JM_RATIO_DENOMINATOR, outcome=focal)
    stability = _jm_ratio_stability(trace, contrast=contrast)
    if ratio is None or denominator is None or stability is None:
        return None
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    return {
        "wave": wave,
        "term": _JM_RATIO_TERM,
        "label": _JM_TERM_LABELS[_JM_RATIO_TERM],
        "focal_outcome": focal,
        "held_outcome": contrast[0],
        "median": float(np.median(ratio)),
        "lo50": float(np.quantile(ratio, 0.25)),
        "hi50": float(np.quantile(ratio, 0.75)),
        "lo": float(np.quantile(ratio, lo_q)),
        "hi": float(np.quantile(ratio, hi_q)),
        "prob_lt_0": float(np.mean(ratio < 0.0)),
        "prob_in_unit": float(np.mean((ratio >= 0.0) & (ratio <= 1.0))),
        "prob_gt_1": float(np.mean(ratio > 1.0)),
        "denominator_lo": float(np.quantile(denominator, lo_q)),
        "denominator_hi": float(np.quantile(denominator, hi_q)),
        **stability,
        "converged": converged,
    }


def _jm_slope_rows(
    trace,
    *,
    outcome_symbols: tuple[str, ...],
    contrast: tuple[str, str],
    ci_prob: float,
    wave: str,
    converged: bool | None,
) -> list[dict]:
    """One fit's reported terms as ``joint_mechanism_slopes.csv`` rows."""
    rows: list[dict] = []

    def _add(term: str, label: str, summary: dict | None) -> None:
        if summary is None:
            return
        rows.append(
            {
                "wave": wave,
                "term": term,
                "label": label,
                **summary,
                "converged": converged,
            }
        )

    for sym in outcome_symbols:
        _add(
            f"beta_mech[{sym}]",
            f"letter-sound slope on {sym} (logit per SD)",
            _jm_term_summary(trace, "beta_mech", ci_prob, outcome=sym),
        )
    delta_label = _JM_TERM_LABELS["delta_ls_decoding"].replace(
        "beta(LS->N) - beta(LS->W)", f"beta(LS->{contrast[0]}) - beta(LS->{contrast[1]})"
    )
    _add("delta_ls_decoding", delta_label, _jm_term_summary(trace, "delta_ls_decoding", ci_prob))
    for term in (
        "rho_outcome",
        "beta_held_on_focal",
        "beta_mech_focal_given_held",
    ):
        _add(term, _JM_TERM_LABELS[term], _jm_term_summary(trace, term, ci_prob))

    # The ratio and its denominator-free companion, gated by the pre-specified
    # stability rule (2026-08-23 joint audit, finding 10).
    stability = _jm_ratio_stability(trace, contrast=contrast)
    share = _jm_term_summary(trace, "share_retained", ci_prob)
    if share is not None:
        # The posterior mean of a ratio with a near-zero denominator is a property
        # of the sampled draws, not of the quantity. Withheld unconditionally.
        share["mean"] = float("nan")
        if stability is not None and not stability["share_retained_stable"]:
            # Undefined, not merely uncertain: blank the summary rather than print
            # numbers a reader would take as an estimate.
            for key in ("median", "lo50", "hi50", "lo", "hi", "prob_pos"):
                share[key] = float("nan")
        _add("share_retained", _JM_TERM_LABELS["share_retained"], share)
    _add(
        "abs_slope_reduction",
        _JM_TERM_LABELS["abs_slope_reduction"],
        _jm_abs_slope_reduction(trace, ci_prob, contrast=contrast),
    )
    if stability is not None:
        for row in rows:
            if row["term"] in ("share_retained", "abs_slope_reduction"):
                row.update(stability)
    return rows


def _jm_ratio_stability(trace, *, contrast: tuple[str, str]) -> dict | None:
    """Pre-specified stability rule for the ``share_retained`` ratio.

    Two instability routes, both checked on the posterior rather than eyeballed:
    the denominator ``beta_mech[focal]`` near zero, and the held-fixed outcome's
    residual scale ``sigma_u_resid[held]`` near zero, which divides the conditional
    slope through ``rho * sigma_focal / sigma_held``. Either makes the ratio
    heavy-tailed, so a finite Monte Carlo summary can look reassuring while the
    quantity has no stable value (2026-08-23 joint audit, finding 10).

    ``None`` when the fit registers no ratio, so the transition design is untouched.
    """
    posterior = trace.posterior
    if "share_retained" not in posterior or "beta_mech" not in posterior:
        return None
    held, focal = contrast
    denominator = np.abs(np.asarray(posterior["beta_mech"].sel(outcome=focal).values))
    prob_denominator = float(np.mean(denominator > _JM_RATIO_MIN_ABS))
    prob_scale = float("nan")
    if "sigma_u_resid" in posterior:
        scale = np.asarray(posterior["sigma_u_resid"].sel(outcome=held).values)
        prob_scale = float(np.mean(scale > _JM_RATIO_MIN_ABS))
    stable = prob_denominator >= _JM_STABILITY_SUPPORT and (
        not np.isfinite(prob_scale) or prob_scale >= _JM_STABILITY_SUPPORT
    )
    return {
        "ratio_min_abs": _JM_RATIO_MIN_ABS,
        "prob_denominator_above_minimum": prob_denominator,
        "prob_held_scale_above_minimum": prob_scale,
        "share_retained_stable": bool(stable),
    }


def _jm_abs_slope_reduction(
    trace, ci_prob: float, *, contrast: tuple[str, str]
) -> dict | None:
    """Per-draw ``|beta_focal| - |beta_focal_given_held|`` on the logit-per-SD scale.

    The denominator-free reading of the same comparison: how much of the
    letter-sound association with the focal outcome disappears when the latent
    held-fixed skill is partialled out. Reportable whether or not the ratio is
    stable, which is why the ratio's failure state has something to fall back on
    (2026-08-23 joint audit, finding 10). Taken from the two slopes directly rather
    than reconstructed through the ratio, so it inherits none of the ratio's
    near-zero-denominator behaviour.
    """
    posterior = trace.posterior
    if (
        "beta_mech_focal_given_held" not in posterior
        or "beta_mech" not in posterior
    ):
        return None
    focal = contrast[1]
    conditional = np.asarray(posterior["beta_mech_focal_given_held"].values).ravel()
    unconditional = np.asarray(
        posterior["beta_mech"].sel(outcome=focal).values
    ).ravel()
    if conditional.shape != unconditional.shape:
        return None
    draws = np.abs(unconditional) - np.abs(conditional)
    draws = draws[np.isfinite(draws)]
    if not draws.size:
        return None
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    return {
        "median": float(np.median(draws)),
        "mean": float(np.mean(draws)),
        "lo50": float(np.quantile(draws, 0.25)),
        "hi50": float(np.quantile(draws, 0.75)),
        "lo": float(np.quantile(draws, lo_q)),
        "hi": float(np.quantile(draws, hi_q)),
        "prob_pos": float(np.mean(draws > 0)),
    }


def _jm_marginal_ppc(
    ctx: StatisticalFitContext,
    *,
    outcome_symbols: tuple[str, ...],
    ci_levels: tuple[float, ...] = (0.5, 0.9),
    trace=None,
    prepared=None,
    name: str = "ppc_summary_marginal",
) -> pd.DataFrame | None:
    """New-child predictive coverage, written to ``ppc_summary_marginal.csv``.

    Why this exists. The levels design carries one bivariate latent residual **per
    child**, and each child contributes exactly two likelihood cells — so the residual
    block is saturated and the ordinary posterior predictive, which conditions on the
    fitted residuals, reproduces every observation. The dev fit's conditional coverage
    is 1.00 at both the 50% and 90% levels with intervals only ~8 items wide on a
    79-item test: not a good fit, an arithmetic consequence of the design. Publishing
    that number alone would read as a passed calibration check.

    This redraws each child's residual from its **estimated** distribution
    (Sigma = diag(sigma) Corr diag(sigma), Cholesky-factorised in closed form for the
    2x2 case) instead of reusing the fitted value, giving the predictive for a *new*
    child with the same covariates — the check that can actually fail. Both files are
    written: ``ppc_summary.csv`` keeps the house cross-family schema, and this is the
    informative companion the results partial reads alongside it.

    Done in NumPy rather than via ``pm.sample_posterior_predictive(var_names=[...])``
    deliberately: naming the residual there still conditioned on its posterior values,
    silently reproducing the conditional numbers under a marginal label.

    ``trace`` / ``prepared`` / ``name`` let a non-anchor wave write its own copy from
    its own persisted posterior, which is what makes every published wave carry the
    check rather than only the artefact-hosting one. Coverage is written pooled and
    **per outcome**, and the table is a *required* artefact.
    """
    trace = ctx.trace if trace is None else trace
    prepared = ctx.prepared if prepared is None else prepared
    written: list[pd.DataFrame] = []
    with guard_optional(
        ctx, f"{name}.csv",
        filename=f"{name}.csv", kind="table", verb="skipped",
    ):
        post = trace.posterior
        required = {"eta", "u_resid", "sigma_u_resid", "rho_outcome"}
        if not required <= set(post.data_vars):
            return None
        # Linear predictor with the fitted residual removed.
        eta = np.asarray(post["eta"].values)
        core = eta - np.asarray(post["u_resid"].values)
        n_obs, n_outcomes = core.shape[-2], core.shape[-1]
        core = core.reshape(-1, n_obs, n_outcomes)
        sigma = np.asarray(post["sigma_u_resid"].values).reshape(-1, n_outcomes)
        rho = np.asarray(post["rho_outcome"].values).reshape(-1)
        if n_outcomes != 2:
            return None

        rng = np.random.default_rng(ctx.sampling.random_seed)
        z = rng.standard_normal((core.shape[0], n_obs, 2))
        # L = [[s0, 0], [s1 rho, s1 sqrt(1 - rho^2)]]; u = L z.
        tail = np.sqrt(np.clip(1.0 - rho**2, 0.0, None))
        u_new = np.stack(
            [
                sigma[:, None, 0] * z[..., 0],
                sigma[:, None, 1]
                * (rho[:, None] * z[..., 0] + tail[:, None] * z[..., 1]),
            ],
            axis=-1,
        )
        p = 1.0 / (1.0 + np.exp(-(core + u_new)))

        row = np.asarray(trace.constant_data["y_post_cell_row"].values).astype(int)
        col = np.asarray(
            trace.constant_data["y_post_cell_outcome"].values
        ).astype(int)
        n_trials = np.array(
            [prepared.n_trials[s] for s in outcome_symbols], dtype=int
        )[col]
        y_rep = rng.binomial(n_trials[None, :], p[:, row, col])  # (draw, cell)

        y_obs = np.asarray(trace.observed_data["y_post"].values, dtype=float)
        finite = np.isfinite(y_obs)
        y_rep, y_obs, col = y_rep[:, finite], y_obs[finite], col[finite]
        # Pooled *and* per-outcome coverage. The two outcomes sit on incompatible
        # denominators (79 items against 6) and the 6-item one is heavily floored,
        # so a pooled figure can hide a badly calibrated leg behind a well
        # calibrated one (#591 follow-up review, robustness gap 2). The pooled row
        # leaves ``outcome`` null and the per-outcome rows carry the symbol — the
        # same convention ``ppc_interval_coverage_by_group`` uses for the
        # conditional table, so a reader filtering one file's split filters the
        # other's identically.
        groups: list[tuple[object, str, np.ndarray]] = [
            (None, "observations", np.ones(y_obs.shape[0], dtype=bool))
        ]
        groups += [
            (symbol, f"observations ({symbol})", col == index)
            for index, symbol in enumerate(outcome_symbols)
        ]
        rows = []
        for level in ci_levels:
            lo = np.quantile(y_rep, (1.0 - level) / 2.0, axis=0)
            hi = np.quantile(y_rep, (1.0 + level) / 2.0, axis=0)
            inside = (y_obs >= lo) & (y_obs <= hi)
            for outcome, unit, mask in groups:
                n = int(np.count_nonzero(mask))
                n_in = int(np.count_nonzero(inside & mask))
                rows.append(
                    {
                        "mode": "count_interval_marginal",
                        "node": "y_post",
                        "outcome": outcome,
                        "unit": unit,
                        "quantity": "observed score (new-child predictive)",
                        "level": float(level),
                        "level_pct": int(round(level * 100)),
                        "n_total": n,
                        "n_inside": n_in,
                        "coverage": float(n_in / n) if n else float("nan"),
                    }
                )
        frame = pd.DataFrame(rows)
        # Required, not optional: the ordinary conditional check is saturated by
        # construction and PSIS-LOO is deliberately not computed, so this is the
        # levels design's only informative predictive check. ``release`` fails a
        # wave whose file is absent (2026-08-23 follow-up review, robustness gap 2).
        save_table(ctx, name, frame, required=True)
        written.append(frame)
    return written[0] if written else None


def _jm_cell_outcome_labels(
    ctx: StatisticalFitContext, outcome_symbols: tuple[str, ...]
) -> list[str] | None:
    """One outcome symbol per flattened ``y_post`` cell, from the saved cell map.

    ``None`` when the map is absent or does not align, so a per-outcome coverage
    split is skipped rather than misaligning a measure with another's counts.
    """
    constant = getattr(ctx.trace, "constant_data", None)
    if constant is None or "y_post_cell_outcome" not in constant:
        return None
    index = np.asarray(constant["y_post_cell_outcome"].values).astype(int)
    if index.size == 0 or int(index.max()) >= len(outcome_symbols):
        return None
    return [outcome_symbols[i] for i in index]


def _jm_primary_fit_plan(
    *,
    outcome_symbols: tuple[str, ...],
    diag_vars: list[str],
    psense_vars: list[str],
    new_child_plan: NewChildPlan,
    marginal_ppc: bool = False,
    compute_loo: bool = True,
) -> PrimaryFitPlan:
    """Declare the joint-mechanism primary lifecycle and custom artefacts.

    The shared runner owns the invariant ordering. These hooks retain the family's
    per-outcome prior/PPC/LOO-PIT views, optional new-child marginal coverage and
    reported-slope power-scaling set.

    The last three were silently absent before the #427 review: the per-outcome
    LOO-PIT calls raised inside :func:`diagnostics.save_joint_loo_pit_plot` because
    the helper hard-required ``posterior['tau']``, which this family does not have.
    ``posterior_var="beta_mech"`` now names the family's own reported coefficient.
    With ``compute_loo=False`` (the saturated levels design) both PSIS-based
    artefacts — LOO and LOO-PIT — are skipped, and the density groups psense needs
    are attached directly instead.
    """
    def _plot_prior(c: StatisticalFitContext) -> None:
        for index, symbol in enumerate(outcome_symbols):
            stem = (
                "prior_predictive_check"
                if index == 0
                else f"prior_predictive_check_{symbol.lower()}"
            )
            _diag.save_prior_predictive_plot(c, symbol, filename_stem=stem)

    def _run_ppc(c: StatisticalFitContext) -> None:
        _diag.sample_posterior_predictive(c, var_names=["y_post"])
        for index, symbol in enumerate(outcome_symbols):
            stem = (
                "posterior_predictive_check"
                if index == 0
                else f"posterior_predictive_check_{symbol.lower()}"
            )
            _diag.save_joint_posterior_predictive_plot(
                c, symbol, filename_stem=stem
            )
        # Coverage is denominator-agnostic for flattened child x outcome cells, but
        # pooling W and N hides outcome-specific miscalibration and weights the two
        # by their observed cell counts -- and these outcomes differ sharply (79
        # items versus a 6-item count floored for 40-72% of children). Publish
        # per-outcome rows and keep the pooled row as the secondary summary the
        # shared coverage sentence reads (2026-08-23 joint audit, lower-priority
        # reporting correction).
        with guard_optional(
            c, "ppc_summary.csv", filename="ppc_summary.csv", kind="table"
        ):
            coverage = _report.ppc_interval_coverage(c.trace, node="y_post")
            frames = [coverage]
            labels = _jm_cell_outcome_labels(c, outcome_symbols)
            if labels is not None:
                frames.append(
                    _report.ppc_interval_coverage_by_group(
                        c.trace, node="y_post", group_labels=labels
                    )
                )
            save_table(
                c,
                "ppc_summary",
                pd.concat(frames, ignore_index=True),
                required=False,
            )
        if marginal_ppc:
            _jm_marginal_ppc(c, outcome_symbols=outcome_symbols)

    def _write_loo_pit(c: StatisticalFitContext) -> None:
        # The generic LOO-PIT would pool tests with different denominators. Note the
        # leave-out unit: this subsets one outcome's flattened cells and keeps no
        # child map, so it leaves out one *cell* while the same child's other
        # transitions, other outcome and fitted intercept stay in — a conditional
        # check, not the leave-one-child-out target the main PSIS-LOO declares. The
        # figure title says so (2026-08-23 follow-up review, finding 4).
        for index, symbol in enumerate(outcome_symbols):
            stem = "loo_pit" if index == 0 else f"loo_pit_{symbol.lower()}"
            _diag.save_joint_loo_pit_plot(
                c,
                symbol,
                filename_stem=stem,
                posterior_var="beta_mech",
            )

    def _validate_new_child(c: StatisticalFitContext) -> None:
        """The validation that *does* match the declared child unit (#626).

        The child's own dependence-block residual is integrated out rather than left
        at its fitted value, so the ELPD and the PIT beside it answer the new-child
        question the ``loo_unit="child"`` declaration has always claimed.

        Run for **both** designs, unlike the LOO-PIT above. The levels design's
        saturated per-child residual is why it computes no conditional PSIS-LOO — and
        it is precisely what this integrates away, so gating this on ``compute_loo``
        would withhold the validation from the design that most needs it.
        """
        write_new_child_validation(c, new_child_plan)

    def _density_groups_for_psense(c: StatisticalFitContext) -> None:
        # The levels design computes no PSIS-LOO (the saturated per-child residual
        # makes it fail its Pareto-k diagnostics en masse; see the run plan), but
        # power scaling still needs the log_likelihood / log_prior groups the LOO
        # step would otherwise have attached — the same direct route the
        # no-LOO mediation families take (#381).
        _diag.compute_log_likelihood_and_prior(c, strict=False)

    def _post_sampling(c: StatisticalFitContext) -> None:
        """Attach psense's density groups where LOO did not, then validate.

        Both designs validate. Only the levels design needs the groups attached
        directly, because it is the one that computes no PSIS-LOO.
        """
        if not compute_loo:
            _density_groups_for_psense(c)
        _validate_new_child(c)

    return PrimaryFitPlan(
        diagnostic_vars=tuple(diag_vars),
        plot_prior_predictive=_plot_prior,
        post_sampling_audit=_post_sampling,
        custom_posterior_predictive=_run_ppc,
        psense_vars=tuple(psense_vars),
        extended_term="delta_ls_decoding",
        include_loo_pit=False,
        # LOO-PIT is importance-sampling-based, so it shares PSIS-LOO's
        # saturation failure in the levels design and is only written where LOO
        # itself is computed.
        post_extended_audit=_write_loo_pit if compute_loo else None,
        compute_loo=compute_loo,
    )


def _jm_write_slopes(
    ctx: StatisticalFitContext,
    rows: list[dict],
    *,
    contrast: tuple[str, str],
) -> pd.DataFrame:
    """Validate and write ``joint_mechanism_slopes.csv``."""
    df = pd.DataFrame(rows)
    missing = _JM_SLOPE_REQUIRED.difference(df.columns)
    if missing:
        raise ValueError(
            f"joint_mechanism_slopes is missing required columns: {sorted(missing)}"
        )
    if not (df["term"] == "delta_ls_decoding").any():
        raise ValueError("joint_mechanism_slopes has no delta_ls_decoding row")
    save_table(ctx, "joint_mechanism_slopes", df)
    print_table(
        metrics_table(
            df.to_dict("records"),
            title=(
                f"Letter-sound slopes + identified {contrast[0]}-{contrast[1]} contrast "
                f"(median, inner 50%, outer {int(ctx.reporting.ci_prob * 100)}%, "
                "equal-tailed)"
            ),
            columns=["wave", "term", "median", "lo50", "hi50", "lo", "hi", "prob_pos"],
        )
    )
    return df


def fit_joint_mechanism(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Fit the bivariate joint mechanism (#421 Tier 3 (1)) in one of two designs.

    ``design="levels"`` is the per-wave levels/concurrent ``{W, N}`` model the issue
    specifies: one fit per timepoint, each with an LKJ residual correlation between
    the two outcomes, reporting both the identified decoding-specificity contrast
    ``delta_ls_decoding`` **and** the identified ``share_retained`` — the within-model
    replacement for the paired-draws ratio the ``ca-010`` / ``ca-011`` pair can only
    approximate. ``design="transition"`` is the phase-stacked ANCOVA companion matched
    to ``mech-096`` / ``mech-101``, which re-reports the Tier-1 Delta on the same
    parameterisation it was originally computed on, with a bivariate child random
    intercept supplying the cross-outcome covariance.
    """
    require_spec(spec, "joint_mechanism")
    plan = _joint_mechanism.resolve_joint_mechanism_run_plan(spec)
    if plan.design == "levels":
        return _fit_joint_mechanism_levels(spec, config, plan)
    return _fit_joint_mechanism_transition(spec, config, plan)


#: Deterministics the levels design publishes and therefore convergence-scans. They
#: are functions of free variables, but a ratio or a correlation can mix far worse
#: than its arguments, so a verdict that omits them does not cover what is published.
_JM_REPORTED_DETERMINISTICS: tuple[str, ...] = (
    "delta_ls_decoding",
    "rho_outcome",
    "beta_held_on_focal",
    "beta_mech_focal_given_held",
    "share_retained",
)


def _jm_wave_eligibility(
    sub,
    *,
    plan: _joint_mechanism.JointMechanismRunPlan,
    outcome_symbols: tuple[str, ...],
    timepoint: int,
) -> dict:
    """Whether one wave clears the prespecified minima, and the counts behind that.

    Three floors, not one. The union count (exposure plus *at least one* outcome)
    bounds neither leg's own sample nor the jointly observed pairs — and it is the
    jointly observed pairs that identify the residual correlation and, through it, the
    conditional slope and its ratio. A wave could clear a union floor while one
    outcome is nearly absent, and would then publish a prior-dominated ``rho_outcome``
    (2026-08-23 follow-up review, robustness gap 1).

    The recorded counts are also the *wave-specific* eligibility ledger. A wave subset
    inherits ``prepared.dropped_rows`` from the four-timepoint panel unchanged, so the
    stored figure is neither the panel's nor this wave's (metadata gap 3).
    """
    # Narrowed here rather than asserted: each is used further down through
    # ``plan``, and ``assert`` would have vanished under ``-O`` (#637 stage 4).
    require_value(plan.min_wave_rows, "min_wave_rows")
    require_value(plan.min_wave_outcome_rows, "min_wave_outcome_rows")
    require_value(plan.min_wave_overlap_rows, "min_wave_overlap_rows")
    usable = ~np.isnan(sub.post_counts[plan.mechanism_symbol])
    observed = {s: ~np.isnan(sub.post_counts[s]) for s in outcome_symbols}
    any_outcome = np.zeros(sub.n_obs, dtype=bool)
    both_outcomes = np.ones(sub.n_obs, dtype=bool)
    for symbol in outcome_symbols:
        any_outcome |= observed[symbol]
        both_outcomes &= observed[symbol]
    n_usable = int(np.count_nonzero(usable & any_outcome))
    per_outcome = {
        symbol: int(np.count_nonzero(usable & observed[symbol]))
        for symbol in outcome_symbols
    }
    n_overlap = int(np.count_nonzero(usable & both_outcomes))
    reasons: list[str] = []
    if n_usable < plan.min_wave_rows:
        reasons.append(f"{n_usable} usable rows < {plan.min_wave_rows}")
    for symbol, count in per_outcome.items():
        if count < plan.min_wave_outcome_rows:
            reasons.append(f"{count} {symbol} cells < {plan.min_wave_outcome_rows}")
    if n_overlap < plan.min_wave_overlap_rows:
        reasons.append(
            f"{n_overlap} jointly observed rows < {plan.min_wave_overlap_rows}"
        )
    return {
        "wave": f"t{timepoint}",
        "timepoint": timepoint,
        "panel_rows_at_wave": int(sub.n_obs),
        "usable_rows": n_usable,
        **{f"cells_{symbol}": count for symbol, count in per_outcome.items()},
        "jointly_observed_rows": n_overlap,
        "wave_eligibility_dropped": int(sub.n_obs) - n_usable,
        "fitted": not reasons,
        "skipped_because": "; ".join(reasons),
    }


def _jm_reported_deterministics(built) -> list[str]:
    """The reported deterministics a built joint-mechanism model actually registers."""
    names = set(built.model.named_vars)
    return [name for name in _JM_REPORTED_DETERMINISTICS if name in names]


def _jm_exposure_logit_sd(built, mechanism_symbol: str) -> float | None:
    """SD of the exposure logit on the rows the factory standardised it over.

    One standard deviation is the unit both slopes are reported in, and the levels
    design re-standardises within each wave — so a cross-wave coefficient range does
    not denote a fixed raw letter-sound increment unless this number is published
    beside it (2026-08-23 follow-up review, robustness gap 8). It is also what makes
    the joint-versus-marginal comparison auditable (finding 2).
    """
    from language_reading_predictors.statistical_models.preprocessing import logit_safe

    prepared = built.prepared
    if mechanism_symbol not in prepared.post_counts:
        return None
    values = logit_safe(
        prepared.post_counts[mechanism_symbol], prepared.n_trials[mechanism_symbol]
    )
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    return float(np.std(values, ddof=1))


def _jm_copy_wave_artifacts(ctx: StatisticalFitContext, *, timepoint: int) -> None:
    """Re-file the artefact-hosting wave's predictive and psense tables per wave.

    The shared runner writes ``ppc_summary_marginal.csv`` and ``psense_summary.csv``
    under their house names. Copying them to the per-wave names means the release
    check requires one uniform bundle for *every* published wave instead of special
    casing the wave that happens to host the fit-level artefacts — and it costs no
    second computation, because both are re-filings of frames already in hand.
    """
    for source, target in (
        ("ppc_summary_marginal", _JM_WAVE_MARGINAL_PPC.format(timepoint=timepoint)),
        ("psense_summary", f"{_JM_WAVE_PSENSE.format(timepoint=timepoint)}_summary"),
    ):
        frame = ctx.tables.get(source)
        if frame is None:
            continue
        save_table(
            ctx, target, frame, index=source == "psense_summary", register=False
        )


def _jm_wave_psense(
    ctx: StatisticalFitContext,
    built,
    trace,
    *,
    plan: _joint_mechanism.JointMechanismRunPlan,
    timepoint: int,
) -> None:
    """Power-scaling sensitivity for one non-hosting wave, from its own posterior.

    Power scaling is importance reweighting over draws already in hand, so it needs
    only the ``log_likelihood`` / ``log_prior`` groups — which a sub-fit trace does
    not carry until they are attached here. Guarded: psense is a secondary diagnostic
    and must not cost a wave its fit, but its absence *is* checked at release, so a
    silent skip cannot pass as a recorded result.
    """
    stem = _JM_WAVE_PSENSE.format(timepoint=timepoint)
    with guard_optional(
        ctx, f"{stem}_summary.csv", filename=f"{stem}_summary.csv", kind="table"
    ):
        names = {rv.name for rv in built.model.free_RVs} | set(built.model.named_vars)
        var_names = plan.psense_vars(names)
        if not var_names:
            return
        with_densities = _diag.attach_log_densities(trace, built.model, strict=False)
        frame = _diag.psense_artifacts(
            with_densities, ctx.output_dir, var_names, stem=stem
        )
        if frame is not None:
            record_artifact(
                ctx, stem, filename=f"{stem}_summary.csv", df=frame
            )


def _fit_joint_mechanism_levels(
    spec: ModelSpec,
    config: str,
    plan: _joint_mechanism.JointMechanismRunPlan,
) -> StatisticalFitContext:
    """Per-wave levels/concurrent bivariate fit (#421 Tier 3 (1); ``jm-001``).

    One cross-sectional fit per timepoint. **Every** published wave is fitted,
    convergence-scanned over its reported deterministics as well as its free random
    variables, given the informative new-child predictive check and power-scaling
    sensitivity, and persisted as a named trace; the wave with the most rows (ties ->
    latest) additionally hosts the fit-level artefacts (``trace.nc``,
    ``diagnostics_summary.json``, the model graph). That rule is operational — which
    fit carries the shared artefacts — and carries no scientific priority, which is
    why nothing downstream selects a wave to headline (2026-08-23 follow-up review,
    finding 1). ``joint_mechanism_fit_diagnostics.csv`` names each wave's trace,
    predictive and power-scaling files so ``release`` can check the whole bundle, and
    no wave is silently dropped.
    """
    outcome_symbols = plan.outcome_symbols
    contrast = plan.contrast
    # Trait covariates are t1-measured, so they broadcast from baseline across the
    # four timepoint rows — the same route ca-010 / ca-011 enter them by. That makes
    # the two conditional slopes comparable in construction; it does not make them
    # nested (see the run plan's comparator_equivalence).
    covariates = plan.declared_adjustment
    predictor_slope_sigma = require_value(
        plan.predictor_slope_sigma,
        "predictor_slope_sigma (the levels design's slope prior)",
    )

    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    ci = ctx.reporting.ci_prob

    section_header("Prepare data")
    prepared_all = load_and_prepare(**plan.prepare_kwargs())
    covariates = tuple(c for c in covariates if c in prepared_all.covariates)
    if covariates != plan.active_adjustment:
        plan = plan.with_active_adjustment(covariates)
        # The ACTIVE plan drives the factory, summaries and this recipe
        # rewrite; config.json keeps the RESOLVER's plan so the #623
        # currency check compares resolution with resolution (2026-08-26
        # batch; latent here — caught by the pipeline-boundary test).
        _report.write_model_recipe(ctx, plan=plan)

    def _build(sub):
        return _factories.build_joint_mechanism_model(
            sub,
            **plan.factory_kwargs(),
        )

    # Usable rows at a wave: the exposure observed (it is never imputed — imputing
    # the focal exposure would bias both slopes toward zero) and at least one outcome
    # observed. A wave below any prespecified floor is skipped *and named*: a silently
    # dropped timepoint would read as "that wave was not estimable" when it was never
    # tried. The per-outcome and overlap floors matter because the residual
    # correlation and the conditional slope are estimated from *jointly* observed
    # pairs, which the union count does not bound (2026-08-23 review, gap 1).
    wave_indices = sorted({int(p) for p in np.unique(prepared_all.phase)})
    wave_subsets: dict[int, object] = {}
    eligibility: list[dict] = []
    skipped: list[str] = []
    for w in wave_indices:
        sub = _subset_prepared(prepared_all, prepared_all.phase == w)
        record = _jm_wave_eligibility(
            sub, plan=plan, outcome_symbols=outcome_symbols, timepoint=w + 1
        )
        eligibility.append(record)
        if not record["fitted"]:
            skipped.append(f"t{w + 1} ({record['skipped_because']})")
            continue
        wave_subsets[w] = sub
    if skipped:
        rprint(
            "[yellow]Joint mechanism: skipped "
            f"{', '.join(skipped)} — below the prespecified wave minima "
            f"(rows >= {plan.min_wave_rows}, each outcome >= "
            f"{plan.min_wave_outcome_rows}, jointly observed >= "
            f"{plan.min_wave_overlap_rows}).[/yellow]"
        )
    if not wave_subsets:
        raise ValueError(
            f"{spec.model_id}: no timepoint meets the prespecified wave minima for "
            f"the exposure {plan.mechanism_symbol!r} and both outcomes."
        )
    # Build after the data filter, so a specification error (a bad contrast, a missing
    # covariate) raises rather than being swallowed as "this wave is not fittable".
    wave_built = {w: _build(sub) for w, sub in wave_subsets.items()}
    # Artefact host = most rows; ties -> latest. An operational choice about which fit
    # carries the shared fit-level files, NOT a claim that the wave is best-powered or
    # substantively primary — every wave now gets the same diagnostic treatment.
    primary_wave = max(wave_built, key=lambda w: (wave_built[w].prepared.n_obs, w))
    for record in eligibility:
        built = wave_built.get(record["timepoint"] - 1)
        record["fitted_rows"] = (
            int(built.prepared.n_obs) if built is not None else None
        )
        record["factory_dropped"] = (
            record["usable_rows"] - int(built.prepared.n_obs)
            if built is not None
            else None
        )
        record["exposure_logit_sd"] = (
            _jm_exposure_logit_sd(built, plan.mechanism_symbol)
            if built is not None
            else None
        )
        record["hosts_fit_artifacts"] = (
            built is not None and record["timepoint"] - 1 == primary_wave
        )
    eligibility_df = pd.DataFrame(eligibility)
    save_table(ctx, "joint_mechanism_wave_eligibility", eligibility_df)

    ctx.prepared = wave_built[primary_wave].prepared
    print_header(ctx)

    slope_rows: list[dict] = []
    ratio_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    gate = None
    for w in sorted(wave_built):
        built = wave_built[w]
        tp = w + 1
        reported = _jm_reported_deterministics(built)
        if w == primary_wave:
            section_header(f"Build model (artefact-hosting wave t{tp})")
            attach_built(ctx, built)
            render_model_graph(ctx)
            model_names = {rv.name for rv in ctx.model.free_RVs} | set(
                ctx.model.named_vars
            )
            diag_vars = plan.diagnostic_vars(model_names)
            psense_vars = plan.psense_vars(model_names)
            gate = shared_stages().run_primary_fit(
                ctx,
                _jm_primary_fit_plan(
                    outcome_symbols=outcome_symbols,
                    diag_vars=diag_vars,
                    psense_vars=psense_vars,
                    new_child_plan=plan.new_child_plan(),
                    # One latent residual per child over two cells: conditional
                    # coverage is structurally 1.00, so publish the new-child view.
                    marginal_ppc=True,
                    compute_loo=plan.compute_loo,
                ),
            )
            _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)
            trace = ctx.trace
            convergence = _diag.subfit_convergence(
                ctx.trace,
                label=f"{spec.model_id} wave t{tp}",
                # Free variables *and* the deterministics this wave publishes: a
                # ratio can mix far worse than the slopes it is built from.
                var_names=[rv.name for rv in ctx.model.free_RVs] + reported,
            )
            convergence["converged"] = bool(
                _report.convergence_gate_clean_passed(gate)
                and convergence.get("converged")
            )
            trace_file = "trace.nc"
            # The fit-level files are written under their house names by the shared
            # runner; copy them to the per-wave names too, so the release check can
            # require one uniform bundle per published wave.
            _jm_copy_wave_artifacts(ctx, timepoint=tp)
        else:
            res = run_subfit(
                ctx,
                built,
                label=f"{spec.model_id} wave t{tp}",
                role="wave",
                posterior_predictive=["y_post"],
                trace_filename=_JM_WAVE_TRACE.format(timepoint=tp),
                extra_var_names=reported,
            )
            trace, convergence = res.trace, res.convergence
            trace_file = res.trace_file
            _jm_marginal_ppc(
                ctx,
                outcome_symbols=outcome_symbols,
                trace=trace,
                prepared=built.prepared,
                name=_JM_WAVE_MARGINAL_PPC.format(timepoint=tp),
            )
            _jm_wave_psense(ctx, built, trace, plan=plan, timepoint=tp)
        slope_rows += _jm_slope_rows(
            trace,
            outcome_symbols=outcome_symbols,
            contrast=contrast,
            ci_prob=ci,
            wave=f"t{tp}",
            converged=bool(convergence.get("converged")),
        )
        ratio_row = _jm_ratio_governance_row(
            trace,
            ci_prob=ci,
            wave=f"t{tp}",
            contrast=contrast,
            converged=bool(convergence.get("converged")),
        )
        if ratio_row is not None:
            ratio_rows.append(ratio_row)
        diagnostic_rows.append(
            {
                "wave": f"t{tp}",
                "timepoint": tp,
                "role": "anchor" if w == primary_wave else "sub-fit",
                "n": built.prepared.n_obs,
                **convergence,
                # The bundle a published wave must carry, named so the release
                # evaluator can check it without re-deriving the convention.
                "trace_file": trace_file,
                "marginal_ppc_file": _JM_WAVE_MARGINAL_PPC.format(timepoint=tp)
                + ".csv",
                "psense_file": _JM_WAVE_PSENSE.format(timepoint=tp) + "_summary.csv",
                "convergence_vars": ", ".join(reported),
            }
        )

    section_header("Decoding-specificity contrast and conditional slope ratio")
    slopes_df = _jm_write_slopes(ctx, slope_rows, contrast=contrast)
    if ratio_rows:
        save_table(ctx, "conditional_slope_ratio", pd.DataFrame(ratio_rows))
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    save_table(ctx, "joint_mechanism_fit_diagnostics", diagnostics_df)
    _plot_joint_mechanism_by_wave(ctx, slopes_df, ci)

    write_run_metadata(
        ctx,
        extra={
            "design": "levels",
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "estimand": (
                "per-wave bivariate letter-sound slopes on {W, N} with an LKJ "
                "within-wave residual correlation; identified decoding-specificity "
                "contrast and share-retained"
            ),
            "joint_dependence": "lkj_residual_within_wave",
            "likelihood": "binomial",
            "mechanism_symbol": plan.mechanism_symbol,
            "outcome_symbols": list(outcome_symbols),
            "contrast": list(contrast),
            "covariates": list(covariates),
            # Requested vs actually-fitted adjustment, with per-term source column,
            # wave and missing-indicator status — the standard record every
            # converted family writes (#258 P1; 2026-08-21 review, finding 6).
            "effective_adjustment": effective_adjustment(
                spec,
                ctx.prepared,
                measure_confounders=plan.confounder_symbols,
                adjust_for=covariates,
                requested_adjust_for=plan.declared_adjustment,
            ),
            "include_group_nuisance": plan.include_group,
            "predictor_slope_sigma": predictor_slope_sigma,
            "artifact_hosting_timepoint": primary_wave + 1,
            "timepoints": [w + 1 for w in sorted(wave_built)],
            "n_published_fits": int(len(diagnostics_df)),
            "all_published_fits_converged": bool(
                not diagnostics_df.empty
                and diagnostics_df["converged"].eq(True).all()
            ),
            # Wave-specific eligibility, kept apart from the panel ``dropped_rows``
            # ledger a wave subset inherits unchanged (2026-08-23 review, gap 3).
            "wave_eligibility": eligibility_df.to_dict("records"),
            "wave_minima": {
                "usable_rows": plan.min_wave_rows,
                "rows_per_outcome": plan.min_wave_outcome_rows,
                "jointly_observed_rows": plan.min_wave_overlap_rows,
            },
            # The exposure is re-standardised within each wave, so one SD is a
            # different raw letter-sound increment at each one; a cross-wave
            # coefficient range is not on a fixed scale without these numbers
            # (2026-08-23 review, gap 8).
            "wave_exposure_logit_sd": {
                str(record["wave"]): record["exposure_logit_sd"]
                for record in eligibility
                if record["exposure_logit_sd"] is not None
            },
            "matched_comparators": list(plan.matched_comparators),
            "comparator_equivalence": plan.comparator_equivalence,
            "output_contract": (
                "joint_mechanism_slopes.csv carries median + inner 50% + outer "
                "reporting interval + P(>0) per wave for both slopes, their "
                "difference, the residual correlation and the conditional slope and "
                "its conditional-to-marginal ratio (whose mean is deliberately "
                "blank); conditional_slope_ratio.csv governs that ratio with a "
                "prespecified denominator-stability rule and the probability mass "
                "below zero, inside [0, 1] and above one; "
                "joint_mechanism_wave_eligibility.csv records why each wave was or "
                "was not fitted; joint_mechanism_fit_diagnostics.csv records every "
                "published wave's convergence and names its trace, new-child "
                "predictive and power-scaling files"
            ),
            "joint_mechanism_slopes": slopes_df.to_dict("records"),
        },
    )
    return finalize_report(ctx)


def _plot_joint_mechanism_by_wave(
    ctx: StatisticalFitContext, df: pd.DataFrame, ci_prob: float
) -> None:
    """Per-wave forest of the two letter-sound slopes and their identified difference.

    One figure per file (PNG + SVG + CSV via ``save_styled_figure``), not a panel, so
    it can be reused on its own. A wave that failed its convergence check is marked
    ``[GATE-FAIL]`` on its row rather than removed: the METHODS.md rule is that a
    non-converged fit is *flagged*, never silently dropped, and the plot previously
    did neither (2026-08-23 follow-up review, finding 1).
    """
    keep = df[df["term"].str.startswith(("beta_mech[", "delta_ls_decoding"))]
    if keep.empty:
        return
    keep = keep.reset_index(drop=True)
    converged = (
        keep["converged"].astype(str).str.lower().isin({"true", "1"})
        if "converged" in keep.columns
        else pd.Series(True, index=keep.index)
    )
    y = np.arange(len(keep))[::-1]
    plt.figure(figsize=(7.2, 0.42 * len(keep) + 1.6))
    colours = [
        COLOUR_BLUE if str(t).startswith("beta_mech[") else "#B45309"
        for t in keep["term"]
    ]
    for i, (_, row) in enumerate(keep.iterrows()):
        plt.errorbar(
            row["median"],
            y[i],
            xerr=[[row["median"] - row["lo"]], [row["hi"] - row["median"]]],
            fmt="o",
            color=colours[i],
            capsize=3,
        )
        plt.plot(
            [row["lo50"], row["hi50"]], [y[i], y[i]], color=colours[i], lw=3.0, alpha=0.7
        )
    plt.axvline(0.0, color="grey", ls=":", lw=1)
    plt.yticks(
        y,
        [
            f"{r['wave']}  {r['term']}"
            + ("" if converged.iloc[i] else "  [GATE-FAIL]")
            for i, (_, r) in enumerate(keep.iterrows())
        ],
        fontsize=8,
    )
    plt.xlabel(
        f"logit per SD of letter sounds (median, inner 50%, outer "
        f"{int(ci_prob * 100)}%)"
    )
    plt.title("Letter-sound slopes and their identified difference, by wave")
    save_styled_figure(ctx.output_dir, "joint_mechanism_by_wave", data=keep)


def _jm_comparator_population(
    ctx: StatisticalFitContext, outcome_symbols: tuple[str, ...]
) -> dict:
    """Fitted-row identity for the matched single-outcome comparison.

    2026-08-23 joint audit, finding 7. ``jm-002`` is described as changing only the
    dependence treatment relative to ``mech-096`` / ``mech-101``, but it is not a
    strict sensitivity: the joint likelihood needs *both* outcomes' baselines, so a
    transition with a valid word-reading baseline but no nonword baseline is in the
    single-outcome fit and not in this one. The exposure standardisation is then
    computed over a different population as well, so a difference between the
    numbers is not attributable to covariance alone. This records what the fit
    actually used so a reader -- and the cross-model comparison -- can quantify the
    gap rather than take the claim on trust.
    """
    record: dict = {
        "n_rows": int(getattr(ctx.prepared, "n_obs", 0) or 0),
        "n_children": int(getattr(ctx.prepared, "n_children", 0) or 0),
        "baseline_rule": (
            "joint: a row enters only when EVERY declared outcome's baseline is "
            "observed, so the fitted set can be smaller than each matched "
            "single-outcome comparator's"
        ),
        "exposure_standardisation": (
            "computed over these rows, so the exposure SD differs from a "
            "comparator fitted on a different row set"
        ),
        "comparison_status": "approximate_not_like_for_like",
    }
    post_counts = getattr(ctx.prepared, "post_counts", None)
    if isinstance(post_counts, dict):
        observed = {}
        for symbol in outcome_symbols:
            values = post_counts.get(symbol)
            if values is not None:
                observed[symbol] = int(np.count_nonzero(np.isfinite(np.asarray(values, dtype=float))))
        if observed:
            record["observed_cells_by_outcome"] = observed
    identity = _report.fitted_subject_identity(ctx.prepared)
    if identity is not None:
        record["fitted_subject_identity"] = identity
    return record


def _fit_joint_mechanism_transition(
    spec: ModelSpec,
    config: str,
    plan: _joint_mechanism.JointMechanismRunPlan,
) -> StatisticalFitContext:
    """Phase-stacked ANCOVA companion (#421 Tier 3 (1); ``jm-002``).

    Matched term-for-term to ``mech-096`` / ``mech-101`` so the Tier-1 Delta is
    re-reported on its original parameterisation, with a bivariate child random
    intercept (LKJ) carrying the cross-outcome covariance the separate fits cannot.
    """
    outcome_symbols = plan.outcome_symbols
    contrast = plan.contrast
    adjust_for = plan.declared_adjustment

    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    ci = ctx.reporting.ci_prob

    section_header("Prepare data")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    ctx.prepared = prepared
    # Drop any adjuster the loader dropped as constant on the fitted rows.
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)
    if adjust_for != plan.active_adjustment:
        plan = plan.with_active_adjustment(adjust_for)
        # The ACTIVE plan drives the factory, summaries and this recipe
        # rewrite; config.json keeps the RESOLVER's plan so the #623
        # currency check compares resolution with resolution (2026-08-26
        # batch; latent here — caught by the pipeline-boundary test).
        _report.write_model_recipe(ctx, plan=plan)

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_joint_mechanism_model(
        prepared,
        **plan.factory_kwargs(),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    model_names = {rv.name for rv in ctx.model.free_RVs} | set(ctx.model.named_vars)
    diag_vars = plan.diagnostic_vars(model_names)
    psense_vars = plan.psense_vars(model_names)
    gate = shared_stages().run_primary_fit(
        ctx,
        _jm_primary_fit_plan(
            outcome_symbols=outcome_symbols,
            diag_vars=diag_vars,
            psense_vars=psense_vars,
            new_child_plan=plan.new_child_plan(),
            compute_loo=plan.compute_loo,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Decoding-specificity contrast")
    slopes_df = _jm_write_slopes(
        ctx,
        _jm_slope_rows(
            ctx.trace,
            outcome_symbols=outcome_symbols,
            contrast=contrast,
            ci_prob=ci,
            wave="stacked",
            converged=_report.convergence_gate_clean_passed(gate),
        ),
        contrast=contrast,
    )

    write_run_metadata(
        ctx,
        extra={
            "design": "transition",
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "estimand": (
                "phase-stacked bivariate letter-sound slopes on {W, N} with a "
                "bivariate (LKJ) child random intercept; within-model operational "
                "contrast between the two adjusted test-score slopes (not a "
                "construct-level decoding-specificity measure)"
            ),
            # 2026-08-23 joint audit, finding 7. The joint fit requires *both*
            # outcomes' baselines, so it drops rows the single-outcome comparators
            # keep, and the exposure standardisation is then computed over a
            # different population too. Recording the row identity makes the
            # comparison's approximation machine-readable instead of leaving the
            # report to assert "like-for-like".
            "comparator_population": _jm_comparator_population(ctx, outcome_symbols),
            "joint_dependence": "lkj_child_intercept",
            "likelihood": "beta_binomial",
            "mechanism_symbol": plan.mechanism_symbol,
            "outcome_symbols": list(outcome_symbols),
            "contrast": list(contrast),
            "adjust_for": list(adjust_for),
            # Requested vs actually-fitted adjustment, incl. both outcomes' own
            # autoregressive baselines (#258 P1; 2026-08-21 review, finding 6).
            "effective_adjustment": effective_adjustment(
                spec,
                ctx.prepared,
                measure_confounders=plan.confounder_symbols,
                adjust_for=adjust_for,
                requested_adjust_for=plan.declared_adjustment,
                baseline_symbols=plan.outcome_symbols,
            ),
            "matched_comparators": list(plan.matched_comparators),
            "comparator_equivalence": plan.comparator_equivalence,
            # The exposure is standardised once over this model's joint union of
            # rows; each marginal comparator re-standardises on its own rows, so the
            # three do not share a unit (2026-08-23 review, finding 2).
            "exposure_logit_sd": _jm_exposure_logit_sd(built, plan.mechanism_symbol),
            "loo_pit_unit": _diag.JOINT_LOO_PIT_UNIT_LABEL,
            "output_contract": (
                "joint_mechanism_slopes.csv carries median + inner 50% + outer "
                "reporting interval + P(>0) for both slopes, their identified "
                "difference and the child-level residual correlation"
            ),
            "joint_mechanism_slopes": slopes_df.to_dict("records"),
        },
    )
    return finalize_report(ctx)
