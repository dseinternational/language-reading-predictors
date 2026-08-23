# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Joint-mechanism orchestration (``kind="joint_mechanism"``, #421 Tier 3).

``fit_joint_mechanism`` fits letter-sound knowledge jointly against word reading
and nonword decoding in one of two designs: a *levels* fit carrying a bivariate
residual dependence block on the observation row, and a *transition* fit over
period transitions. Only the levels design yields ``rho_outcome`` and the
conditional slopes, so those terms are emitted conditionally, and a wave with too
few usable rows is not fitted at all — a residual correlation from a handful of
children would be prior-dominated. Nothing here is randomised: every slope is an
adjusted association.
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
    save_table,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.plotting import save_styled_figure
from language_reading_predictors.statistical_models.preprocessing import (
    _subset_prepared,
    load_and_prepare,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
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


#: Terms reported for both designs, in report order. ``rho_outcome`` and the two
#: conditional-slope terms exist only where the dependence block sits on the
#: observation row (the ``levels`` design), so they are emitted conditionally.
_JM_TERM_LABELS: dict[str, str] = {
    # 2026-08-23 joint audit, finding 4. "Decoding specificity" is a construct-level
    # claim this model does not identify: W and N differ in item count (79 vs 6),
    # score distribution, discrimination, reliability and floor/ceiling behaviour,
    # and nothing here calibrates them to a common latent outcome scale. A single
    # common ability loading differently on the two tests produces a non-zero slope
    # contrast on its own. What IS identified is an operational contrast between two
    # adjusted test-score associations, so that is what the label says.
    "delta_ls_decoding": (
        "Delta = beta(LS->N) - beta(LS->W): operational test-score slope contrast "
        "(logit per SD), not a construct-level decoding-specificity measure"
    ),
    "rho_outcome": "residual correlation between the two outcomes",
    "beta_held_on_focal": "implied coefficient of the held-fixed outcome",
    "beta_mech_focal_given_held": "beta(LS->W) holding nonword decoding fixed",
    # Likewise a ratio of two adjusted associations, not a mediation proportion or
    # a causal path fraction.
    "share_retained": (
        "ratio of adjusted associations: beta(LS->W) holding latent decoding fixed, "
        "over the unconditional beta(LS->W)"
    ),
    "abs_slope_reduction": (
        "absolute reduction in beta(LS->W) when latent decoding is held fixed "
        "(logit per SD) - the denominator-free companion to the ratio"
    ),
}

#: Deliberately small logit-scale identifiability threshold for the ``share_retained``
#: ratio's two instability routes (2026-08-23 joint audit, finding 10), matching the
#: historical-joint residual-scale rule's 0.05-logit convention. A ratio is reported
#: only when the posterior supports, with at least ``_JM_STABILITY_SUPPORT``
#: probability, BOTH that its denominator ``beta_mech[focal]`` is away from zero and
#: that the held-fixed outcome's residual scale ``sigma_u_resid[held]`` is away from
#: zero (it divides the conditional slope). Neither is a minimum-important-effect
#: threshold. A finite Monte Carlo mean over a heavy-tailed ratio looks reassuring
#: precisely when the quantity is least meaningful, which is why the mean is
#: withheld for this term regardless.
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
    """
    if name not in trace.posterior:
        return None
    var = trace.posterior[name]
    if outcome is not None:
        var = var.sel(outcome=outcome)
    draws = np.asarray(var.values).ravel()
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
) -> None:
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
    """
    with guard_optional(
        ctx, "ppc_summary_marginal.csv",
        filename="ppc_summary_marginal.csv", kind="table", verb="skipped",
    ):
        post = ctx.trace.posterior
        required = {"eta", "u_resid", "sigma_u_resid", "rho_outcome"}
        if not required <= set(post.data_vars):
            return
        # Linear predictor with the fitted residual removed.
        eta = np.asarray(post["eta"].values)
        core = eta - np.asarray(post["u_resid"].values)
        n_obs, n_outcomes = core.shape[-2], core.shape[-1]
        core = core.reshape(-1, n_obs, n_outcomes)
        sigma = np.asarray(post["sigma_u_resid"].values).reshape(-1, n_outcomes)
        rho = np.asarray(post["rho_outcome"].values).reshape(-1)
        if n_outcomes != 2:
            return

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

        row = np.asarray(ctx.trace.constant_data["y_post_cell_row"].values).astype(int)
        col = np.asarray(
            ctx.trace.constant_data["y_post_cell_outcome"].values
        ).astype(int)
        n_trials = np.array(
            [ctx.prepared.n_trials[s] for s in outcome_symbols], dtype=int
        )[col]
        y_rep = rng.binomial(n_trials[None, :], p[:, row, col])  # (draw, cell)

        y_obs = np.asarray(ctx.trace.observed_data["y_post"].values, dtype=float)
        finite = np.isfinite(y_obs)
        y_rep, y_obs = y_rep[:, finite], y_obs[finite]
        n = int(y_obs.shape[0])
        rows = []
        for level in ci_levels:
            lo = np.quantile(y_rep, (1.0 - level) / 2.0, axis=0)
            hi = np.quantile(y_rep, (1.0 + level) / 2.0, axis=0)
            n_in = int(np.count_nonzero((y_obs >= lo) & (y_obs <= hi)))
            rows.append(
                {
                    "mode": "count_interval_marginal",
                    "node": "y_post",
                    "unit": "observations",
                    "quantity": "observed score (new-child predictive)",
                    "level": float(level),
                    "level_pct": int(round(level * 100)),
                    "n_total": n,
                    "n_inside": n_in,
                    "coverage": float(n_in / n) if n else float("nan"),
                }
            )
        frame = pd.DataFrame(rows)
        save_table(ctx, "ppc_summary_marginal", frame, required=False)


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
        # The generic LOO-PIT would pool tests with different denominators.
        for index, symbol in enumerate(outcome_symbols):
            stem = "loo_pit" if index == 0 else f"loo_pit_{symbol.lower()}"
            _diag.save_joint_loo_pit_plot(
                c,
                symbol,
                filename_stem=stem,
                posterior_var="beta_mech",
            )

    def _density_groups_for_psense(c: StatisticalFitContext) -> None:
        # The levels design computes no PSIS-LOO (the saturated per-child residual
        # makes it fail its Pareto-k diagnostics en masse; see the run plan), but
        # power scaling still needs the log_likelihood / log_prior groups the LOO
        # step would otherwise have attached — the same direct route the
        # no-LOO mediation families take (#381).
        _diag.compute_log_likelihood_and_prior(c, strict=False)

    return PrimaryFitPlan(
        diagnostic_vars=tuple(diag_vars),
        plot_prior_predictive=_plot_prior,
        post_sampling_audit=None if compute_loo else _density_groups_for_psense,
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


def _fit_joint_mechanism_levels(
    spec: ModelSpec,
    config: str,
    plan: _joint_mechanism.JointMechanismRunPlan,
) -> StatisticalFitContext:
    """Per-wave levels/concurrent bivariate fit (#421 Tier 3 (1); ``jm-001``).

    One cross-sectional fit per timepoint, mirroring the ``concurrent`` family's
    published shape: the diagnostic-anchor wave (most rows; ties -> latest) carries
    the trace / gate / PPC artefacts, every wave's convergence is recorded in
    ``joint_mechanism_fit_diagnostics.csv``, and no wave is silently dropped.
    """
    outcome_symbols = plan.outcome_symbols
    contrast = plan.contrast
    # Trait covariates are t1-measured, so they broadcast from baseline across the
    # four timepoint rows — exactly as ca-010 / ca-011 enter them, which is what
    # makes the identified share_retained a like-for-like replacement.
    covariates = plan.declared_adjustment
    predictor_slope_sigma = plan.predictor_slope_sigma
    assert predictor_slope_sigma is not None

    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    ci = ctx.reporting.ci_prob

    section_header("Prepare data")
    prepared_all = load_and_prepare(**plan.prepare_kwargs())
    covariates = tuple(c for c in covariates if c in prepared_all.covariates)
    if covariates != plan.active_adjustment:
        plan = plan.with_active_adjustment(covariates)
        ctx.resolved_plan = plan
        _report.write_model_recipe(ctx)

    def _build(sub):
        return _factories.build_joint_mechanism_model(
            sub,
            **plan.factory_kwargs(),
        )

    # Usable rows at a wave: the exposure observed (it is never imputed — imputing
    # the focal exposure would bias both slopes toward zero) and at least one outcome
    # observed. A wave below the floor is skipped *and named*: a silently dropped
    # timepoint would read as "that wave was not estimable" when it was never tried.
    wave_indices = sorted({int(p) for p in np.unique(prepared_all.phase)})
    wave_subsets: dict[int, object] = {}
    skipped: list[str] = []
    for w in wave_indices:
        sub = _subset_prepared(prepared_all, prepared_all.phase == w)
        usable = ~np.isnan(sub.post_counts[plan.mechanism_symbol])
        any_outcome = np.zeros(sub.n_obs, dtype=bool)
        for symbol in outcome_symbols:
            any_outcome |= ~np.isnan(sub.post_counts[symbol])
        n_usable = int(np.count_nonzero(usable & any_outcome))
        assert plan.min_wave_rows is not None
        if n_usable < plan.min_wave_rows:
            skipped.append(f"t{w + 1} ({n_usable} usable rows)")
            continue
        wave_subsets[w] = sub
    if skipped:
        rprint(
            "[yellow]Joint mechanism: skipped "
            f"{', '.join(skipped)} — fewer than {plan.min_wave_rows} rows with the "
            "exposure and at least one outcome observed.[/yellow]"
        )
    if not wave_subsets:
        raise ValueError(
            f"{spec.model_id}: no timepoint has at least {plan.min_wave_rows} rows "
            f"with the exposure {plan.mechanism_symbol!r} and an outcome observed."
        )
    # Build after the data filter, so a specification error (a bad contrast, a missing
    # covariate) raises rather than being swallowed as "this wave is not fittable".
    wave_built = {w: _build(sub) for w, sub in wave_subsets.items()}
    # Diagnostic anchor = most rows; ties -> latest. An operational artefact-selection
    # rule, not a claim that the wave is best-powered or substantively primary.
    primary_wave = max(wave_built, key=lambda w: (wave_built[w].prepared.n_obs, w))

    ctx.prepared = wave_built[primary_wave].prepared
    print_header(ctx)

    slope_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    gate = None
    for w in sorted(wave_built):
        built = wave_built[w]
        tp = w + 1
        if w == primary_wave:
            section_header(f"Build model (anchor wave t{tp})")
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
                label=f"{spec.model_id} anchor wave t{tp}",
                var_names=[rv.name for rv in ctx.model.free_RVs],
            )
            convergence["converged"] = bool(
                _report.convergence_gate_clean_passed(gate)
                and convergence.get("converged")
            )
        else:
            res = run_subfit(
                ctx, built, label=f"{spec.model_id} wave t{tp}", role="wave"
            )
            trace, convergence = res.trace, res.convergence
        slope_rows += _jm_slope_rows(
            trace,
            outcome_symbols=outcome_symbols,
            contrast=contrast,
            ci_prob=ci,
            wave=f"t{tp}",
            converged=bool(convergence.get("converged")),
        )
        diagnostic_rows.append(
            {
                "wave": f"t{tp}",
                "timepoint": tp,
                "role": "anchor" if w == primary_wave else "sub-fit",
                "n": built.prepared.n_obs,
                **convergence,
            }
        )

    section_header("Decoding-specificity contrast and share retained")
    slopes_df = _jm_write_slopes(ctx, slope_rows, contrast=contrast)
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
            "diagnostic_anchor_timepoint": primary_wave + 1,
            "timepoints": [w + 1 for w in sorted(wave_built)],
            "n_published_fits": int(len(diagnostics_df)),
            "all_published_fits_converged": bool(
                not diagnostics_df.empty
                and diagnostics_df["converged"].eq(True).all()
            ),
            "matched_comparators": list(plan.matched_comparators),
            "output_contract": (
                "joint_mechanism_slopes.csv carries median + inner 50% + outer "
                "reporting interval + P(>0) per wave for both slopes, their "
                "difference, the residual correlation and the conditional slope / "
                "share retained; joint_mechanism_fit_diagnostics.csv records every "
                "published wave's convergence"
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
    it can be reused on its own.
    """
    keep = df[df["term"].str.startswith(("beta_mech[", "delta_ls_decoding"))]
    if keep.empty:
        return
    keep = keep.reset_index(drop=True)
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
    plt.yticks(y, [f"{r['wave']}  {r['term']}" for _, r in keep.iterrows()], fontsize=8)
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
        ctx.resolved_plan = plan
        _report.write_model_recipe(ctx)

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
            "output_contract": (
                "joint_mechanism_slopes.csv carries median + inner 50% + outer "
                "reporting interval + P(>0) for both slopes, their identified "
                "difference and the child-level residual correlation"
            ),
            "joint_mechanism_slopes": slopes_df.to_dict("records"),
        },
    )
    return finalize_report(ctx)
