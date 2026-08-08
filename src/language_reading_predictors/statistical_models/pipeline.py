# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end fit pipeline for the statistical models.

``fit_mechanism(spec, config)`` is the entry point for LRP56/57/58 and companions,
``fit_mediation`` for the g-formula decompositions, and ``fit_adjusted`` /
``fit_lcsm`` / ``fit_concurrent`` for the LRP65/LRP67/LRP-CA companions.

The ITT and joint families have moved to :mod:`pipelines` (#394 step 5), followed
by DiD, dose-response, gain- and level-factors, block exposure and aligned (step
6). Their entry points are re-exported here so model modules and tests keep their
import path; ``MIGRATED_FAMILIES`` in ``test_pipeline_boundaries.py`` is the
authoritative list, checked against the package contents.

The shared mechanics they used to carry inline now live in :mod:`runtime` (the
stage binding and spec validation), :mod:`publication` (banners, report template,
model graph), :mod:`adjustment` (the fitted adjustment-set record),
:mod:`prior_artifacts`, :mod:`ppc_artifacts` and :mod:`figure_artifacts`. The
remaining families migrate in the same way.

Each pipeline:

1. Loads data via :func:`preprocessing.load_and_prepare`.
2. Builds the PyMC model via the appropriate factory.
3. Writes prior-panel plots.
4. Runs prior predictive, posterior sampling (nutpie), LOO, posterior
   predictive.
5. Saves ``trace.nc`` (with the prior / prior_predictive / log_prior groups
   attached — issue #125 step 0b), ``config.json``, ``diagnostics_summary.json``
   (the pass/fail convergence gate), ``priors_table.csv`` and the standard
   diagnostic plots to ``output/statistical_models/models/{model_id}-{config}/``.
6. Copies ``docs/models/{model_id}/index.qmd`` and the shared
   ``docs/models/_partials/`` alongside the artefacts so the Quarto report can be
   rendered in-place.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import (
    COLOUR_BLUE,
    COLOUR_RED,
    FIGSIZE_LG,
)
from rich import print as rprint
from scipy.special import expit

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    datasets as _datasets,
    diagnostics as _diag,
    factories as _factories,
    historical as _historical,
    lcf_inference as _lcf_inference,
    lcf_summaries as _lcf_summaries,
    mechanism as _mechanism,
    reporting as _report,
    survival as _survival,
)
from language_reading_predictors.statistical_models.plotting import (
    save_styled_figure,
)
from language_reading_predictors.statistical_models.adjustment import (
    effective_adjustment,
)
from language_reading_predictors.statistical_models.artifacts import (
    guard_optional,
    record_artifact,
    save_table,
)
from language_reading_predictors.statistical_models.concurrent import (
    resolve_concurrent_run_plan,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.factories import default_of
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_forest_plot,
    write_child_fit,
    write_panel_child_fit,
    write_panel_trajectory,
)
from language_reading_predictors.statistical_models.growth import (
    resolve_growth_run_plan,
)
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
)

# Compatibility re-exports: these families now live in ``pipelines/`` (#394 steps
# 5-6). The ``x as x`` form marks them as deliberate re-exports rather than unused
# imports, and keeps ``from ...pipeline import fit_itt`` working for every model
# module and test until step 8 migrates the callers.
from language_reading_predictors.statistical_models.pipelines.aligned import (
    fit_aligned as fit_aligned,
)
from language_reading_predictors.statistical_models.pipelines.block_exposure import (
    fit_block_exposure as fit_block_exposure,
)
from language_reading_predictors.statistical_models.pipelines.did import (
    fit_did as fit_did,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import (
    fit_dose_response as fit_dose_response,
)
from language_reading_predictors.statistical_models.pipelines.gain_factors import (
    fit_gain_factors as fit_gain_factors,
)
from language_reading_predictors.statistical_models.pipelines.itt import (
    fit_itt as fit_itt,
)
from language_reading_predictors.statistical_models.pipelines.joint import (
    fit_joint as fit_joint,
)
from language_reading_predictors.statistical_models.pipelines.level_factors import (
    fit_level_factors as fit_level_factors,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    growth_contrast_pushforward_rows,
    horseshoe_pushforward_rows,
    marginal_pushforward_rows,
    pushforward_n_trials,
    pushforward_outcome_label,
    write_indicator_prior_check,
    write_prior_pushforward,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset_prepared,
    load_and_prepare,
    load_and_prepare_lagged_outcome,
    load_longitudinal_panel,
    load_wave_panel,
    logit_safe,
    split_confounders_by_timing,
    split_covariates_by_wave,
    standardise,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    print_loo_row,
    render_model_graph,
)
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    run_ppc,
    run_sampling_and_loo,
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import (
    PrimaryFitPlan,
)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _raw_covariate_confounders(confounders: Iterable[str]) -> tuple[str, ...]:
    """The confounders that are raw covariates, needing ``covariates=`` loading.

    A mediation adjustment set mixes two kinds of confounder: bounded-count skill
    measures (E, R, ...) that arrive via ``prepared.pre_logit`` (they are in
    ``ITT_OUTCOMES`` or ``spec.extra['outcomes']``), and revised-DAG raw covariates
    (hearing ``hs``/``hs_missing``, speech ``deapp_c``, phonological memory
    ``erbto`` + missing indicators; #246) that must be requested as ``covariates``.
    A symbol is a raw covariate exactly when it is not a bounded-count measure.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    return tuple(c for c in confounders if c not in MEASURES)


def _survival_summary(
    trace, *, ci_prob: float, hazard_link: str, use_treatment: bool
) -> pd.DataFrame:
    """Off-floor discrete-time hazard summary (log-hazard, hazard ratio, P>0).

    Reports the treatment hazard shift and baseline-covariate slopes on the
    log-hazard scale (with ``exp`` as a hazard ratio and ``P(effect > 0)``), plus
    the per-interval baseline off-floor probability for an untreated child at mean
    covariates, on the model's ``hazard_link`` scale. Equal-tailed intervals at
    ``ci_prob`` with the posterior median as the point estimate (the suite convention).
    """
    post = trace.posterior
    lo, hi = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2

    def _row(term: str, draws: np.ndarray, *, as_ratio: bool) -> dict:
        d = np.asarray(draws).reshape(-1)
        return {
            "term": term,
            "median": float(np.median(d)),
            "ci_low": float(np.quantile(d, lo)),
            "ci_high": float(np.quantile(d, hi)),
            "hazard_ratio": float(np.exp(np.median(d))) if as_ratio else float("nan"),
            "P(>0)": float(np.mean(d > 0)) if as_ratio else float("nan"),
        }

    rows: list[dict] = []
    if use_treatment:
        rows.append(_row("tau (log hazard shift, treated)", post["tau"].values, as_ratio=True))
    for name in sorted(v for v in post.data_vars if str(v).startswith("beta_")):
        rows.append(_row(f"{name} (log hazard, per SD)", post[name].values, as_ratio=True))

    alpha = post["alpha"].stack(sample=("chain", "draw")).transpose("interval", "sample")
    labels = [str(v) for v in alpha.coords["interval"].values]
    for i, lab in enumerate(labels):
        a = alpha.values[i]
        base = 1.0 - np.exp(-np.exp(a)) if hazard_link == "cloglog" else 1.0 / (1.0 + np.exp(-a))
        rows.append(
            {
                "term": f"baseline off-floor prob [{lab}]",
                "median": float(np.median(base)),
                "ci_low": float(np.quantile(base, lo)),
                "ci_high": float(np.quantile(base, hi)),
                "hazard_ratio": float("nan"),
                "P(>0)": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fit_survival(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Discrete-time off-floor survival fit for a floored outcome P / N (#230 §5).

    Fits a person-period discrete-time hazard for the *time* to come off the floor,
    generalising the single-transition off-floor estimand of the LRPITT09/11 floor
    rule to all four waves. Treatment enters as an intervention-aligned hazard shift;
    the estimand is prognostic (both arms are treated by t4).
    """
    require_spec(spec, "survival", outcome=True)
    ctx = make_context(spec, config)

    section_header("Prepare data")
    panel = _survival.prepare_survival(spec.outcome_symbol)
    ctx.prepared = panel
    print_header(ctx)
    rprint(
        f"  Survival at-risk set: {panel.n_at_risk_children} children at the "
        f"{spec.outcome_symbol} floor at t1 contribute {panel.n_obs} person-period rows; "
        f"{panel.n_events} off-floor events."
    )
    if panel.dropped_rows:
        rprint(
            f"  [yellow]{panel.dropped_rows} at-risk child(ren) contributed no rows "
            "(t2 score unobserved, so no interval could be placed) and are excluded.[/yellow]"
        )
    for name, k in panel.imputed_covariate_rows.items():
        if k:
            rprint(
                f"  [yellow]{k} row(s) had a missing baseline {name}; mean-imputed (z=0).[/yellow]"
            )

    hazard_link = spec.extra.get("hazard_link", "cloglog")
    use_treatment = bool(spec.extra.get("use_treatment", True))

    section_header("Build model")
    built = _survival.build_survival_model(
        panel, hazard_link=hazard_link, use_treatment=use_treatment
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    diag_vars = (
        ["alpha"]
        + [f"beta_{n}" for n in panel.covariates]
        + (["tau"] if use_treatment else [])
    )

    # Reference adoption of the shared primary-fit lifecycle (#394 design 2):
    # the invariant sequence lives in ``stages.run_primary_fit`` and the family
    # declares only its execution profile.
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            ppc_var_names=("y_event",),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_rate_plot(
                c, spec.outcome_symbol, node="y_event"
            ),
            extended_term="tau" if use_treatment else None,
        ),
    )

    section_header("Off-floor hazard summary")
    summary = _survival_summary(
        ctx.trace, ci_prob=ctx.reporting.ci_prob, hazard_link=hazard_link,
        use_treatment=use_treatment,
    )
    save_table(ctx, "survival_summary", summary)
    print_table(
        ranked_dataframe_table(
            summary,
            title=(
                f"Off-floor discrete-time hazard ({spec.outcome_symbol}, {hazard_link}); "
                "positive = raises Pr(off-floor); prognostic, not a randomised effect"
            ),
            columns=list(summary.columns),
            rank_column=False,
            precision=3,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "n_at_risk_children": panel.n_at_risk_children,
            "n_events": panel.n_events,
            "hazard_link": hazard_link,
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Joint-mechanism pipeline (#421 Tier 3): bivariate letter-sound -> {W, N}
# ---------------------------------------------------------------------------

#: Terms reported for both designs, in report order. ``rho_outcome`` and the two
#: conditional-slope terms exist only where the dependence block sits on the
#: observation row (the ``levels`` design), so they are emitted conditionally.
_JM_TERM_LABELS: dict[str, str] = {
    "delta_ls_decoding": "Delta = beta(LS->N) - beta(LS->W) (decoding specificity)",
    "rho_outcome": "residual correlation between the two outcomes",
    "beta_held_on_focal": "implied coefficient of the held-fixed outcome",
    "beta_mech_focal_given_held": "beta(LS->W) holding nonword decoding fixed",
    "share_retained": "share of beta(LS->W) retained when decoding is held fixed",
}

#: Fewest usable rows a wave needs before the levels design will fit it. A bivariate
#: residual covariance on a handful of children is prior-dominated, and reporting a
#: correlation from it would misrepresent what the joint fit adds.
_JM_MIN_WAVE_ROWS = 10

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
        "share_retained",
    ):
        _add(term, _JM_TERM_LABELS[term], _jm_term_summary(trace, term, ci_prob))
    return rows


def _jm_diag_vars(
    trace_or_model_names: set[str],
    *,
    design: str,
    adjust_for: tuple[str, ...],
    confounder_symbols: tuple[str, ...],
    include_group: bool,
) -> list[str]:
    """Reported parameters for the convergence gate and the prior-vs-posterior panel."""
    names = ["alpha", "beta_mech", "delta_ls_decoding"]
    if include_group or "G" in confounder_symbols:
        names.append("beta_group_nuisance" if design == "levels" else "beta_G")
    if "A" in confounder_symbols:
        names.append("gamma_A")
    names += [f"gamma_{c}" for c in adjust_for]
    if design == "levels":
        names += [
            "sigma_u_resid",
            "rho_outcome",
            "beta_mech_focal_given_held",
            "share_retained",
        ]
    else:
        names += ["gamma_own", "alpha_phase", "kappa", "sigma_u_child", "rho_outcome"]
    return [n for n in names if n in trace_or_model_names]


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


def _jm_standard_artefacts(
    ctx: StatisticalFitContext,
    *,
    outcome_symbols: tuple[str, ...],
    diag_vars: list[str],
    psense_vars: list[str],
    marginal_ppc: bool = False,
) -> dict | None:
    """The shared post-sampling artefact set for a joint-mechanism fit.

    Everything the house technical report expects of a published fit, in the order
    ``stages.py`` uses elsewhere: summary diagnostics, **power-scaling prior
    sensitivity** (#381 — the mechanism family already audits its reported slopes and
    this family must not be the one exemption), per-outcome posterior-predictive
    overlays, the **``ppc_summary.csv`` coverage statistic** (#318), the convergence
    gate, per-outcome LOO-PIT, the trace and the prior-vs-posterior panel.

    The last three were silently absent before the #427 review: the per-outcome
    LOO-PIT calls raised inside :func:`diagnostics.save_joint_loo_pit_plot` because
    the helper hard-required ``posterior['tau']``, which this family does not have.
    ``posterior_var="beta_mech"`` now names the family's own reported coefficient.
    """
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported coefficients (#381). The
    # share_retained ratio is deliberately excluded: psense summarises a location
    # shift, which is not a stable description of a ratio whose denominator has
    # posterior mass on both sides of small values.
    _diag.run_psense(ctx, var_names=psense_vars)

    section_header("Posterior predictive")
    _diag.sample_posterior_predictive(ctx, var_names=["y_post"])
    for index, symbol in enumerate(outcome_symbols):
        stem = (
            "posterior_predictive_check"
            if index == 0
            else f"posterior_predictive_check_{symbol.lower()}"
        )
        _diag.save_joint_posterior_predictive_plot(ctx, symbol, filename_stem=stem)
    # Coverage statistic (#318). Per-observation interval coverage is
    # denominator-agnostic — each flattened child x outcome cell is scored against
    # its own predictive draws — so it is well defined on the flattened ``y_post``
    # even though the distribution overlays must be split per outcome above.
    with guard_optional(ctx, "ppc_summary.csv", filename="ppc_summary.csv", kind="table"):
        coverage = _report.ppc_interval_coverage(ctx.trace, node="y_post")
        save_table(ctx, "ppc_summary", coverage, required=False)
    if marginal_ppc:
        _jm_marginal_ppc(ctx, outcome_symbols=outcome_symbols)

    section_header("Extended diagnostics")
    gate = _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    # The generic LOO-PIT would pool flattened cells from tests with different
    # denominators (79 items vs 6); save one calibrated plot per outcome instead.
    # ``causal_term`` only selects which parameter the rank and ESS-evolution plots
    # focus on — it carries no causal claim, and nothing in this family is causal.
    # Naming the headline contrast matters: left unset, the ESS-evolution plot tries
    # to draw every variable (563 subplots for the transition design) and is skipped
    # entirely, which is how a required diagnostic goes quietly missing.
    _diag.run_extended_diagnostics(
        ctx, causal_term="delta_ls_decoding", include_loo_pit=False
    )
    for index, symbol in enumerate(outcome_symbols):
        stem = "loo_pit" if index == 0 else f"loo_pit_{symbol.lower()}"
        _diag.save_joint_loo_pit_plot(
            ctx, symbol, filename_stem=stem, posterior_var="beta_mech"
        )
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)
    return gate


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
    e = spec.extra
    design = str(e.get("design", "levels"))
    if design not in {"levels", "transition"}:
        raise ValueError(
            f"{spec.model_id}: joint-mechanism design must be 'levels' or "
            f"'transition'; got {design!r}"
        )
    if design == "levels":
        return _fit_joint_mechanism_levels(spec, config)
    return _fit_joint_mechanism_transition(spec, config)


def _jm_common_settings(spec: ModelSpec) -> dict:
    e = spec.extra
    return {
        "mechanism_symbol": spec.mechanism_symbol or "L",
        "outcome_symbols": tuple(e.get("outcome_symbols", ("W", "N"))),
        "contrast": tuple(e.get("contrast", ("N", "W"))),
        "confounder_symbols": tuple(e.get("confounder_symbols", ("G", "A"))),
        "include_group": bool(e.get("include_group", True)),
    }


def _fit_joint_mechanism_levels(
    spec: ModelSpec, config: str
) -> StatisticalFitContext:
    """Per-wave levels/concurrent bivariate fit (#421 Tier 3 (1); ``jm-001``).

    One cross-sectional fit per timepoint, mirroring the ``concurrent`` family's
    published shape: the diagnostic-anchor wave (most rows; ties -> latest) carries
    the trace / gate / PPC artefacts, every wave's convergence is recorded in
    ``joint_mechanism_fit_diagnostics.csv``, and no wave is silently dropped.
    """
    e = spec.extra
    s = _jm_common_settings(spec)
    outcome_symbols = s["outcome_symbols"]
    contrast = s["contrast"]
    # Trait covariates are t1-measured, so they broadcast from baseline across the
    # four timepoint rows — exactly as ca-010 / ca-011 enter them, which is what
    # makes the identified share_retained a like-for-like replacement.
    covariates = tuple(e.get("covariates", ()))
    predictor_slope_sigma = float(e.get("predictor_slope_sigma", 0.3))

    ctx = make_context(spec, config)
    ci = ctx.reporting.ci_prob

    section_header("Prepare data")
    prepared_all = load_and_prepare(
        phase_mode="levels",
        outcomes=(*outcome_symbols, s["mechanism_symbol"]),
        baseline_covariates=covariates,
    )
    covariates = tuple(c for c in covariates if c in prepared_all.covariates)

    def _build(sub):
        return _factories.build_joint_mechanism_model(
            sub,
            design="levels",
            mechanism_symbol=s["mechanism_symbol"],
            outcome_symbols=outcome_symbols,
            contrast=contrast,
            adjust_for=covariates,
            confounder_symbols=s["confounder_symbols"],
            include_group=s["include_group"],
            predictor_slope_sigma=predictor_slope_sigma,
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
        usable = ~np.isnan(sub.post_counts[s["mechanism_symbol"]])
        any_outcome = np.zeros(sub.n_obs, dtype=bool)
        for symbol in outcome_symbols:
            any_outcome |= ~np.isnan(sub.post_counts[symbol])
        n_usable = int(np.count_nonzero(usable & any_outcome))
        if n_usable < _JM_MIN_WAVE_ROWS:
            skipped.append(f"t{w + 1} ({n_usable} usable rows)")
            continue
        wave_subsets[w] = sub
    if skipped:
        rprint(
            "[yellow]Joint mechanism: skipped "
            f"{', '.join(skipped)} — fewer than {_JM_MIN_WAVE_ROWS} rows with the "
            "exposure and at least one outcome observed.[/yellow]"
        )
    if not wave_subsets:
        raise ValueError(
            f"{spec.model_id}: no timepoint has at least {_JM_MIN_WAVE_ROWS} rows "
            f"with the exposure {s['mechanism_symbol']!r} and an outcome observed."
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
            section_header("Prior predictive")
            _diag.run_prior_predictive(ctx, draws=1000)
            for index, symbol in enumerate(outcome_symbols):
                stem = (
                    "prior_predictive_check"
                    if index == 0
                    else f"prior_predictive_check_{symbol.lower()}"
                )
                _diag.save_prior_predictive_plot(ctx, symbol, filename_stem=stem)
            run_sampling_and_loo(ctx)
            trace = ctx.trace
            model_names = {rv.name for rv in ctx.model.free_RVs} | set(
                ctx.trace.posterior.data_vars
            )
            diag_vars = _jm_diag_vars(
                model_names,
                design="levels",
                adjust_for=covariates,
                confounder_symbols=s["confounder_symbols"],
                include_group=s["include_group"],
            )
            psense_vars = [
                v
                for v in ("beta_mech", "delta_ls_decoding", "rho_outcome",
                          "beta_mech_focal_given_held")
                if v in ctx.trace.posterior
            ]
            gate = _jm_standard_artefacts(
                ctx,
                outcome_symbols=outcome_symbols,
                diag_vars=diag_vars,
                psense_vars=psense_vars,
                # One latent residual per child over two cells: the conditional
                # coverage is structurally 1.00, so publish the new-child version too.
                marginal_ppc=True,
            )
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
            trace, convergence = _sample_model(
                built.model, ctx.sampling, label=f"{spec.model_id} wave t{tp}"
            )
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
            "mechanism_symbol": s["mechanism_symbol"],
            "outcome_symbols": list(outcome_symbols),
            "contrast": list(contrast),
            "covariates": list(covariates),
            "include_group_nuisance": s["include_group"],
            "predictor_slope_sigma": predictor_slope_sigma,
            "diagnostic_anchor_timepoint": primary_wave + 1,
            "timepoints": [w + 1 for w in sorted(wave_built)],
            "n_published_fits": int(len(diagnostics_df)),
            "all_published_fits_converged": bool(
                not diagnostics_df.empty
                and diagnostics_df["converged"].eq(True).all()
            ),
            "matched_comparators": ["lrp-rli-ca-010", "lrp-rli-ca-011"],
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


def _fit_joint_mechanism_transition(
    spec: ModelSpec, config: str
) -> StatisticalFitContext:
    """Phase-stacked ANCOVA companion (#421 Tier 3 (1); ``jm-002``).

    Matched term-for-term to ``mech-096`` / ``mech-101`` so the Tier-1 Delta is
    re-reported on its original parameterisation, with a bivariate child random
    intercept (LKJ) carrying the cross-outcome covariance the separate fits cannot.
    """
    e = spec.extra
    s = _jm_common_settings(spec)
    outcome_symbols = s["outcome_symbols"]
    contrast = s["contrast"]
    adjust_for = tuple(e.get("adjust_for", ()))

    ctx = make_context(spec, config)
    ci = ctx.reporting.ci_prob

    section_header("Prepare data")
    pre_adj, post_adj = split_covariates_by_wave(adjust_for)
    prepared = load_and_prepare(
        phase_mode="all",
        outcomes=(*outcome_symbols, s["mechanism_symbol"]),
        covariates=pre_adj,
        post_covariates=post_adj,
    )
    ctx.prepared = prepared
    # Drop any adjuster the loader dropped as constant on the fitted rows.
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_joint_mechanism_model(
        prepared,
        design="transition",
        mechanism_symbol=s["mechanism_symbol"],
        outcome_symbols=outcome_symbols,
        contrast=contrast,
        adjust_for=adjust_for,
        confounder_symbols=s["confounder_symbols"],
        include_group=s["include_group"],
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    for index, symbol in enumerate(outcome_symbols):
        stem = (
            "prior_predictive_check"
            if index == 0
            else f"prior_predictive_check_{symbol.lower()}"
        )
        _diag.save_prior_predictive_plot(ctx, symbol, filename_stem=stem)

    run_sampling_and_loo(ctx)

    model_names = {rv.name for rv in ctx.model.free_RVs} | set(
        ctx.trace.posterior.data_vars
    )
    diag_vars = _jm_diag_vars(
        model_names,
        design="transition",
        adjust_for=adjust_for,
        confounder_symbols=s["confounder_symbols"],
        include_group=s["include_group"],
    )
    psense_vars = [
        v
        for v in ("beta_mech", "delta_ls_decoding", "rho_outcome")
        if v in ctx.trace.posterior
    ]
    gate = _jm_standard_artefacts(
        ctx,
        outcome_symbols=outcome_symbols,
        diag_vars=diag_vars,
        psense_vars=psense_vars,
    )

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
                "bivariate (LKJ) child random intercept; identified "
                "decoding-specificity contrast"
            ),
            "joint_dependence": "lkj_child_intercept",
            "likelihood": "beta_binomial",
            "mechanism_symbol": s["mechanism_symbol"],
            "outcome_symbols": list(outcome_symbols),
            "contrast": list(contrast),
            "adjust_for": list(adjust_for),
            "matched_comparators": ["lrp-rli-mech-096", "lrp-rli-mech-101"],
            "output_contract": (
                "joint_mechanism_slopes.csv carries median + inner 50% + outer "
                "reporting interval + P(>0) for both slopes, their identified "
                "difference and the child-level residual correlation"
            ),
            "joint_mechanism_slopes": slopes_df.to_dict("records"),
        },
    )
    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Mechanism pipeline (LRP56 / LRP57 / LRP58)
# ---------------------------------------------------------------------------


def fit_mechanism(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    require_spec(spec, "mechanism", mechanism=True)

    ctx = make_context(spec, config)
    # Some mechanism fits keep the HSGP curve and need a higher target_accept for
    # the residual boundary divergences (LRP58/71/158); honour it with the shared
    # CLI > model-specific > preset precedence.

    section_header("Prepare data")
    # Data preparation and construction live in the family-owned ``mechanism``
    # module (#438) so that a leave-one-out refit for ``reloo`` builds the *same*
    # model as this fit rather than a re-derived lookalike. Behaviour-preserving
    # relocation: the loader-argument derivation, confounder filtering and factory
    # keyword mapping moved verbatim.
    plan = _mechanism.resolve_mechanism_plan(spec)
    prepared = plan.prepared
    ctx.prepared = prepared
    adjust_for = plan.adjust_for
    confounders = list(plan.confounders)
    moderator_symbol = spec.extra.get("moderator_symbol")
    mechanism_is_covariate = bool(spec.extra.get("mechanism_is_covariate", False))

    print_header(ctx)

    section_header("Build model")
    built = _mechanism.build_mechanism_for_plan(plan)
    attach_built(ctx, built)

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W")

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _mech_vars = ["alpha", "beta_G", "gamma_own", "kappa"]
    _mech_vars += [f"gamma_{s}" for s in confounders if s in MEASURES]
    _mech_vars += [f"gamma_{c}" for c in adjust_for]
    if "A" in confounders and not spec.extra.get("use_age_gp", False):
        _mech_vars.append("gamma_A")
    if spec.extra.get("use_subject_random_intercept", True):
        _mech_vars.append("sigma_child")
    if spec.extra.get("linear_mechanism", False):
        _mech_vars.append("beta_mech")
    if moderator_symbol is not None:
        _mech_vars.append("gamma_mod")
        if spec.extra.get("include_interaction", True):
            _mech_vars.append("gamma_int")
    _diag.summary_diagnostics(ctx, var_names=_mech_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381). For the
    # HSGP mechanism curve the estimand is the shape, governed by the deliberately
    # tight ``eta_main_prior`` amplitude the prior review flagged; the linear slope
    # ``beta_mech`` is already in ``_mech_vars``, so add the GP amplitude and
    # lengthscale only when the nonparametric curve is fitted.
    _mech_psense_vars = list(_mech_vars)
    if not spec.extra.get("linear_mechanism", False):
        _mech_psense_vars += ["f_mech__eta", "f_mech__ell"]
    _diag.run_psense(ctx, var_names=_mech_psense_vars)

    run_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=_mech_vars)
    _diag.run_extended_diagnostics(ctx)

    # Mechanism curve: f_mech vs mech_post_logit grid (logit-contribution scale only).
    section_header("Mechanism curve")
    _write_mechanism_curve(ctx)
    # Items-scale companion (#319): the same curve as exposure items -> predicted
    # outcome items, with a computed worked-example contrast. The worked dict is
    # folded into config.json below so the report partial renders the caption from
    # computed numbers.
    _items_worked = _write_mechanism_items(ctx)
    _write_readiness_threshold(ctx)

    # Record the adjustment set that was actually FITTED — with each term's source
    # column, measurement wave and missing-indicator status — not just the requested
    # symbols. ``spec.adjustment`` alone materially misdescribed the model, because
    # the ``adjust_for`` covariates never reached config.json (#258 review, P1).
    meta_extra = {
        "loo_elpd": float(ctx.loo.elpd),
        "adjustment": spec.adjustment,
        "effective_adjustment": effective_adjustment(
            spec,
            prepared,
            measure_confounders=tuple(
                s for s in confounders if s in ("G", "A") or s in MEASURES
            ),
            adjust_for=adjust_for,
            baseline_symbol=spec.extra.get("adjust_baseline_symbol", "W"),
        ),
    }
    # Items-scale worked-example reference points (#319): recorded so the caption
    # numbers are computed, not hand-written, and the quantiles are auditable.
    if _items_worked:
        meta_extra["mechanism_items"] = _items_worked
    if mechanism_is_covariate:
        # Record the exposure's raw-units anchor so a report can translate the
        # per-SD ``beta_mech`` into raw score points: the factory re-standardises
        # the loader z on the kept rows, so +1 SD of the fitted exposure is
        # ``loader_sd * sd(z_kept)`` raw points.
        meta_extra["mechanism_is_covariate"] = True
        _sc = ctx.prepared.covariate_scalers.get(spec.mechanism_symbol)
        if _sc is not None:
            _z_kept = np.asarray(
                ctx.prepared.covariates[spec.mechanism_symbol], dtype=float
            )
            meta_extra["mechanism_exposure_sd_raw"] = float(
                _sc.sd * np.nanstd(_z_kept, ddof=1)
            )
            meta_extra["mechanism_exposure_mean_raw"] = float(
                _sc.mean + _sc.sd * np.nanmean(_z_kept)
            )

    # Linear-moderation summary (gamma_int / gamma_mod), when a moderator is set.
    if moderator_symbol is not None:
        section_header("Interaction summary")
        gi = _report.gamma_interaction_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
        gi_df = pd.DataFrame([gi])
        save_table(ctx, "interaction_summary", gi_df)
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in gi.items()],
                title=(
                    f"Linear moderation by {moderator_symbol} "
                    f"- {int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["moderator_symbol"] = moderator_symbol
        meta_extra["interaction_summary"] = gi

    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_mech_vars)

    # Per-child fitted-vs-observed panels (#317 fig 2), one per period transition.
    write_child_fit(
        ctx,
        outcome_symbol=spec.outcome_symbol,
        wave=ctx.prepared.phase,
        child_idx=ctx.prepared.child_idx,
        off_floor=False,
        obs_node="y_post",
        x_label="period transition",
    )

    write_run_metadata(ctx, extra=meta_extra)

    return finalize_report(ctx)


def _write_mechanism_curve(ctx: StatisticalFitContext) -> None:
    """Posterior adjusted dose-response of the mechanism predictor on the outcome.

    With the HSGP ``f_mech`` on (the default) this is the non-parametric curve. When
    the model uses the linear slope instead (``linear_mechanism=True``, so no
    ``f_mech`` variable exists) it falls back to the straight
    ``beta_mech * z(logit(predictor))`` band — the predictor's linear logit
    contribution (at the mean of any moderator) — so the adjusted predictor->outcome
    relationship is still shown rather than left implicit in a coefficient. Both
    branches hold the adjustment set fixed and write the same CSV/PNG schema, except
    for the x column: ``mech_logit`` for a bounded-count measure exposure,
    ``mech_x`` (the raw covariate score) for a covariate exposure
    (``mechanism_is_covariate``, always linear). Guarded by the caller.
    """
    post = ctx.trace.posterior

    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    sym = ctx.spec.mechanism_symbol
    is_covariate = bool(ctx.spec.extra.get("mechanism_is_covariate", False))
    if is_covariate:
        # Covariate exposure: x is the raw score (the loader scaler inverted); the
        # model's z is the loader z re-standardised on the kept rows, exactly as
        # the factory did it.
        z_loaded = np.asarray(ctx.prepared.covariates[sym], dtype=float)
        _scaler = ctx.prepared.covariate_scalers.get(sym)
        x_vals = _scaler.inverse(z_loaded) if _scaler is not None else z_loaded
        z_L, _ = standardise(z_loaded)
        x_col, x_label = "mech_x", f"{sym} (raw score)"
    elif bool(ctx.spec.extra.get("mechanism_at_pre", False)):
        # Lagged form: the factory fits the mechanism on its period-start (pre)
        # logit, so the reported curve must use that same vector on the same rows.
        # Using the post logit here would plot and label the fitted pre-slope
        # against the wrong exposure — pre/post differ materially (#405 review).
        mech_logit = np.asarray(ctx.prepared.pre_logit[sym], dtype=float)
        x_vals = mech_logit
        z_L, _ = standardise(mech_logit)
        x_col, x_label = "mech_logit", f"logit({sym}_pre)"
    else:
        N = MEASURES[sym].n_trials
        mech_logit = logit_safe(ctx.prepared.post_counts[sym], N)
        x_vals = mech_logit
        # z the same standardisation the factory applied to the logit input.
        z_L, _ = standardise(mech_logit)
        x_col, x_label = "mech_logit", f"logit({sym}_post)"

    if "f_mech" in post:
        f = post["f_mech"].stack(sample=("chain", "draw")).values  # (n_obs, n_sample)
        kind = "GP"
    elif "beta_mech" in post:
        # Linear mechanism: the predictor enters as beta_mech * z. Build the
        # per-observation contribution so the band mirrors the GP branch (an exact
        # straight line).
        b = post["beta_mech"].stack(sample=("chain", "draw")).values  # (n_sample,)
        f = z_L[:, None] * b[None, :]  # (n_obs, n_sample)
        kind = "linear"
    else:
        # No f_mech / beta_mech in the posterior — e.g. a phase_specific_mechanism
        # fit, whose per-phase f_mech is not registered under either name, so the
        # curve would be silently skipped. Warn loudly rather than no-op (issue
        # #273); register the phase-specific curve as pm.Deterministic("f_mech",
        # ..., dims="obs_id") in the factory if such a model is ever shipped.
        rprint(
            "[yellow]_write_mechanism_curve: no 'f_mech'/'beta_mech' in the "
            f"posterior for {ctx.spec.model_id} (phase_specific_mechanism?); "
            "no mechanism_curve.csv/plot written.[/yellow]"
        )
        return

    order = np.argsort(x_vals)
    x = x_vals[order]
    f_ord = f[order]
    mean = f_ord.mean(axis=1)
    lo = np.quantile(f_ord, 0.055, axis=1)
    hi = np.quantile(f_ord, 0.945, axis=1)
    lo50 = np.quantile(f_ord, 0.25, axis=1)
    hi50 = np.quantile(f_ord, 0.75, axis=1)
    save_table(
        ctx,
        "mechanism_curve",
        pd.DataFrame(
            {x_col: x, "f_mean": mean, "f_lo": lo, "f_hi": hi,
             "f_lo50": lo50, "f_hi50": hi50}
        ),
        register=False,
    )
    outcome = ctx.spec.outcome_symbol or "W"

    # Preserve a posterior end-to-end contrast on the outcome-items scale for
    # the key-findings box (#320).  The contrast compares the lowest and highest
    # observed exposure values while setting any moderator to its standardised
    # mean (zero).  Removing the fitted mechanism and moderator contributions
    # from eta before adding the two endpoint contributions keeps every other
    # fitted row characteristic fixed and retains the posterior dependence that
    # the pointwise curve CSV alone cannot reconstruct.
    eta = (
        post["eta"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )
    eta_base = eta - f
    if "gamma_mod" in post and "z_moderator" in ctx.trace.constant_data:
        z_mod = np.asarray(ctx.trace.constant_data["z_moderator"].values).reshape(-1)
        gamma_mod = post["gamma_mod"].stack(sample=("chain", "draw")).values
        eta_base = eta_base - z_mod[:, None] * gamma_mod[None, :]
        if "gamma_int" in post:
            z_mech = np.asarray(
                ctx.trace.constant_data["z_mech_logit"].values
            ).reshape(-1)
            gamma_int = post["gamma_int"].stack(sample=("chain", "draw")).values
            eta_base = eta_base - (
                z_mech[:, None] * z_mod[:, None] * gamma_int[None, :]
            )
    endpoint_items = (
        expit(eta_base + f_ord[-1][None, :])
        - expit(eta_base + f_ord[0][None, :])
    ).mean(axis=0) * float(ctx.prepared.n_trials[outcome])
    lo_q = (1 - ctx.reporting.ci_prob) / 2
    if is_covariate:
        exposure_low = float(x[0])
        exposure_high = float(x[-1])
        exposure_unit = f"{sym} raw-score units"
    else:
        # Invert the Haldane-corrected logit used by preprocessing so the
        # headline exposure range is in test items, not log-odds.
        N = ctx.prepared.n_trials[sym]
        exposure_low = float(np.clip((N + 1) * expit(x[0]) - 0.5, 0, N))
        exposure_high = float(np.clip((N + 1) * expit(x[-1]) - 0.5, 0, N))
        exposure_unit = f"{sym} items"
    mechanism_summary = pd.DataFrame(
        [
            {
                "exposure_low": exposure_low,
                "exposure_high": exposure_high,
                "exposure_unit": exposure_unit,
                "items_median": float(np.median(endpoint_items)),
                "items_lo": float(np.quantile(endpoint_items, lo_q)),
                "items_hi": float(np.quantile(endpoint_items, 1 - lo_q)),
                "items_lo50": float(np.quantile(endpoint_items, 0.25)),
                "items_hi50": float(np.quantile(endpoint_items, 0.75)),
                "prob_pos": float(np.mean(endpoint_items > 0)),
            }
        ]
    )
    save_table(ctx, "mechanism_summary", mechanism_summary)
    plt.figure(figsize=FIGSIZE_LG)
    plt.plot(x, mean, color=COLOUR_BLUE, lw=2)
    plt.fill_between(x, lo, hi, color=COLOUR_BLUE, alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel("predictor logit contribution")
    plt.title(f"Mechanism curve ({kind}): {sym} -> {outcome}")
    # mechanism_curve.csv (the plotted band) is written just above.
    save_styled_figure(ctx.output_dir, "mechanism_curve")


#: Friendly labels for covariate mechanism exposures (no ``Measure`` entry, so no
#: label registry). Falls back to the symbol for anything not listed.
_COVARIATE_EXPOSURE_LABELS = {
    "erbto": "Phonological memory (word/nonword repetition)",
    "deapp_c": "Speech production (DEAP)",
}


def _write_mechanism_items(ctx: StatisticalFitContext) -> dict:
    """Items-scale mechanism dose-response curve + worked example (#319).

    Companion to ``_write_mechanism_curve``: the logit-scale CSV/plot remain the
    analyst's object; this renders the same fitted curve on the items scale
    (exposure items -> predicted outcome items) with a credible ribbon and one
    computed worked-example contrast between fixed quantiles of the observed
    exposure. Returns the ``worked`` dict (quantile reference points + the
    computed caption) so ``fit_mechanism`` can persist it to ``config.json`` for
    the report partial. Never raises through the fit — a failure logs and returns
    ``{}``.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.mechanism_items import (
        write_mechanism_items_artifacts,
    )

    try:
        spec = ctx.spec
        sym = spec.mechanism_symbol
        outcome = spec.outcome_symbol or "W"
        is_covariate = bool(spec.extra.get("mechanism_is_covariate", False))

        if is_covariate:
            z_loaded = np.asarray(ctx.prepared.covariates[sym], dtype=float)
            scaler = ctx.prepared.covariate_scalers.get(sym)
            x_exposure = scaler.inverse(z_loaded) if scaler is not None else z_loaded
            exposure_label = _COVARIATE_EXPOSURE_LABELS.get(sym, sym)
            exposure_n_trials = None
        elif bool(spec.extra.get("mechanism_at_pre", False)):
            # Lagged form: the fitted curve (via z_mech_logit) is on the pre
            # exposure, so the items-scale x-axis must be the pre counts, not the
            # post counts — otherwise the worked-example quantiles land on the
            # wrong distribution and the axis is mislabelled (#405 review).
            x_exposure = np.asarray(ctx.prepared.pre_counts[sym], dtype=float)
            exposure_label = f"{MEASURES[sym].label} (period start)"
            exposure_n_trials = MEASURES[sym].n_trials
        else:
            x_exposure = np.asarray(ctx.prepared.post_counts[sym], dtype=float)
            exposure_label = MEASURES[sym].label
            exposure_n_trials = MEASURES[sym].n_trials

        # The mechanism factory always fits a Beta-Binomial likelihood, so the
        # y-axis is an item count. Floored (off-floor Bernoulli) mechanism
        # outcomes are a future addition (#319 design note); wire the flag when
        # such a model ships.
        ref_quantiles = tuple(spec.extra.get("items_ref_quantiles", (0.25, 0.75)))
        worked = write_mechanism_items_artifacts(
            ctx.output_dir,
            ctx.trace,
            x_exposure=x_exposure,
            outcome_symbol=outcome,
            outcome_label=MEASURES[outcome].label,
            n_trials_outcome=MEASURES[outcome].n_trials,
            exposure_label=exposure_label,
            exposure_is_covariate=is_covariate,
            exposure_n_trials=exposure_n_trials,
            ci_prob=ctx.reporting.ci_prob,
            ref_quantiles=ref_quantiles,
            outcome_off_floor=False,
        )
        _write_mechanism_prior_pushforward(
            ctx,
            x_exposure=x_exposure,
            outcome=outcome,
            exposure_label=exposure_label,
            ref_quantiles=ref_quantiles,
        )
        # ``mechanism_curve_items.csv`` is written inside the helper (which takes
        # an output directory, not a context); record it for the manifest.
        record_artifact(ctx, "mechanism_curve_items", required=False)
        return worked
    except Exception as exc:  # pragma: no cover - defensive; logit curve stands alone
        rprint(f"[yellow]Items-scale mechanism curve failed: {exc}[/yellow]")
        write_prior_pushforward(
            ctx,
            [
                _report.unavailable_pushforward(
                    estimand="mechanism_curve",
                    estimand_label="the mechanism dose-response contrast",
                    role="association",
                    reason=f"the items-scale mechanism curve could not be built: {exc}",
                )
            ],
        )
        return {}


def _write_mechanism_prior_pushforward(
    ctx: StatisticalFitContext,
    *,
    x_exposure: np.ndarray,
    outcome: str,
    exposure_label: str,
    ref_quantiles: tuple[float, float],
) -> None:
    """Estimand-scale prior check for the mechanism family (#381).

    The mechanism deliverable is a worked contrast — the predicted items-scale
    difference between two fixed quantiles of the observed exposure — so that,
    not a coefficient, is what the prior has to be pushed through. Runs
    :func:`mechanism_items.mechanism_items_curve` on the ``prior`` group, which
    reconstructs the HSGP ``f_mech`` curve or the linear ``beta_mech`` slope by
    the same route as the posterior version. This is the check the prior-analysis
    review asked for most directly: the GP amplitude prior is deliberately tight,
    and its implied curve range is what says whether a flat fitted curve is
    evidence of no dose-response or an artefact of the prior.

    Never raises: this rides on the items-curve writer, and a prior check that
    could abort the fitted curve it accompanies would trade a bigger deliverable
    for a smaller one.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.mechanism_items import (
        mechanism_items_curve,
    )

    label = "the mechanism dose-response contrast"
    try:
        n_trials = MEASURES[outcome].n_trials
        q_lo = int(round(100 * ref_quantiles[0]))
        q_hi = int(round(100 * ref_quantiles[1]))
        label = (
            f"the predicted difference on {MEASURES[outcome].label} between the "
            f"{q_hi}th and {q_lo}th percentile of {exposure_label}"
        )
        source = getattr(ctx, "prior_samples", None) or ctx.trace
        _, worked = mechanism_items_curve(
            source,
            x_exposure=x_exposure,
            n_trials_outcome=n_trials,
            ci_prob=ctx.reporting.ci_prob,
            ref_quantiles=ref_quantiles,
            group="prior",
        )
        # ``logit_difference_*`` is the curve's rise between the two quantiles on
        # the linear-predictor scale and ``outcome_difference_*`` the same contrast
        # in items, so the two scales describe one quantity rather than two.
        rows = [
            _report.labelled_pushforward(
                {
                    "prior_logit_median": worked["logit_difference_median"],
                    "prior_logit_lo": worked["logit_difference_lo"],
                    "prior_logit_hi": worked["logit_difference_hi"],
                    "prior_items_median": worked["outcome_difference_median"],
                    "prior_items_lo50": worked["outcome_difference_lo50"],
                    "prior_items_hi50": worked["outcome_difference_hi50"],
                    "prior_items_lo": worked["outcome_difference_lo"],
                    "prior_items_hi": worked["outcome_difference_hi"],
                    "n_trials": int(n_trials),
                },
                estimand=f"mechanism_curve ({worked['curve_kind']})",
                estimand_label=label,
                role="association",
            )
        ]
    except Exception as exc:  # noqa: BLE001 - absence must stay legible
        rows = [
            _report.unavailable_pushforward(
                estimand="mechanism_curve",
                estimand_label=label,
                role="association",
                reason=str(exc),
            )
        ]
    with guard_optional(
        ctx, "mechanism prior pushforward",
        filename="prior_pushforward.csv", kind="table", verb="not written",
    ):
        write_prior_pushforward(ctx, rows)


def _write_readiness_threshold(ctx: StatisticalFitContext) -> None:
    """Readiness-threshold summary for the mechanism curve (#230 §2/§5).

    Post-processes the fitted nonparametric mechanism curve (``f_mech``) into a
    posterior for the predictor count at which the outcome rises *fastest* — the
    "knee" (the steepest rise, not the onset), via
    :func:`reporting.readiness_threshold`. Only the GP mechanism has a curve to
    find a knee in; linear / phase-specific fits (no ``f_mech``) are skipped
    quietly. Writes ``readiness_threshold.csv`` and a plot. Guarded by the
    caller.
    """
    post = ctx.trace.posterior
    if "f_mech" not in post:
        return

    from language_reading_predictors.statistical_models.measures import MEASURES

    sym = ctx.spec.mechanism_symbol
    outcome = ctx.spec.outcome_symbol or "W"
    is_covariate = bool(ctx.spec.extra.get("mechanism_is_covariate", False))
    f = post["f_mech"].stack(sample=("chain", "draw")).values  # (n_obs, n_sample)

    if is_covariate:
        # Continuous-covariate exposure (e.g. LRP92 sessions): locate the knee in the
        # exposure's own raw units (scaler-inverted, as in _write_mechanism_curve),
        # not a bounded count. The per-obs exposure aligns with f_mech's row order.
        z_loaded = np.asarray(ctx.prepared.covariates[sym], dtype=float)
        scaler = ctx.prepared.covariate_scalers.get(sym)
        x_obs = scaler.inverse(z_loaded) if scaler is not None else z_loaded
        try:
            summary = _report.readiness_threshold(
                ctx.trace, exposure_values=x_obs, ci_prob=ctx.reporting.ci_prob
            )
        except ValueError as exc:
            rprint(f"[yellow]_write_readiness_threshold: {exc}; skipped.[/yellow]")
            return
        x_label = f"{sym} (raw score)"
    else:
        N = MEASURES[sym].n_trials
        try:
            summary = _report.readiness_threshold(
                ctx.trace, n_trials=N, ci_prob=ctx.reporting.ci_prob
            )
        except ValueError as exc:
            rprint(f"[yellow]_write_readiness_threshold: {exc}; skipped.[/yellow]")
            return
        # Mean curve on the raw count scale (inverse Haldane-corrected logit, as in
        # reporting._readiness_knee) with the knee posterior overlaid.
        ell = np.asarray(ctx.trace.constant_data["mech_post_logit"].values).reshape(-1)
        x_obs = np.clip((N + 1.0) / (1.0 + np.exp(-ell)) - 0.5, 0.0, float(N))
        x_label = f"{sym} (raw count, out of {N})"

    save_table(ctx, "readiness_threshold", pd.DataFrame([summary]), register=False)

    order = np.argsort(x_obs)
    x = x_obs[order]
    mean = f[order].mean(axis=1)
    plt.figure(figsize=FIGSIZE_LG)
    plt.plot(x, mean, color=COLOUR_BLUE, lw=2)
    plt.axvspan(
        summary["knee_count_ci_low"],
        summary["knee_count_ci_high"],
        color=COLOUR_RED,
        alpha=0.15,
        label=f"knee {int(round(ctx.reporting.ci_prob * 100))}% CI",
    )
    plt.axvline(
        summary["knee_count_median"], color=COLOUR_RED, lw=1.5, label="knee median"
    )
    plt.xlabel(x_label)
    plt.ylabel(f"{outcome} logit contribution")
    plt.title(f"Readiness threshold (steepest rise): {sym} -> {outcome}")
    plt.legend(fontsize=8)
    save_styled_figure(ctx.output_dir, "readiness_threshold")


# ---------------------------------------------------------------------------
# Mediation pipeline (LRP59)
# ---------------------------------------------------------------------------

_T3_SENSITIVITY_TIME = 3  # post-RCT wave used for the temporal-ordering check


def _fit_t3_sensitivity(
    ctx: StatisticalFitContext,
    spec: ModelSpec,
    *,
    confounders: tuple[str, ...],
    mediator_kind: str,
    route_symbols: tuple[str, ...],
):
    """Temporal-ordering sensitivity fit for the mediation models (issue #84).

    Refits the *identical* mediation model but with the outcome measured at a
    later wave (t3) while the mediator stays at t2, so the mediator precedes the
    outcome in time. The t2 -> t3 increment is **not randomised** (both arms are
    treated after t2), so this is a triangulation point for the contemporaneous
    measurement caveat, not a cleaner causal estimate. Returns the g-formula
    decomposition DataFrame for the t3-outcome variant.
    """
    import pymc as pm

    from language_reading_predictors.statistical_models import mediation as _med

    outcome_symbol = spec.outcome_symbol or "W"
    # Match the primary fit's load set so a mediator/confounder outside
    # ITT_OUTCOMES (TE, N) is present in the lagged-outcome frame too.
    _extra_outcomes = spec.extra.get("outcomes")
    _lag_kwargs = (
        {"outcomes": tuple(_extra_outcomes)} if _extra_outcomes is not None else {}
    )
    prepared_t3 = load_and_prepare_lagged_outcome(
        outcome_symbol,
        outcome_time=_T3_SENSITIVITY_TIME,
        covariates=_raw_covariate_confounders(confounders),
        **_lag_kwargs,
    )
    built_t3, med_t3 = _factories.build_mediation_model(
        prepared_t3,
        mediator_symbol=spec.mechanism_symbol or "L",
        outcome_symbol=outcome_symbol,
        confounder_symbols=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
    )
    s = ctx.sampling
    with built_t3.model:
        trace_t3 = pm.sample(
            draws=s.draws,
            tune=s.tune,
            chains=s.chains,
            cores=s.cores,
            target_accept=s.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=s.random_seed,
            progressbar=False,
        )
    # Gate this temporal-ordering sensitivity sub-fit (bypasses the primary gate).
    conv = _diag.subfit_convergence(trace_t3, label=f"{spec.model_id} t3 sensitivity")
    df_t3 = _med.decompose(
        trace_t3,
        med_t3,
        ci_prob=ctx.reporting.ci_prob,
    )
    # Persist the verdict onto the published rows: this sub-fit bypasses the primary
    # gate, and the verdict was previously computed then discarded so the t3 table
    # shipped with no convergence flag (this review's finding B1). Flows through to
    # both mediation_summary_t3.csv and the mediation_t3_sensitivity metadata block.
    df_t3["converged"] = conv["converged"]
    return df_t3


def prepare_mediation_data(spec: ModelSpec):
    """Load the exact rows and fitted confounder set for a mediation spec.

    Kept separate from sampling so reporting-only regenerators can reconstruct the
    mediation sample and its mediator standardiser without refitting the posterior.
    """
    require_spec(spec, "mediation")
    # Phase 0 only (t1 -> t2): the single randomised contrast. One row per child.
    mediator_symbol = spec.mechanism_symbol or "L"
    # Drop the structural markers and the mediator's own baseline ({mediator}_t1,
    # handled inside the factory) from the adjustment set; the rest are confounders.
    # The set mixes bounded-count skill measures (E, R — arriving via pre_logit) and
    # revised-DAG RAW covariates (hearing ``hs``/``hs_missing``, speech ``deapp_c``,
    # phonological memory ``erbto`` + indicators; #246), which must be requested as
    # covariates and are taken from the t1 pre-row (treatment-unaffected). Models
    # with no raw covariates get ``covariates=()`` — a no-op, so LRP59/62/64/66 and
    # the #263 mediation family are unchanged unless a spec adds raw confounders.
    confounders = tuple(
        s
        for s in spec.adjustment
        if s not in ("G", "A", "W_pre", f"{mediator_symbol}_t1")
    )
    _raw_cov = _raw_covariate_confounders(confounders)
    # A mediator or confounder outside ``ITT_OUTCOMES`` (e.g. taught-expressive TE,
    # nonword N) must be requested via ``extra["outcomes"]`` so it is loaded; this
    # also restricts the complete-case mask to the symbols the model uses (mirrors
    # fit_itt).
    _extra_outcomes = spec.extra.get("outcomes")
    _outcome_time = spec.extra.get("outcome_time")
    if _outcome_time is not None:
        # Longitudinal-ordering primary fit (LRP76): the mediator stays at t2 but
        # the outcome is taken from a later wave (t3/t4), so the mediator strictly
        # precedes the outcome — promoting the temporal-ordering check from a
        # sensitivity to the primary estimand. The t2 -> t{outcome_time} increment
        # is NOT randomised (both arms treated after t2), so this is a
        # triangulation design, read under stated assumptions, not a cleaner τ.
        _lag_outcomes = (
            tuple(_extra_outcomes) if _extra_outcomes is not None else ITT_OUTCOMES
        )
        prepared = load_and_prepare_lagged_outcome(
            spec.outcome_symbol or "W",
            outcome_time=int(_outcome_time),
            outcomes=_lag_outcomes,
            covariates=_raw_cov,
        )
    elif _extra_outcomes is not None:
        prepared = load_and_prepare(
            phase_mode="itt",
            outcomes=tuple(_extra_outcomes),
            covariates=_raw_cov,
            drop_missing_pre=bool(spec.extra.get("drop_missing_pre", True)),
        )
    else:
        prepared = load_and_prepare(phase_mode="itt", covariates=_raw_cov)
    # A missing-indicator can be constant on the ITT-phase rows (SP/RW are near-
    # complete at t1) and be dropped by the loader; keep only confounders actually
    # present, so no vacuous coefficient is fitted for a dropped covariate.
    confounders = tuple(
        c for c in confounders if c in prepared.covariates or c in prepared.pre_logit
    )
    return prepared, confounders


def fit_mediation(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """ITT-phase mediation decomposition (LRP59): how much of G -> W flows via L."""
    require_spec(spec, "mediation")
    from language_reading_predictors.statistical_models import mediation as _med

    ctx = make_context(spec, config)

    section_header("Prepare data")
    mediator_symbol = spec.mechanism_symbol or "L"
    _outcome_time = spec.extra.get("outcome_time")
    prepared, confounders = prepare_mediation_data(spec)
    ctx.prepared = prepared

    print_header(ctx)

    section_header("Build model")

    mediator_kind = spec.extra.get("mediator_kind", "beta_binomial")
    route_symbols = tuple(spec.extra.get("route_symbols", ()))
    # Off-floor (Bernoulli) OUTCOME for a heavily-floored outcome such as nonword N
    # (#228 item 12): the outcome leg becomes a Bernoulli on the off-floor indicator
    # (node "y_offfloor") and the g-formula reports NIE/NDE on the off-floor
    # risk-difference scale. Default "beta_binomial" keeps every existing med model
    # byte-identical.
    outcome_kind = spec.extra.get("outcome_kind", "beta_binomial")
    off_floor = outcome_kind == "bernoulli_offfloor"
    outcome_node = "y_offfloor" if off_floor else "y_post"
    built, med_data = _factories.build_mediation_model(
        prepared,
        mediator_symbol=mediator_symbol,
        outcome_symbol=spec.outcome_symbol or "W",
        confounder_symbols=confounders,
        mediator_kind=mediator_kind,
        route_symbols=route_symbols,
        outcome_kind=outcome_kind,
    )
    attach_built(ctx, built)

    # The mediator observed node differs by kind: Beta-Binomial "{mediator}_post"
    # vs the Gaussian composite "M_post".
    is_gaussian = mediator_kind == "gaussian_composite"
    mediator_node = "M_post" if is_gaussian else f"{mediator_symbol}_post"
    # Diagnose every scalar coefficient the model actually built (deterministics
    # and the observed mediator/outcome nodes are not free RVs), so the list
    # tracks the fitted confounder set instead of a hand-maintained constant.
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # The mediator likelihood is the FIRST observed RV, so name the outcome node
    # explicitly — else the plot overlays mediator draws on the outcome's counts.
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W", node=outcome_node)

    run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=coef_vars)
    # Power-scaling prior sensitivity (#381): this family does not compute LOO,
    # so add the log groups explicitly, then power-scale the reported parameters.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=coef_vars)

    run_ppc(ctx, var_names=[mediator_node, outcome_node])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=coef_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=coef_vars)

    section_header("Mediation decomposition (g-formula)")
    _interventional = spec.extra.get("estimand") == "interventional"
    med_df = _med.decompose(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        interventional=_interventional,
    )
    save_table(ctx, "mediation_summary", med_df)
    # Print the primary decomposition table before the (slow, ~21x-decompose) sensitivity
    # sweep, so the main NDE/NIE result shows under its own section header rather than
    # under the sensitivity header and only after the sweep finishes (#289 review).
    print_table(
        ranked_dataframe_table(
            med_df,
            title=(
                "Mediation (intervention-helps; off-floor risk difference)"
                if off_floor
                else f"Mediation (intervention-helps; items out of {med_data.n_trials_W})"
            ),
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # Unmeasured mediator-outcome confounding sensitivity for the NIE (#230): sweep a
    # bias off b_M and report the tipping point at which the indirect effect's CI
    # includes 0 (a Bayesian E-value analogue). Quantifies the no-unmeasured-
    # confounding assumption the decomposition otherwise only states.
    section_header("Mediation NIE sensitivity (unmeasured confounding)")
    sens_sweep, sens_summary = _med.sensitivity_sweep(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        interventional=_interventional,
    )
    save_table(ctx, "mediation_sensitivity", sens_sweep)
    save_table(
        ctx,
        "mediation_sensitivity_summary",
        pd.DataFrame([sens_summary]),
        register=False,
    )
    if sens_summary["already_null_at_zero"]:
        rprint(
            "  NIE not credibly nonzero at delta=0 — sensitivity analysis N/A "
            "(no indirect effect to explain away)."
        )
    elif sens_summary["robust_over_full_sweep"]:
        rprint(
            f"  NIE robust across the full sweep (CI excludes 0 up to "
            f"delta={sens_sweep['delta'].max():.2f} logit)."
        )
    else:
        rprint(
            f"  NIE tipping point delta*={sens_summary['tipping_delta']:.3f} logit "
            f"({sens_summary['tipping_frac_of_bM']:.0%} of the fitted b_M+b_GM) — an "
            "unmeasured mediator-outcome confounder that strong would null the NIE."
        )

    # Named-confounder anchor (#324): place the fitted/observed intervention-session
    # associations on the abstract delta surface.  Only the signed-off L-mediator
    # code-route targets produce this artefact; missing source fits degrade to an
    # explicit not-available row and never abort the mediation fit.
    from language_reading_predictors.statistical_models import (
        mediation_calibration as _med_cal,
    )

    is_calibration = _med_cal.generate_is_calibration(
        spec,
        config=config,
        output_dir=ctx.output_dir,
        prepared=ctx.prepared,
        med=med_data,
        sweep=sens_sweep,
        sensitivity_summary=sens_summary,
    )
    if is_calibration is not None:
        save_table(ctx, "mediation_is_calibration", is_calibration)
        cal = is_calibration.iloc[0]
        if cal["status"] == "ok":
            rprint(f"  IS calibration: {cal['verdict']} (delta={cal['delta_is_point']:.3f})")
        else:
            rprint(f"  IS calibration: not available ({cal.get('reason', 'unknown reason')})")

    # --- Temporal-ordering sensitivity: outcome at t3, mediator still at t2 ---
    # Triangulation for the contemporaneous-measurement caveat (issue #84): the
    # mediator now precedes the outcome in time. NB the t2 -> t3 increment is not
    # randomised (both arms treated after t2), so read this as triangulation only.
    # Skipped when the primary fit is ALREADY longitudinal (outcome_time set, LRP76)
    # — the sensitivity would double-lag and duplicate the primary estimand.
    med_df_t3 = None
    if _outcome_time is None and not _interventional:
        section_header("Temporal-ordering sensitivity (outcome at t3)")
        med_df_t3 = _fit_t3_sensitivity(
            ctx,
            spec,
            confounders=confounders,
            mediator_kind=mediator_kind,
            route_symbols=route_symbols,
        )
        save_table(ctx, "mediation_summary_t3", med_df_t3)
        print_table(
            ranked_dataframe_table(
                med_df_t3,
                title="Temporal-ordering sensitivity (outcome W at t3; NOT randomised)",
                columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    _summary = {r["quantity"]: r for r in med_df.to_dict("records")}
    # Record the REQUESTED adjustment set and the confounders ACTUALLY fitted
    # separately (#246 review, P2). A raw covariate can be dropped by the loader
    # when its missing-indicator is constant on the ITT rows; recording only
    # ``spec.adjustment`` would then imply a coefficient that was never estimated.
    _requested_raw = _raw_covariate_confounders(
        s for s in spec.adjustment if s not in ("G", "A", "W_pre", f"{mediator_symbol}_t1")
    )
    _extra_meta = {
        "adjustment": spec.adjustment,
        "effective_confounders": list(confounders),
        "dropped_confounders": [c for c in _requested_raw if c not in confounders],
        "estimand": "interventional" if _interventional else "natural",
        "outcome_kind": outcome_kind,
        "companion_of": spec.extra.get("companion_of"),
        "n_obs": prepared.n_obs,
        "mediation": _summary,
    }
    if med_df_t3 is not None:
        _extra_meta["mediation_t3_sensitivity"] = {
            r["quantity"]: r for r in med_df_t3.to_dict("records")
        }
    if _outcome_time is not None:
        _extra_meta["outcome_time"] = int(_outcome_time)
    if is_calibration is not None:
        _extra_meta["is_calibration"] = is_calibration.iloc[0].to_dict()
    write_run_metadata(ctx, extra=_extra_meta)

    return finalize_report(ctx)


def fit_mediation_period_stacked(
    spec: ModelSpec, config: str = "dev"
) -> StatisticalFitContext:
    """Period-stacked g-formula mediation on the gain-factor scaffold (MED-092, #229).

    The LRP59 mediator + outcome design refit over **all stacked period
    transitions** (``phase_mode="all"``), with the per-period on-intervention
    indicator as the exposure and the gain-factor machinery (phase intercepts,
    per-leg child random intercepts). Writes the all-period decomposition to
    ``mediation_summary.csv`` and the period-1 (ITT-anchored, LRP59-comparable)
    row restriction to ``mediation_summary_p1.csv``. No t3 temporal-ordering
    sensitivity is fitted — the stacked design already spans every window, and
    its mediator/outcome remain contemporaneous within each period by design.
    The #324 named-IS calibration deliberately excludes this model: its exposure is
    an ignorability-based per-period treatment indicator, not the randomised phase-0
    group used by the single- and two-mediator calibrations. Importing their
    treated-arm benchmark here would silently change its interpretation (#335
    placement decision).
    """
    require_spec(spec, "mediation")
    from language_reading_predictors.statistical_models import mediation as _med

    ctx = make_context(spec, config)

    section_header("Prepare data")
    mediator_symbol = spec.mechanism_symbol or "L"
    outcome_symbol = spec.outcome_symbol or "W"
    # Structural markers aside, the adjustment list is the confounder set; the
    # raw covariates take the gain-factor timing split (hearing contemporaneous,
    # speech/phonological memory at the t1 baseline — the A1 timing decision).
    confounders = tuple(
        s
        for s in spec.adjustment
        if s not in ("T", "A", "W_pre", f"{mediator_symbol}_pre")
    )
    raw_cov = _raw_covariate_confounders(confounders)
    pre_adj, post_adj = split_covariates_by_wave(raw_cov)
    baseline_adj, post_adj = split_confounders_by_timing(post_adj)
    measure_confounders = tuple(c for c in confounders if c not in raw_cov)
    prepared = load_and_prepare(
        phase_mode="all",
        outcomes=(outcome_symbol, mediator_symbol, *measure_confounders),
        covariates=pre_adj,
        post_covariates=post_adj,
        baseline_covariates=baseline_adj,
    )
    ctx.prepared = prepared
    # Keep only confounders actually present (a constant ``_missing`` indicator
    # is dropped by the loader and gets no coefficient).
    confounders = tuple(
        c for c in confounders if c in prepared.covariates or c in prepared.pre_logit
    )

    print_header(ctx)

    section_header("Build model")
    built, med_data = _factories.build_period_stacked_mediation_model(
        prepared,
        mediator_symbol=mediator_symbol,
        outcome_symbol=outcome_symbol,
        confounder_symbols=confounders,
    )
    attach_built(ctx, built)

    mediator_node = f"{mediator_symbol}_post"
    # Scalar coefficients from the model itself, plus the per-phase intercept
    # vectors (the convergence gate scans every free RV regardless).
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)
    diag_vars = [*coef_vars, "a_phase", "b_phase"]

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome_symbol, node="y_post")

    run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity (#381): this family does not compute LOO,
    # so add the log groups explicitly, then power-scale the reported parameters.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=[mediator_node, "y_post"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Mediation decomposition (period-stacked g-formula)")
    med_df = _med.decompose_period_stacked(
        ctx.trace, med_data, ci_prob=ctx.reporting.ci_prob
    )
    save_table(ctx, "mediation_summary", med_df)
    print_table(
        ranked_dataframe_table(
            med_df,
            title=(
                "Per-period mediation, all stacked periods "
                f"(on-intervention; words out of {med_data.n_trials_W})"
            ),
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # Period-1 restriction: the same posterior averaged over the randomised,
    # all-untreated-baseline transition only — the LRP59-comparable readout
    # (mirrors the gain-factor family's period-1 treatment marginal, #247 P2).
    med_df_p1 = _med.decompose_period_stacked(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        row_mask=med_data.phase_idx == 0,
    )
    save_table(ctx, "mediation_summary_p1", med_df_p1)
    print_table(
        ranked_dataframe_table(
            med_df_p1,
            title="Period-1 restriction (randomised window; LRP59-comparable)",
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    section_header("Mediation NIE sensitivity (unmeasured confounding)")
    sens_sweep, sens_summary = _med.sensitivity_sweep(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        decompose_fn=_med.decompose_period_stacked,
        interaction_name="b_trtM",
    )
    save_table(ctx, "mediation_sensitivity", sens_sweep)
    save_table(
        ctx,
        "mediation_sensitivity_summary",
        pd.DataFrame([sens_summary]),
        register=False,
    )
    if sens_summary["already_null_at_zero"]:
        rprint(
            "  NIE not credibly nonzero at delta=0 — sensitivity analysis N/A "
            "(no indirect effect to explain away)."
        )
    elif sens_summary["robust_over_full_sweep"]:
        rprint(
            f"  NIE robust across the full sweep (CI excludes 0 up to "
            f"delta={sens_sweep['delta'].max():.2f} logit)."
        )
    else:
        rprint(
            f"  NIE tipping point delta*={sens_summary['tipping_delta']:.3f} logit "
            f"({sens_summary['tipping_frac_of_bM']:.0%} of the fitted b_M+b_trtM) — an "
            "unmeasured mediator-outcome confounder that strong would null the NIE."
        )

    _requested_raw = _raw_covariate_confounders(
        s for s in spec.adjustment if s not in ("T", "A", "W_pre", f"{mediator_symbol}_pre")
    )
    write_run_metadata(
        ctx,
        extra={
            "adjustment": spec.adjustment,
            "effective_confounders": list(confounders),
            "dropped_confounders": [c for c in _requested_raw if c not in confounders],
            "n_obs": prepared.n_obs,
            "exposure": "on_intervention (per-period; gain-factor ignorability)",
            "mediation": {r["quantity"]: r for r in med_df.to_dict("records")},
            "mediation_p1": {r["quantity"]: r for r in med_df_p1.to_dict("records")},
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Two-mediator decomposition pipeline (LRP64)
# ---------------------------------------------------------------------------


def fit_mediation_multi(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """ITT-phase two-mediator decomposition (LRP64): G -> W via letter-sound and vocab.

    Mirrors :func:`fit_mediation` but builds the two-mediator joint model
    (:func:`factories.build_two_mediator_model`) and runs the two-mediator
    g-formula (:func:`mediation.decompose_two_mediator`), reporting the joint
    indirect effect as the headline plus the (ordering-dependent) path-specific
    indirect effects.
    """
    require_spec(spec, "mediation_multi")
    from language_reading_predictors.statistical_models import mediation as _med

    ctx = make_context(spec, config)

    section_header("Prepare data")
    # Phase 0 only (t1 -> t2): the single randomised contrast. One row per child.
    mediators = tuple(spec.extra.get("mediators", ("L", "E")))
    # Drop the structural symbols and the two mediator baselines ({m}_t1) from the
    # adjustment set; whatever remains are the measured mediator-outcome confounders
    # C. Keyed off ``mediators`` so a non-(L, E) pair excludes its own baselines
    # (LRP64 -> L_t1/E_t1; LRP66 -> L_t1/B_t1). The set mixes bounded-count measures
    # (E, R — via pre_logit) and revised-DAG raw covariates (hs/deapp_c/erbto; #246 —
    # requested as covariates, taken from the t1 pre-row); ``covariates=()`` is a
    # no-op for models with no raw confounders.
    _mediator_baselines = tuple(f"{m}_t1" for m in mediators)
    confounders = tuple(
        s
        for s in spec.adjustment
        if s not in ("G", "A", "W_pre", *_mediator_baselines)
    )
    _raw_cov = _raw_covariate_confounders(confounders)
    _calibration = spec.extra.get("named_confounder_calibration")
    _calibration_symbol = (
        str(_calibration.get("symbol", "attend")) if _calibration else None
    )
    # A named-confounder calibration needs the observed covariate but must not add
    # it to the fitted natural-effects model: IS is treatment-affected, so
    # conditioning on it would not identify the NDE/NIE. It is loaded only for the
    # post-fit, treated-arm omitted-variable-bias benchmark (#335).
    _loaded_cov = tuple(
        dict.fromkeys(
            [*_raw_cov, *([_calibration_symbol] if _calibration_symbol else [])]
        )
    )
    # A floored second mediator (e.g. nonword decoding N, med-081) is not in the
    # default ITT outcome set, so load exactly the requested outcomes when given.
    _load_outcomes = spec.extra.get("outcomes")
    if _load_outcomes is not None:
        prepared = load_and_prepare(
            phase_mode="itt", covariates=_loaded_cov, outcomes=tuple(_load_outcomes)
        )
    else:
        prepared = load_and_prepare(phase_mode="itt", covariates=_loaded_cov)
    # Drop any missing-indicator constant on the ITT-phase rows (see fit_mediation).
    confounders = tuple(
        c for c in confounders if c in prepared.covariates or c in prepared.pre_logit
    )
    ctx.prepared = prepared

    print_header(ctx)

    section_header("Build model")

    second_offfloor = bool(spec.extra.get("second_mediator_offfloor", False))
    built, med_data = _factories.build_two_mediator_model(
        prepared,
        outcome_symbol=spec.outcome_symbol or "W",
        mediator_symbols=mediators,
        confounder_symbols=confounders,
        chain=bool(spec.extra.get("chain", False)),
        second_mediator_offfloor=second_offfloor,
    )
    attach_built(ctx, built)

    # Diagnose every scalar coefficient the model actually built, so the list
    # tracks the fitted confounder set instead of a hand-maintained constant
    # (mirrors fit_mediation).
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # The mediator likelihood is the FIRST observed RV, so name the outcome node
    # explicitly — else the plot overlays mediator draws on the outcome's counts.
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W", node="y_post")

    run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=coef_vars)
    # Power-scaling prior sensitivity (#381): this family does not compute LOO,
    # so add the log groups explicitly, then power-scale the reported parameters.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=coef_vars)

    _m2_node = f"{mediators[1]}_offfloor" if second_offfloor else f"{mediators[1]}_post"
    run_ppc(ctx, var_names=[f"{mediators[0]}_post", _m2_node, "y_post"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=coef_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=coef_vars)

    section_header("Two-mediator decomposition (g-formula)")
    med_df = _med.decompose_two_mediator(
        ctx.trace,
        med_data,
        hdi_prob=ctx.reporting.ci_prob,
        order=tuple(spec.extra.get("order", ("L", "E"))),
    )
    save_table(ctx, "mediation_summary", med_df)
    print_table(
        ranked_dataframe_table(
            med_df,
            title=(
                f"Two-mediator decomposition (intervention-helps; words out of "
                f"{med_data.n_trials_W})"
            ),
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    section_header("Per-leg NIE sensitivity (unmeasured confounding)")
    sens_sweep, sens_summary = _med.sensitivity_sweep_two_mediator(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        order=tuple(spec.extra.get("order", ("L", "E"))),
    )
    save_table(ctx, "mediation_sensitivity", sens_sweep)
    save_table(ctx, "mediation_sensitivity_summary", sens_summary)
    for row in sens_summary.to_dict("records"):
        mediator = row["mediator"]
        if row["already_null_at_zero"]:
            rprint(
                f"  NIE_{mediator} is not credibly nonzero at delta=0 — no "
                "non-zero path-specific effect to explain away."
            )
        elif row["robust_over_full_sweep"]:
            max_delta = sens_sweep.loc[
                sens_sweep["mediator"] == mediator, "delta"
            ].max()
            rprint(
                f"  NIE_{mediator} remains nonzero across its full sweep "
                f"(delta <= {max_delta:.2f} logit)."
            )
        else:
            rprint(
                f"  NIE_{mediator} tipping point delta*={row['tipping_delta']:.3f} "
                f"({row['tipping_frac_of_effective_slope']:.0%} of the fitted "
                "treatment-arm mediator->outcome slope)."
            )
        if not row["joint_already_null_at_zero"]:
            if row["joint_robust_over_full_sweep"]:
                rprint(
                    f"  NIE_joint remains nonzero across the {mediator}-leg sweep."
                )
            else:
                rprint(
                    f"  NIE_joint reaches zero at delta="
                    f"{row['joint_tipping_delta']:.3f} when attenuating the "
                    f"{mediator} leg."
                )

    calibration_df = None
    if _calibration_symbol:
        section_header("Named-confounder calibration (intervention sessions)")
        calibration_df = _med.calibrate_session_confounding(
            built.prepared,
            med_data,
            sens_summary,
            session_symbol=_calibration_symbol,
        )
        save_table(ctx, "mediation_is_calibration", calibration_df)
        for conclusion in calibration_df["conclusion"]:
            rprint(f"  {conclusion}")

    _summary = {r["quantity"]: r for r in med_df.to_dict("records")}
    # Requested vs actually-fitted confounders, recorded separately (#246 review, P2).
    _requested_raw = _raw_covariate_confounders(
        s
        for s in spec.adjustment
        if s not in ("G", "A", "W_pre", *(f"{m}_t1" for m in mediators))
    )
    write_run_metadata(
        ctx,
        extra={
            "adjustment": spec.adjustment,
            "effective_confounders": list(confounders),
            "dropped_confounders": [c for c in _requested_raw if c not in confounders],
            "n_obs": built.prepared.n_obs,
            "mediators": list(mediators),
            "n_trials_W": med_data.n_trials_W,
            "mediation": _summary,
            "mediation_sensitivity": sens_summary.to_dict("records"),
            "named_confounder_calibration": (
                calibration_df.to_dict("records") if calibration_df is not None else None
            ),
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Adjusted pipeline (LRP65) — between-child baseline predictors of gain
# ---------------------------------------------------------------------------

# Human-readable labels for the LRP65 predictor keys (for tables / forest plot).
_ADJ_LABELS = {
    "L": "Letter sounds (T1)",
    "lang": "Language composite (T1)",
    "B": "Blending (T1)",
    "age": "Age (T1)",
    "blocks": "Non-verbal MA (T1)",
    "behav": "Behaviour (T1)",
    # Revised-DAG upstream traits, entered as tested covariates (#247).
    "hs": "Hearing status (T1)",
    "hs_missing": "Hearing missing (indicator)",
    "deapp_c": "Speech production (T1)",
    "deapp_c_missing": "Speech missing (indicator)",
    "erbto": "Phonological memory (T1)",
    "erbto_missing": "Phon. memory missing (indicator)",
    "mumedupost16": "SES: mother post-16 educ.",
    "dadedupost16": "SES: father post-16 educ.",
}


def _adj_label(key: str) -> str:
    return _ADJ_LABELS.get(key, key)


def _sample_model(model, sampling, *, label: str = "sub-fit"):
    """Sample a sub-model (bivariate / sensitivity / prior-sweep) with nutpie.

    Mirrors :func:`diagnostics.sample_posterior` but is standalone, so the sub-fit
    traces never overwrite the headline ``ctx.trace`` / ``trace.nc``. A convergence
    check runs on the result and warns loudly if the sub-fit failed the gate, since
    these traces bypass the primary ``diagnostics_summary.json`` gate.

    Returns ``(trace, conv)`` where ``conv`` is the
    :func:`diagnostics.subfit_convergence` verdict dict (``converged``/``max_rhat``/
    ``min_ess``/``min_bfmi``/``n_divergences``). The caller persists the verdict onto
    the sub-fit's published CSV: previously it was computed and discarded, so the
    bivariate / prior-sweep / SES sensitivity tables were reported with no convergence
    flag despite bypassing the primary gate (this review's finding B1).
    """
    import pymc as pm

    with model:
        trace = pm.sample(
            draws=sampling.draws,
            tune=sampling.tune,
            chains=sampling.chains,
            cores=sampling.cores,
            target_accept=sampling.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=sampling.random_seed,
            progressbar=False,
        )
    conv = _diag.subfit_convergence(
        trace, label=label, var_names=[rv.name for rv in model.free_RVs]
    )
    return trace, conv


def _beta_summary(trace, name: str, ci_prob: float) -> dict:
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


def _plot_associations(ctx: StatisticalFitContext, df: pd.DataFrame, hdi: float) -> None:
    y = np.arange(len(df))[::-1]
    plt.figure(figsize=(7.0, 0.6 * len(df) + 1.6))
    plt.errorbar(
        df["adj_mean"], y + 0.12,
        xerr=[df["adj_mean"] - df["adj_lo"], df["adj_hi"] - df["adj_mean"]],
        fmt="o", color=COLOUR_BLUE, capsize=3, label="adjusted (mutual)",
    )
    plt.errorbar(
        df["biv_mean"], y - 0.12,
        xerr=[df["biv_mean"] - df["biv_lo"], df["biv_hi"] - df["biv_mean"]],
        fmt="s", color="#999999", capsize=3, label="bivariate (baseline-only)",
    )
    plt.axvline(0.0, color="grey", ls=":", lw=1)
    plt.yticks(y, df["label"])
    plt.xlabel(
        f"Standardised coefficient (per-SD, logit scale); {int(hdi * 100)}% interval"
    )
    plt.title("LRP65: baseline predictors of word-reading gain (between-child)")
    plt.legend(fontsize=8, loc="best")
    save_styled_figure(
        ctx.output_dir, "predictor_associations", data=df
    )


def _natural_scale_contrasts(
    ctx: StatisticalFitContext, prepared, headline: list, outcome: str, hdi: float
) -> pd.DataFrame:
    """Predicted +1 SD contrast for each predictor on the natural (words) scale.

    For two children with the *same* baseline word reading (held at the sample
    mean) who differ by one standard deviation on a single predictor (others at
    their mean), the model-implied difference in word-reading count at the final
    wave — i.e. the differential gain, in words out of ``N``. Computed per
    posterior draw then summarised, so the interval carries the full uncertainty.
    This turns the per-SD logit coefficients into something a teacher can read.
    """
    from scipy.special import expit

    post = ctx.trace.posterior
    N = prepared.n_trials[outcome]
    mean_pre_logit = float(np.mean(prepared.pre_logit[outcome]))

    def draws(name: str) -> np.ndarray:
        return post[name].stack(sample=("chain", "draw")).values

    # All standardised predictors at their mean (z = 0); baseline at sample mean.
    base_eta = draws("alpha") + draws("gamma_own") * mean_pre_logit
    base_words = N * expit(base_eta)

    lo_q, hi_q = (1 - hdi) / 2, 1 - (1 - hdi) / 2
    rows = []
    for k in headline:
        delta = N * expit(base_eta + draws(f"beta_{k}")) - base_words
        rows.append(
            {
                "predictor": k,
                "label": _adj_label(k),
                "delta_words_median": float(np.median(delta)),
                "delta_words_mean": float(np.mean(delta)),
                "delta_words_lo": float(np.quantile(delta, lo_q)),
                "delta_words_hi": float(np.quantile(delta, hi_q)),
                "delta_words_lo50": float(np.quantile(delta, 0.25)),
                "delta_words_hi50": float(np.quantile(delta, 0.75)),
                "prob_pos": float(np.mean(delta > 0)),
            }
        )
    return pd.DataFrame(rows)


def fit_horseshoe(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Regularized-horseshoe predictor-ranking fit (LRPHS, #116 Phase E).

    An independent Bayesian sensitivity cross-check on the gradient-boosting
    predictor ranking: one horseshoe regression (gain or level, per ``spec.extra``)
    over the full construct predictor set, ranked by posterior
    ``P(|beta| > delta)``. Writes ``predictor_ranking.csv`` alongside the standard
    trace / diagnostics / LOO / PPC artefacts. Not causal — a which-predictors
    -carry-signal read to compare against the GB cluster ranking.
    """
    require_spec(spec, "horseshoe")
    e = spec.extra
    outcome = spec.outcome_symbol or "W"
    gain = bool(e.get("gain", True))
    predictors = list(e["predictors"])
    lang_symbols = tuple(e.get("language_composite_symbols", ["R", "E", "F"]))
    covariates = tuple(e.get("covariates", ()))
    delta = float(e.get("delta", 0.1))
    tau0 = float(e.get("tau0", 0.1))
    slab_scale = float(e.get("slab_scale", 2.0))
    slab_df = float(e.get("slab_df", 4.0))
    post_time = int(e.get("post_time", 4))
    phase_mode = e.get("phase_mode", "span" if gain else "levels")

    # 94% intervals, matching the LRP65 adjusted-model convention.
    ctx = make_context(spec, config, ci_prob=0.89)
    # The horseshoe has a funnel geometry (global-local scales); lift target_accept
    # above the tier default so the sampler takes smaller steps near the neck.

    section_header("Prepare data")
    measure_syms = tuple(
        dict.fromkeys(
            [outcome]
            + [p for p in predictors if p not in ("age", "lang", *covariates)]
            + list(lang_symbols)
        )
    )
    prepared = load_and_prepare(
        phase_mode=phase_mode,
        post_time=post_time,
        outcomes=measure_syms,
        covariates=covariates,
    )
    ctx.prepared = prepared
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_horseshoe_model(
        prepared,
        outcome_symbol=outcome,
        predictors=predictors,
        gain=gain,
        tau0=tau0,
        slab_scale=slab_scale,
        slab_df=slab_df,
        language_composite_symbols=lang_symbols,
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome)

    run_sampling_and_loo(ctx)

    # Coupling term present in the model: gamma_own (gain) or the fixed age slope
    # gamma_A (level) — but the level model suppresses gamma_A when age is itself a
    # horseshoe-ranked predictor (build_horseshoe_model), so only list it then.
    if gain:
        coupling_vars = ["gamma_own"]
    elif "age" not in predictors:
        coupling_vars = ["gamma_A"]
    else:
        coupling_vars = []
    diag_vars = ["alpha", *coupling_vars, "kappa", "hs_tau", "hs_c2", "beta"]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Predictor ranking")
    ranking = _report.horseshoe_ranking(ctx.trace, delta=delta)
    save_table(ctx, "predictor_ranking", ranking)
    print_table(ranked_dataframe_table(ranking.head(10), title="Horseshoe predictor ranking (top 10)"))
    write_prior_pushforward(ctx, horseshoe_pushforward_rows(ctx, predictors, outcome))

    meta_extra = {
        "framing": "gain" if gain else "level",
        "phase_mode": phase_mode,
        "predictors": predictors,
        "covariates": list(covariates),
        "delta": delta,
        "tau0": tau0,
        "slab_scale": slab_scale,
        "slab_df": slab_df,
        "gb_reference": e.get("gb_reference"),
        "ranking_top": ranking.head(3)[["predictor", "p_abs_gt_delta"]].to_dict(
            "records"
        ),
    }
    write_run_metadata(ctx, extra=meta_extra)

    return finalize_report(ctx)


def fit_adjusted(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Between-child adjusted fit (LRP65): independent T1 predictors of gain.

    Headline = the mutually-adjusted between-child regression (one row per child,
    T1 baselines, full-study gain ``W_last | W_T1``). Also fits, per the brief:
    the bivariate (baseline-only-adjusted) association for each predictor; a
    prior-sensitivity sweep over the predictor-slope sigma; and a complete-case
    SES sensitivity fit. Writes ``predictor_associations.csv`` (+ forest plot),
    ``prior_sensitivity.csv`` and ``ses_sensitivity.csv`` alongside the standard
    trace / diagnostics / LOO / PPC artefacts.
    """
    require_spec(spec, "adjusted")
    e = spec.extra
    outcome = spec.outcome_symbol or "W"
    post_time = int(e.get("post_time", 4))
    predictor_symbols = list(e.get("predictor_symbols", ["L", "B"]))
    lang_symbols = tuple(e.get("language_composite_symbols", ["R", "E", "F"]))
    covariates = list(e.get("covariates", ["blocks", "behav"]))
    ses_covs = list(e.get("ses_covariates", ["mumedupost16"]))
    # The slope-prior default is sourced from the factory signature (single source
    # of truth) so this fallback cannot drift from the reconciled scale — prior-
    # critical-review 2026-07-07, recommendation 3; #209 review. The sweep default
    # brackets that scale from the looser side (no factory param mirrors it).
    sigma0 = float(
        e.get(
            "predictor_slope_sigma",
            default_of(_factories.build_adjusted_model, "predictor_slope_sigma"),
        )
    )
    prior_sens = list(e.get("prior_sensitivity_sigmas", [0.5, 0.7]))
    use_age = bool(e.get("use_age_predictor", True))

    # 94% intervals (the brief's convention) rather than the project-wide 95%.
    ctx = make_context(spec, config, ci_prob=0.89)
    hdi = ctx.reporting.ci_prob

    section_header("Prepare data")
    measure_outcomes = tuple(
        dict.fromkeys([outcome, *predictor_symbols, *lang_symbols])
    )
    prepared = load_and_prepare(
        phase_mode="span",
        post_time=post_time,
        outcomes=measure_outcomes,
        covariates=tuple(covariates),
    )
    ctx.prepared = prepared
    # Drop any covariate the loader removed as constant on the fitted rows (e.g. a
    # `_missing` indicator that is all-zero once the complete cases are kept) so the
    # model never requests a coefficient for a term that was never estimated (#247).
    covariates = [c for c in covariates if c in prepared.covariates]
    # Headline predictor key order: skills, language composite, age, tested covariates.
    headline = (
        list(predictor_symbols)
        + ["lang"]
        + (["age"] if use_age else [])
        + covariates
    )
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_adjusted_model(
        prepared,
        outcome_symbol=outcome,
        predictors=headline,
        language_composite_symbols=lang_symbols,
        predictor_slope_sigma=sigma0,
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome)

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    beta_names = [f"beta_{k}" for k in headline]
    _diag.summary_diagnostics(
        ctx, var_names=["alpha", "gamma_own", "kappa", *beta_names]
    )

    run_ppc(ctx)
    _adjusted_diag_vars = ["alpha", "gamma_own", "kappa", *beta_names]
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=_adjusted_diag_vars)

    section_header("Extended diagnostics")
    # Capture the primary gate verdict so the sub-fit tables can label their
    # primary-derived rows (the adjusted/mutual associations and the headline-sigma
    # prior-sweep rows come from ``ctx.trace``, which this gate covers) consistently
    # with the sub-fits' own ``subfit_convergence`` flags (this review's finding B1).
    _primary_gate = _diag.write_diagnostics_summary(ctx, var_names=_adjusted_diag_vars)
    _primary_converged = _report.convergence_gate_clean_passed(_primary_gate)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_adjusted_diag_vars)

    # --- Adjusted vs bivariate associations --------------------------------
    section_header("Predictor associations (adjusted vs bivariate)")
    adjusted = {k: _beta_summary(ctx.trace, f"beta_{k}", hdi) for k in headline}
    bivariate: dict[str, dict] = {}
    biv_converged: dict[str, object] = {}
    for k in headline:
        b = _factories.build_adjusted_model(
            prepared,
            outcome_symbol=outcome,
            predictors=[k],
            language_composite_symbols=lang_symbols,
            predictor_slope_sigma=sigma0,
        )
        t, conv = _sample_model(b.model, ctx.sampling, label=f"{spec.model_id} bivariate {k}")
        bivariate[k] = _beta_summary(t, f"beta_{k}", hdi)
        biv_converged[k] = conv["converged"]

    rows = []
    for k in headline:
        a, bv = adjusted[k], bivariate[k]
        rows.append(
            {
                "predictor": k,
                "label": _adj_label(k),
                "adj_median": a["median"],
                "adj_mean": a["mean"],
                "adj_lo": a["lo"],
                "adj_hi": a["hi"],
                "adj_lo50": a["lo50"],
                "adj_hi50": a["hi50"],
                "adj_prob_pos": a["prob_pos"],
                "biv_median": bv["median"],
                "biv_mean": bv["mean"],
                "biv_lo": bv["lo"],
                "biv_hi": bv["hi"],
                "biv_lo50": bv["lo50"],
                "biv_hi50": bv["hi50"],
                "biv_prob_pos": bv["prob_pos"],
                # Convergence flags: the adjusted column is the primary (gated) fit;
                # the bivariate column is a sub-fit that bypasses the primary gate (B1).
                "adj_converged": _primary_converged,
                "biv_converged": biv_converged[k],
            }
        )
    assoc_df = pd.DataFrame(rows)
    # Missing-data-indicator coefficients are subgroup mean-offsets under the
    # missing-indicator method, not interpretable predictor associations — the same
    # basis on which the prior table now labels them nuisance (the missing-indicator
    # sweep in _prior_table_overrides; #384 review, Frank). Keep them out of the
    # reported associations table + forest so it does not contradict that nuisance
    # label; they remain in the fitted model (as adjusters) and in the full
    # diagnostics summary above.
    _missing_mask = assoc_df["predictor"].astype(str).str.endswith("_missing")
    if _missing_mask.any():
        assoc_df = assoc_df[~_missing_mask].reset_index(drop=True)
    save_table(ctx, "predictor_associations", assoc_df)
    _pf_assoc = assoc_df
    # Estimand-scale prior check on the headline adjusted associations (#381).
    # Driven off the association table just written, not off ``headline``: the
    # missing-data indicators are dropped from that table as nuisance
    # subgroup offsets, and a prior row for a term the report does not show
    # would contradict the nuisance labelling it was dropped for.
    _pf_n = pushforward_n_trials(ctx, outcome)
    _pf_outcome = pushforward_outcome_label(ctx, outcome)
    write_prior_pushforward(
        ctx,
        marginal_pushforward_rows(
            ctx,
            [
                (
                    f"beta_{r.predictor}",
                    f"the adjusted association of +1 SD {r.label} with {_pf_outcome}",
                )
                for r in _pf_assoc.itertuples()
            ],
            n_trials=_pf_n,
            convention="forward",
        ),
    )
    print_table(
        ranked_dataframe_table(
            assoc_df,
            title=(
                f"Predictor associations (per-SD, logit; {int(hdi * 100)}% interval)"
            ),
            columns=[
                "label", "adj_mean", "adj_lo", "adj_hi", "adj_prob_pos",
                "biv_mean", "biv_lo", "biv_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )
    _plot_associations(ctx, assoc_df, hdi)

    # --- Prior sensitivity (does the clear-zero conclusion move?) ----------
    section_header("Prior sensitivity")
    ps_rows = []
    for sig in [sigma0, *prior_sens]:
        if sig == sigma0:
            tr = ctx.trace
            sig_converged = _primary_converged  # headline sigma is the gated primary
        else:
            b = _factories.build_adjusted_model(
                prepared,
                outcome_symbol=outcome,
                predictors=headline,
                language_composite_symbols=lang_symbols,
                predictor_slope_sigma=sig,
            )
            tr, conv = _sample_model(
                b.model, ctx.sampling, label=f"{spec.model_id} prior-sweep sigma={sig}"
            )
            sig_converged = conv["converged"]
        for k in headline:
            ps_rows.append(
                {
                    "sigma": sig,
                    "predictor": k,
                    **_beta_summary(tr, f"beta_{k}", hdi),
                    "converged": sig_converged,
                }
            )
    ps_df = pd.DataFrame(ps_rows)
    save_table(ctx, "prior_sensitivity", ps_df)

    # --- SES complete-case sensitivity -------------------------------------
    section_header("SES sensitivity (complete cases)")
    ses_df = None
    ses_n = None
    ses_error = None
    try:
        prepared_ses = load_and_prepare(
            phase_mode="span",
            post_time=post_time,
            outcomes=measure_outcomes,
            covariates=tuple(covariates + ses_covs),
        )
        # Re-filter against the SES-complete subset: a `_missing` indicator can go
        # constant on this smaller subset even if it survived the headline fit, and the
        # loader then drops it — so rebuild the predictor list here too, or
        # ``build_adjusted_model`` would KeyError on the dropped term (#287 review). The
        # non-covariate predictors (skills / lang / age) are always kept.
        ses_headline = [
            k for k in headline if k not in covariates or k in prepared_ses.covariates
        ]
        ses_covs_fit = [c for c in ses_covs if c in prepared_ses.covariates]
        ses_predictors = ses_headline + ses_covs_fit
        b = _factories.build_adjusted_model(
            prepared_ses,
            outcome_symbol=outcome,
            predictors=ses_predictors,
            language_composite_symbols=lang_symbols,
            predictor_slope_sigma=sigma0,
        )
        t, conv = _sample_model(b.model, ctx.sampling, label=f"{spec.model_id} SES complete-case")
        ses_n = int(b.prepared.n_children)
        ses_rows = [
            {
                "predictor": k,
                "label": _adj_label(k),
                "n_children": ses_n,
                **_beta_summary(t, f"beta_{k}", hdi),
                "converged": conv["converged"],
            }
            for k in ses_predictors
        ]
        ses_df = pd.DataFrame(ses_rows)
        save_table(ctx, "ses_sensitivity", ses_df)
        rprint(f"  SES sensitivity fit on {ses_n} complete-case children")
    except Exception as exc:  # pragma: no cover
        # Record the failure (type + message + traceback) rather than swallowing
        # it to a one-line warning: a genuine bug (missing column, factory error)
        # should not silently produce a "successful" reporting run with no
        # ses_sensitivity.csv. The error is surfaced in the run metadata.
        import traceback

        ses_error = f"{type(exc).__name__}: {exc}"
        rprint(f"[red]SES sensitivity fit failed: {ses_error}[/red]")
        rprint(f"[yellow]{traceback.format_exc()}[/yellow]")

    # --- Natural-scale interpretation (predicted gain, in words) -----------
    section_header("Predicted gain on the natural (words) scale")
    words_df = _natural_scale_contrasts(ctx, ctx.prepared, headline, outcome, hdi)
    save_table(ctx, "predicted_gain_words", words_df)
    print_table(
        ranked_dataframe_table(
            words_df,
            title=(
                f"Predicted differential gain per +1 SD (words out of "
                f"{ctx.prepared.n_trials[outcome]}; {int(hdi * 100)}% interval)"
            ),
            columns=[
                "label", "delta_words_mean", "delta_words_lo",
                "delta_words_hi", "prob_pos",
            ],
            rank_column=False,
            precision=2,
        )
    )

    # --- Influence (does the fit rest on a few children?) ------------------
    section_header("Influence (PSIS-LOO Pareto-k)")
    infl_df, k_thr, n_flagged = _diag.influence_diagnostics(ctx)
    if infl_df is not None:
        save_table(ctx, "influence", infl_df)
        rprint(
            f"  max Pareto-k = {infl_df['pareto_k'].max():.2f}; "
            f"{n_flagged} of {len(infl_df)} children exceed k = {k_thr:.2f}"
        )
    else:
        rprint("[yellow]Pareto-k unavailable from LOO; influence check skipped[/yellow]")

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "design": "between_child",
            "post_time": post_time,
            "predictors": headline,
            "predictor_slope_sigma": sigma0,
            "prior_sensitivity_sigmas": prior_sens,
            "language_composite_symbols": list(lang_symbols),
            "n_children": int(ctx.prepared.n_children),
            "ses_n_children": ses_n,
            "ses_error": ses_error,
            "associations": rows,
            "predicted_gain_words": words_df.to_dict("records"),
            "max_pareto_k": (
                float(infl_df["pareto_k"].max()) if infl_df is not None else None
            ),
            "n_pareto_k_flagged": n_flagged,
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Longitudinal dynamic pipeline (LRP67 LCSM)
# ---------------------------------------------------------------------------


def _coef_row(label: str, draws, hdi_prob: float) -> dict:
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


# ---------------------------------------------------------------------------
# Concurrent conditional-associations family (LRP-CA, #312, workstream #314)
# ---------------------------------------------------------------------------

_CA_LABELS = {
    "W": "Word reading",
    "L": "Letter sounds",
    "B": "Blending",
    "TR": "Taught receptive vocab",
    "TE": "Taught expressive vocab",
    "R": "Receptive vocab",
    "E": "Expressive vocab",
    "age": "Age",
}


def _ca_label(sym: str) -> str:
    return _CA_LABELS.get(sym, sym)


def _ca_wave_predictors(
    wave_prepared, predictor_symbols: list[str]
) -> tuple[list[str], list[str]]:
    """Split ``predictor_symbols`` into those usable at this wave and those dropped.

    A predictor is usable only if its same-wave logit has positive, finite variance on
    the wave's rows — otherwise the factory's ``standardise`` would raise (an all-missing
    or constant predictor at a wave carries no association and cannot be standardised).
    Returns ``(available, dropped)`` preserving input order.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    available, dropped = [], []
    for sym in predictor_symbols:
        vals = np.asarray(wave_prepared.post_counts.get(sym), dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size < 2:
            dropped.append(sym)
            continue
        sd = float(np.nanstd(logit_safe(vals, MEASURES[sym].n_trials), ddof=1))
        (available if np.isfinite(sd) and sd > 0 else dropped).append(sym)
    return available, dropped


def _ca_concurrent_terms(wave_prepared, predictor_symbols: list[str]) -> list:
    """``ConcurrentTerm`` list for a wave's items-scale marginals (#312).

    Recomputes, per predictor, the same-wave logit SD (matching the factory's
    ``standardise``), the mean item count (the ``+k items`` operating point) and a
    per-measure items increment ``k = max(1, round(N / 10))`` — so a fixed ``+5`` does
    not span 3 %-50 % of predictor scales that differ tenfold (the #310/#325 caveat,
    applied here from the outset).
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    terms = []
    for sym in predictor_symbols:
        m = MEASURES[sym]
        vals = np.asarray(wave_prepared.post_counts[sym], dtype=float)
        _z, scaler = standardise(logit_safe(vals, m.n_trials))
        mean_items = float(np.nanmean(vals))
        k = max(1, round(m.n_trials / 10))
        terms.append(
            _report.ConcurrentTerm(
                label=sym,
                coef=f"beta_{sym}",
                sd_logit=float(scaler.sd),
                n_items=m.n_trials,
                mean_items=mean_items,
                k_items=k,
            )
        )
    return terms


def _ca_margin_fields(prefix: str, row: pd.Series) -> dict[str, float]:
    """Wide probability/items fields for one ``+1 SD`` concurrent marginal row."""
    return {
        f"{prefix}_ame_{scale}_{stat}": float(row[f"{scale}_{stat}"])
        for scale in ("prob", "items")
        for stat in ("median", "lo", "hi", "lo50", "hi50")
    }


def _ca_sd_margin(df: pd.DataFrame, predictor: str) -> pd.Series:
    """Return the unique ``+1 SD`` marginal row for ``predictor``."""
    rows = df[(df["term"] == predictor) & (df["scale"] == "+1 SD")]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one +1 SD marginal for {predictor!r}; found {len(rows)}"
        )
    return rows.iloc[0]


_CA_MARGIN_STATS = ("median", "lo", "hi", "lo50", "hi50")
_CA_ASSOCIATION_REQUIRED = {
    "timepoint",
    "predictor",
    "label",
    "n",
    "predictor_n",
    "predictor_imputed_n",
    "ame_contrast",
    "adj_median",
    "adj_mean",
    "adj_lo",
    "adj_hi",
    "adj_lo50",
    "adj_hi50",
    "adj_prob_pos",
    "biv_median",
    "biv_mean",
    "biv_lo",
    "biv_hi",
    "biv_lo50",
    "biv_hi50",
    "biv_prob_pos",
    "adj_converged",
    "biv_converged",
} | {
    f"{prefix}_ame_{scale}_{stat}"
    for prefix in ("adj", "biv")
    for scale in ("prob", "items")
    for stat in _CA_MARGIN_STATS
}
_CA_MARGINAL_REQUIRED = {
    "timepoint",
    "adjustment",
    "term",
    "role",
    "scale",
    "prob_median",
    "prob_lo",
    "prob_hi",
    "prob_lo50",
    "prob_hi50",
    "items_median",
    "items_lo",
    "items_hi",
    "items_lo50",
    "items_hi50",
    "prob_pos",
    "label",
    "converged",
}
_CA_DIAGNOSTIC_REQUIRED = {
    "timepoint",
    "fit_kind",
    "predictor",
    "n",
    "n_predictors",
    "converged",
    "max_rhat",
    "min_ess",
    "min_bfmi",
    "n_divergences",
}


def _write_concurrent_outputs(
    ctx: StatisticalFitContext,
    *,
    association_rows: list[dict],
    marginal_frames: list[pd.DataFrame],
    diagnostic_rows: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and write the three concurrent-family output tables.

    The explicit cross-table checks make the issue #312 contract executable: every
    wave-predictor association must have adjusted and bivariate ``+1 SD`` natural-scale
    rows and a matching fit-diagnostics row, while every wave has one adjusted-fit
    diagnostics row. This prevents a future refactor from silently publishing only one
    side of the requested adjusted/bivariate comparison.
    """
    association_df = pd.DataFrame(association_rows)
    marginal_df = pd.concat(marginal_frames, ignore_index=True)
    diagnostic_df = pd.DataFrame(diagnostic_rows)

    for name, frame, required in (
        ("concurrent_associations", association_df, _CA_ASSOCIATION_REQUIRED),
        ("concurrent_marginals", marginal_df, _CA_MARGINAL_REQUIRED),
        ("concurrent_fit_diagnostics", diagnostic_df, _CA_DIAGNOSTIC_REQUIRED),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")

    association_pairs = {
        (int(row.timepoint), str(row.predictor))
        for row in association_df[["timepoint", "predictor"]].itertuples(index=False)
    }
    expected_marginals = {
        (timepoint, predictor, adjustment)
        for timepoint, predictor in association_pairs
        for adjustment in ("adjusted", "bivariate")
    }
    sd_marginals = marginal_df[marginal_df["scale"] == "+1 SD"]
    actual_marginals = {
        (int(row.timepoint), str(row.term), str(row.adjustment))
        for row in sd_marginals[
            ["timepoint", "term", "adjustment"]
        ].itertuples(index=False)
    }
    if actual_marginals != expected_marginals:
        missing = sorted(expected_marginals - actual_marginals)
        extra = sorted(actual_marginals - expected_marginals)
        raise ValueError(
            "concurrent_marginals +1 SD cross-product mismatch: "
            f"missing={missing}, extra={extra}"
        )

    expected_adjusted = {timepoint for timepoint, _ in association_pairs}
    adjusted_diagnostics = diagnostic_df[
        diagnostic_df["fit_kind"] == "adjusted"
    ]
    actual_adjusted = {
        int(row.timepoint)
        for row in adjusted_diagnostics[["timepoint"]].itertuples(index=False)
    }
    bivariate_diagnostics = diagnostic_df[
        diagnostic_df["fit_kind"] == "bivariate"
    ]
    actual_bivariate = {
        (int(row.timepoint), str(row.predictor))
        for row in bivariate_diagnostics[
            ["timepoint", "predictor"]
        ].itertuples(index=False)
    }
    if actual_adjusted != expected_adjusted or actual_bivariate != association_pairs:
        raise ValueError(
            "concurrent_fit_diagnostics does not cover every published fit: "
            f"adjusted={sorted(actual_adjusted)}, "
            f"bivariate={sorted(actual_bivariate)}"
        )

    for name, frame in (
        ("concurrent_associations", association_df),
        ("concurrent_marginals", marginal_df),
        ("concurrent_fit_diagnostics", diagnostic_df),
    ):
        save_table(ctx, name, frame)

    return association_df, marginal_df, diagnostic_df


def fit_concurrent(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Per-wave concurrent conditional associations (LRP-CA, #312).

    Fits, at each timepoint, a between-child Beta-Binomial regression of the focal
    outcome's *level* on the standardised same-wave logits of a predictor skill set
    (plus age and a group nuisance term) — "at wave t, among children alike on age and
    the other skills, +n of predictor X is associated with +m of the outcome". Every
    coefficient is an **adjusted association**; the family makes no causal claim, so
    conditioning on contemporaneous (post-treatment) skill levels is intentional.

    Design (issue #312): four separate cross-sectional fits, reported side by side. The
    diagnostic-anchor wave (most rows; ties → latest) is the fit that carries the
    standard trace / convergence-gate / PPC artefacts; the other waves and every
    bivariate (single-predictor, unadjusted) fit are sub-fits. Every published fit has
    R-hat, ESS, BFMI and divergence diagnostics recorded in
    ``concurrent_fit_diagnostics.csv``. ``concurrent_associations.csv`` carries the
    adjusted and bivariate logit coefficients plus matched +1-SD probability/items
    marginals (wave × predictor); ``concurrent_marginals.csv`` carries both fit kinds'
    detailed probability/items marginals (wave × predictor × {+1 SD, +k items}).
    """
    require_spec(spec, "concurrent", outcome=True)
    # Resolve and validate the family contract before the context resets an output
    # directory or the loader reads any data (#394 pillar 4). One plan drives
    # preparation, the teaching recipe and config.json.
    plan = resolve_concurrent_run_plan(spec)
    outcome = plan.outcome_symbol
    predictor_symbols = list(plan.predictor_symbols)
    # Trait covariates (non-verbal ability, hearing, speech, phonological memory),
    # aligned with the gains panel. They are t1-measured, so they enter as
    # baseline covariates broadcast across the waves (there is no per-wave value).
    covariates = list(plan.covariates)
    include_age = plan.include_age
    include_group = plan.include_group
    # ``predictor_slope_sigma`` is None on the plan when a spec does not set it, so the
    # build_concurrent_model default is filled via default_of here — the anti-drift
    # single source #394 retains until typed family defaults replace it.
    sigma0 = (
        float(plan.predictor_slope_sigma)
        if plan.predictor_slope_sigma is not None
        else float(
            default_of(_factories.build_concurrent_model, "predictor_slope_sigma")
        )
    )

    from language_reading_predictors.statistical_models.measures import MEASURES

    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    hdi = ctx.reporting.ci_prob
    N_focal = MEASURES[outcome].n_trials

    section_header("Prepare data")
    prepared_all = load_and_prepare(**plan.prepare_kwargs())

    # Timepoints present; each wave's row count and its usable predictor set (a
    # predictor whose same-wave logit has positive variance on the wave's rows —
    # anything constant/all-missing at that wave is dropped, and a wave with no usable
    # predictor is skipped below).
    wave_indices = sorted({int(p) for p in np.unique(prepared_all.phase)})
    wave_subsets: dict[int, object] = {}
    wave_n: dict[int, int] = {}
    wave_preds: dict[int, list[str]] = {}
    dropped_by_wave: dict[int, list[str]] = {}
    for w in wave_indices:
        sub = _subset_prepared(prepared_all, prepared_all.phase == w)
        keep = ~np.isnan(sub.post_counts[outcome])
        sub = _subset_prepared(sub, keep)
        wave_subsets[w] = sub
        wave_n[w] = sub.n_obs
        wave_preds[w], dropped_by_wave[w] = _ca_wave_predictors(sub, predictor_symbols)
    # Diagnostic anchor = most complete-outcome rows; tie → latest timepoint. This is
    # an operational artefact-selection rule, not a claim that the wave is best-powered
    # or substantively primary. Choose it ONLY among waves that actually have a usable
    # predictor: a wave whose predictors are all constant/all-missing is skipped in the
    # fit loop, so making it the anchor would leave ``wave_fits[primary_wave]`` unset
    # and crash the fit.
    fittable_waves = [w for w in wave_indices if wave_preds[w]]
    if not fittable_waves:
        raise ValueError(
            f"{spec.model_id}: no wave has a usable predictor (all "
            f"{predictor_symbols} are constant/all-missing at every timepoint); "
            "cannot fit the concurrent model."
        )
    primary_wave = max(fittable_waves, key=lambda w: (wave_n[w], w))

    # Provisional; replaced with the primary-wave subset once known so the report's
    # header / n_obs describe the gated fit.
    ctx.prepared = wave_subsets[primary_wave]
    print_header(ctx)

    def _build(sub, preds, *, age, group):
        return _factories.build_concurrent_model(
            sub,
            outcome_symbol=outcome,
            predictor_symbols=preds,
            covariates=covariates,
            include_age=age,
            include_group=group,
            predictor_slope_sigma=sigma0,
        )

    # ---- Fit each wave's mutually-adjusted model --------------------------------
    wave_fits: dict[int, dict] = {}
    for w in wave_indices:
        sub = wave_subsets[w]
        preds = wave_preds[w]
        tp = w + 1  # 1-based timepoint for reports
        if not preds:
            rprint(f"[yellow]Concurrent: wave t{tp} has no usable predictors; skipped.[/yellow]")
            continue
        if w == primary_wave:
            section_header(f"Build model (primary wave t{tp})")
            built = _build(sub, preds, age=include_age, group=include_group)
            attach_built(ctx, built)
            render_model_graph(ctx)
            section_header("Prior predictive")
            _diag.run_prior_predictive(ctx, draws=1000)
            _diag.save_prior_predictive_plot(ctx, outcome)
            run_sampling_and_loo(ctx)
            trace = ctx.trace
            convergence = None  # populated below after the full primary gate
        else:
            built = _build(sub, preds, age=include_age, group=include_group)
            trace, conv = _sample_model(
                built.model, ctx.sampling, label=f"{spec.model_id} wave t{tp}"
            )
            convergence = conv
        wave_fits[w] = {
            "trace": trace,
            "prepared": built.prepared,
            "preds": preds,
            "convergence": convergence,
        }

    # ---- Primary-wave diagnostics + standard artefacts --------------------------
    section_header("Summary diagnostics (primary wave)")
    prim = wave_fits[primary_wave]
    beta_names = [f"beta_{s}" for s in prim["preds"]]
    diag_vars = ["alpha", "kappa", *beta_names]
    if include_age:
        diag_vars.append("beta_age")
    if include_group:
        diag_vars.append("beta_group_nuisance")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)
    run_ppc(ctx)
    section_header("Extended diagnostics (primary wave)")
    _primary_gate = _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _primary_conv = _diag.subfit_convergence(
        ctx.trace,
        label=f"{spec.model_id} primary wave t{primary_wave + 1}",
        var_names=[rv.name for rv in ctx.model.free_RVs],
    )
    _primary_conv["converged"] = bool(
        _report.convergence_gate_clean_passed(_primary_gate)
        and _primary_conv.get("converged")
    )
    wave_fits[primary_wave]["convergence"] = _primary_conv
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)
    # Estimand-scale prior check on the primary wave's adjusted associations
    # (#381) — one row per predictor, on the same ``+1 SD`` forward-shift scale
    # ``concurrent_marginals`` reports below. Only the primary wave persists a
    # prior group; the other waves are refits sampled without one.
    write_prior_pushforward(
        ctx,
        marginal_pushforward_rows(
            ctx,
            [
                (
                    f"beta_{s}",
                    f"the adjusted association of +1 SD {_ca_label(s)} with "
                    f"{MEASURES[outcome].label} at t{primary_wave + 1}",
                )
                for s in prim["preds"]
            ],
            n_trials=N_focal,
            convention="forward",
        ),
    )

    # ---- Adjusted vs bivariate coefficients + natural-scale marginals -----------
    section_header("Concurrent associations (adjusted vs bivariate)")
    assoc_rows: list[dict] = []
    marg_frames: list[pd.DataFrame] = []
    fit_diagnostic_rows: list[dict] = []
    for w in wave_indices:
        if w not in wave_fits:
            continue
        tp = w + 1
        fit = wave_fits[w]
        sub, preds, trace = fit["prepared"], fit["preds"], fit["trace"]
        adj_conv = fit["convergence"]
        fit_diagnostic_rows.append(
            {
                "timepoint": tp,
                "fit_kind": "adjusted",
                "predictor": "all",
                "n": sub.n_obs,
                "n_predictors": len(preds),
                **adj_conv,
            }
        )

        # Natural-scale marginals for the mutually-adjusted associations at this wave.
        terms = _ca_concurrent_terms(sub, preds)
        terms_by_symbol = {term.label: term for term in terms}
        adj_mdf = _report.concurrent_marginals(
            trace, terms=terms, n_trials=N_focal, ci_prob=hdi
        )
        adj_mdf.insert(0, "timepoint", tp)
        adj_mdf.insert(1, "adjustment", "adjusted")
        adj_mdf["label"] = adj_mdf["term"].map(_ca_label)
        adj_mdf["converged"] = adj_conv["converged"]
        marg_frames.append(adj_mdf)

        # Per-predictor: adjusted beta (this wave's full fit) + bivariate beta (refit).
        for sym in preds:
            adj = _beta_summary(trace, f"beta_{sym}", hdi)
            b = _build(sub, [sym], age=False, group=False)
            bt, bconv = _sample_model(
                b.model, ctx.sampling, label=f"{spec.model_id} t{tp} bivariate {sym}"
            )
            biv = _beta_summary(bt, f"beta_{sym}", hdi)
            biv_mdf = _report.concurrent_marginals(
                bt,
                terms=[terms_by_symbol[sym]],
                n_trials=N_focal,
                ci_prob=hdi,
            )
            biv_mdf.insert(0, "timepoint", tp)
            biv_mdf.insert(1, "adjustment", "bivariate")
            biv_mdf["label"] = biv_mdf["term"].map(_ca_label)
            biv_mdf["converged"] = bconv["converged"]
            marg_frames.append(biv_mdf)

            adj_sd = _ca_sd_margin(adj_mdf, sym)
            biv_sd = _ca_sd_margin(biv_mdf, sym)
            predictor_n = int(np.isfinite(sub.post_counts[sym]).sum())
            assoc_rows.append(
                {
                    "timepoint": tp,
                    "predictor": sym,
                    "label": _ca_label(sym),
                    "n": sub.n_obs,
                    "predictor_n": predictor_n,
                    "predictor_imputed_n": sub.n_obs - predictor_n,
                    "ame_contrast": "+1 SD",
                    "adj_median": adj["median"],
                    "adj_mean": adj["mean"],
                    "adj_lo": adj["lo"],
                    "adj_hi": adj["hi"],
                    "adj_lo50": adj["lo50"],
                    "adj_hi50": adj["hi50"],
                    "adj_prob_pos": adj["prob_pos"],
                    **_ca_margin_fields("adj", adj_sd),
                    "biv_median": biv["median"],
                    "biv_mean": biv["mean"],
                    "biv_lo": biv["lo"],
                    "biv_hi": biv["hi"],
                    "biv_lo50": biv["lo50"],
                    "biv_hi50": biv["hi50"],
                    "biv_prob_pos": biv["prob_pos"],
                    **_ca_margin_fields("biv", biv_sd),
                    "adj_converged": adj_conv["converged"],
                    "biv_converged": bconv["converged"],
                }
            )
            fit_diagnostic_rows.append(
                {
                    "timepoint": tp,
                    "fit_kind": "bivariate",
                    "predictor": sym,
                    "n": sub.n_obs,
                    "n_predictors": 1,
                    **bconv,
                }
            )

    assoc_df, marg_df, fit_diagnostics_df = _write_concurrent_outputs(
        ctx,
        association_rows=assoc_rows,
        marginal_frames=marg_frames,
        diagnostic_rows=fit_diagnostic_rows,
    )
    print_table(
        ranked_dataframe_table(
            assoc_df,
            title=f"Concurrent associations (per-SD, logit; {int(hdi * 100)}% interval)",
            columns=[
                "timepoint", "label", "adj_mean", "adj_lo", "adj_hi", "adj_prob_pos",
                "biv_mean", "biv_lo", "biv_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )
    _plot_concurrent(ctx, assoc_df, hdi, primary_tp=primary_wave + 1)

    all_fits_converged = bool(
        not fit_diagnostics_df.empty
        and fit_diagnostics_df["converged"].eq(True).all()
    )
    meta_extra = {
        "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
        "estimand": "concurrent conditional associations (per wave)",
        "predictors": prim["preds"],
        "predictors_requested": predictor_symbols,
        "dropped_by_wave": {f"t{w + 1}": dropped_by_wave[w] for w in wave_indices},
        "primary_timepoint": primary_wave + 1,
        "diagnostic_anchor_timepoint": primary_wave + 1,
        "timepoints": [w + 1 for w in wave_indices],
        "wave_n": {f"t{w + 1}": wave_n[w] for w in wave_indices},
        "include_age": include_age,
        "include_group_nuisance": include_group,
        "bivariate_adjustment": "predictor only; age, group and other skills omitted",
        "averaging_population": "all fitted rows at the wave (descriptive)",
        "predictor_slope_sigma": sigma0,
        "standardisation": (
            "same-wave Haldane-corrected logit, standardised within each wave"
        ),
        "n_published_fits": int(len(fit_diagnostics_df)),
        "all_published_fits_converged": all_fits_converged,
        "n_failed_or_unchecked_fits": int(
            (~fit_diagnostics_df["converged"].eq(True)).sum()
        ),
        "output_contract": (
            "concurrent_associations.csv contains adjusted and bivariate logit, "
            "probability and items summaries for +1 SD; concurrent_marginals.csv "
            "contains both fit kinds for +1 SD and +k items"
        ),
    }
    write_run_metadata(ctx, extra=meta_extra)

    return finalize_report(ctx)


def _plot_concurrent(
    ctx: StatisticalFitContext, df: pd.DataFrame, hdi: float, *, primary_tp: int
) -> None:
    """Forest of adjusted vs bivariate coefficients for the primary wave (#312)."""
    if df.empty:
        return
    d = df[df["timepoint"] == primary_tp].reset_index(drop=True)
    if d.empty:
        return
    y = np.arange(len(d))[::-1]
    plt.figure(figsize=(7.0, 0.6 * len(d) + 1.6))
    plt.errorbar(
        d["adj_mean"], y + 0.12,
        xerr=[d["adj_mean"] - d["adj_lo"], d["adj_hi"] - d["adj_mean"]],
        fmt="o", color=COLOUR_BLUE, capsize=3, label="adjusted (mutual)",
    )
    plt.errorbar(
        d["biv_mean"], y - 0.12,
        xerr=[d["biv_mean"] - d["biv_lo"], d["biv_hi"] - d["biv_mean"]],
        fmt="s", color="#999999", capsize=3, label="bivariate (unadjusted)",
    )
    plt.axvline(0.0, color="grey", ls=":", lw=1)
    plt.yticks(y, d["label"])
    plt.xlabel(
        f"Standardised coefficient (per-SD, logit scale); {int(hdi * 100)}% interval"
    )
    plt.title(f"Concurrent associations at t{primary_tp} (between-child)")
    plt.legend(fontsize=8, loc="best")
    # NB: distinct stem from ``concurrent_associations.csv`` (the full wave×predictor
    # table) — save_styled_figure(data=...) writes a sidecar ``{stem}.csv`` of just the
    # plotted (primary-wave) rows, which would otherwise clobber the full table.
    save_styled_figure(ctx.output_dir, "concurrent_associations_forest", data=d)


def fit_lcsm(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Latent change-score model (LRP67 + the lagged coupling suite, #250).

    Fits the coupled McArdle latent change-score model with process noise and
    reports the per-target coupling tables. ``spec.extra`` selects the shape:
    the LRP67 default couples every other measure into the reading change; the
    lagged reverse-coupling models (LCSM-081/181/082) pass an explicit
    ``couplings`` map plus ``arm_window_intercepts`` (the crossover-aware
    arm x window change intercepts, with the window-1 randomised contrast
    written to ``itt_window1_contrast.csv``) and a shared adjuster
    ``covariate_block``. ``dominance_pair`` adds the SD-standardised
    reciprocal-dominance contrast (``dominance_summary.csv``).
    ``lagged_change_couplings`` (LCSM-091, #229 spec 2) adds prior-transition
    latent-change terms (``h_{src}``) to the named targets' change equations.
    """
    require_spec(spec, "lcsm")

    ctx = make_context(spec, config)

    section_header("Prepare data")
    outcomes = tuple(spec.extra.get("outcomes", ("W", "L", "E")))
    reading_symbol = spec.outcome_symbol or "W"
    couplings_in = spec.extra.get("couplings")
    couplings: dict[str, tuple[str, ...]] = (
        {tgt: tuple(srcs) for tgt, srcs in couplings_in.items()}
        if couplings_in
        else {reading_symbol: tuple(s for s in outcomes if s != reading_symbol)}
    )
    lagged_in = spec.extra.get("lagged_change_couplings")
    lagged_change_couplings: dict[str, tuple[str, ...]] = (
        {tgt: tuple(srcs) for tgt, srcs in lagged_in.items()} if lagged_in else {}
    )
    arm_window = bool(spec.extra.get("arm_window_intercepts", False))
    covariate_block = tuple(spec.extra.get("covariate_block", ()))
    covariate_targets = tuple(spec.extra.get("covariate_targets", ()))
    # Loader needs from the covariate block: the hearing dummies come via
    # include_hearing; everything else names a per-wave source column (its
    # ``_missing`` companion is derived, not loaded).
    include_hearing = any(n in ("hs", "hs_missing") for n in covariate_block)
    wave_cov_cols = tuple(
        dict.fromkeys(
            n
            for n in covariate_block
            if n not in ("hs", "hs_missing") and not n.endswith("_missing")
        )
    )
    panel = load_wave_panel(
        outcomes=outcomes,
        wave_covariates=wave_cov_cols,
        include_hearing=include_hearing,
    )
    ctx.prepared = panel

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_lcsm_model(
        panel,
        reading_symbol=reading_symbol,
        couplings=couplings,
        lagged_change_couplings=lagged_change_couplings or None,
        arm_window_intercepts=arm_window,
        covariate_block=covariate_block,
        covariate_targets=covariate_targets,
        coupling_prior_sigma=spec.extra.get(
            "coupling_prior_sigma",
            default_of(_factories.build_lcsm_model, "coupling_prior_sigma"),
        ),
        use_process_noise=spec.extra.get("use_process_noise", True),
        shared_process_noise=spec.extra.get("shared_process_noise", False),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    # Coupling parameter names mirror the factory's rule: single target keeps
    # LRP67's ``g_{src}``; multiple targets carry the target (``g_{src}_{tgt}``).
    single_target = len(couplings) == 1
    coupling_names = {
        (src, tgt): (f"g_{src}" if single_target else f"g_{src}_{tgt}")
        for tgt, srcs in couplings.items()
        for src in srcs
    }
    # Lagged change-on-change names mirror the factory's rule on the lag map.
    single_lag_target = len(lagged_change_couplings) == 1
    lagged_names = {
        (src, tgt): (f"h_{src}" if single_lag_target else f"h_{src}_{tgt}")
        for tgt, srcs in lagged_change_couplings.items()
        for src in srcs
    }
    diag_vars = list(coupling_names.values())
    diag_vars += list(lagged_names.values())
    diag_vars += [f"b_{name}" for name in covariate_block]
    diag_vars += ["a_change", "b_self", "d_age", "sigma1", "kappa"]
    if spec.extra.get("use_process_noise", True):
        diag_vars.append("sigma_proc")

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # One check per measure: ``y_obs`` flattens every measure into a single vector, so
    # a lone overlay would pool scales with different maxima. The headline reading
    # symbol keeps the unsuffixed filename the report partial expects.
    for _sym in outcomes:
        _diag.save_prior_predictive_plot(
            ctx,
            _sym,
            node="y_obs",
            filename_stem=(
                "prior_predictive_check"
                if _sym == reading_symbol
                else f"prior_predictive_check_{_sym.lower()}"
            ),
        )

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=["y_obs"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(
        ctx, causal_term=diag_vars[0] if diag_vars else None
    )
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Per-target coupling table — the headline "what predicts whose change"
    # output. For LRP67 (single reading target) this reproduces the historical
    # reading-change table, labels included.
    section_header("Change-coupling summary")
    post = ctx.trace.posterior
    rows = [
        _coef_row(
            f"{pname} (prior {src} -> {tgt} change)",
            post[pname].values,
            ctx.reporting.ci_prob,
        )
        for (src, tgt), pname in coupling_names.items()
    ]
    rows += [
        _coef_row(
            f"{pname} (prior {src} change -> {tgt} change)",
            post[pname].values,
            ctx.reporting.ci_prob,
        )
        for (src, tgt), pname in lagged_names.items()
    ]
    for name in covariate_block:
        rows.append(
            _coef_row(
                f"b_{name} ({name} -> {'/'.join(covariate_targets)} change)",
                post[f"b_{name}"].values,
                ctx.reporting.ci_prob,
            )
        )
    for tgt in couplings:
        # LRP67's historical row labels are kept verbatim for the single
        # reading-target shape.
        legacy = single_target and tgt == reading_symbol
        rows.append(
            _coef_row(
                f"b_self[{tgt}] (reading self-feedback)"
                if legacy
                else f"b_self[{tgt}] ({tgt} self-feedback)",
                post["b_self"].sel(outcome=tgt).values,
                ctx.reporting.ci_prob,
            )
        )
        if not arm_window:
            rows.append(
                _coef_row(
                    f"a_change[{tgt}] (reading baseline change)"
                    if legacy
                    else f"a_change[{tgt}] ({tgt} baseline change)",
                    post["a_change"].sel(outcome=tgt).values,
                    ctx.reporting.ci_prob,
                )
            )
        rows.append(
            _coef_row(
                f"d_age[{tgt}] (age -> reading change)"
                if legacy
                else f"d_age[{tgt}] (age -> {tgt} change)",
                post["d_age"].sel(outcome=tgt).values,
                ctx.reporting.ci_prob,
            )
        )
    coupling_df = pd.DataFrame(rows)
    save_table(ctx, "coupling_summary", coupling_df)
    print_table(
        ranked_dataframe_table(
            coupling_df,
            title=(
                f"Change couplings - {int(ctx.reporting.ci_prob * 100)}% CI "
                "(equal-tailed)"
            ),
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # Window-1 randomised contrast on the latent change scale (immediate -
    # waitlist), the built-in consistency check against the ITT suite. Only the
    # arm x window shape carries it.
    itt_rows: list[dict] = []
    if arm_window:
        section_header("Window-1 randomised contrast (ITT consistency check)")
        for s in outcomes:
            itt_rows.append(
                _coef_row(
                    f"itt_w1[{s}] (immediate - waitlist, window-1 latent change)",
                    post["itt_w1_contrast"].sel(outcome=s).values,
                    ctx.reporting.ci_prob,
                )
            )
        itt_df = pd.DataFrame(itt_rows)
        save_table(ctx, "itt_window1_contrast", itt_df)
        print_table(
            ranked_dataframe_table(
                itt_df,
                title="Window-1 arm contrast (latent logit change)",
                columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    # Reciprocal-dominance contrast (LCSM-082): per draw, standardise each
    # direction's coupling by the model's own latent scales (g* = g *
    # sd(prior source levels) / sd(target changes)) and report |g*_AB| - |g*_BA|.
    dom_rows: list[dict] = []
    dominance_pair = spec.extra.get("dominance_pair")
    if dominance_pair:
        a, b = dominance_pair
        section_header(f"Reciprocal dominance: {a} <-> {b}")
        x = post["x_latent"]

        def _std_coupling(src: str, tgt: str):
            g = post[coupling_names[(src, tgt)]]
            sd_src = x.isel(wave=slice(0, -1)).sel(outcome=src).std(
                dim=("child", "wave")
            )
            sd_dt = x.sel(outcome=tgt).diff("wave").std(dim=("child", "wave"))
            return g * sd_src / sd_dt

        g_ab = _std_coupling(a, b)  # prior a -> b change
        g_ba = _std_coupling(b, a)  # prior b -> a change
        contrast = abs(g_ab) - abs(g_ba)
        dom_rows = [
            _coef_row(f"std g ({a} -> {b} change)", g_ab.values, ctx.reporting.ci_prob),
            _coef_row(f"std g ({b} -> {a} change)", g_ba.values, ctx.reporting.ci_prob),
            _coef_row(
                f"|std g {a}->{b}| - |std g {b}->{a}| (dominance)",
                contrast.values,
                ctx.reporting.ci_prob,
            ),
        ]
        dom_df = pd.DataFrame(dom_rows)
        save_table(ctx, "dominance_summary", dom_df)
        print_table(
            ranked_dataframe_table(
                dom_df,
                title="SD-standardised reciprocal couplings",
                columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    # Per-child fitted-vs-observed panels (#317 fig 2) for the focal reading target.
    write_panel_child_fit(ctx, latent_name="x_latent", focal_symbol=reading_symbol)

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "outcomes": list(outcomes),
            "reading_symbol": reading_symbol,
            "couplings": {tgt: list(srcs) for tgt, srcs in couplings.items()},
            "lagged_change_couplings": {
                tgt: list(srcs) for tgt, srcs in lagged_change_couplings.items()
            },
            "arm_window_intercepts": arm_window,
            "covariate_block": list(covariate_block),
            "covariate_targets": list(covariate_targets),
            "coupling_summary": rows,
            **({"itt_window1_contrast": itt_rows} if itt_rows else {}),
            **({"dominance_summary": dom_rows} if dom_rows else {}),
        },
    )

    return finalize_report(ctx)


def fit_growth(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Joint multivariate latent growth-curve model (LRP69 core / LRP70 factor).

    Characterises each verbal/reading measure's within-child trajectory across the
    four RLI waves and reports whether **baseline non-verbal ability** (``blocks``)
    predicts trajectory shape: ``gamma`` on the growth *rate* (the headline Q5
    estimand) and ``delta`` on the baseline *level*. With ``use_shared_factor`` a
    rank-1 shared growth-tempo factor couples the slopes and the block-design ->
    common-tempo association is read out post-hoc. Every non-randomised term is an
    **adjusted, latent-GA-confounded association**, never causal (locked DAG,
    ``notes/202606231600-dag-revision-consolidated.md``).
    """
    require_spec(spec, "growth")

    # Resolve and validate the family contract before the context resets an output
    # directory or the loader reads any data (#394 pillar 4). One plan drives
    # preparation, factory arguments, the teaching recipe and config.json.
    plan = resolve_growth_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    outcomes = plan.outcomes
    baseline_cov = plan.baseline_covariate
    use_factor = plan.use_shared_factor
    age_ability = plan.age_ability_interaction

    section_header("Prepare data")
    panel = load_wave_panel(**plan.prepare_kwargs())
    ctx.prepared = panel

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_growth_model(panel, **plan.factory_kwargs())
    attach_built(ctx, built)

    render_model_graph(ctx)

    diag_vars = [
        "gamma", "delta", "beta", "alpha", "sigma_slope", "sigma_intercept", "kappa"
    ]
    if use_factor:
        diag_vars.append("loading")
    if age_ability:
        # LRP85 (#228 item 10): baseline-age main effect + the headline
        # age0 × ability interaction on the growth rate.
        diag_vars.extend(["gamma_age", "gamma_int"])

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # One check per measure: ``y_obs`` flattens (child, wave, outcome) into a single
    # vector, so a lone overlay would pool scales with different maxima. The first
    # outcome keeps the unsuffixed filename the report partial expects.
    for _i, _sym in enumerate(outcomes):
        _diag.save_prior_predictive_plot(
            ctx,
            _sym,
            node="y_obs",
            filename_stem=(
                "prior_predictive_check"
                if _i == 0
                else f"prior_predictive_check_{_sym.lower()}"
            ),
        )

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=["y_obs"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=None)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Headline Q5 output: baseline non-verbal ability -> trajectory shape. The
    # gamma (growth-rate) rows are the answer; delta (level) and beta (mean slope)
    # round out the trajectory characterisation. All adjusted associations.
    section_header("Non-verbal ability -> trajectory shape (Q5)")
    gs = _report.growth_association_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
    save_table(ctx, "growth_association_summary", gs)
    save_forest_plot(ctx, ["gamma"], name="gamma_forest.png")
    print_table(
        ranked_dataframe_table(
            gs[gs["coefficient"] == "gamma"],
            title="Baseline non-verbal ability -> growth rate (gamma, logit; 95% ETI)",
            columns=[
                "outcome", "median", "lo89", "hi89", "prob_positive",
                "favoured_direction_label",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # Factor layer: is there *residual* coupling between baseline non-verbal
    # ability and the common growth tempo, beyond what the model already
    # attributes to block-design directly (the gamma/delta terms)? Block-design
    # enters the trajectory as a predictor, so G_tempo is the shared tempo net of
    # that modelled effect — this correlation is therefore a *residual*
    # association, not the total "does ability predict tempo". Read out post-hoc as
    # the per-draw correlation between each child's latent tempo G_i and their
    # standardised block-design score: independent a priori, but the posterior can
    # still correlate G and blocks through the likelihood. Descriptive only.
    tempo_corr: dict[str, float] | None = None
    if use_factor and "G_tempo" in ctx.trace.posterior:
        G = (
            ctx.trace.posterior["G_tempo"]
            .stack(sample=("chain", "draw"))
            .transpose("child", "sample")
            .values
        )  # (N, S)
        zb = np.asarray(panel.baseline[baseline_cov], dtype=float)  # (N,)
        Gc = G - G.mean(axis=0, keepdims=True)
        zc = (zb - zb.mean())[:, None]
        denom = np.sqrt((Gc**2).sum(0) * (zc**2).sum(0)) + 1e-12
        corr = (Gc * zc).sum(0) / denom  # (S,)
        lo_q = (1 - ctx.reporting.ci_prob) / 2
        tempo_corr = {
            "median": float(np.median(corr)),
            "lo": float(np.quantile(corr, lo_q)),
            "hi": float(np.quantile(corr, 1 - lo_q)),
            "lo50": float(np.quantile(corr, 0.25)),
            "hi50": float(np.quantile(corr, 0.75)),
            "prob_pos": float(np.mean(corr > 0)),
        }
        save_table(ctx, "growth_tempo_corr", pd.DataFrame([tempo_corr]))
        rprint(
            f"[bold]blocks <-> growth-tempo residual corr:[/bold] {tempo_corr['median']:+.3f} "
            f"[{tempo_corr['lo']:+.3f}, {tempo_corr['hi']:+.3f}] "
            f"P(>0)={tempo_corr['prob_pos']:.3f}"
        )

    # Data-space figures (#317): per-measure cohort trajectory (no arm — growth's
    # "arm" is a latent tempo, not an observed randomised arm) and per-child
    # fitted-vs-observed panels for a focal outcome.
    write_panel_trajectory(ctx, latent_name="theta")
    write_panel_child_fit(ctx, latent_name="theta", focal_symbol=outcomes[0])

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "outcomes": list(outcomes),
            "baseline_covariate": baseline_cov,
            "use_shared_factor": use_factor,
            "growth_association_summary": gs.to_dict("records"),
            **({"blocks_tempo_corr": tempo_corr} if tempo_corr else {}),
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Historical group-by-wave growth (RLMHG, #165 - first non-RLI dataset)
# ---------------------------------------------------------------------------


def fit_historical_growth(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Historical group-by-wave growth model (RLMHG, #165).

    A descriptive natural-history growth model for a non-RLI historical cohort
    (the Byrne reading-language-memory study), run through the shared
    statistical-model pipeline so it uses the same sampler, convergence gate,
    output layout and report conventions as the intervention models. It is
    **not** an intervention-effect model - ``group`` carries no treatment
    semantics (see :func:`factories.build_historical_growth_model`).
    """
    require_spec(spec, "historical_growth")

    ctx = make_context(spec, config)

    section_header("Prepare data")
    study_id = spec.extra.get("study_id", spec.study_id)
    measure = spec.extra.get("measure", spec.outcome_symbol or "basread")
    waves = tuple(spec.extra.get("waves", (1, 2, 3)))
    extension_waves = tuple(spec.extra.get("extension_waves", ()))
    dataset, measures = _datasets.resolve_dataset(study_id)
    if measure not in measures:
        raise KeyError(f"measure {measure!r} not registered for study {study_id!r}")
    panel = load_longitudinal_panel(
        dataset,
        [measures[measure]],
        waves=waves,
        complete_case=True,
        extension_waves=extension_waves,
    )
    ctx.prepared = panel

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_historical_growth_model(
        panel,
        measure=measure,
        eta_prior_sigma=spec.extra.get(
            "eta_prior_sigma",
            default_of(_factories.build_historical_growth_model, "eta_prior_sigma"),
        ),
        sigma_subject_prior_sigma=spec.extra.get(
            "sigma_subject_prior_sigma",
            default_of(
                _factories.build_historical_growth_model, "sigma_subject_prior_sigma"
            ),
        ),
        kappa_prior_sigma=spec.extra.get(
            "kappa_prior_sigma",
            default_of(_factories.build_historical_growth_model, "kappa_prior_sigma"),
        ),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    diag_vars = ["eta_cell", "sigma_subject", "kappa"]
    diag_vars += [
        v
        for v in (
            "growth_first_next_items",
            "growth_next_last_items",
            "growth_first_last_items",
        )
        if v in ctx.model.named_vars
    ]

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, measure, node="score")

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=["score"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Descriptive summaries: observed complete-case baseline (the Table 2 audit
    # target), posterior group-by-wave fitted means, and within-group / between-
    # group growth in items.
    section_header("Growth summaries")
    measure_label = measures[measure].label
    baseline = _historical.observed_baseline(panel, measure, measure_label)
    save_table(ctx, "observed_complete_case_baseline", baseline)
    cells = _historical.cell_summary(ctx.trace, panel, measure, measure_label, baseline)
    save_table(ctx, "posterior_cell_summary", cells)
    growth = _historical.growth_summary(ctx.trace, panel, measure)
    save_table(ctx, "posterior_growth_summary", growth)
    write_prior_pushforward(
        ctx, growth_contrast_pushforward_rows(ctx, panel, measure)
    )
    print_table(
        ranked_dataframe_table(
            growth,
            title=(
                f"{measure_label} growth (items) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
            ),
            columns=["label", "readgrp_label", "mean", "q_lo", "q_hi", "p_gt_0"],
            rank_column=False,
            precision=2,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "study_id": study_id,
            "measure": measure,
            "measure_label": measure_label,
            "n_trials": panel.n_trials[measure],
            "waves": list(waves),
            "extension_waves": list(extension_waves),
            "groups": dict(
                zip(panel.group_codes, panel.group_labels, strict=True)
            ),
            "n_subjects": panel.n_subjects,
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Byrne (RLM) Phase B/D fits (#338): adjusted, horseshoe, corr_factor, joint
# ---------------------------------------------------------------------------


def _rlm_nuisance_names(frame) -> list[str]:
    """The group-nuisance coefficient names the RLM factories create."""
    codes = sorted(frame.group_labels)
    counts = {c: int((frame.group_code == c).sum()) for c in codes}
    reference = max(counts, key=lambda c: (counts[c], -c))
    return [
        "beta_group_nuisance_"
        + frame.group_labels[c].lower().replace(" ", "_").replace("-", "_")
        for c in codes
        if c != reference
    ]


def _rlm_natural_scale_contrasts(
    ctx: StatisticalFitContext, frame, headline: list, hdi: float
) -> pd.DataFrame:
    """Predicted +1 SD contrast per predictor on the items scale (RLM span frame).

    The Byrne analogue of :func:`_natural_scale_contrasts`: for two children with
    the same pre-wave outcome score (held at the sample mean) who differ by one
    SD on a single predictor, the model-implied difference in outcome items at
    the later wave, per posterior draw.
    """
    from scipy.special import expit

    post = ctx.trace.posterior
    outcome = frame.outcome
    N = frame.n_trials[outcome]
    mean_pre_logit = float(np.mean(frame.pre_logit[outcome]))

    def draws(name: str) -> np.ndarray:
        return post[name].stack(sample=("chain", "draw")).values

    base_eta = draws("alpha") + draws("gamma_own") * mean_pre_logit
    base_items = N * expit(base_eta)
    lo_q, hi_q = (1 - hdi) / 2, 1 - (1 - hdi) / 2
    rows = []
    for k in headline:
        delta = N * expit(base_eta + draws(f"beta_{k}")) - base_items
        rows.append(
            {
                "predictor": k,
                "label": frame.predictor_labels.get(k, k),
                "delta_words_median": float(np.median(delta)),
                "delta_words_mean": float(np.mean(delta)),
                "delta_words_lo": float(np.quantile(delta, lo_q)),
                "delta_words_hi": float(np.quantile(delta, hi_q)),
                "delta_words_lo50": float(np.quantile(delta, 0.25)),
                "delta_words_hi50": float(np.quantile(delta, 0.75)),
                "prob_pos": float(np.mean(delta > 0)),
            }
        )
    return pd.DataFrame(rows)


def fit_rlm_adjusted(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne between-child adjusted fit (#338 Phase D, ``lrp-rlm-adj-001``).

    The RLI ``fit_adjusted`` shape on the Byrne span frame: the mutually-adjusted
    wave-1-predictors -> later-wave outcome regression (pooled three-group with
    non-interpretable group-nuisance dummies, per the 2026-07-16 sign-off), the
    per-predictor bivariate comparison fits, a slope-prior sensitivity sweep and
    the items-scale +1 SD contrasts. Writes ``predictor_associations.csv``,
    ``predicted_gain_words.csv`` and ``prior_sensitivity.csv`` so the shared
    ``adjusted`` report partial and key-findings builder apply unchanged. Every
    coefficient is an adjusted association - nothing in this cohort is causal.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        load_rlm_span_frame,
    )

    require_spec(spec, "adjusted")
    e = spec.extra
    outcome = spec.outcome_symbol or "basread"
    predictor_measures = tuple(
        e.get("predictor_measures", ("bpvs", "trog", "basdig", "bassim", "basnum"))
    )
    include_age = bool(e.get("use_age_predictor", True))
    pre_wave = int(e.get("pre_wave", 1))
    post_wave = int(e.get("post_wave", 3))
    sigma0 = float(
        e.get(
            "predictor_slope_sigma",
            default_of(_factories.build_rlm_adjusted_model, "predictor_slope_sigma"),
        )
    )
    prior_sens = list(e.get("prior_sensitivity_sigmas", [0.5, 0.7]))

    # 94% intervals, matching the RLI adjusted-family convention.
    ctx = make_context(spec, config, ci_prob=0.89)
    hdi = ctx.reporting.ci_prob

    section_header("Prepare data")
    frame = load_rlm_span_frame(
        outcome=outcome,
        predictor_measures=predictor_measures,
        include_age=include_age,
        pre_wave=pre_wave,
        post_wave=post_wave,
    )
    ctx.prepared = frame
    headline = list(frame.predictors)
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_adjusted_model(
        frame, predictors=headline, predictor_slope_sigma=sigma0
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome, node="y_post")

    run_sampling_and_loo(ctx)

    beta_names = [f"beta_{k}" for k in headline]
    nuisance = _rlm_nuisance_names(frame)
    diag_vars = ["alpha", "gamma_own", "kappa", *beta_names, *nuisance]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx)

    section_header("Extended diagnostics")
    _primary_gate = _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _primary_converged = _report.convergence_gate_clean_passed(_primary_gate)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # --- Adjusted vs bivariate associations --------------------------------
    section_header("Predictor associations (adjusted vs bivariate)")
    adjusted = {k: _beta_summary(ctx.trace, f"beta_{k}", hdi) for k in headline}
    bivariate: dict[str, dict] = {}
    biv_converged: dict[str, object] = {}
    for k in headline:
        b = _factories.build_rlm_adjusted_model(
            frame, predictors=[k], predictor_slope_sigma=sigma0
        )
        t, conv = _sample_model(
            b.model, ctx.sampling, label=f"{spec.model_id} bivariate {k}"
        )
        bivariate[k] = _beta_summary(t, f"beta_{k}", hdi)
        biv_converged[k] = conv["converged"]
    rows = []
    for k in headline:
        a, bv = adjusted[k], bivariate[k]
        rows.append(
            {
                "predictor": k,
                "label": frame.predictor_labels.get(k, k),
                "adj_median": a["median"],
                "adj_mean": a["mean"],
                "adj_lo": a["lo"],
                "adj_hi": a["hi"],
                "adj_lo50": a["lo50"],
                "adj_hi50": a["hi50"],
                "adj_prob_pos": a["prob_pos"],
                "biv_median": bv["median"],
                "biv_mean": bv["mean"],
                "biv_lo": bv["lo"],
                "biv_hi": bv["hi"],
                "biv_lo50": bv["lo50"],
                "biv_hi50": bv["hi50"],
                "biv_prob_pos": bv["prob_pos"],
                "adjusted_converged": _primary_converged,
                "bivariate_converged": biv_converged[k],
            }
        )
    assoc = pd.DataFrame(rows)
    save_table(ctx, "predictor_associations", assoc)
    _pf_assoc = assoc
    # Estimand-scale prior check on the headline adjusted associations (#381).
    # Driven off the association table just written, not off ``headline``: the
    # missing-data indicators are dropped from that table as nuisance
    # subgroup offsets, and a prior row for a term the report does not show
    # would contradict the nuisance labelling it was dropped for.
    _pf_n = pushforward_n_trials(ctx, outcome)
    _pf_outcome = pushforward_outcome_label(ctx, outcome)
    write_prior_pushforward(
        ctx,
        marginal_pushforward_rows(
            ctx,
            [
                (
                    f"beta_{r.predictor}",
                    f"the adjusted association of +1 SD {r.label} with {_pf_outcome}",
                )
                for r in _pf_assoc.itertuples()
            ],
            n_trials=_pf_n,
            convention="forward",
        ),
    )
    print_table(
        ranked_dataframe_table(
            assoc,
            title=f"Wave-{pre_wave} predictors of {outcome} at wave {post_wave} "
            f"(adjusted vs bivariate) - {int(hdi * 100)}% CI",
            columns=[
                "label", "adj_mean", "adj_lo", "adj_hi", "adj_prob_pos",
                "biv_mean", "biv_prob_pos",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Items-scale contrasts (the key-findings headline) ------------------
    section_header("Items-scale +1 SD contrasts")
    gain_words = _rlm_natural_scale_contrasts(ctx, frame, headline, hdi)
    save_table(ctx, "predicted_gain_words", gain_words)

    # --- Prior-sensitivity sweep over the slope sigma ------------------------
    section_header("Prior sensitivity (slope sigma)")
    sens_rows = []
    for sig in [sigma0, *prior_sens]:
        if sig == sigma0:
            t, conv = ctx.trace, {"converged": _primary_converged}
        else:
            b = _factories.build_rlm_adjusted_model(
                frame, predictors=headline, predictor_slope_sigma=float(sig)
            )
            t, conv = _sample_model(
                b.model, ctx.sampling, label=f"{spec.model_id} sigma={sig}"
            )
        for k in headline:
            s = _beta_summary(t, f"beta_{k}", hdi)
            sens_rows.append(
                {
                    "predictor_slope_sigma": float(sig),
                    "predictor": k,
                    "mean": s["mean"],
                    "lo": s["lo"],
                    "hi": s["hi"],
                    "prob_pos": s["prob_pos"],
                    "subfit_converged": conv["converged"],
                }
            )
    sens = pd.DataFrame(sens_rows)
    save_table(ctx, "prior_sensitivity", sens)

    write_run_metadata(
        ctx,
        extra={
            "study_id": "rlm",
            "outcome": outcome,
            "pre_wave": pre_wave,
            "post_wave": post_wave,
            "predictors": headline,
            "group_nuisance_terms": nuisance,
            "n_children": frame.n_children,
            "predictor_slope_sigma": sigma0,
        },
    )
    return finalize_report(ctx)


def fit_rlm_horseshoe(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne horseshoe predictor-ranking fit (#338 Phase D, ``lrp-rlm-hs-001``).

    The RLI gain-framing ``fit_horseshoe`` on the Byrne span frame: one
    regularised-horseshoe regression over the wave-1 predictor set (age
    included), ranked by posterior ``P(|beta| > delta)``. Writes
    ``predictor_ranking.csv`` so the shared ``horseshoe`` partial and
    key-findings builder apply unchanged. There is no gradient-boosting layer
    for the Byrne cohort, so no ``horseshoe_vs_gb.csv`` comparison is written -
    the cross-check partner here is ``lrp-rlm-adj-001``.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        load_rlm_span_frame,
    )

    require_spec(spec, "horseshoe")
    e = spec.extra
    outcome = spec.outcome_symbol or "basread"
    predictor_measures = tuple(
        e.get("predictor_measures", ("bpvs", "trog", "basdig", "bassim", "basnum"))
    )
    include_age = bool(e.get("use_age_predictor", True))
    pre_wave = int(e.get("pre_wave", 1))
    post_wave = int(e.get("post_wave", 3))
    delta = float(e.get("delta", 0.1))
    tau0 = float(e.get("tau0", 0.1))
    slab_scale = float(e.get("slab_scale", 2.0))
    slab_df = float(e.get("slab_df", 4.0))

    ctx = make_context(spec, config, ci_prob=0.89)

    section_header("Prepare data")
    frame = load_rlm_span_frame(
        outcome=outcome,
        predictor_measures=predictor_measures,
        include_age=include_age,
        pre_wave=pre_wave,
        post_wave=post_wave,
    )
    ctx.prepared = frame
    predictors = list(frame.predictors)
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_horseshoe_model(
        frame,
        predictors=predictors,
        tau0=tau0,
        slab_scale=slab_scale,
        slab_df=slab_df,
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome, node="y_post")

    run_sampling_and_loo(ctx)

    nuisance = _rlm_nuisance_names(frame)
    diag_vars = ["alpha", "gamma_own", "kappa", "hs_tau", "hs_c2", "beta", *nuisance]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Predictor ranking")
    ranking = _report.horseshoe_ranking(ctx.trace, delta=delta)
    save_table(ctx, "predictor_ranking", ranking)
    print_table(
        ranked_dataframe_table(ranking, title="Horseshoe predictor ranking")
    )
    write_prior_pushforward(ctx, horseshoe_pushforward_rows(ctx, predictors, outcome))

    write_run_metadata(
        ctx,
        extra={
            "study_id": "rlm",
            "framing": "gain",
            "outcome": outcome,
            "pre_wave": pre_wave,
            "post_wave": post_wave,
            "predictors": predictors,
            "group_nuisance_terms": nuisance,
            "delta": delta,
            "tau0": tau0,
            "slab_scale": slab_scale,
            "slab_df": slab_df,
            "ranking_top": ranking.head(3)[["predictor", "p_abs_gt_delta"]].to_dict(
                "records"
            ),
        },
    )
    return finalize_report(ctx)


def fit_rlm_corr_factor(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne correlated-domain-factor measurement fit (#338 Phase B, mm-001).

    Measurement-only (per the 2026-07-16 sign-off): loadings/communalities and
    the domain-factor correlation matrix over the wave-3 nine-measure battery,
    no structural leg. Writes ``loadings_summary.csv``,
    ``factor_correlation.csv`` and ``factor_correlation_summary.csv`` in the
    RLI ``corr_factor`` schema so the shared partial and key-findings builder
    apply unchanged. LOO is not computed, matching the RLI corr-factor family.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        load_rlm_wave_battery,
    )

    require_spec(spec, "corr_factor")
    e = spec.extra
    wave = int(e.get("wave", 3))
    domains = {k: tuple(v) for k, v in e["domains"].items()}
    reliability = float(e.get("single_indicator_reliability", 0.8))
    lkj_eta = float(e.get("lkj_eta", 2.0))
    comm_alpha = float(
        e.get("comm_alpha", default_of(_factories.build_rlm_corr_factor_model, "comm_alpha"))
    )
    comm_beta = float(
        e.get("comm_beta", default_of(_factories.build_rlm_corr_factor_model, "comm_beta"))
    )

    ctx = make_context(spec, config)
    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0

    section_header("Prepare data")
    symbols = tuple(dict.fromkeys(s for syms in domains.values() for s in syms))
    battery = load_rlm_wave_battery(wave=wave, measure_symbols=symbols)
    ctx.prepared = battery
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_corr_factor_model(
        battery,
        domains=domains,
        single_indicator_reliability=reliability,
        comm_alpha=comm_alpha,
        comm_beta=comm_beta,
        lkj_eta=lkj_eta,
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_dist_overlay(ctx)

    run_sampling_and_loo(ctx, compute_loo=False)

    diag_vars = ["lambda_free", "sigma_free", "factor_corr_pairs"]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity (#381), as in the RLI ``corr_factor`` family:
    # LOO is skipped here, so the log groups have to be added explicitly before the
    # reported loadings, residual scales and factor correlations can be power-scaled.
    # #381 exempted this model on the grounds that its posterior had not converged;
    # since the #383 ``LKJCorr`` fix it does (0 divergences, max R-hat 1.0004), so the
    # exemption no longer applies and a latent-factor model is exactly where an
    # unmeasured prior dependence would matter most.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=["Z_obs"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    # Indicator-scale prior check (#381) — the measurement families' stand-in for
    # the estimand pushforward the outcome families get. AFTER save_trace, which
    # is what attaches the prior/prior_predictive groups to ctx.trace on a fresh
    # fit: called earlier, the check found no prior_predictive group and skipped
    # silently (#383) — the re-emitted reporting artefacts never showed it
    # because a reused trace arrives with its groups already on disk.
    write_indicator_prior_check(ctx, ["Z_obs"])
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    post = ctx.trace.posterior

    # --- Loadings + communalities (the measurement headline) ----------------
    section_header("Loadings + communalities")
    from language_reading_predictors.statistical_models import (
        rlm_corr_factor_summaries as _rlm_summaries,
    )

    load_df = _rlm_summaries.loadings_communalities_table(post, domains, lo_q=lo_q)
    save_table(ctx, "loadings_summary", load_df)
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings, correlations + communalities - {int(hdi * 100)}% CI",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Factor correlation matrix + per-pair summary ------------------------
    section_header("Factor correlation")
    corr_df = _rlm_summaries.factor_correlation_matrix(post)
    save_table(ctx, "factor_correlation", corr_df, index=True)
    corr_summary_df = _rlm_summaries.factor_correlation_pairs(post, lo_q=lo_q)
    save_table(ctx, "factor_correlation_summary", corr_summary_df)
    print_table(
        ranked_dataframe_table(
            corr_summary_df,
            title=f"Domain-factor correlations - {int(hdi * 100)}% CI",
            columns=["domain_i", "domain_j", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "study_id": "rlm",
            "wave": wave,
            "domains": {k: list(v) for k, v in domains.items()},
            "single_indicator_reliability": reliability,
            "n_children": battery.n_children,
            "structural_leg": False,
        },
    )
    return finalize_report(ctx)


def fit_rlm_joint_growth(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne joint correlated growth fit (#338 Phase B, ``lrp-rlm-jc-001``).

    Fits :func:`factories.build_rlm_joint_growth_model` over a small measure set
    and reports the between-child cross-measure correlation matrix of the
    stable child levels (the headline), plus per-measure fitted cells and
    common-window growth via the shared historical summaries. LOO is not
    computed: the model has one likelihood node per measure, so a single
    pointwise PSIS-LOO is not defined for it (documented in the report).
    """
    require_spec(spec, "historical_joint")
    e = spec.extra
    study_id = e.get("study_id", spec.study_id)
    measure_syms = tuple(e.get("measures", ("basread", "bpvs", "basdig")))
    waves = tuple(e.get("waves", (1, 2, 3)))
    extension_waves = tuple(e.get("extension_waves", ()))

    ctx = make_context(spec, config)

    section_header("Prepare data")
    dataset, measures = _datasets.resolve_dataset(study_id)
    for m in measure_syms:
        if m not in measures:
            raise KeyError(f"measure {m!r} not registered for study {study_id!r}")
    panel = load_longitudinal_panel(
        dataset,
        [measures[m] for m in measure_syms],
        waves=waves,
        complete_case=True,
        extension_waves=extension_waves,
    )
    ctx.prepared = panel
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_joint_growth_model(
        panel,
        measures=measure_syms,
        eta_prior_sigma=e.get("eta_prior_sigma", 1.5),
        sigma_subject_prior_sigma=e.get("sigma_subject_prior_sigma", 1.0),
        kappa_prior_sigma=e.get("kappa_prior_sigma", 50.0),
        lkj_eta=e.get("lkj_eta", 2.0),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # One likelihood node per measure, so emit one check per measure rather than a
    # pooled overlay: these scales have different maxima and pooling their counts has
    # no interpretable predictive distribution (same reasoning as the joint family's
    # symbol-suffixed checks).
    for _sym in measure_syms:
        _diag.save_prior_predictive_plot(
            ctx,
            _sym,
            node=f"score_{_sym}",
            filename_stem=f"prior_predictive_check_{_sym.lower()}",
        )

    run_sampling_and_loo(ctx, compute_loo=False)

    diag_vars = ["eta_cell", "sigma_subject", "kappa", "measure_corr_pairs"]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381). This family
    # is ``compute_loo=False`` (one likelihood node per measure, so a single pointwise
    # PSIS-LOO is undefined — not a likelihood PyMC cannot evaluate), so the groups
    # psense needs are not attached by the sampling stage and have to be requested
    # here. ``strict=False`` because psense is a secondary diagnostic and must not
    # crash a fit: today both groups are in fact refused, but by a *naming* seam
    # rather than an intractable likelihood — the model draws
    # ``pm.LKJCorr("measure_corr_chol", ...)`` and PyMC stores its value variable as
    # ``measure_corr_chol_cholesky``, which ``get_untransformed_name`` mangles (see
    # notes/202607261700-psense-coverage-backfill.md and the upstream draft in
    # notes/assets/). That is plausibly fixable upstream; when it is, this call site
    # needs no change. Meanwhile the fit degrades to a warning and gets no psense,
    # which is a *measured and declined* exemption rather than the silent absence it
    # was before.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=[f"score_{m}" for m in measure_syms])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0
    post = ctx.trace.posterior

    # --- Cross-measure correlation of stable child levels (the headline) ----
    section_header("Cross-measure correlation")
    corr_draws = post["measure_corr"]
    mnames = [str(m) for m in post["measure"].values]
    corr_df = pd.DataFrame(
        corr_draws.mean(dim=("chain", "draw")).values, index=mnames, columns=mnames
    )
    save_table(ctx, "measure_correlation", corr_df, index=True)
    corr_stacked = corr_draws.stack(sample=("chain", "draw"))
    labels = {
        m: str(measures[m].label) if m in measures else m for m in mnames
    }
    corr_rows = []
    for i, mi in enumerate(mnames):
        for j, mj in enumerate(mnames):
            if j <= i:
                continue
            pair = np.asarray(
                corr_stacked.isel(measure=i, measure_b=j).values
            ).reshape(-1)
            corr_rows.append(
                {
                    "measure_i": mi,
                    "measure_j": mj,
                    "label_i": labels[mi],
                    "label_j": labels[mj],
                    "median": float(np.median(pair)),
                    "mean": float(np.mean(pair)),
                    "lo": float(np.quantile(pair, lo_q)),
                    "hi": float(np.quantile(pair, 1 - lo_q)),
                    "lo50": float(np.quantile(pair, 0.25)),
                    "hi50": float(np.quantile(pair, 0.75)),
                    "prob_pos": float(np.mean(pair > 0)),
                }
            )
    corr_summary_df = pd.DataFrame(corr_rows)
    save_table(ctx, "measure_correlation_summary", corr_summary_df)
    print_table(
        ranked_dataframe_table(
            corr_summary_df,
            title=f"Between-child cross-measure correlations - {int(hdi * 100)}% CI",
            columns=["label_i", "label_j", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # --- Per-measure fitted cells + growth (shared historical summaries) ----
    section_header("Per-measure growth summaries")
    # One estimand-scale prior row per measure per contrast (#381), accumulated
    # across the loop so the joint fit writes a single table rather than each
    # measure overwriting the last.
    _pf_rows: list[dict[str, object]] = []
    for m in measure_syms:
        label = measures[m].label
        baseline = _historical.observed_baseline(panel, m, label)
        save_table(
            ctx, f"observed_complete_case_baseline_{m}", baseline, register=False
        )
        cells = _historical.cell_summary(
            ctx.trace,
            panel,
            m,
            label,
            baseline,
            mean_var=f"mean_items_{m}",
            fitted_var=f"fitted_mean_items_obs_{m}",
        )
        save_table(ctx, f"posterior_cell_summary_{m}", cells, register=False)
        growth = _historical.growth_summary(
            ctx.trace, panel, m, fitted_var=f"fitted_mean_items_obs_{m}"
        )
        save_table(ctx, f"posterior_growth_summary_{m}", growth)
        _pf_rows.extend(
            growth_contrast_pushforward_rows(
                ctx,
                panel,
                m,
                fitted_var=f"fitted_mean_items_obs_{m}",
                prefix=f"{m}:",
            )
        )
    write_prior_pushforward(ctx, _pf_rows)

    write_run_metadata(
        ctx,
        extra={
            "study_id": study_id,
            "measures": list(measure_syms),
            "measure_labels": {m: measures[m].label for m in measure_syms},
            "waves": list(waves),
            "extension_waves": list(extension_waves),
            "n_subjects": panel.n_subjects,
            "loo_elpd": None,
        },
    )
    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Correlated-domain-factor measurement model (LRPMM01, #134)
# ---------------------------------------------------------------------------


_DEFAULT_DOMAINS = {
    "vocabulary": ("R", "E"),
    "code": ("L", "B"),
    "grammar": ("F", "T"),
}


def fit_correlated_factor(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Correlated-domain-factor measurement model (LRPMM01, #134).

    Fits a reflective CFA with correlated vocabulary / code / grammar factors over
    the standardised T1 skill indicators, plus a structural Beta-Binomial leg for
    the reading-gain outcome, and reports the loadings / communalities, the factor
    correlation matrix, and the measurement-error-corrected factor->gain slopes.
    A triangulation / measurement model, not causal (every factor->gain slope is a
    latent-ability-confounded adjusted association; #115 ID-2).
    """
    require_spec(spec, "corr_factor")

    # #383 settings coherence, checked BEFORE make_context resets the output
    # directory (the #455 principle): each loading parameterisation has knobs the
    # other would silently ignore, so a spec mixing them is declaring settings the
    # fitted model does not use.
    _loading_prior = str(
        spec.extra.get(
            "loading_prior",
            default_of(_factories.build_correlated_factor_model, "loading_prior"),
        )
    )
    if _loading_prior not in {"communality", "free"}:
        raise ValueError(
            f"Spec {spec.model_id}: loading_prior must be 'communality' or 'free'; "
            f"got {_loading_prior!r}"
        )
    _free_knobs = sorted(
        k for k in ("loading_mu", "loading_sigma", "residual_sigma") if k in spec.extra
    )
    _comm_knobs = sorted(k for k in ("comm_alpha", "comm_beta") if k in spec.extra)
    if _loading_prior == "communality" and _free_knobs:
        raise ValueError(
            f"Spec {spec.model_id}: {_free_knobs} only apply to "
            "loading_prior='free'; the communality parameterisation would silently "
            "ignore them. Set loading_prior='free' or drop the knobs."
        )
    if _loading_prior == "free" and _comm_knobs:
        raise ValueError(
            f"Spec {spec.model_id}: {_comm_knobs} only apply to "
            "loading_prior='communality'; the free parameterisation would silently "
            "ignore them. Drop the knobs or use the default parameterisation."
        )

    ctx = make_context(spec, config)
    # The correlated-factor CFA is a small-n latent model; even with the factor
    # scores marginalised out of the measurement likelihood a few boundary
    # divergences survive at the tier-default target_accept, so lift it via the spec
    # (the strict gate requires zero), as the horseshoe fit does for its funnel.

    section_header("Prepare data")
    domains = {
        k: tuple(v) for k, v in (spec.extra.get("domains") or _DEFAULT_DOMAINS).items()
    }
    outcome = spec.outcome_symbol or "W"
    structural_covs = tuple(spec.extra.get("structural_covariates", ("blocks",)))
    # #228 item 14 (errors-in-variables mechanism): optionally regress the outcome on a
    # SUBSET of the fitted factors (e.g. just "code") and/or add the randomised arm G as
    # an adjusted-association covariate. Defaults reproduce mm-001/101 exactly.
    _sf = spec.extra.get("structural_factors")
    structural_factors = tuple(_sf) if _sf is not None else None
    use_group = bool(spec.extra.get("use_group", False))
    indicator_syms = tuple(dict.fromkeys(s for v in domains.values() for s in v))
    measure_outcomes = tuple(dict.fromkeys((outcome, *indicator_syms)))
    prepared = load_and_prepare(
        phase_mode="span",
        post_time=int(spec.extra.get("post_time", 4)),
        outcomes=measure_outcomes,
        covariates=structural_covs,
    )
    ctx.prepared = prepared
    # A structural covariate can go constant on the fitted span rows — e.g. an
    # ``erbto_missing`` indicator that is all-zero because phonological memory is
    # observed for every fitted child at t1 — so the loader drops it. Re-filter to the
    # covariates actually present, mirroring the mechanism/mediation pipelines'
    # #247/#258 re-filter, so the factory is not asked for a coefficient on a dropped
    # covariate (it raises KeyError otherwise) and the effective set is honest.
    _dropped_structural = tuple(c for c in structural_covs if c not in prepared.covariates)
    if _dropped_structural:
        structural_covs = tuple(c for c in structural_covs if c in prepared.covariates)
        rprint(
            "[yellow]fit_correlated_factor: dropped constant structural covariate(s) "
            f"{list(_dropped_structural)} (not in prepared.covariates on the fitted "
            "rows).[/yellow]"
        )
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_correlated_factor_model(
        prepared,
        outcome_symbol=outcome,
        domains=domains,
        structural_covariates=structural_covs,
        structural_factors=structural_factors,
        use_group=use_group,
        use_age=spec.extra.get("use_age", True),
        loading_prior=_loading_prior,
        comm_alpha=spec.extra.get(
            "comm_alpha",
            default_of(_factories.build_correlated_factor_model, "comm_alpha"),
        ),
        comm_beta=spec.extra.get(
            "comm_beta",
            default_of(_factories.build_correlated_factor_model, "comm_beta"),
        ),
        loading_mu=spec.extra.get(
            "loading_mu",
            default_of(_factories.build_correlated_factor_model, "loading_mu"),
        ),
        loading_sigma=spec.extra.get(
            "loading_sigma",
            default_of(_factories.build_correlated_factor_model, "loading_sigma"),
        ),
        residual_sigma=spec.extra.get(
            "residual_sigma",
            default_of(_factories.build_correlated_factor_model, "residual_sigma"),
        ),
        predictor_slope_sigma=spec.extra.get(
            "predictor_slope_sigma",
            default_of(
                _factories.build_correlated_factor_model, "predictor_slope_sigma"
            ),
        ),
        focal_slope_sigma=spec.extra.get(
            "focal_slope_sigma",
            default_of(_factories.build_correlated_factor_model, "focal_slope_sigma"),
        ),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    summary_vars = [
        # ``communality`` is the free RV under the default communality
        # parameterisation (#383) and a Deterministic under the legacy free pair;
        # either way it is a reported quantity, so gate it explicitly alongside
        # the derived lambda_load / sigma_indicator.
        "alpha", "gamma_own", "kappa", "beta_factor", "lambda_load", "sigma_indicator",
        "communality",
        # The headline factor correlations MUST be in the gated set: they are what
        # the report releases, and the global checks (divergences, BFMI) are not a
        # substitute for parameter-specific R-hat / ESS on them. ``factor_corr``
        # itself is unusable for this — its constant unit diagonal has undefined
        # R-hat and zero variance — so the factory exposes the unique off-diagonals
        # as ``factor_corr_pairs``. ``factor_z`` is the latent-score offset the
        # structural leg consumes; gate it too.
        "factor_z",
    ]
    # Only present when there are >= 2 domains (a single factor has no off-diagonal).
    if len(domains) > 1:
        summary_vars.append("factor_corr_pairs")
    if spec.extra.get("use_age", True):
        summary_vars.append("beta_age")
    summary_vars += [f"beta_{c}" for c in structural_covs]
    if use_group:
        summary_vars.append("beta_G")

    section_header("Prior predictive")
    # Draw the full prior, not just the two observed nodes (#381). Restricting
    # ``var_names`` to ``["Z_obs", "y_post"]`` left the persisted ``prior`` group
    # completely empty, so ``save_prior_posterior_plot`` below had nothing to
    # overlay and these three fits shipped with no prior-vs-posterior figure at
    # all — the one measurement family the prior-analysis review most wanted to
    # see. The default (all free RVs + deterministics + observed nodes) is what
    # every other family uses, and ``run_prior_predictive`` falls back to the
    # minimal set on failure.
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome, node="y_post")

    # Two observed nodes (the indicator matrix Z_obs + the structural y_post) make
    # a single-target PSIS-LOO ambiguous, so LOO is skipped here as in the
    # mediation family; this is a measurement / triangulation model, not a
    # predictive one, and #134 turns on the loadings / communalities, not on LOO.
    run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=summary_vars)
    # Power-scaling prior sensitivity (#381): this family does not compute LOO,
    # so add the log groups explicitly, then power-scale the reported parameters.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=summary_vars)

    # Sample both observed nodes (the indicator matrix + the structural outcome).
    # These are two SEPARATE checks, not a joint predictive draw: the factor scores
    # condition on the observed indicator data (``Z_d``), not on the replicated
    # ``Z_obs``, so a replicated indicator is independent of the replicated factor
    # it loads on. ``Z_obs`` is a marginal check of the measurement covariance;
    # ``y_post`` is a check of the structural leg *conditional on the observed
    # indicators*. Together they do not certify the joint model. See the
    # predictive-simulation caveat in ``build_correlated_factor_model``.
    run_ppc(ctx, var_names=["Z_obs", "y_post"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=summary_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    # Indicator-scale prior check (#381). Only ``Z_obs`` is the indicator matrix;
    # ``y_post`` is the structural outcome and is covered by the ordinary
    # prior-predictive plot above. AFTER save_trace, which is what attaches the
    # prior/prior_predictive groups to ctx.trace on a fresh fit: called earlier,
    # the check found no prior_predictive group and skipped silently (#383).
    write_indicator_prior_check(ctx, ["Z_obs"])
    _diag.save_prior_posterior_plot(ctx, var_names=summary_vars)

    post = ctx.trace.posterior
    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0

    # --- Loadings + communalities (the measurement headline) ---
    section_header("Loadings + communalities")
    from language_reading_predictors.statistical_models import (
        corr_factor_summaries as _cf_summaries,
    )

    # The RLI factor loadings live under ``lambda_load`` (the Byrne model uses
    # ``loading``); the residual sigma is free, so lambda is a coefficient on the
    # unit-variance factor, not in general a correlation — the standardised loading /
    # indicator-factor correlation reported alongside is sqrt(communality).
    load_df = _cf_summaries.loadings_communalities_table(
        post, domains, lo_q=lo_q, loading_var="lambda_load"
    )
    save_table(ctx, "loadings_summary", load_df)
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings, correlations + communalities - {int(hdi * 100)}% CI (equal-tailed)",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Factor correlation matrix ---
    section_header("Factor correlation")
    corr_df = _cf_summaries.factor_correlation_matrix(post)
    # Domain names are also used by the structural leg below (beta_factor dims).
    dnames = [str(d) for d in post["domain"].values]
    save_table(ctx, "factor_correlation", corr_df, index=True)
    # The bare mean matrix above is kept for the heatmap, but the house rule is
    # "never a bare point estimate": persist each unique off-diagonal pair with a
    # posterior mean, equal-tailed interval and tail probability alongside it.
    corr_summary_df = _cf_summaries.factor_correlation_pairs(post, lo_q=lo_q)
    save_table(ctx, "factor_correlation_summary", corr_summary_df)

    # --- Structural slopes: factor -> reading gain (adjusted associations) ---
    section_header("Structural slopes (factor -> gain)")
    # The structural leg regresses on all domain factors (beta_factor dims "domain")
    # unless structural_factors isolated a subset (dims "struct_domain", #228 item 14).
    struct_names = list(structural_factors) if structural_factors is not None else dnames
    _bf_dim = "struct_domain" if structural_factors is not None else "domain"
    struct_rows = [
        _coef_row(f"beta_{d}", post["beta_factor"].isel({_bf_dim: k}).values, hdi)
        for k, d in enumerate(struct_names)
    ]
    extra_terms = (
        (["beta_G"] if use_group else [])
        + (["beta_age"] if spec.extra.get("use_age", True) else [])
        + [f"beta_{c}" for c in structural_covs]
    )
    struct_rows += [_coef_row(t, post[t].values, hdi) for t in extra_terms]
    struct_df = pd.DataFrame(struct_rows)
    save_table(ctx, "structural_summary", struct_df)
    print_table(
        ranked_dataframe_table(
            struct_df,
            title=(
                f"Structural slopes (factor -> gain; adjusted associations) - "
                f"{int(hdi * 100)}% CI"
            ),
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "domains": {k: list(v) for k, v in domains.items()},
            "loadings_summary": load_df.to_dict("records"),
            "factor_correlation": corr_df.to_dict(),
            "structural_summary": struct_df.to_dict("records"),
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313)
# ---------------------------------------------------------------------------


_LCF_DEFAULT_DOMAINS = {
    "vocabulary": ("R", "E", "TR", "TE"),
    "code": ("L", "B"),
    "grammar": ("F", "T"),
}


# The LCF exact child-level log-likelihood and constrained-scale log-prior
# recovery are substantive numerical algorithms, isolated in ``lcf_inference``
# so they are testable independently of report publication (#394, pillar 7).
# Re-exported here under their historical private names for existing callers.
_lcf_child_log_likelihood = _lcf_inference.child_log_likelihood
_lcf_log_prior = _lcf_inference.log_prior


def _lcf_stitch_loo(ctx: StatisticalFitContext, built) -> None:
    """Pointwise PSIS-LOO for the longitudinal CFA (custom, per-child stitch).

    The masked-cell likelihood is one ``MvNormal`` per observed-cell pattern, so
    there is no single observed node ``az.loo`` can key on. Compute each pattern
    node's per-child log-likelihood, stitch them into one ``(chain, draw, child)``
    array over all children, and run pointwise LOO on that — the honest per-child
    predictive score an invariance comparison (#313) would use. Attach the exact
    constrained-scale prior terms through the companion workaround above. As in the
    other LOO-enabled families, a failure is fatal: a reporting run must not silently
    complete without the likelihood, prior and predictive diagnostics its output
    contract requires.
    """
    import arviz as az
    import xarray as xr

    stitched = _lcf_child_log_likelihood(ctx.trace, built)
    if "log_likelihood" not in ctx.trace.children:
        ctx.trace["log_likelihood"] = xr.Dataset()
    ctx.trace.log_likelihood["lcf_child"] = stitched
    ctx.trace["log_prior"] = _lcf_log_prior(ctx.trace, ctx.model)
    ctx.loo = az.loo(ctx.trace, var_name="lcf_child", pointwise=True)
    _report.write_loo_summary(ctx)
    print_loo_row(ctx)


# LCF descriptive/comparison summaries live in ``lcf_summaries`` (#394 pillar 6);
# re-exported here under their historical private names for existing callers.
_lcf_observed_domain_corr = _lcf_summaries.observed_domain_corr
_lcf_items_scale = _lcf_summaries.items_scale
_lcf_observed_conditional_slope = _lcf_summaries.observed_conditional_slope
_lcf_concurrent_comparison = _lcf_summaries.concurrent_comparison


def fit_longitudinal_corr_factor(
    spec: ModelSpec, config: str = "dev"
) -> StatisticalFitContext:
    """Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313).

    Fits the four-wave extension of the ``corr_factor`` CFA over the child×wave
    panel: correlated vocabulary / code / grammar factors at every timepoint, with a
    trait/state across-wave structure and the factor scores marginalised out. Reports
    the per-wave latent skill correlation matrices, the conditional (partial) latent
    slopes, the loadings / communalities, and a descriptive comparison against the
    observed same-wave correlations (the #312 anchor). The quantities differ in their
    aggregation and conditioning, so no magnitude ordering is required. A measurement
    / triangulation model — every quantity is a descriptive association, never causal.
    """
    require_spec(spec, "long_corr_factor")

    # #383 settings coherence, checked BEFORE make_context resets the output
    # directory (the #455 principle), mirroring fit_correlated_factor: each loading
    # parameterisation has knobs the other would silently ignore, so a spec mixing
    # them is declaring settings the fitted model does not use.
    _loading_prior = str(
        spec.extra.get(
            "loading_prior",
            default_of(
                _factories.build_longitudinal_corr_factor_model, "loading_prior"
            ),
        )
    )
    if _loading_prior not in {"communality", "free"}:
        raise ValueError(
            f"Spec {spec.model_id}: loading_prior must be 'communality' or 'free'; "
            f"got {_loading_prior!r}"
        )
    _free_knobs = sorted(
        k for k in ("loading_sigma", "residual_sigma") if k in spec.extra
    )
    _comm_knobs = sorted(k for k in ("comm_alpha", "comm_beta") if k in spec.extra)
    if _loading_prior == "communality" and _free_knobs:
        raise ValueError(
            f"Spec {spec.model_id}: {_free_knobs} only apply to "
            "loading_prior='free'; the communality parameterisation would silently "
            "ignore them. Set loading_prior='free' or drop the knobs."
        )
    if _loading_prior == "free" and _comm_knobs:
        raise ValueError(
            f"Spec {spec.model_id}: {_comm_knobs} only apply to "
            "loading_prior='communality'; the free parameterisation would silently "
            "ignore them. Drop the knobs or use the default parameterisation."
        )

    ctx = make_context(spec, config)
    # A small-n latent model; even fully marginalised a few boundary divergences can
    # survive at the tier default, so lift target_accept via the spec (as mm-001 does).

    section_header("Prepare data")
    domains = {
        k: tuple(v)
        for k, v in (spec.extra.get("domains") or _LCF_DEFAULT_DOMAINS).items()
    }
    indicators = tuple(dict.fromkeys(s for v in domains.values() for s in v))
    panel = load_wave_panel(outcomes=indicators)
    ctx.prepared = panel
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_longitudinal_corr_factor_model(
        panel,
        domains=domains,
        loading_prior=_loading_prior,
        comm_alpha=spec.extra.get(
            "comm_alpha",
            default_of(_factories.build_longitudinal_corr_factor_model, "comm_alpha"),
        ),
        comm_beta=spec.extra.get(
            "comm_beta",
            default_of(_factories.build_longitudinal_corr_factor_model, "comm_beta"),
        ),
        loading_sigma=spec.extra.get(
            "loading_sigma",
            default_of(_factories.build_longitudinal_corr_factor_model, "loading_sigma"),
        ),
        residual_sigma=spec.extra.get(
            "residual_sigma",
            default_of(_factories.build_longitudinal_corr_factor_model, "residual_sigma"),
        ),
        lkj_eta=spec.extra.get(
            "lkj_eta",
            default_of(_factories.build_longitudinal_corr_factor_model, "lkj_eta"),
        ),
        factor_mean_sigma=spec.extra.get(
            "factor_mean_sigma",
            default_of(
                _factories.build_longitudinal_corr_factor_model, "factor_mean_sigma"
            ),
        ),
        trait_share_a=spec.extra.get(
            "trait_share_a",
            default_of(
                _factories.build_longitudinal_corr_factor_model, "trait_share_a"
            ),
        ),
        trait_share_b=spec.extra.get(
            "trait_share_b",
            default_of(
                _factories.build_longitudinal_corr_factor_model, "trait_share_b"
            ),
        ),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    z_nodes = built.extras["z_nodes"]
    summary_vars = [
        # ``communality`` is the free RV under the default pooled-budget
        # parameterisation (#383 follow-up) and a Deterministic under the legacy
        # free pair; either way it is a reported quantity, so gate it explicitly
        # alongside the derived lambda_load / sigma_indicator. ``within_share``
        # is the within-wave share of the pooled unit variance that the budget
        # allocates (lambda**2 + sigma**2 in both modes).
        "lambda_load",
        "sigma_indicator",
        "communality",
        "within_share",
        "trait_share",
        # The headline: gate exactly the released per-wave off-diagonal correlations
        # (the full matrix's constant unit diagonal has undefined R-hat).
        "factor_corr_pairs",
    ]

    section_header("Prior predictive")
    # Dedupe: ``communality`` is itself a free RV under the default
    # parameterisation, so appending it unconditionally would double it. The
    # derived lambda_load / sigma_indicator / within_share are named explicitly —
    # as Deterministics they are no longer covered by the free-RV listing, and the
    # prior-vs-posterior overlay below needs their prior draws.
    prior_vars = list(
        dict.fromkeys(
            [
                *(rv.name for rv in built.model.free_RVs),
                "communality",
                "lambda_load",
                "sigma_indicator",
                "within_share",
                "factor_corr_pairs",
                *z_nodes,
            ]
        )
    )
    _diag.run_prior_predictive(ctx, draws=1000, var_names=prior_vars)
    _diag.save_prior_predictive_dist_overlay(ctx)

    # Automatic single-target LOO is ambiguous with per-pattern observed nodes, so
    # sampling runs without it and the per-child stitch below computes LOO instead.
    run_sampling_and_loo(ctx, compute_loo=False)

    section_header("LOO-PSIS (per-child stitch)")
    _lcf_stitch_loo(ctx, built)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=summary_vars)

    run_ppc(ctx, var_names=z_nodes)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=summary_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=None)
    # ``communality`` joins the power-scaled set (#383 follow-up): under the
    # pooled-budget parameterisation it is the free measurement parameter behind
    # the reported loadings table, exactly the place a prior dependence would live.
    _diag.run_psense(ctx, var_names=["factor_corr_pairs", "trait_share", "communality"])
    _diag.save_trace(ctx)
    # Indicator-scale prior check (#381), pooled across the missingness-pattern
    # blocks: each block is its own observed node but the indicators are shared.
    # AFTER save_trace, which is what attaches the prior/prior_predictive groups
    # to ctx.trace on a fresh fit: called earlier, the check found no
    # prior_predictive group and skipped silently (#383).
    write_indicator_prior_check(ctx, z_nodes)
    _diag.save_prior_posterior_plot(ctx, var_names=summary_vars)

    post = ctx.trace.posterior
    hdi = ctx.reporting.ci_prob
    lo_q = (1 - hdi) / 2

    # --- Loadings + communalities (the measurement layer) ---
    section_header("Loadings + communalities")
    dom_of = built.extras["domain_of"]
    load_rows = []
    for j, name in enumerate(str(s) for s in post["indicator"].values):
        lam_d = post["lambda_load"].isel(indicator=j).values.reshape(-1)
        com_d = post["communality"].isel(indicator=j).values.reshape(-1)
        corr_d = np.sqrt(com_d)
        load_rows.append(
            {
                "indicator": name,
                "domain": dom_of.get(name, "?"),
                "loading_mean": float(np.mean(lam_d)),
                "loading_lo": float(np.quantile(lam_d, lo_q)),
                "loading_hi": float(np.quantile(lam_d, 1 - lo_q)),
                "correlation_mean": float(np.mean(corr_d)),
                "communality_mean": float(np.mean(com_d)),
                "communality_lo": float(np.quantile(com_d, lo_q)),
                "communality_hi": float(np.quantile(com_d, 1 - lo_q)),
            }
        )
    load_df = pd.DataFrame(load_rows)
    save_table(ctx, "loadings_summary", load_df)
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings + communalities - {int(hdi * 100)}% CI (equal-tailed)",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Per-wave latent factor correlations (the headline) ---
    section_header("Per-wave latent factor correlations")
    corr_df = _report.longitudinal_factor_correlations(ctx.trace, ci_prob=hdi)
    save_table(ctx, "factor_correlation_by_wave", corr_df)
    print_table(
        ranked_dataframe_table(
            corr_df,
            title=f"Per-wave latent factor correlations - {int(hdi * 100)}% ETI",
            columns=["wave", "domain_i", "domain_j", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # --- Conditional (partial) latent slopes ---
    section_header("Conditional latent slopes")
    slope_df = _report.longitudinal_conditional_slopes(ctx.trace, ci_prob=hdi)
    save_table(ctx, "latent_conditional_slopes", slope_df)

    # --- Trait / state (across-wave) structure ---
    section_header("Trait / state structure")
    ts_rows = []
    for j, d in enumerate(str(s) for s in post["domain"].values):
        pi_d = post["trait_share"].isel(domain=j).values.reshape(-1)
        ts_rows.append(
            {
                "domain": d,
                "trait_share_mean": float(np.mean(pi_d)),
                "trait_share_lo": float(np.quantile(pi_d, lo_q)),
                "trait_share_hi": float(np.quantile(pi_d, 1 - lo_q)),
            }
        )
    ts_df = pd.DataFrame(ts_rows)
    save_table(ctx, "trait_state_summary", ts_df)

    # --- Latent-versus-observed comparison (#312 triangulation anchor) --------
    section_header("Latent-versus-observed correlation comparison")
    obs_df = _lcf_observed_domain_corr(built)
    xcheck_df = _report.disattenuation_crosscheck(corr_df, obs_df)
    save_table(ctx, "disattenuation_crosscheck", xcheck_df)
    n_latent_below = int((~xcheck_df["latent_ge_observed"]).sum())
    n_latent_at_or_above = len(xcheck_df) - n_latent_below
    rprint(
        "[cyan]Latent-versus-observed comparison: "
        f"{n_latent_below} wave/pair(s) are below and {n_latent_at_or_above} are at "
        "or above the mean observed indicator-pair magnitude. This is a descriptive "
        "gap direction between different estimands, not a pass/fail ordering.[/cyan]"
    )

    # --- Items-scale translation for the headline pairs ---
    section_header("Items-scale translation (selected pairs)")
    items_df = _lcf_items_scale(ctx, built)
    save_table(ctx, "latent_items_slopes", items_df)

    # --- Directed comparison with matching concurrent associations (#312) ---
    section_header("Directed LCF-versus-concurrent comparison")
    concurrent_df = _lcf_concurrent_comparison(ctx, built)
    save_table(ctx, "lcf_concurrent_comparison", concurrent_df)
    n_ca_available = int(concurrent_df["ca_available"].sum())
    if n_ca_available < len(concurrent_df):
        rprint(
            "[yellow]Directed #312 comparison is incomplete: "
            f"{n_ca_available}/{len(concurrent_df)} matching concurrent rows were "
            "found under this output root/config. Fit CA002--006 at the same tier "
            "to populate the missing side.[/yellow]"
        )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "domains": {k: list(v) for k, v in domains.items()},
            "invariance": built.extras["invariance"],
            "n_used_children": built.extras["n_used_children"],
            "loadings_summary": load_df.to_dict("records"),
            "factor_correlation_by_wave": corr_df.to_dict("records"),
            "trait_state_summary": ts_df.to_dict("records"),
            # Keep the legacy key for output compatibility; neither count is a gate.
            "disattenuation_reversals": n_latent_below,
            "latent_below_observed_count": n_latent_below,
            "latent_observed_comparison": "descriptive; no required ordering",
            "lcf_concurrent_comparison_rows": int(len(concurrent_df)),
            "lcf_concurrent_comparison_available": n_ca_available,
        },
    )

    return finalize_report(ctx)
