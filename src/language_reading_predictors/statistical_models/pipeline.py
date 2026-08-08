# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end fit pipeline for the statistical models.

``fit_adjusted`` / ``fit_lcsm`` / ``fit_concurrent`` are the entry points for the
LRP65/LRP67/LRP-CA companions, alongside the growth, historical and measurement
families that have yet to migrate.

The ITT and joint families moved to :mod:`pipelines` first (#394 step 5), then the
DiD, dose-response, gain- and level-factors, block-exposure and aligned families,
then mechanism, joint mechanism and mediation (step 6). Their entry points are
re-exported here so model modules and tests keep their import path;
``MIGRATED_FAMILIES`` in ``test_pipeline_boundaries.py`` is the authoritative list,
checked against the package contents.

The shared mechanics they used to carry inline now live in :mod:`runtime` (the
stage binding and spec validation), :mod:`publication` (banners, report template,
model graph), :mod:`adjustment` (the fitted adjustment-set record),
:mod:`prior_artifacts`, :mod:`ppc_artifacts` and :mod:`figure_artifacts`, with the
sub-fit sampler in :mod:`diagnostics` beside the primary one it mirrors. The
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import COLOUR_BLUE
from rich import print as rprint

from language_reading_predictors.models._reporting import (
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
    reporting as _report,
    survival as _survival,
)
from language_reading_predictors.statistical_models.plotting import (
    save_styled_figure,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.concurrent import (
    resolve_concurrent_run_plan,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.diagnostics import sample_subfit
from language_reading_predictors.statistical_models.factories import default_of
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_forest_plot,
    write_panel_child_fit,
    write_panel_trajectory,
)
from language_reading_predictors.statistical_models.growth import (
    resolve_growth_run_plan,
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
from language_reading_predictors.statistical_models.pipelines.joint_mechanism import (
    fit_joint_mechanism as fit_joint_mechanism,
)
from language_reading_predictors.statistical_models.pipelines.level_factors import (
    fit_level_factors as fit_level_factors,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import (
    fit_mechanism as fit_mechanism,
)
from language_reading_predictors.statistical_models.pipelines.mediation import (
    fit_mediation as fit_mediation,
    fit_mediation_multi as fit_mediation_multi,
    fit_mediation_period_stacked as fit_mediation_period_stacked,
    prepare_mediation_data as prepare_mediation_data,
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
    load_longitudinal_panel,
    load_wave_panel,
    logit_safe,
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
# Survival pipeline (LRP-RLI-SURV)
# ---------------------------------------------------------------------------


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
        t, conv = sample_subfit(b.model, ctx.sampling, label=f"{spec.model_id} bivariate {k}")
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
            tr, conv = sample_subfit(
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
        t, conv = sample_subfit(b.model, ctx.sampling, label=f"{spec.model_id} SES complete-case")
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
            trace, conv = sample_subfit(
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
            bt, bconv = sample_subfit(
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
        t, conv = sample_subfit(
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
            t, conv = sample_subfit(
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
