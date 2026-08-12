# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Between-child adjusted-associations orchestration (``kind="adjusted"``, LRP65).

``fit_adjusted`` is the mutually-adjusted between-child regression of full-study
gain on the T1 predictor set — one row per child — with three sensitivity layers
beside it: the bivariate baseline-only association for each predictor, a sweep over
the predictor-slope prior scale, and a complete-case SES fit. ``fit_rlm_adjusted``
is the same family on the Byrne (RLM) span frame, pooled across three groups with
non-interpretable group-nuisance dummies; ``definitions.KINDS`` keys both as
``adjusted``, and they publish the same ``predictor_associations.csv`` schema, so
they live together here.

Nothing in either cohort is randomised. Every predictor slope is an adjusted
association, and the natural-scale contrasts translate a +1 SD difference into
outcome items for readability — not into an effect of changing the predictor.
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
    adjusted as _adjusted,
    diagnostics as _diag,
    factories as _factories,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.plotting import save_styled_figure
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    marginal_pushforward_rows,
    pushforward_n_trials,
    pushforward_outcome_label,
    write_prior_pushforward,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
)
from language_reading_predictors.statistical_models.reporting import beta_summary
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    run_ppc,
    run_sampling_and_loo,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.subfits import run_subfit


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
    plan = _adjusted.resolve_adjusted_run_plan(spec)
    if plan.port != "rli":
        raise ValueError(f"{spec.model_id}: RLM settings require fit_rlm_adjusted")
    outcome = plan.outcome_symbol
    post_time = plan.post_time
    assert post_time is not None
    lang_symbols = plan.language_composite_symbols
    ses_covs = list(plan.ses_covariates)
    sigma0 = plan.predictor_slope_sigma
    prior_sens = list(plan.prior_sensitivity_sigmas)

    # 94% intervals (the brief's convention) rather than the project-wide 95%.
    ctx = make_context(spec, config, ci_prob=0.89)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    hdi = ctx.reporting.ci_prob

    section_header("Prepare data")
    prepared = load_and_prepare(**plan.rli_prepare_kwargs())
    ctx.prepared = prepared
    # Drop any covariate the loader removed as constant on the fitted rows (e.g. a
    # `_missing` indicator that is all-zero once the complete cases are kept) so the
    # model never requests a coefficient for a term that was never estimated (#247).
    covariates = tuple(
        symbol
        for symbol in plan.declared_covariates
        if symbol in prepared.covariates
    )
    if covariates != plan.active_covariates:
        plan = plan.with_active_covariates(covariates)
        ctx.resolved_plan = plan
        _report.write_model_recipe(ctx)
    # Headline predictor key order: skills, language composite, age, tested covariates.
    headline = list(plan.headline_predictors())
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_adjusted_model(
        prepared,
        **plan.rli_factory_kwargs(),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome)

    run_sampling_and_loo(ctx, compute_loo=plan.compute_loo)

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
    adjusted = {k: beta_summary(ctx.trace, f"beta_{k}", hdi) for k in headline}
    bivariate: dict[str, dict] = {}
    biv_converged: dict[str, object] = {}
    for k in headline:
        b = _factories.build_adjusted_model(
            prepared,
            **plan.rli_factory_kwargs(predictors=(k,)),
        )
        res = run_subfit(
            ctx, b, label=f"{spec.model_id} bivariate {k}", role="bivariate"
        )
        bivariate[k] = beta_summary(res.trace, f"beta_{k}", hdi)
        biv_converged[k] = res.converged

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
                **{
                    **plan.rli_factory_kwargs(),
                    "predictor_slope_sigma": sig,
                },
            )
            res = run_subfit(
                ctx,
                b,
                label=f"{spec.model_id} prior-sweep sigma={sig}",
                role="prior_sweep",
            )
            tr = res.trace
            sig_converged = res.converged
        for k in headline:
            ps_rows.append(
                {
                    "sigma": sig,
                    "predictor": k,
                    **beta_summary(tr, f"beta_{k}", hdi),
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
            **plan.rli_prepare_kwargs(include_ses=True),
        )
        # Re-filter against the SES-complete subset: a `_missing` indicator can go
        # constant on this smaller subset even if it survived the headline fit, and the
        # loader then drops it — so rebuild the predictor list here too, or
        # ``build_adjusted_model`` would KeyError on the dropped term (#287 review). The
        # non-covariate predictors (skills / lang / age) are always kept.
        ses_headline = [
            k
            for k in headline
            if k not in plan.active_covariates or k in prepared_ses.covariates
        ]
        ses_covs_fit = [c for c in ses_covs if c in prepared_ses.covariates]
        ses_predictors = ses_headline + ses_covs_fit
        b = _factories.build_adjusted_model(
            prepared_ses,
            **plan.rli_factory_kwargs(predictors=ses_predictors),
        )
        res = run_subfit(
            ctx, b, label=f"{spec.model_id} SES complete-case", role="sensitivity"
        )
        ses_n = int(b.prepared.n_children)
        ses_rows = [
            {
                "predictor": k,
                "label": _adj_label(k),
                "n_children": ses_n,
                **beta_summary(res.trace, f"beta_{k}", hdi),
                "converged": res.converged,
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


def rlm_nuisance_names(frame) -> list[str]:
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
    plan = _adjusted.resolve_adjusted_run_plan(spec)
    if plan.port != "rlm":
        raise ValueError(f"{spec.model_id}: RLI settings require fit_adjusted")
    outcome = plan.outcome_symbol
    pre_wave = plan.pre_wave
    post_wave = plan.post_wave
    assert pre_wave is not None and post_wave is not None
    sigma0 = plan.predictor_slope_sigma
    prior_sens = list(plan.prior_sensitivity_sigmas)

    # 94% intervals, matching the RLI adjusted-family convention.
    ctx = make_context(spec, config, ci_prob=0.89)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)
    hdi = ctx.reporting.ci_prob

    section_header("Prepare data")
    frame = load_rlm_span_frame(**plan.rlm_prepare_kwargs())
    ctx.prepared = frame
    headline = list(frame.predictors)
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_adjusted_model(
        frame, **plan.rlm_factory_kwargs(headline)
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome, node="y_post")

    run_sampling_and_loo(ctx, compute_loo=plan.compute_loo)

    nuisance = rlm_nuisance_names(frame)
    diag_vars = plan.diagnostic_vars(headline, nuisance)
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
    adjusted = {k: beta_summary(ctx.trace, f"beta_{k}", hdi) for k in headline}
    bivariate: dict[str, dict] = {}
    biv_converged: dict[str, object] = {}
    for k in headline:
        b = _factories.build_rlm_adjusted_model(
            frame, **plan.rlm_factory_kwargs((k,))
        )
        res = run_subfit(
            ctx, b, label=f"{spec.model_id} bivariate {k}", role="bivariate"
        )
        bivariate[k] = beta_summary(res.trace, f"beta_{k}", hdi)
        biv_converged[k] = res.converged
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
            t, sig_converged = ctx.trace, _primary_converged
        else:
            b = _factories.build_rlm_adjusted_model(
                frame,
                **{
                    **plan.rlm_factory_kwargs(headline),
                    "predictor_slope_sigma": float(sig),
                },
            )
            res = run_subfit(
                ctx, b, label=f"{spec.model_id} sigma={sig}", role="prior_sweep"
            )
            t = res.trace
            sig_converged = res.converged
        for k in headline:
            s = beta_summary(t, f"beta_{k}", hdi)
            sens_rows.append(
                {
                    "predictor_slope_sigma": float(sig),
                    "predictor": k,
                    "mean": s["mean"],
                    "lo": s["lo"],
                    "hi": s["hi"],
                    "prob_pos": s["prob_pos"],
                    "subfit_converged": sig_converged,
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
