# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Byrne joint correlated-growth orchestration (``kind="historical_joint"``, #338).

``fit_rlm_joint_growth`` fits a small measure set jointly and reports the
between-child cross-measure correlation matrix of the stable child levels — the
headline — plus per-measure fitted cells and common-window growth through the
shared historical summaries. LOO is not computed: the model carries one
likelihood node per measure, so a single pointwise PSIS-LOO is not defined for
it. Descriptive throughout; the cohort is observational.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_longitudinal_panel,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    growth_contrast_pushforward_rows,
    write_prior_pushforward,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
)
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    run_ppc,
    run_sampling_and_loo,
    write_run_metadata,
)


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
