# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Historical group-by-wave growth orchestration (``kind="historical_growth"``, #165).

``fit_historical_growth`` is a descriptive natural-history growth model for the
Byrne reading-language-memory cohort — the first non-RLI dataset — run through the
shared pipeline so it uses the same sampler, convergence gate, output layout and
report conventions as the intervention models. It is **not** an
intervention-effect model: ``group`` carries no treatment semantics.
"""

from __future__ import annotations

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
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.historical_growth import (
    resolve_historical_growth_run_plan,
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

    # Resolve every family setting before ``make_context`` starts an output
    # transaction or the panel loader reads study data (#394 pillar 4).
    plan = resolve_historical_growth_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    study_id = plan.study_id
    measure = plan.measure
    dataset, measures = _datasets.resolve_dataset(study_id)
    panel = load_longitudinal_panel(
        dataset,
        [measures[measure]],
        **plan.prepare_kwargs(),
    )
    ctx.prepared = panel

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_historical_growth_model(
        panel,
        **plan.factory_kwargs(),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    diag_vars = plan.diagnostic_vars(ctx.model.named_vars)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, measure, node=plan.observation_node)

    run_sampling_and_loo(ctx, compute_loo=plan.compute_loo)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=[plan.observation_node])

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
            "waves": list(plan.waves),
            "extension_waves": list(plan.extension_waves),
            "groups": dict(
                zip(panel.group_codes, panel.group_labels, strict=True)
            ),
            "n_subjects": panel.n_subjects,
        },
    )

    return finalize_report(ctx)
