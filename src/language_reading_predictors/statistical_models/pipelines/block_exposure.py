# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Block-exposure orchestration (LRP-RLI-BX).

``fit_block_exposure`` relates outcome level to cumulative teaching-block
exposure, splitting confounders by measurement timing so pre-exposure covariates
and wave-varying covariates enter on the right footing.
"""

from __future__ import annotations

import pandas as pd

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    block_exposure as _block_exposure,
    diagnostics as _diag,
    factories as _factories,
    reporting as _report,
)
from language_reading_predictors.statistical_models.adjustment import (
    effective_adjustment,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_association_forest,
    save_forest_plot,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    marginal_pushforward_rows,
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
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import PrimaryFitPlan


def fit_block_exposure(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Block-2 taught-vocabulary staggered block-active exposure fit (LRPBX, #228 item 5).

    Data-load like ``fit_level_factors`` (per-timepoint levels frame + the revised-DAG
    adjuster wiring); effect readout like ``fit_did`` (the focal ``delta`` + its
    items-scale AME). ``delta`` is an association (parallel-trends), so the factor
    summary flags no causal term.
    """
    require_spec(spec, "block_exposure", outcome=True)

    # Resolve the complete family contract before ``make_context`` can reset the
    # output directory and before preparation reads the RLI data (#394 pillar 4).
    plan = _block_exposure.resolve_block_exposure_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    sym = plan.outcome_symbol
    prepared = load_and_prepare(**plan.prepare_kwargs())
    # A constant ``_missing`` indicator may be removed by the loader. The effective
    # adjustment set, factory, diagnostic names and report table must all agree.
    adjust_for = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    ctx.prepared = prepared
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_block_exposure_model(
        prepared,
        **plan.factory_kwargs(effective_adjustment=adjust_for),
    )
    attach_built(ctx, built)
    diag_vars = plan.diagnostic_vars(effective_adjustment=adjust_for)
    coef_names = plan.coefficient_names(effective_adjustment=adjust_for)

    render_model_graph(ctx)

    # ``delta`` is the focal (association) effect — gets the prior-sensitivity +
    # forest evidence, exactly as the level-factor group term does.
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            ppc_var_names=(plan.observation_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, sym, node=plan.observation_node
            ),
            # The family's established post-trace order — overlay, forest, then
            # power scaling — now declared to the runner rather than performed
            # after it (#637 stage 4). Same figures, same order, one owner.
            psense_timing="after_trace",
            psense_vars=(plan.focal_term,),
            after_trace_audit=lambda c: (
                _diag.save_prior_posterior_plot(c, var_names=diag_vars),
                save_forest_plot(
                    c, [plan.focal_term], name="delta_forest.png",
                    title=(
                        "Block-active exposure effect "
                        "(forest, reference line at 0)"
                    ),
                ),
            )
            and None,
            extended_term=plan.focal_term,
            compute_loo=plan.compute_loo,
        ),
    )

    section_header("Factor summary")
    # No randomised contrast: block-active exposure is an association (parallel trends),
    # so no term is flagged causal.
    fs = _report.factor_summary(
        ctx.trace,
        coef_names,
        ci_prob=ctx.reporting.ci_prob,
        causal_terms=(),
    )
    save_table(ctx, "factor_summary", fs)
    save_association_forest(ctx, coef_names, ())
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({sym}) - {int(ctx.reporting.ci_prob * 100)}% CrI",
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    section_header("Block-2 exposure effect summary")
    from language_reading_predictors.statistical_models.measures import MEASURES

    bx_s = _report.block_exposure_summary(
        ctx.trace,
        ci_prob=ctx.reporting.ci_prob,
        n_trials=1 if plan.off_floor else MEASURES[sym].n_trials,
    )
    bx_df = pd.DataFrame([bx_s])
    save_table(ctx, "block_exposure_summary", bx_df)
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in bx_s.items()],
            title=(
                f"block-2 exposure effect ({sym}) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed); "
                "association (parallel-trends), positive = more taught-word learning"
            ),
            columns=["metric", "value"],
        )
    )
    # Estimand-scale prior check (#381), through the same transform
    # ``block_exposure_summary`` uses for ``delta_items_*``: a forward shift from
    # the un-exposed linear predictor ``eta_base``, not the ITT net-out core.
    write_prior_pushforward(
        ctx,
        marginal_pushforward_rows(
            ctx,
            [(plan.focal_term, "the block-active exposure association")],
            n_trials=1 if plan.off_floor else MEASURES[sym].n_trials,
            convention="forward",
            eta_name="eta_base",
        ),
    )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "block_exposure_summary": bx_s,
            "effective_adjustment": effective_adjustment(
                spec,
                built.prepared,
                adjust_for=adjust_for,
                requested_adjust_for=plan.adjust_for,
                ability_covariate=plan.ability_covariate,
            ),
        },
    )
    return finalize_report(ctx)
