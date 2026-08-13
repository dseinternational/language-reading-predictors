# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Joint available-case modified ITT orchestration and contrast companions.

``fit_joint`` fits the available-case modified ITT suite's outcomes jointly — optionally with an LKJ
residual correlation — and writes the per-outcome tau summaries, the contrast
matrix and the taught-vs-not-taught generalisation contrasts. It shares the ITT
family's analysis-set and PPC-calibration audits, imported from
:mod:`language_reading_predictors.statistical_models.pipelines.itt` (#394 step 5).
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
    diagnostics as _diag,
    factories as _factories,
    joint as _joint,
    reporting as _report,
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
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_contrast_heatmap,
    save_forest_plot,
)
from language_reading_predictors.statistical_models.fitted_payloads import JointPayload
from language_reading_predictors.statistical_models.pipelines.itt import (
    write_analysis_audit,
    write_ppc_calibration,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    write_prior_pushforward,
)
from language_reading_predictors.statistical_models.publication import (
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


def fit_joint(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    require_spec(spec, "joint")

    # Resolve the complete family contract before ``make_context`` opens an output
    # transaction or the loader reads intervention data (#394 pillar 4).
    plan = _joint.resolve_joint_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    # A joint model may target an explicit outcome set (e.g. the taught-vs-not-
    # taught contrast in LRPITT15/15b); load exactly those so the complete-case mask is
    # not driven by the eight standardised outcomes. Defaults to ITT_OUTCOMES.
    joint_outcomes = plan.outcomes
    prepared = load_and_prepare(**plan.prepare_kwargs())
    ctx.prepared = prepared

    section_header("Build model")

    built = _factories.build_joint_model(
        prepared,
        **plan.factory_kwargs(),
    )
    attach_built(ctx, built)
    write_analysis_audit(ctx, built.prepared, joint_outcomes)

    render_model_graph(ctx)

    _joint_vars = plan.diagnostic_vars()

    def _plot_joint_prior(c: StatisticalFitContext) -> None:
        for index, symbol in enumerate(joint_outcomes):
            stem = (
                "prior_predictive_check"
                if index == 0
                else f"prior_predictive_check_{symbol.lower()}"
            )
            _diag.save_prior_predictive_plot(c, symbol, filename_stem=stem)

    def _run_joint_ppc(c: StatisticalFitContext) -> None:
        _diag.sample_posterior_predictive(c, var_names=["y_post"])
        for index, symbol in enumerate(joint_outcomes):
            stem = (
                "posterior_predictive_check"
                if index == 0
                else f"posterior_predictive_check_{symbol.lower()}"
            )
            _diag.save_joint_posterior_predictive_plot(
                c, symbol, filename_stem=stem
            )
        write_ppc_calibration(c, built.prepared, joint_outcomes)
        # Coverage is denominator-agnostic for flattened child × outcome cells;
        # the per-outcome overlays and calibration remain the figure/table views.
        with guard_optional(
            c, "ppc_summary.csv", filename="ppc_summary.csv", kind="table"
        ):
            coverage = _report.ppc_interval_coverage(c.trace, node="y_post")
            save_table(c, "ppc_summary", coverage, required=False)

    def _write_joint_loo_pit(c: StatisticalFitContext) -> None:
        # The generic LOO-PIT would pool tests with different denominators.
        for index, symbol in enumerate(joint_outcomes):
            stem = "loo_pit" if index == 0 else f"loo_pit_{symbol.lower()}"
            _diag.save_joint_loo_pit_plot(c, symbol, filename_stem=stem)

    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(_joint_vars),
            plot_prior_predictive=_plot_joint_prior,
            custom_posterior_predictive=_run_joint_ppc,
            psense_timing="family_tail",
            extended_term="tau",
            include_loo_pit=False,
            post_extended_audit=_write_joint_loo_pit,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=_joint_vars)
    # Power-scaling prior sensitivity (#381) on the causal term only, matching the
    # ITT family this shares an estimand with — ``tau`` is vector-valued here, so
    # psense expands it to one row per outcome and the report can say which of the
    # jointly-fitted effects lean on the prior.
    _diag.run_psense(ctx, var_names=["tau"])
    # The probability-scale AMEs in tau_summary.csv are the headline effects. This
    # forest is deliberately retained as an explicitly labelled secondary view of
    # the conditional-logit coefficients.
    save_forest_plot(
        ctx,
        ["tau"],
        title="Secondary conditional-logit coefficients (forest, reference line at 0)",
    )

    section_header("Treatment-effect summary")
    outcomes = list(ctx.trace.posterior["outcome"].values)
    tau_df = _report.tau_summary_joint(
        ctx.trace,
        outcomes,
        ci_prob=ctx.reporting.ci_prob,
        G=built.prepared.G,
    )
    save_table(ctx, "tau_summary", tau_df)
    print_table(
        ranked_dataframe_table(
            tau_df,
            title=(
                "Probability-scale AME by outcome - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
            ),
            columns=[
                "outcome",
                "ame_prob_median",
                "ame_prob_lo",
                "ame_prob_hi",
                "prob_ame_pos",
            ],
            rank_column=False,
        )
    )

    # Items-scale counterpart for the key-findings range-plus-count headline
    # (#320).  The joint tau table is deliberately on the common logit scale;
    # this separate counterfactual pushforward preserves comparability there
    # while giving each outcome its own readable item-scale marginal and ROPE
    # probabilities where a project-agreed minimally-important difference exists.
    from language_reading_predictors.statistical_models.measures import ROPE_DELTA

    joint_marginal = _report.joint_treatment_marginals(
        ctx.trace,
        outcomes=outcomes,
        G=built.prepared.G,
        n_trials=built.prepared.n_trials,
        deltas=ROPE_DELTA,
        ci_prob=ctx.reporting.ci_prob,
    )
    save_table(ctx, "joint_treatment_marginal", joint_marginal)

    # Estimand-scale prior check, one row per outcome (#381). A joint fit has one
    # tau and one item denominator per outcome, so the single-row ITT schema
    # cannot describe it; the rows run through ``_joint_ame_draws``, the same core
    # the posterior marginals above use.
    try:
        pf_rows = _report.joint_prior_pushforward(
            ctx.prior_samples,
            outcomes=outcomes,
            G=built.prepared.G,
            n_trials=built.prepared.n_trials,
            ci_prob=ctx.reporting.ci_prob,
        )
    except Exception as exc:  # noqa: BLE001 - absence must stay legible
        pf_rows = [
            _report.unavailable_pushforward(
                estimand="tau",
                estimand_label="the per-outcome treatment effects",
                role="causal",
                reason=str(exc),
            )
        ]
    write_prior_pushforward(ctx, pf_rows)

    contrast = _report.tau_contrast_matrix(
        ctx.trace, outcomes, G=built.prepared.G, scale="probability"
    )
    save_table(ctx, "tau_contrast_matrix", contrast, index=True)
    save_contrast_heatmap(ctx, contrast)

    logit_contrast = _report.tau_contrast_matrix(
        ctx.trace, outcomes, G=built.prepared.G, scale="logit"
    )
    save_table(ctx, "tau_contrast_matrix_logit", logit_contrast, index=True)

    payload = built.require_payload(JointPayload, family="joint")
    meta_extra: dict = {
        "loo_elpd": float(ctx.loo.elpd),
        "joint_structure": payload.joint_dependence,
        "loo_unit": payload.loo_unit,
        "outcomes": list(joint_outcomes),
    }

    # Two-outcome contrast (LRPITT15/15b/16). ``difference = (a, b)`` reports the
    # headline probability-scale AME[a] - AME[b] and retains tau[a] - tau[b] as a
    # secondary conditional-logit contrast.
    difference = plan.difference
    if difference is not None:
        pair = difference
        section_header("Treatment-effect difference")
        diff_s = _report.tau_difference_summary(
            ctx.trace,
            outcomes,
            pair,
            ci_prob=ctx.reporting.ci_prob,
            G=built.prepared.G,
            metadata=plan.difference_metadata(),
        )
        diff_df = pd.DataFrame([diff_s])
        save_table(ctx, "tau_difference", diff_df)
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in diff_s.items()],
                title=(
                    f"AME[{pair[0]}] - AME[{pair[1]}] (probability-scale headline; "
                    f"logit secondary) - {int(ctx.reporting.ci_prob * 100)}% CI "
                    "(equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["tau_difference"] = diff_s
        meta_extra["difference_metadata"] = plan.difference_metadata()

    write_run_metadata(ctx, extra=meta_extra)

    return finalize_report(ctx)
