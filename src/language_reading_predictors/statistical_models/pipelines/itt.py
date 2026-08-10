# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Available-case modified ITT orchestration (the LRP-RLI-ITT suite + companions).

``fit_itt`` is the family entry point: it resolves the typed run plan, prepares
the analysis rows, builds the model, runs the shared primary-fit sequence and
writes the family's scientific summaries. Heavily-floored outcomes divert to
``fit_itt_floor_rule``, which fits the binary off-the-floor headline estimand and
its flagged graded secondary from the same resolved plan.

The declaration side — settings, run-plan resolution, the analysis-set audit and
the PPC calibration audit — stays in :mod:`language_reading_predictors.statistical_models.itt`;
this module is orchestration only (#394 step 5).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
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
    save_forest_plot,
    save_proportion_at_zero_plot,
    save_rope_plot,
    write_arm_overlap,
    write_predicted_scores,
)
from language_reading_predictors.statistical_models.itt import (
    IttRunPlan,
    build_itt_from_plan,
    itt_diagnostic_variables,
    prepare_itt_data,
    resolve_itt_run_plan,
    write_itt_analysis_audit,
    write_itt_ppc_calibration,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    restrict_to_baseline_floored,
    restrict_to_off_floor,
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
from language_reading_predictors.statistical_models.subfits import run_subfit


def emit_itt_extras(
    ctx: StatisticalFitContext,
    built,
    *,
    n_trials: int,
    overlay_vars: list[str],
    term: str = "tau",
    varying_term: str = "tau_i",
    moderators: Sequence[tuple[str, np.ndarray]] | None = None,
    score_mean_link: str = "logit",
) -> None:
    """Area 1/4 extras for an ITT-style fit (issue #125).

    Writes ``prior_pushforward.csv`` (the estimand-scale prior check), the causal
    forest, the prior-vs-posterior overlay, and power-scaling sensitivity. Reads
    the persisted ``prior`` group (on ``ctx.prior_samples``) and the full trace,
    so call after ``save_trace``. ``n_trials=1`` gives the risk-difference scale
    for the binary off-floor model. ``moderators`` carries any treatment
    interactions so the prior is pushed through the same full-contribution AME.
    """
    with guard_optional(
        ctx, "prior pushforward",
        filename="prior_pushforward.csv", kind="table", verb="skipped",
    ):
        pf = _report.prior_pushforward(
            ctx.prior_samples,
            G=built.prepared.G,
            n_trials=n_trials,
            term=term,
            varying_term=varying_term,
            moderators=moderators,
            ci_prob=ctx.reporting.ci_prob,
            score_mean_link=score_mean_link,
        )
        save_table(ctx, "prior_pushforward", pd.DataFrame([pf]), required=False)
    save_forest_plot(ctx, [term])
    _diag.save_prior_posterior_plot(ctx, var_names=overlay_vars)
    _diag.run_psense(ctx, var_names=[term])


def itt_diag_vars(
    plan: IttRunPlan,
    adjust_for: tuple[str, ...],
    *,
    likelihood: str = "beta_binomial",
) -> list[str]:
    """Compatibility wrapper for the ITT family's diagnostic contract."""

    return itt_diagnostic_variables(plan, adjust_for, likelihood=likelihood)


def write_analysis_audit(
    ctx: StatisticalFitContext,
    prepared,
    outcomes: Sequence[str],
) -> None:
    """Compatibility wrapper for the ITT family's analysis-set audit."""

    write_itt_analysis_audit(
        ctx,
        prepared,
        outcomes,
        loader=load_and_prepare,
    )


def write_ppc_calibration(
    ctx: StatisticalFitContext,
    prepared,
    outcomes: Sequence[str],
    *,
    node: str = "y_post",
    filename: str = "posterior_predictive_calibration.csv",
) -> pd.DataFrame:
    """Compatibility wrapper for the ITT family's PPC calibration audit."""

    return write_itt_ppc_calibration(
        ctx,
        prepared,
        outcomes,
        node=node,
        filename=filename,
    )


def fit_itt(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    require_spec(spec, "itt", outcome=True)

    # Resolve and validate the family contract before the context resets an output
    # directory or the loader reads any data. From this point onward preparation,
    # factory arguments, diagnostics and the teaching recipe consume one plan.
    plan = resolve_itt_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    prepared, adjust_for = prepare_itt_data(plan, loader=load_and_prepare)
    ctx.prepared = prepared

    print_header(ctx)

    section_header("Build model")

    # Heavily-floored outcomes (P, N) take the post-hoc, data-adaptive
    # floor-rule branch in this reanalysis: a binary transition estimand as the
    # exploratory headline plus graded secondary checks (#119/#341).
    if plan.floor_rule:
        return fit_itt_floor_rule(ctx, spec, plan, prepared, adjust_for)

    built = build_itt_from_plan(
        plan,
        prepared,
        effective_adjustment=adjust_for,
        builder=_factories.build_itt_model,
    )
    attach_built(ctx, built)
    write_analysis_audit(ctx, built.prepared, (spec.outcome_symbol,))

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=itt_diag_vars(plan, adjust_for))

    run_ppc(ctx)
    write_ppc_calibration(ctx, built.prepared, (spec.outcome_symbol,))

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=itt_diag_vars(plan, adjust_for))
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol)
    _diag.run_extended_diagnostics(ctx, causal_term="tau")

    _diag.save_trace(ctx)

    # Area 1/4 extras that read the attached prior group or the full trace:
    # the prior pushforward to the items scale (estimand-scale prior check), the
    # tau forest, the prior-vs-posterior overlay, and power-scaling sensitivity.
    # Net out the full per-row treatment contribution: the age-varying ``tau_i``
    # is picked up automatically by the AME core; a linear tau moderator adds
    # ``gamma_tau_int·z_M`` (Part B). Latent today — no registered ITT spec sets
    # ``tau_moderator_symbol`` — but wired so a heterogeneity fit reports the
    # model-implied effect, not ``tau`` alone.
    tau_moderators = built.extras.get("tau_interaction_moderators", [])
    score_mean_link = plan.score_mean_link
    n_trials_own = int(built.prepared.n_trials[spec.outcome_symbol])
    emit_itt_extras(
        ctx, built, n_trials=n_trials_own,
        overlay_vars=itt_diag_vars(plan, adjust_for),
        moderators=tau_moderators,
        score_mean_link=score_mean_link,
    )

    # Treatment-effect summary on both scales.
    section_header("Available-case modified ITT estimate summary")
    tau_s = _report.tau_summary_itt(
        ctx.trace,
        ci_prob=ctx.reporting.ci_prob,
        # built.prepared is the (possibly row-subset) frame the model was fit
        # on, so G aligns with eta's obs_id axis (finding #2 in issue #78).
        G=built.prepared.G,
        moderators=tau_moderators,
        score_mean_link=score_mean_link,
    )
    tau_df = pd.DataFrame([tau_s])
    save_table(ctx, "tau_summary", tau_df)
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in tau_s.items()],
            title=f"tau ({spec.outcome_symbol}) - {int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)",
            columns=["metric", "value"],
        )
    )

    # ROPE-anchored continuous summary on the items scale
    # (notes/202606261304-evidence-strength-and-rope-reporting.md). Emitted for
    # graded outcomes with an agreed minimally-important difference (delta);
    # floored outcomes (P/N) take the floor-rule path and a probability-scale delta.
    from language_reading_predictors.statistical_models.measures import (
        ROPE_DELTA,
        rope_delta_grid,
    )

    delta_items = ROPE_DELTA.get(spec.outcome_symbol)
    if delta_items is not None:
        rope_s = _report.rope_summary(
            ctx.trace,
            G=built.prepared.G,
            n_trials=int(built.prepared.n_trials[spec.outcome_symbol]),
            delta=delta_items,
            ci_prob=ctx.reporting.ci_prob,
            moderators=tau_moderators,
            score_mean_link=score_mean_link,
        )
        rope_df = pd.DataFrame([rope_s])
        save_table(ctx, "rope_summary", rope_df)
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in rope_s.items()],
                title=f"ROPE summary ({spec.outcome_symbol}, delta={delta_items:g} items)",
                columns=["metric", "value"],
            )
        )
        save_rope_plot(
            ctx,
            spec.outcome_symbol,
            built.prepared.G,
            int(built.prepared.n_trials[spec.outcome_symbol]),
            delta_items,
            moderators=tau_moderators,
            split=True,
            score_mean_link=score_mean_link,
        )

        # δ-sensitivity sweep (issue #144): P(benefit ≥ δ) at the adopted δ and a
        # stricter 2·δ (word reading at δ = 1 and 2), for every graded outcome.
        sens_df = _report.rope_sensitivity(
            ctx.trace,
            G=built.prepared.G,
            n_trials=int(built.prepared.n_trials[spec.outcome_symbol]),
            deltas=rope_delta_grid(spec.outcome_symbol),
            moderators=tau_moderators,
            score_mean_link=score_mean_link,
        )
        save_table(ctx, "rope_sensitivity", sens_df)

    # Predicted-scores contrast panel + icon array (#316): what the model says
    # about actual test scores for a new child, treated vs untreated. No child
    # random intercept in the single-outcome ITT, so the prediction population
    # is the fitted sample's covariate profiles.
    write_predicted_scores(
        ctx,
        outcome_symbol=spec.outcome_symbol,
        G=built.prepared.G,
        n_trials=int(built.prepared.n_trials[spec.outcome_symbol]),
        term="tau",
        moderators=tau_moderators,
        delta=delta_items,
        population=(
            "new child; covariate profiles drawn from the fitted available-case "
            "modified ITT analysis rows"
        ),
        contrast_status=(
            "randomised assigned-arm contrast (available-case modified ITT estimate)"
        ),
        split=True,
        score_mean_link=score_mean_link,
    )

    # Intervention vs no-intervention overlap (two individual figures): the
    # arm-mean expected-outcome posterior and the new-child predictive outcome,
    # each drawn as smoothed overlapping density curves. Same reference rows and
    # contrast arithmetic as the predicted-scores panel above.
    write_arm_overlap(
        ctx,
        outcome_symbol=spec.outcome_symbol,
        G=built.prepared.G,
        n_trials=int(built.prepared.n_trials[spec.outcome_symbol]),
        term="tau",
        moderators=tau_moderators,
        population=(
            "new child; covariate profiles drawn from the fitted available-case "
            "modified ITT analysis rows"
        ),
        contrast_status=(
            "randomised assigned-arm contrast (available-case modified ITT estimate)"
        ),
        score_mean_link=score_mean_link,
    )

    # Tau-moderator (Part B / HTE) summary: the effect-modification coefficient
    # gamma_tau_int and the moderator main effect gamma_tau_mod, when a linear
    # tau moderator was fit. Returns {} (nothing written) for the standard
    # main-effect ITT models, so this is a no-op unless the moderator is present.
    tau_mod_s = _report.tau_moderation_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
    if tau_mod_s:
        tau_mod_df = pd.DataFrame([tau_mod_s])
        save_table(ctx, "tau_moderation_summary", tau_mod_df)

    missingness_metadata: dict[str, object] | None = None
    if plan.missingness_sensitivity_required_for_release:
        from language_reading_predictors.statistical_models.itt_missingness import (
            missingness_source_path,
            run_missingness_subfit,
        )

        archive_option = getattr(
            getattr(ctx, "run_options", None), "rli_randomised_archive", None
        )
        archive_path = missingness_source_path(archive_option)
        if archive_path is None:
            if os.environ.get("DSE_LRP_REUSE_TRACE"):
                raise FileNotFoundError(
                    "reuse-trace mode cannot rebuild the mandatory word-reading "
                    "missingness bundle without --rli-randomised-archive; the "
                    "previous complete output has not been replaced"
                )
            missingness_metadata = {
                "status": "not_run",
                "reason": "--rli-randomised-archive was not supplied",
            }
            rprint(
                "[yellow]Required word-reading missing-data sensitivity not run: "
                "supply --rli-randomised-archive. The primary fit is retained, "
                "but its scientific release will be withheld as incomplete.[/yellow]"
            )
        else:
            section_header("Full-randomised-cohort missing-data sensitivity")
            missingness_metadata = run_missingness_subfit(
                ctx,
                archive_path,
                plan=plan.missingness_plan,
                runner=run_subfit,
            )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "tau_summary": tau_s,
            "adjust_for": list(adjust_for),
            "itt_missingness_sensitivity": missingness_metadata,
        },
    )

    return finalize_report(ctx)


def fit_itt_floor_rule(
    ctx: StatisticalFitContext,
    spec: ModelSpec,
    plan: IttRunPlan,
    prepared,
    adjust_for: tuple[str, ...],
) -> StatisticalFitContext:
    """Floor-rule fit for heavily-floored outcomes P / N (#119).

    Fits two age-only models: the post-hoc exploratory binary off-floor
    transition (Bernoulli on ``post > 0`` among observed baseline zeros) and a
    flagged, detection-limited secondary graded Beta-Binomial. Writes
    ``tau_summary.csv`` (off-floor exploratory headline), the per-arm mover table,
    the proportion-at-zero PPC, and ``tau_summary_graded.csv``. The floor rule is
    post-hoc and data-adaptive in this reanalysis, although its mechanical gate is
    applied arm-blind.
    """
    from language_reading_predictors.statistical_models import floor as _floor

    own = plan.outcome_symbol

    # Data-adaptive gate: the outcome must actually qualify (>= 40% at zero at
    # t2). Applying it arm-blind avoids using treatment labels in this mechanical
    # classification, but does not make the post-hoc choice pre-specified.
    p0 = _floor.proportion_at_zero(prepared, own)
    if not _floor.is_floored(prepared, own):
        raise ValueError(
            f"floor_rule set for {own!r}, but only {p0:.0%} of its post-scores "
            f"are at zero at t2 (threshold {_floor.FLOOR_THRESHOLD:.0%}); the "
            "post-hoc floor gate is arm-blind - remove floor_rule or "
            "check the data."
        )

    # Make eligibility and its missingness visible before restricting. The
    # registered loader retains missing pre-scores for P/N, so without this table
    # those children would disappear silently when np.isclose(NaN, floor) is false.
    eligibility = _floor.baseline_floor_eligibility_by_arm(prepared, own)
    save_table(ctx, "baseline_floor_eligibility", eligibility)
    eligibility_sensitivity = _floor.baseline_floor_status_bounds(prepared, own)
    save_table(ctx, "floor_eligibility_sensitivity", eligibility_sensitivity)
    transition_missingness = _floor.binary_transition_missingness_bounds(prepared, own)
    save_table(ctx, "floor_transition_missingness_bounds", transition_missingness)
    print_table(
        ranked_dataframe_table(
            eligibility,
            title=f"Observed baseline-floor eligibility by arm ({own})",
            columns=[
                "arm",
                "n_loaded",
                "n_post_observed",
                "n_pre_observed",
                "n_pre_missing",
                "n_pre_floor",
                "n_pre_above_floor",
                "n_exploratory_eligible",
            ],
            rank_column=False,
            precision=0,
        )
    )

    # Restrict the exploratory headline to children with an *observed* baseline
    # score of zero. This targets Pr(post > 0 | observed pre == 0), rather than
    # prevalence over everyone. Baseline status is pre-randomisation, so the arm
    # contrast remains causally valid for this observed subgroup, subject to the
    # missingness assumptions stated in the report.
    at_risk = restrict_to_baseline_floored(prepared, own)
    n_eligible = int(eligibility["n_exploratory_eligible"].sum())
    # This equality relies on the single-outcome loader requiring this outcome's
    # post-score; revisit it before applying the floor rule to a joint outcome load.
    if at_risk.n_obs != n_eligible:
        raise RuntimeError(
            f"floor-rule eligibility count drift for {own!r}: restriction kept "
            f"{at_risk.n_obs}, eligibility table reports {n_eligible}"
        )
    # Guard: the subgroup ITT is only identified if the at-risk subset keeps both
    # arms and enough rows. If a future floored outcome had (say) all baseline-floored
    # children in one arm, tau would be unidentified and the headline posterior
    # degenerate — fail loudly rather than publish it (issue #267 review).
    _n_arms = int(np.unique(at_risk.G).size)
    if at_risk.n_obs < 10 or _n_arms < 2:
        raise ValueError(
            f"floor rule for {own!r}: the baseline-floored at-risk subset is "
            f"degenerate (n={at_risk.n_obs}, arms present={_n_arms}) — the subgroup "
            "contrast Pr(post>0 | observed pre==0) is not identified. Re-check "
            "the floor rule / "
            "data or fit a different estimand."
        )
    write_analysis_audit(ctx, at_risk, (own,))
    missing_by_arm = ", ".join(
        f"{row.arm}: {int(row.n_pre_missing)}"
        for row in eligibility.itertuples(index=False)
    )
    rprint(
        f"  Floor rule: {own} is {p0:.0%} floored at t2 "
        f"(>= {_floor.FLOOR_THRESHOLD:.0%}); the post-hoc exploratory headline is "
        f"Pr(off-floor at t2 | observed at floor at t1) on {at_risk.n_obs} "
        f"eligible children (of {prepared.n_obs} with an available t2 outcome). "
        f"Missing baseline eligibility by arm — {missing_by_arm}. A graded "
        "Beta-Binomial over all children and a graded contrast among off-floor "
        "children are flagged secondaries."
    )

    # ----- EXPLORATORY HEADLINE: binary transition among observed baseline zeros. -----
    section_header(
        "Build model (post-hoc headline: off-floor transition among observed "
        "baseline-floor children)"
    )
    built = build_itt_from_plan(
        plan,
        at_risk,
        effective_adjustment=adjust_for,
        likelihood="bernoulli_offfloor",
        builder=_factories.build_itt_model,
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol or "W")
    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(
        ctx,
        var_names=itt_diag_vars(plan, adjust_for, likelihood="bernoulli_offfloor"),
    )

    run_ppc(ctx, var_names=["y_offfloor"])
    write_ppc_calibration(
        ctx,
        built.prepared,
        (own,),
        node="y_offfloor",
    )

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(
        ctx,
        var_names=itt_diag_vars(plan, adjust_for, likelihood="bernoulli_offfloor"),
    )
    _diag.run_extended_diagnostics(ctx, causal_term="tau")
    _diag.save_trace(ctx)

    # Off-floor estimand is a risk difference (Pr off-floor), so the items scale is
    # n_trials = 1; no age-varying term in the floor-rule model.
    emit_itt_extras(
        ctx, built, n_trials=1, varying_term="",
        overlay_vars=itt_diag_vars(plan, adjust_for, likelihood="bernoulli_offfloor"),
    )

    section_header(
        "Off-floor available-case modified ITT estimate "
        "(post-hoc exploratory headline)"
    )
    off = _report.tau_summary_offfloor(
        ctx.trace, ci_prob=ctx.reporting.ci_prob, G=built.prepared.G
    )
    save_table(ctx, "tau_summary", pd.DataFrame([off]))
    print_table(
        metrics_table(
            [{"metric": k, "value": v} for k, v in off.items()],
            title=(
                f"off-floor transition tau ({own}, observed baseline-floor subgroup) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed); positive = "
                "intervention raises Pr(off-floor at t2 | observed at floor at t1)"
            ),
            columns=["metric", "value"],
        )
    )

    movers = _report.offfloor_mover_table(built.prepared, own)
    save_table(ctx, "offfloor_movers", movers)
    print_table(
        ranked_dataframe_table(
            movers,
            title=f"Off-floor movers by arm ({own})",
            columns=["arm", "n", "off_floor", "at_floor", "prop_off_floor"],
            rank_column=False,
            precision=3,
        )
    )

    # ROPE-anchored card on the off-floor RISK-DIFFERENCE scale (issue #125 Area 4;
    # #130 follow-up). delta is a probability (risk difference), n_trials = 1; the
    # 10 pp value is confirmed by the education lead (2026-07-01, issue #144).
    from language_reading_predictors.statistical_models.measures import (
        ROPE_DELTA_PROB,
        ROPE_DELTA_PROB_GRID,
    )

    delta_prob = ROPE_DELTA_PROB.get(own)
    if delta_prob is not None:
        rope_s = _report.rope_summary(
            ctx.trace,
            G=built.prepared.G,
            n_trials=1,
            delta=delta_prob,
            ci_prob=ctx.reporting.ci_prob,
            varying_term="",
        )
        rope_s["provisional_delta"] = False  # 10 pp signed off (#144, 2026-07-01)
        rope_s["delta_scale"] = "risk_difference"
        save_table(ctx, "rope_summary", pd.DataFrame([rope_s]))
        save_rope_plot(
            ctx, own, built.prepared.G, 1, delta_prob, varying_term="", split=True
        )

        # δ-sensitivity sweep on the risk-difference scale (issue #144): 10/15/20 pp.
        sens_df = _report.rope_sensitivity(
            ctx.trace,
            G=built.prepared.G,
            n_trials=1,
            deltas=ROPE_DELTA_PROB_GRID,
            varying_term="",
        )
        save_table(ctx, "rope_sensitivity", sens_df)

    # Paired off-floor probability display + risk-difference density + icon
    # array (#316): the floor rule's binary estimand drawn as two bars with
    # credible intervals rather than a score distribution.
    write_predicted_scores(
        ctx,
        outcome_symbol=own,
        G=built.prepared.G,
        n_trials=1,
        term="tau",
        varying_term="",
        likelihood="bernoulli",
        delta=delta_prob,
        population=(
            "new child; covariate profiles drawn from the baseline-floored "
            "at-risk analysis rows"
        ),
        contrast_status=(
            "randomised assigned-arm contrast (post-hoc subgroup available-case "
            "modified ITT estimate)"
        ),
        event_label="off the floor at t2",
        split=True,
    )

    # Intervention vs no-intervention overlap: only the arm-mean off-floor
    # probability posterior is meaningful here — a single binary outcome has no
    # smooth predictive density, so the predictive figure is not emitted.
    write_arm_overlap(
        ctx,
        outcome_symbol=own,
        G=built.prepared.G,
        n_trials=1,
        term="tau",
        varying_term="",
        likelihood="bernoulli",
        population=(
            "new child; covariate profiles drawn from the baseline-floored "
            "at-risk analysis rows"
        ),
        contrast_status=(
            "randomised assigned-arm contrast (post-hoc subgroup available-case "
            "modified ITT estimate)"
        ),
        event_label="off the floor at t2",
    )

    def _fit_secondary(built_x, *, label: str, trace_filename: str):
        # Gate every free variable: a well-mixed tau cannot rescue a non-mixing
        # kappa/alpha/age term because those nuisance parameters determine the
        # fitted mean and posterior predictive distribution (#341). Secondary
        # estimates are publication artefacts too, so the trace is persisted:
        # every convergence value and posterior stays auditable independently of
        # the exploratory off-floor fit.
        res = run_subfit(
            ctx,
            built_x,
            label=label,
            role="secondary",
            posterior_predictive=["y_post"],
            trace_filename=trace_filename,
        )
        summ = _report.tau_summary_itt(
            res.trace, ci_prob=ctx.reporting.ci_prob, G=built_x.prepared.G
        )
        summ.update(res.convergence)
        summ["trace_file"] = res.trace_file
        return res.trace, summ

    # ----- SECONDARY (flagged cross-check): graded Beta-Binomial over ALL children.
    # Not the exploratory headline — it mixes already-off-floor children into a mover analysis and
    # is detection-limited; read only beside the mover table, never alone (#119).
    section_header("Build model (SECONDARY cross-check: graded Beta-Binomial, all children)")
    built_g = build_itt_from_plan(
        plan,
        prepared,
        effective_adjustment=adjust_for,
        likelihood="beta_binomial",
        builder=_factories.build_itt_model,
    )
    trace_g, graded = _fit_secondary(
        built_g,
        label=f"{spec.model_id} graded cross-check",
        trace_filename="trace_graded_secondary.nc",
    )
    save_table(ctx, "tau_summary_graded", pd.DataFrame([graded]))

    # ----- SECONDARY (flagged): graded contrast AMONG the off-floor children.
    # The #119 hurdle branch reads the graded score conditional on having come off
    # the floor. Two caveats keep this honest: (1) conditioning on post>0 is
    # POST-randomisation (selection on outcome), so the contrast is NOT a clean
    # randomised effect; and (2) this fits a *plain* Beta-Binomial to the post>0
    # subset — an untruncated proxy for the conditional-above-floor mean
    # E[post | post>0], because a zero-truncated Beta-Binomial is not cleanly
    # supported here (its vectorised logcdf is undefined); the untruncated fit
    # slightly overstates the conditional mean (issue #267 review). Reported flagged,
    # never as an ITT estimand.
    hurdle = None
    off_floor_data = restrict_to_off_floor(prepared, own)
    if off_floor_data.n_obs >= 8 and int(np.unique(off_floor_data.G).size) == 2:
        section_header(
            "Build model (SECONDARY: graded contrast among off-floor children | post>0)"
        )
        built_h = build_itt_from_plan(
            plan,
            off_floor_data,
            effective_adjustment=adjust_for,
            likelihood="beta_binomial",
            builder=_factories.build_itt_model,
        )
        _trace_h, hurdle = _fit_secondary(
            built_h,
            label=f"{spec.model_id} off-floor-subset graded contrast",
            trace_filename="trace_hurdle_secondary.nc",
        )
        hurdle["n_off_floor"] = int(off_floor_data.n_obs)
        hurdle["untruncated_proxy"] = True
        save_table(ctx, "tau_summary_hurdle", pd.DataFrame([hurdle]))
    else:
        rprint(
            f"[yellow]hurdle conditional-above-floor secondary skipped for {own}: "
            f"only {off_floor_data.n_obs} off-floor rows (need >= 8, both arms).[/yellow]"
        )

    # Proportion-at-zero PPC on the graded model: assess whether the graded
    # Beta-Binomial reproduces the observed floor.
    ppc0 = _report.proportion_at_zero_ppc(built_g.prepared, own, trace_g)
    save_proportion_at_zero_plot(ctx, own, ppc0)
    save_table(
        ctx,
        "proportion_at_zero_ppc",
        pd.DataFrame([{k: v for k, v in ppc0.items() if k != "rep"}]),
        register=False,
    )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "floor_rule": {
                "outcome": own,
                "proportion_at_zero": p0,
                "threshold": _floor.FLOOR_THRESHOLD,
                "status": "post_hoc_data_adaptive",
                "arm_blind_gate": True,
                "exploratory_estimand": (
                    "Pr(off-floor at t2 | observed at floor at t1)"
                ),
                "at_risk_n": int(at_risk.n_obs),
                "total_n": int(prepared.n_obs),
                "baseline_missing_n": int(eligibility["n_pre_missing"].sum()),
                "eligibility_by_arm": eligibility.to_dict(orient="records"),
                "eligibility_status_sensitivity": eligibility_sensitivity.to_dict(
                    orient="records"
                ),
                "transition_missingness_bounds": transition_missingness.to_dict(
                    orient="records"
                ),
            },
            "tau_offfloor_exploratory": off,
            "tau_graded_secondary": graded,
            "tau_hurdle_secondary": hurdle,
            "proportion_at_zero_ppc": {k: v for k, v in ppc0.items() if k != "rep"},
            "adjust_for": list(adjust_for),
        },
    )

    return finalize_report(ctx)
