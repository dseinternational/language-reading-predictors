# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mediation orchestration (``kind="mediation"``, LRP59 / LRP64).

The three entry points share the g-formula NDE/NIE decomposition in
:mod:`language_reading_predictors.statistical_models.mediation`:
``fit_mediation`` runs the single-mediator split by counterfactual simulation,
``fit_mediation_period_stacked`` stacks the period transitions, and
``fit_mediation_multi`` decomposes two mediators. None of them computes ordinary
PSIS-LOO — the published quantity is a simulated contrast, not a pointwise
predictive density. The temporal-ordering sensitivity refits the outcome one wave
later so the mediator precedes it in time; that increment is not randomised, so it
is triangulation only (#84).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    mediation_settings as _settings,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.fitted_payloads import (
    MediationPayload,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    load_and_prepare_lagged_outcome,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    render_model_graph,
)
from language_reading_predictors.statistical_models.release import (
    MEDIATION_T3_TRACE_FILENAME,
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


def _raw_covariate_confounders(confounders: Iterable[str]) -> tuple[str, ...]:
    """The confounders that are raw covariates, needing ``covariates=`` loading.

    A mediation adjustment set mixes two kinds of confounder: bounded-count skill
    measures (E, R, ...) that arrive via ``prepared.pre_logit`` (they are in the
    default or typed outcome set), and revised-DAG raw covariates
    (hearing ``hs``/``hs_missing``, speech ``deapp_c``, phonological memory
    ``erbto`` + missing indicators; #246) that must be requested as ``covariates``.
    A symbol is a raw covariate exactly when it is not a bounded-count measure.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    return tuple(c for c in confounders if c not in MEASURES)


_T3_SENSITIVITY_TIME = 3  # post-RCT wave used for the temporal-ordering check


def _leg_contract(plan, built) -> dict:
    """Declared-versus-fitted leg contract for ``config.json`` (#585 finding 1).

    Records the common pre-exposure vector, the per-leg baseline terms the plan
    resolved, the complete-case rule those terms imply, and the coefficient names
    the built graph actually carries — so a term can no longer be declared
    without being fitted, or fitted without being declared, unnoticed.
    """

    def terms(items):
        return [
            {"symbol": t.symbol, "coefficient": t.coefficient, "form": t.form}
            for t in items
        ]

    cross = plan.mediator_cross_baselines
    mediator_terms = (
        {symbol: terms(items) for symbol, items in cross.items()}
        if isinstance(cross, dict)
        else terms(cross)
    )
    return {
        "common_baselines": list(plan.common_baselines),
        "pre_required": list(plan.pre_required),
        "mediator_cross_baselines": mediator_terms,
        "outcome_cross_baselines": terms(plan.outcome_cross_baselines),
        "fitted_coefficients": sorted(
            rv.name for rv in built.model.free_RVs if rv.ndim == 0
        ),
    }


def _fit_t3_sensitivity(
    ctx: StatisticalFitContext,
    spec: ModelSpec,
    *,
    plan: _settings.MediationRunPlan,
):
    """Temporal-ordering sensitivity fit for the mediation models (issue #84).

    Refits the *identical* mediation model but with the outcome measured at a
    later wave (t3) while the mediator stays at t2, so the mediator precedes the
    outcome in time. The t2 -> t3 increment is **not randomised** (both arms are
    treated after t2), so this is a triangulation point for the contemporaneous
    measurement caveat, not a cleaner causal estimate. Returns the g-formula
    decomposition DataFrame for the t3-outcome variant.
    """
    from language_reading_predictors.statistical_models import mediation as _med

    outcome_symbol = plan.outcome_symbol
    lag_kwargs = (
        {"outcomes": plan.outcomes} if plan.outcomes is not None else {}
    )
    prepared_t3 = load_and_prepare_lagged_outcome(
        outcome_symbol,
        outcome_time=_T3_SENSITIVITY_TIME,
        covariates=_raw_covariate_confounders(plan.effective_confounders),
        # Same complete-case rule as the primary fit, so the sensitivity is the
        # identical design on a later outcome rather than a different sample.
        pre_required=plan.pre_required,
        **lag_kwargs,
    )
    built_t3, med_t3 = _factories.build_mediation_model(
        prepared_t3,
        mediator_symbol=plan.mediator_symbol,
        outcome_symbol=outcome_symbol,
        confounder_symbols=plan.effective_confounders,
        mediator_kind=plan.mediator_kind,
        outcome_kind=plan.outcome_kind,
        route_symbols=plan.route_symbols,
        # The sensitivity must rebuild the SAME legs (#585): dropping the common
        # baseline terms here would silently make it a different specification.
        mediator_cross_baselines=plan.mediator_cross_baselines,
        outcome_cross_baselines=plan.outcome_cross_baselines,
        # ... and the same outcome score-mean link (#619): a sensitivity that
        # silently swapped the link would be a different specification, which is the
        # very thing the two lines above exist to prevent.
        score_mean_link=plan.score_mean_link,
    )
    # Gate this temporal-ordering sensitivity sub-fit (bypasses the primary gate).
    # ``convergence_scope="all"`` keeps the scan this fit has always used: every
    # variable ArviZ reports, deterministics included, which is stricter than the
    # free-RV scan the other sub-fits take.
    res_t3 = run_subfit(
        ctx,
        built_t3,
        label=f"{spec.model_id} t3 sensitivity",
        role="sensitivity",
        trace_filename=MEDIATION_T3_TRACE_FILENAME,
        convergence_scope="all",
    )
    conv = res_t3.convergence
    df_t3 = _med.decompose(
        res_t3.trace,
        med_t3,
        ci_prob=ctx.reporting.ci_prob,
        # Same estimand branch as the primary (#631 finding 9): an interventional
        # companion's t3 rows must carry IDE/IIE labels, not the natural NDE/NIE
        # default — the numbers coincide, the schema and interpretation do not.
        interventional=plan.estimand == "interventional",
        score_mean_link=plan.score_mean_link,
    )
    # Persist the verdict onto the published rows: this sub-fit bypasses the primary
    # gate, and the verdict was previously computed then discarded so the t3 table
    # shipped with no convergence flag (this review's finding B1). Flows through to
    # both mediation_summary_t3.csv and the mediation_t3_sensitivity metadata block.
    df_t3["converged"] = conv["converged"]
    df_t3["trace_file"] = res_t3.trace_file
    return df_t3


def _prepare_mediation_data(plan: _settings.MediationRunPlan):
    """Execute the single-mediator loader contract from a resolved plan."""
    if plan.entrypoint != "single":
        raise ValueError("prepare_mediation_data does not accept a period-stacked plan")
    kwargs = plan.prepare_kwargs()
    if plan.outcome_time is not None:
        outcome_symbol = kwargs.pop("outcome_symbol")
        prepared = load_and_prepare_lagged_outcome(outcome_symbol, **kwargs)
    else:
        prepared = load_and_prepare(**kwargs)
    confounders = tuple(
        symbol
        for symbol in plan.declared_confounders
        if symbol in prepared.covariates or symbol in prepared.pre_logit
    )
    return prepared, confounders


def prepare_mediation_data(spec: ModelSpec):
    """Load the exact rows and fitted confounder set for a mediation spec.

    Kept separate from sampling so reporting-only regenerators can reconstruct the
    mediation sample and its mediator standardiser without refitting the posterior.
    """
    require_spec(spec, "mediation")
    return _prepare_mediation_data(_settings.resolve_mediation_run_plan(spec))


def _simulation_record() -> dict[str, int]:
    """The g-formula's inner-simulation settings, for the fit's own metadata.

    What separates a reported decomposition from a reproducible one: the
    counterfactual cells are *drawn*, so the seed and the per-draw replicate
    count belong in the fit rather than only in the source defaults (#585
    section C). ``gate_derived_estimands`` bounds the resulting Monte-Carlo
    error; this records what produced it. All three fit entry points write it,
    and ``test_mediation`` pins these values to the ``decompose*`` defaults.
    """
    from language_reading_predictors.statistical_models import mediation as _med

    return {
        "seed": _med.G_FORMULA_SEED,
        "replicates_per_draw": _med.G_FORMULA_REPLICATES,
    }


def fit_mediation(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """ITT-phase mediation decomposition (LRP59): how much of G -> W flows via L."""
    require_spec(spec, "mediation")
    from language_reading_predictors.statistical_models import mediation as _med

    plan = _settings.resolve_mediation_run_plan(spec)
    if plan.entrypoint != "single":
        raise ValueError(
            f"{spec.model_id}: period-stacked settings require "
            "fit_mediation_period_stacked"
        )
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    prepared, confounders = _prepare_mediation_data(plan)
    if confounders != plan.effective_confounders:
        plan = plan.with_effective_confounders(confounders)
        # The ACTIVE plan drives the factory, summaries and this recipe
        # rewrite; config.json keeps the RESOLVER's plan so the #623
        # currency check compares resolution with resolution. The
        # loader's constant-column removals stay recorded in extra
        # (2026-08-26 batch).
        _report.write_model_recipe(ctx, plan=plan)
    ctx.prepared = prepared

    print_header(ctx)

    section_header("Build model")

    outcome_kind = plan.outcome_kind
    off_floor = outcome_kind == "bernoulli_offfloor"
    mediator_node, outcome_node = plan.observation_nodes
    built, med_data = _factories.build_mediation_model(
        prepared,
        **plan.factory_kwargs(),
    )
    attach_built(ctx, built)

    # Diagnose every scalar coefficient the model actually built (deterministics
    # and the observed mediator/outcome nodes are not free RVs), so the list
    # tracks the fitted confounder set instead of a hand-maintained constant.
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)

    render_model_graph(ctx)

    # The mediator likelihood is the FIRST observed RV, so name the outcome node
    # explicitly — else the plot overlays mediator draws on the outcome's counts.
    # Power-scaling prior sensitivity (#381): this family does not compute LOO,
    # so add the log groups explicitly, then power-scale the reported parameters.
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(coef_vars),
            ppc_var_names=(mediator_node, outcome_node),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, plan.outcome_symbol, node=outcome_node
            ),
            prepare_psense=lambda c: _diag.compute_log_likelihood_and_prior(
                c, strict=False
            ),
            compute_loo=plan.compute_loo,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=coef_vars)

    section_header("Mediation decomposition (g-formula)")
    _interventional = plan.estimand == "interventional"
    med_df = _med.decompose(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        interventional=_interventional,
        # The link the factory BUILT, so the g-formula accumulates every
        # counterfactual cell on the response scale the outcome likelihood used
        # (#619).
        score_mean_link=built.require_payload(
            MediationPayload, family="mediation"
        ).score_mean_link,
    )
    save_table(ctx, "mediation_summary", med_df)
    # Extend the convergence gate to the POST-PROCESSED headline effects (#585
    # finding 8): the all-free-RV gate never sees the g-formula draws.
    # Pass the exact branch the fit produced (#631 finding 10): the gate now
    # fails on a requested quantity the summary lacks, so a natural/interventional
    # union would always fail one branch's three names.
    _diag.gate_derived_estimands(
        ctx,
        med_df,
        quantities=(
            ("total", "IDE", "IIE") if _interventional else ("total", "NDE", "NIE")
        ),
    )
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

    # Mediator-coefficient TIPPING analysis (#230, renamed #585 finding 6): sweep a
    # one-directional bias off b_M and report the delta at which the indirect
    # effect's interval first includes 0. It is a coefficient-scale bias model for
    # the mediator->outcome slope, NOT an E-value: an E-value is a specific
    # risk-ratio sensitivity measure (VanderWeele & Ding 2017), and this sweep
    # neither introduces an unmeasured variable nor works on that scale.
    section_header("Mediator-coefficient tipping analysis")
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
            f"({sens_summary['tipping_frac_of_bM']:.0%} of the fitted b_M+b_GM) — a "
            "mediator->outcome slope bias that large would null the NIE."
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
    # The interventional companions are NO LONGER exempt (#585 finding 2): they are
    # the same fitted model under a different label, so exempting them left them
    # with strictly less evidence than the parent whose numbers they reproduce.
    med_df_t3 = None
    if plan.outcome_time is None:
        section_header("Temporal-ordering sensitivity (outcome at t3)")
        med_df_t3 = _fit_t3_sensitivity(
            ctx,
            spec,
            plan=plan,
        )
        save_table(ctx, "mediation_summary_t3", med_df_t3)
        print_table(
            ranked_dataframe_table(
                med_df_t3,
                title=(
                    "Temporal-ordering sensitivity "
                    f"(outcome {plan.outcome_symbol} at t3; NOT randomised)"
                ),
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
    # ``dropped_confounders`` compares the FULL declared set, not only the raw
    # covariates (#585 finding 3): a declared bounded-measure confounder that
    # never reached the graph used to be invisible here. The resolver now refuses
    # to leave one unloaded, so this should only ever list constant indicators.
    _extra_meta = {
        "adjustment": spec.adjustment,
        "effective_confounders": list(confounders),
        "dropped_confounders": [
            c for c in plan.declared_confounders if c not in confounders
        ],
        "estimand": "interventional" if _interventional else "natural",
        "outcome_kind": outcome_kind,
        "companion_of": plan.companion_of,
        # The FITTED row count: the factory drops rows with a missing mediator or
        # outcome post-score after preparation, so the pre-factory count
        # overstated n for every lagged fit (#585).
        "n_obs": built.prepared.n_obs,
        "n_obs_prepared": prepared.n_obs,
        "leg_contract": _leg_contract(plan, built),
        "mediation": _summary,
        "simulation": _simulation_record(),
    }
    if med_df_t3 is not None:
        _extra_meta["mediation_t3_sensitivity"] = {
            r["quantity"]: r for r in med_df_t3.to_dict("records")
        }
    if plan.outcome_time is not None:
        _extra_meta["outcome_time"] = plan.outcome_time
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
    per-leg child random intercepts).

    The exposure is ``T = (G == 1) | (phase >= 1)``, so only period 1 holds
    untreated rows. Since #585 (finding 5) the **period-1** restriction is the
    primary readout in ``mediation_summary.csv``; the all-period average, whose
    untreated counterfactual is extrapolation in every post-crossover period,
    goes to ``mediation_summary_all_periods.csv`` under an explicit label, and
    ``period_treatment_support.csv`` records the per-period arm counts behind the
    distinction. No t3 temporal-ordering
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

    plan = _settings.resolve_mediation_run_plan(spec)
    if plan.entrypoint != "period_stacked":
        raise ValueError(
            f"{spec.model_id}: single-mediator settings require fit_mediation"
        )
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    mediator_symbol = plan.mediator_symbol
    outcome_symbol = plan.outcome_symbol
    prepared = load_and_prepare(**plan.period_prepare_kwargs())
    ctx.prepared = prepared
    # Keep only confounders actually present (a constant ``_missing`` indicator
    # is dropped by the loader and gets no coefficient).
    confounders = tuple(
        symbol
        for symbol in plan.declared_confounders
        if symbol in prepared.covariates or symbol in prepared.pre_logit
    )
    if confounders != plan.effective_confounders:
        plan = plan.with_effective_confounders(confounders)
        # The ACTIVE plan drives the factory, summaries and this recipe
        # rewrite; config.json keeps the RESOLVER's plan so the #623
        # currency check compares resolution with resolution. The
        # loader's constant-column removals stay recorded in extra
        # (2026-08-26 batch).
        _report.write_model_recipe(ctx, plan=plan)

    print_header(ctx)

    section_header("Build model")
    built, med_data = _factories.build_period_stacked_mediation_model(
        prepared,
        **plan.period_factory_kwargs(),
    )
    attach_built(ctx, built)

    mediator_node = f"{mediator_symbol}_post"
    # Scalar coefficients from the model itself, plus the per-phase intercept
    # vectors (the convergence gate scans every free RV regardless).
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)
    diag_vars = [*coef_vars, "a_phase", "b_phase"]

    render_model_graph(ctx)

    # Power-scaling prior sensitivity (#381): this family does not compute LOO,
    # so add the log groups explicitly, then power-scale the reported parameters.
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            ppc_var_names=(mediator_node, "y_post"),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, outcome_symbol, node="y_post"
            ),
            prepare_psense=lambda c: _diag.compute_log_likelihood_and_prior(
                c, strict=False
            ),
            compute_loo=plan.compute_loo,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Empirical treatment support by period (#585 finding 5). ``T`` is
    # ``(G == 1) | (phase >= 1)``, so only period 1 has untreated rows: after the
    # wait-list crossover every child is on the programme. Persisted so a reader
    # can see which standardisation cells the counterfactual actually rests on.
    section_header("Treatment support by period")
    support_df = pd.DataFrame(
        [
            {
                "period": int(ph) + 1,
                "n_rows": int((med_data.phase_idx == ph).sum()),
                "n_treated": int(med_data.trt[med_data.phase_idx == ph].sum()),
                "n_untreated": int(
                    (1.0 - med_data.trt)[med_data.phase_idx == ph].sum()
                ),
            }
            for ph in sorted(set(med_data.phase_idx.tolist()))
        ]
    )
    support_df["both_arms_supported"] = (support_df["n_treated"] > 0) & (
        support_df["n_untreated"] > 0
    )
    save_table(ctx, "period_treatment_support", support_df)
    print_table(
        ranked_dataframe_table(
            support_df,
            title="Empirical treatment support by period",
            columns=["period", "n_rows", "n_treated", "n_untreated",
                     "both_arms_supported"],
            rank_column=False,
            precision=0,
        )
    )
    _supported = support_df.loc[support_df["both_arms_supported"], "period"].tolist()
    _unsupported = support_df.loc[~support_df["both_arms_supported"], "period"].tolist()
    if _unsupported:
        rprint(
            f"  Period(s) {_unsupported} have no untreated rows — their untreated "
            "counterfactual is model extrapolation, not standardisation."
        )

    # PRIMARY (#585 finding 5): the period-1 restriction. It is the only window in
    # which both arms are observed, so it is the only decomposition whose untreated
    # counterfactual is supported by data. The all-period average used to be the
    # headline; it toggles T = 0 on periods where every child is treated, so
    # adjustment cannot recover positivity there.
    section_header("Mediation decomposition (period 1; randomised window)")
    med_df = _med.decompose_period_stacked(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        row_mask=med_data.phase_idx == 0,
    )
    save_table(ctx, "mediation_summary", med_df)
    _diag.gate_derived_estimands(ctx, med_df, quantities=("total", "NDE", "NIE"))
    print_table(
        ranked_dataframe_table(
            med_df,
            title=(
                "Period-1 mediation, randomised window "
                f"(on-intervention; words out of {med_data.n_trials_W})"
            ),
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # Secondary, explicitly labelled: the all-period standardised contrast. Kept
    # for continuity and as a shape check, never as the headline.
    section_header("All-period contrast (model extrapolation)")
    med_df_all = _med.decompose_period_stacked(
        ctx.trace, med_data, ci_prob=ctx.reporting.ci_prob
    )
    save_table(ctx, "mediation_summary_all_periods", med_df_all)
    print_table(
        ranked_dataframe_table(
            med_df_all,
            title=(
                "All-period contrast — MODEL EXTRAPOLATION "
                f"(no untreated rows in period(s) {_unsupported or 'none'})"
            ),
            columns=["quantity", "words_mean", "words_lo", "words_hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    section_header("Mediator-coefficient tipping analysis")
    sens_sweep, sens_summary = _med.sensitivity_sweep(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        decompose_fn=_med.decompose_period_stacked,
        interaction_name="b_trtM",
        # Sweep the PRIMARY (period-1) estimand, not the extrapolated average.
        row_mask=med_data.phase_idx == 0,
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
            f"({sens_summary['tipping_frac_of_bM']:.0%} of the fitted b_M+b_trtM) — a "
            "mediator->outcome slope bias that large would null the NIE."
        )

    write_run_metadata(
        ctx,
        extra={
            "adjustment": spec.adjustment,
            "effective_confounders": list(confounders),
            "dropped_confounders": [
                c for c in plan.declared_confounders if c not in confounders
            ],
            "n_obs": built.prepared.n_obs,
            "n_obs_prepared": prepared.n_obs,
            "leg_contract": _leg_contract(plan, built),
            "exposure": "on_intervention (per-period; gain-factor ignorability)",
            "period_treatment_support": support_df.to_dict("records"),
            "supported_periods": _supported,
            "unsupported_periods": _unsupported,
            # ``mediation`` is now the PERIOD-1 headline (#585 finding 5); the
            # all-period average is recorded beside it as an extrapolation.
            "mediation": {r["quantity"]: r for r in med_df.to_dict("records")},
            "simulation": _simulation_record(),
            "mediation_all_periods": {
                r["quantity"]: r for r in med_df_all.to_dict("records")
            },
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

    plan = _settings.resolve_mediation_multi_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    mediators = plan.mediators
    calibration = plan.named_confounder_calibration
    calibration_symbol = calibration.symbol if calibration else None
    prepared = load_and_prepare(**plan.prepare_kwargs())
    # Drop any missing-indicator constant on the ITT-phase rows (see fit_mediation).
    confounders = tuple(
        symbol
        for symbol in plan.declared_confounders
        if symbol in prepared.covariates or symbol in prepared.pre_logit
    )
    if confounders != plan.effective_confounders:
        plan = plan.with_effective_confounders(confounders)
        # The ACTIVE plan drives the factory, summaries and this recipe
        # rewrite; config.json keeps the RESOLVER's plan so the #623
        # currency check compares resolution with resolution. The
        # loader's constant-column removals stay recorded in extra
        # (2026-08-26 batch).
        _report.write_model_recipe(ctx, plan=plan)
    ctx.prepared = prepared

    print_header(ctx)

    section_header("Build model")

    built, med_data = _factories.build_two_mediator_model(
        prepared,
        **plan.factory_kwargs(),
    )
    attach_built(ctx, built)

    # Diagnose every scalar coefficient the model actually built, so the list
    # tracks the fitted confounder set instead of a hand-maintained constant
    # (mirrors fit_mediation).
    coef_vars = sorted(rv.name for rv in built.model.free_RVs if rv.ndim == 0)

    render_model_graph(ctx)

    # The mediator likelihood is the FIRST observed RV, so name the outcome node
    # explicitly — else the plot overlays mediator draws on the outcome's counts.
    # Power-scaling prior sensitivity (#381): this family does not compute LOO,
    # so add the log groups explicitly, then power-scale the reported parameters.
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(coef_vars),
            ppc_var_names=plan.observation_nodes,
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, plan.outcome_symbol, node="y_post"
            ),
            prepare_psense=lambda c: _diag.compute_log_likelihood_and_prior(
                c, strict=False
            ),
            compute_loo=plan.compute_loo,
        ),
    )
    _diag.save_prior_posterior_plot(ctx, var_names=coef_vars)

    section_header("Two-mediator decomposition (g-formula)")
    med_df = _med.decompose_two_mediator(
        ctx.trace,
        med_data,
        hdi_prob=ctx.reporting.ci_prob,
        order=plan.order,
    )
    save_table(ctx, "mediation_summary", med_df)
    _diag.gate_derived_estimands(
        ctx,
        med_df,
        quantities=(
            "total",
            "NDE",
            "NIE_joint",
            f"NIE_{mediators[0]}",
            f"NIE_{mediators[1]}",
        ),
    )
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

    section_header("Per-leg mediator-coefficient tipping analysis")
    sens_sweep, sens_summary = _med.sensitivity_sweep_two_mediator(
        ctx.trace,
        med_data,
        ci_prob=ctx.reporting.ci_prob,
        order=plan.order,
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
    if calibration_symbol:
        section_header("Named-confounder calibration (intervention sessions)")
        calibration_df = _med.calibrate_session_confounding(
            built.prepared,
            med_data,
            sens_summary,
            session_symbol=calibration_symbol,
        )
        save_table(ctx, "mediation_is_calibration", calibration_df)
        for conclusion in calibration_df["conclusion"]:
            rprint(f"  {conclusion}")

    _summary = {r["quantity"]: r for r in med_df.to_dict("records")}
    # Requested vs actually-fitted confounders, recorded separately (#246 review, P2).
    write_run_metadata(
        ctx,
        extra={
            "adjustment": spec.adjustment,
            "effective_confounders": list(confounders),
            # Full declared set, not just the raw covariates (#585 finding 3).
            "dropped_confounders": [
                c for c in plan.declared_confounders if c not in confounders
            ],
            "n_obs": built.prepared.n_obs,
            "n_obs_prepared": prepared.n_obs,
            "leg_contract": _leg_contract(plan, built),
            "mediators": list(mediators),
            "n_trials_W": med_data.n_trials_W,
            "mediation": _summary,
            "simulation": _simulation_record(),
            "mediation_sensitivity": sens_summary.to_dict("records"),
            "named_confounder_calibration": (
                calibration_df.to_dict("records") if calibration_df is not None else None
            ),
        },
    )

    return finalize_report(ctx)
