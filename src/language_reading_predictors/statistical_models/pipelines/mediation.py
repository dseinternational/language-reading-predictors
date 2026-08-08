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
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    load_and_prepare_lagged_outcome,
    split_confounders_by_timing,
    split_covariates_by_wave,
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
