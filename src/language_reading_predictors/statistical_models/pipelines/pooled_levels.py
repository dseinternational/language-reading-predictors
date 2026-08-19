# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Wave-pooled level-association orchestration (``kind="pooled_levels"``).

``fit_pooled_levels`` fits one Beta-Binomial likelihood over every child-wave row,
regressing a bounded-count outcome on a same-wave skill exposure with a child
random intercept. It is the pooled counterpart of the per-wave ``concurrent``
family and the levels counterpart of ``mechanism``; see
``statistical_models/pooled_levels.py`` for why it is a family rather than a flag
on either neighbour.
"""

from __future__ import annotations

from rich import print as rprint

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    pooled_levels as _pooled,
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
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.preprocessing import load_and_prepare
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


def fit_pooled_levels(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Wave-pooled between-child level association for one exposure/outcome pair."""
    require_spec(spec, "pooled_levels", outcome=True)

    plan = _pooled.resolve_pooled_levels_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    ctx.prepared = prepared
    print_header(ctx)
    rprint(
        f"  Pooled levels: {prepared.n_obs} child-wave rows from "
        f"{prepared.n_children} children across waves "
        f"{', '.join('t' + str(w) for w in plan.waves)}."
    )
    # Complete-case accounting for the exposure (#553): ``require_observed`` drops
    # the child-wave rows whose covariate exposure (or declared adjuster) was
    # mean-imputed. The loader does not report that count, so it is recovered by
    # loading the same frame without the filter and differencing the rows — the
    # number is part of the analysis population and is recorded beside the fit.
    n_dropped_require_observed: int | None = None
    if plan.require_observed:
        unfiltered = load_and_prepare(
            **{**plan.prepare_kwargs(), "require_observed": ()}
        )
        n_dropped_require_observed = int(unfiltered.n_obs - prepared.n_obs)
        rprint(
            f"  [yellow]{n_dropped_require_observed} child-wave row(s) whose "
            f"{', '.join(plan.require_observed)} value was imputed were dropped "
            "(require_observed).[/yellow]"
        )
    if not plan.use_wave_intercepts:
        rprint(
            "  [yellow]No wave intercepts: beta_mech also carries the secular "
            "co-movement of the two measures across waves. Comparator only.[/yellow]"
        )

    section_header("Build model")
    built = _pooled.build_pooled_levels_model(prepared, **plan.factory_kwargs())
    payload = built.payload
    rprint(
        f"  Fitted rows: {payload.n_fitted_rows} from {payload.n_children} children."
    )
    if payload.n_dropped_incomplete:
        rprint(
            f"  [yellow]{payload.n_dropped_incomplete} child-wave row(s) were missing "
            "the outcome, the exposure or a same-wave skill adjuster and were "
            "dropped.[/yellow]"
        )
    if payload.exposure_kind == "raw_covariate" and payload.exposure_sd_raw is not None:
        rprint(
            f"  Covariate exposure {plan.mechanism_symbol}: +1 SD of the fitted "
            f"exposure is {payload.exposure_sd_raw:.2f} raw points (mean "
            f"{payload.exposure_mean_raw:.2f})."
        )
    attach_built(ctx, built)
    render_model_graph(ctx)

    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=plan.diagnostic_vars(tuple(prepared.covariates)),
            ppc_var_names=(plan.observation_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, plan.outcome_symbol, node=plan.observation_node
            ),
            extended_term=plan.focal_term,
            compute_loo=plan.compute_loo,
        ),
    )

    section_header("Pooled level association")
    summary = _pooled.pooled_levels_summary(
        ctx.trace,
        ci_prob=ctx.reporting.ci_prob,
        n_trials=MEASURES[plan.outcome_symbol].n_trials,
    )
    save_table(ctx, "pooled_levels_summary", summary)
    print_table(
        ranked_dataframe_table(
            summary,
            title=(
                f"Wave-pooled level association "
                f"({plan.mechanism_symbol} -> {plan.outcome_symbol}); "
                "every term is an adjusted association, none is causal"
            ),
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    # Record the adjustment set that was actually FITTED — every coefficient-bearing
    # adjuster with its source column and measurement wave, plus anything the loader
    # dropped as constant — so config.json never implies a term the posterior lacks
    # (the same audit record the mechanism and factor families write).
    fitted_adjust_for = tuple(
        c for c in plan.adjust_for if c in prepared.covariates
    )
    meta_extra: dict = {
        "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
        "n_child_wave_rows": int(built.payload.n_fitted_rows),
        "n_dropped_incomplete_rows": int(built.payload.n_dropped_incomplete),
        "use_wave_intercepts": plan.use_wave_intercepts,
        "exposure_kind": built.payload.exposure_kind,
        "skill_symbols": list(plan.skill_symbols),
        "effective_adjustment": _levels_effective_adjustment(
            spec, prepared, plan, fitted_adjust_for
        ),
    }
    if plan.require_observed:
        meta_extra["require_observed"] = list(plan.require_observed)
        meta_extra["n_dropped_require_observed_rows"] = n_dropped_require_observed
    if built.payload.exposure_kind == "raw_covariate":
        # The raw-units anchor of the per-SD slopes (the mechanism family's
        # ``mechanism_exposure_sd_raw`` convention), so a report can read
        # ``beta_between`` / ``beta_within`` back in score points.
        meta_extra["mechanism_is_covariate"] = True
        meta_extra["mechanism_exposure_sd_raw"] = built.payload.exposure_sd_raw
        meta_extra["mechanism_exposure_mean_raw"] = built.payload.exposure_mean_raw
    write_run_metadata(ctx, extra=meta_extra)
    return finalize_report(ctx)


def _levels_effective_adjustment(spec, prepared, plan, fitted_adjust_for):
    """The fitted adjustment record, in levels-model vocabulary.

    :func:`effective_adjustment` speaks the transition families' language — age at
    the period's ``pre`` row, per-row adjusters at its ``post`` row. In a levels model
    the row *is* the wave, so both are the same wave as the outcome and exposure;
    only the t1-broadcast ability baseline keeps a distinct wave.
    """
    record = effective_adjustment(
        spec,
        prepared,
        # Same-wave skill adjusters (#553) are bounded-count measures taken at the
        # row's wave — the ``measure`` kind, relabelled ``same_wave`` below.
        measure_confounders=(("G",) if plan.include_group else ())
        + ("A",)
        + tuple(plan.skill_symbols),
        adjust_for=fitted_adjust_for,
        requested_adjust_for=plan.adjust_for,
        ability_covariate=plan.ability_covariate,
    )
    for term in record["fitted"]:
        if term["wave"] in ("pre", "post"):
            term["wave"] = "same_wave"
    return record
