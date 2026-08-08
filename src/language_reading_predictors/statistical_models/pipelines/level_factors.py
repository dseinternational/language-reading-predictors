# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Level-factors orchestration (LRP-RLI-LF, the companion levels view).

``fit_level_factors`` models the score at each timepoint with no own baseline,
carrying group×time and ability×time as per-timepoint coefficient vectors. Only
the t2 group contrast is a clean randomised effect; later timepoints are
post-crossover and flagged as associations. It takes the revised-DAG exogenous
confounders but no measure-skill adjusters — conditioning a levels model on
another skill's contemporaneous level would condition on a post-treatment
mediator of the group×time effect (#247).
"""

from __future__ import annotations

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
    save_rope_plot,
    write_child_fit,
    write_group_trajectory,
)
from language_reading_predictors.statistical_models.level_factors import (
    resolve_level_factors_run_plan,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
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


def fit_level_factors(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    require_spec(spec, "level_factors", outcome=True)
    # Resolve and validate the family contract before the context resets an output
    # directory or the loader reads any data (#389 finding 6). One plan then drives
    # preparation, factory arguments, the teaching recipe and config.json. The
    # covariate wave-split is resolved into the plan (#247 timing; review finding A1:
    # the language-proximal SP/RW confounders load at the pre-randomisation baseline
    # so the t2 randomised contrast is not conditioned on a treatment-affected
    # descendant; hearing is exogenous and stays contemporaneous). The level model
    # takes no measure-skill adjusters (post-treatment mediators).
    plan = resolve_level_factors_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    ability_covariate = plan.ability_covariate
    off_floor = plan.off_floor
    obs_node = plan.obs_node

    section_header("Prepare data")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    # Fail before model construction if the loaded panel cannot identify the
    # declared quantities — t2 missing a randomised arm, or a non-finite ability
    # value on a fitted row (#389 acceptance criterion; plan-owned so the guard
    # and the declared contract cannot drift apart).
    plan.validate_prepared(prepared)
    # Re-filter after loading — a constant ``_missing`` indicator is dropped by the
    # loader and must not be built or reported as adjusted-for.
    adjust_for = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    ctx.prepared = prepared
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_level_factors_model(
        prepared, **plan.factory_kwargs(effective_adjustment=adjust_for)
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, spec.outcome_symbol, node=obs_node)

    run_sampling_and_loo(ctx)

    _lf_diag = plan.diag_vars(effective_adjustment=adjust_for)
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=_lf_diag)

    run_ppc(ctx, var_names=[obs_node])

    section_header("Extended diagnostics")
    # For the shipped group-by-time LF models the flagged-causal term is the t2
    # element of the per-timepoint group vector, ``b_grp_time`` (``b_grp_time[1]``,
    # which reporting.level_t2_marginal_effect reads into the causal ROPE card), so
    # it must get the same prior-sensitivity + forest evidence as tau/beta_trt
    # rather than being skipped (issue #273). Names come from the plan (#389
    # finding 6): the run plan is the single source of truth.
    _causal_lf = plan.causal_vector
    _diag.write_diagnostics_summary(ctx, var_names=_lf_diag)
    _diag.run_extended_diagnostics(ctx, causal_term=_causal_lf)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_lf_diag)
    save_forest_plot(ctx, [_causal_lf])
    _diag.run_psense(ctx, var_names=[_causal_lf])

    section_header("Factor summary")
    # Only the t2 group contrast (b_grp_time[1]) is the clean randomised effect;
    # the other timepoints are post-crossover (see the level-model caveat).
    causal = plan.causal_terms
    _lf_coefs = plan.coefficient_names(effective_adjustment=adjust_for)
    fs = _report.factor_summary(
        ctx.trace, _lf_coefs, ci_prob=ctx.reporting.ci_prob, causal_terms=causal
    )
    save_table(ctx, "factor_summary", fs)
    save_association_forest(ctx, _lf_coefs, causal)
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({spec.outcome_symbol}) - {int(ctx.reporting.ci_prob * 100)}% CrI",
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    meta_extra = {
        "loo_elpd": float(ctx.loo.elpd),
        # Requested vs actually-fitted adjustment set, incl. dropped-constant
        # covariates (#247). The level model carries no skill baselines — only the
        # exogenous raw-covariate confounders (hs/deapp_c/erbto) at the split wave.
        "effective_adjustment": effective_adjustment(
            spec, built.prepared, adjust_for=adjust_for, ability_covariate=ability_covariate
        ),
    }
    # ROPE-anchored continuous report for the one causal term — the t2 randomised
    # contrast b_grp_time[1] (notes/202606261304-...). The level model enters group
    # as a per-timepoint vector and also carries a group×ability interaction, so the
    # t2 items-scale AME nets both group terms out at the t2 rows
    # (reporting.level_t2_marginal_effect) rather than reusing the gain core. Emitted
    # when the t2 contrast exists (group_by_time): graded outcomes with an agreed items
    # delta (ROPE_DELTA -> W/R/E/L/B) report on the items scale; the floored outcomes P
    # and N report the off-floor risk difference (A4, 2026-07-13) — previously they got
    # no probability-scale card at all; F/T (no agreed delta) are still skipped.
    from language_reading_predictors.statistical_models.measures import (
        ROPE_DELTA,
        ROPE_DELTA_PROB,
        ROPE_DELTA_PROB_GRID,
    )

    delta_items = ROPE_DELTA.get(spec.outcome_symbol)
    delta_prob = ROPE_DELTA_PROB.get(spec.outcome_symbol)
    _gbt = plan.group_by_time
    _graded_card = delta_items is not None and not off_floor and _gbt
    _offfloor_card = off_floor and delta_prob is not None and _gbt
    if _graded_card or _offfloor_card:
        ability = (
            built.prepared.covariates[ability_covariate]
            if ability_covariate is not None
            else None
        )
        contrast_draws, ame_prob = _report.level_t2_marginal_effect(
            ctx.trace,
            phase=built.prepared.phase,
            G=built.prepared.G,
            ability=ability,
        )
        if _graded_card:
            n_marg = int(built.prepared.n_trials[spec.outcome_symbol])
            delta = delta_items
            title = (
                f"ROPE summary (t2 contrast, {spec.outcome_symbol}, "
                f"delta={delta_items:g} items)"
            )
        else:
            # Off-floor (Bernoulli) t2 contrast: expit(eta) = Pr(off-floor), so the
            # probability-scale AME from level_t2_marginal_effect IS the off-floor risk
            # difference (n_trials = 1), matching the gain-factor off-floor path.
            n_marg = 1
            delta = delta_prob
            title = (
                f"ROPE summary (t2 off-floor risk difference, "
                f"{spec.outcome_symbol}, delta={delta_prob:g})"
            )
        items = ame_prob * n_marg
        # Estimand-scale prior pushforward for the t2 term (#389 finding 3): the
        # prior-predictive counterpart of this card, pushed through the same t2
        # net-out transform, so the level family emits the check the ITT / gain
        # families do (it previously lacked one). Same n_trials scaling as the AME.
        # Guarded like every other secondary artefact here: the ``prior`` group is
        # grafted on by the (itself guarded) ``_attach_prior_groups``, and reading it
        # off a trace that has none raises AttributeError. This runs after sampling,
        # so an unguarded failure would lose a completed reporting-tier fit over a
        # prior check. Degrade to a warning and no CSV instead.
        try:
            pf = _report.level_prior_pushforward(
                ctx.trace,
                phase=built.prepared.phase,
                G=built.prepared.G,
                n_trials=n_marg,
                ability=ability,
                ci_prob=ctx.reporting.ci_prob,
            )
        except Exception as exc:
            rprint(f"[yellow]prior_pushforward skipped: {exc}[/yellow]")
        else:
            save_table(ctx, "prior_pushforward", pd.DataFrame([pf]))
            meta_extra["prior_pushforward"] = pf
        rope_s = _report.rope_card(
            contrast_draws, items, delta=delta, ci_prob=ctx.reporting.ci_prob
        )
        if _offfloor_card:
            rope_s["provisional_delta"] = False  # 10 pp signed off (#144, 2026-07-01)
            rope_s["delta_scale"] = "risk_difference"
        rope_df = pd.DataFrame([rope_s])
        save_table(ctx, "rope_summary", rope_df)
        meta_extra["rope_summary"] = rope_s
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in rope_s.items()],
                title=title,
                columns=["metric", "value"],
            )
        )
        save_rope_plot(
            ctx, spec.outcome_symbol, None, n_marg, delta, items=items, split=True
        )
        if _offfloor_card:
            # δ-sensitivity sweep on the risk-difference grid (10/15/20 pp), mirroring
            # the gain-factor off-floor path (#144). Built from the same ``items``
            # (risk-difference) draws so it cannot drift from the headline card.
            sens_rows = []
            for d in ROPE_DELTA_PROB_GRID:
                d = float(d)
                p_benefit = float(np.mean(items >= d))
                sens_rows.append(
                    {
                        "delta_items": d,
                        "prob_benefit_ge_delta": p_benefit,
                        "prob_in_rope": float(np.mean(np.abs(items) <= d)),
                        "prob_harm_ge_delta": float(np.mean(items <= -d)),
                        "benefit_label": _report.evidence_label(p_benefit),
                    }
                )
            sens_df = pd.DataFrame(sens_rows)
            save_table(ctx, "rope_sensitivity", sens_df)

    # Data-space figures (#317): population per-arm score trajectory (the crossover
    # picture — only the t2 gap is randomised) and per-child fitted-vs-observed panels.
    write_group_trajectory(
        ctx,
        outcome_symbol=spec.outcome_symbol,
        arm=built.prepared.G,
        wave=built.prepared.phase,
        child_idx=built.prepared.child_idx,
        off_floor=off_floor,
        obs_node=obs_node,
    )
    write_child_fit(
        ctx,
        outcome_symbol=spec.outcome_symbol,
        wave=built.prepared.phase,
        child_idx=built.prepared.child_idx,
        off_floor=off_floor,
        obs_node=obs_node,
    )

    write_run_metadata(ctx, extra=meta_extra)
    return finalize_report(ctx)
