# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Level-factors orchestration (LRP-RLI-LF, the companion levels view).

``fit_level_factors`` models the score at each timepoint with no own baseline,
carrying group×time and ability×time as per-timepoint coefficient vectors. Under
the default t1-referenced parameterisation (#552) the arm-by-time vector is a
pre-randomisation balance term ``arm_gap_t1`` plus per-wave changes
``d_grp_time[t]``; only the t2 change is a clean randomised effect (a
difference-in-differences of adjusted levels); later timepoints are
post-crossover and flagged as associations. It takes the revised-DAG exogenous
confounders but no measure-skill adjusters — conditioning a levels model on
another skill's contemporaneous level would condition on a post-treatment
mediator of the group×time effect (#247).
"""

from __future__ import annotations

import numpy as np
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
    reporting as _report,
)
from language_reading_predictors.statistical_models.adjustment import (
    effective_adjustment,
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
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import PrimaryFitPlan


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

    _lf_diag = plan.diag_vars(effective_adjustment=adjust_for)
    # For the shipped group-by-time LF models the flagged-causal term is the t2
    # element of the arm-by-time vector — ``d_grp_time[t2]`` under the t1-referenced
    # parameterisation (#552), ``b_grp_time[1]`` under the free comparator — which
    # reporting.level_t2_marginal_effect reads into the causal ROPE card, so the
    # vector must get the same prior-sensitivity + forest evidence as tau/beta_trt
    # rather than being skipped (issue #273). Names come from the plan (#389
    # finding 6): the run plan is the single source of truth. The forest also shows
    # the balance term beside the changes under the t1 reference, so the reader sees
    # the pre-randomisation gap the changes are measured from.
    _causal_lf = plan.causal_vector
    _forest_vars = [*plan.balance_terms, _causal_lf]
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(_lf_diag),
            ppc_var_names=(obs_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, spec.outcome_symbol, node=obs_node
            ),
            psense_timing="family_tail",
            extended_term=_causal_lf,
        ),
    )
    # Preserve the family's established post-trace order: overlay, forest, then
    # power scaling. The plan opts out of the standard pre-PPC sensitivity slot.
    _diag.save_prior_posterior_plot(ctx, var_names=_lf_diag)
    save_forest_plot(ctx, _forest_vars)
    # Power scaling covers the focal arm terms **and** the free nuisance scales
    # (#584 finding 6): the stored suite flags ``sigma_child`` in every fit and
    # ``kappa`` in most graded ones, and an audit that scans only the arm terms
    # establishes focal-term behaviour rather than prior/likelihood robustness. The
    # gate and the key-findings box still read the focal row only, so a nuisance
    # conflict is disclosed in the psense table without silently blocking release.
    _diag.run_psense(ctx, var_names=[*_forest_vars, *plan.nuisance_terms])

    section_header("Factor summary")
    # Only the t2 group contrast (plan.causal_terms) is the clean randomised effect;
    # the other timepoints are post-crossover (see the level-model caveat). Under the
    # t1 reference the balance term and the derived per-wave levels view carry their
    # own roles so they are never read as effects or as adjusted associations.
    causal = plan.causal_terms
    _lf_coefs = plan.coefficient_names(effective_adjustment=adjust_for)
    fs = _report.factor_summary(
        ctx.trace,
        _lf_coefs,
        ci_prob=ctx.reporting.ci_prob,
        causal_terms=causal,
        role_overrides=plan.factor_summary_roles(),
    )
    save_table(ctx, "factor_summary", fs)
    # The association forest shows the adjusted associations only: the focal vector
    # (causal element), the balance term and the derived levels view are excluded.
    save_association_forest(
        ctx,
        [
            c
            for c in _lf_coefs
            if c not in plan.balance_terms and c not in plan.levels_view_terms
        ],
        causal,
    )
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
            spec,
            built.prepared,
            adjust_for=adjust_for,
            requested_adjust_for=plan.adjust_for,
            ability_covariate=ability_covariate,
        ),
    }
    # ROPE-anchored continuous report for the one causal term — the t2 randomised
    # contrast (plan.focal_term; notes/202606261304-...). The level model enters group
    # as a per-timepoint vector and also carries a group×ability interaction, so the
    # t2 items-scale AME nets the focal contrast and the interaction out at the t2
    # rows (reporting.level_t2_marginal_effect) rather than reusing the gain core;
    # under the t1 reference the balance term stays in both arms and only the t2
    # change is added back (#552). Emitted
    # when the t2 contrast exists (group_by_time): graded outcomes with an agreed items
    # delta report on the items scale — since the ½-natural-maturation δ ratifications
    # (F/T adopted 2026-07-20, ratified 2026-08-19) every graded LF outcome has one in
    # ROPE_DELTA, so no graded fit is skipped today — and the floored outcomes P and N
    # report the off-floor risk difference (A4, 2026-07-13). An outcome absent from
    # both delta maps would still be skipped.
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
            contrast_term=plan.focal_vector,
            contrast_index=plan.focal_index,
            balance_term=plan.standardisation_balance_term,
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
        # prior check. Degrade to a warning and no CSV instead -- through
        # ``guard_optional``, so the skip and its cause land in
        # ``artifact_manifest.json`` rather than scrolling away in a warning the
        # manifest never records (#584 lower-severity 1).
        with guard_optional(
            ctx, "prior_pushforward", filename="prior_pushforward.csv", kind="table"
        ):
            pf = _report.level_prior_pushforward(
                ctx.trace,
                phase=built.prepared.phase,
                G=built.prepared.G,
                n_trials=n_marg,
                ability=ability,
                ci_prob=ctx.reporting.ci_prob,
                contrast_term=plan.focal_vector,
                contrast_index=plan.focal_index,
                balance_term=plan.standardisation_balance_term,
            )
            save_table(ctx, "prior_pushforward", pd.DataFrame([pf]), required=False)
            meta_extra["prior_pushforward"] = pf
        # The external rope_card still emits the retired 90% band; strip it so the
        # level family's rope_summary.csv matches the median + 50% + 89% convention
        # the other families publish (2026-07-17 standard; 2026-08-20 review,
        # finding 3).
        rope_s = _report.drop_retired_90_band(
            _report.rope_card(
                contrast_draws, items, delta=delta, ci_prob=ctx.reporting.ci_prob
            )
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
