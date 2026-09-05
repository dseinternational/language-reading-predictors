# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Gain-factors orchestration (LRP-RLI-GF, the DAG-focused ANCOVA family).

``fit_gain_factors`` stacks every on-intervention and untreated period and
regresses a period's post-score on its own pre-score with a child random
intercept. The randomised on-intervention term is the *only* causal coefficient,
and its marginal effect is averaged over the period-1 transition alone; every
covariate is an explicit adjusted association. Heavily-floored outcomes take the
suite floor rule, with the binary off-floor-at-pre indicator as the always-on
baseline main effect (#391 finding 2).
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
    write_predicted_scores,
)
from language_reading_predictors.statistical_models.fitted_payloads import (
    GainFactorsPayload,
)
from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsRunPlan,
    resolve_gain_factors_run_plan,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    standardise,
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


def _gf_association_terms(
    plan: GainFactorsRunPlan,
    built: _factories.BuiltModel,
    *,
    adjust_for: tuple[str, ...],
    off_floor: bool,
) -> list[_report.AssociationTerm]:
    """Per-covariate ``AssociationTerm`` list for the gain items-scale marginals (#310).

    Reconstructs — from the *fitted* subset ``built.prepared`` — the exact standardised
    term vectors ``build_gain_factors_model`` used, so each covariate's ``+1 SD``
    perturbation is pushed through :func:`reporting.association_marginals` on the same
    scale the model was built on. The own baseline and skill baselines enter the linear
    predictor on the **raw logit** scale (their ``main_scale`` is that logit's SD) while
    their interactions use the standardised vector; age and cognitive ability are
    standardised throughout (``main_scale = 1``). Raw-covariate adjusters enter
    standardised with no interactions; their ``_missing`` companions are nuisance 0/1
    indicators (a ``+1 SD`` shift on them is not an interpretable association) and are
    skipped. On the off-floor (Bernoulli) path the own baseline is the binary
    off-floor-at-pre indicator (``gamma_own_offfloor``), matching the factory: its
    perturbation is the at-floor -> off-floor switch, not a ``+1 SD`` shift (#391
    finding 2 decision).
    """
    from scipy.special import expit as _expit
    from scipy.special import logit as _logit

    AT = _report.AssociationTerm
    bp = built.prepared
    own = plan.outcome_symbol
    skill_symbols = plan.skill_symbols
    ability_covariate = plan.ability_covariate
    treated_only = plan.treated_only

    # Standardised term vectors + main-effect scales, matching the factory on kept rows.
    term_vecs: dict[str, np.ndarray] = {"age": np.asarray(bp.A_std, dtype=float)}
    scales: dict[str, float] = {"age": 1.0}
    if ability_covariate is not None:
        z_ab, _ = standardise(bp.covariates[ability_covariate])
        term_vecs["ability"] = z_ab
        scales["ability"] = 1.0
    if off_floor:
        # Mirror the factory: "own" on the off-floor path is the binary
        # off-floor-at-pre indicator (raw 0/1), used for both the main effect and
        # any interaction naming it (#391 finding 2 decision). A "+1" perturbation
        # is the at-floor -> off-floor switch, not a +1 SD shift.
        term_vecs["own"] = (np.asarray(bp.pre_counts[own], dtype=float) > 0).astype(float)
        scales["own"] = 1.0
    else:
        z_own, s_own = standardise(bp.pre_logit[own])
        term_vecs["own"] = z_own
        scales["own"] = s_own.sd
    for s in skill_symbols:
        z_s, sc = standardise(bp.pre_logit[s])
        term_vecs[s] = z_s
        scales[s] = sc.sd
    # The treatment indicator: a covariate marginal holds it fixed, but a ``trt ×
    # covariate`` interaction still moves with the covariate, so it must be available as
    # a partner. Omitted under treated_only (then constant, and the factory drops it).
    if not treated_only:
        term_vecs["trt"] = ((bp.G == 1) | (bp.phase >= 1)).astype(float)
    active_interactions = plan.active_interactions

    def _ints_for(key: str) -> tuple[tuple[str, np.ndarray], ...]:
        out: list[tuple[str, np.ndarray]] = []
        for pair in active_interactions:
            if key not in pair:
                continue
            other = pair[0] if pair[1] == key else pair[1]
            if other not in term_vecs:  # partner unavailable (e.g. trt under treated_only)
                continue
            out.append(
                (f"gamma_int_{pair[0]}_{pair[1]}", np.asarray(term_vecs[other], dtype=float))
            )
        return tuple(out)

    def _sd_items(sd_logit: float, p: float, n: int) -> float:
        # Items equivalent of +1 SD of a bounded-count covariate, at its mean
        # proportion. The fitted baselines are Haldane logits,
        # log((y + 0.5) / (n - y + 0.5)), whose implied proportion is
        # (y + 0.5) / (n + 1) — so the exact inverse maps a proportion change onto
        # items through n + 1, not n (#575 finding 3).
        return float((n + 1) * (_expit(_logit(p) + sd_logit) - p))

    def _k_for(n: int) -> int:
        # Per-measure items increment (#575 finding 3), the concurrent family's
        # convention: a fixed +5 was a third of the 6-item nonword scale and half
        # the 10-item blending one.
        return max(1, round(n / 10))

    terms: list[_report.AssociationTerm] = []
    if not off_floor:
        p_own = float(np.mean(_expit(bp.pre_logit[own])))
        n_own = int(bp.n_trials[own])
        terms.append(
            AT("own", "gamma_own", scales["own"], _ints_for("own"),
               n_items=n_own, mean_prop=p_own,
               sd_items=_sd_items(scales["own"], p_own, n_own),
               k_items=_k_for(n_own))
        )
    else:
        # The binary off-floor-at-pre indicator association (#391 finding 2
        # decision): the "+1" perturbation is the at-floor -> off-floor switch.
        # Passing the observed indicator as toggle_vector makes the marginal use
        # the net-out-and-toggle idiom rather than a forward shift, so rows
        # already off the floor at pre contrast the actual 0 -> 1 switch instead
        # of an out-of-support 1 -> 2 move (code review 2026-08-20, finding 2).
        terms.append(
            AT("own", "gamma_own_offfloor", 1.0, _ints_for("own"),
               perturbation_label="off-floor at pre (0 to 1)",
               toggle_vector=term_vecs["own"])
        )
    terms.append(AT("age", "gamma_A", 1.0, _ints_for("age")))
    if ability_covariate is not None:
        terms.append(AT("ability", "gamma_ability", 1.0, _ints_for("ability")))
    for s in skill_symbols:
        p_s = float(np.mean(_expit(bp.pre_logit[s])))
        n_s = int(bp.n_trials[s])
        terms.append(
            AT(s, f"gamma_{s}", scales[s], _ints_for(s),
               n_items=n_s, mean_prop=p_s,
               sd_items=_sd_items(scales[s], p_s, n_s),
               k_items=_k_for(n_s))
        )
    for c in adjust_for:
        if c.endswith("_missing"):
            continue
        # The design column is the loader-standardised vector, so binariness must
        # be judged on the RAW support recovered through the carried scaler: the
        # standardised values of a 0/1 covariate are never literally {0, 1}, which
        # left the #575-finding-3 toggle branch unreachable and published hearing
        # as an uninterpretable "+1 SD" forward shift (#631 finding 4).
        values = np.asarray(bp.covariates[c], dtype=float)
        raw = np.asarray(bp.covariate_scalers[c].inverse(values), dtype=float)
        if np.all(np.isclose(raw, 0.0) | np.isclose(raw, 1.0)):
            # A binary status covariate (hearing: 1 = impaired, 0 = clear) has no
            # "+1 SD" — a continuous forward shift leaves its support (#575
            # finding 3). The net-out-and-toggle idiom contrasts the actual
            # 0 -> 1 switch instead, exactly as the off-floor own indicator does.
            # eta carries gamma_c * (x - mean) / sd, so a raw 0 -> 1 switch shifts
            # eta by gamma_c / sd and the toggle vector must be the raw indicator.
            terms.append(
                AT(c, f"gamma_{c}", 1.0 / float(bp.covariate_scalers[c].sd), (),
                   perturbation_label=f"{c} toggled 0 to 1",
                   toggle_vector=np.isclose(raw, 1.0).astype(float))
            )
        else:
            sd_c = float(np.std(values, ddof=1))
            terms.append(AT(c, f"gamma_{c}", sd_c, ()))
    return terms


def fit_gain_factors(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    require_spec(spec, "gain_factors", outcome=True)

    # Resolve and validate the family contract before the context resets an output
    # directory or the loader reads any data (#391 finding 6). One plan then drives
    # preparation, factory arguments, the teaching recipe and config.json. The
    # covariate wave-split (#247 timing: language-proximal SP/RW confounders at the
    # pre-randomisation baseline, hearing contemporaneous) is resolved into the plan.
    plan = resolve_gain_factors_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    skill_symbols = plan.skill_symbols
    ability_covariate = plan.ability_covariate
    treated_only = plan.treated_only
    off_floor = plan.off_floor
    # Explicitly associational moderation variant (#391 finding 3 decision): the
    # netted treatment marginal is still computed (the #391 finding 1 netting is
    # exactly what these variants exist to exercise) but no term — including
    # beta_trt — is labelled causal; the randomised headline lives in the
    # interaction-free primary.
    moderation_variant = plan.moderation_variant
    obs_node = plan.obs_node

    section_header("Prepare data")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    # Re-filter after loading — a constant ``_missing`` indicator is dropped by the
    # loader and must not be built or reported as adjusted-for. This is the
    # loader-frame filter only; the factory re-filters on the FINAL analysis mask.
    loader_adjust_for = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    ctx.prepared = prepared
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_gain_factors_model(
        prepared, **plan.factory_kwargs(effective_adjustment=loader_adjust_for)
    )
    attach_built(ctx, built)
    payload = built.require_payload(GainFactorsPayload, family="gain_factors")
    # The score-mean link the factory BUILT, not the one the module declared. Every
    # natural-scale summary below maps through it, so a floor-link posterior cannot
    # publish ordinary-link items (#596). Read at function scope because the
    # association marginals run for the treated-only companions too, which never
    # enter the randomised-contrast block.
    link = payload.score_mean_link
    # The adjusters the built model ACTUALLY carries (#575 finding 1): the
    # factory's focal-outcome / treated-only masks can make a loader-varying
    # indicator constant (gf-005/105/205's erbto_missing), and the factory now
    # drops such intercept aliases and records them. Everything downstream —
    # diagnostic vars, coefficient names, the effective-adjustment record, the
    # association marginals — must describe that model, not the loader frame.
    adjust_for = payload.effective_adjust_for

    render_model_graph(ctx)

    # The diagnostic FOCAL term, not a causal designation: for a moderation
    # variant beta_trt is still the term whose mixing, ESS evolution, forest and
    # prior sensitivity a reader needs, but it is presented as a model-dependent
    # association everywhere (factor_summary role, priors-table override, the
    # results partial and the key-findings box all branch on moderation_variant)
    # — the plot titles here are deliberately neutral ("Rank plot", "Effect
    # posterior"), so focusing them on beta_trt asserts nothing causal.
    _focal_gf = None if treated_only else "beta_trt"
    _gf_diag = plan.diagnostic_vars(effective_adjustment=adjust_for)
    _gf_coef_names = plan.coefficient_names(effective_adjustment=adjust_for)
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(_gf_diag),
            ppc_var_names=(obs_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, spec.outcome_symbol, node=obs_node
            ),
            # The family's established post-trace order — overlay, optional
            # forest, then optional power scaling — declared to the runner
            # (#637 stage 4). A treated-only fit has no focal term, so it
            # declares ``skip`` rather than reaching the slot and doing nothing.
            psense_timing="after_trace" if _focal_gf is not None else "skip",
            psense_vars=(_focal_gf,) if _focal_gf is not None else None,
            after_trace_audit=lambda c: (
                _diag.save_prior_posterior_plot(c, var_names=_gf_diag),
                save_forest_plot(c, [_focal_gf]) if _focal_gf is not None else None,
            )
            and None,
            extended_term=_focal_gf,
        ),
    )

    section_header("Factor summary")
    # A moderation variant's beta_trt is NOT flagged causal: its interaction-aware
    # marginal is model-dependent (partly informed by post-crossover data), so every
    # row — the treatment term included — reads as an adjusted association there.
    gf_causal_terms: tuple[str, ...] = (
        () if moderation_variant else ("beta_trt",)
    )
    fs = _report.factor_summary(
        ctx.trace, _gf_coef_names, ci_prob=ctx.reporting.ci_prob,
        causal_terms=gf_causal_terms,
    )
    save_table(ctx, "factor_summary", fs)
    save_association_forest(ctx, _gf_coef_names, gf_causal_terms)
    print_table(
        ranked_dataframe_table(
            fs,
            title=f"Factor summary ({spec.outcome_symbol}) - {int(ctx.reporting.ci_prob * 100)}% CrI",
            columns=["term", "role", "median", "lo", "hi", "prob_positive"],
            rank_column=False,
            precision=3,
        )
    )

    # Realised per-period, per-arm fitted-row support (#575 finding 5): the
    # machine-readable record behind the analysis-population prose, and the
    # evidence that period 1 retained both randomised arms after the final mask
    # (the factory refuses to build a causal fit otherwise).
    support_df = pd.DataFrame(
        payload.period_arm_support,
        columns=["period", "arm", "n_rows", "n_children"],
    )
    save_table(ctx, "analysis_support", support_df)

    meta_extra = {
        "loo_elpd": float(ctx.loo.elpd),
        "loo_unit": plan.loo_unit,
        "loo_note": plan.loo_note,
        "treated_only": treated_only,
        # Requested vs actually-fitted adjustment set, incl. dropped-constant
        # covariates (#247 / #258 review P1). Skills enter at the pre baseline; the
        # raw-covariate confounders (hs/deapp_c/erbto) at the split wave.
        "effective_adjustment": effective_adjustment(
            spec,
            built.prepared,
            adjust_for=adjust_for,
            requested_adjust_for=plan.adjust_for,
            ability_covariate=ability_covariate,
            baseline_symbol=spec.outcome_symbol,
            skill_baselines=skill_symbols,
            descriptive_skills=plan.descriptive_skills,
        ),
        # Adjusters removed by the factory's final-mask re-filter (#575 finding
        # 1) — distinct from the loader-frame constants already inside
        # dropped_constant, and empty for most fits.
        "post_mask_dropped_adjusters": list(payload.post_mask_dropped_adjusters),
        "analysis_support": [
            {
                "period": int(period),
                "arm": arm,
                "n_rows": int(n_rows),
                "n_children": int(n_children),
            }
            for period, arm, n_rows, n_children in payload.period_arm_support
        ],
    }
    # Items-scale marginal effect of the treatment term. Skipped when
    # treated_only (the on-intervention indicator is then constant and beta_trt
    # is absent).
    if not treated_only:
        trt = ((built.prepared.G == 1) | (built.prepared.phase >= 1)).astype(float)
        # The marginal treatment effect is averaged over the **period-1** rows only
        # (#247 P2): period 1 is the genuinely randomised, all-untreated-baseline
        # transition, so its switch-on-vs-off contrast is the available-case
        # modified ITT estimate. The post-crossover transitions (phase >= 1) carry no untreated
        # observations and baselines that may already be treatment-affected, so
        # pooling them yields a model-based transported contrast, not that estimate.
        # The logit-scale beta_trt posterior itself is unchanged; only its
        # probability/items-scale marginalisation is restricted.
        p1_mask = built.prepared.phase == 0
        # Net out the *full* per-row treatment contribution — ``beta_trt`` plus every
        # fitted treatment interaction (``gamma_int_trt_*``) — so the marginal effect
        # reflects the modelled heterogeneity, not ``beta_trt`` alone. The factory
        # exposes the exact standardised moderator vectors it used.
        trt_moderators = payload.trt_interaction_moderators
        # Off-floor models are Bernoulli on Pr(post > 0); the "items" scale then
        # collapses to the off-floor risk difference (n_trials = 1).
        n_marg = 1 if off_floor else built.prepared.n_trials[spec.outcome_symbol]
        tme = _report.treatment_marginal_effect(
            ctx.trace,
            trt=trt,
            n_trials=n_marg,
            moderators=trt_moderators,
            ci_prob=ctx.reporting.ci_prob,
            row_mask=p1_mask,
            score_mean_link=link,
        )
        save_table(ctx, "treatment_marginal", pd.DataFrame([tme]))
        meta_extra["treatment_marginal"] = tme
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in tme.items()],
                title="Treatment items-scale marginal effect",
                columns=["metric", "value"],
            )
        )

        # Prior pushforward on the same scale (estimand-scale prior check, #125).
        with guard_optional(
            ctx, "prior pushforward", filename="prior_pushforward.csv", kind="table"
        ):
            pf = _report.prior_pushforward(
                ctx.prior_samples, G=trt, n_trials=n_marg,
                term="beta_trt", varying_term="", moderators=trt_moderators,
                ci_prob=ctx.reporting.ci_prob, row_mask=p1_mask,
                score_mean_link=link,
            )
            save_table(ctx, "prior_pushforward", pd.DataFrame([pf]), required=False)

        # ROPE-anchored continuous report for the one causal term (beta_trt),
        # mirroring fit_itt (notes/202606261304-evidence-strength-and-rope-
        # reporting.md): separates direction (pd) from a *meaningful* benefit
        # (P(items >= delta)). Graded outcomes with an agreed items-scale delta
        # (ROPE_DELTA -> W/R/E/L/B) use the items scale; the floored outcome P (off-
        # floor) uses the provisional risk-difference delta (ROPE_DELTA_PROB, #130
        # follow-up); F/T have no agreed delta and are skipped.
        from language_reading_predictors.statistical_models.measures import (
            ROPE_DELTA,
            ROPE_DELTA_PROB,
            ROPE_DELTA_PROB_GRID,
        )

        delta_items = ROPE_DELTA.get(spec.outcome_symbol)
        delta_prob = ROPE_DELTA_PROB.get(spec.outcome_symbol)
        if delta_items is not None and not off_floor:
            rope_s = _report.rope_summary(
                ctx.trace,
                G=trt,
                n_trials=n_marg,
                delta=delta_items,
                ci_prob=ctx.reporting.ci_prob,
                term="beta_trt",
                varying_term="",
                moderators=trt_moderators,
                row_mask=p1_mask,
                score_mean_link=link,
                # Treatment interactions make beta_trt and the AME diverge in sign, so
                # the reported direction follows the marginal effect, not the coefficient (#391).
                direction_from_ame=True,
            )
            rope_df = pd.DataFrame([rope_s])
            save_table(ctx, "rope_summary", rope_df)
            meta_extra["rope_summary"] = rope_s
            print_table(
                metrics_table(
                    [{"metric": k, "value": v} for k, v in rope_s.items()],
                    title=f"ROPE summary ({spec.outcome_symbol}, delta={delta_items:g} items)",
                    columns=["metric", "value"],
                )
            )
            save_rope_plot(
                ctx, spec.outcome_symbol, trt, n_marg, delta_items,
                term="beta_trt", varying_term="", moderators=trt_moderators,
                row_mask=p1_mask, split=True, score_mean_link=link,
            )
        elif off_floor and delta_prob is not None:
            # Off-floor risk-difference ROPE, matching the floored ITT path
            # (#125 Area 4). The 10 pp δ was signed off by the education lead
            # (2026-07-01, #144), so it is NOT provisional; the ITT floored path
            # sets provisional_delta=False and this mirrors it.
            rope_s = _report.rope_summary(
                ctx.trace, G=trt, n_trials=1, delta=delta_prob,
                ci_prob=ctx.reporting.ci_prob, term="beta_trt", varying_term="",
                moderators=trt_moderators, row_mask=p1_mask,
                direction_from_ame=True,  # direction from the off-floor RD AME, not beta_trt (#391)
            )
            rope_s["provisional_delta"] = False  # 10 pp signed off (#144, 2026-07-01)
            rope_s["delta_scale"] = "risk_difference"
            save_table(ctx, "rope_summary", pd.DataFrame([rope_s]))
            meta_extra["rope_summary"] = rope_s
            save_rope_plot(
                ctx, spec.outcome_symbol, trt, 1, delta_prob,
                term="beta_trt", varying_term="", moderators=trt_moderators,
                row_mask=p1_mask, split=True,
            )
            # δ-sensitivity sweep on the risk-difference scale (#144): 10/15/20 pp,
            # the grid the sign-off mandates (mirrors the floored ITT path).
            sens_df = _report.rope_sensitivity(
                ctx.trace, G=trt, n_trials=1, deltas=ROPE_DELTA_PROB_GRID,
                term="beta_trt", varying_term="", moderators=trt_moderators,
                row_mask=p1_mask,
            )
            save_table(ctx, "rope_sensitivity", sens_df)

        # Predicted-scores contrast panel + icon array (#316), averaged over the
        # same period-1 reference rows as treatment_marginal.csv and integrating
        # the child random intercept for a *new* typical child (the fitted
        # children's intercepts are swapped for fresh population draws).
        write_predicted_scores(
            ctx,
            outcome_symbol=spec.outcome_symbol,
            G=trt,
            n_trials=n_marg,
            term="beta_trt",
            varying_term="",
            moderators=trt_moderators,
            row_mask=p1_mask,
            likelihood="bernoulli" if off_floor else "beta_binomial",
            score_mean_link=link,
            child_re=True,
            child_idx=built.prepared.child_idx,
            delta=delta_prob if off_floor else delta_items,
            population=(
                "covariate profiles drawn from the period-1 "
                "randomised-transition rows"
            ),
            contrast_status=(
                "model-dependent interaction-aware contrast (associational "
                "moderation variant; partly informed by post-crossover data)"
                if moderation_variant
                else "randomised on-intervention contrast (period-1 anchor)"
            ),
            event_label="off the floor at the period end",
            split=True,
        )

        # --- Period-1-only refit sensitivity (#575 finding 2) ---
        # The model of record stacks every transition, so beta_trt is fitted on a
        # likelihood whose shared parameters (period effects, child intercepts,
        # covariate slopes) borrow from post-crossover rows; the period-1
        # restriction above is applied only when averaging the marginal. This
        # refit keeps the identical specification but drops every post-crossover
        # row from the likelihood, so the comparison quantifies that borrowing
        # for the causal headline. With each child observed once, the child
        # intercept is only jointly identified with the overdispersion and leans
        # on its prior; that is the price of keeping the refit a pure row
        # restriction rather than a second specification.
        if plan.period1_sensitivity_required:
            section_header("Period-1-only refit sensitivity")
            from dataclasses import replace as _dc_replace

            from language_reading_predictors.statistical_models.preprocessing import (
                _subset_prepared,
            )
            from language_reading_predictors.statistical_models.subfits import (
                run_subfit,
            )

            p1_frame = _dc_replace(
                _subset_prepared(prepared, np.asarray(prepared.phase) == 0),
                n_phases=1,
            )
            p1_built = _factories.build_gain_factors_model(
                p1_frame,
                **plan.factory_kwargs(effective_adjustment=loader_adjust_for),
            )
            res = run_subfit(
                ctx,
                p1_built,
                label="period1_only",
                role="sensitivity",
                trace_filename="trace_period1_only.nc",
                extra_var_names=["beta_trt"],
            )
            p1_payload = p1_built.require_payload(
                GainFactorsPayload, family="gain_factors"
            )
            trt_p1 = (
                (p1_built.prepared.G == 1) | (p1_built.prepared.phase >= 1)
            ).astype(float)
            tme_p1 = _report.treatment_marginal_effect(
                res.trace,
                trt=trt_p1,
                n_trials=n_marg,
                moderators=p1_payload.trt_interaction_moderators,
                ci_prob=ctx.reporting.ci_prob,
                row_mask=None,
                score_mean_link=link,
            )
            _b = ctx.trace.posterior["beta_trt"].values.ravel()
            _b1 = res.trace.posterior["beta_trt"].values.ravel()
            _lo_q = (1 - ctx.reporting.ci_prob) / 2
            _hi_q = 1 - _lo_q

            def _p1_row(
                fit_label: str,
                b_draws: np.ndarray,
                tme_row: dict[str, float],
                built_x: _factories.BuiltModel[GainFactorsPayload],
            ) -> dict[str, object]:
                return {
                    "fit": fit_label,
                    "n_rows": int(built_x.prepared.n_obs),
                    "n_children": int(built_x.prepared.n_children),
                    "beta_trt_median": float(np.median(b_draws)),
                    "beta_trt_lo": float(np.quantile(b_draws, _lo_q)),
                    "beta_trt_hi": float(np.quantile(b_draws, _hi_q)),
                    "trt_items_median": tme_row["trt_items_median"],
                    "trt_items_lo": tme_row["trt_items_lo"],
                    "trt_items_hi": tme_row["trt_items_hi"],
                    "prob_trt_pos": tme_row["prob_trt_pos"],
                }
            p1_df = pd.DataFrame(
                [
                    {
                        **_p1_row("primary_period_stacked", _b, tme, built),
                        "converged": None,
                        "max_rhat": None,
                        "min_ess": None,
                    },
                    {
                        **_p1_row("period1_only", _b1, tme_p1, p1_built),
                        "converged": res.convergence.get("converged"),
                        "max_rhat": res.convergence.get("max_rhat"),
                        "min_ess": res.convergence.get("min_ess"),
                    },
                ]
            )
            save_table(ctx, "period1_sensitivity", p1_df)
            meta_extra["period1_sensitivity"] = {
                "beta_trt_shift": float(np.median(_b1) - np.median(_b)),
                "items_shift": float(
                    tme_p1["trt_items_median"] - tme["trt_items_median"]
                ),
                "period1_converged": res.convergence.get("converged"),
            }
            print_table(
                metrics_table(
                    [
                        {"metric": k, "value": v}
                        for k, v in meta_extra["period1_sensitivity"].items()
                    ],
                    title="Period-1-only refit sensitivity",
                    columns=["metric", "value"],
                )
            )

    # --- Per-covariate items-scale association marginals (#310) ---
    # The adjusted-association analogue of the treatment marginal: for each covariate
    # (own baseline, age, cognitive ability, skill baselines, raw-covariate adjusters)
    # push a +1 SD perturbation — and, for the bounded-count baselines, a +k-items one —
    # through the posterior onto the probability / items scales. Runs for the treated_only
    # (…b) companions too (they keep the covariate associations even without beta_trt).
    # Averaging population = ALL stacked rows (row_mask=None): these are descriptive
    # associations, not the randomised period-1 contrast, so every fitted observation
    # counts. That choice is recorded in config.json (meta_extra) as well as the note.
    assoc_terms = _gf_association_terms(
        plan, built, adjust_for=adjust_for, off_floor=off_floor
    )
    if assoc_terms:
        n_assoc = 1 if off_floor else built.prepared.n_trials[spec.outcome_symbol]
        am = _report.association_marginals(
            ctx.trace,
            terms=assoc_terms,
            n_trials=n_assoc,
            off_floor=off_floor,
            ci_prob=ctx.reporting.ci_prob,
            row_mask=None,
            score_mean_link=link,
        )
        save_table(ctx, "association_marginals", am)
        meta_extra["association_marginals"] = {
            "averaging_population": "all_stacked_rows",
            # Per-measure increments (#575 finding 3): max(1, round(n/10)) per
            # bounded-count term, capped at fit time to the items the scale has
            # left; None for the standardised / toggled terms.
            "k_items": {t.label: t.k_items for t in assoc_terms},
            "terms": [t.label for t in assoc_terms],
        }
        print_table(
            ranked_dataframe_table(
                am,
                title=f"Association marginals ({spec.outcome_symbol}) - items scale",
                columns=["term", "scale", "items_median", "items_lo", "items_hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    # Per-child fitted-vs-observed panels (#317 fig 2), one per period transition.
    write_child_fit(
        ctx,
        outcome_symbol=spec.outcome_symbol,
        wave=built.prepared.phase,
        child_idx=built.prepared.child_idx,
        off_floor=off_floor,
        obs_node="y_offfloor" if off_floor else "y_post",
        x_label="period transition",
    )

    write_run_metadata(ctx, extra=meta_extra)
    return finalize_report(ctx)
