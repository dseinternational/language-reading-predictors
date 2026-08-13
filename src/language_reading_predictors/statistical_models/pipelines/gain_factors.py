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
from language_reading_predictors.statistical_models.gain_factors import (
    resolve_active_interactions,
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


def _gf_coef_names(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    """Factor coefficients to report in the LRPGF factor table (interpretable
    terms only; nuisance alpha/alpha_phase/kappa/sigma_child are excluded).

    ``adjust_for`` overrides the requested ``spec.extra['adjust_for']`` with the
    **actually fitted** set (a constant ``_missing`` indicator is dropped by the
    loader and gets no ``gamma_{c}`` coefficient), so the pipeline passes the
    post-filter tuple; ``None`` falls back to the requested set (used off the fit
    path, e.g. in tests)."""
    extra = spec.extra
    treated_only = bool(extra.get("treated_only", False))
    adj = extra.get("adjust_for", ()) if adjust_for is None else adjust_for
    names: list[str] = []
    if not treated_only:
        names.append("beta_trt")
    # The graded gamma_own drops on the off-floor (Bernoulli) path (A4) — see
    # build_gain_factors_model. The binary off-floor-at-pre indicator main effect
    # ``gamma_own_offfloor`` always stands in for it there (#391 finding 2
    # decision, 2026-07-22), so it is reported unconditionally in its place.
    active_interactions = resolve_active_interactions(
        extra.get("interactions", ()), treated_only=treated_only
    )
    if extra.get("likelihood") != "bernoulli_offfloor":
        names.append("gamma_own")
    else:
        names.append("gamma_own_offfloor")
    names.append("gamma_A")
    if extra.get("ability_covariate"):
        names.append("gamma_ability")
    names += [f"gamma_{s}" for s in extra.get("skill_symbols", ())]
    names += [f"gamma_{c}" for c in adj]
    names += [f"gamma_int_{a}_{b}" for a, b in active_interactions]
    return names


def _gf_diag_vars(
    spec: ModelSpec, adjust_for: tuple[str, ...] | None = None
) -> list[str]:
    # No kappa under the off-floor Bernoulli likelihood.
    tail = (
        ["sigma_child"]
        if spec.extra.get("likelihood") == "bernoulli_offfloor"
        else ["kappa", "sigma_child"]
    )
    # Include the per-phase intercept vector, mirroring the level-factor plan's
    # alpha_time (issue #274 item 2); the gate already covers it via the free-RV
    # scan, this keeps the human-readable diagnostics.csv consistent across the
    # two families.
    return ["alpha", "alpha_phase", *_gf_coef_names(spec, adjust_for), *tail]


def _gf_association_terms(
    spec: ModelSpec,
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
    own = spec.outcome_symbol
    extra = spec.extra
    skill_symbols = tuple(extra.get("skill_symbols", ()))
    ability_covariate = extra.get("ability_covariate")
    interactions = tuple(tuple(p) for p in extra.get("interactions", ()))
    treated_only = bool(extra.get("treated_only", False))

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
    active_interactions = resolve_active_interactions(
        interactions, treated_only=treated_only
    )

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
        # Items equivalent of +1 SD of a bounded-count covariate, at its mean proportion.
        return float(n * (_expit(_logit(p) + sd_logit) - p))

    terms: list[_report.AssociationTerm] = []
    if not off_floor:
        p_own = float(np.mean(_expit(bp.pre_logit[own])))
        n_own = int(bp.n_trials[own])
        terms.append(
            AT("own", "gamma_own", scales["own"], _ints_for("own"),
               n_items=n_own, mean_prop=p_own, sd_items=_sd_items(scales["own"], p_own, n_own))
        )
    else:
        # The binary off-floor-at-pre indicator association (#391 finding 2
        # decision): the "+1" perturbation is the at-floor -> off-floor switch.
        terms.append(
            AT("own", "gamma_own_offfloor", 1.0, _ints_for("own"),
               perturbation_label="off-floor at pre (0 to 1)")
        )
    terms.append(AT("age", "gamma_A", 1.0, _ints_for("age")))
    if ability_covariate is not None:
        terms.append(AT("ability", "gamma_ability", 1.0, _ints_for("ability")))
    for s in skill_symbols:
        p_s = float(np.mean(_expit(bp.pre_logit[s])))
        n_s = int(bp.n_trials[s])
        terms.append(
            AT(s, f"gamma_{s}", scales[s], _ints_for(s),
               n_items=n_s, mean_prop=p_s, sd_items=_sd_items(scales[s], p_s, n_s))
        )
    for c in adjust_for:
        if c.endswith("_missing"):
            continue
        sd_c = float(np.std(np.asarray(bp.covariates[c], dtype=float), ddof=1))
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
    # loader and must not be built or reported as adjusted-for.
    adjust_for = tuple(c for c in plan.adjust_for if c in prepared.covariates)
    ctx.prepared = prepared
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_gain_factors_model(
        prepared, **plan.factory_kwargs(effective_adjustment=adjust_for)
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    # The diagnostic FOCAL term, not a causal designation: for a moderation
    # variant beta_trt is still the term whose mixing, ESS evolution, forest and
    # prior sensitivity a reader needs, but it is presented as a model-dependent
    # association everywhere (factor_summary role, priors-table override, the
    # results partial and the key-findings box all branch on moderation_variant)
    # — the plot titles here are deliberately neutral ("Rank plot", "Effect
    # posterior"), so focusing them on beta_trt asserts nothing causal.
    _focal_gf = None if treated_only else "beta_trt"
    _gf_diag = _gf_diag_vars(spec, adjust_for)
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(_gf_diag),
            ppc_var_names=(obs_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, spec.outcome_symbol, node=obs_node
            ),
            psense_timing="family_tail",
            extended_term=_focal_gf,
        ),
    )
    # Preserve the family's established post-trace order: overlay, optional
    # forest, then optional power scaling. Treated-only fits have no focal term.
    _diag.save_prior_posterior_plot(ctx, var_names=_gf_diag)
    if _focal_gf is not None:
        save_forest_plot(ctx, [_focal_gf])
        _diag.run_psense(ctx, var_names=[_focal_gf])

    section_header("Factor summary")
    # A moderation variant's beta_trt is NOT flagged causal: its interaction-aware
    # marginal is model-dependent (partly informed by post-crossover data), so every
    # row — the treatment term included — reads as an adjusted association there.
    gf_causal_terms: tuple[str, ...] = (
        () if moderation_variant else ("beta_trt",)
    )
    fs = _report.factor_summary(
        ctx.trace, _gf_coef_names(spec, adjust_for), ci_prob=ctx.reporting.ci_prob,
        causal_terms=gf_causal_terms,
    )
    save_table(ctx, "factor_summary", fs)
    save_association_forest(ctx, _gf_coef_names(spec, adjust_for), gf_causal_terms)
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
        "treated_only": treated_only,
        # Requested vs actually-fitted adjustment set, incl. dropped-constant
        # covariates (#247 / #258 review P1). Skills enter at the pre baseline; the
        # raw-covariate confounders (hs/deapp_c/erbto) at the split wave.
        "effective_adjustment": effective_adjustment(
            spec,
            built.prepared,
            adjust_for=adjust_for,
            ability_covariate=ability_covariate,
            baseline_symbol=spec.outcome_symbol,
            skill_baselines=skill_symbols,
        ),
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
        trt_moderators = built.extras.get("trt_interaction_moderators", [])
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
                row_mask=p1_mask, split=True,
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
        spec, built, adjust_for=adjust_for, off_floor=off_floor
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
        )
        save_table(ctx, "association_marginals", am)
        meta_extra["association_marginals"] = {
            "averaging_population": "all_stacked_rows",
            "k_items": 5,
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
