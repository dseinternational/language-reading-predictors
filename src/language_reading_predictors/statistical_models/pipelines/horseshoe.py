# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regularised-horseshoe predictor-ranking orchestration (``kind="horseshoe"``).

``fit_horseshoe`` refits the construct predictor set under a regularised horseshoe
prior and ranks predictors by posterior ``P(|beta| > delta)``, as an independent
Bayesian cross-check on the gradient-boosting ranking. ``fit_rlm_horseshoe`` is the
Byrne (RLM) port; there is no gradient-boosting layer for that cohort, so its
cross-check partner is the Byrne adjusted fit rather than a GB comparison table.
Both ports read the RLM span frame through :func:`.adjusted.rlm_nuisance_names`.

A horseshoe ranking is an association ranking. It says which predictors carry
signal once the others are in the model, not which of them would change the
outcome if intervened on.
"""

from __future__ import annotations

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    factories as _factories,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.pipelines.adjusted import (
    rlm_nuisance_names,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    horseshoe_pushforward_rows,
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
    run_ppc,
    run_sampling_and_loo,
    write_run_metadata,
)


def fit_horseshoe(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Regularized-horseshoe predictor-ranking fit (LRPHS, #116 Phase E).

    An independent Bayesian sensitivity cross-check on the gradient-boosting
    predictor ranking: one horseshoe regression (gain or level, per ``spec.extra``)
    over the full construct predictor set, ranked by posterior
    ``P(|beta| > delta)``. Writes ``predictor_ranking.csv`` alongside the standard
    trace / diagnostics / LOO / PPC artefacts. Not causal — a which-predictors
    -carry-signal read to compare against the GB cluster ranking.
    """
    require_spec(spec, "horseshoe")
    e = spec.extra
    outcome = spec.outcome_symbol or "W"
    gain = bool(e.get("gain", True))
    predictors = list(e["predictors"])
    lang_symbols = tuple(e.get("language_composite_symbols", ["R", "E", "F"]))
    covariates = tuple(e.get("covariates", ()))
    delta = float(e.get("delta", 0.1))
    tau0 = float(e.get("tau0", 0.1))
    slab_scale = float(e.get("slab_scale", 2.0))
    slab_df = float(e.get("slab_df", 4.0))
    post_time = int(e.get("post_time", 4))
    phase_mode = e.get("phase_mode", "span" if gain else "levels")

    # 94% intervals, matching the LRP65 adjusted-model convention.
    ctx = make_context(spec, config, ci_prob=0.89)
    # The horseshoe has a funnel geometry (global-local scales); lift target_accept
    # above the tier default so the sampler takes smaller steps near the neck.

    section_header("Prepare data")
    measure_syms = tuple(
        dict.fromkeys(
            [outcome]
            + [p for p in predictors if p not in ("age", "lang", *covariates)]
            + list(lang_symbols)
        )
    )
    prepared = load_and_prepare(
        phase_mode=phase_mode,
        post_time=post_time,
        outcomes=measure_syms,
        covariates=covariates,
    )
    ctx.prepared = prepared
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_horseshoe_model(
        prepared,
        outcome_symbol=outcome,
        predictors=predictors,
        gain=gain,
        tau0=tau0,
        slab_scale=slab_scale,
        slab_df=slab_df,
        language_composite_symbols=lang_symbols,
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome)

    run_sampling_and_loo(ctx)

    # Coupling term present in the model: gamma_own (gain) or the fixed age slope
    # gamma_A (level) — but the level model suppresses gamma_A when age is itself a
    # horseshoe-ranked predictor (build_horseshoe_model), so only list it then.
    if gain:
        coupling_vars = ["gamma_own"]
    elif "age" not in predictors:
        coupling_vars = ["gamma_A"]
    else:
        coupling_vars = []
    diag_vars = ["alpha", *coupling_vars, "kappa", "hs_tau", "hs_c2", "beta"]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Predictor ranking")
    ranking = _report.horseshoe_ranking(ctx.trace, delta=delta)
    save_table(ctx, "predictor_ranking", ranking)
    print_table(ranked_dataframe_table(ranking.head(10), title="Horseshoe predictor ranking (top 10)"))
    write_prior_pushforward(ctx, horseshoe_pushforward_rows(ctx, predictors, outcome))

    meta_extra = {
        "framing": "gain" if gain else "level",
        "phase_mode": phase_mode,
        "predictors": predictors,
        "covariates": list(covariates),
        "delta": delta,
        "tau0": tau0,
        "slab_scale": slab_scale,
        "slab_df": slab_df,
        "gb_reference": e.get("gb_reference"),
        "ranking_top": ranking.head(3)[["predictor", "p_abs_gt_delta"]].to_dict(
            "records"
        ),
    }
    write_run_metadata(ctx, extra=meta_extra)

    return finalize_report(ctx)


def fit_rlm_horseshoe(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne horseshoe predictor-ranking fit (#338 Phase D, ``lrp-rlm-hs-001``).

    The RLI gain-framing ``fit_horseshoe`` on the Byrne span frame: one
    regularised-horseshoe regression over the wave-1 predictor set (age
    included), ranked by posterior ``P(|beta| > delta)``. Writes
    ``predictor_ranking.csv`` so the shared ``horseshoe`` partial and
    key-findings builder apply unchanged. There is no gradient-boosting layer
    for the Byrne cohort, so no ``horseshoe_vs_gb.csv`` comparison is written -
    the cross-check partner here is ``lrp-rlm-adj-001``.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        load_rlm_span_frame,
    )

    require_spec(spec, "horseshoe")
    e = spec.extra
    outcome = spec.outcome_symbol or "basread"
    predictor_measures = tuple(
        e.get("predictor_measures", ("bpvs", "trog", "basdig", "bassim", "basnum"))
    )
    include_age = bool(e.get("use_age_predictor", True))
    pre_wave = int(e.get("pre_wave", 1))
    post_wave = int(e.get("post_wave", 3))
    delta = float(e.get("delta", 0.1))
    tau0 = float(e.get("tau0", 0.1))
    slab_scale = float(e.get("slab_scale", 2.0))
    slab_df = float(e.get("slab_df", 4.0))

    ctx = make_context(spec, config, ci_prob=0.89)

    section_header("Prepare data")
    frame = load_rlm_span_frame(
        outcome=outcome,
        predictor_measures=predictor_measures,
        include_age=include_age,
        pre_wave=pre_wave,
        post_wave=post_wave,
    )
    ctx.prepared = frame
    predictors = list(frame.predictors)
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_horseshoe_model(
        frame,
        predictors=predictors,
        tau0=tau0,
        slab_scale=slab_scale,
        slab_df=slab_df,
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome, node="y_post")

    run_sampling_and_loo(ctx)

    nuisance = rlm_nuisance_names(frame)
    diag_vars = ["alpha", "gamma_own", "kappa", "hs_tau", "hs_c2", "beta", *nuisance]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    section_header("Predictor ranking")
    ranking = _report.horseshoe_ranking(ctx.trace, delta=delta)
    save_table(ctx, "predictor_ranking", ranking)
    print_table(
        ranked_dataframe_table(ranking, title="Horseshoe predictor ranking")
    )
    write_prior_pushforward(ctx, horseshoe_pushforward_rows(ctx, predictors, outcome))

    write_run_metadata(
        ctx,
        extra={
            "study_id": "rlm",
            "framing": "gain",
            "outcome": outcome,
            "pre_wave": pre_wave,
            "post_wave": post_wave,
            "predictors": predictors,
            "group_nuisance_terms": nuisance,
            "delta": delta,
            "tau0": tau0,
            "slab_scale": slab_scale,
            "slab_df": slab_df,
            "ranking_top": ranking.head(3)[["predictor", "p_abs_gt_delta"]].to_dict(
                "records"
            ),
        },
    )
    return finalize_report(ctx)
