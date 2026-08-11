# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dose-response orchestration (LRP-RLI-DOSE, #104 Phase 2).

``fit_dose_response`` reuses the mechanism-family backbone — Beta-Binomial
conditional change, phase intercepts, subject random intercept — with cumulative
intervention sessions as the focal predictor. It also owns
:func:`write_dose_slope_summary`, the per-period dose-slope table that the DiD
family's dose companions publish from their own fits (#394 step 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    dose_response as _dose_response,
    factories as _factories,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    marginal_pushforward_rows,
    pushforward_outcome_label,
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


def fit_dose_response(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Period-resolved dose-response fit (#104 Phase 2).

    Reuses the mechanism-family backbone (Beta-Binomial conditional change,
    phase intercepts, subject random intercept) but the focal predictor is the
    per-period intervention **dose** (``attend``), entered with partial-pooled
    period-specific slopes. See :func:`factories.build_dose_response_model`.
    """
    require_spec(spec, "dose_response")

    # Resolve family settings before ``make_context`` can reset the output
    # directory and before preparation reads RLI data (#394 pillar 4).
    plan = _dose_response.resolve_dose_response_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    prepared = load_and_prepare(**plan.prepare_kwargs())
    ctx.prepared = prepared

    print_header(ctx)

    section_header("Build model")

    built = _factories.build_dose_response_model(
        prepared,
        **plan.factory_kwargs(),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, plan.outcome_symbol)

    run_sampling_and_loo(ctx, compute_loo=plan.compute_loo)

    section_header("Summary diagnostics")
    dose_vars = plan.diagnostic_vars()
    _diag.summary_diagnostics(ctx, var_names=dose_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=dose_vars)

    run_ppc(ctx, var_names=[plan.observation_node])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=dose_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=plan.focal_term)

    section_header("Dose-slope summary")
    write_dose_slope_summary(
        ctx,
        period_varying=plan.period_varying_dose,
        dose_covariate=plan.dose_covariate,
    )

    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=dose_vars)
    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "adjustment": spec.adjustment,
            "period_varying_dose": plan.period_varying_dose,
            "ability_adjust_symbols": list(plan.ability_adjust_symbols),
        },
    )

    return finalize_report(ctx)


def _summarise_draws(
    values: np.ndarray, ci_prob: float, *, include_p_pos: bool = True
) -> dict[str, float]:
    """Mean, equal-tailed CI and (optionally) P(>0) for a 1-D array of draws.

    ``ci_prob`` is the interval *coverage* probability (equal-tailed), read from
    ``ctx.reporting.ci_prob`` — see the naming note in ``context.make_context`` (#170).
    ``include_p_pos=False`` omits the directional ``P(>0)`` for a strictly-positive
    quantity (e.g. a between-period SD) where it is trivially 1 and meaningless.
    """
    lo_q = (1.0 - ci_prob) / 2.0
    out = {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "lo": float(np.quantile(values, lo_q)),
        "hi": float(np.quantile(values, 1.0 - lo_q)),
        # Inner 50% equal-tailed band alongside the headline ci_prob interval.
        "lo50": float(np.quantile(values, 0.25)),
        "hi50": float(np.quantile(values, 0.75)),
    }
    if include_p_pos:
        out["p_pos"] = float(np.mean(values > 0.0))
    return out


def write_dose_slope_summary(
    ctx: StatisticalFitContext,
    *,
    period_varying: bool,
    dose_covariate: str = "attend",
) -> None:
    """Posterior dose slope (overall + per-period) on the per-1-SD logit scale."""
    post = ctx.trace.posterior
    ci_prob = ctx.reporting.ci_prob
    rows: list[dict[str, object]] = []

    def _draws(name: str) -> np.ndarray:
        return post[name].stack(sample=("chain", "draw")).values

    if period_varying:
        rows.append({"term": "dose_overall", **_summarise_draws(_draws("mu_dose"), ci_prob)})
        bdp = _draws("beta_dose_phase")  # (phase, sample)
        for p in range(bdp.shape[0]):
            rows.append(
                {"term": f"dose_period{p + 1}", **_summarise_draws(bdp[p], ci_prob)}
            )
        rows.append(
            {
                "term": "sigma_dose_between_period",
                **_summarise_draws(_draws("sigma_dose"), ci_prob, include_p_pos=False),
            }
        )
    else:
        rows.append({"term": "dose_pooled", **_summarise_draws(_draws("beta_dose"), ci_prob)})

    df = pd.DataFrame(rows)
    dose_scaler = ctx.prepared.covariate_scalers[dose_covariate]
    # Persist the original standardisation so downstream named-confounder
    # calibration can put slopes from separately fitted outcomes onto one common
    # per-session scale.  Older artefacts are reconstructible from the data, but
    # new fits should be self-describing (#324).
    df["dose_mean_sessions"] = float(dose_scaler.mean)
    df["dose_sd_sessions"] = float(dose_scaler.sd)
    save_table(ctx, "dose_slope_summary", df)

    # Natural-scale average marginal association for the key-findings box
    # (#320): increase the standardised session dose by 1 on every fitted row,
    # using that row's period-specific slope where the model varies it by period.
    eta = (
        post["eta"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )
    if period_varying:
        # The period dimension is named "phase" in the dose_response family but
        # "dose_phase" in the DiD dose companions; derive it rather than hardcode
        # one spelling (a hardcoded "phase" crashed did-007 with a missing-dim
        # ValueError before it could write its report).
        stacked_bdp = post["beta_dose_phase"].stack(sample=("chain", "draw"))
        phase_dim = next(d for d in stacked_bdp.dims if d != "sample")
        phase_slopes = stacked_bdp.transpose(phase_dim, "sample").values
        delta_eta = phase_slopes[np.asarray(ctx.prepared.phase, dtype=int)]
    else:
        slope = post["beta_dose"].stack(sample=("chain", "draw")).values
        delta_eta = np.broadcast_to(slope[None, :], eta.shape)
    outcome = ctx.spec.outcome_symbol or "W"
    items = (
        expit(eta + delta_eta) - expit(eta)
    ).mean(axis=0) * float(ctx.prepared.n_trials[outcome])
    lo_q = (1 - ci_prob) / 2
    marginal = pd.DataFrame(
        [
            {
                "items_median": float(np.median(items)),
                "items_lo": float(np.quantile(items, lo_q)),
                "items_hi": float(np.quantile(items, 1 - lo_q)),
                "items_lo50": float(np.quantile(items, 0.25)),
                "items_hi50": float(np.quantile(items, 0.75)),
                "prob_pos": float(np.mean(items > 0)),
            }
        ]
    )
    save_table(ctx, "dose_marginal_summary", marginal)
    print_table(
        metrics_table(
            [
                {"metric": r["term"], "value": r["mean"], "lo": r["lo"], "hi": r["hi"]}
                for r in rows
            ],
            title=(
                f"Dose slope (logit / 1 SD dose) - {int(ci_prob * 100)}% CI (equal-tailed)"
            ),
            columns=["metric", "value", "lo", "hi"],
        )
    )
    # Estimand-scale prior check on the reported dose marginal (#381). Shared by
    # ``fit_dose_response`` and the DiD dose companions, which both route their
    # slope reporting through this writer. ``mu_dose`` is the period-varying
    # family's overall slope, the same term ``dose_overall`` summarises above;
    # ``forward`` matches the ``expit(eta + delta) - expit(eta)`` marginal
    # computed a few lines up, so prior and estimate share one transform.
    write_prior_pushforward(
        ctx,
        marginal_pushforward_rows(
            ctx,
            [
                (
                    "mu_dose" if period_varying else "beta_dose",
                    "the association of a +1 SD session-dose step with "
                    f"{pushforward_outcome_label(ctx, outcome)}",
                )
            ],
            n_trials=int(ctx.prepared.n_trials[outcome]),
            convention="forward",
        ),
    )
