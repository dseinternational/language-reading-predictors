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
    Standardiser,
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
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import PrimaryFitPlan


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

    dose_vars = plan.diagnostic_vars()
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(dose_vars),
            ppc_var_names=(plan.observation_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, plan.outcome_symbol
            ),
            extended_term=plan.focal_term,
            compute_loo=plan.compute_loo,
            # The trace is intentionally persisted after the dose-slope summary,
            # preserving this family's established artefact order.
            save_trace=False,
        ),
    )

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
    dose_scaler: Standardiser | None = None,
    marginal_row_mask: np.ndarray | None = None,
) -> None:
    """Posterior dose slope (overall + per-period) on the per-1-SD logit scale.

    ``dose_scaler`` is the standardisation the fitted slope is per-1-SD *of*,
    persisted as ``dose_mean_sessions`` / ``dose_sd_sessions``. The default
    (``None``) reads the loader scaler ``ctx.prepared.covariate_scalers[dose_covariate]``,
    which is correct for the dose_response family — its factory fits the
    loader-standardised dose directly. The DiD dose companions must pass their
    fitted payload's treated-rows scaler instead: ``build_did_model``
    re-standardises sessions among treated P1/P2 rows only, so the loader scaler
    would misstate their per-session calibration (and contradict the
    ``dose_standardization`` block the same fit records in ``config.json``).

    ``marginal_row_mask`` restricts the natural-scale ``dose_marginal_summary``
    average (and the matching prior pushforward) to a boolean subset of the
    fitted rows. The default (``None``, all rows) is the dose_response family's
    definition — its zero-dose rows anchor the slope at dose = 0 on one common
    standardised scale, so a +1 SD step is meaningful on every row. The DiD dose
    companions must pass their treated-row mask: their dose is treated-centred
    with untreated rows hard-coded to zero, so a dose step on an untreated row
    is not a supported counterfactual of that design and the coherent averaging
    population for the intensive-margin estimand is the treated rows.
    """
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
    if dose_scaler is None:
        dose_scaler = ctx.prepared.covariate_scalers[dose_covariate]
    # Persist the standardisation the slope is per-1-SD of, so downstream
    # named-confounder calibration can put slopes from separately fitted
    # outcomes onto one common per-session scale.  Older artefacts are
    # reconstructible from the data, but new fits should be self-describing
    # (#324).
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
    if marginal_row_mask is not None:
        keep = np.asarray(marginal_row_mask)
        if keep.ndim != 1 or keep.dtype != bool or keep.shape[0] != eta.shape[0]:
            raise ValueError(
                f"marginal_row_mask must be a 1-D boolean mask with "
                f"{eta.shape[0]} entries; got dtype {keep.dtype}, shape {keep.shape}."
            )
        if not keep.any():
            raise ValueError("marginal_row_mask selects no rows for the dose marginal.")
        eta = eta[keep]
        delta_eta = delta_eta[keep]
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
                # Self-describing averaging population (see marginal_row_mask
                # in the docstring): the DiD dose companions average over
                # treated rows only; the dose_response family over all rows.
                "n_rows": int(eta.shape[0]),
                "row_population": (
                    "all fitted rows"
                    if marginal_row_mask is None
                    else "masked rows (DiD dose: treated rows only)"
                ),
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
    # computed a few lines up, and ``marginal_row_mask`` carries through, so
    # prior and estimate share one transform and one averaging population.
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
            row_mask=marginal_row_mask,
        ),
    )
