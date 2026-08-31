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

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from language_reading_predictors.statistical_models.likelihood import (
    apply_score_mean_link,
)

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
    pushforward_outcome_label,
    PriorEvidenceUnavailable,
    require_prior_evidence,
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

    section_header("Dose support and contrast")
    # ``built.prepared``, never the loader's ``prepared``: the factory drops rows
    # (a missing outcome, or a child with no verified t1 ability) and the payload's
    # arrays are aligned to what it kept. dose-177 loads 157 rows and fits 156.
    contrast = resolve_dose_contrast(
        built.payload, np.asarray(built.prepared.phase, dtype=int)
    )
    save_table(ctx, "dose_support", contrast.support_table)

    section_header("Dose-slope summary")
    write_dose_slope_summary(
        ctx,
        period_varying=plan.period_varying_dose,
        dose_covariate=plan.dose_covariate,
        dose_scaler=built.payload.dose_scaler,
        marginal_row_mask=built.payload.treated,
        between_term=(
            "beta_dose_between" if plan.decompose_between_within else None
        ),
        include_presence_term=True,
        contrast_std=contrast.delta_std,
        contrast_sessions=contrast.delta_sessions,
        contrast_label=contrast.label,
        contrast_kind=plan.dose_contrast,
        support_note=contrast.note,
        # The link the factory BUILT, so the dose marginal and its prior pushforward
        # both map onto the response scale the likelihood used (#619).
        score_mean_link=built.payload.score_mean_link,
    )

    section_header("Dose calibration checks")
    write_dose_band_ppc(ctx, payload=built.payload)

    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=dose_vars)
    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "loo_unit": plan.loo_unit,
            "loo_note": plan.loo_note,
            # The exposure and the adjustment set are different facts about a fit and
            # are recorded separately (#587 finding 10). ``effective_adjustment`` used
            # to be filled from ``prepared.covariates``, which for this family is the
            # exposure alone, so every dose fit recorded its adjustment set as
            # ``["attend"]`` while omitting arm, age and the baselines it fitted.
            "exposure": plan.exposure,
            "dose_margin": plan.dose_margin,
            "effective_adjustment": _dose_effective_adjustment(plan, ctx.prepared),
            "coefficient_meanings": plan.coefficient_meanings(),
            "adjustment": spec.adjustment,
            "period_varying_dose": plan.period_varying_dose,
            "decompose_between_within": plan.decompose_between_within,
            "ability_adjust_symbols": list(plan.ability_adjust_symbols),
            "ability_baseline_wave": plan.ability_baseline_wave,
            "dose_standardization": {
                "scope": "fitted on-intervention rows",
                "mean_sessions": float(built.payload.dose_scaler.mean),
                "sd_sessions": float(built.payload.dose_scaler.sd),
                "n_treated_rows": int(built.payload.treated.sum()),
                "n_fitted_rows": int(ctx.prepared.n_obs),
            },
        },
    )

    return finalize_report(ctx)


def _dose_effective_adjustment(plan, prepared) -> dict:
    """The adjustment set this fit actually conditioned on — never the exposure."""
    terms: list[dict[str, object]] = [
        {
            "term": f"{plan.adjust_baseline_symbol}_pre",
            "kind": "autoregressive_baseline",
            "source_column": prepared.column_map.get(
                plan.adjust_baseline_symbol, plan.adjust_baseline_symbol
            ),
            "wave": "pre",
        }
    ]
    if plan.adjust_group:
        terms.append(
            {
                "term": "beta_arm_late",
                "kind": "treatment_history",
                "source_column": "group",
                "wave": "time_invariant",
                "note": "post-crossover periods only; period 1 arm is theta_treated",
            }
        )
    if plan.adjust_age:
        terms.append(
            {"term": "A", "kind": "covariate", "source_column": "age", "wave": "pre"}
        )
    for symbol in plan.ability_adjust_symbols:
        terms.append(
            {
                "term": f"{symbol}_pre",
                "kind": "ability_baseline",
                "source_column": prepared.column_map.get(symbol, symbol),
                "wave": plan.ability_baseline_wave,
            }
        )
    if plan.dose_stage_covariate is not None:
        terms.append(
            {
                "term": plan.dose_stage_covariate,
                "kind": "flagged_collider_sensitivity",
                "source_column": plan.dose_stage_covariate,
                "wave": "pre",
            }
        )
    return {
        "requested": list(plan.ability_adjust_symbols),
        "fitted": terms,
        "exposure": plan.exposure,
        "dropped_constant": list(getattr(prepared, "dropped_covariates", ())),
    }


@dataclass(frozen=True)
class DoseContrast:
    """A support-respecting items-scale dose contrast and the evidence for it."""

    delta_std: np.ndarray
    delta_sessions: np.ndarray
    label: str
    note: str
    support_table: pd.DataFrame


def resolve_dose_contrast(payload, phase_idx: np.ndarray) -> DoseContrast:
    """Within-period interquartile contrast over observed treated attendance (#587 finding 3).

    The pre-#587 headline added **one global SD — 30.7 sessions — to every fitted
    row**, including the 25 period-1 waitlist rows that attended none (no treated
    period-1 child attended fewer than 45) and 84 of 156 rows whose shifted value
    exceeded their own period's observed maximum. The inverse-logit arithmetic was
    right; the reported item magnitude was mostly extrapolation, and no
    posterior-predictive check at observed doses validated it.

    The replacement moves each treated row from its **own period's** observed lower
    quartile of sessions to that period's upper quartile. Every endpoint is an
    observed attendance level in the period it is applied to, so the contrast is
    inside support by construction, and the raw-session width is recorded per period
    rather than left for the reader to infer from a standardised coefficient.
    """
    treated = np.asarray(payload.treated, dtype=bool)
    sd = float(payload.dose_scaler.sd)
    delta_sessions = np.zeros(treated.shape[0], dtype=float)
    records: list[dict[str, object]] = []
    for phase, (lo, q1, q3, hi) in enumerate(payload.phase_support):
        rows = treated & (phase_idx == phase)
        width = float(q3 - q1) if np.isfinite(q3) and np.isfinite(q1) else 0.0
        delta_sessions[rows] = width
        records.append(
            {
                "period": phase + 1,
                "n_treated_rows": int(rows.sum()),
                "sessions_min": lo,
                "sessions_q1": q1,
                "sessions_q3": q3,
                "sessions_max": hi,
                "contrast_sessions": width,
                "contrast_within_support": bool(
                    np.isfinite(hi) and np.isfinite(lo) and q1 >= lo and q3 <= hi
                ),
            }
        )
    label = "an interquartile increase in that period's observed session attendance"
    widths = [r["contrast_sessions"] for r in records if r["n_treated_rows"]]
    note = (
        "Each on-intervention row is moved from its own period's observed lower "
        "quartile of sessions to that period's upper quartile ("
        + ", ".join(
            f"period {r['period']}: {r['sessions_q1']:.0f}→{r['sessions_q3']:.0f}"
            for r in records
            if r["n_treated_rows"]
        )
        + f" sessions; median width {float(np.median(widths)) if widths else 0.0:.0f}). "
        "Both endpoints are observed attendance levels in the period they are applied "
        "to, so no profile is extrapolated beyond the fitted dose support. Untreated "
        "rows are excluded: a session step where there is no intervention is not a "
        "counterfactual this design supports."
    )
    return DoseContrast(
        delta_std=delta_sessions / sd,
        delta_sessions=delta_sessions,
        label=label,
        note=note,
        support_table=pd.DataFrame(records),
    )


def write_dose_band_ppc(ctx: StatisticalFitContext, *, payload) -> None:
    """Observed-vs-predicted calibration by arm x period x dose band (#587 finding 15).

    The family's existing checks are a marginal score-density overlay and an
    all-observation interval-coverage statistic. Neither can see the failure that
    would matter most here: a model that fits the overall score distribution while
    getting the *dose* gradient wrong within an arm and period. This cross-tabulates
    posterior-predictive coverage by assigned arm, period and terciles of observed
    treated attendance (with untreated rows as their own band), so a systematic
    miss at one end of the dose range is visible rather than averaged away.
    """
    pp = getattr(ctx.trace, "posterior_predictive", None)
    if pp is None or "y_post" not in pp:
        return
    draws = pp["y_post"].stack(sample=("chain", "draw")).transpose("obs_id", "sample").values
    observed = np.asarray(ctx.trace.observed_data["y_post"].values, dtype=float)
    phase_idx = np.asarray(ctx.prepared.phase, dtype=int)
    arm = np.asarray(ctx.prepared.G, dtype=int)
    treated = np.asarray(payload.treated, dtype=bool)
    raw = np.asarray(payload.raw_attend, dtype=float)

    band = np.full(raw.shape[0], "none", dtype=object)
    if treated.any():
        cuts = np.percentile(raw[treated], [100 / 3, 200 / 3])
        band[treated] = np.where(
            raw[treated] <= cuts[0], "low", np.where(raw[treated] <= cuts[1], "mid", "high")
        )
    lo, hi = np.percentile(draws, [5.5, 94.5], axis=1)
    inside = (observed >= lo) & (observed <= hi)
    median = np.median(draws, axis=1)

    records: list[dict[str, object]] = []
    for a in sorted(set(arm.tolist())):
        for p in range(int(phase_idx.max()) + 1):
            for b in ("none", "low", "mid", "high"):
                rows = (arm == a) & (phase_idx == p) & (band == b)
                n = int(rows.sum())
                if not n:
                    continue
                records.append(
                    {
                        "arm": "immediate" if a == 1 else "waitlist",
                        "period": p + 1,
                        "dose_band": b,
                        "n": n,
                        "sessions_mean": float(raw[rows].mean()),
                        "observed_mean": float(observed[rows].mean()),
                        "predicted_median_mean": float(median[rows].mean()),
                        "mean_residual": float((observed[rows] - median[rows]).mean()),
                        "coverage_89": float(inside[rows].mean()),
                    }
                )
    save_table(ctx, "dose_band_calibration", pd.DataFrame(records))


def dose_marginal_draws(
    group,
    *,
    phase_idx: np.ndarray,
    delta_std: np.ndarray,
    n_trials: int,
    period_varying: bool,
    row_mask: np.ndarray | None = None,
    eta_name: str = "eta",
    score_mean_link: str = "logit",
) -> np.ndarray:
    """The family's items-scale dose contrast, from one draws group (#587 finding 5).

    **One transform, used by both paths.** The posterior marginal and the matching
    prior pushforward must be the same function of the same rows, phase slopes,
    contrast and denominator, or the "check" compares two different quantities. They
    were not: the posterior indexed ``beta_dose_phase`` by each row's phase while the
    prior path broadcast the scalar ``mu_dose`` to every row, dropping ``sigma_dose``
    and the phase deviations entirely — a mean absolute discrepancy of 1.3 items per
    prior draw on the stored word-reading fit. Both now call this.

    ``group`` is a ``prior`` or ``posterior`` xarray group carrying ``eta_name`` and
    the slope variables. ``delta_std`` is the per-row step in units of the fitted
    (treated-row) standardised dose; ``row_mask`` restricts the averaging population.
    Returns one items-scale value per draw.
    """
    eta = (
        group[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )
    if period_varying:
        stacked = group["beta_dose_phase"].stack(sample=("chain", "draw"))
        phase_dim = next(d for d in stacked.dims if d != "sample")
        slopes = stacked.transpose(phase_dim, "sample").values
        per_row = slopes[np.asarray(phase_idx, dtype=int)]
    else:
        scalar = group["beta_dose"].stack(sample=("chain", "draw")).values.ravel()
        per_row = np.broadcast_to(scalar[None, :], eta.shape)
    delta_eta = per_row * np.asarray(delta_std, dtype=float)[:, None]
    if row_mask is not None:
        keep = np.asarray(row_mask)
        if keep.ndim != 1 or keep.dtype != bool or keep.shape[0] != eta.shape[0]:
            raise ValueError(
                f"row_mask must be a 1-D boolean mask with {eta.shape[0]} entries; "
                f"got dtype {keep.dtype}, shape {keep.shape}."
            )
        if not keep.any():
            raise ValueError("row_mask selects no rows for the dose marginal.")
        eta, delta_eta = eta[keep], delta_eta[keep]
    # Map both operating points through the fitted score mean before differencing.
    # Under a non-identity link the response-scale change is not the logit-scale one
    # rescaled, so an ordinary-logit transform on a floor-link posterior would report
    # a dose marginal the likelihood never modelled (#619). Because this one helper
    # serves both the posterior marginal and its prior pushforward, passing the link
    # here keeps that pair on the same scale by construction.
    return (
        apply_score_mean_link(expit(eta + delta_eta), score_mean_link)
        - apply_score_mean_link(expit(eta), score_mean_link)
    ).mean(axis=0) * float(n_trials)


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
    between_term: str | None = None,
    include_presence_term: bool = False,
    contrast_std: np.ndarray | None = None,
    contrast_sessions: np.ndarray | None = None,
    contrast_label: str = "a +1 SD session-dose step",
    contrast_kind: str = "one_sd_all_rows",
    support_note: str = "",
    score_mean_link: str = "logit",
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
    fitted rows. Both families now pass their treated-row mask: the dose is
    treated-centred with untreated rows contributing exactly zero, so a dose step
    on an untreated row is not a supported counterfactual of either design and the
    coherent averaging population for an intensive-margin estimand is the treated
    rows. (Before #587 the dose_response family averaged over *all* rows and
    stepped every one by one global SD — 30.7 sessions, on a scale inflated by the
    period-1 structural zeros. 84 of 156 shifted rows then sat above their own
    period's observed maximum, and the 25 period-1 waitlist rows were shifted from
    0 to 30.7 sessions although no treated period-1 child attended fewer than 45.)

    ``contrast_std`` is the per-row step in units of the fitted standardised dose;
    ``contrast_sessions`` the same step in raw sessions, for the self-describing
    columns. The default (``None``) is a +1 SD step on every retained row, which is
    what the DiD dose companions report. ``between_term`` names a between-child
    slope to summarise alongside the period slopes when the exposure is split.
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
    if between_term is not None and between_term in post:
        # The Mundlak between-child component (#587 finding 2). Reported beside the
        # period slopes and never merged into them: "children who attended more" and
        # "a period when this child attended more" are different questions, and a lone
        # coefficient over a random intercept silently returns a blend of the two.
        rows.append(
            {"term": "dose_between_child", **_summarise_draws(_draws(between_term), ci_prob)}
        )
    if include_presence_term and "theta_treated" in post:
        # The extensive margin, kept in the same table so a reader cannot mistake a
        # dose slope for the effect of being on the intervention at all. Opt-in so the
        # DiD dose companions, which have their own ``theta_treated`` and their own
        # audit in #576, keep their existing table unchanged.
        rows.append(
            {"term": "on_intervention", **_summarise_draws(_draws("theta_treated"), ci_prob)}
        )

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

    # Natural-scale average marginal association for the key-findings box (#320),
    # computed through the one shared transform the prior pushforward also uses.
    outcome = ctx.spec.outcome_symbol or "W"
    n_trials = int(ctx.prepared.n_trials[outcome])
    phase_idx = np.asarray(ctx.prepared.phase, dtype=int)
    n_rows_total = int(phase_idx.shape[0])
    focal_term_name = "mu_dose" if period_varying else "beta_dose"
    if contrast_std is None:
        contrast_std = np.ones(n_rows_total, dtype=float)
    contrast_std = np.asarray(contrast_std, dtype=float)
    if contrast_std.shape != (n_rows_total,):
        raise ValueError(
            f"contrast_std must have {n_rows_total} entries; got {contrast_std.shape}"
        )
    items = dose_marginal_draws(
        post,
        phase_idx=phase_idx,
        delta_std=contrast_std,
        n_trials=n_trials,
        period_varying=period_varying,
        row_mask=marginal_row_mask,
        score_mean_link=score_mean_link,
    )
    lo_q = (1 - ci_prob) / 2
    kept = (
        np.ones(n_rows_total, dtype=bool)
        if marginal_row_mask is None
        else np.asarray(marginal_row_mask)
    )
    sessions = (
        np.full(n_rows_total, float(dose_scaler.sd))
        if contrast_sessions is None
        else np.asarray(contrast_sessions, dtype=float)
    )
    marginal = pd.DataFrame(
        [
            {
                "items_median": float(np.median(items)),
                "items_lo": float(np.quantile(items, lo_q)),
                "items_hi": float(np.quantile(items, 1 - lo_q)),
                "items_lo50": float(np.quantile(items, 0.25)),
                "items_hi50": float(np.quantile(items, 0.75)),
                "prob_pos": float(np.mean(items > 0)),
                # Self-describing contrast and averaging population, so a reader
                # never has to guess how big the step was or who it was averaged
                # over (#587 finding 3).
                "contrast_kind": contrast_kind,
                "contrast_label": contrast_label,
                "contrast_sessions_median": float(np.median(sessions[kept])),
                "contrast_sessions_min": float(sessions[kept].min()),
                "contrast_sessions_max": float(sessions[kept].max()),
                "n_rows": int(kept.sum()),
                "n_rows_fitted": n_rows_total,
                "row_population": (
                    "all fitted rows"
                    if marginal_row_mask is None
                    else "on-intervention (treated) rows only"
                ),
                "support_note": support_note,
                # The published estimand names itself where the family records one
                # (#576 finding 1), so a reader of the CSV never has to infer which
                # of the fit's several dose quantities is the headline.
                "focal_estimand": str(
                    getattr(getattr(ctx, "resolved_plan", None), "focal_estimand", "")
                ),
                "swept_coefficient": focal_term_name,
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
    # Estimand-scale prior check on the reported dose marginal (#381, repaired by
    # #587 finding 5). It runs through :func:`dose_marginal_draws` — the *same*
    # function, rows, phase-indexed slopes, contrast and denominator as the posterior
    # marginal above. The previous path called the generic scalar-term writer, which
    # broadcast ``mu_dose`` to every row and silently omitted ``sigma_dose`` and the
    # phase deviations, so the "matching" prior check was of a different quantity.
    focal = focal_term_name
    source = getattr(ctx, "prior_samples", None) or ctx.trace
    try:
        prior_group = require_prior_evidence(
            source,
            terms=(focal, "eta"),
            what="the dose-marginal prior check",
        )
    except PriorEvidenceUnavailable as exc:
        pushforward_rows = [
            _report.unavailable_pushforward(
                estimand=focal,
                estimand_label=(
                    f"the association of {contrast_label} with "
                    f"{pushforward_outcome_label(ctx, outcome)}"
                ),
                role="association",
                reason=str(exc),
            )
        ]
    else:
        # Past the availability check the transform must succeed or fail the fit
        # (#637 stage 1) — it is the same function, rows and denominator the
        # posterior marginal above uses.
        prior_items = dose_marginal_draws(
            prior_group,
            phase_idx=phase_idx,
            delta_std=contrast_std,
            n_trials=n_trials,
            period_varying=period_varying,
            row_mask=marginal_row_mask,
            score_mean_link=score_mean_link,
        )
        prior_logit = (
            prior_group[focal].stack(sample=("chain", "draw")).values.ravel()
        )
        pushforward_rows = [
            _report.labelled_pushforward(
                _report.pushforward_values(
                    prior_logit, prior_items, n_trials=n_trials, ci_prob=ci_prob
                ),
                estimand=focal,
                estimand_label=(
                    f"the association of {contrast_label} with "
                    f"{pushforward_outcome_label(ctx, outcome)}"
                ),
                role="association",
            )
        ]
    write_prior_pushforward(ctx, pushforward_rows)
