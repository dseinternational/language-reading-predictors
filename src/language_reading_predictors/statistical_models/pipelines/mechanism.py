# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mechanism orchestration (``kind="mechanism"``, LRP56 / LRP57 / LRP58).

``fit_mechanism`` relates one measure's period change to another measure's
period-start level across all phases, with subject random intercepts and either a
linear slope or an HSGP shape, plus optional linear moderation. The exposure →
outcome coupling is an *adjusted association* — the DAG adjustment set is
conditioned on and recorded, but the exposure is not randomised — so the fitted
curve, its items-scale translation and the readiness threshold are all read as
associations, never as "X drives Y".
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import COLOUR_BLUE, COLOUR_RED, FIGSIZE_LG
from rich import print as rprint
from scipy.special import expit

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    diagnostics as _diag,
    mechanism as _mechanism,
    reporting as _report,
)
from language_reading_predictors.statistical_models.adjustment import (
    effective_adjustment,
)
from language_reading_predictors.statistical_models.artifacts import (
    guard_optional,
    record_artifact,
    save_table,
)
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.figure_artifacts import (
    write_child_fit,
)
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.plotting import save_styled_figure
from language_reading_predictors.statistical_models.prior_artifacts import (
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


def _mechanism_run_plan(
    ctx: StatisticalFitContext,
) -> _mechanism.MechanismRunPlan:
    """Return the pre-IO plan attached by ``fit_mechanism`` or reconstruct it."""
    resolved = getattr(ctx, "resolved_plan", None)
    if isinstance(resolved, _mechanism.MechanismRunPlan):
        return _mechanism.validate_mechanism_run_plan(ctx.spec, resolved)
    return _mechanism.resolve_mechanism_run_plan(ctx.spec)


def fit_mechanism(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    require_spec(spec, "mechanism", mechanism=True)
    # Resolve and validate every mechanism setting before ``make_context`` resets an
    # output staging directory or the loader reads any data (#394 pillar 4).
    run_plan = _mechanism.resolve_mechanism_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = run_plan
    _report.write_model_recipe(ctx)
    # Some mechanism fits keep the HSGP curve and need a higher target_accept for
    # the residual boundary divergences (LRP58/71/158); honour it with the shared
    # CLI > model-specific > preset precedence.

    section_header("Prepare data")
    # Data preparation and construction live in the family-owned ``mechanism``
    # module (#438) so that a leave-one-out refit for ``reloo`` builds the *same*
    # model as this fit rather than a re-derived lookalike. Behaviour-preserving
    # relocation: the loader-argument derivation, confounder filtering and factory
    # keyword mapping moved verbatim.
    plan = _mechanism.resolve_mechanism_plan(spec, run_plan=run_plan)
    prepared = plan.prepared
    ctx.prepared = prepared
    adjust_for = plan.adjust_for
    confounders = list(plan.confounders)
    moderator_symbol = run_plan.moderator_symbol
    mechanism_is_covariate = run_plan.mechanism_is_covariate

    print_header(ctx)

    section_header("Build model")
    built = _mechanism.build_mechanism_for_plan(plan)
    attach_built(ctx, built)

    render_model_graph(ctx)

    _mech_vars = _mechanism.mechanism_diagnostic_vars(plan)
    # Power-scaling prior sensitivity on the reported parameters (#381). For the
    # HSGP mechanism curve the estimand is the shape, governed by the deliberately
    # tight ``eta_main_prior`` amplitude the prior review flagged; the linear slope
    # ``beta_mech`` is already in ``_mech_vars``, so add the GP amplitude and
    # lengthscale only when the nonparametric curve is fitted.
    _mech_psense_vars = list(_mech_vars)
    if not run_plan.linear_mechanism:
        _mech_psense_vars += ["f_mech__eta", "f_mech__ell"]
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(_mech_vars),
            ppc_var_names=(run_plan.observation_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_plot(
                c, spec.outcome_symbol or "W"
            ),
            psense_vars=tuple(_mech_psense_vars),
            # The curve and interaction summaries precede persistence by design.
            save_trace=False,
        ),
    )

    # Mechanism curve: f_mech vs mech_post_logit grid (logit-contribution scale only).
    section_header("Mechanism curve")
    _write_mechanism_curve(ctx)
    # Items-scale companion (#319): the same curve as exposure items -> predicted
    # outcome items, with a computed worked-example contrast. The worked dict is
    # folded into config.json below so the report partial renders the caption from
    # computed numbers.
    _items_worked = _write_mechanism_items(ctx)
    _write_mechanism_slope_summary(ctx)
    _write_dispersion_summary(ctx)
    _write_readiness_threshold(ctx)
    _write_exposure_support(ctx)

    # Record the adjustment set that was actually FITTED — with each term's source
    # column, measurement wave and missing-indicator status — not just the requested
    # symbols. ``spec.adjustment`` alone materially misdescribed the model, because
    # the ``adjust_for`` covariates never reached config.json (#258 review, P1).
    meta_extra = {
        "loo_elpd": float(ctx.loo.elpd),
        "adjustment": spec.adjustment,
        "effective_adjustment": effective_adjustment(
            spec,
            prepared,
            measure_confounders=tuple(
                s for s in confounders if s in ("G", "A") or s in MEASURES
            ),
            adjust_for=adjust_for,
            # The typed ability adjuster is declared apart from ``adjust_for`` (it
            # loads from t1 via ``baseline_covariates``) but is fitted as an ordinary
            # standardised covariate, so it belongs in the *requested* list too —
            # otherwise the record shows it fitted but never asked for.
            requested_adjust_for=run_plan.adjust_for
            + ((run_plan.ability_covariate,) if run_plan.ability_covariate else ()),
            baseline_symbol=run_plan.adjust_baseline_symbol,
            # The fitted moderation terms. ``gamma_mod`` carries a coefficient on
            # every moderated fit, and for age moderation it *is* the age adjustment
            # (the factory drops the separate linear ``gamma_A`` as collinear), so
            # omitting it left a coefficient-bearing conditioning term unnamed
            # (#586 finding 9).
            moderator_symbol=run_plan.moderator_symbol,
            moderator_is_covariate=run_plan.moderator_is_covariate,
            moderator_interaction=(
                run_plan.moderator_symbol is not None and run_plan.include_interaction
            ),
            # The #603 / #604 focal exposure terms. They are not adjusters, but they
            # do carry coefficients, and the fitted record exists to name every
            # coefficient a fit estimates (#586 finding 9) — so they are recorded
            # under their own kinds rather than left out or relabelled.
            exposure_terms=_exposure_term_records(run_plan, prepared),
        ),
    }
    # Items-scale worked-example reference points (#319): recorded so the caption
    # numbers are computed, not hand-written, and the quantiles are auditable.
    if _items_worked:
        meta_extra["mechanism_items"] = _items_worked
    # The exposure unit, on the rows this fit actually used. ``beta_mech`` is "per SD
    # of the exposure", and the factory re-standardises on its own kept rows — so two
    # mechanism fits of the same exposure on different outcomes report slopes in
    # *different raw increments*. Any cross-model contrast that treats them as
    # commensurate has to be able to check that, and could not before (2026-08-23
    # joint-mechanism follow-up review, finding 2).
    _exposure_sd = _mechanism_exposure_logit_sd(ctx, run_plan)
    if _exposure_sd is not None:
        meta_extra["exposure_logit_sd"] = _exposure_sd
    if mechanism_is_covariate:
        # Record the exposure's raw-units anchor so a report can translate the
        # per-SD ``beta_mech`` into raw score points: the factory re-standardises
        # the loader z on the kept rows, so +1 SD of the fitted exposure is
        # ``loader_sd * sd(z_kept)`` raw points.
        meta_extra["mechanism_is_covariate"] = True
        _sc = ctx.prepared.covariate_scalers.get(spec.mechanism_symbol)
        if _sc is not None:
            _z_kept = np.asarray(
                ctx.prepared.covariates[spec.mechanism_symbol], dtype=float
            )
            meta_extra["mechanism_exposure_sd_raw"] = float(
                _sc.sd * np.nanstd(_z_kept, ddof=1)
            )
            meta_extra["mechanism_exposure_mean_raw"] = float(
                _sc.mean + _sc.sd * np.nanmean(_z_kept)
            )

    # Linear-moderation summary (gamma_int / gamma_mod), when a moderator is set.
    if moderator_symbol is not None:
        section_header("Interaction summary")
        gi = _report.gamma_interaction_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
        gi_df = pd.DataFrame([gi])
        save_table(ctx, "interaction_summary", gi_df)
        print_table(
            metrics_table(
                [{"metric": k, "value": v} for k, v in gi.items()],
                title=(
                    f"Linear moderation by {moderator_symbol} "
                    f"- {int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
                ),
                columns=["metric", "value"],
            )
        )
        meta_extra["moderator_symbol"] = moderator_symbol
        meta_extra["interaction_summary"] = gi
        # Words-scale re-expression of the interaction (2026-08-19): the sign of
        # ``gamma_int`` on a bounded outcome is not a statement about items, so
        # the interquartile-cell contrast in items, with its logit-additive
        # benchmark, is published alongside it.
        write_moderation_items(ctx)

    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=_mech_vars)

    # Per-child fitted-vs-observed panels (#317 fig 2), one per period transition.
    write_child_fit(
        ctx,
        outcome_symbol=spec.outcome_symbol,
        wave=ctx.prepared.phase,
        child_idx=ctx.prepared.child_idx,
        off_floor=False,
        obs_node=run_plan.observation_node,
        x_label="period transition",
    )

    write_run_metadata(ctx, extra=meta_extra)

    return finalize_report(ctx)


def _exposure_term_records(
    run_plan: _mechanism.MechanismRunPlan, prepared
) -> tuple[dict, ...]:
    """Fitted-record entries for the family's focal exposure coefficients.

    Empty for a pooled design, whose single ``beta_mech`` (or ``f_mech``) is already
    described by ``resolved_run_plan``. The sensitivities replace it with several
    coefficients answering different questions, so each is named with the question it
    answers and an explicit non-adjuster ``kind``.
    """
    symbol = run_plan.mechanism_symbol
    if run_plan.mechanism_is_covariate:
        source = symbol
        scale = "standardised raw covariate"
        wave = prepared.covariate_time.get(symbol, "unknown")
    else:
        source = prepared.column_map.get(symbol, symbol)
        scale = "standardised logit of the exposure count"
        wave = "pre" if run_plan.mechanism_at_pre else "post"
    common = {
        "exposure": symbol,
        "source_column": source,
        "wave": wave,
        "missing_indicator": False,
    }
    records: list[dict] = []
    if run_plan.decompose_between_within:
        records.append(
            {
                "term": "beta_between",
                "kind": "exposure_between_child",
                "scale": f"child fitted-row mean of the {scale}",
                "question": "do children whose average exposure is higher score "
                "higher on the outcome?",
                **common,
            }
        )
        if not run_plan.phase_varying_slope:
            records.append(
                {
                    "term": "beta_within",
                    "kind": "exposure_within_child",
                    "scale": f"deviation from the child's mean of the {scale}",
                    "question": "when a child's own exposure moves, does their "
                    "outcome move with it?",
                    **common,
                }
            )
    if run_plan.phase_varying_slope:
        records.append(
            {
                "term": "beta_mech_phase",
                "kind": "exposure_phase_varying_slope",
                "scale": (
                    f"per-period slope on the deviation from the child's mean of the {scale}"
                    if run_plan.decompose_between_within
                    else f"per-period slope on the {scale}"
                ),
                "question": "does the exposure-outcome slope differ across the "
                "three period transitions?",
                **common,
            }
        )
    return tuple(records)


def _mechanism_exposure_logit_sd(
    ctx: StatisticalFitContext, run_plan: _mechanism.MechanismRunPlan
) -> float | None:
    """SD of the exposure logit on this fit's rows, or ``None`` when undefined.

    Bounded-count exposures only: a covariate exposure is already standardised by the
    loader and records its raw anchor separately, and a lagged (``mechanism_at_pre``)
    exposure reads a different column. Returning ``None`` rather than a wrong number
    keeps the cross-model comparison honest about what it can and cannot prove.
    """
    if run_plan.mechanism_is_covariate or run_plan.mechanism_at_pre:
        return None
    symbol = run_plan.mechanism_symbol
    prepared = ctx.prepared
    if prepared is None or symbol not in prepared.post_counts:
        return None
    from language_reading_predictors.statistical_models.preprocessing import logit_safe

    values = np.asarray(
        logit_safe(prepared.post_counts[symbol], prepared.n_trials[symbol]),
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    return float(np.std(values, ddof=1))


def _write_exposure_support(ctx: StatisticalFitContext) -> None:
    """Fitted exposure support by period and randomised arm (#586 finding 2).

    A pooled exposure range hides whether the range is populated the same way in
    every period and both arms. mech-191's did not: its lowest sessions values were
    the entire period-1 waitlist arm, so the bottom of a "0 to 94 sessions" curve was
    an arm-and-period contrast, not a dose one — and no published artefact showed it.
    One row per (phase, arm) cell with the count and the exposure quantiles, so
    structural non-overlap is visible rather than inferred.
    """
    run_plan = _mechanism_run_plan(ctx)
    sym = run_plan.mechanism_symbol
    prepared = ctx.prepared
    if run_plan.mechanism_is_covariate:
        scaler = prepared.covariate_scalers.get(sym)
        z = np.asarray(prepared.covariates[sym], dtype=float)
        values = scaler.inverse(z) if scaler is not None else z
        unit = f"{sym} raw score"
    elif sym in prepared.post_counts:
        values = np.asarray(prepared.post_counts[sym], dtype=float)
        unit = f"{sym} items"
    else:  # pragma: no cover - the factory keep-mask guarantees one of the above
        return

    arm_label = {0: "wait-list", 1: "immediate"}
    rows = []
    for phase in sorted(set(int(p) for p in prepared.phase)):
        for arm in sorted(set(int(g) for g in prepared.G)):
            cell = values[(prepared.phase == phase) & (prepared.G == arm)]
            if not cell.size:
                continue
            rows.append(
                {
                    "phase": phase,
                    "period": f"t{phase + 1}->t{phase + 2}",
                    "arm": arm_label.get(arm, str(arm)),
                    "exposure_unit": unit,
                    "n_rows": int(cell.size),
                    "n_at_zero": int((cell <= 0).sum()),
                    "min": float(np.min(cell)),
                    "q25": float(np.quantile(cell, 0.25)),
                    "median": float(np.median(cell)),
                    "q75": float(np.quantile(cell, 0.75)),
                    "max": float(np.max(cell)),
                }
            )
    if rows:
        save_table(ctx, "exposure_support", pd.DataFrame(rows), register=False)


def _mechanism_exposure_axis(ctx: StatisticalFitContext) -> tuple[np.ndarray, str, str]:
    """The fitted rows' exposure values, and the axis column/label for the logit view.

    A covariate exposure is plotted in its own raw score; a bounded-count exposure
    on its logit-safe scale, taken from the period-start column under
    ``mechanism_at_pre`` because that is the vector the factory fitted (plotting the
    post logit there would label the fitted pre-slope against the wrong exposure,
    #405 review).
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import logit_safe

    sym = ctx.spec.mechanism_symbol
    run_plan = _mechanism_run_plan(ctx)
    if run_plan.mechanism_is_covariate:
        z_loaded = np.asarray(ctx.prepared.covariates[sym], dtype=float)
        scaler = ctx.prepared.covariate_scalers.get(sym)
        x_vals = scaler.inverse(z_loaded) if scaler is not None else z_loaded
        return x_vals, "mech_x", f"{sym} (raw score)"
    if run_plan.mechanism_at_pre:
        x_vals = np.asarray(ctx.prepared.pre_logit[sym], dtype=float)
        return x_vals, "mech_logit", f"logit({sym}_pre)"
    x_vals = np.asarray(
        logit_safe(ctx.prepared.post_counts[sym], MEASURES[sym].n_trials), dtype=float
    )
    return x_vals, "mech_logit", f"logit({sym}_post)"


def _write_mechanism_curve(ctx: StatisticalFitContext) -> None:
    """The analyst's logit-scale view of the fitted exposure term.

    One row per distinct exposure value on the fitted rows, carrying the mechanism
    term's contribution to the outcome logit **standardised over those rows** — the
    same reference population as the family's declared natural-scale estimand
    (#602), so the two views describe one quantity on two scales rather than two
    quantities. For the HSGP curve and the pooled linear slope the contribution
    depends on the exposure alone, so the standardisation is a no-op and this is the
    familiar band; for the between/within split (#603) and the per-period slopes
    (#604) the contribution is row-specific and the average over the fitted rows is
    what makes a single plotted curve well defined.

    The x column is ``mech_logit`` for a bounded-count measure exposure and
    ``mech_x`` (the raw covariate score) for a covariate exposure
    (``mechanism_is_covariate``). Guarded by the caller.
    """
    from language_reading_predictors.statistical_models.mechanism_items import (
        resolve_mechanism_terms,
    )

    sym = ctx.spec.mechanism_symbol
    x_vals, x_col, x_label = _mechanism_exposure_axis(ctx)

    try:
        terms = resolve_mechanism_terms(
            ctx.trace, x_exposure=x_vals, exposure_n_trials=None, group="posterior"
        )
    except KeyError as exc:
        # No recognisable exposure term in the posterior — e.g. a
        # ``phase_specific_mechanism`` fit, whose per-phase curve is registered under
        # neither name. Warn loudly rather than no-op (issue #273).
        rprint(
            f"[yellow]_write_mechanism_curve: {exc}; no mechanism_curve.csv/plot "
            f"written for {ctx.spec.model_id}.[/yellow]"
        )
        return

    # ``x_vals`` is already the fitted regressor's own scale here (the logit, or the
    # raw covariate score), so the exposure -> contribution map is exact on the
    # observed grid and needs no transform.
    xs = np.unique(x_vals)
    f_ord = np.stack(
        [
            np.broadcast_to(
                terms.contribution_at(float(v)),
                (x_vals.size, terms.fitted.shape[1]),
            ).mean(axis=0)
            for v in xs
        ]
    )  # (U, S)
    save_table(
        ctx,
        "mechanism_curve",
        pd.DataFrame(
            {
                x_col: xs,
                "f_mean": f_ord.mean(axis=1),
                "f_lo": np.quantile(f_ord, 0.055, axis=1),
                "f_hi": np.quantile(f_ord, 0.945, axis=1),
                "f_lo50": np.quantile(f_ord, 0.25, axis=1),
                "f_hi50": np.quantile(f_ord, 0.75, axis=1),
            }
        ),
        register=False,
    )
    outcome = ctx.spec.outcome_symbol or "W"
    plt.figure(figsize=FIGSIZE_LG)
    plt.plot(xs, f_ord.mean(axis=1), color=COLOUR_BLUE, lw=2)
    plt.fill_between(
        xs,
        np.quantile(f_ord, 0.055, axis=1),
        np.quantile(f_ord, 0.945, axis=1),
        color=COLOUR_BLUE,
        alpha=0.2,
    )
    plt.xlabel(x_label)
    plt.ylabel("predictor logit contribution")
    plt.title(f"Mechanism curve ({terms.kind}): {sym} -> {outcome}")
    # mechanism_curve.csv (the plotted band) is written just above.
    save_styled_figure(ctx.output_dir, "mechanism_curve")


def _write_dispersion_summary(ctx: StatisticalFitContext) -> pd.DataFrame | None:
    """``dispersion_summary.csv``: the concentration posterior and its implied VIF.

    The Beta-Binomial concentration is a nuisance parameter only if the prior leaves
    the ordinary hypothesis "no extra-Binomial variation" available. On a
    high-denominator outcome the family's shared ``kappa ~ HalfNormal(50)`` does not:
    with ``alpha + beta = kappa`` the variance inflation over Binomial is

        VIF = (kappa + n) / (kappa + 1) = 1 + (n - 1) / (kappa + 1)

    so being within 10% of Binomial variance needs ``kappa >= 10 (n - 1) - 1`` —
    779 at n = 79 and 1689 at n = 170 — and ``HalfNormal(50)`` gives that region
    vanishing mass (#605). This table publishes the fitted concentration beside the
    prior it was drawn against, the implied variance inflation, and the posterior
    probability of being within 10% of Binomial, so a reader can see whether the data
    moved ``kappa`` or the prior carried it.

    Written for every mechanism fit — the question is about the *registered* prior,
    so the default fits are exactly where the answer matters.
    """
    post = ctx.trace.posterior
    if "kappa" not in post:
        return None
    run_plan = _mechanism_run_plan(ctx)
    outcome = run_plan.outcome_symbol
    n_trials = int(MEASURES[outcome].n_trials)
    kappa = np.asarray(post["kappa"].values, dtype=float).ravel()
    inflation = (kappa + n_trials) / (kappa + 1.0)
    ci_prob = float(ctx.reporting.ci_prob)
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    family = run_plan.kappa_prior_family
    default_sigma = 0.25 if family == "halfnormal_inverse_sqrt" else 50.0
    sigma = default_sigma if run_plan.kappa_sigma is None else float(run_plan.kappa_sigma)
    row = {
        "outcome_symbol": outcome,
        "n_trials": n_trials,
        "kappa_prior_family": family,
        "kappa_prior_sigma": sigma,
        "kappa_prior_label": (
            f"1/sqrt(kappa) ~ HalfNormal({sigma:g})"
            if family == "halfnormal_inverse_sqrt"
            else f"kappa ~ HalfNormal({sigma:g})"
        ),
        "reaches_near_binomial": family == "halfnormal_inverse_sqrt",
        "kappa_median": float(np.median(kappa)),
        "kappa_lo": float(np.quantile(kappa, lo_q)),
        "kappa_hi": float(np.quantile(kappa, hi_q)),
        "variance_inflation_median": float(np.median(inflation)),
        "variance_inflation_lo": float(np.quantile(inflation, lo_q)),
        "variance_inflation_hi": float(np.quantile(inflation, hi_q)),
        # The concentration a near-Binomial fit would need at this denominator.
        "kappa_for_10pct_of_binomial": 10.0 * (n_trials - 1) - 1.0,
        "prob_within_10pct_of_binomial": float(np.mean(inflation <= 1.1)),
        "ci_prob": ci_prob,
        "n_obs": int(ctx.prepared.n_obs),
    }
    frame = pd.DataFrame([row])
    save_table(ctx, "dispersion_summary", frame, required=False)
    return frame


def _write_mechanism_slope_summary(ctx: StatisticalFitContext) -> pd.DataFrame | None:
    """Per-question exposure-slope table for the #603 / #604 sensitivities.

    The pooled designs have one exposure coefficient and ``diagnostics.csv`` already
    carries it. The two sensitivities have several, and the whole point of each is
    that the several answer *different questions* — so the table names the question
    beside every coefficient rather than leaving a reader to map ``beta_between``
    onto "do children who generally know more letter sounds generally read more
    words?" themselves. Returns ``None`` (writing nothing) for a pooled design.

    Every row is an adjusted association. The within-child coefficient removes
    stable between-child confounding, including the stable part of latent general
    ability; it does not make the exposure temporally prior to the outcome, and it
    does not remove time-varying confounding or reverse causation. A difference
    between period slopes is evidence against pooling, not evidence that the
    relationship changed over time.
    """
    run_plan = _mechanism_run_plan(ctx)
    if not (run_plan.decompose_between_within or run_plan.phase_varying_slope):
        return None
    post = ctx.trace.posterior
    ci_prob = float(ctx.reporting.ci_prob)
    exposure = run_plan.mechanism_symbol
    outcome = run_plan.outcome_symbol
    unit = (
        f"{outcome} logit per 1 SD of the standardised {exposure} exposure"
        if run_plan.mechanism_is_covariate
        else f"{outcome} logit per 1 SD of the {exposure} logit"
    )
    rows: list[dict] = []

    def add(name: str, draws, question: str, causal_note: str, **extra) -> None:
        row = _report.coef_row(name, draws, ci_prob)
        row.update(
            question=question,
            causal_status="adjusted association",
            causal_note=causal_note,
            unit=unit,
            exposure_symbol=exposure,
            outcome_symbol=outcome,
            n_obs=int(ctx.prepared.n_obs),
            ci_prob=ci_prob,
            **extra,
        )
        rows.append(row)

    if run_plan.decompose_between_within and "beta_between" in post:
        add(
            "beta_between",
            post["beta_between"].values,
            "Between children: do children whose study-average exposure is higher "
            "also tend to score higher on the outcome?",
            "Cross-sectional between-child comparison; confounded by every stable "
            "child characteristic, latent general ability included.",
            component="between",
        )
    if run_plan.phase_varying_slope and "beta_mech_phase" in post:
        if "mu_mech" in post:
            add(
                "mu_mech",
                post["mu_mech"].values,
                "Shared mean of the per-period slopes: the pooled exposure "
                "association the per-period slopes are shrunk toward.",
                "Partially-pooled mean of three adjusted associations; only the "
                "first period is randomised-arm-clean, and the exposure is not "
                "randomised in any of them.",
                component="phase_mean",
            )
        if "sigma_mech_phase" in post:
            add(
                "sigma_mech_phase",
                post["sigma_mech_phase"].values,
                "How much the exposure slope varies between periods — the "
                "quantity that answers whether pooling was safe.",
                "A scale, not an association; a large value is evidence against "
                "pooling, not evidence the mechanism changed over time.",
                component="phase_scale",
            )
        per_phase = post["beta_mech_phase"]
        phase_dim = next(d for d in per_phase.dims if d not in ("chain", "draw"))
        for index in range(per_phase.sizes[phase_dim]):
            add(
                f"beta_mech_phase[{index}]",
                per_phase.isel({phase_dim: index}).values,
                f"Within period t{index + 1}->t{index + 2}"
                + (
                    " (the randomised transition)"
                    if index == 0
                    else " (post-crossover: both arms on the intervention)"
                )
                + ": how the outcome tracks the exposure in that period alone.",
                "Adjusted association. The exposure is not randomised in any "
                "period; from period 2 both arms are on the intervention, so a "
                "period difference also differs in age, treatment history and "
                "measurement position.",
                component="phase_slope",
                phase=index,
                period=f"t{index + 1}->t{index + 2}",
            )
    elif run_plan.decompose_between_within and "beta_within" in post:
        add(
            "beta_within",
            post["beta_within"].values,
            "Within a child: when a child's own exposure moves away from their "
            "study average, does their outcome move with it?",
            "Removes stable between-child confounding, including the stable part "
            "of latent general ability. Exposure and outcome are still measured at "
            "the same wave, so this is not temporally ordered and does not rule "
            "out time-varying confounding or reverse causation.",
            component="within",
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    save_table(ctx, "mechanism_slope_summary", frame)
    print_table(
        metrics_table(
            [
                {
                    "coefficient": r["coefficient"],
                    "median": f"{r['median']:+.3f}",
                    f"{int(round(ci_prob * 100))}% CI": f"[{r['lo']:+.3f}, {r['hi']:+.3f}]",
                    "P(>0)": f"{r['prob_pos']:.3f}",
                }
                for r in rows
            ],
            title=f"Exposure slopes by question - {exposure} -> {outcome}",
            columns=["coefficient", "median", f"{int(round(ci_prob * 100))}% CI", "P(>0)"],
        )
    )
    return frame


#: Friendly labels for covariate mechanism exposures (no ``Measure`` entry, so no
#: label registry). Falls back to the symbol for anything not listed.
_COVARIATE_EXPOSURE_LABELS = {
    "erbto": "Phonological memory (word/nonword repetition)",
    "deapp_c": "Speech production (DEAP)",
}


def _mechanism_items_axis(ctx: StatisticalFitContext) -> tuple[np.ndarray, str, int | None]:
    """The fitted rows' exposure in natural units, its label and its item ceiling."""
    from language_reading_predictors.statistical_models.measures import MEASURES

    run_plan = _mechanism_run_plan(ctx)
    sym = ctx.spec.mechanism_symbol
    if run_plan.mechanism_is_covariate:
        z_loaded = np.asarray(ctx.prepared.covariates[sym], dtype=float)
        scaler = ctx.prepared.covariate_scalers.get(sym)
        x_exposure = scaler.inverse(z_loaded) if scaler is not None else z_loaded
        return x_exposure, _COVARIATE_EXPOSURE_LABELS.get(sym, sym), None
    if run_plan.mechanism_at_pre:
        # Lagged form: the fitted term (via z_mech_logit) is on the pre exposure, so
        # the items-scale x-axis must be the pre counts, not the post counts —
        # otherwise the worked-example quantiles land on the wrong distribution and
        # the axis is mislabelled (#405 review).
        return (
            np.asarray(ctx.prepared.pre_counts[sym], dtype=float),
            f"{MEASURES[sym].label} (period start)",
            MEASURES[sym].n_trials,
        )
    return (
        np.asarray(ctx.prepared.post_counts[sym], dtype=float),
        MEASURES[sym].label,
        MEASURES[sym].n_trials,
    )


def _write_mechanism_items(ctx: StatisticalFitContext) -> dict:
    """The family's declared natural-scale estimand: tables, figure, prior check.

    Writes ``mechanism_summary.csv`` — the headline interquartile contrast first,
    the secondary observed-range contrast second, each tagged with its
    machine-readable ``estimand`` — and ``mechanism_curve_items.csv`` plus the
    items-scale figure, all from **one** computation over the fitted rows (#602).
    Returns the ``worked`` dict so ``fit_mechanism`` can persist the reference
    points to ``config.json`` for the report partial.

    The summary table is the deliverable and is written outside the guard: only the
    figure and the prior pushforward are best-effort, because a plotting failure
    must not cost the headline number the key-findings box reads.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.mechanism_items import (
        mechanism_summary_table,
        write_mechanism_items_artifacts,
    )

    spec = ctx.spec
    run_plan = _mechanism_run_plan(ctx)
    sym = spec.mechanism_symbol
    outcome = spec.outcome_symbol or "W"
    is_covariate = run_plan.mechanism_is_covariate
    x_exposure, exposure_label, exposure_n_trials = _mechanism_items_axis(ctx)
    ref_quantiles = run_plan.items_ref_quantiles

    # The mechanism factory always fits a Beta-Binomial likelihood, so the y-axis is
    # an item count. Floored (off-floor Bernoulli) mechanism outcomes are a future
    # addition (#319 design note); wire the flag when such a model ships.
    worked = write_mechanism_items_artifacts(
        ctx.output_dir,
        ctx.trace,
        x_exposure=x_exposure,
        outcome_symbol=outcome,
        outcome_label=MEASURES[outcome].label,
        n_trials_outcome=MEASURES[outcome].n_trials,
        exposure_label=exposure_label,
        exposure_is_covariate=is_covariate,
        exposure_n_trials=exposure_n_trials,
        ci_prob=ctx.reporting.ci_prob,
        ref_quantiles=ref_quantiles,
        outcome_off_floor=False,
    )
    save_table(
        ctx,
        "mechanism_summary",
        mechanism_summary_table(
            worked,
            exposure_unit=(
                f"{sym} raw-score units" if is_covariate else f"{sym} items"
            ),
        ),
    )
    # ``mechanism_curve_items.csv`` is written inside the helper (which takes an
    # output directory, not a context); record it for the manifest.
    record_artifact(ctx, "mechanism_curve_items", required=False)
    with guard_optional(
        ctx,
        "mechanism prior pushforward",
        filename="prior_pushforward.csv",
        kind="table",
        verb="not written",
    ):
        _write_mechanism_prior_pushforward(
            ctx,
            x_exposure=x_exposure,
            outcome=outcome,
            exposure_label=exposure_label,
            exposure_n_trials=None if is_covariate else exposure_n_trials,
            ref_quantiles=ref_quantiles,
        )
    return worked


def _write_mechanism_prior_pushforward(
    ctx: StatisticalFitContext,
    *,
    x_exposure: np.ndarray,
    outcome: str,
    exposure_label: str,
    exposure_n_trials: int | None,
    ref_quantiles: tuple[float, float],
) -> None:
    """Estimand-scale prior check for the mechanism family (#381).

    The mechanism deliverable is a worked contrast — the predicted items-scale
    difference between two fixed quantiles of the observed exposure — so that,
    not a coefficient, is what the prior has to be pushed through. Runs
    :func:`mechanism_items.mechanism_items_curve` on the ``prior`` group, which
    reconstructs the HSGP ``f_mech`` curve or the linear ``beta_mech`` slope by
    the same route as the posterior version. This is the check the prior-analysis
    review asked for most directly: the GP amplitude prior is deliberately tight,
    and its implied curve range is what says whether a flat fitted curve is
    evidence of no dose-response or an artefact of the prior.

    Raises only for a defect (#637 stage 1). Absent prior evidence — no persisted
    prior group — is recorded as an ``unavailable`` row, because that is an honest
    absence; a resolution failure, a mismatched exposure vector or an unregistered
    outcome is not, and the previous blanket handler turned all three into a table
    reading "check unavailable" that release accepted on presence alone.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.mechanism_items import (
        mechanism_items_curve,
    )

    n_trials = MEASURES[outcome].n_trials
    q_lo = int(round(100 * ref_quantiles[0]))
    q_hi = int(round(100 * ref_quantiles[1]))
    label = (
        f"the predicted difference on {MEASURES[outcome].label} between the "
        f"{q_hi}th and {q_lo}th percentile of {exposure_label}"
    )
    source = getattr(ctx, "prior_samples", None) or ctx.trace
    try:
        require_prior_evidence(
            source, terms=("eta",), what="the mechanism-curve prior check"
        )
        _, worked = mechanism_items_curve(
            source,
            x_exposure=x_exposure,
            n_trials_outcome=n_trials,
            exposure_n_trials=exposure_n_trials,
            ci_prob=ctx.reporting.ci_prob,
            ref_quantiles=ref_quantiles,
            group="prior",
        )
        # ``logit_difference_*`` is the curve's rise between the two quantiles on
        # the linear-predictor scale and ``outcome_difference_*`` the same contrast
        # in items, so the two scales describe one quantity rather than two.
        rows = [
            _report.labelled_pushforward(
                {
                    "prior_logit_median": worked["logit_difference_median"],
                    "prior_logit_lo": worked["logit_difference_lo"],
                    "prior_logit_hi": worked["logit_difference_hi"],
                    "prior_items_median": worked["outcome_difference_median"],
                    "prior_items_lo50": worked["outcome_difference_lo50"],
                    "prior_items_hi50": worked["outcome_difference_hi50"],
                    "prior_items_lo": worked["outcome_difference_lo"],
                    "prior_items_hi": worked["outcome_difference_hi"],
                    "n_trials": int(n_trials),
                },
                estimand=f"mechanism_curve ({worked['curve_kind']})",
                estimand_label=label,
                role="association",
            )
        ]
    except PriorEvidenceUnavailable as exc:
        # Narrow by design (#637 stage 1). ``mechanism_items_curve`` resolves the
        # exposure term the model actually built, so a KeyError or a mismatched
        # exposure vector past this point is a defect in the curve, not absent
        # prior evidence, and must fail the fit.
        rows = [
            _report.unavailable_pushforward(
                estimand="mechanism_curve",
                estimand_label=label,
                role="association",
                reason=str(exc),
            )
        ]
    write_prior_pushforward(ctx, rows)


def _items_scale_knee(
    ctx: StatisticalFitContext, *, x_obs: np.ndarray, is_covariate: bool
) -> dict:
    """The steepest interval of the **expected-items** curve, ``items_``-prefixed.

    ``readiness_threshold`` locates the steepest interval of the *latent-logit*
    contribution. The expected-items derivative is
    ``d E[y] / dx = N * p (1 - p) * d eta / dx``, so its maximum can sit at a
    different exposure value — which is the question a reader who asks "where do
    predicted words rise fastest?" is actually asking (#602). It is computed here
    under the **declared reference population**, by binning the standardised
    items-scale curve rather than relabelling the logit statistic, and is published
    *alongside* the logit interval because the two answer different questions.

    Returns an empty dict, not a raised error, when it cannot be computed: the
    latent-logit interval is the established deliverable and must not be lost to a
    companion statistic.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.mechanism_items import (
        standardised_items_by_row,
    )

    outcome = ctx.spec.outcome_symbol or "W"
    x_exposure, _label, exposure_n_trials = _mechanism_items_axis(ctx)
    try:
        items_rows = standardised_items_by_row(
            ctx.trace,
            x_exposure=x_exposure,
            n_trials_outcome=MEASURES[outcome].n_trials,
            exposure_n_trials=None if is_covariate else exposure_n_trials,
        )
        items = _report.readiness_threshold(
            ctx.trace,
            exposure_values=np.asarray(x_obs, dtype=float),
            ci_prob=ctx.reporting.ci_prob,
            curve=items_rows,
            scale="expected_items",
        )
    except (KeyError, ValueError) as exc:
        rprint(
            f"[yellow]_items_scale_knee: {exc}; the items-scale steepest interval "
            "is omitted (the latent-logit one still stands).[/yellow]"
        )
        return {}
    keep = (
        "knee_count_median",
        "knee_count_ci_low",
        "knee_count_ci_high",
        "half_rise_count_median",
        "slope_below_knee_median",
        "slope_above_knee_median",
        "increasing_frac",
        "steepest_interval_index",
        "steepest_interval_share",
        "boundary_pinned",
        "prob_slope_above_gt_below",
        "knee_well_defined",
        "scale",
    )
    return {f"items_{k}": items[k] for k in keep if k in items}


def _write_readiness_threshold(ctx: StatisticalFitContext) -> None:
    """Readiness-threshold summary for the mechanism curve (#230 §2/§5).

    Post-processes the fitted nonparametric mechanism curve (``f_mech``) into a
    posterior for the predictor count at which the outcome rises *fastest* — the
    "knee" (the steepest rise, not the onset), via
    :func:`reporting.readiness_threshold`. Only the GP mechanism has a curve to
    find a knee in; linear / phase-specific fits (no ``f_mech``) are skipped
    quietly. Writes ``readiness_threshold.csv`` and a plot. Guarded by the
    caller.
    """
    post = ctx.trace.posterior
    if "f_mech" not in post:
        return

    from language_reading_predictors.statistical_models.measures import MEASURES

    sym = ctx.spec.mechanism_symbol
    outcome = ctx.spec.outcome_symbol or "W"
    is_covariate = _mechanism_run_plan(ctx).mechanism_is_covariate
    f = post["f_mech"].stack(sample=("chain", "draw")).values  # (n_obs, n_sample)

    if is_covariate:
        # Continuous-covariate exposure (e.g. LRP92 sessions): locate the knee in the
        # exposure's own raw units (scaler-inverted, as in _write_mechanism_curve),
        # not a bounded count. The per-obs exposure aligns with f_mech's row order.
        z_loaded = np.asarray(ctx.prepared.covariates[sym], dtype=float)
        scaler = ctx.prepared.covariate_scalers.get(sym)
        x_obs = scaler.inverse(z_loaded) if scaler is not None else z_loaded
        try:
            summary = _report.readiness_threshold(
                ctx.trace, exposure_values=x_obs, ci_prob=ctx.reporting.ci_prob
            )
        except ValueError as exc:
            rprint(f"[yellow]_write_readiness_threshold: {exc}; skipped.[/yellow]")
            return
        x_label = f"{sym} (raw score)"
    else:
        N = MEASURES[sym].n_trials
        try:
            summary = _report.readiness_threshold(
                ctx.trace, n_trials=N, ci_prob=ctx.reporting.ci_prob
            )
        except ValueError as exc:
            rprint(f"[yellow]_write_readiness_threshold: {exc}; skipped.[/yellow]")
            return
        # Mean curve on the raw count scale (inverse Haldane-corrected logit, as in
        # reporting._readiness_knee) with the knee posterior overlaid.
        ell = np.asarray(ctx.trace.constant_data["mech_post_logit"].values).reshape(-1)
        x_obs = np.clip((N + 1.0) / (1.0 + np.exp(-ell)) - 0.5, 0.0, float(N))
        x_label = f"{sym} (raw count, out of {N})"

    summary.update(_items_scale_knee(ctx, x_obs=x_obs, is_covariate=is_covariate))
    save_table(ctx, "readiness_threshold", pd.DataFrame([summary]), register=False)

    order = np.argsort(x_obs)
    x = x_obs[order]
    mean = f[order].mean(axis=1)
    plt.figure(figsize=FIGSIZE_LG)
    plt.plot(x, mean, color=COLOUR_BLUE, lw=2)
    plt.axvspan(
        summary["knee_count_ci_low"],
        summary["knee_count_ci_high"],
        color=COLOUR_RED,
        alpha=0.15,
        label=f"knee {int(round(ctx.reporting.ci_prob * 100))}% CI",
    )
    plt.axvline(
        summary["knee_count_median"], color=COLOUR_RED, lw=1.5, label="knee median"
    )
    plt.xlabel(x_label)
    plt.ylabel(f"{outcome} logit contribution")
    plt.title(f"Readiness threshold (steepest rise): {sym} -> {outcome}")
    plt.legend(fontsize=8)
    save_styled_figure(ctx.output_dir, "readiness_threshold")


# ---------------------------------------------------------------------------
# Words-scale moderation contrast (2026-08-19)
# ---------------------------------------------------------------------------

#: Quantiles of the fitted exposure and moderator values at which the words-scale
#: moderation cells are placed (the interquartile cells of the fitted rows).
MODERATION_ITEMS_QUANTILES = (0.25, 0.75)


def _exact_affine_map(values: np.ndarray, transform, z: np.ndarray, what: str):
    """Return ``x -> a * transform(x) + b`` with ``z == a * transform(values) + b``.

    The factory standardises the exposure logit and the moderator (logit or raw
    covariate) on the kept rows; the stored ``constant_data`` vectors are that
    standardisation. Recovering it as an exact affine map of the transformed
    natural-unit values — and refusing anything that is not exact — is the
    row-identity guard: a re-loaded frame whose rows or values differ from the
    fitted ones cannot silently produce a table. ``transform`` is the logit for a
    measure (``logit_safe`` with its item count bound) and the identity for a raw
    covariate.
    """
    t = np.asarray(transform(np.asarray(values, dtype=float)), dtype=float)
    z = np.asarray(z, dtype=float)
    if t.shape != z.shape:
        raise ValueError(f"{what}: {t.shape[0]} re-loaded rows vs {z.shape[0]} fitted")
    if np.ptp(t) == 0:
        raise ValueError(f"{what}: constant on the fitted rows; no cells to form")
    a, b = np.polyfit(t, z, 1)
    err = float(np.max(np.abs(a * t + b - z)))
    if err > 1e-6:
        raise ValueError(
            f"{what}: the re-loaded values are not the fitted standardised vector "
            f"(max deviation {err:.3g}); rows or data differ from the fit"
        )
    return lambda x: a * np.asarray(transform(np.asarray(x, dtype=float)), dtype=float) + b


def _cells(values: np.ndarray, *, snap: bool) -> tuple[float, float] | None:
    """Interquartile low/high cell values, snapped to observed values if asked.

    Falls back to the 10th/90th percentiles when the quartiles coincide (a
    heavily floored moderator), and returns ``None`` if even those coincide.
    """
    values = np.asarray(values, dtype=float)
    for q_lo, q_hi in (MODERATION_ITEMS_QUANTILES, (0.1, 0.9)):
        lo, hi = float(np.quantile(values, q_lo)), float(np.quantile(values, q_hi))
        if snap:
            observed = np.unique(values)
            lo = float(observed[np.argmin(np.abs(observed - lo))])
            hi = float(observed[np.argmin(np.abs(observed - hi))])
        else:
            lo, hi = round(lo, 1), round(hi, 1)
        if hi > lo:
            return lo, hi
    return None


def moderation_items_rows(
    post: Any,
    constant: Any,
    *,
    exposure_counts: np.ndarray,
    exposure_n_trials: int,
    moderator_values: np.ndarray,
    moderator_n_trials: int | None,
    outcome_n_trials: int,
    ci_prob: float,
    exposure_symbol: str,
    moderator_symbol: str,
    outcome_symbol: str,
    moderator_unit: str,
) -> list[dict]:
    """The moderated mechanism's interaction re-expressed on the outcome's items scale.

    ``gamma_int`` is a product term on the latent logit scale, and on a bounded
    outcome its sign is not a statement about items: below the midpoint of the
    scale the logit is concave, so two positive effects that are *additive in
    items* show a negative logit product, and logit-additivity is items-scale
    synergy. This table therefore evaluates the fitted surface in items at the
    interquartile cells of the fitted exposure and moderator values — every other
    term held at its fitted value for each row and averaged over rows, exactly as
    the family's end-to-end curve contrast does::

        E(x, m) = N_outcome * mean_rows expit(eta_base + f(x) + gamma_mod z(m)
                                                + gamma_int z(x) z(m))

    and reports (i) the four cell expectations, (ii) the exposure increment
    ``E(x_hi, m) - E(x_lo, m)`` at the low and at the high moderator cell, (iii)
    their difference — the items-scale interaction — (iv) the same difference
    with ``gamma_int`` set to zero, which is what logit-additivity would have
    shown in items (the bounded-scale benchmark), and (v) the logit-scale
    interaction over the same cells, ``gamma_int * dz(x) * dz(m)``. ``post`` and
    ``constant`` are the posterior and ``constant_data`` groups of the fit's
    trace; ``exposure_counts`` / ``moderator_values`` are the fitted rows'
    values in their natural units (counts for a measure, raw units for a
    covariate moderator, ``moderator_n_trials=None``), which must reproduce the
    stored standardised vectors exactly. Added 2026-08-19
    (``notes/202608182200-findings-by-question.md``, question 5 and section 8).
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
    )
    from language_reading_predictors.statistical_models.reporting import coef_row

    exposure_counts = np.asarray(exposure_counts, dtype=float)
    moderator_values = np.asarray(moderator_values, dtype=float)
    z_x_obs = np.asarray(constant["z_mech_logit"].values, dtype=float).reshape(-1)
    z_m_obs = np.asarray(constant["z_moderator"].values, dtype=float).reshape(-1)
    z_x = _exact_affine_map(
        exposure_counts,
        lambda v: logit_safe(v, exposure_n_trials),
        z_x_obs,
        "exposure",
    )
    if moderator_n_trials is None:
        z_m = _exact_affine_map(moderator_values, lambda v: v, z_m_obs, "moderator")
    else:
        z_m = _exact_affine_map(
            moderator_values,
            lambda v: logit_safe(v, moderator_n_trials),
            z_m_obs,
            "moderator",
        )

    x_cells = _cells(exposure_counts, snap=True)
    m_cells = _cells(moderator_values, snap=moderator_n_trials is not None)
    if x_cells is None or m_cells is None:
        return []
    x_lo, x_hi = x_cells
    m_lo, m_hi = m_cells

    eta = post["eta"].stack(sample=("chain", "draw")).transpose("obs_id", "sample").values
    if "f_mech" in post:
        f = post["f_mech"].stack(sample=("chain", "draw")).values
        f = np.asarray(f).reshape(eta.shape[0], -1)
        # The curve depends on the exposure only, so its value at a cell is its
        # value on any fitted row with that exposure count (cells are snapped
        # to observed counts above).
        def f_at(x: float) -> np.ndarray:
            return f[int(np.flatnonzero(exposure_counts == x)[0])]
    elif "beta_mech" in post:
        beta = post["beta_mech"].stack(sample=("chain", "draw")).values
        f = z_x_obs[:, None] * beta[None, :]

        def f_at(x: float) -> np.ndarray:
            return z_x(x) * beta
    else:
        raise KeyError("posterior has neither 'f_mech' nor 'beta_mech'")
    gamma_mod = post["gamma_mod"].stack(sample=("chain", "draw")).values
    gamma_int = post["gamma_int"].stack(sample=("chain", "draw")).values
    eta_base = (
        eta
        - f
        - z_m_obs[:, None] * gamma_mod[None, :]
        - (z_x_obs * z_m_obs)[:, None] * gamma_int[None, :]
    )

    def expected_items(x: float, m: float, g_int: np.ndarray) -> np.ndarray:
        eta_cell = (
            eta_base
            + f_at(x)[None, :]
            + gamma_mod[None, :] * z_m(m)
            + g_int[None, :] * (z_x(x) * z_m(m))
        )
        return expit(eta_cell).mean(axis=0) * float(outcome_n_trials)

    zero = np.zeros_like(gamma_int)
    cells = {
        (x_lo, m_lo): expected_items(x_lo, m_lo, gamma_int),
        (x_hi, m_lo): expected_items(x_hi, m_lo, gamma_int),
        (x_lo, m_hi): expected_items(x_lo, m_hi, gamma_int),
        (x_hi, m_hi): expected_items(x_hi, m_hi, gamma_int),
    }
    inc_lo = cells[(x_hi, m_lo)] - cells[(x_lo, m_lo)]
    inc_hi = cells[(x_hi, m_hi)] - cells[(x_lo, m_hi)]
    interaction = inc_hi - inc_lo
    additive = (
        expected_items(x_hi, m_hi, zero) - expected_items(x_lo, m_hi, zero)
    ) - (expected_items(x_hi, m_lo, zero) - expected_items(x_lo, m_lo, zero))
    logit_interaction = gamma_int * (z_x(x_hi) - z_x(x_lo)) * (z_m(m_hi) - z_m(m_lo))

    common = {
        "exposure_symbol": exposure_symbol,
        "exposure_unit": f"{exposure_symbol} items",
        "moderator_symbol": moderator_symbol,
        "moderator_unit": moderator_unit,
        "outcome_symbol": outcome_symbol,
        "outcome_unit": f"{outcome_symbol} items",
        "n_obs": int(eta.shape[0]),
        "ci_prob": float(ci_prob),
    }

    def row(label, draws, quantity, scale, xl, xh, ml, mh) -> dict:
        r = coef_row(label, draws, ci_prob)
        r.update(
            quantity=quantity,
            scale=scale,
            exposure_low=xl,
            exposure_high=xh,
            moderator_low=ml,
            moderator_high=mh,
            **common,
        )
        return r

    rows = [
        row(
            f"E[{outcome_symbol} | {exposure_symbol}={x:g}, {moderator_symbol}={m:g}]",
            cells[(x, m)],
            "cell_mean",
            "items",
            x,
            x,
            m,
            m,
        )
        for (x, m) in cells
    ]
    rows += [
        row(
            f"{exposure_symbol} {x_lo:g}->{x_hi:g} increment at {moderator_symbol}={m_lo:g}",
            inc_lo,
            "increment_at_moderator_low",
            "items",
            x_lo,
            x_hi,
            m_lo,
            m_lo,
        ),
        row(
            f"{exposure_symbol} {x_lo:g}->{x_hi:g} increment at {moderator_symbol}={m_hi:g}",
            inc_hi,
            "increment_at_moderator_high",
            "items",
            x_lo,
            x_hi,
            m_hi,
            m_hi,
        ),
        row(
            "items-scale interaction (increment at high minus low moderator)",
            interaction,
            "interaction",
            "items",
            x_lo,
            x_hi,
            m_lo,
            m_hi,
        ),
        row(
            "items-scale interaction if logit-additive (gamma_int = 0)",
            additive,
            "interaction_if_logit_additive",
            "items",
            x_lo,
            x_hi,
            m_lo,
            m_hi,
        ),
        row(
            "logit-scale interaction over the same cells",
            logit_interaction,
            "interaction_logit",
            "logit",
            x_lo,
            x_hi,
            m_lo,
            m_hi,
        ),
    ]
    return rows


#: Raw-unit labels for the covariate moderators a mechanism fit may declare.
_COVARIATE_MODERATOR_UNITS = {"A": "months", "erbto": "raw-score points"}


def write_moderation_items(ctx: Any) -> pd.DataFrame | None:
    """Write ``moderation_items.csv`` for a moderated fit (see :func:`moderation_items_rows`).

    Returns the table, or ``None`` when the fit has no ``gamma_int`` (no
    moderator, or a main-effect-only companion) or when the exposure is not a
    post-score measure count (a covariate or period-start exposure has no items
    cells to form; none is registered with a moderator). Usable over a stored fit
    with a lightweight context carrying ``spec``, ``prepared`` (the *fitted* rows,
    as the factory subsets them), ``trace``, ``output_dir`` and
    ``reporting.ci_prob`` — it reads the posterior and the fitted rows only, so it
    needs no refit.
    """
    run_plan = _mechanism_run_plan(ctx)
    if run_plan.moderator_symbol is None or not run_plan.include_interaction:
        return None
    if run_plan.mechanism_is_covariate or run_plan.mechanism_at_pre:
        rprint(
            "[yellow]write_moderation_items: the exposure is not a post-score "
            f"measure count for {ctx.spec.model_id}; no moderation_items.csv "
            "written.[/yellow]"
        )
        return None
    post = ctx.trace.posterior
    if "gamma_int" not in post:
        return None
    prepared = ctx.prepared
    mech = run_plan.mechanism_symbol
    mod = run_plan.moderator_symbol
    outcome = run_plan.outcome_symbol
    if run_plan.moderator_is_covariate:
        if mod == "A":
            moderator_values = np.asarray(prepared.A_months, dtype=float)
        else:
            z_loaded = np.asarray(prepared.covariates[mod], dtype=float)
            scaler = prepared.covariate_scalers.get(mod)
            moderator_values = scaler.inverse(z_loaded) if scaler is not None else z_loaded
        moderator_n_trials = None
        moderator_unit = _COVARIATE_MODERATOR_UNITS.get(mod, f"{mod} raw units")
    else:
        moderator_values = np.asarray(prepared.post_counts[mod], dtype=float)
        moderator_n_trials = int(prepared.n_trials[mod])
        moderator_unit = f"{mod} items"
    rows = moderation_items_rows(
        post,
        ctx.trace.constant_data,
        exposure_counts=np.asarray(prepared.post_counts[mech], dtype=float),
        exposure_n_trials=int(prepared.n_trials[mech]),
        moderator_values=moderator_values,
        moderator_n_trials=moderator_n_trials,
        outcome_n_trials=int(prepared.n_trials[outcome]),
        ci_prob=float(ctx.reporting.ci_prob),
        exposure_symbol=mech,
        moderator_symbol=mod,
        outcome_symbol=outcome,
        moderator_unit=moderator_unit,
    )
    if not rows:
        rprint(
            "[yellow]write_moderation_items: the fitted exposure or moderator has no "
            f"distinct interquartile cells for {ctx.spec.model_id}; no "
            "moderation_items.csv written.[/yellow]"
        )
        return None
    df = pd.DataFrame(rows)
    save_table(ctx, "moderation_items", df, required=False)
    return df
