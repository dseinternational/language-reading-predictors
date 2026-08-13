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
    _write_readiness_threshold(ctx)

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
            baseline_symbol=run_plan.adjust_baseline_symbol,
        ),
    }
    # Items-scale worked-example reference points (#319): recorded so the caption
    # numbers are computed, not hand-written, and the quantiles are auditable.
    if _items_worked:
        meta_extra["mechanism_items"] = _items_worked
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


def _write_mechanism_curve(ctx: StatisticalFitContext) -> None:
    """Posterior adjusted dose-response of the mechanism predictor on the outcome.

    With the HSGP ``f_mech`` on (the default) this is the non-parametric curve. When
    the model uses the linear slope instead (``linear_mechanism=True``, so no
    ``f_mech`` variable exists) it falls back to the straight
    ``beta_mech * z(logit(predictor))`` band — the predictor's linear logit
    contribution (at the mean of any moderator) — so the adjusted predictor->outcome
    relationship is still shown rather than left implicit in a coefficient. Both
    branches hold the adjustment set fixed and write the same CSV/PNG schema, except
    for the x column: ``mech_logit`` for a bounded-count measure exposure,
    ``mech_x`` (the raw covariate score) for a covariate exposure
    (``mechanism_is_covariate``, always linear). Guarded by the caller.
    """
    post = ctx.trace.posterior

    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    sym = ctx.spec.mechanism_symbol
    run_plan = _mechanism_run_plan(ctx)
    is_covariate = run_plan.mechanism_is_covariate
    if is_covariate:
        # Covariate exposure: x is the raw score (the loader scaler inverted); the
        # model's z is the loader z re-standardised on the kept rows, exactly as
        # the factory did it.
        z_loaded = np.asarray(ctx.prepared.covariates[sym], dtype=float)
        _scaler = ctx.prepared.covariate_scalers.get(sym)
        x_vals = _scaler.inverse(z_loaded) if _scaler is not None else z_loaded
        z_L, _ = standardise(z_loaded)
        x_col, x_label = "mech_x", f"{sym} (raw score)"
    elif run_plan.mechanism_at_pre:
        # Lagged form: the factory fits the mechanism on its period-start (pre)
        # logit, so the reported curve must use that same vector on the same rows.
        # Using the post logit here would plot and label the fitted pre-slope
        # against the wrong exposure — pre/post differ materially (#405 review).
        mech_logit = np.asarray(ctx.prepared.pre_logit[sym], dtype=float)
        x_vals = mech_logit
        z_L, _ = standardise(mech_logit)
        x_col, x_label = "mech_logit", f"logit({sym}_pre)"
    else:
        N = MEASURES[sym].n_trials
        mech_logit = logit_safe(ctx.prepared.post_counts[sym], N)
        x_vals = mech_logit
        # z the same standardisation the factory applied to the logit input.
        z_L, _ = standardise(mech_logit)
        x_col, x_label = "mech_logit", f"logit({sym}_post)"

    if "f_mech" in post:
        f = post["f_mech"].stack(sample=("chain", "draw")).values  # (n_obs, n_sample)
        kind = "GP"
    elif "beta_mech" in post:
        # Linear mechanism: the predictor enters as beta_mech * z. Build the
        # per-observation contribution so the band mirrors the GP branch (an exact
        # straight line).
        b = post["beta_mech"].stack(sample=("chain", "draw")).values  # (n_sample,)
        f = z_L[:, None] * b[None, :]  # (n_obs, n_sample)
        kind = "linear"
    else:
        # No f_mech / beta_mech in the posterior — e.g. a phase_specific_mechanism
        # fit, whose per-phase f_mech is not registered under either name, so the
        # curve would be silently skipped. Warn loudly rather than no-op (issue
        # #273); register the phase-specific curve as pm.Deterministic("f_mech",
        # ..., dims="obs_id") in the factory if such a model is ever shipped.
        rprint(
            "[yellow]_write_mechanism_curve: no 'f_mech'/'beta_mech' in the "
            f"posterior for {ctx.spec.model_id} (phase_specific_mechanism?); "
            "no mechanism_curve.csv/plot written.[/yellow]"
        )
        return

    order = np.argsort(x_vals)
    x = x_vals[order]
    f_ord = f[order]
    mean = f_ord.mean(axis=1)
    lo = np.quantile(f_ord, 0.055, axis=1)
    hi = np.quantile(f_ord, 0.945, axis=1)
    lo50 = np.quantile(f_ord, 0.25, axis=1)
    hi50 = np.quantile(f_ord, 0.75, axis=1)
    save_table(
        ctx,
        "mechanism_curve",
        pd.DataFrame(
            {x_col: x, "f_mean": mean, "f_lo": lo, "f_hi": hi,
             "f_lo50": lo50, "f_hi50": hi50}
        ),
        register=False,
    )
    outcome = ctx.spec.outcome_symbol or "W"

    # Preserve a posterior end-to-end contrast on the outcome-items scale for
    # the key-findings box (#320).  The contrast compares the lowest and highest
    # observed exposure values while setting any moderator to its standardised
    # mean (zero).  Removing the fitted mechanism and moderator contributions
    # from eta before adding the two endpoint contributions keeps every other
    # fitted row characteristic fixed and retains the posterior dependence that
    # the pointwise curve CSV alone cannot reconstruct.
    eta = (
        post["eta"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )
    eta_base = eta - f
    if "gamma_mod" in post and "z_moderator" in ctx.trace.constant_data:
        z_mod = np.asarray(ctx.trace.constant_data["z_moderator"].values).reshape(-1)
        gamma_mod = post["gamma_mod"].stack(sample=("chain", "draw")).values
        eta_base = eta_base - z_mod[:, None] * gamma_mod[None, :]
        if "gamma_int" in post:
            z_mech = np.asarray(
                ctx.trace.constant_data["z_mech_logit"].values
            ).reshape(-1)
            gamma_int = post["gamma_int"].stack(sample=("chain", "draw")).values
            eta_base = eta_base - (
                z_mech[:, None] * z_mod[:, None] * gamma_int[None, :]
            )
    endpoint_items = (
        expit(eta_base + f_ord[-1][None, :])
        - expit(eta_base + f_ord[0][None, :])
    ).mean(axis=0) * float(ctx.prepared.n_trials[outcome])
    lo_q = (1 - ctx.reporting.ci_prob) / 2
    if is_covariate:
        exposure_low = float(x[0])
        exposure_high = float(x[-1])
        exposure_unit = f"{sym} raw-score units"
    else:
        # Invert the Haldane-corrected logit used by preprocessing so the
        # headline exposure range is in test items, not log-odds.
        N = ctx.prepared.n_trials[sym]
        exposure_low = float(np.clip((N + 1) * expit(x[0]) - 0.5, 0, N))
        exposure_high = float(np.clip((N + 1) * expit(x[-1]) - 0.5, 0, N))
        exposure_unit = f"{sym} items"
    mechanism_summary = pd.DataFrame(
        [
            {
                "exposure_low": exposure_low,
                "exposure_high": exposure_high,
                "exposure_unit": exposure_unit,
                "items_median": float(np.median(endpoint_items)),
                "items_lo": float(np.quantile(endpoint_items, lo_q)),
                "items_hi": float(np.quantile(endpoint_items, 1 - lo_q)),
                "items_lo50": float(np.quantile(endpoint_items, 0.25)),
                "items_hi50": float(np.quantile(endpoint_items, 0.75)),
                "prob_pos": float(np.mean(endpoint_items > 0)),
            }
        ]
    )
    save_table(ctx, "mechanism_summary", mechanism_summary)
    plt.figure(figsize=FIGSIZE_LG)
    plt.plot(x, mean, color=COLOUR_BLUE, lw=2)
    plt.fill_between(x, lo, hi, color=COLOUR_BLUE, alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel("predictor logit contribution")
    plt.title(f"Mechanism curve ({kind}): {sym} -> {outcome}")
    # mechanism_curve.csv (the plotted band) is written just above.
    save_styled_figure(ctx.output_dir, "mechanism_curve")


#: Friendly labels for covariate mechanism exposures (no ``Measure`` entry, so no
#: label registry). Falls back to the symbol for anything not listed.
_COVARIATE_EXPOSURE_LABELS = {
    "erbto": "Phonological memory (word/nonword repetition)",
    "deapp_c": "Speech production (DEAP)",
}


def _write_mechanism_items(ctx: StatisticalFitContext) -> dict:
    """Items-scale mechanism dose-response curve + worked example (#319).

    Companion to ``_write_mechanism_curve``: the logit-scale CSV/plot remain the
    analyst's object; this renders the same fitted curve on the items scale
    (exposure items -> predicted outcome items) with a credible ribbon and one
    computed worked-example contrast between fixed quantiles of the observed
    exposure. Returns the ``worked`` dict (quantile reference points + the
    computed caption) so ``fit_mechanism`` can persist it to ``config.json`` for
    the report partial. Never raises through the fit — a failure logs and returns
    ``{}``.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.mechanism_items import (
        write_mechanism_items_artifacts,
    )

    try:
        spec = ctx.spec
        run_plan = _mechanism_run_plan(ctx)
        sym = spec.mechanism_symbol
        outcome = spec.outcome_symbol or "W"
        is_covariate = run_plan.mechanism_is_covariate

        if is_covariate:
            z_loaded = np.asarray(ctx.prepared.covariates[sym], dtype=float)
            scaler = ctx.prepared.covariate_scalers.get(sym)
            x_exposure = scaler.inverse(z_loaded) if scaler is not None else z_loaded
            exposure_label = _COVARIATE_EXPOSURE_LABELS.get(sym, sym)
            exposure_n_trials = None
        elif run_plan.mechanism_at_pre:
            # Lagged form: the fitted curve (via z_mech_logit) is on the pre
            # exposure, so the items-scale x-axis must be the pre counts, not the
            # post counts — otherwise the worked-example quantiles land on the
            # wrong distribution and the axis is mislabelled (#405 review).
            x_exposure = np.asarray(ctx.prepared.pre_counts[sym], dtype=float)
            exposure_label = f"{MEASURES[sym].label} (period start)"
            exposure_n_trials = MEASURES[sym].n_trials
        else:
            x_exposure = np.asarray(ctx.prepared.post_counts[sym], dtype=float)
            exposure_label = MEASURES[sym].label
            exposure_n_trials = MEASURES[sym].n_trials

        # The mechanism factory always fits a Beta-Binomial likelihood, so the
        # y-axis is an item count. Floored (off-floor Bernoulli) mechanism
        # outcomes are a future addition (#319 design note); wire the flag when
        # such a model ships.
        ref_quantiles = run_plan.items_ref_quantiles
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
        _write_mechanism_prior_pushforward(
            ctx,
            x_exposure=x_exposure,
            outcome=outcome,
            exposure_label=exposure_label,
            ref_quantiles=ref_quantiles,
        )
        # ``mechanism_curve_items.csv`` is written inside the helper (which takes
        # an output directory, not a context); record it for the manifest.
        record_artifact(ctx, "mechanism_curve_items", required=False)
        return worked
    except Exception as exc:  # pragma: no cover - defensive; logit curve stands alone
        rprint(f"[yellow]Items-scale mechanism curve failed: {exc}[/yellow]")
        write_prior_pushforward(
            ctx,
            [
                _report.unavailable_pushforward(
                    estimand="mechanism_curve",
                    estimand_label="the mechanism dose-response contrast",
                    role="association",
                    reason=f"the items-scale mechanism curve could not be built: {exc}",
                )
            ],
        )
        return {}


def _write_mechanism_prior_pushforward(
    ctx: StatisticalFitContext,
    *,
    x_exposure: np.ndarray,
    outcome: str,
    exposure_label: str,
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

    Never raises: this rides on the items-curve writer, and a prior check that
    could abort the fitted curve it accompanies would trade a bigger deliverable
    for a smaller one.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.mechanism_items import (
        mechanism_items_curve,
    )

    label = "the mechanism dose-response contrast"
    try:
        n_trials = MEASURES[outcome].n_trials
        q_lo = int(round(100 * ref_quantiles[0]))
        q_hi = int(round(100 * ref_quantiles[1]))
        label = (
            f"the predicted difference on {MEASURES[outcome].label} between the "
            f"{q_hi}th and {q_lo}th percentile of {exposure_label}"
        )
        source = getattr(ctx, "prior_samples", None) or ctx.trace
        _, worked = mechanism_items_curve(
            source,
            x_exposure=x_exposure,
            n_trials_outcome=n_trials,
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
    except Exception as exc:  # noqa: BLE001 - absence must stay legible
        rows = [
            _report.unavailable_pushforward(
                estimand="mechanism_curve",
                estimand_label=label,
                role="association",
                reason=str(exc),
            )
        ]
    with guard_optional(
        ctx, "mechanism prior pushforward",
        filename="prior_pushforward.csv", kind="table", verb="not written",
    ):
        write_prior_pushforward(ctx, rows)


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
