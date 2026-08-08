# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
End-to-end fit pipeline for the statistical models.

What is left here is the families that have yet to migrate: survival, the LRP67
LCSM, growth and its historical companion, and the Byrne measurement and
joint-growth fits — including the correlated-factor models whose specialised
likelihood and log-prior recovery #394 design point 7 wants isolated.

The ITT and joint families moved to :mod:`pipelines` first (#394 step 5), then the
DiD, dose-response, gain- and level-factors, block-exposure and aligned families,
then mechanism, joint mechanism and mediation, then adjusted, horseshoe and
concurrent with the Byrne ports of the first two (step 6). Their entry points are
re-exported here so model modules and tests keep their import path;
``MIGRATED_FAMILIES`` in ``test_pipeline_boundaries.py`` is the authoritative list,
checked against the package contents.

The shared mechanics they used to carry inline now live in :mod:`runtime` (the
stage binding and spec validation), :mod:`publication` (banners, report template,
model graph), :mod:`adjustment` (the fitted adjustment-set record),
:mod:`prior_artifacts`, :mod:`ppc_artifacts` and :mod:`figure_artifacts`, with the
sub-fit sampler in :mod:`diagnostics` beside the primary one it mirrors and the
posterior-summary helpers in :mod:`reporting`. The remaining families migrate in
the same way.

Each pipeline:

1. Loads data via :func:`preprocessing.load_and_prepare`.
2. Builds the PyMC model via the appropriate factory.
3. Writes prior-panel plots.
4. Runs prior predictive, posterior sampling (nutpie), LOO, posterior
   predictive.
5. Saves ``trace.nc`` (with the prior / prior_predictive / log_prior groups
   attached — issue #125 step 0b), ``config.json``, ``diagnostics_summary.json``
   (the pass/fail convergence gate), ``priors_table.csv`` and the standard
   diagnostic plots to ``output/statistical_models/models/{model_id}-{config}/``.
6. Copies ``docs/models/{model_id}/index.qmd`` and the shared
   ``docs/models/_partials/`` alongside the artefacts so the Quarto report can be
   rendered in-place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
    section_header,
)
from language_reading_predictors.statistical_models import (
    datasets as _datasets,
    diagnostics as _diag,
    factories as _factories,
    historical as _historical,
    lcf_inference as _lcf_inference,
    lcf_summaries as _lcf_summaries,
    reporting as _report,
    survival as _survival,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
)
from language_reading_predictors.statistical_models.factories import default_of
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_forest_plot,
    write_panel_child_fit,
    write_panel_trajectory,
)
from language_reading_predictors.statistical_models.growth import (
    resolve_growth_run_plan,
)

# Compatibility re-exports: these families now live in ``pipelines/`` (#394 steps
# 5-6). The ``x as x`` form marks them as deliberate re-exports rather than unused
# imports, and keeps ``from ...pipeline import fit_itt`` working for every model
# module and test until step 8 migrates the callers.
from language_reading_predictors.statistical_models.pipelines.adjusted import (
    fit_adjusted as fit_adjusted,
    fit_rlm_adjusted as fit_rlm_adjusted,
)
from language_reading_predictors.statistical_models.pipelines.aligned import (
    fit_aligned as fit_aligned,
)
from language_reading_predictors.statistical_models.pipelines.block_exposure import (
    fit_block_exposure as fit_block_exposure,
)
from language_reading_predictors.statistical_models.pipelines.concurrent import (
    fit_concurrent as fit_concurrent,
)
from language_reading_predictors.statistical_models.pipelines.did import (
    fit_did as fit_did,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import (
    fit_dose_response as fit_dose_response,
)
from language_reading_predictors.statistical_models.pipelines.gain_factors import (
    fit_gain_factors as fit_gain_factors,
)
from language_reading_predictors.statistical_models.pipelines.horseshoe import (
    fit_horseshoe as fit_horseshoe,
    fit_rlm_horseshoe as fit_rlm_horseshoe,
)
from language_reading_predictors.statistical_models.pipelines.itt import (
    fit_itt as fit_itt,
)
from language_reading_predictors.statistical_models.pipelines.joint import (
    fit_joint as fit_joint,
)
from language_reading_predictors.statistical_models.pipelines.joint_mechanism import (
    fit_joint_mechanism as fit_joint_mechanism,
)
from language_reading_predictors.statistical_models.pipelines.level_factors import (
    fit_level_factors as fit_level_factors,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import (
    fit_mechanism as fit_mechanism,
)
from language_reading_predictors.statistical_models.pipelines.mediation import (
    fit_mediation as fit_mediation,
    fit_mediation_multi as fit_mediation_multi,
    fit_mediation_period_stacked as fit_mediation_period_stacked,
    prepare_mediation_data as prepare_mediation_data,
)
from language_reading_predictors.statistical_models.prior_artifacts import (
    growth_contrast_pushforward_rows,
    write_indicator_prior_check,
    write_prior_pushforward,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    load_longitudinal_panel,
    load_wave_panel,
)
from language_reading_predictors.statistical_models.publication import (
    print_header,
    print_loo_row,
    render_model_graph,
)
from language_reading_predictors.statistical_models.runtime import (
    attach_built,
    finalize_report,
    require_spec,
    run_ppc,
    run_sampling_and_loo,
    shared_stages,
    write_run_metadata,
)
from language_reading_predictors.statistical_models.stages import (
    PrimaryFitPlan,
)


# ---------------------------------------------------------------------------
# Survival pipeline (LRP-RLI-SURV)
# ---------------------------------------------------------------------------


def _survival_summary(
    trace, *, ci_prob: float, hazard_link: str, use_treatment: bool
) -> pd.DataFrame:
    """Off-floor discrete-time hazard summary (log-hazard, hazard ratio, P>0).

    Reports the treatment hazard shift and baseline-covariate slopes on the
    log-hazard scale (with ``exp`` as a hazard ratio and ``P(effect > 0)``), plus
    the per-interval baseline off-floor probability for an untreated child at mean
    covariates, on the model's ``hazard_link`` scale. Equal-tailed intervals at
    ``ci_prob`` with the posterior median as the point estimate (the suite convention).
    """
    post = trace.posterior
    lo, hi = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2

    def _row(term: str, draws: np.ndarray, *, as_ratio: bool) -> dict:
        d = np.asarray(draws).reshape(-1)
        return {
            "term": term,
            "median": float(np.median(d)),
            "ci_low": float(np.quantile(d, lo)),
            "ci_high": float(np.quantile(d, hi)),
            "hazard_ratio": float(np.exp(np.median(d))) if as_ratio else float("nan"),
            "P(>0)": float(np.mean(d > 0)) if as_ratio else float("nan"),
        }

    rows: list[dict] = []
    if use_treatment:
        rows.append(_row("tau (log hazard shift, treated)", post["tau"].values, as_ratio=True))
    for name in sorted(v for v in post.data_vars if str(v).startswith("beta_")):
        rows.append(_row(f"{name} (log hazard, per SD)", post[name].values, as_ratio=True))

    alpha = post["alpha"].stack(sample=("chain", "draw")).transpose("interval", "sample")
    labels = [str(v) for v in alpha.coords["interval"].values]
    for i, lab in enumerate(labels):
        a = alpha.values[i]
        base = 1.0 - np.exp(-np.exp(a)) if hazard_link == "cloglog" else 1.0 / (1.0 + np.exp(-a))
        rows.append(
            {
                "term": f"baseline off-floor prob [{lab}]",
                "median": float(np.median(base)),
                "ci_low": float(np.quantile(base, lo)),
                "ci_high": float(np.quantile(base, hi)),
                "hazard_ratio": float("nan"),
                "P(>0)": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fit_survival(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Discrete-time off-floor survival fit for a floored outcome P / N (#230 §5).

    Fits a person-period discrete-time hazard for the *time* to come off the floor,
    generalising the single-transition off-floor estimand of the LRPITT09/11 floor
    rule to all four waves. Treatment enters as an intervention-aligned hazard shift;
    the estimand is prognostic (both arms are treated by t4).
    """
    require_spec(spec, "survival", outcome=True)
    ctx = make_context(spec, config)

    section_header("Prepare data")
    panel = _survival.prepare_survival(spec.outcome_symbol)
    ctx.prepared = panel
    print_header(ctx)
    rprint(
        f"  Survival at-risk set: {panel.n_at_risk_children} children at the "
        f"{spec.outcome_symbol} floor at t1 contribute {panel.n_obs} person-period rows; "
        f"{panel.n_events} off-floor events."
    )
    if panel.dropped_rows:
        rprint(
            f"  [yellow]{panel.dropped_rows} at-risk child(ren) contributed no rows "
            "(t2 score unobserved, so no interval could be placed) and are excluded.[/yellow]"
        )
    for name, k in panel.imputed_covariate_rows.items():
        if k:
            rprint(
                f"  [yellow]{k} row(s) had a missing baseline {name}; mean-imputed (z=0).[/yellow]"
            )

    hazard_link = spec.extra.get("hazard_link", "cloglog")
    use_treatment = bool(spec.extra.get("use_treatment", True))

    section_header("Build model")
    built = _survival.build_survival_model(
        panel, hazard_link=hazard_link, use_treatment=use_treatment
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    diag_vars = (
        ["alpha"]
        + [f"beta_{n}" for n in panel.covariates]
        + (["tau"] if use_treatment else [])
    )

    # Reference adoption of the shared primary-fit lifecycle (#394 design 2):
    # the invariant sequence lives in ``stages.run_primary_fit`` and the family
    # declares only its execution profile.
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            ppc_var_names=("y_event",),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_rate_plot(
                c, spec.outcome_symbol, node="y_event"
            ),
            extended_term="tau" if use_treatment else None,
        ),
    )

    section_header("Off-floor hazard summary")
    summary = _survival_summary(
        ctx.trace, ci_prob=ctx.reporting.ci_prob, hazard_link=hazard_link,
        use_treatment=use_treatment,
    )
    save_table(ctx, "survival_summary", summary)
    print_table(
        ranked_dataframe_table(
            summary,
            title=(
                f"Off-floor discrete-time hazard ({spec.outcome_symbol}, {hazard_link}); "
                "positive = raises Pr(off-floor); prognostic, not a randomised effect"
            ),
            columns=list(summary.columns),
            rank_column=False,
            precision=3,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "n_at_risk_children": panel.n_at_risk_children,
            "n_events": panel.n_events,
            "hazard_link": hazard_link,
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Longitudinal dynamic pipeline (LRP67 LCSM)
# ---------------------------------------------------------------------------


def _coef_row(label: str, draws, hdi_prob: float) -> dict:
    """Posterior mean, equal-tailed central interval and ``P(coef > 0)``.

    Equal-tailed quantiles at coverage ``hdi_prob`` — the same convention as
    :func:`reporting.tau_summary_itt` (not a highest-density interval).
    """
    d = np.asarray(draws).reshape(-1)
    lo_q = (1 - hdi_prob) / 2
    return {
        "coefficient": label,
        "median": float(np.median(d)),
        "mean": float(np.mean(d)),
        "lo": float(np.quantile(d, lo_q)),
        "hi": float(np.quantile(d, 1 - lo_q)),
        "lo50": float(np.quantile(d, 0.25)),
        "hi50": float(np.quantile(d, 0.75)),
        "prob_pos": float(np.mean(d > 0)),
    }


def fit_lcsm(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Latent change-score model (LRP67 + the lagged coupling suite, #250).

    Fits the coupled McArdle latent change-score model with process noise and
    reports the per-target coupling tables. ``spec.extra`` selects the shape:
    the LRP67 default couples every other measure into the reading change; the
    lagged reverse-coupling models (LCSM-081/181/082) pass an explicit
    ``couplings`` map plus ``arm_window_intercepts`` (the crossover-aware
    arm x window change intercepts, with the window-1 randomised contrast
    written to ``itt_window1_contrast.csv``) and a shared adjuster
    ``covariate_block``. ``dominance_pair`` adds the SD-standardised
    reciprocal-dominance contrast (``dominance_summary.csv``).
    ``lagged_change_couplings`` (LCSM-091, #229 spec 2) adds prior-transition
    latent-change terms (``h_{src}``) to the named targets' change equations.
    """
    require_spec(spec, "lcsm")

    ctx = make_context(spec, config)

    section_header("Prepare data")
    outcomes = tuple(spec.extra.get("outcomes", ("W", "L", "E")))
    reading_symbol = spec.outcome_symbol or "W"
    couplings_in = spec.extra.get("couplings")
    couplings: dict[str, tuple[str, ...]] = (
        {tgt: tuple(srcs) for tgt, srcs in couplings_in.items()}
        if couplings_in
        else {reading_symbol: tuple(s for s in outcomes if s != reading_symbol)}
    )
    lagged_in = spec.extra.get("lagged_change_couplings")
    lagged_change_couplings: dict[str, tuple[str, ...]] = (
        {tgt: tuple(srcs) for tgt, srcs in lagged_in.items()} if lagged_in else {}
    )
    arm_window = bool(spec.extra.get("arm_window_intercepts", False))
    covariate_block = tuple(spec.extra.get("covariate_block", ()))
    covariate_targets = tuple(spec.extra.get("covariate_targets", ()))
    # Loader needs from the covariate block: the hearing dummies come via
    # include_hearing; everything else names a per-wave source column (its
    # ``_missing`` companion is derived, not loaded).
    include_hearing = any(n in ("hs", "hs_missing") for n in covariate_block)
    wave_cov_cols = tuple(
        dict.fromkeys(
            n
            for n in covariate_block
            if n not in ("hs", "hs_missing") and not n.endswith("_missing")
        )
    )
    panel = load_wave_panel(
        outcomes=outcomes,
        wave_covariates=wave_cov_cols,
        include_hearing=include_hearing,
    )
    ctx.prepared = panel

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_lcsm_model(
        panel,
        reading_symbol=reading_symbol,
        couplings=couplings,
        lagged_change_couplings=lagged_change_couplings or None,
        arm_window_intercepts=arm_window,
        covariate_block=covariate_block,
        covariate_targets=covariate_targets,
        coupling_prior_sigma=spec.extra.get(
            "coupling_prior_sigma",
            default_of(_factories.build_lcsm_model, "coupling_prior_sigma"),
        ),
        use_process_noise=spec.extra.get("use_process_noise", True),
        shared_process_noise=spec.extra.get("shared_process_noise", False),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    # Coupling parameter names mirror the factory's rule: single target keeps
    # LRP67's ``g_{src}``; multiple targets carry the target (``g_{src}_{tgt}``).
    single_target = len(couplings) == 1
    coupling_names = {
        (src, tgt): (f"g_{src}" if single_target else f"g_{src}_{tgt}")
        for tgt, srcs in couplings.items()
        for src in srcs
    }
    # Lagged change-on-change names mirror the factory's rule on the lag map.
    single_lag_target = len(lagged_change_couplings) == 1
    lagged_names = {
        (src, tgt): (f"h_{src}" if single_lag_target else f"h_{src}_{tgt}")
        for tgt, srcs in lagged_change_couplings.items()
        for src in srcs
    }
    diag_vars = list(coupling_names.values())
    diag_vars += list(lagged_names.values())
    diag_vars += [f"b_{name}" for name in covariate_block]
    diag_vars += ["a_change", "b_self", "d_age", "sigma1", "kappa"]
    if spec.extra.get("use_process_noise", True):
        diag_vars.append("sigma_proc")

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # One check per measure: ``y_obs`` flattens every measure into a single vector, so
    # a lone overlay would pool scales with different maxima. The headline reading
    # symbol keeps the unsuffixed filename the report partial expects.
    for _sym in outcomes:
        _diag.save_prior_predictive_plot(
            ctx,
            _sym,
            node="y_obs",
            filename_stem=(
                "prior_predictive_check"
                if _sym == reading_symbol
                else f"prior_predictive_check_{_sym.lower()}"
            ),
        )

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=["y_obs"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(
        ctx, causal_term=diag_vars[0] if diag_vars else None
    )
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Per-target coupling table — the headline "what predicts whose change"
    # output. For LRP67 (single reading target) this reproduces the historical
    # reading-change table, labels included.
    section_header("Change-coupling summary")
    post = ctx.trace.posterior
    rows = [
        _coef_row(
            f"{pname} (prior {src} -> {tgt} change)",
            post[pname].values,
            ctx.reporting.ci_prob,
        )
        for (src, tgt), pname in coupling_names.items()
    ]
    rows += [
        _coef_row(
            f"{pname} (prior {src} change -> {tgt} change)",
            post[pname].values,
            ctx.reporting.ci_prob,
        )
        for (src, tgt), pname in lagged_names.items()
    ]
    for name in covariate_block:
        rows.append(
            _coef_row(
                f"b_{name} ({name} -> {'/'.join(covariate_targets)} change)",
                post[f"b_{name}"].values,
                ctx.reporting.ci_prob,
            )
        )
    for tgt in couplings:
        # LRP67's historical row labels are kept verbatim for the single
        # reading-target shape.
        legacy = single_target and tgt == reading_symbol
        rows.append(
            _coef_row(
                f"b_self[{tgt}] (reading self-feedback)"
                if legacy
                else f"b_self[{tgt}] ({tgt} self-feedback)",
                post["b_self"].sel(outcome=tgt).values,
                ctx.reporting.ci_prob,
            )
        )
        if not arm_window:
            rows.append(
                _coef_row(
                    f"a_change[{tgt}] (reading baseline change)"
                    if legacy
                    else f"a_change[{tgt}] ({tgt} baseline change)",
                    post["a_change"].sel(outcome=tgt).values,
                    ctx.reporting.ci_prob,
                )
            )
        rows.append(
            _coef_row(
                f"d_age[{tgt}] (age -> reading change)"
                if legacy
                else f"d_age[{tgt}] (age -> {tgt} change)",
                post["d_age"].sel(outcome=tgt).values,
                ctx.reporting.ci_prob,
            )
        )
    coupling_df = pd.DataFrame(rows)
    save_table(ctx, "coupling_summary", coupling_df)
    print_table(
        ranked_dataframe_table(
            coupling_df,
            title=(
                f"Change couplings - {int(ctx.reporting.ci_prob * 100)}% CI "
                "(equal-tailed)"
            ),
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # Window-1 randomised contrast on the latent change scale (immediate -
    # waitlist), the built-in consistency check against the ITT suite. Only the
    # arm x window shape carries it.
    itt_rows: list[dict] = []
    if arm_window:
        section_header("Window-1 randomised contrast (ITT consistency check)")
        for s in outcomes:
            itt_rows.append(
                _coef_row(
                    f"itt_w1[{s}] (immediate - waitlist, window-1 latent change)",
                    post["itt_w1_contrast"].sel(outcome=s).values,
                    ctx.reporting.ci_prob,
                )
            )
        itt_df = pd.DataFrame(itt_rows)
        save_table(ctx, "itt_window1_contrast", itt_df)
        print_table(
            ranked_dataframe_table(
                itt_df,
                title="Window-1 arm contrast (latent logit change)",
                columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    # Reciprocal-dominance contrast (LCSM-082): per draw, standardise each
    # direction's coupling by the model's own latent scales (g* = g *
    # sd(prior source levels) / sd(target changes)) and report |g*_AB| - |g*_BA|.
    dom_rows: list[dict] = []
    dominance_pair = spec.extra.get("dominance_pair")
    if dominance_pair:
        a, b = dominance_pair
        section_header(f"Reciprocal dominance: {a} <-> {b}")
        x = post["x_latent"]

        def _std_coupling(src: str, tgt: str):
            g = post[coupling_names[(src, tgt)]]
            sd_src = x.isel(wave=slice(0, -1)).sel(outcome=src).std(
                dim=("child", "wave")
            )
            sd_dt = x.sel(outcome=tgt).diff("wave").std(dim=("child", "wave"))
            return g * sd_src / sd_dt

        g_ab = _std_coupling(a, b)  # prior a -> b change
        g_ba = _std_coupling(b, a)  # prior b -> a change
        contrast = abs(g_ab) - abs(g_ba)
        dom_rows = [
            _coef_row(f"std g ({a} -> {b} change)", g_ab.values, ctx.reporting.ci_prob),
            _coef_row(f"std g ({b} -> {a} change)", g_ba.values, ctx.reporting.ci_prob),
            _coef_row(
                f"|std g {a}->{b}| - |std g {b}->{a}| (dominance)",
                contrast.values,
                ctx.reporting.ci_prob,
            ),
        ]
        dom_df = pd.DataFrame(dom_rows)
        save_table(ctx, "dominance_summary", dom_df)
        print_table(
            ranked_dataframe_table(
                dom_df,
                title="SD-standardised reciprocal couplings",
                columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
                rank_column=False,
                precision=3,
            )
        )

    # Per-child fitted-vs-observed panels (#317 fig 2) for the focal reading target.
    write_panel_child_fit(ctx, latent_name="x_latent", focal_symbol=reading_symbol)

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "outcomes": list(outcomes),
            "reading_symbol": reading_symbol,
            "couplings": {tgt: list(srcs) for tgt, srcs in couplings.items()},
            "lagged_change_couplings": {
                tgt: list(srcs) for tgt, srcs in lagged_change_couplings.items()
            },
            "arm_window_intercepts": arm_window,
            "covariate_block": list(covariate_block),
            "covariate_targets": list(covariate_targets),
            "coupling_summary": rows,
            **({"itt_window1_contrast": itt_rows} if itt_rows else {}),
            **({"dominance_summary": dom_rows} if dom_rows else {}),
        },
    )

    return finalize_report(ctx)


def fit_growth(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Joint multivariate latent growth-curve model (LRP69 core / LRP70 factor).

    Characterises each verbal/reading measure's within-child trajectory across the
    four RLI waves and reports whether **baseline non-verbal ability** (``blocks``)
    predicts trajectory shape: ``gamma`` on the growth *rate* (the headline Q5
    estimand) and ``delta`` on the baseline *level*. With ``use_shared_factor`` a
    rank-1 shared growth-tempo factor couples the slopes and the block-design ->
    common-tempo association is read out post-hoc. Every non-randomised term is an
    **adjusted, latent-GA-confounded association**, never causal (locked DAG,
    ``notes/202606231600-dag-revision-consolidated.md``).
    """
    require_spec(spec, "growth")

    # Resolve and validate the family contract before the context resets an output
    # directory or the loader reads any data (#394 pillar 4). One plan drives
    # preparation, factory arguments, the teaching recipe and config.json.
    plan = resolve_growth_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    outcomes = plan.outcomes
    baseline_cov = plan.baseline_covariate
    use_factor = plan.use_shared_factor
    age_ability = plan.age_ability_interaction

    section_header("Prepare data")
    panel = load_wave_panel(**plan.prepare_kwargs())
    ctx.prepared = panel

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_growth_model(panel, **plan.factory_kwargs())
    attach_built(ctx, built)

    render_model_graph(ctx)

    diag_vars = [
        "gamma", "delta", "beta", "alpha", "sigma_slope", "sigma_intercept", "kappa"
    ]
    if use_factor:
        diag_vars.append("loading")
    if age_ability:
        # LRP85 (#228 item 10): baseline-age main effect + the headline
        # age0 × ability interaction on the growth rate.
        diag_vars.extend(["gamma_age", "gamma_int"])

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # One check per measure: ``y_obs`` flattens (child, wave, outcome) into a single
    # vector, so a lone overlay would pool scales with different maxima. The first
    # outcome keeps the unsuffixed filename the report partial expects.
    for _i, _sym in enumerate(outcomes):
        _diag.save_prior_predictive_plot(
            ctx,
            _sym,
            node="y_obs",
            filename_stem=(
                "prior_predictive_check"
                if _i == 0
                else f"prior_predictive_check_{_sym.lower()}"
            ),
        )

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=["y_obs"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=None)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Headline Q5 output: baseline non-verbal ability -> trajectory shape. The
    # gamma (growth-rate) rows are the answer; delta (level) and beta (mean slope)
    # round out the trajectory characterisation. All adjusted associations.
    section_header("Non-verbal ability -> trajectory shape (Q5)")
    gs = _report.growth_association_summary(ctx.trace, ci_prob=ctx.reporting.ci_prob)
    save_table(ctx, "growth_association_summary", gs)
    save_forest_plot(ctx, ["gamma"], name="gamma_forest.png")
    print_table(
        ranked_dataframe_table(
            gs[gs["coefficient"] == "gamma"],
            title="Baseline non-verbal ability -> growth rate (gamma, logit; 95% ETI)",
            columns=[
                "outcome", "median", "lo89", "hi89", "prob_positive",
                "favoured_direction_label",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # Factor layer: is there *residual* coupling between baseline non-verbal
    # ability and the common growth tempo, beyond what the model already
    # attributes to block-design directly (the gamma/delta terms)? Block-design
    # enters the trajectory as a predictor, so G_tempo is the shared tempo net of
    # that modelled effect — this correlation is therefore a *residual*
    # association, not the total "does ability predict tempo". Read out post-hoc as
    # the per-draw correlation between each child's latent tempo G_i and their
    # standardised block-design score: independent a priori, but the posterior can
    # still correlate G and blocks through the likelihood. Descriptive only.
    tempo_corr: dict[str, float] | None = None
    if use_factor and "G_tempo" in ctx.trace.posterior:
        G = (
            ctx.trace.posterior["G_tempo"]
            .stack(sample=("chain", "draw"))
            .transpose("child", "sample")
            .values
        )  # (N, S)
        zb = np.asarray(panel.baseline[baseline_cov], dtype=float)  # (N,)
        Gc = G - G.mean(axis=0, keepdims=True)
        zc = (zb - zb.mean())[:, None]
        denom = np.sqrt((Gc**2).sum(0) * (zc**2).sum(0)) + 1e-12
        corr = (Gc * zc).sum(0) / denom  # (S,)
        lo_q = (1 - ctx.reporting.ci_prob) / 2
        tempo_corr = {
            "median": float(np.median(corr)),
            "lo": float(np.quantile(corr, lo_q)),
            "hi": float(np.quantile(corr, 1 - lo_q)),
            "lo50": float(np.quantile(corr, 0.25)),
            "hi50": float(np.quantile(corr, 0.75)),
            "prob_pos": float(np.mean(corr > 0)),
        }
        save_table(ctx, "growth_tempo_corr", pd.DataFrame([tempo_corr]))
        rprint(
            f"[bold]blocks <-> growth-tempo residual corr:[/bold] {tempo_corr['median']:+.3f} "
            f"[{tempo_corr['lo']:+.3f}, {tempo_corr['hi']:+.3f}] "
            f"P(>0)={tempo_corr['prob_pos']:.3f}"
        )

    # Data-space figures (#317): per-measure cohort trajectory (no arm — growth's
    # "arm" is a latent tempo, not an observed randomised arm) and per-child
    # fitted-vs-observed panels for a focal outcome.
    write_panel_trajectory(ctx, latent_name="theta")
    write_panel_child_fit(ctx, latent_name="theta", focal_symbol=outcomes[0])

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd),
            "outcomes": list(outcomes),
            "baseline_covariate": baseline_cov,
            "use_shared_factor": use_factor,
            "growth_association_summary": gs.to_dict("records"),
            **({"blocks_tempo_corr": tempo_corr} if tempo_corr else {}),
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Historical group-by-wave growth (RLMHG, #165 - first non-RLI dataset)
# ---------------------------------------------------------------------------


def fit_historical_growth(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Historical group-by-wave growth model (RLMHG, #165).

    A descriptive natural-history growth model for a non-RLI historical cohort
    (the Byrne reading-language-memory study), run through the shared
    statistical-model pipeline so it uses the same sampler, convergence gate,
    output layout and report conventions as the intervention models. It is
    **not** an intervention-effect model - ``group`` carries no treatment
    semantics (see :func:`factories.build_historical_growth_model`).
    """
    require_spec(spec, "historical_growth")

    ctx = make_context(spec, config)

    section_header("Prepare data")
    study_id = spec.extra.get("study_id", spec.study_id)
    measure = spec.extra.get("measure", spec.outcome_symbol or "basread")
    waves = tuple(spec.extra.get("waves", (1, 2, 3)))
    extension_waves = tuple(spec.extra.get("extension_waves", ()))
    dataset, measures = _datasets.resolve_dataset(study_id)
    if measure not in measures:
        raise KeyError(f"measure {measure!r} not registered for study {study_id!r}")
    panel = load_longitudinal_panel(
        dataset,
        [measures[measure]],
        waves=waves,
        complete_case=True,
        extension_waves=extension_waves,
    )
    ctx.prepared = panel

    print_header(ctx)

    section_header("Build model")
    built = _factories.build_historical_growth_model(
        panel,
        measure=measure,
        eta_prior_sigma=spec.extra.get(
            "eta_prior_sigma",
            default_of(_factories.build_historical_growth_model, "eta_prior_sigma"),
        ),
        sigma_subject_prior_sigma=spec.extra.get(
            "sigma_subject_prior_sigma",
            default_of(
                _factories.build_historical_growth_model, "sigma_subject_prior_sigma"
            ),
        ),
        kappa_prior_sigma=spec.extra.get(
            "kappa_prior_sigma",
            default_of(_factories.build_historical_growth_model, "kappa_prior_sigma"),
        ),
    )
    attach_built(ctx, built)

    render_model_graph(ctx)

    diag_vars = ["eta_cell", "sigma_subject", "kappa"]
    diag_vars += [
        v
        for v in (
            "growth_first_next_items",
            "growth_next_last_items",
            "growth_first_last_items",
        )
        if v in ctx.model.named_vars
    ]

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, measure, node="score")

    run_sampling_and_loo(ctx)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381).
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=["score"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    # Descriptive summaries: observed complete-case baseline (the Table 2 audit
    # target), posterior group-by-wave fitted means, and within-group / between-
    # group growth in items.
    section_header("Growth summaries")
    measure_label = measures[measure].label
    baseline = _historical.observed_baseline(panel, measure, measure_label)
    save_table(ctx, "observed_complete_case_baseline", baseline)
    cells = _historical.cell_summary(ctx.trace, panel, measure, measure_label, baseline)
    save_table(ctx, "posterior_cell_summary", cells)
    growth = _historical.growth_summary(ctx.trace, panel, measure)
    save_table(ctx, "posterior_growth_summary", growth)
    write_prior_pushforward(
        ctx, growth_contrast_pushforward_rows(ctx, panel, measure)
    )
    print_table(
        ranked_dataframe_table(
            growth,
            title=(
                f"{measure_label} growth (items) - "
                f"{int(ctx.reporting.ci_prob * 100)}% CI (equal-tailed)"
            ),
            columns=["label", "readgrp_label", "mean", "q_lo", "q_hi", "p_gt_0"],
            rank_column=False,
            precision=2,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "study_id": study_id,
            "measure": measure,
            "measure_label": measure_label,
            "n_trials": panel.n_trials[measure],
            "waves": list(waves),
            "extension_waves": list(extension_waves),
            "groups": dict(
                zip(panel.group_codes, panel.group_labels, strict=True)
            ),
            "n_subjects": panel.n_subjects,
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Byrne (RLM) Phase B/D fits (#338): corr_factor, joint growth
# ---------------------------------------------------------------------------


def fit_rlm_corr_factor(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne correlated-domain-factor measurement fit (#338 Phase B, mm-001).

    Measurement-only (per the 2026-07-16 sign-off): loadings/communalities and
    the domain-factor correlation matrix over the wave-3 nine-measure battery,
    no structural leg. Writes ``loadings_summary.csv``,
    ``factor_correlation.csv`` and ``factor_correlation_summary.csv`` in the
    RLI ``corr_factor`` schema so the shared partial and key-findings builder
    apply unchanged. LOO is not computed, matching the RLI corr-factor family.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        load_rlm_wave_battery,
    )

    require_spec(spec, "corr_factor")
    e = spec.extra
    wave = int(e.get("wave", 3))
    domains = {k: tuple(v) for k, v in e["domains"].items()}
    reliability = float(e.get("single_indicator_reliability", 0.8))
    lkj_eta = float(e.get("lkj_eta", 2.0))
    comm_alpha = float(
        e.get("comm_alpha", default_of(_factories.build_rlm_corr_factor_model, "comm_alpha"))
    )
    comm_beta = float(
        e.get("comm_beta", default_of(_factories.build_rlm_corr_factor_model, "comm_beta"))
    )

    ctx = make_context(spec, config)
    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0

    section_header("Prepare data")
    symbols = tuple(dict.fromkeys(s for syms in domains.values() for s in syms))
    battery = load_rlm_wave_battery(wave=wave, measure_symbols=symbols)
    ctx.prepared = battery
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_corr_factor_model(
        battery,
        domains=domains,
        single_indicator_reliability=reliability,
        comm_alpha=comm_alpha,
        comm_beta=comm_beta,
        lkj_eta=lkj_eta,
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_dist_overlay(ctx)

    run_sampling_and_loo(ctx, compute_loo=False)

    diag_vars = ["lambda_free", "sigma_free", "factor_corr_pairs"]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity (#381), as in the RLI ``corr_factor`` family:
    # LOO is skipped here, so the log groups have to be added explicitly before the
    # reported loadings, residual scales and factor correlations can be power-scaled.
    # #381 exempted this model on the grounds that its posterior had not converged;
    # since the #383 ``LKJCorr`` fix it does (0 divergences, max R-hat 1.0004), so the
    # exemption no longer applies and a latent-factor model is exactly where an
    # unmeasured prior dependence would matter most.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=["Z_obs"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    # Indicator-scale prior check (#381) — the measurement families' stand-in for
    # the estimand pushforward the outcome families get. AFTER save_trace, which
    # is what attaches the prior/prior_predictive groups to ctx.trace on a fresh
    # fit: called earlier, the check found no prior_predictive group and skipped
    # silently (#383) — the re-emitted reporting artefacts never showed it
    # because a reused trace arrives with its groups already on disk.
    write_indicator_prior_check(ctx, ["Z_obs"])
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    post = ctx.trace.posterior

    # --- Loadings + communalities (the measurement headline) ----------------
    section_header("Loadings + communalities")
    from language_reading_predictors.statistical_models import (
        rlm_corr_factor_summaries as _rlm_summaries,
    )

    load_df = _rlm_summaries.loadings_communalities_table(post, domains, lo_q=lo_q)
    save_table(ctx, "loadings_summary", load_df)
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings, correlations + communalities - {int(hdi * 100)}% CI",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Factor correlation matrix + per-pair summary ------------------------
    section_header("Factor correlation")
    corr_df = _rlm_summaries.factor_correlation_matrix(post)
    save_table(ctx, "factor_correlation", corr_df, index=True)
    corr_summary_df = _rlm_summaries.factor_correlation_pairs(post, lo_q=lo_q)
    save_table(ctx, "factor_correlation_summary", corr_summary_df)
    print_table(
        ranked_dataframe_table(
            corr_summary_df,
            title=f"Domain-factor correlations - {int(hdi * 100)}% CI",
            columns=["domain_i", "domain_j", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "study_id": "rlm",
            "wave": wave,
            "domains": {k: list(v) for k, v in domains.items()},
            "single_indicator_reliability": reliability,
            "n_children": battery.n_children,
            "structural_leg": False,
        },
    )
    return finalize_report(ctx)


def fit_rlm_joint_growth(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Byrne joint correlated growth fit (#338 Phase B, ``lrp-rlm-jc-001``).

    Fits :func:`factories.build_rlm_joint_growth_model` over a small measure set
    and reports the between-child cross-measure correlation matrix of the
    stable child levels (the headline), plus per-measure fitted cells and
    common-window growth via the shared historical summaries. LOO is not
    computed: the model has one likelihood node per measure, so a single
    pointwise PSIS-LOO is not defined for it (documented in the report).
    """
    require_spec(spec, "historical_joint")
    e = spec.extra
    study_id = e.get("study_id", spec.study_id)
    measure_syms = tuple(e.get("measures", ("basread", "bpvs", "basdig")))
    waves = tuple(e.get("waves", (1, 2, 3)))
    extension_waves = tuple(e.get("extension_waves", ()))

    ctx = make_context(spec, config)

    section_header("Prepare data")
    dataset, measures = _datasets.resolve_dataset(study_id)
    for m in measure_syms:
        if m not in measures:
            raise KeyError(f"measure {m!r} not registered for study {study_id!r}")
    panel = load_longitudinal_panel(
        dataset,
        [measures[m] for m in measure_syms],
        waves=waves,
        complete_case=True,
        extension_waves=extension_waves,
    )
    ctx.prepared = panel
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_rlm_joint_growth_model(
        panel,
        measures=measure_syms,
        eta_prior_sigma=e.get("eta_prior_sigma", 1.5),
        sigma_subject_prior_sigma=e.get("sigma_subject_prior_sigma", 1.0),
        kappa_prior_sigma=e.get("kappa_prior_sigma", 50.0),
        lkj_eta=e.get("lkj_eta", 2.0),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    section_header("Prior predictive")
    _diag.run_prior_predictive(ctx, draws=1000)
    # One likelihood node per measure, so emit one check per measure rather than a
    # pooled overlay: these scales have different maxima and pooling their counts has
    # no interpretable predictive distribution (same reasoning as the joint family's
    # symbol-suffixed checks).
    for _sym in measure_syms:
        _diag.save_prior_predictive_plot(
            ctx,
            _sym,
            node=f"score_{_sym}",
            filename_stem=f"prior_predictive_check_{_sym.lower()}",
        )

    run_sampling_and_loo(ctx, compute_loo=False)

    diag_vars = ["eta_cell", "sigma_subject", "kappa", "measure_corr_pairs"]
    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=diag_vars)
    # Power-scaling prior sensitivity on the reported parameters (#381). This family
    # is ``compute_loo=False`` (one likelihood node per measure, so a single pointwise
    # PSIS-LOO is undefined — not a likelihood PyMC cannot evaluate), so the groups
    # psense needs are not attached by the sampling stage and have to be requested
    # here. ``strict=False`` because psense is a secondary diagnostic and must not
    # crash a fit: today both groups are in fact refused, but by a *naming* seam
    # rather than an intractable likelihood — the model draws
    # ``pm.LKJCorr("measure_corr_chol", ...)`` and PyMC stores its value variable as
    # ``measure_corr_chol_cholesky``, which ``get_untransformed_name`` mangles (see
    # notes/202607261700-psense-coverage-backfill.md and the upstream draft in
    # notes/assets/). That is plausibly fixable upstream; when it is, this call site
    # needs no change. Meanwhile the fit degrades to a warning and gets no psense,
    # which is a *measured and declined* exemption rather than the silent absence it
    # was before.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=diag_vars)

    run_ppc(ctx, var_names=[f"score_{m}" for m in measure_syms])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=diag_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    _diag.save_prior_posterior_plot(ctx, var_names=diag_vars)

    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0
    post = ctx.trace.posterior

    # --- Cross-measure correlation of stable child levels (the headline) ----
    section_header("Cross-measure correlation")
    corr_draws = post["measure_corr"]
    mnames = [str(m) for m in post["measure"].values]
    corr_df = pd.DataFrame(
        corr_draws.mean(dim=("chain", "draw")).values, index=mnames, columns=mnames
    )
    save_table(ctx, "measure_correlation", corr_df, index=True)
    corr_stacked = corr_draws.stack(sample=("chain", "draw"))
    labels = {
        m: str(measures[m].label) if m in measures else m for m in mnames
    }
    corr_rows = []
    for i, mi in enumerate(mnames):
        for j, mj in enumerate(mnames):
            if j <= i:
                continue
            pair = np.asarray(
                corr_stacked.isel(measure=i, measure_b=j).values
            ).reshape(-1)
            corr_rows.append(
                {
                    "measure_i": mi,
                    "measure_j": mj,
                    "label_i": labels[mi],
                    "label_j": labels[mj],
                    "median": float(np.median(pair)),
                    "mean": float(np.mean(pair)),
                    "lo": float(np.quantile(pair, lo_q)),
                    "hi": float(np.quantile(pair, 1 - lo_q)),
                    "lo50": float(np.quantile(pair, 0.25)),
                    "hi50": float(np.quantile(pair, 0.75)),
                    "prob_pos": float(np.mean(pair > 0)),
                }
            )
    corr_summary_df = pd.DataFrame(corr_rows)
    save_table(ctx, "measure_correlation_summary", corr_summary_df)
    print_table(
        ranked_dataframe_table(
            corr_summary_df,
            title=f"Between-child cross-measure correlations - {int(hdi * 100)}% CI",
            columns=["label_i", "label_j", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # --- Per-measure fitted cells + growth (shared historical summaries) ----
    section_header("Per-measure growth summaries")
    # One estimand-scale prior row per measure per contrast (#381), accumulated
    # across the loop so the joint fit writes a single table rather than each
    # measure overwriting the last.
    _pf_rows: list[dict[str, object]] = []
    for m in measure_syms:
        label = measures[m].label
        baseline = _historical.observed_baseline(panel, m, label)
        save_table(
            ctx, f"observed_complete_case_baseline_{m}", baseline, register=False
        )
        cells = _historical.cell_summary(
            ctx.trace,
            panel,
            m,
            label,
            baseline,
            mean_var=f"mean_items_{m}",
            fitted_var=f"fitted_mean_items_obs_{m}",
        )
        save_table(ctx, f"posterior_cell_summary_{m}", cells, register=False)
        growth = _historical.growth_summary(
            ctx.trace, panel, m, fitted_var=f"fitted_mean_items_obs_{m}"
        )
        save_table(ctx, f"posterior_growth_summary_{m}", growth)
        _pf_rows.extend(
            growth_contrast_pushforward_rows(
                ctx,
                panel,
                m,
                fitted_var=f"fitted_mean_items_obs_{m}",
                prefix=f"{m}:",
            )
        )
    write_prior_pushforward(ctx, _pf_rows)

    write_run_metadata(
        ctx,
        extra={
            "study_id": study_id,
            "measures": list(measure_syms),
            "measure_labels": {m: measures[m].label for m in measure_syms},
            "waves": list(waves),
            "extension_waves": list(extension_waves),
            "n_subjects": panel.n_subjects,
            "loo_elpd": None,
        },
    )
    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Correlated-domain-factor measurement model (LRPMM01, #134)
# ---------------------------------------------------------------------------


_DEFAULT_DOMAINS = {
    "vocabulary": ("R", "E"),
    "code": ("L", "B"),
    "grammar": ("F", "T"),
}


def fit_correlated_factor(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Correlated-domain-factor measurement model (LRPMM01, #134).

    Fits a reflective CFA with correlated vocabulary / code / grammar factors over
    the standardised T1 skill indicators, plus a structural Beta-Binomial leg for
    the reading-gain outcome, and reports the loadings / communalities, the factor
    correlation matrix, and the measurement-error-corrected factor->gain slopes.
    A triangulation / measurement model, not causal (every factor->gain slope is a
    latent-ability-confounded adjusted association; #115 ID-2).
    """
    require_spec(spec, "corr_factor")

    # #383 settings coherence, checked BEFORE make_context resets the output
    # directory (the #455 principle): each loading parameterisation has knobs the
    # other would silently ignore, so a spec mixing them is declaring settings the
    # fitted model does not use.
    _loading_prior = str(
        spec.extra.get(
            "loading_prior",
            default_of(_factories.build_correlated_factor_model, "loading_prior"),
        )
    )
    if _loading_prior not in {"communality", "free"}:
        raise ValueError(
            f"Spec {spec.model_id}: loading_prior must be 'communality' or 'free'; "
            f"got {_loading_prior!r}"
        )
    _free_knobs = sorted(
        k for k in ("loading_mu", "loading_sigma", "residual_sigma") if k in spec.extra
    )
    _comm_knobs = sorted(k for k in ("comm_alpha", "comm_beta") if k in spec.extra)
    if _loading_prior == "communality" and _free_knobs:
        raise ValueError(
            f"Spec {spec.model_id}: {_free_knobs} only apply to "
            "loading_prior='free'; the communality parameterisation would silently "
            "ignore them. Set loading_prior='free' or drop the knobs."
        )
    if _loading_prior == "free" and _comm_knobs:
        raise ValueError(
            f"Spec {spec.model_id}: {_comm_knobs} only apply to "
            "loading_prior='communality'; the free parameterisation would silently "
            "ignore them. Drop the knobs or use the default parameterisation."
        )

    ctx = make_context(spec, config)
    # The correlated-factor CFA is a small-n latent model; even with the factor
    # scores marginalised out of the measurement likelihood a few boundary
    # divergences survive at the tier-default target_accept, so lift it via the spec
    # (the strict gate requires zero), as the horseshoe fit does for its funnel.

    section_header("Prepare data")
    domains = {
        k: tuple(v) for k, v in (spec.extra.get("domains") or _DEFAULT_DOMAINS).items()
    }
    outcome = spec.outcome_symbol or "W"
    structural_covs = tuple(spec.extra.get("structural_covariates", ("blocks",)))
    # #228 item 14 (errors-in-variables mechanism): optionally regress the outcome on a
    # SUBSET of the fitted factors (e.g. just "code") and/or add the randomised arm G as
    # an adjusted-association covariate. Defaults reproduce mm-001/101 exactly.
    _sf = spec.extra.get("structural_factors")
    structural_factors = tuple(_sf) if _sf is not None else None
    use_group = bool(spec.extra.get("use_group", False))
    indicator_syms = tuple(dict.fromkeys(s for v in domains.values() for s in v))
    measure_outcomes = tuple(dict.fromkeys((outcome, *indicator_syms)))
    prepared = load_and_prepare(
        phase_mode="span",
        post_time=int(spec.extra.get("post_time", 4)),
        outcomes=measure_outcomes,
        covariates=structural_covs,
    )
    ctx.prepared = prepared
    # A structural covariate can go constant on the fitted span rows — e.g. an
    # ``erbto_missing`` indicator that is all-zero because phonological memory is
    # observed for every fitted child at t1 — so the loader drops it. Re-filter to the
    # covariates actually present, mirroring the mechanism/mediation pipelines'
    # #247/#258 re-filter, so the factory is not asked for a coefficient on a dropped
    # covariate (it raises KeyError otherwise) and the effective set is honest.
    _dropped_structural = tuple(c for c in structural_covs if c not in prepared.covariates)
    if _dropped_structural:
        structural_covs = tuple(c for c in structural_covs if c in prepared.covariates)
        rprint(
            "[yellow]fit_correlated_factor: dropped constant structural covariate(s) "
            f"{list(_dropped_structural)} (not in prepared.covariates on the fitted "
            "rows).[/yellow]"
        )
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_correlated_factor_model(
        prepared,
        outcome_symbol=outcome,
        domains=domains,
        structural_covariates=structural_covs,
        structural_factors=structural_factors,
        use_group=use_group,
        use_age=spec.extra.get("use_age", True),
        loading_prior=_loading_prior,
        comm_alpha=spec.extra.get(
            "comm_alpha",
            default_of(_factories.build_correlated_factor_model, "comm_alpha"),
        ),
        comm_beta=spec.extra.get(
            "comm_beta",
            default_of(_factories.build_correlated_factor_model, "comm_beta"),
        ),
        loading_mu=spec.extra.get(
            "loading_mu",
            default_of(_factories.build_correlated_factor_model, "loading_mu"),
        ),
        loading_sigma=spec.extra.get(
            "loading_sigma",
            default_of(_factories.build_correlated_factor_model, "loading_sigma"),
        ),
        residual_sigma=spec.extra.get(
            "residual_sigma",
            default_of(_factories.build_correlated_factor_model, "residual_sigma"),
        ),
        predictor_slope_sigma=spec.extra.get(
            "predictor_slope_sigma",
            default_of(
                _factories.build_correlated_factor_model, "predictor_slope_sigma"
            ),
        ),
        focal_slope_sigma=spec.extra.get(
            "focal_slope_sigma",
            default_of(_factories.build_correlated_factor_model, "focal_slope_sigma"),
        ),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    summary_vars = [
        # ``communality`` is the free RV under the default communality
        # parameterisation (#383) and a Deterministic under the legacy free pair;
        # either way it is a reported quantity, so gate it explicitly alongside
        # the derived lambda_load / sigma_indicator.
        "alpha", "gamma_own", "kappa", "beta_factor", "lambda_load", "sigma_indicator",
        "communality",
        # The headline factor correlations MUST be in the gated set: they are what
        # the report releases, and the global checks (divergences, BFMI) are not a
        # substitute for parameter-specific R-hat / ESS on them. ``factor_corr``
        # itself is unusable for this — its constant unit diagonal has undefined
        # R-hat and zero variance — so the factory exposes the unique off-diagonals
        # as ``factor_corr_pairs``. ``factor_z`` is the latent-score offset the
        # structural leg consumes; gate it too.
        "factor_z",
    ]
    # Only present when there are >= 2 domains (a single factor has no off-diagonal).
    if len(domains) > 1:
        summary_vars.append("factor_corr_pairs")
    if spec.extra.get("use_age", True):
        summary_vars.append("beta_age")
    summary_vars += [f"beta_{c}" for c in structural_covs]
    if use_group:
        summary_vars.append("beta_G")

    section_header("Prior predictive")
    # Draw the full prior, not just the two observed nodes (#381). Restricting
    # ``var_names`` to ``["Z_obs", "y_post"]`` left the persisted ``prior`` group
    # completely empty, so ``save_prior_posterior_plot`` below had nothing to
    # overlay and these three fits shipped with no prior-vs-posterior figure at
    # all — the one measurement family the prior-analysis review most wanted to
    # see. The default (all free RVs + deterministics + observed nodes) is what
    # every other family uses, and ``run_prior_predictive`` falls back to the
    # minimal set on failure.
    _diag.run_prior_predictive(ctx, draws=1000)
    _diag.save_prior_predictive_plot(ctx, outcome, node="y_post")

    # Two observed nodes (the indicator matrix Z_obs + the structural y_post) make
    # a single-target PSIS-LOO ambiguous, so LOO is skipped here as in the
    # mediation family; this is a measurement / triangulation model, not a
    # predictive one, and #134 turns on the loadings / communalities, not on LOO.
    run_sampling_and_loo(ctx, compute_loo=False)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=summary_vars)
    # Power-scaling prior sensitivity (#381): this family does not compute LOO,
    # so add the log groups explicitly, then power-scale the reported parameters.
    _diag.compute_log_likelihood_and_prior(ctx, strict=False)
    _diag.run_psense(ctx, var_names=summary_vars)

    # Sample both observed nodes (the indicator matrix + the structural outcome).
    # These are two SEPARATE checks, not a joint predictive draw: the factor scores
    # condition on the observed indicator data (``Z_d``), not on the replicated
    # ``Z_obs``, so a replicated indicator is independent of the replicated factor
    # it loads on. ``Z_obs`` is a marginal check of the measurement covariance;
    # ``y_post`` is a check of the structural leg *conditional on the observed
    # indicators*. Together they do not certify the joint model. See the
    # predictive-simulation caveat in ``build_correlated_factor_model``.
    run_ppc(ctx, var_names=["Z_obs", "y_post"])

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=summary_vars)
    _diag.run_extended_diagnostics(ctx)
    _diag.save_trace(ctx)
    # Indicator-scale prior check (#381). Only ``Z_obs`` is the indicator matrix;
    # ``y_post`` is the structural outcome and is covered by the ordinary
    # prior-predictive plot above. AFTER save_trace, which is what attaches the
    # prior/prior_predictive groups to ctx.trace on a fresh fit: called earlier,
    # the check found no prior_predictive group and skipped silently (#383).
    write_indicator_prior_check(ctx, ["Z_obs"])
    _diag.save_prior_posterior_plot(ctx, var_names=summary_vars)

    post = ctx.trace.posterior
    hdi = ctx.reporting.ci_prob
    lo_q = (1.0 - hdi) / 2.0

    # --- Loadings + communalities (the measurement headline) ---
    section_header("Loadings + communalities")
    from language_reading_predictors.statistical_models import (
        corr_factor_summaries as _cf_summaries,
    )

    # The RLI factor loadings live under ``lambda_load`` (the Byrne model uses
    # ``loading``); the residual sigma is free, so lambda is a coefficient on the
    # unit-variance factor, not in general a correlation — the standardised loading /
    # indicator-factor correlation reported alongside is sqrt(communality).
    load_df = _cf_summaries.loadings_communalities_table(
        post, domains, lo_q=lo_q, loading_var="lambda_load"
    )
    save_table(ctx, "loadings_summary", load_df)
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings, correlations + communalities - {int(hdi * 100)}% CI (equal-tailed)",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Factor correlation matrix ---
    section_header("Factor correlation")
    corr_df = _cf_summaries.factor_correlation_matrix(post)
    # Domain names are also used by the structural leg below (beta_factor dims).
    dnames = [str(d) for d in post["domain"].values]
    save_table(ctx, "factor_correlation", corr_df, index=True)
    # The bare mean matrix above is kept for the heatmap, but the house rule is
    # "never a bare point estimate": persist each unique off-diagonal pair with a
    # posterior mean, equal-tailed interval and tail probability alongside it.
    corr_summary_df = _cf_summaries.factor_correlation_pairs(post, lo_q=lo_q)
    save_table(ctx, "factor_correlation_summary", corr_summary_df)

    # --- Structural slopes: factor -> reading gain (adjusted associations) ---
    section_header("Structural slopes (factor -> gain)")
    # The structural leg regresses on all domain factors (beta_factor dims "domain")
    # unless structural_factors isolated a subset (dims "struct_domain", #228 item 14).
    struct_names = list(structural_factors) if structural_factors is not None else dnames
    _bf_dim = "struct_domain" if structural_factors is not None else "domain"
    struct_rows = [
        _coef_row(f"beta_{d}", post["beta_factor"].isel({_bf_dim: k}).values, hdi)
        for k, d in enumerate(struct_names)
    ]
    extra_terms = (
        (["beta_G"] if use_group else [])
        + (["beta_age"] if spec.extra.get("use_age", True) else [])
        + [f"beta_{c}" for c in structural_covs]
    )
    struct_rows += [_coef_row(t, post[t].values, hdi) for t in extra_terms]
    struct_df = pd.DataFrame(struct_rows)
    save_table(ctx, "structural_summary", struct_df)
    print_table(
        ranked_dataframe_table(
            struct_df,
            title=(
                f"Structural slopes (factor -> gain; adjusted associations) - "
                f"{int(hdi * 100)}% CI"
            ),
            columns=["coefficient", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    write_run_metadata(
        ctx,
        extra={
            "domains": {k: list(v) for k, v in domains.items()},
            "loadings_summary": load_df.to_dict("records"),
            "factor_correlation": corr_df.to_dict(),
            "structural_summary": struct_df.to_dict("records"),
        },
    )

    return finalize_report(ctx)


# ---------------------------------------------------------------------------
# Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313)
# ---------------------------------------------------------------------------


_LCF_DEFAULT_DOMAINS = {
    "vocabulary": ("R", "E", "TR", "TE"),
    "code": ("L", "B"),
    "grammar": ("F", "T"),
}


# The LCF exact child-level log-likelihood and constrained-scale log-prior
# recovery are substantive numerical algorithms, isolated in ``lcf_inference``
# so they are testable independently of report publication (#394, pillar 7).
# Re-exported here under their historical private names for existing callers.
_lcf_child_log_likelihood = _lcf_inference.child_log_likelihood
_lcf_log_prior = _lcf_inference.log_prior


def _lcf_stitch_loo(ctx: StatisticalFitContext, built) -> None:
    """Pointwise PSIS-LOO for the longitudinal CFA (custom, per-child stitch).

    The masked-cell likelihood is one ``MvNormal`` per observed-cell pattern, so
    there is no single observed node ``az.loo`` can key on. Compute each pattern
    node's per-child log-likelihood, stitch them into one ``(chain, draw, child)``
    array over all children, and run pointwise LOO on that — the honest per-child
    predictive score an invariance comparison (#313) would use. Attach the exact
    constrained-scale prior terms through the companion workaround above. As in the
    other LOO-enabled families, a failure is fatal: a reporting run must not silently
    complete without the likelihood, prior and predictive diagnostics its output
    contract requires.
    """
    import arviz as az
    import xarray as xr

    stitched = _lcf_child_log_likelihood(ctx.trace, built)
    if "log_likelihood" not in ctx.trace.children:
        ctx.trace["log_likelihood"] = xr.Dataset()
    ctx.trace.log_likelihood["lcf_child"] = stitched
    ctx.trace["log_prior"] = _lcf_log_prior(ctx.trace, ctx.model)
    ctx.loo = az.loo(ctx.trace, var_name="lcf_child", pointwise=True)
    _report.write_loo_summary(ctx)
    print_loo_row(ctx)


# LCF descriptive/comparison summaries live in ``lcf_summaries`` (#394 pillar 6);
# re-exported here under their historical private names for existing callers.
_lcf_observed_domain_corr = _lcf_summaries.observed_domain_corr
_lcf_items_scale = _lcf_summaries.items_scale
_lcf_observed_conditional_slope = _lcf_summaries.observed_conditional_slope
_lcf_concurrent_comparison = _lcf_summaries.concurrent_comparison


def fit_longitudinal_corr_factor(
    spec: ModelSpec, config: str = "dev"
) -> StatisticalFitContext:
    """Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313).

    Fits the four-wave extension of the ``corr_factor`` CFA over the child×wave
    panel: correlated vocabulary / code / grammar factors at every timepoint, with a
    trait/state across-wave structure and the factor scores marginalised out. Reports
    the per-wave latent skill correlation matrices, the conditional (partial) latent
    slopes, the loadings / communalities, and a descriptive comparison against the
    observed same-wave correlations (the #312 anchor). The quantities differ in their
    aggregation and conditioning, so no magnitude ordering is required. A measurement
    / triangulation model — every quantity is a descriptive association, never causal.
    """
    require_spec(spec, "long_corr_factor")

    # #383 settings coherence, checked BEFORE make_context resets the output
    # directory (the #455 principle), mirroring fit_correlated_factor: each loading
    # parameterisation has knobs the other would silently ignore, so a spec mixing
    # them is declaring settings the fitted model does not use.
    _loading_prior = str(
        spec.extra.get(
            "loading_prior",
            default_of(
                _factories.build_longitudinal_corr_factor_model, "loading_prior"
            ),
        )
    )
    if _loading_prior not in {"communality", "free"}:
        raise ValueError(
            f"Spec {spec.model_id}: loading_prior must be 'communality' or 'free'; "
            f"got {_loading_prior!r}"
        )
    _free_knobs = sorted(
        k for k in ("loading_sigma", "residual_sigma") if k in spec.extra
    )
    _comm_knobs = sorted(k for k in ("comm_alpha", "comm_beta") if k in spec.extra)
    if _loading_prior == "communality" and _free_knobs:
        raise ValueError(
            f"Spec {spec.model_id}: {_free_knobs} only apply to "
            "loading_prior='free'; the communality parameterisation would silently "
            "ignore them. Set loading_prior='free' or drop the knobs."
        )
    if _loading_prior == "free" and _comm_knobs:
        raise ValueError(
            f"Spec {spec.model_id}: {_comm_knobs} only apply to "
            "loading_prior='communality'; the free parameterisation would silently "
            "ignore them. Drop the knobs or use the default parameterisation."
        )

    ctx = make_context(spec, config)
    # A small-n latent model; even fully marginalised a few boundary divergences can
    # survive at the tier default, so lift target_accept via the spec (as mm-001 does).

    section_header("Prepare data")
    domains = {
        k: tuple(v)
        for k, v in (spec.extra.get("domains") or _LCF_DEFAULT_DOMAINS).items()
    }
    indicators = tuple(dict.fromkeys(s for v in domains.values() for s in v))
    panel = load_wave_panel(outcomes=indicators)
    ctx.prepared = panel
    print_header(ctx)

    section_header("Build model")
    built = _factories.build_longitudinal_corr_factor_model(
        panel,
        domains=domains,
        loading_prior=_loading_prior,
        comm_alpha=spec.extra.get(
            "comm_alpha",
            default_of(_factories.build_longitudinal_corr_factor_model, "comm_alpha"),
        ),
        comm_beta=spec.extra.get(
            "comm_beta",
            default_of(_factories.build_longitudinal_corr_factor_model, "comm_beta"),
        ),
        loading_sigma=spec.extra.get(
            "loading_sigma",
            default_of(_factories.build_longitudinal_corr_factor_model, "loading_sigma"),
        ),
        residual_sigma=spec.extra.get(
            "residual_sigma",
            default_of(_factories.build_longitudinal_corr_factor_model, "residual_sigma"),
        ),
        lkj_eta=spec.extra.get(
            "lkj_eta",
            default_of(_factories.build_longitudinal_corr_factor_model, "lkj_eta"),
        ),
        factor_mean_sigma=spec.extra.get(
            "factor_mean_sigma",
            default_of(
                _factories.build_longitudinal_corr_factor_model, "factor_mean_sigma"
            ),
        ),
        trait_share_a=spec.extra.get(
            "trait_share_a",
            default_of(
                _factories.build_longitudinal_corr_factor_model, "trait_share_a"
            ),
        ),
        trait_share_b=spec.extra.get(
            "trait_share_b",
            default_of(
                _factories.build_longitudinal_corr_factor_model, "trait_share_b"
            ),
        ),
    )
    attach_built(ctx, built)
    render_model_graph(ctx)

    z_nodes = built.extras["z_nodes"]
    summary_vars = [
        # ``communality`` is the free RV under the default pooled-budget
        # parameterisation (#383 follow-up) and a Deterministic under the legacy
        # free pair; either way it is a reported quantity, so gate it explicitly
        # alongside the derived lambda_load / sigma_indicator. ``within_share``
        # is the within-wave share of the pooled unit variance that the budget
        # allocates (lambda**2 + sigma**2 in both modes).
        "lambda_load",
        "sigma_indicator",
        "communality",
        "within_share",
        "trait_share",
        # The headline: gate exactly the released per-wave off-diagonal correlations
        # (the full matrix's constant unit diagonal has undefined R-hat).
        "factor_corr_pairs",
    ]

    section_header("Prior predictive")
    # Dedupe: ``communality`` is itself a free RV under the default
    # parameterisation, so appending it unconditionally would double it. The
    # derived lambda_load / sigma_indicator / within_share are named explicitly —
    # as Deterministics they are no longer covered by the free-RV listing, and the
    # prior-vs-posterior overlay below needs their prior draws.
    prior_vars = list(
        dict.fromkeys(
            [
                *(rv.name for rv in built.model.free_RVs),
                "communality",
                "lambda_load",
                "sigma_indicator",
                "within_share",
                "factor_corr_pairs",
                *z_nodes,
            ]
        )
    )
    _diag.run_prior_predictive(ctx, draws=1000, var_names=prior_vars)
    _diag.save_prior_predictive_dist_overlay(ctx)

    # Automatic single-target LOO is ambiguous with per-pattern observed nodes, so
    # sampling runs without it and the per-child stitch below computes LOO instead.
    run_sampling_and_loo(ctx, compute_loo=False)

    section_header("LOO-PSIS (per-child stitch)")
    _lcf_stitch_loo(ctx, built)

    section_header("Summary diagnostics")
    _diag.summary_diagnostics(ctx, var_names=summary_vars)

    run_ppc(ctx, var_names=z_nodes)

    section_header("Extended diagnostics")
    _diag.write_diagnostics_summary(ctx, var_names=summary_vars)
    _diag.run_extended_diagnostics(ctx, causal_term=None)
    # ``communality`` joins the power-scaled set (#383 follow-up): under the
    # pooled-budget parameterisation it is the free measurement parameter behind
    # the reported loadings table, exactly the place a prior dependence would live.
    _diag.run_psense(ctx, var_names=["factor_corr_pairs", "trait_share", "communality"])
    _diag.save_trace(ctx)
    # Indicator-scale prior check (#381), pooled across the missingness-pattern
    # blocks: each block is its own observed node but the indicators are shared.
    # AFTER save_trace, which is what attaches the prior/prior_predictive groups
    # to ctx.trace on a fresh fit: called earlier, the check found no
    # prior_predictive group and skipped silently (#383).
    write_indicator_prior_check(ctx, z_nodes)
    _diag.save_prior_posterior_plot(ctx, var_names=summary_vars)

    post = ctx.trace.posterior
    hdi = ctx.reporting.ci_prob
    lo_q = (1 - hdi) / 2

    # --- Loadings + communalities (the measurement layer) ---
    section_header("Loadings + communalities")
    dom_of = built.extras["domain_of"]
    load_rows = []
    for j, name in enumerate(str(s) for s in post["indicator"].values):
        lam_d = post["lambda_load"].isel(indicator=j).values.reshape(-1)
        com_d = post["communality"].isel(indicator=j).values.reshape(-1)
        corr_d = np.sqrt(com_d)
        load_rows.append(
            {
                "indicator": name,
                "domain": dom_of.get(name, "?"),
                "loading_mean": float(np.mean(lam_d)),
                "loading_lo": float(np.quantile(lam_d, lo_q)),
                "loading_hi": float(np.quantile(lam_d, 1 - lo_q)),
                "correlation_mean": float(np.mean(corr_d)),
                "communality_mean": float(np.mean(com_d)),
                "communality_lo": float(np.quantile(com_d, lo_q)),
                "communality_hi": float(np.quantile(com_d, 1 - lo_q)),
            }
        )
    load_df = pd.DataFrame(load_rows)
    save_table(ctx, "loadings_summary", load_df)
    print_table(
        ranked_dataframe_table(
            load_df,
            title=f"Loadings + communalities - {int(hdi * 100)}% CI (equal-tailed)",
            columns=[
                "indicator", "domain", "loading_mean", "correlation_mean",
                "communality_mean", "communality_lo", "communality_hi",
            ],
            rank_column=False,
            precision=3,
        )
    )

    # --- Per-wave latent factor correlations (the headline) ---
    section_header("Per-wave latent factor correlations")
    corr_df = _report.longitudinal_factor_correlations(ctx.trace, ci_prob=hdi)
    save_table(ctx, "factor_correlation_by_wave", corr_df)
    print_table(
        ranked_dataframe_table(
            corr_df,
            title=f"Per-wave latent factor correlations - {int(hdi * 100)}% ETI",
            columns=["wave", "domain_i", "domain_j", "mean", "lo", "hi", "prob_pos"],
            rank_column=False,
            precision=3,
        )
    )

    # --- Conditional (partial) latent slopes ---
    section_header("Conditional latent slopes")
    slope_df = _report.longitudinal_conditional_slopes(ctx.trace, ci_prob=hdi)
    save_table(ctx, "latent_conditional_slopes", slope_df)

    # --- Trait / state (across-wave) structure ---
    section_header("Trait / state structure")
    ts_rows = []
    for j, d in enumerate(str(s) for s in post["domain"].values):
        pi_d = post["trait_share"].isel(domain=j).values.reshape(-1)
        ts_rows.append(
            {
                "domain": d,
                "trait_share_mean": float(np.mean(pi_d)),
                "trait_share_lo": float(np.quantile(pi_d, lo_q)),
                "trait_share_hi": float(np.quantile(pi_d, 1 - lo_q)),
            }
        )
    ts_df = pd.DataFrame(ts_rows)
    save_table(ctx, "trait_state_summary", ts_df)

    # --- Latent-versus-observed comparison (#312 triangulation anchor) --------
    section_header("Latent-versus-observed correlation comparison")
    obs_df = _lcf_observed_domain_corr(built)
    xcheck_df = _report.disattenuation_crosscheck(corr_df, obs_df)
    save_table(ctx, "disattenuation_crosscheck", xcheck_df)
    n_latent_below = int((~xcheck_df["latent_ge_observed"]).sum())
    n_latent_at_or_above = len(xcheck_df) - n_latent_below
    rprint(
        "[cyan]Latent-versus-observed comparison: "
        f"{n_latent_below} wave/pair(s) are below and {n_latent_at_or_above} are at "
        "or above the mean observed indicator-pair magnitude. This is a descriptive "
        "gap direction between different estimands, not a pass/fail ordering.[/cyan]"
    )

    # --- Items-scale translation for the headline pairs ---
    section_header("Items-scale translation (selected pairs)")
    items_df = _lcf_items_scale(ctx, built)
    save_table(ctx, "latent_items_slopes", items_df)

    # --- Directed comparison with matching concurrent associations (#312) ---
    section_header("Directed LCF-versus-concurrent comparison")
    concurrent_df = _lcf_concurrent_comparison(ctx, built)
    save_table(ctx, "lcf_concurrent_comparison", concurrent_df)
    n_ca_available = int(concurrent_df["ca_available"].sum())
    if n_ca_available < len(concurrent_df):
        rprint(
            "[yellow]Directed #312 comparison is incomplete: "
            f"{n_ca_available}/{len(concurrent_df)} matching concurrent rows were "
            "found under this output root/config. Fit CA002--006 at the same tier "
            "to populate the missing side.[/yellow]"
        )

    write_run_metadata(
        ctx,
        extra={
            "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
            "domains": {k: list(v) for k, v in domains.items()},
            "invariance": built.extras["invariance"],
            "n_used_children": built.extras["n_used_children"],
            "loadings_summary": load_df.to_dict("records"),
            "factor_correlation_by_wave": corr_df.to_dict("records"),
            "trait_state_summary": ts_df.to_dict("records"),
            # Keep the legacy key for output compatibility; neither count is a gate.
            "disattenuation_reversals": n_latent_below,
            "latent_below_observed_count": n_latent_below,
            "latent_observed_comparison": "descriptive; no required ordering",
            "lcf_concurrent_comparison_rows": int(len(concurrent_df)),
            "lcf_concurrent_comparison_available": n_ca_available,
        },
    )

    return finalize_report(ctx)
