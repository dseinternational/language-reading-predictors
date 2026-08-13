# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Multivariate latent growth-curve orchestration (``kind="growth"``, LRP69/70).

``fit_growth`` characterises each verbal and reading measure's within-child
trajectory across the four RLI waves and reports whether baseline non-verbal
ability predicts trajectory shape — ``gamma`` on the growth rate, ``delta`` on the
baseline level — optionally coupling the slopes through a rank-1 shared
growth-tempo factor. Every non-randomised term is an adjusted, latent-ability-
confounded association, never causal (locked DAG,
``notes/202606231600-dag-revision-consolidated.md``).
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
from language_reading_predictors.statistical_models.figure_artifacts import (
    save_forest_plot,
    write_panel_child_fit,
    write_panel_trajectory,
)
from language_reading_predictors.statistical_models.growth import (
    resolve_growth_run_plan,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_wave_panel,
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

    # One check per measure: ``y_obs`` flattens (child, wave, outcome) into a single
    # vector, so a lone overlay would pool scales with different maxima. The first
    # outcome keeps the unsuffixed filename the report partial expects.
    def _plot_prior_predictive(c: StatisticalFitContext) -> None:
        for index, symbol in enumerate(outcomes):
            _diag.save_prior_predictive_plot(
                c,
                symbol,
                node="y_obs",
                filename_stem=(
                    "prior_predictive_check"
                    if index == 0
                    else f"prior_predictive_check_{symbol.lower()}"
                ),
            )

    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=tuple(diag_vars),
            ppc_var_names=("y_obs",),
            plot_prior_predictive=_plot_prior_predictive,
        ),
    )
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
