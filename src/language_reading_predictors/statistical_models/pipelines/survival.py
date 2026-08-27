# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Discrete-time off-floor survival orchestration (``kind="survival"``, #230 §5).

``fit_survival`` fits a person-period discrete-time hazard for the *time* a
floored outcome (P or N) takes to come off the floor, generalising the
single-transition off-floor estimand of the ITT floor rule to all four waves.
Treatment enters as an intervention-aligned hazard shift. Under the default
randomised window, tau is a model-based, available-case modified-ITT assignment
contrast in the first interval within the baseline at-floor subgroup (#631
finding 11); the family still releases no causal headline.

This is the reference adoption of the shared primary-fit lifecycle: its
mid-section is one :func:`stages.SharedFitStages.run_primary_fit` call driven by a
:class:`stages.PrimaryFitPlan` (#394 step 4).
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
    reporting as _report,
    survival as _survival,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
    make_context,
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


def _survival_summary(
    trace,
    *,
    ci_prob: float,
    hazard_link: str,
    use_treatment: bool,
    treatment_window: str = "randomised",
) -> pd.DataFrame:
    """Off-floor discrete-time hazard summary (log-hazard, ratio, P>0).

    Reports the treatment hazard term and baseline-covariate slopes on the
    log-hazard scale (with ``exp`` as a hazard ratio under the cloglog link — the
    column is named ``odds_ratio`` under the logistic-hazard sensitivity link,
    where ``exp`` is an odds ratio, 2026-08-21 survival review, finding 4 — and
    ``P(effect > 0)``), plus the per-interval off-floor probability at mean
    covariates on the model's ``hazard_link`` scale. Under the default
    ``treatment_window="randomised"`` the first-interval row is the untreated
    baseline and the later rows are the identified both-arms-treated interval
    hazards; under the legacy pooled comparator the later "untreated" rows are
    prior-mediated extrapolations (no untreated children exist there) and are
    labelled as such. Equal-tailed intervals at ``ci_prob`` with the posterior
    median as the point estimate (the suite convention).
    """
    post = trace.posterior
    lo, hi = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    ratio_col = "hazard_ratio" if hazard_link == "cloglog" else "odds_ratio"

    def _row(term: str, draws: np.ndarray, *, as_ratio: bool) -> dict:
        d = np.asarray(draws).reshape(-1)
        return {
            "term": term,
            "median": float(np.median(d)),
            "ci_low": float(np.quantile(d, lo)),
            "ci_high": float(np.quantile(d, hi)),
            ratio_col: float(np.exp(np.median(d))) if as_ratio else float("nan"),
            "P(>0)": float(np.mean(d > 0)) if as_ratio else float("nan"),
        }

    rows: list[dict] = []
    if use_treatment:
        tau_label = (
            "tau (log hazard contrast, randomised interval t1->t2)"
            if treatment_window == "randomised"
            else "tau (log hazard shift, treated, pooled)"
        )
        rows.append(_row(tau_label, post["tau"].values, as_ratio=True))
    for name in sorted(v for v in post.data_vars if str(v).startswith("beta_")):
        rows.append(_row(f"{name} (log hazard, per SD)", post[name].values, as_ratio=True))

    alpha = post["alpha"].stack(sample=("chain", "draw")).transpose("interval", "sample")
    labels = [str(v) for v in alpha.coords["interval"].values]
    for i, lab in enumerate(labels):
        a = alpha.values[i]
        base = 1.0 - np.exp(-np.exp(a)) if hazard_link == "cloglog" else 1.0 / (1.0 + np.exp(-a))
        if not use_treatment:
            term = f"off-floor prob [{lab}] (no treatment term)"
        elif treatment_window == "randomised":
            term = (
                f"baseline off-floor prob [{lab}] (untreated)"
                if i == 0
                else f"off-floor prob [{lab}] (both arms treated)"
            )
        else:
            term = (
                f"baseline off-floor prob [{lab}] (untreated)"
                if i == 0
                else f"baseline off-floor prob [{lab}] (untreated extrapolation; "
                "no untreated children in this interval)"
            )
        rows.append(
            {
                "term": term,
                "median": float(np.median(base)),
                "ci_low": float(np.quantile(base, lo)),
                "ci_high": float(np.quantile(base, hi)),
                ratio_col: float("nan"),
                "P(>0)": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fit_survival(spec: ModelSpec, config: str = "dev") -> StatisticalFitContext:
    """Discrete-time off-floor survival fit for a floored outcome P / N (#230 §5).

    Fits a person-period discrete-time hazard for the *time* to come off the floor,
    generalising the single-transition off-floor estimand of the LRPITT09/11 floor
    rule to all four waves. Treatment enters as an intervention-aligned hazard shift;
    under the default randomised window tau is an available-case modified-ITT
    assignment contrast in the first interval within the baseline at-floor
    subgroup (#631 finding 11), and the family releases no causal headline.
    """
    require_spec(spec, "survival", outcome=True)

    # Resolve the complete family contract before ``make_context`` can reset the
    # output directory and before preparation reads the RLI data (#394 pillar 4).
    plan = _survival.resolve_survival_run_plan(spec)
    ctx = make_context(spec, config)
    ctx.resolved_plan = plan
    _report.write_model_recipe(ctx)

    section_header("Prepare data")
    panel = _survival.prepare_survival(**plan.prepare_kwargs())
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

    hazard_link = plan.hazard_link
    use_treatment = plan.use_treatment

    section_header("Build model")
    built = _survival.build_survival_model(panel, **plan.factory_kwargs())
    attach_built(ctx, built)
    render_model_graph(ctx)

    diag_vars = plan.diagnostic_vars(panel.covariates)

    # Reference adoption of the shared primary-fit lifecycle (#394 design 2):
    # the invariant sequence lives in ``stages.run_primary_fit`` and the family
    # declares only its execution profile.
    shared_stages().run_primary_fit(
        ctx,
        PrimaryFitPlan(
            diagnostic_vars=diag_vars,
            ppc_var_names=(plan.observation_node,),
            plot_prior_predictive=lambda c: _diag.save_prior_predictive_rate_plot(
                c, plan.outcome_symbol, node=plan.observation_node
            ),
            extended_term=plan.focal_term,
            compute_loo=plan.compute_loo,
        ),
    )

    section_header("Off-floor hazard summary")
    summary = _survival_summary(
        ctx.trace,
        ci_prob=ctx.reporting.ci_prob,
        hazard_link=hazard_link,
        use_treatment=use_treatment,
        treatment_window=plan.treatment_window,
    )
    save_table(ctx, "survival_summary", summary)
    tau_reading = (
        "tau = available-case modified-ITT interval-1 assignment contrast "
        "(at-floor subgroup)"
        if plan.treatment_window == "randomised"
        else "pooled tau is prior-mediated beyond interval 1; prognostic"
    )
    print_table(
        ranked_dataframe_table(
            summary,
            title=(
                f"Off-floor discrete-time hazard ({plan.outcome_symbol}, {hazard_link}); "
                f"positive = raises Pr(off-floor); {tau_reading}"
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
