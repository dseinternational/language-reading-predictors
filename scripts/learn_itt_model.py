#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Explore the registered ITT-001 model with synthetic scores.

See docs/learning/itt-model-walkthrough.md. This example builds the production
model and uses its summary functions. Its results describe simulated children.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit

from language_reading_predictors import paths
from language_reading_predictors.data_variables import Variables
from language_reading_predictors.statistical_models.itt import (
    build_itt_from_plan,
    resolve_itt_run_plan,
)
from language_reading_predictors.statistical_models.lrp_rli_itt_001 import SPEC
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.posteriors import REPORTING_CI_PROB
from language_reading_predictors.statistical_models.ppc_audit import (
    score_ppc_distribution_shape,
)
from language_reading_predictors.statistical_models.predictive_checks import (
    prior_pushforward,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    logit_safe,
    standardise,
)
from language_reading_predictors.statistical_models.priors import priors_table
from language_reading_predictors.statistical_models.summaries.itt import (
    _itt_ame_draws,
    tau_summary_itt,
)


def synthetic_scores(n_children: int, *, seed: int) -> pd.DataFrame:
    """One baseline and one follow-up score per simulated child."""
    rng = np.random.default_rng(seed)
    measure = MEASURES[SPEC.outcome_symbol]
    n_items = measure.n_trials
    group = rng.permutation(np.arange(n_children) % 2)
    age = rng.uniform(60, 120, size=n_children)
    age_standardised, _ = standardise(age)
    baseline = rng.binomial(n_items, expit(rng.normal(-0.4, 0.9, n_children)))
    baseline_logit = logit_safe(baseline, n_items)

    mean_probability = expit(-0.1 + baseline_logit + 0.25 * group + 0.05 * age_standardised)
    concentration = 25.0
    child_probability = rng.beta(
        mean_probability * concentration,
        (1 - mean_probability) * concentration,
    )
    follow_up = rng.binomial(n_items, child_probability)
    rows = []
    for child in range(n_children):
        for wave, score in ((1, baseline[child]), (2, follow_up[child])):
            rows.append(
                {
                    Variables.SUBJECT_ID: f"synthetic-{child + 1:03}",
                    Variables.TIME: wave,
                    Variables.GROUP: 1 if group[child] else 2,
                    Variables.AGE: age[child] + 6 * (wave - 1),
                    measure.column: int(score),
                }
            )
    return pd.DataFrame(rows)


def draw_score_check(observed: np.ndarray, replicated: np.ndarray, output: Path) -> None:
    """Compare score distributions, including the range of replicated scores."""
    fig = Figure(figsize=(7, 4))
    ax = fig.subplots()
    bins = np.arange(MEASURES[SPEC.outcome_symbol].n_trials + 2) - 0.5
    ax.hist(replicated.ravel(), bins=bins, density=True, alpha=0.5, label="Simulated scores")
    ax.hist(observed, bins=bins, density=True, histtype="step", linewidth=2, label="Observed synthetic scores")
    ax.set(xlabel="Items answered correctly", ylabel="Proportion per item bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=paths.output_root() / "learning" / "itt-001")
    parser.add_argument("--children", type=int, default=80)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--prior-only", action="store_true", help="inspect assumptions before fitting")
    args = parser.parse_args()
    if args.children < 4 or args.chains < 2 or min(args.draws, args.tune) < 1:
        parser.error("use at least four children, two chains, and positive draws and tune")

    plan = resolve_itt_run_plan(SPEC)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "synthetic_scores.csv"
    synthetic_scores(args.children, seed=args.seed).to_csv(data_path, index=False)
    prepared = load_and_prepare(path=data_path, **plan.prepare_kwargs())
    built = build_itt_from_plan(plan, prepared, effective_adjustment=())
    n_items = built.prepared.n_trials[plan.outcome_symbol]
    observed = built.prepared.post_counts[plan.outcome_symbol]
    priors_table(built.model).to_csv(output / "priors.csv", index=False)
    print(f"Synthetic example: {prepared.n_children} children, {n_items} items, files in {output}")

    with built.model:
        prior = pm.sample_prior_predictive(draws=500, random_seed=args.seed)
    prior_summary = prior_pushforward(
        prior,
        G=built.prepared.G,
        n_trials=n_items,
        ci_prob=REPORTING_CI_PROB,
    )
    pd.DataFrame([prior_summary]).to_csv(output / "prior_effect_summary.csv", index=False)
    draw_score_check(observed, prior.prior_predictive["y_post"].values, output / "prior_score_check.png")
    if args.prior_only:
        return

    with built.model:
        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.chains,
            target_accept=0.95,
            nuts_sampler="nutpie",
            random_seed=args.seed,
            progressbar=False,
        )
        pm.sample_posterior_predictive(
            trace,
            var_names=["y_post"],
            extend_inferencedata=True,
            random_seed=args.seed,
            progressbar=False,
        )

    diagnostics = az.summary(trace, var_names=["alpha", "tau", "gamma_own", "gamma_A", "kappa"], kind="diagnostics")
    diagnostics.to_csv(output / "sampling_diagnostics.csv")
    print(diagnostics.to_string())
    print(f"Divergent transitions: {int(trace.sample_stats['diverging'].sum())}")
    replicated = trace.posterior_predictive["y_post"].stack(sample=("chain", "draw"))
    score_ppc_distribution_shape(
        plan.outcome_symbol,
        replicated.transpose("sample", "obs_id").values,
        observed,
        n_trials=n_items,
        ci_prob=0.95,
    ).to_csv(output / "posterior_predictive_checks.csv", index=False)
    draw_score_check(observed, replicated.values, output / "posterior_score_check.png")

    # Calculate the two conditions within each draw before summarising differences.
    _, effect_probability = _itt_ame_draws(trace, G=built.prepared.G)
    effect_items = effect_probability * n_items
    pd.DataFrame({"effect_items": effect_items}).to_csv(output / "effect_draws.csv", index=False)
    summary = tau_summary_itt(trace, G=built.prepared.G, ci_prob=REPORTING_CI_PROB)
    pd.DataFrame([summary]).to_csv(output / "treatment_summary.csv", index=False)
    lower, upper = np.quantile(effect_items, ((1 - REPORTING_CI_PROB) / 2, (1 + REPORTING_CI_PROB) / 2))
    print(
        f"Synthetic assigned-arm difference: median {np.median(effect_items):.2f} items; "
        f"89% equal-tailed interval {lower:.2f} to {upper:.2f} items"
    )


if __name__ == "__main__":
    main()
