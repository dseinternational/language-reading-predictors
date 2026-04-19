# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared sampling / diagnostics helpers for LRP52-LRP58.

Each LRP model runs the same diagnostic suite:

1. Prior predictive check (1000 draws).
2. Sampling via NUTS (nutpie backend).
3. Summary diagnostics (Rhat, ESS, divergences), trace, energy, pair plots.
4. LOO-PSIS via ArviZ.
5. Posterior predictive draws.

Everything is written to ``context.output_dir`` and the trace persisted as
``trace.nc`` (NetCDF InferenceData).
"""

from __future__ import annotations

import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
)
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)


def run_prior_predictive(
    context: StatisticalFitContext,
    draws: int = 1000,
    var_names: list[str] | None = None,
) -> None:
    with context.model:
        prior = pm.sample_prior_predictive(
            draws=draws,
            var_names=var_names,
            random_seed=context.sampling.random_seed,
        )
    context.prior_samples = prior


def sample_posterior(context: StatisticalFitContext) -> None:
    s = context.sampling
    with context.model:
        trace = pm.sample(
            draws=s.draws,
            tune=s.tune,
            chains=s.chains,
            cores=s.cores,
            target_accept=s.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=s.random_seed,
            progressbar=False,
        )
    context.trace = trace


def compute_log_likelihood_and_loo(context: StatisticalFitContext) -> None:
    with context.model:
        context.trace = pm.compute_log_likelihood(context.trace)
    context.loo = az.loo(context.trace)


def summary_diagnostics(
    context: StatisticalFitContext,
    var_names: list[str] | None = None,
    max_vars_for_pairs: int = 8,
) -> None:
    out = context.output_dir
    os.makedirs(out, exist_ok=True)

    # Narrow to scalar RVs by default so the summary table is readable.
    if var_names is None:
        scalar_vars = []
        for rv in context.model.unobserved_RVs:
            try:
                if int(np.prod(rv.shape.eval())) <= 2:
                    scalar_vars.append(rv.name)
            except Exception:
                continue
        var_names = scalar_vars

    if var_names:
        summary = az.summary(
            context.trace,
            var_names=var_names,
            round_to=3,
            hdi_prob=context.reporting.hdi,
        )
        summary.to_csv(os.path.join(out, "diagnostics.csv"))
        context.tables["diagnostics"] = summary

        hdi_pct = int(round(context.reporting.hdi * 100))
        hdi_lo = f"hdi_{(100 - hdi_pct) / 2:g}%"
        hdi_hi = f"hdi_{100 - (100 - hdi_pct) / 2:g}%"
        wanted = [
            c
            for c in ["mean", "sd", hdi_lo, hdi_hi, "ess_bulk", "ess_tail", "r_hat"]
            if c in summary.columns
        ]
        display_df = summary.reset_index().rename(columns={"index": "variable"})
        print_table(
            ranked_dataframe_table(
                display_df,
                title=f"Posterior diagnostics (HDI {hdi_pct}%)",
                columns=["variable", *wanted],
                rank_column=False,
                precision=3,
            )
        )

    # Trace
    az.plot_trace(context.trace, combined=True, var_names=var_names or None)
    plt.savefig(os.path.join(out, "trace_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Energy
    az.plot_energy(context.trace)
    plt.savefig(os.path.join(out, "energy_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Posterior
    if var_names:
        az.plot_posterior(
            context.trace.posterior,
            var_names=var_names,
            hdi_prob=context.reporting.hdi,
        )
        plt.savefig(
            os.path.join(out, "posterior_plot.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        if len(var_names) <= max_vars_for_pairs:
            az.plot_pair(context.trace, var_names=var_names, kind="kde", divergences=True)
            plt.savefig(
                os.path.join(out, "pair_plot.png"), dpi=300, bbox_inches="tight"
            )
            plt.close()


def sample_posterior_predictive(
    context: StatisticalFitContext,
    var_names: list[str] | None = None,
) -> None:
    with context.model:
        context.trace = pm.sample_posterior_predictive(
            context.trace,
            var_names=var_names,
            extend_inferencedata=True,
            random_seed=context.sampling.random_seed,
            progressbar=False,
        )


def save_trace(context: StatisticalFitContext, filename: str = "trace.nc") -> str:
    path = os.path.join(context.output_dir, filename)
    context.trace.to_netcdf(path)
    return path
