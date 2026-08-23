# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dispersion prior-family sensitivity for the high-denominator ITT outcomes.

The registered suite puts ``kappa ~ HalfNormal(50)`` on the Beta-Binomial
concentration. That prior is candidly documented as only partly permissive, and
the 2026-08-22 ITT audit (finding 5) quantified what it excludes: the
Beta-Binomial variance inflation over Binomial is ``(n + kappa) / (1 + kappa)``,
so at the prior median ``kappa ~ 33.7`` the *prior* already enforces about 2.1x
inflation at ``n = 40``, 3.3x at ``n = 80`` and 5.9x at ``n = 170``. Coming within
10% of Binomial at ``n = 170`` needs ``kappa > 1689``, which has effectively zero
mass. The near-Binomial limit — for a bounded count, the perfectly ordinary
hypothesis "this measure shows no extra-Binomial dispersion" — is off the table
*a priori*, and the registered ``kappa_sigma`` sweep cannot test it either: it
covers only L and W, and even its widest ``HalfNormal(200)`` gives the region
zero prior mass.

This sweep tests it, by varying the prior **family** rather than its scale. The
alternative is the dispersion-scale parameterisation the RLM historical families
already use, ``1 / sqrt(kappa) ~ HalfNormal(sigma)``, which does reach
``kappa >> n``. Measured over 400k draws, P(variance within 10% of Binomial):

======================================  ======  ======  =======
prior                                    n=79    n=80    n=170
======================================  ======  ======  =======
``HalfNormal(50)`` on kappa (registered)  0.000   0.000   0.000
``HalfNormal(200)`` on kappa (sweep max)  0.000   0.000   0.000
``HalfNormal(0.25)`` on 1/sqrt(kappa)     0.114   0.113   0.078
``HalfNormal(0.50)`` on 1/sqrt(kappa)     0.058   0.057   0.039
======================================  ======  ======  =======

Scope is every graded high-denominator ITT outcome: **R** and **E**
(``n_trials`` 170), **EI** (80) and **W** (79). W is here despite already having a
``kappa_sigma`` sweep, because that sweep varies the prior's *scale* and this
finding is about its *family* — the table above shows ``HalfNormal(200)``, the
widest cell that sweep reaches, giving the near-Binomial region 0.000 mass at
n = 79 just as ``HalfNormal(50)`` does. W had therefore never been tested against
this hypothesis, which matters because it is the suite's model of record
(``lrp-rli-itt-010``, five registered fits). P (92) is excluded: its floor-rule
headline is a Bernoulli off-floor indicator with no ``kappa`` at all. Its flagged
graded secondary does carry one, and is noted as out of scope here rather than
silently omitted — it is an explicitly exploratory sub-fit, not a headline.

Deliberately a **separate artefact**, written to
``output/statistical_models/dispersion_prior_sensitivity/``, exactly as the P/N
floor grid is kept out of the standard sweep. The 44-cell
``tau_prior_sensitivity.csv`` grid is bound to registered primaries by hash and
checked against ``sensitivity._standard_expected_cells``; adding cells to it
would invalidate stored sweeps. Nothing here feeds a release gate — it is
recorded evidence about a prior choice, not a pass/fail.

Usage::

    python scripts/dispersion_prior_sensitivity.py                    # dev (fast)
    python scripts/dispersion_prior_sensitivity.py --config reporting
    python scripts/dispersion_prior_sensitivity.py --outcomes R --config test

Read the output as: does the treatment effect move when the model is *allowed*
to conclude there is no extra-Binomial dispersion? The first run (R, E, EI) found
the answer is outcome-specific — no AME moved materially anywhere, but for E the
registered prior was genuinely binding, its concentration posterior moving from
126 to 475 with predictive calibration improving at both levels, while R and EI
were unaffected. E's registered models now declare the dispersion-scale prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
from rich.console import Console

import dse_research_utils.statistics.models.sampling as _sampling
from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.factories import build_itt_model
from language_reading_predictors.statistical_models.itt import (
    prepare_itt_data,
    resolve_itt_run_plan,
)
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.pipelines.itt import load_and_prepare
from language_reading_predictors.statistical_models.sampling_quality import (
    sampling_quality,
)

_console = Console()

OUTPUT_SUBDIR = "dispersion_prior_sensitivity"
FILENAME = "dispersion_prior_sensitivity.csv"

#: Graded outcomes whose denominator makes the enforced-overdispersion floor
#: material, with the registered primary each cell is matched to.
DISPERSION_SENSITIVITY_MODEL_IDS = {
    "W": "lrp-rli-itt-010",
    "R": "lrp-rli-itt-005",
    "E": "lrp-rli-itt-006",
    "EI": "lrp-rli-itt-029",
}

#: ``(family, sigma)``. The first is the registered prior, refitted here so the
#: comparison is like-for-like within one run rather than against a stored trace
#: sampled under different settings.
DISPERSION_SENSITIVITY_CELLS = (
    ("halfnormal_concentration", 50.0),
    ("halfnormal_inverse_sqrt", 0.25),
    ("halfnormal_inverse_sqrt", 0.5),
)


def _registered_spec(symbol: str):
    import importlib

    model_id = DISPERSION_SENSITIVITY_MODEL_IDS[symbol]
    module = importlib.import_module(
        "language_reading_predictors.statistical_models."
        + model_id.replace("-", "_").replace("lrp_rli", "lrp_rli")
    )
    return module.SPEC


def _variance_inflation(kappa: np.ndarray, n_trials: int) -> np.ndarray:
    """Beta-Binomial variance as a multiple of the Binomial variance."""
    return (n_trials + kappa) / (1.0 + kappa)


def _fit_cell(prepared, symbol, family, sigma, sampling, seed):
    built = build_itt_model(
        prepared,
        outcome_symbol=symbol,
        cross_symbols=(),
        use_age_linear=True,
        use_own_baseline=True,
        kappa_sigma=sigma,
        kappa_prior_family=family,
    )
    with built.model:
        trace = pm.sample(
            draws=sampling.draws,
            tune=sampling.tune,
            chains=sampling.chains,
            cores=sampling.cores,
            target_accept=sampling.target_accept,
            random_seed=seed,
            nuts_sampler="nutpie",
            progressbar=False,
        )
        trace = pm.sample_posterior_predictive(
            trace,
            extend_inferencedata=True,
            random_seed=seed,
            progressbar=False,
        )
    return built, trace


def _row(symbol, family, sigma, built, trace, ci_prob):
    n_trials = MEASURES[symbol].n_trials
    G = np.asarray(built.prepared.G, dtype=float)
    _tau, ame = _report._itt_ame_draws(trace, G=G)
    kappa = np.asarray(trace.posterior["kappa"].values, dtype=float).ravel()
    inflation = _variance_inflation(kappa, n_trials)
    lo_q, hi_q = (1 - ci_prob) / 2, 1 - (1 - ci_prob) / 2
    signals = sampling_quality(trace)

    # The suite's own coverage statistic, so a cell is comparable with the
    # ``ppc_summary.csv`` the primary fits publish rather than a hand-rolled
    # variant that could differ in its interval convention.
    cov = _report.ppc_interval_coverage(trace, node="y_post")
    by_level = {
        int(round(float(r.level) * 100)): (int(r.n_inside), int(r.n_total))
        for r in cov.itertuples()
    }
    inside50, total50 = by_level.get(50, (0, 0))
    inside90, total90 = by_level.get(90, (0, 0))

    return {
        "outcome": symbol,
        "primary_model_id": DISPERSION_SENSITIVITY_MODEL_IDS[symbol],
        "n_trials": n_trials,
        "n_obs": int(built.prepared.n_obs),
        "kappa_prior_family": family,
        "kappa_prior_sigma": sigma,
        "is_registered_prior": family == "halfnormal_concentration" and sigma == 50.0,
        "ame_prob_median": float(np.median(ame)),
        "ame_prob_lo": float(np.quantile(ame, lo_q)),
        "ame_prob_hi": float(np.quantile(ame, hi_q)),
        "ame_items_median": float(np.median(ame) * n_trials),
        "prob_ame_pos": float(np.mean(ame > 0)),
        "kappa_median": float(np.median(kappa)),
        "kappa_lo": float(np.quantile(kappa, lo_q)),
        "kappa_hi": float(np.quantile(kappa, hi_q)),
        "variance_inflation_median": float(np.median(inflation)),
        "prob_within_10pct_of_binomial": float(np.mean(inflation <= 1.1)),
        "ppc_coverage_50": (inside50 / total50) if total50 else float("nan"),
        "ppc_coverage_90": (inside90 / total90) if total90 else float("nan"),
        "ppc_n_inside_90": inside90,
        "ppc_n_total_90": total90,
        "converged": bool(
            not signals.unassessable
            and signals.max_rhat <= 1.01
            and signals.min_ess >= 400
            and signals.n_divergences == 0
            and signals.min_bfmi is not None
            and signals.min_bfmi >= 0.3
        ),
        "max_rhat": float(signals.max_rhat),
        "min_ess": float(signals.min_ess),
        "n_divergences": int(signals.n_divergences or 0),
        "ci_prob": ci_prob,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dev", choices=["dev", "test", "rep-lite", "reporting"])
    parser.add_argument("--outcomes", nargs="*", default=list(DISPERSION_SENSITIVITY_MODEL_IDS))
    parser.add_argument("--ci-prob", type=float, default=0.89)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.output_dir:
        _paths.set_output_root(args.output_dir)

    unknown = [o for o in args.outcomes if o not in DISPERSION_SENSITIVITY_MODEL_IDS]
    if unknown:
        raise SystemExit(
            f"unknown outcome(s): {', '.join(unknown)}; "
            f"registered: {', '.join(DISPERSION_SENSITIVITY_MODEL_IDS)}"
        )

    sampling = _sampling.get_sampling_configuration(
        args.config, random_seed=args.seed
    )
    out_dir = Path(_paths.stat_models_dir()).parent / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    _console.print(f"Output root: {_paths.describe_output_root()}")
    _console.print(f"Writing to: {out_dir}")

    rows: list[dict] = []
    for symbol in args.outcomes:
        spec = _registered_spec(symbol)
        plan = resolve_itt_run_plan(spec)
        prepared, _adjust = prepare_itt_data(plan, loader=load_and_prepare)
        for index, (family, sigma) in enumerate(DISPERSION_SENSITIVITY_CELLS):
            label = f"{symbol}: {family} sigma={sigma}"
            _console.print(f"  fitting {label} ...")
            built, trace = _fit_cell(
                prepared, symbol, family, sigma, sampling, args.seed + index
            )
            row = _row(symbol, family, sigma, built, trace, args.ci_prob)
            rows.append(row)
            _console.print(
                f"    AME {row['ame_items_median']:+.2f} items "
                f"[{row['ame_prob_lo'] * row['n_trials']:+.2f}, "
                f"{row['ame_prob_hi'] * row['n_trials']:+.2f}], "
                f"kappa {row['kappa_median']:.1f}, "
                f"inflation {row['variance_inflation_median']:.2f}x, "
                f"PPC90 {row['ppc_coverage_90']:.2f}, "
                f"converged={row['converged']}"
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / FILENAME, index=False)
    (out_dir / "run_settings.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "seed": args.seed,
                "ci_prob": args.ci_prob,
                "cells": [list(c) for c in DISPERSION_SENSITIVITY_CELLS],
                "model_ids": DISPERSION_SENSITIVITY_MODEL_IDS,
                "sampling": {
                    "draws": sampling.draws,
                    "tune": sampling.tune,
                    "chains": sampling.chains,
                    "target_accept": sampling.target_accept,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _console.print(f"\nWrote {len(frame)} cells to {out_dir / FILENAME}")


if __name__ == "__main__":
    main()
