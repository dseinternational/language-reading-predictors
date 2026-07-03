# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Treatment-effect (tau) prior-sensitivity sweep for the ITT suite (issue #141).

Refits representative single-outcome ITT models across a grid of tau prior SDs and
reports whether the headline conclusion is stable — posterior direction
(``pd = P(tau > 0)``), the logit and items-scale effect, and the interval width —
rather than a binary significant/not read.

The two-tier proposal keeps the wider ``Normal(0, 0.5)`` for proximal outcomes and
tightens the broad standardised-transfer (distal) outcomes to ``Normal(0, 0.3)``.
This sweep is the evidence for that choice: distal R/E should be stable (already
near-null) and proximal L/W should keep their direction across defensible SDs.

Usage::

    python scripts/tau_prior_sensitivity.py                 # dev config (fast)
    python scripts/tau_prior_sensitivity.py --config test   # more draws
    python scripts/tau_prior_sensitivity.py --outcomes R E L W

Writes ``output/statistical_models/tau_prior_sensitivity/tau_prior_sensitivity.csv``
and prints the table.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pymc as pm

import dse_research_utils.statistics.models.sampling as _sampling
from language_reading_predictors.statistical_models.factories import build_itt_model
from language_reading_predictors.statistical_models.measures import MEASURES

# The default sweep grid: distal outcomes get the sub-0.5 SDs the tier proposes,
# proximal outcomes bracket the 0.5 default. Both include the *other* tier's
# anchor so the comparison is symmetric.
DISTAL_SIGMAS = (0.2, 0.25, 0.3, 0.5)
PROXIMAL_SIGMAS = (0.25, 0.5, 0.75)
DEFAULT_OUTCOMES = ("R", "E", "L", "W")


def _sigmas_for(symbol: str) -> tuple[float, ...]:
    from language_reading_predictors.statistical_models.measures import is_distal

    return DISTAL_SIGMAS if is_distal(symbol) else PROXIMAL_SIGMAS


def _fit_one(prepared, symbol: str, sigma: float, sampling) -> dict:
    from language_reading_predictors.statistical_models.reporting import tau_summary_itt

    built = build_itt_model(
        prepared,
        outcome_symbol=symbol,
        cross_symbols=(),
        use_age_linear=True,
        tau_sigma=sigma,
    )
    with built.model:
        trace = pm.sample(
            draws=sampling.draws,
            tune=sampling.tune,
            chains=sampling.chains,
            cores=sampling.cores,
            target_accept=sampling.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=sampling.random_seed,
            progressbar=False,
        )
    s = tau_summary_itt(trace, ci_prob=0.95, G=built.prepared.G)
    n_trials = MEASURES[symbol].n_trials
    tau_draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values
    r_hat = float(pm.stats.rhat(trace, var_names=["tau"])["tau"].values)
    return {
        "outcome": symbol,
        "n_trials": n_trials,
        "tau_sigma": sigma,
        "n": int(built.prepared.n_obs),
        "pd": s["prob_tau_pos"],
        "tau_logit_mean": s["tau_logit_mean"],
        "tau_logit_lo": s["tau_logit_lo"],
        "tau_logit_hi": s["tau_logit_hi"],
        "ci_width_logit": s["tau_logit_hi"] - s["tau_logit_lo"],
        "tau_sd_logit": float(np.std(tau_draws)),
        "items_mean": s["tau_prob_mean"] * n_trials,
        "items_lo": s["tau_prob_lo"] * n_trials,
        "items_hi": s["tau_prob_hi"] * n_trials,
        "r_hat": r_hat,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="dev", help="sampling preset (dev/test/reporting)")
    ap.add_argument("--outcomes", nargs="+", default=list(DEFAULT_OUTCOMES))
    ap.add_argument(
        "--out-dir",
        default=os.path.join(
            "output", "statistical_models", "tau_prior_sensitivity"
        ),
    )
    args = ap.parse_args()

    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    sampling = _sampling.get_sampling_configuration(args.config, random_seed=20260701)
    prepared = load_and_prepare(phase_mode="itt")
    print(
        f"Loaded {prepared.n_obs} rows / {prepared.n_children} children; "
        f"config={args.config} (draws={sampling.draws}, tune={sampling.tune}, "
        f"chains={sampling.chains})"
    )

    rows = []
    for symbol in args.outcomes:
        for sigma in _sigmas_for(symbol):
            print(f"  fitting {symbol}  tau ~ Normal(0, {sigma}) ...", flush=True)
            rows.append(_fit_one(prepared, symbol, sigma, sampling))

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "tau_prior_sensitivity.csv")
    df.to_csv(out_csv, index=False)

    show = df.copy()
    for c in ("pd", "tau_logit_mean", "tau_logit_lo", "tau_logit_hi",
              "ci_width_logit", "tau_sd_logit", "items_mean", "items_lo",
              "items_hi", "r_hat"):
        show[c] = show[c].round(3)
    print("\n=== tau prior sensitivity ===")
    print(show.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
