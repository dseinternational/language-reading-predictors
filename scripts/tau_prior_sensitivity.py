# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prior-sensitivity and unadjusted benchmarks for the ITT suite (#141, #341).

Refits representative single-outcome ITT models across a grid of tau prior SDs and
reports whether the headline conclusion is stable — posterior direction
(``pd = P(AME > 0)``), the logit and items-scale effect, and the interval width —
rather than a binary significant/not read.

The two-tier proposal keeps the wider ``Normal(0, 0.5)`` for proximal outcomes and
tightens the broad standardised-transfer (distal) outcomes to ``Normal(0, 0.3)``.
This sweep audits that post-data regularisation choice: the distal outcomes should be stable
across defensible SDs and proximal L/W should keep their direction. The default
now covers **every distal member** — the clearly-null R/E and the *borderline*
UR/UE/T/F — so the certifying sweep is not limited to the null outcomes (issue
#267): the printed table lets a reviewer check whether any evidence-ladder
boundary moves between SD 0.3 and 0.5 for the borderline members. It separately
compares the own-baseline precision prior ``Normal(1, 0.25)`` with
``Normal(1, 0.5)`` for the headline L/W outcomes and fits an unadjusted
randomised-arm benchmark (no baseline or age precision terms). It also varies
the Beta-Binomial concentration prior across ``HalfNormal(25)``, ``HalfNormal(50)``,
``HalfNormal(100)`` and ``HalfNormal(200)`` for L/W, spanning stronger
overdispersion through a much more permissive near-Binomial region.

Usage::

    python scripts/tau_prior_sensitivity.py                 # dev config (fast)
    python scripts/tau_prior_sensitivity.py --config test   # more draws
    python scripts/tau_prior_sensitivity.py --outcomes R E UR UE T F L W

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
from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.factories import build_itt_model
from language_reading_predictors.statistical_models.measures import MEASURES

# The default sweep grid: distal outcomes get the sub-0.5 SDs the tier proposes,
# proximal outcomes bracket the 0.5 default. Both include the *other* tier's
# anchor so the comparison is symmetric.
DISTAL_SIGMAS = (0.2, 0.25, 0.3, 0.5)
PROXIMAL_SIGMAS = (0.25, 0.5, 0.75)
# Every distal member (clearly-null R/E and borderline UR/UE/T/F) plus the L/W
# proximal anchors — so the certifying sweep is not limited to the null outcomes
# (issue #267).
DEFAULT_OUTCOMES = ("R", "E", "UR", "UE", "T", "F", "L", "W")
BASELINE_SENSITIVITY_OUTCOMES = ("L", "W")
BASELINE_SIGMAS = (0.25, 0.5)
CONCENTRATION_SENSITIVITY_OUTCOMES = ("L", "W")
KAPPA_SIGMAS = (25.0, 50.0, 100.0, 200.0)


def _sigmas_for(symbol: str) -> tuple[float, ...]:
    from language_reading_predictors.statistical_models.measures import is_distal

    return DISTAL_SIGMAS if is_distal(symbol) else PROXIMAL_SIGMAS


def _adopted_tau_sigma(symbol: str) -> float:
    from language_reading_predictors.statistical_models.measures import is_distal

    return 0.3 if is_distal(symbol) else 0.5


def _fit_one(
    prepared,
    symbol: str,
    tau_sigma: float,
    sampling,
    config: str,
    *,
    gamma_own_sigma: float | None = 0.25,
    kappa_sigma: float | None = 50.0,
    use_precision_terms: bool = True,
    sensitivity_axis: str = "tau_sigma",
) -> dict:
    from language_reading_predictors.statistical_models import diagnostics as _diag
    from language_reading_predictors.statistical_models.reporting import tau_summary_itt

    built = build_itt_model(
        prepared,
        outcome_symbol=symbol,
        cross_symbols=(),
        use_age_linear=use_precision_terms,
        use_own_baseline=use_precision_terms,
        tau_sigma=tau_sigma,
        gamma_own_sigma=gamma_own_sigma,
        kappa_sigma=kappa_sigma,
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
    kappa_draws = (
        trace.posterior["kappa"].stack(sample=("chain", "draw")).values
        if "kappa" in trace.posterior
        else np.array([np.nan])
    )
    free_names = [rv.name for rv in built.model.free_RVs]
    convergence = _diag.subfit_convergence(
        trace,
        label=f"{symbol} {sensitivity_axis}",
        var_names=free_names,
    )
    return {
        "config": config,
        "outcome": symbol,
        "n_trials": n_trials,
        "sensitivity_axis": sensitivity_axis,
        "tau_sigma": tau_sigma,
        "gamma_own_sigma": gamma_own_sigma,
        "kappa_sigma": kappa_sigma,
        "use_precision_terms": use_precision_terms,
        "n": int(built.prepared.n_obs),
        "pd": s["prob_tau_pos"],
        "tau_logit_mean": s["tau_logit_mean"],
        "tau_logit_lo": s["tau_logit_lo"],
        "tau_logit_hi": s["tau_logit_hi"],
        "ci_width_logit": s["tau_logit_hi"] - s["tau_logit_lo"],
        "tau_sd_logit": float(np.std(tau_draws)),
        "kappa_median": float(np.nanmedian(kappa_draws)),
        "items_mean": s["tau_prob_mean"] * n_trials,
        "items_lo": s["tau_prob_lo"] * n_trials,
        "items_hi": s["tau_prob_hi"] * n_trials,
        "converged": convergence["converged"],
        "max_rhat": convergence["max_rhat"],
        "min_ess": convergence["min_ess"],
        "min_bfmi": convergence["min_bfmi"],
        "n_divergences": convergence["n_divergences"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="dev", help="sampling preset (dev/test/reporting)")
    ap.add_argument("--outcomes", nargs="+", default=list(DEFAULT_OUTCOMES))
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output dir (default: <output-root>/statistical_models/"
            "tau_prior_sensitivity)."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root for this run (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR); the relative layout is unchanged. Default: "
            "repo-local output/."
        ),
    )
    args = ap.parse_args()

    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")
    args.out_dir = args.out_dir or os.path.join(
        str(_paths.stat_dir()), "tau_prior_sensitivity"
    )

    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    sampling = _sampling.get_sampling_configuration(args.config, random_seed=20260701)
    # Prepare each outcome separately, matching its registered single-outcome
    # model. A shared eight-outcome frame would both omit taught outcomes (UR/UE)
    # and impose an unintended cross-outcome complete-case restriction.
    prepared_by_symbol = {
        symbol: load_and_prepare(phase_mode="itt", outcomes=(symbol,))
        for symbol in args.outcomes
    }
    loaded_counts = ", ".join(
        f"{symbol}={prepared.n_obs}"
        for symbol, prepared in prepared_by_symbol.items()
    )
    print(
        f"Loaded separate outcome frames ({loaded_counts}); "
        f"config={args.config} (draws={sampling.draws}, tune={sampling.tune}, "
        f"chains={sampling.chains})"
    )

    rows = []
    for symbol in args.outcomes:
        for sigma in _sigmas_for(symbol):
            print(f"  fitting {symbol}  tau ~ Normal(0, {sigma}) ...", flush=True)
            rows.append(
                _fit_one(
                    prepared_by_symbol[symbol],
                    symbol,
                    sigma,
                    sampling,
                    args.config,
                    sensitivity_axis="tau_sigma",
                )
            )

    # Own-baseline sensitivity is a separate axis rather than a Cartesian product
    # with every tau value: it answers the prior-audit question without multiplying
    # the already large sweep. Restrict it to the two headline reading anchors.
    for symbol in BASELINE_SENSITIVITY_OUTCOMES:
        if symbol not in args.outcomes:
            continue
        tau_sigma = _adopted_tau_sigma(symbol)
        for baseline_sigma in BASELINE_SIGMAS:
            print(
                f"  fitting {symbol}  gamma_own ~ Normal(1, {baseline_sigma}) ...",
                flush=True,
            )
            rows.append(
                _fit_one(
                    prepared_by_symbol[symbol],
                    symbol,
                    tau_sigma,
                    sampling,
                    args.config,
                    gamma_own_sigma=baseline_sigma,
                    sensitivity_axis="gamma_own_sigma",
                )
            )

        print(f"  fitting {symbol}  unadjusted randomised-arm benchmark ...", flush=True)
        rows.append(
            _fit_one(
                prepared_by_symbol[symbol],
                symbol,
                tau_sigma,
                sampling,
                args.config,
                gamma_own_sigma=None,
                use_precision_terms=False,
                sensitivity_axis="unadjusted_benchmark",
            )
        )

    # Dispersion-prior sensitivity is kept as its own axis. The larger scales are
    # important for the 79- and 32-item headline tests because the adopted
    # HalfNormal(50) puts little mass in the near-Binomial region for high
    # denominators; a stable treatment contrast should not depend materially on
    # that restriction.
    for symbol in CONCENTRATION_SENSITIVITY_OUTCOMES:
        if symbol not in args.outcomes:
            continue
        tau_sigma = _adopted_tau_sigma(symbol)
        for kappa_sigma in KAPPA_SIGMAS:
            print(
                f"  fitting {symbol}  kappa ~ HalfNormal({kappa_sigma:g}) ...",
                flush=True,
            )
            rows.append(
                _fit_one(
                    prepared_by_symbol[symbol],
                    symbol,
                    tau_sigma,
                    sampling,
                    args.config,
                    kappa_sigma=kappa_sigma,
                    sensitivity_axis="kappa_sigma",
                )
            )

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "tau_prior_sensitivity.csv")
    df.to_csv(out_csv, index=False)

    show = df.copy()
    for c in (
        "pd",
        "tau_logit_mean",
        "tau_logit_lo",
        "tau_logit_hi",
        "ci_width_logit",
        "tau_sd_logit",
        "kappa_median",
        "items_mean",
        "items_lo",
        "items_hi",
        "max_rhat",
        "min_ess",
        "min_bfmi",
    ):
        show[c] = show[c].round(3)
    print("\n=== tau prior sensitivity ===")
    print(show.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
