# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Readiness-threshold post-processing for a mechanism fit (#230 §5).

Locates the "knee" of a fitted letter-sound -> word-reading HSGP curve — the
letter-sound count around which reading takes off — from a mechanism model's saved
trace, answering "does reading move only above ~k letter sounds?". Pure
post-processing: it re-fits nothing, reading the ``f_mech`` posterior and the
``mech_post_logit`` constant-data node written by the mechanism fit (the default
target is ``lrp-rli-mech-058``, letter sounds -> word reading).

Writes ``readiness_threshold.csv`` into the model's output directory and prints the
summary. Run after the mechanism model has been fitted::

    python scripts/fit_statistical_model.py lrp-rli-mech-058 --config reporting
    python scripts/readiness_threshold.py --model lrp-rli-mech-058 --config reporting
"""

from __future__ import annotations

import argparse
import json

import arviz as az
import pandas as pd
from rich import print as rprint

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.measures import MEASURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="lrp-rli-mech-058", help="Mechanism model id.")
    parser.add_argument("--config", default="dev", help="Sampling config the trace was fit under.")
    parser.add_argument("--n-bins", type=int, default=6, help="Quantile bins over the letter-sound range.")
    parser.add_argument("--output-dir", default=None, help="Override the output root.")
    args = parser.parse_args()
    _paths.set_output_root(args.output_dir)

    model_dir = _paths.stat_models_dir() / f"{args.model}-{args.config}"
    trace_path = model_dir / "trace.nc"
    if not trace_path.exists():
        raise SystemExit(
            f"No trace at {trace_path}. Fit the model first: "
            f"python scripts/fit_statistical_model.py {args.model} --config {args.config}"
        )

    # The predictor whose count scale the knee is reported in (letter sounds, L = 32).
    mech_symbol = "L"
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as fh:
            mech_symbol = json.load(fh).get("mechanism_symbol") or mech_symbol
    n_trials = MEASURES[mech_symbol].n_trials

    trace = az.from_netcdf(trace_path)
    summary = _report.readiness_threshold(trace, n_trials=n_trials, n_bins=args.n_bins)

    out_path = model_dir / "readiness_threshold.csv"
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    rprint(f"[green]Readiness threshold for {args.model} ({mech_symbol}, n={n_trials}):[/green]")
    for k, v in summary.items():
        rprint(f"  {k}: {v}")
    rprint(f"wrote {out_path}")


if __name__ == "__main__":
    main()
