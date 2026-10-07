# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Locate the steepest interval of a saved mechanism association curve.

The legacy output name is ``readiness_threshold.csv``. The location describes
where the fitted curve rises fastest. It does not identify the onset of reading,
a causal threshold or a minimum prerequisite. Interpret its boundary and
stability checks with the fitted sample and the model's assumptions.

This script post-processes the saved logit-scale ``f_mech`` curve and does not
refit the model. The fit pipeline also writes this summary and a separate
expected-items summary.

Run::

    python scripts/readiness_threshold.py --model lrp-rli-mech-058 --config reporting"""

from __future__ import annotations

from language_reading_predictors.statistical_models.summaries import readiness as _readiness_summary


import argparse
import json

import arviz as az
import pandas as pd
from rich import print as rprint

from language_reading_predictors import paths as _paths

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
    summary = _readiness_summary.readiness_threshold(trace, n_trials=n_trials, n_bins=args.n_bins)

    out_path = model_dir / "readiness_threshold.csv"
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    rprint(f"[green]Readiness threshold for {args.model} ({mech_symbol}, n={n_trials}):[/green]")
    for k, v in summary.items():
        rprint(f"  {k}: {v}")
    rprint(f"wrote {out_path}")


if __name__ == "__main__":
    main()
