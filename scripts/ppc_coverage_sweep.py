# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pool posterior-predictive coverage (``ppc_summary.csv``) across fitted models.

Motivation (notes/202607261405-binomial-exchangeability-item-difficulty-review.md):
heterogeneous item difficulty makes a bounded count *conditionally* underdispersed
relative to the Binomial (a Poisson-binomial has at-or-below-binomial variance given
ability), which the Beta-Binomial cannot express — its variance floor is the Binomial.
The observable symptom is predictive OVERcoverage: 50 % / 90 % prediction bands
covering more than 50 % / 90 % of observations. This script pools the per-fit
``ppc_summary.csv`` coverage rows by outcome symbol and family so the suite-level
pattern is visible at a glance, without any refitting.

Small-denominator caveat: central intervals of a discrete count distribution
overcover mechanically (a "50 %" interval on a 10-item score covers at least 50 %),
so compare measures of similar length with each other rather than reading any single
row against an absolute nominal level.

Usage:
    python scripts/ppc_coverage_sweep.py
    python scripts/ppc_coverage_sweep.py --models-dir /path/to/output/statistical_models/models
    python scripts/ppc_coverage_sweep.py --config reporting --out output/statistical_models/comparison/ppc_coverage.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(models_dir: Path, config: str) -> pd.DataFrame:
    rows = []
    skipped = []
    for model_dir in sorted(models_dir.glob(f"*-{config}")):
        ppc_path = model_dir / "ppc_summary.csv"
        cfg_path = model_dir / "config.json"
        if not ppc_path.exists() or not cfg_path.exists():
            skipped.append(model_dir.name)
            continue
        cfg = json.loads(cfg_path.read_text())
        ppc = pd.read_csv(ppc_path)
        ppc["model_id"] = cfg.get("model_id", model_dir.name)
        ppc["kind"] = cfg.get("kind", "?")
        ppc["outcome_symbol"] = cfg.get("outcome_symbol") or "-"
        rows.append(ppc)
    if skipped:
        print(f"[skipped {len(skipped)} dirs without ppc_summary.csv/config.json]")
    if not rows:
        raise SystemExit(f"No ppc_summary.csv found under {models_dir} for config {config!r}")
    return pd.concat(rows, ignore_index=True)


def pooled(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grp = df.groupby(keys + ["level_pct"], dropna=False)
    out = grp.agg(
        n_models=("model_id", "nunique"),
        n_total=("n_total", "sum"),
        n_inside=("n_inside", "sum"),
        mean_coverage=("coverage", "mean"),
    ).reset_index()
    out["pooled_coverage"] = out["n_inside"] / out["n_total"]
    out["excess"] = out["pooled_coverage"] - out["level_pct"] / 100.0
    return out.sort_values(["level_pct", "excess"], ascending=[True, False])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=Path("output/statistical_models/models"),
        help="Directory holding {model_id}-{config} output folders",
    )
    ap.add_argument("--config", default="reporting", help="Config suffix to sweep (default: reporting)")
    ap.add_argument("--out", type=Path, default=None, help="Optional CSV path for the long per-model table")
    args = ap.parse_args()

    df = collect(args.models_dir, args.config)
    print(f"Collected {len(df)} coverage rows from {df['model_id'].nunique()} models ({args.config}).")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote long table to {args.out}")

    for mode, sub in df.groupby("mode"):
        print(f"\n=== mode: {mode} — pooled coverage by outcome symbol ===")
        tab = pooled(sub, ["outcome_symbol"])
        print(tab.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))
        print(f"\n=== mode: {mode} — pooled coverage by family (kind) ===")
        print(pooled(sub, ["kind"]).to_string(index=False, float_format=lambda v: f"{v:0.3f}"))


if __name__ == "__main__":
    main()
