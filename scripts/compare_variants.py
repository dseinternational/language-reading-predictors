# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Paired-fold comparison across fitted model variants.

Reads ``output/models/{model_id}/cv_scores.csv`` for each model listed
on the command line. All listed models must share the same
cross-validation splits (same ``GroupKFold`` n_splits, same seed, same
data rows) so per-fold scores can be paired by fold number. This is
the standard case across variants of a single task — e.g. ``lrp02``
and all its variants use ``GroupKFold(n_splits=10)`` at seed 47.

For each metric the script reports:

- mean score per model,
- mean paired difference (model B − model A),
- paired-sample *t*-test *p*-value,
- Wilcoxon signed-rank *p*-value (more robust to non-normal residuals),
- win count (folds where B < A for error metrics, B > A for R²).

Output is a markdown table so it can be pasted straight into notes or
PR descriptions.

Usage
-----

::

    python scripts/compare_variants.py lrp02 lrp02_select02 lrp02_log lrp02_select02_log
    python scripts/compare_variants.py lrp02_log lrp02_select02_log --metrics mae r2
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from scipy import stats

_ROOT_DIR = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT_DIR / "output" / "models"

# Metrics where lower is better (error metrics) vs higher is better (R²).
_METRICS_LOWER_IS_BETTER = {"mae", "rmse", "medae"}
_METRICS_HIGHER_IS_BETTER = {"r2"}
_ALL_METRICS = sorted(_METRICS_LOWER_IS_BETTER | _METRICS_HIGHER_IS_BETTER)


def _load_cv(model_id: str) -> pd.DataFrame:
    path = _MODELS_DIR / model_id / "cv_scores.csv"
    if not path.exists():
        msg = f"no cv_scores.csv for {model_id!r} at {path}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(path)
    if "fold" not in df.columns:
        msg = f"{path} missing 'fold' column"
        raise ValueError(msg)
    return df.set_index("fold")


def _pairwise_table(
    frames: dict[str, pd.DataFrame], metric: str
) -> pd.DataFrame:
    """Per-pair comparison rows for a single metric."""
    lower_is_better = metric in _METRICS_LOWER_IS_BETTER

    rows = []
    for a, b in combinations(frames.keys(), 2):
        scores_a = frames[a][metric]
        scores_b = frames[b][metric]

        common = scores_a.index.intersection(scores_b.index)
        if len(common) == 0:
            continue
        sa = scores_a.loc[common].to_numpy(dtype=float)
        sb = scores_b.loc[common].to_numpy(dtype=float)

        diff = sb - sa
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")

        t_res = stats.ttest_rel(sb, sa, nan_policy="omit")
        # Wilcoxon rejects identical vectors; guard for it.
        if np.all(diff == 0):
            w_p = float("nan")
        else:
            try:
                w_res = stats.wilcoxon(diff, zero_method="wilcox", nan_policy="omit")
                w_p = float(w_res.pvalue)
            except ValueError:
                w_p = float("nan")

        wins_b = (
            int(np.sum(sb < sa)) if lower_is_better else int(np.sum(sb > sa))
        )

        rows.append(
            {
                "a": a,
                "b": b,
                "mean_a": float(np.mean(sa)),
                "mean_b": float(np.mean(sb)),
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "t_p": float(t_res.pvalue),
                "wilcoxon_p": w_p,
                "b_wins": wins_b,
                "n_folds": int(len(common)),
            }
        )

    return pd.DataFrame(rows)


def _format_p(p: float) -> str:
    if np.isnan(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _format_markdown(df: pd.DataFrame, metric: str) -> str:
    direction = (
        "lower is better"
        if metric in _METRICS_LOWER_IS_BETTER
        else "higher is better"
    )
    header = (
        f"### {metric.upper()} ({direction})\n\n"
        "| A | B | mean(A) | mean(B) | mean(B−A) | std(B−A) | "
        "paired *t* p | Wilcoxon p | B wins |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = [
        f"| `{r.a}` | `{r.b}` | {r.mean_a:.3f} | {r.mean_b:.3f} | "
        f"{r.mean_diff:+.3f} | {r.std_diff:.3f} | "
        f"{_format_p(r.t_p)} | {_format_p(r.wilcoxon_p)} | "
        f"{r.b_wins}/{r.n_folds} |"
        for r in df.itertuples()
    ]
    return header + "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired-fold comparison of fitted model variants."
    )
    parser.add_argument(
        "models",
        nargs="+",
        help="Model ids to compare (at least 2). Each must have "
        "output/models/{id}/cv_scores.csv.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=_ALL_METRICS,
        choices=_ALL_METRICS,
        help=f"Metrics to compare. Default: all ({_ALL_METRICS}).",
    )
    args = parser.parse_args()

    if len(args.models) < 2:
        print("[bold red]Need at least 2 models to compare.[/bold red]")
        raise SystemExit(1)

    frames = {mid: _load_cv(mid) for mid in args.models}

    # Sanity check: every pair must share fold indices.
    ref_folds = next(iter(frames.values())).index
    for mid, df in frames.items():
        if not df.index.equals(ref_folds):
            print(
                f"[yellow]Warning: {mid!r} has fold indices "
                f"{list(df.index)} but reference is {list(ref_folds)}. "
                "Only intersecting folds will be compared."
            )

    print(
        f"[green]Comparing {len(args.models)} models on "
        f"{len(ref_folds)} folds: {args.models}[/green]\n"
    )

    for metric in args.metrics:
        table = _pairwise_table(frames, metric)
        print(_format_markdown(table, metric))


if __name__ == "__main__":
    main()
