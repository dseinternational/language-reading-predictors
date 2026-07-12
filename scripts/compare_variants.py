# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Paired-fold comparison across fitted model variants.

Reads ``output/models/{model_id}/cv_scores.csv`` for each model listed
on the command line. All listed models must share the same
cross-validation splits (same ``GroupKFold`` n_splits, same seed, same
data rows) so per-fold scores can be paired by fold number. This is
the standard case across variants of a single task — e.g. ``LRP-RLI-GBL-012``
and all its variants use ``GroupKFold(n_splits=10)`` at seed 47.

For each metric the script reports:

- mean score per model,
- mean paired difference (model B − model A) and its SD,
- the **Nadeau–Bengio corrected** resampled paired *t*-test *p*-value,
- win count (folds where B < A for error metrics, B > A for R²).

CV fold scores are **not** independent — every pair of training sets shares most
of the ~54 children — so a *naive* paired *t*-test / Wilcoxon over the folds is
badly anti-conservative (Dietterich 1998; Nadeau & Bengio 2003, DOI
10.1023/A:1024068626366) and would overstate the evidence for a variant. The
reported *p* therefore inflates the variance of the mean difference by
``(1/K + n_test/n_train)`` (for K-fold, ``n_test/n_train = 1/(K−1)``); read it as a
rough guide alongside the mean difference and win-count, not a licence for a
"``p < 0.05``" claim. Output is a markdown table for notes / PR descriptions.

Usage
-----

::

    python scripts/compare_variants.py lrp-rli-gbl-012 lrp-rli-gbl-012_select02 lrp-rli-gbl-012_log lrp-rli-gbl-012_select02_log
    python scripts/compare_variants.py lrp-rli-gbl-012_log lrp-rli-gbl-012_select02_log --metrics mae r2
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from rich import print
from scipy import stats

from language_reading_predictors import paths as _paths

# Metrics where lower is better (error metrics) vs higher is better (R²).
_METRICS_LOWER_IS_BETTER = {"mae", "rmse", "medae"}
_METRICS_HIGHER_IS_BETTER = {"r2"}
_ALL_METRICS = sorted(_METRICS_LOWER_IS_BETTER | _METRICS_HIGHER_IS_BETTER)


def _nadeau_bengio_p(diff: np.ndarray) -> float:
    """Corrected resampled paired *t*-test p-value (Nadeau & Bengio 2003).

    K-fold CV scores are not independent — every pair of training sets overlaps
    heavily — so the naive paired *t*-test underestimates the variance of the mean
    difference and is anti-conservative (Dietterich 1998). The corrected test
    inflates that variance by ``(1/K + n_test/n_train)``; for K-fold the
    test/train size ratio is ``1/(K−1)``. Two-sided, ``df = K − 1``. Returns NaN
    when there are fewer than two folds or the differences are degenerate.
    """
    k = int(diff.size)
    if k < 2:
        return float("nan")
    var = float(np.var(diff, ddof=1))
    if not np.isfinite(var) or var == 0.0:
        return float("nan")
    correction = 1.0 / k + 1.0 / (k - 1)
    t_stat = float(np.mean(diff)) / np.sqrt(correction * var)
    return float(2.0 * stats.t.sf(abs(t_stat), df=k - 1))


def _load_cv(model_id: str) -> pd.DataFrame:
    path = _paths.gb_models_dir() / model_id / "cv_scores.csv"
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

        # Filter to fold pairs where both metrics are finite. ``ttest_rel``
        # with ``nan_policy="omit"`` drops NaNs per-vector and breaks
        # pairing when only one side is missing.
        valid = np.isfinite(sa) & np.isfinite(sb)
        sa = sa[valid]
        sb = sb[valid]

        diff = sb - sa
        mean_diff = float(np.mean(diff)) if diff.size else float("nan")
        std_diff = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")

        # Nadeau–Bengio corrected resampled t-test: the naive paired t / Wilcoxon
        # treat overlapping-fold scores as independent and overstate the evidence.
        nb_p = _nadeau_bengio_p(diff)

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
                "nb_p": nb_p,
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
        "corrected *t* p | B wins |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = [
        f"| `{r.a}` | `{r.b}` | {r.mean_a:.3f} | {r.mean_b:.3f} | "
        f"{r.mean_diff:+.3f} | {r.std_diff:.3f} | "
        f"{_format_p(r.nb_p)} | "
        f"{r.b_wins}/{r.n_folds} |"
        for r in df.itertuples()
    ]
    footer = (
        "\n_Corrected t p_: Nadeau–Bengio resampled paired t-test "
        "(fold scores are dependent; read as a guide, not a `p<0.05` claim).\n"
    )
    return header + "\n".join(rows) + footer


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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root to read from (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR). Default: repo-local output/."
        ),
    )
    args = parser.parse_args()
    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")

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

    # Validate that requested metrics exist in all frames.
    available_metrics = set.intersection(
        *(set(df.columns) for df in frames.values())
    )
    for metric in args.metrics:
        if metric not in available_metrics:
            missing_in = [mid for mid, df in frames.items() if metric not in df.columns]
            print(
                f"[bold red]Metric {metric!r} not found in: "
                f"{', '.join(missing_in)}[/bold red]"
            )
            raise SystemExit(1)

    for metric in args.metrics:
        table = _pairwise_table(frames, metric)
        print(_format_markdown(table, metric))


if __name__ == "__main__":
    main()
