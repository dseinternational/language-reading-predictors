# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pooled-periods descriptive scatters: start-of-period level vs per-period gain.

Generates all 22 predictor-vs-gain panels in one pass (unlike the one-point-per-
child set in this folder, which keeps one archivable script per figure).

Recipe, per panel:

    * keep the three wave-to-wave transitions (waves 1-3; ``_gain`` is NaN at 4),
    * x = the predictor measure's level at the start of the period (plain column,
      e.g. ``yarclet``),
    * y = that period's gain in the outcome (the ``_gain`` column, e.g.
      ``ewrswr_gain``),
    * drop rows missing either, then plot a faint scatter with a quantile-binned
      median (solid) and mean (dashed) over ~5 bins (bins with <4 points skipped)
      and a dotted zero line.

Caveat: a child contributes up to three points (one per period), so these are
repeated measures from the same ~54 children. The point count (~150) is larger
than the number of children and the points are correlated, so the binned lines
look more precise than the independent-child evidence supports, and start-of-
period level mixes baseline and mid-study ages. Each panel's printout reports
both counts. For a one-point-per-child view, use the plotNN_*.py scripts here.

Measure -> column map (x uses the level column; y uses ``<level>_gain``):
    word reading ``ewrswr`` · letter sounds ``yarclet`` · phonetic spelling
    ``spphon`` · nonword reading ``nonword`` · blending ``blending`` ·
    expressive vocabulary ``eowpvt``. (Letter sounds is ``yarclet``, NOT the
    phonetic-spelling column ``spphon``.)

Run::

    python scripts/descriptive/pooled_periods.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from language_reading_predictors import paths as _paths

PERIODS = (1, 2, 3)  # wave-to-wave transitions; _gain is NaN at wave 4
N_BINS = 5  # quantile bins across x
MIN_BIN_N = 4  # skip bins with fewer points than this
DPI = 140

_DATA_PATH = _paths.DATA_DIR / "rli_data_long.csv"


def _out_dir():
    return _paths.output_root() / "descriptive" / "pooled"

# measure -> (plain name, short slug, x-axis annotation)
MEASURES = {
    "ewrswr": ("word reading", "wordreading", ""),
    "yarclet": ("letter sounds", "lettersounds", " (YARC-LSK)"),
    "spphon": ("phonetic spelling", "phoneticspelling", ""),
    "nonword": ("nonword reading", "nonword", ""),
    "blending": ("blending", "blending", " (phonological awareness)"),
    "eowpvt": ("expressive vocabulary", "expressivevocab", ""),
}

# (panel number, predictor x level column, outcome y level column)
PANELS = [
    (1, "ewrswr", "spphon"), (2, "ewrswr", "nonword"), (3, "ewrswr", "blending"),
    (4, "ewrswr", "yarclet"), (5, "ewrswr", "eowpvt"),
    (6, "yarclet", "ewrswr"), (7, "yarclet", "spphon"), (8, "yarclet", "nonword"),
    (9, "yarclet", "blending"), (10, "yarclet", "eowpvt"),
    (11, "eowpvt", "ewrswr"), (12, "eowpvt", "spphon"), (13, "eowpvt", "nonword"),
    (14, "eowpvt", "blending"), (15, "eowpvt", "yarclet"),
    (16, "blending", "ewrswr"), (17, "blending", "spphon"), (18, "blending", "nonword"),
    (19, "blending", "yarclet"), (20, "blending", "eowpvt"),
    (21, "spphon", "ewrswr"), (22, "nonword", "ewrswr"),
]


def build_panel(df: pd.DataFrame, x_col: str, y_gain_col: str) -> pd.DataFrame:
    """One row per child-period: start-of-period x level vs that period's gain."""
    periods = df[df["time"].isin(PERIODS)]
    panel = pd.DataFrame(
        {
            "x": periods[x_col],
            "y": periods[y_gain_col],
            "subject_id": periods["subject_id"],
        }
    )
    return panel.dropna(subset=["x", "y"])


def binned_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Quantile-bin x; return per-bin median-x, median-y, mean-y, count."""
    bins = pd.qcut(panel["x"], N_BINS, duplicates="drop")
    grouped = panel.groupby(bins, observed=True)
    summary = grouped.agg(
        x_center=("x", "median"),
        median_y=("y", "median"),
        mean_y=("y", "mean"),
        n=("y", "size"),
    )
    return summary[summary["n"] >= MIN_BIN_N].sort_values("x_center")


def make_figure(df: pd.DataFrame, number: int, x_col: str, y_col: str) -> None:
    """Build and save one pooled-periods panel."""
    x_name, x_slug, x_annot = MEASURES[x_col]
    y_name, y_slug, _ = MEASURES[y_col]
    y_gain_col = f"{y_col}_gain"

    panel = build_panel(df, x_col, y_gain_col)
    summary = binned_summary(panel)
    n_children = panel["subject_id"].nunique()
    out_name = f"plot{number:02d}_{x_slug}_vs_{y_slug}_gain.png"
    print(
        f"{out_name}: n = {len(panel)} period-observations from {n_children} "
        f"children ({len(summary)} usable bins)"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0, zorder=1)
    ax.scatter(
        panel["x"], panel["y"], s=28, alpha=0.3, color="#4C72B0", zorder=2,
        label=f"child-periods (n = {len(panel)})",
    )
    if not summary.empty:
        ax.plot(
            summary["x_center"], summary["median_y"], color="#C44E52",
            linewidth=2.0, marker="o", zorder=3, label="binned median",
        )
        ax.plot(
            summary["x_center"], summary["mean_y"], color="#C44E52",
            linewidth=2.0, linestyle="--", marker="s", zorder=3, label="binned mean",
        )

    ax.set_xlabel(f"{x_name} at period start{x_annot}")
    ax.set_ylabel(f"{y_name} gain (per period)")
    ax.set_title(f"{x_name.capitalize()} vs {y_name} gain (pooled periods)")
    ax.legend(frameon=False, fontsize=9)
    ax.margins(x=0.03)
    fig.tight_layout()

    fig.savefig(_out_dir() / out_name, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root for this run (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR). Default: repo-local output/."
        ),
    )
    args = parser.parse_args()
    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")

    df = pd.read_csv(_DATA_PATH)
    out_dir = _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for number, x_col, y_col in PANELS:
        make_figure(df, number, x_col, y_col)
    print(f"wrote {len(PANELS)} figures to {out_dir}")


if __name__ == "__main__":
    main()
