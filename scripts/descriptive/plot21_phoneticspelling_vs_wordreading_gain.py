# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Descriptive scatter: baseline phonetic spelling vs word reading total gain.

x = phonetic spelling (``spphon``) level at time 1 (the fixed baseline).
y = word reading (``ewrswr``) total gain = level at the child's last available
    wave minus level at time 1. One point per child.

Standalone by design: this script loads ``data/rli_data_long.csv`` directly and
hardcodes its own columns and labels and resolves its output path through
``language_reading_predictors.paths`` so scratch-output runs stay
consistent. Duplication across the descriptive plot scripts is intentional --
do not refactor shared logic into a helper.

Run::

    python scripts/descriptive/plot21_phoneticspelling_vs_wordreading_gain.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from language_reading_predictors import paths as _paths

# --- Per-figure configuration (the only lines that differ between scripts) ---
X_COL = "spphon"  # predictor measure (level column)
Y_COL = "ewrswr"  # outcome measure (level column)
X_LABEL = "baseline phonetic spelling"
Y_LABEL = "word reading total gain (baseline to last wave)"
TITLE = "Baseline phonetic spelling vs word reading gain"
OUT_NAME = "plot21_phoneticspelling_vs_wordreading_gain.png"

# --- Recipe constants (shared by every predictor-vs-gain script) ---
N_BINS = 5  # quantile bins across x
MIN_BIN_N = 4  # skip bins with fewer points than this
DPI = 140

_DATA_PATH = _paths.DATA_DIR / "rli_data_long.csv"


def _out_dir():
    return _paths.output_root() / "descriptive"


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """One row per child: baseline predictor level vs total outcome gain.

    x = ``X_COL`` at time 1. y = ``Y_COL`` at the last available wave (max time
    with a non-null value, requiring time > 1) minus ``Y_COL`` at time 1.
    """
    baseline = df[df["time"] == 1].set_index("subject_id")
    x = baseline[X_COL]
    y_baseline = baseline[Y_COL]

    last = (
        df.dropna(subset=[Y_COL])
        .sort_values("time")
        .groupby("subject_id")
        .tail(1)
        .set_index("subject_id")
    )
    last = last[last["time"] > 1]  # need a later wave to define a gain
    y_last = last[Y_COL]

    panel = pd.DataFrame({"x": x, "y": y_last - y_baseline})
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
    summary = summary[summary["n"] >= MIN_BIN_N].sort_values("x_center")
    return summary


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
    panel = build_panel(df)
    summary = binned_summary(panel)

    print(f"{OUT_NAME}: n = {len(panel)} children ({len(summary)} usable bins)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0, zorder=1)
    ax.scatter(
        panel["x"], panel["y"], s=28, alpha=0.3, color="#4C72B0", zorder=2,
        label=f"children (n = {len(panel)})",
    )
    if not summary.empty:
        ax.plot(
            summary["x_center"], summary["median_y"], color="#C44E52",
            linewidth=2.0, marker="o", zorder=3, label="binned median",
        )
        ax.plot(
            summary["x_center"], summary["mean_y"], color="#C44E52",
            linewidth=2.0, linestyle="--", marker="s", zorder=3,
            label="binned mean",
        )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(TITLE)
    ax.legend(frameon=False, fontsize=9)
    ax.margins(x=0.03)
    fig.tight_layout()

    out_dir = _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
