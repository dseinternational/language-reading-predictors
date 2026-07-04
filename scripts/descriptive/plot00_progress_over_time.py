# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Descriptive trajectory grid: each measure's level over the four waves.

A 2x4 small-multiples grid, one panel per measure. Within each panel:
faint per-child lines (level against timepoint) plus a bold mean-per-wave line.
Shows the developmental profile across the reading/phonics, language and
vocabulary battery for the responder/non-responder picture. Non-verbal MA
(block design) is excluded here because it is measured at baseline only and so
has no trajectory over the four waves.

Standalone by design: this script loads ``data/rli_data_long.csv`` directly and
hardcodes its own columns and labels and resolves its output path through
``language_reading_predictors.paths`` so scratch-output runs stay
consistent. Duplication across the descriptive plot scripts is intentional --
do not refactor shared logic into a helper.

Run::

    python scripts/descriptive/plot00_progress_over_time.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from language_reading_predictors import paths as _paths

# --- Measures shown, in grid order (column, plain-English panel title) ---
MEASURES = [
    ("ewrswr", "Word reading"),
    ("yarclet", "Letter sounds (YARC-LSK)"),
    ("spphon", "Phonetic spelling"),
    ("nonword", "Nonword reading"),
    ("blending", "Blending"),
    ("eowpvt", "Expressive vocabulary"),
    ("rowpvt", "Receptive vocabulary"),
    ("trog", "Grammar (TROG)"),
]
TIMEPOINTS = [1, 2, 3, 4]
DPI = 140
OUT_NAME = "plot00_progress_over_time.png"

_DATA_PATH = _paths.DATA_DIR / "rli_data_long.csv"


def _out_dir():
    return _paths.output_root() / "descriptive"


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

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True)
    for ax, (col, title) in zip(axes.flat, MEASURES, strict=True):
        # Faint individual trajectories.
        for _, child in df.groupby("subject_id"):
            series = child.dropna(subset=[col]).sort_values("time")
            if len(series) >= 2:
                ax.plot(
                    series["time"], series[col], color="#4C72B0",
                    alpha=0.15, linewidth=0.8, zorder=1,
                )
        # Bold mean-per-wave line.
        wave_mean = df.groupby("time")[col].mean().reindex(TIMEPOINTS)
        ax.plot(
            wave_mean.index, wave_mean.to_numpy(), color="#C44E52",
            linewidth=2.5, marker="o", zorder=3, label="mean",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks(TIMEPOINTS)
        ax.margins(x=0.05)

    for ax in axes[-1, :]:
        ax.set_xlabel("timepoint")
    for ax in axes[:, 0]:
        ax.set_ylabel("score")

    fig.suptitle("Measure trajectories over the four waves", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_dir = _out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
