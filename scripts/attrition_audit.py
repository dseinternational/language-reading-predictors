# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Attrition / outcome-missingness audit for the RLI waves (#230 §3).

Issue #230 §3 asks whether the statistical suite needs an MNAR / informative-attrition
sensitivity analysis, on the grounds that the complete-case comparators only handle
*covariate* missingness, not *outcome* dropout. This audit is the evidence for that
decision: it counts, per outcome and per wave, how many children have a non-missing
score, and how many are lost across the windows the headline estimands actually use —

- the randomised **ITT window** t1 -> t2 (a child with a t1 baseline but no t2 post),
- the **DiD crossover window** t1 -> t2 -> t3 (a t2 score but no t3), and
- the final **maintenance wave** t4 (a t3 score but no t4),

so a reviewer can see where — if anywhere — outcome attrition could bias an estimand.
It fits no model and makes no causal claim; it only tabulates observed missingness.

Standalone by design: reads ``data/rli_data_long.csv`` directly and resolves its output
path through ``language_reading_predictors.paths``. Writes
``attrition_audit.csv`` to ``output/audit/`` (gitignored) and prints the table.

Run::

    python scripts/attrition_audit.py
"""

from __future__ import annotations

import argparse

import pandas as pd

from language_reading_predictors import paths as _paths
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.measures import MEASURES

# Outcomes audited: the eight standardised ITT outcomes plus nonword reading, in a
# reading order. Columns and labels come from MEASURES so this cannot drift.
OUTCOMES: tuple[str, ...] = ("W", "R", "E", "L", "B", "F", "T", "P", "N")
WAVES: tuple[int, ...] = (1, 2, 3, 4)

_DATA_PATH = _paths.DATA_DIR / "rli_data_long.csv"


def _wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.pivot_table(index=V.SUBJECT_ID, columns=V.TIME, values=column)


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Per-outcome wave completeness and window-attrition counts."""
    rows: list[dict] = []
    for sym in OUTCOMES:
        col = MEASURES[sym].column
        w = _wide(df, col)
        present = {t: w[t].notna() if t in w.columns else pd.Series(False, index=w.index) for t in WAVES}
        # Window attrition: had the earlier score, missing the next one.
        itt_attr = int((present[1] & ~present[2]).sum())  # t1 baseline, no t2 post
        did_attr = int((present[2] & ~present[3]).sum())  # t2, no t3 (crossover window)
        t4_attr = int((present[3] & ~present[4]).sum())  # t3, no t4 (maintenance wave)
        rows.append(
            {
                "outcome": sym,
                "measure": MEASURES[sym].label,
                "n_t1": int(present[1].sum()),
                "n_t2": int(present[2].sum()),
                "n_t3": int(present[3].sum()),
                "n_t4": int(present[4].sum()),
                "itt_window_attrition": itt_attr,
                "did_window_attrition": did_attr,
                "t4_attrition": t4_attr,
            }
        )
    return pd.DataFrame(rows)


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
    n_subjects = int(df[V.SUBJECT_ID].nunique())
    table = audit(df)

    out_dir = _paths.output_root() / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attrition_audit.csv"
    table.to_csv(out_path, index=False)

    print(f"\nTotal subjects: {n_subjects}")
    print("\n=== outcome missingness by wave + window attrition ===")
    print(table.to_string(index=False))
    itt = int(table["itt_window_attrition"].sum())
    did = int(table["did_window_attrition"].sum())
    print(
        f"\nITT-window attrition (all outcomes): {itt}; "
        f"DiD-window attrition: {did}; "
        f"max t4 attrition on any outcome: {int(table['t4_attrition'].max())}."
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
