# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Analysis-set and outcome-missingness audit for the RLI trial (#230/#341).

The published trial randomised 57 children, but the archived modelling dataset
contains 54. The first table therefore distinguishes absence from the dataset
from within-dataset outcome missingness. The second counts, per outcome and wave,
how many archived children have a non-missing score and how many are lost across
the windows the headline estimands use —

- the randomised **ITT window** t1 -> t2 (a child with a t1 baseline but no t2 post),
- the **DiD crossover window** t1 -> t2 -> t3 (a t2 score but no t3), and
- the final **maintenance wave** t4 (a t3 score but no t4),

so a reviewer can see where — if anywhere — outcome attrition could bias an estimand.
It fits no model and makes no causal claim; it only tabulates observed missingness.

Standalone by design: reads ``data/rli_data_long.csv`` directly and resolves its output
path through ``language_reading_predictors.paths``. Writes
``analysis_set_audit.csv`` and ``attrition_audit.csv`` to ``output/audit/``
(gitignored) and prints both tables.

Run::

    python scripts/attrition_audit.py
"""

from __future__ import annotations

import argparse

import pandas as pd

from language_reading_predictors import paths as _paths
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.itt_audit import (
    CONTROL_G,
    INTERVENTION_G,
    RLI_RANDOMISED_BY_G,
)
from language_reading_predictors.statistical_models.measures import MEASURES

# Outcomes audited: the eight standardised ITT outcomes plus nonword reading, in a
# reading order. Columns and labels come from MEASURES so this cannot drift.
OUTCOMES: tuple[str, ...] = ("W", "R", "E", "L", "B", "F", "T", "P", "N")
WAVES: tuple[int, ...] = (1, 2, 3, 4)

_DATA_PATH = _paths.DATA_DIR / "rli_data_long.csv"

# Raw trial group codes are 1/2; the model layer recodes them to 1/0. Keep the
# published allocation counts in itt_audit.py as the single source of truth.
RANDOMISED_BY_GROUP: dict[int, tuple[str, int]] = {
    1: ("intervention", RLI_RANDOMISED_BY_G[INTERVENTION_G]),
    2: ("control", RLI_RANDOMISED_BY_G[CONTROL_G]),
}


def _wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # pivot (not pivot_table): for an audit the counts must be trustworthy, so a
    # duplicate (subject_id, time) row should *raise* rather than be silently averaged
    # by the default aggfunc='mean' (#298 review). The panel is balanced today (216 =
    # 54x4), so this is a guard, not a behaviour change.
    return df.pivot(index=V.SUBJECT_ID, columns=V.TIME, values=column)


def analysis_set_audit(df: pd.DataFrame) -> pd.DataFrame:
    """Compare published randomised allocation with children in the CSV."""
    subject_groups = df[[V.SUBJECT_ID, V.GROUP]].drop_duplicates()
    per_subject = subject_groups.groupby(V.SUBJECT_ID, dropna=False)[V.GROUP].nunique(
        dropna=False
    )
    if not bool((per_subject == 1).all()):
        bad = per_subject[per_subject != 1].index.tolist()
        raise ValueError(f"group assignment is not unique for subject(s): {bad}")

    observed_group = subject_groups.set_index(V.SUBJECT_ID)[V.GROUP]
    invalid = sorted(set(observed_group.dropna().unique()) - set(RANDOMISED_BY_GROUP))
    if observed_group.isna().any() or invalid:
        raise ValueError(
            "analysis-set audit requires complete group codes 1/2; "
            f"invalid values: {invalid}"
        )

    rows: list[dict[str, str | int]] = []
    for code, (arm, randomised_n) in RANDOMISED_BY_GROUP.items():
        dataset_n = int((observed_group == code).sum())
        if dataset_n > randomised_n:
            raise ValueError(
                f"archived {arm} count {dataset_n} exceeds published randomised "
                f"count {randomised_n}"
            )
        rows.append(
            {
                "arm": arm,
                "randomised_n": randomised_n,
                "dataset_n": dataset_n,
                "absent_from_dataset_n": randomised_n - dataset_n,
            }
        )
    rows.append(
        {
            "arm": "total",
            "randomised_n": sum(v[1] for v in RANDOMISED_BY_GROUP.values()),
            "dataset_n": int(observed_group.size),
            "absent_from_dataset_n": (
                sum(v[1] for v in RANDOMISED_BY_GROUP.values())
                - int(observed_group.size)
            ),
        }
    )
    return pd.DataFrame(rows)


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Per-outcome wave completeness and window-attrition counts."""
    rows: list[dict] = []
    for sym in OUTCOMES:
        col = MEASURES[sym].column
        w = _wide(df, col)
        present = {t: w[t].notna() if t in w.columns else pd.Series(False, index=w.index) for t in WAVES}
        # Single-leg attrition: had the earlier score, missing the next one. Column names
        # are per-LEG (not per-chain): the full DiD crossover chain t1->t2->t3 is complete
        # for an outcome iff BOTH itt_window_attrition (t1->t2) and t2_to_t3_attrition are 0
        # (#298 review — avoids over-reading a single column as the whole chain).
        itt_attr = int((present[1] & ~present[2]).sum())  # t1 baseline, no t2 post
        t2_to_t3_attr = int((present[2] & ~present[3]).sum())  # t2, no t3 (crossover leg)
        t4_attr = int((present[3] & ~present[4]).sum())  # t3, no t4 (maintenance leg)
        rows.append(
            {
                "outcome": sym,
                "measure": MEASURES[sym].label,
                "n_t1": int(present[1].sum()),
                "n_t2": int(present[2].sum()),
                "n_t3": int(present[3].sum()),
                "n_t4": int(present[4].sum()),
                "itt_window_attrition": itt_attr,
                "t2_to_t3_attrition": t2_to_t3_attr,
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
    analysis_set = analysis_set_audit(df)
    table = audit(df)

    out_dir = _paths.output_root() / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_set_path = out_dir / "analysis_set_audit.csv"
    analysis_set.to_csv(analysis_set_path, index=False)
    out_path = out_dir / "attrition_audit.csv"
    table.to_csv(out_path, index=False)

    print("\n=== published randomised allocation -> archived dataset ===")
    print(analysis_set.to_string(index=False))
    print(
        "\nThe outcome table below is conditional on the "
        f"{n_subjects}-child archived dataset; it cannot audit outcomes for the "
        "three randomised children who are absent from that file."
    )
    print("\n=== within-dataset outcome missingness by wave + window attrition ===")
    print(table.to_string(index=False))
    itt = int(table["itt_window_attrition"].sum())
    t2t3 = int(table["t2_to_t3_attrition"].sum())
    print(
        f"\nITT-window (t1->t2) attrition (all outcomes): {itt}; "
        f"t2->t3 attrition: {t2t3}; "
        f"max t4 attrition on any outcome: {int(table['t4_attrition'].max())}. "
        "(The DiD chain t1->t2->t3 is complete for an outcome iff both are 0.)"
    )
    print(f"wrote {analysis_set_path}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
