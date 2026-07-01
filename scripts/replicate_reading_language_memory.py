# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproduce first-pass summaries from the reading-language-memory data.

This script is the audit/replication entry point for the Byrne, MacDonald, and
Buckley longitudinal dataset copied under ``data/reading-language-memory/``.
It writes CSVs under ``output/reading_language_memory/replication/``:

- complete-case Table 2-style means and standard deviations for waves 1-3;
- available-case wave summaries for the same measures;
- sample and observation-count audits;
- raw and age-adjusted correlations with BAS word reading.

The prepared extract does not currently include the visual recall variables
reported in the paper's correlation tables, so the correlation reproduction is
partial by design.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    REPO_ROOT
    / "data"
    / "reading-language-memory"
    / "reading_language_memory_data_long.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "output" / "reading_language_memory" / "replication"
)

PAPER_WAVES: tuple[int, ...] = (1, 2, 3)

GROUP_LABELS: dict[int, str] = {
    1: "Down syndrome",
    2: "Average readers",
    3: "Reading-matched",
}

MEASURE_LABELS: dict[str, str] = {
    "basread": "BAS word reading",
    "woco": "WORD reading comprehension",
    "basspel": "BAS spelling",
    "bpvs": "BPVS receptive vocabulary",
    "trog": "TROG receptive grammar",
    "bassim": "BAS similarities/verbal reasoning",
    "basdig": "BAS recall of digits",
    "basnum": "BAS number skills",
    "basmat": "BAS matrices/non-verbal reasoning",
}

TABLE2_MEASURES: tuple[str, ...] = (
    "basread",
    "woco",
    "basspel",
    "bpvs",
    "trog",
    "bassim",
    "basdig",
    "basnum",
)

CORRELATION_PREDICTORS: tuple[str, ...] = (
    "age",
    "woco",
    "basspel",
    "trog",
    "bpvs",
    "basdig",
    "bassim",
    "basnum",
    "basmat",
)

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"subject_id", "time", "readgrp", "sex", "age", "basread"}
    | set(TABLE2_MEASURES)
    | set(CORRELATION_PREDICTORS)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce first-pass reading-language-memory summaries.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Long-format CSV to read (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        missing_cols = ", ".join(missing)
        raise ValueError(f"{path} is missing required columns: {missing_cols}")

    df = df.copy()
    df["time"] = df["time"].astype(int)
    df["readgrp"] = df["readgrp"].astype(int)
    df["readgrp_label"] = df["readgrp"].map(GROUP_LABELS)
    return df.sort_values(["readgrp", "subject_id", "time"]).reset_index(drop=True)


def sample_overview(df: pd.DataFrame) -> pd.DataFrame:
    subjects = df[["subject_id", "readgrp", "readgrp_label", "sex"]].drop_duplicates()
    return (
        subjects.groupby(["readgrp", "readgrp_label"], dropna=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            sex_0=("sex", lambda s: int((s == 0).sum())),
            sex_1=("sex", lambda s: int((s == 1).sum())),
        )
        .reset_index()
    )


def observation_counts(df: pd.DataFrame, measures: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, group_df in df.groupby("readgrp", sort=True):
        for wave, wave_df in group_df.groupby("time", sort=True):
            for measure in measures:
                rows.append(
                    {
                        "readgrp": group,
                        "readgrp_label": GROUP_LABELS[group],
                        "time": wave,
                        "measure": measure,
                        "measure_label": MEASURE_LABELS[measure],
                        "n_nonmissing": int(wave_df[measure].notna().sum()),
                    }
                )
    return pd.DataFrame(rows)


def available_case_summary(df: pd.DataFrame, measures: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paper_df = df[df["time"].isin(PAPER_WAVES)]
    for measure in measures:
        for (group, wave), part in paper_df.groupby(["readgrp", "time"], sort=True):
            values = part[measure].dropna()
            rows.append(
                {
                    "measure": measure,
                    "measure_label": MEASURE_LABELS[measure],
                    "readgrp": group,
                    "readgrp_label": GROUP_LABELS[group],
                    "time": wave,
                    "n": int(values.size),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def complete_case_summary(df: pd.DataFrame, measures: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paper_df = df[df["time"].isin(PAPER_WAVES)]
    for measure in measures:
        for group, part in paper_df.groupby("readgrp", sort=True):
            wide = part.pivot(index="subject_id", columns="time", values=measure)
            complete = wide.dropna(subset=list(PAPER_WAVES))
            row: dict[str, object] = {
                "measure": measure,
                "measure_label": MEASURE_LABELS[measure],
                "readgrp": group,
                "readgrp_label": GROUP_LABELS[group],
                "n_complete": int(len(complete)),
            }
            for wave in PAPER_WAVES:
                values = complete[wave]
                row[f"time_{wave}_mean"] = values.mean()
                row[f"time_{wave}_sd"] = values.std(ddof=1)
            rows.append(row)
    return pd.DataFrame(rows)


def pearson_r(x: pd.Series, y: pd.Series) -> float:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.pearsonr(x, y).statistic)


def p_value_from_r(r: float, df: int) -> float:
    if math.isnan(r) or df <= 0:
        return math.nan
    if abs(r) >= 1:
        return 0.0
    t_stat = abs(r) * math.sqrt(df / (1 - r**2))
    return float(2 * stats.t.sf(t_stat, df))


def residualize_on_age(series: pd.Series, age: pd.Series) -> pd.Series:
    matrix = np.column_stack([np.ones(len(age)), age.to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(matrix, series.to_numpy(dtype=float), rcond=None)
    return pd.Series(series.to_numpy(dtype=float) - matrix @ beta, index=series.index)


def correlations_with_reading(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paper_df = df[df["time"].isin(PAPER_WAVES)]

    for group, group_df in paper_df.groupby("readgrp", sort=True):
        for wave, wave_df in group_df.groupby("time", sort=True):
            for predictor in CORRELATION_PREDICTORS:
                raw = wave_df[["basread", predictor]].dropna()
                raw_n = int(len(raw))
                raw_df = raw_n - 2
                raw_r = pearson_r(raw["basread"], raw[predictor]) if raw_n >= 3 else math.nan
                rows.append(
                    {
                        "correlation_type": "raw",
                        "readgrp": group,
                        "readgrp_label": GROUP_LABELS[group],
                        "time": wave,
                        "target": "basread",
                        "target_label": MEASURE_LABELS["basread"],
                        "predictor": predictor,
                        "predictor_label": (
                            "Age in months"
                            if predictor == "age"
                            else MEASURE_LABELS[predictor]
                        ),
                        "n": raw_n,
                        "df": raw_df if raw_n >= 3 else math.nan,
                        "r": raw_r,
                        "p": p_value_from_r(raw_r, raw_df),
                    }
                )

                if predictor == "age":
                    continue
                partial = wave_df[["basread", predictor, "age"]].dropna()
                partial_n = int(len(partial))
                partial_df = partial_n - 3
                if partial_n >= 4:
                    basread_resid = residualize_on_age(partial["basread"], partial["age"])
                    predictor_resid = residualize_on_age(partial[predictor], partial["age"])
                    partial_r = pearson_r(basread_resid, predictor_resid)
                else:
                    partial_r = math.nan
                rows.append(
                    {
                        "correlation_type": "age_adjusted",
                        "readgrp": group,
                        "readgrp_label": GROUP_LABELS[group],
                        "time": wave,
                        "target": "basread",
                        "target_label": MEASURE_LABELS["basread"],
                        "predictor": predictor,
                        "predictor_label": MEASURE_LABELS[predictor],
                        "n": partial_n,
                        "df": partial_df if partial_n >= 4 else math.nan,
                        "r": partial_r,
                        "p": p_value_from_r(partial_r, partial_df),
                    }
                )

    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sample_overview.csv": sample_overview(df),
        "observations_by_group_time_measure.csv": observation_counts(
            df,
            MEASURE_LABELS.keys(),
        ),
        "table2_available_case_summary.csv": available_case_summary(
            df,
            TABLE2_MEASURES,
        ),
        "table2_complete_case_summary.csv": complete_case_summary(
            df,
            TABLE2_MEASURES,
        ),
        "basread_correlations.csv": correlations_with_reading(df),
    }

    paths: list[Path] = []
    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path, index=False, float_format="%.6g")
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    df = load_data(args.data)
    paths = write_outputs(df, args.output_dir)

    print(f"Read {args.data}")
    print(f"Rows: {len(df):,}; subjects: {df['subject_id'].nunique():,}")
    print(f"Wrote {len(paths)} files to {args.output_dir}:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
