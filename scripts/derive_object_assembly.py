# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Derive the Object Assembly column into the analysis files from the deposit.

WPPSI-III Object Assembly at t1 is the second non-verbal subtest the original
trial reported, and it is present in the committed deposit
(``data/dse-rli-trial-data-archive.csv``) but in neither derived analysis file.
It exists so the suite's single-subtest ability adjustment can be checked
against a two-indicator composite (``Variables.OBJASS_C``).

The deposit and the analysis files carry **different anonymised subject
labels**, so the column cannot be joined on an identifier.  It is joined on the
same 71-field row fingerprint the word-reading missingness loader already uses
to reconcile the two, which that loader proves is one-to-one across the 54
analysed children.  No identifier crosswalk is written: the fingerprint is
computed, used and discarded.

``--check`` recomputes the column and compares it with what is committed,
without writing.  ``tests/test_derive_object_assembly.py`` runs that mode, so
the committed column cannot drift from the deposit unnoticed.

Usage::

    uv run python scripts/derive_object_assembly.py --check
    uv run python scripts/derive_object_assembly.py --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from language_reading_predictors import paths as _paths
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.itt_missingness import (
    ARCHIVE_TO_LOCAL_WIDE,
    RLI_ARCHIVE_LOCAL_CSV,
    _row_fingerprints,
)

#: The deposit's own name for the subtest, and the wide file's wave suffix.
ARCHIVE_COLUMN = "object_ass_raw_t1"
WIDE_COLUMN = f"{V.OBJASS}1"

LONG_PATH = _paths.DATA_DIR / "rli_data_long.csv"
WIDE_PATH = _paths.DATA_DIR / "rli_data_wide.csv"


def object_assembly_by_wide_row(
    *,
    archive_path: Path = RLI_ARCHIVE_LOCAL_CSV,
    wide: pd.DataFrame | None = None,
) -> pd.Series:
    """Object Assembly aligned to ``rli_data_wide.csv`` row order.

    Raises when the fingerprint match is not one-to-one over the 54 analysed
    children, so a changed data file fails here rather than silently
    mis-assigning a score to the wrong child.
    """
    archive = pd.read_csv(
        archive_path, encoding="utf-8-sig", skipinitialspace=True, na_values=["", " "]
    )
    if ARCHIVE_COLUMN not in archive.columns:
        raise ValueError(f"deposit has no {ARCHIVE_COLUMN!r} column: {archive_path}")
    included = archive[archive["included"] == 1]

    if wide is None:
        wide = pd.read_csv(WIDE_PATH)

    source_columns = [c for c in ARCHIVE_TO_LOCAL_WIDE if c in included.columns]
    local_columns = [ARCHIVE_TO_LOCAL_WIDE[c] for c in source_columns]
    missing = [c for c in local_columns if c not in wide.columns]
    if missing:
        raise ValueError(f"analysis file is missing reconciliation columns: {missing}")

    source = _row_fingerprints(
        included[source_columns].rename(columns=ARCHIVE_TO_LOCAL_WIDE)
    )
    local = _row_fingerprints(wide[local_columns])
    if not source.is_unique or not local.is_unique:
        raise ValueError("row fingerprints are not one-to-one")
    if set(source) != set(local):
        raise ValueError("the deposit's included rows do not reconcile with the repository")

    by_fingerprint = dict(zip(source.to_numpy(), included[ARCHIVE_COLUMN].to_numpy(), strict=True))
    values = pd.Series(
        [by_fingerprint[f] for f in local.to_numpy()], index=wide.index, dtype=float
    )
    if values.isna().any():
        raise ValueError("Object Assembly is unexpectedly missing for a reconciled child")
    return values


def _format(value: float | None) -> str:
    """Render a subtest score as the integer count it is, or empty when absent."""
    if value is None:
        return ""
    return str(int(value))


def _append_column(path: Path, name: str, values: list[float | None]) -> None:
    """Append one column to a CSV **textually**, leaving every existing byte alone.

    Deliberately not a ``read_csv``/``to_csv`` round trip. Re-serialising the
    frame rewrites every float, and pandas' repr differs from the committed text
    in the last bit for two columns in each file — a ~2e-16 relative change that
    is numerically meaningless but would still alter committed research data and
    its digest for no reason.
    """
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    trailing = raw.endswith(newline)
    lines = raw.splitlines()
    if len(lines) != len(values) + 1:
        raise ValueError(
            f"{path.name}: {len(lines) - 1} data rows but {len(values)} values"
        )
    header, *rows = lines
    if name in header.split(","):
        raise ValueError(f"{path.name}: column {name!r} is already present")
    out = [f"{header},{name}"] + [
        f"{row},{_format(value)}" for row, value in zip(rows, values, strict=True)
    ]
    path.write_bytes((newline.join(out) + (newline if trailing else "")).encode("utf-8"))


def _values(write: bool) -> tuple[list[float | None], list[float | None]]:
    wide = pd.read_csv(WIDE_PATH)
    long = pd.read_csv(LONG_PATH)
    derived = object_assembly_by_wide_row(wide=wide)
    # The long file carries the subtest at t1 only, exactly as Block Design does;
    # every consumer broadcasts it from there.
    by_subject = dict(zip(wide[V.SUBJECT_ID], derived, strict=True))
    wide_values: list[float | None] = [float(v) for v in derived]
    long_values: list[float | None] = [
        float(by_subject[s]) if t == 1 else None
        for s, t in zip(long[V.SUBJECT_ID], long[V.TIME], strict=True)
    ]
    return wide_values, long_values


def _apply(write: bool) -> int:
    wide_values, long_values = _values(write)

    if not write:
        wide = pd.read_csv(WIDE_PATH)
        long = pd.read_csv(LONG_PATH)
        problems: list[str] = []
        for path, frame, column, expected in (
            (WIDE_PATH, wide, WIDE_COLUMN, wide_values),
            (LONG_PATH, long, V.OBJASS, long_values),
        ):
            if column not in frame.columns:
                problems.append(f"{path.name}: {column!r} is not present")
                continue
            stored = pd.to_numeric(frame[column], errors="coerce")
            want = pd.Series(expected, dtype="float64")
            if not stored.fillna(-1).eq(want.fillna(-1)).all():
                problems.append(f"{path.name}: {column!r} differs from the deposit")
        for problem in problems:
            print(f"MISMATCH {problem}", file=sys.stderr)
        if problems:
            return 1
        print(
            f"Object Assembly matches the deposit in {WIDE_PATH.name} and {LONG_PATH.name}"
        )
        return 0

    _append_column(WIDE_PATH, WIDE_COLUMN, wide_values)
    _append_column(LONG_PATH, V.OBJASS, long_values)
    print(f"Appended {WIDE_COLUMN!r} to {WIDE_PATH} and {V.OBJASS!r} to {LONG_PATH}")
    print("Update RLI_LOCAL_WIDE_SHA256 in statistical_models/itt_missingness.py.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Compare without writing")
    mode.add_argument("--write", action="store_true", help="Write the column")
    args = parser.parse_args()
    return _apply(write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
