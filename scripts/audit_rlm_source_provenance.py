# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproduce the Byrne/RLM source reconciliation and repaired CSV audit.

The identifying SPSS source remains in the private ``research-data-analysis`` Git
history and must not be copied into this repository. This audit reads that pinned
blob into a temporary file, compares only non-identifying assessment fields, and
prints no names or source codes.

Run::

    python scripts/audit_rlm_source_provenance.py \
        --source-repository /path/to/research-data-analysis
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from language_reading_predictors import paths

SOURCE_REPOSITORY = "dsegroup/research-data-analysis"
SOURCE_REVISION = "f36df93fe946b975cd701867a117e9ac188a1551"
SOURCE_PATH = "projects/reading-language-memory/original/data12345.sav"
SOURCE_GIT_BLOB_SHA1 = "591e14ceee2ffe61fe8af8e51ea35ac86ae8436f"
SOURCE_SHA256 = "e3cac5ff644ab9126fba25803e677f9492e3e076d8d611d8b3c0aa7ea322952c"

LEGACY_SECONDARY_REVISION = "fab4e8f0b513cd2f275ae1a29bed4c695d7f1ef6"
LEGACY_SECONDARY_PATH = "projects/reading-language-memory/original/data12345.csv"
LEGACY_SECONDARY_GIT_BLOB_SHA1 = "a7a7c2a3ca97c8a5caf591e7beae349492c72129"
LEGACY_SECONDARY_SHA256 = "e36e0e2dd880031870d57dd7e2620a27c9cc9c67ee58760f50285725d756997e"

SECONDARY_REVISION = "79ceb55086d502a739b2fd153990916c58096a25"
SECONDARY_PATH = "projects/reading-language-memory/original/data12345.csv"
SECONDARY_GIT_BLOB_SHA1 = "2cac37086cd299cf958ff51c2554faa54b3e7c70"
SECONDARY_SHA256 = "a9098f69e6854d31c665c7f2544ce8f7a26a4417606590bb00c0f18eed96ee6d"

PREPARED_WIDE_PATH = paths.DATA_DIR / "reading-language-memory" / "reading_language_memory_data_wide.csv"
PREPARED_LONG_PATH = paths.DATA_DIR / "reading-language-memory" / "reading_language_memory_data_long.csv"
PREPARED_WIDE_SHA256 = "b2262d6b3b7102594b3424c4a72f4237dc84087a7b18f6fc815ccdcd0d10a55c"
PREPARED_LONG_SHA256 = "68ea2e9c847c908b7217431af76abd45a940099ced2bfd9acf4dd69ba7e2e5f6"
PROVENANCE_MANIFEST_PATH = paths.DATA_DIR / "reading-language-memory" / "source_provenance.json"

EXPECTED_SHARED_FIELDS = 52
EXPECTED_SHARED_COLUMNS = {
    "readgrp",
    *(f"age{wave}" for wave in range(1, 6)),
    *(f"speed{wave}" for wave in range(3, 6)),
    *(f"basmat{wave}" for wave in range(3, 6)),
    *(
        f"{stub}{wave}"
        for stub in ("bassim", "basdig", "basnum", "basread", "basspel", "trog", "woco", "bpvs")
        for wave in range(1, 6)
    ),
}
EXPECTED_SOURCE_IDENTIFIER_FIELDS = {"name", "subno"}
EXPECTED_SOURCE_BASMAT_FIELDS = ["basmat3", "basmat4", "basmat5"]
EXPECTED_GROUP_COUNTS = {"1": 24, "2": 42, "3": 31}
EXPECTED_LEGACY_SECONDARY_GROUP_COUNTS = {"1": 23, "2": 42, "3": 31}
EXPECTED_OMITTED_SUBJECT_ID = "ID_25873B41B04B6AE6"
EXPECTED_OMITTED_GROUP = 1
EXPECTED_OMITTED_WAVES = [1, 2, 3]

_LONG_STUBS = (
    "age",
    "basmat",
    "bassim",
    "basdig",
    "basnum",
    "basread",
    "basspel",
    "trog",
    "woco",
    "bpvs",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _require_digest(payload: bytes, expected: str, label: str) -> None:
    observed = _sha256(payload)
    if observed != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, observed {observed}")


def _git_file(repository: Path, revision: str, path: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "show", f"{revision}:{path}"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not read {revision}:{path} from {repository}") from exc
    return completed.stdout


def _read_sav(payload: bytes) -> pd.DataFrame:
    try:
        import pyreadstat
    except ImportError as exc:  # pragma: no cover - dependency failure has its own message
        raise RuntimeError("pyreadstat is required to audit the pinned SPSS source") from exc

    with tempfile.NamedTemporaryFile(suffix=".sav") as handle:
        handle.write(payload)
        handle.flush()
        frame, _ = pyreadstat.read_sav(handle.name)
    frame.columns = frame.columns.str.lower()
    return frame


def _normalise_value(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _row_fingerprint(row: pd.Series, columns: list[str]) -> str:
    payload = "\x1f".join(_normalise_value(row[column]) for column in columns)
    return hashlib.sha256(payload.encode()).hexdigest()


def _fingerprints(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame.apply(_row_fingerprint, axis=1, columns=columns)


def _counter_difference_size(left: Counter[str], right: Counter[str]) -> int:
    return sum((left - right).values())


def _group_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(int(group)): int(count) for group, count in frame["readgrp"].value_counts().sort_index().items()}


def compare_source_frames(
    source: pd.DataFrame,
    legacy_secondary: pd.DataFrame,
    secondary: pd.DataFrame,
    prepared: pd.DataFrame,
) -> dict[str, Any]:
    """Compare participant rows without using identifying source columns."""
    for frame in (source, legacy_secondary, secondary, prepared):
        frame.columns = frame.columns.str.lower()

    shared = sorted(
        set(source.columns)
        & set(legacy_secondary.columns)
        & set(secondary.columns)
        & set(prepared.columns)
    )
    if len(shared) != EXPECTED_SHARED_FIELDS:
        raise ValueError(f"Expected {EXPECTED_SHARED_FIELDS} shared source fields, found {len(shared)}")
    if set(shared) != EXPECTED_SHARED_COLUMNS:
        raise ValueError("The shared source-field inventory differs from the signed-off RLM record")
    source_only = set(source.columns) - set(shared)
    if source_only != EXPECTED_SOURCE_IDENTIFIER_FIELDS:
        raise ValueError("The SPSS source contains an unexpected field outside the retained assessment inventory")
    source_basmat_fields = sorted(column for column in shared if column.startswith("basmat"))
    if source_basmat_fields != EXPECTED_SOURCE_BASMAT_FIELDS:
        raise ValueError("The SPSS source does not contain the expected native BASMAT3-BASMAT5 fields")

    source_fingerprints = _fingerprints(source, shared)
    legacy_secondary_fingerprints = _fingerprints(legacy_secondary, shared)
    secondary_fingerprints = _fingerprints(secondary, shared)
    prepared_fingerprints = _fingerprints(prepared, shared)
    source_counter = Counter(source_fingerprints)
    legacy_secondary_counter = Counter(legacy_secondary_fingerprints)
    secondary_counter = Counter(secondary_fingerprints)
    prepared_counter = Counter(prepared_fingerprints)

    if source_counter != prepared_counter:
        raise ValueError("The prepared extract does not match the pinned SPSS source on all shared fields")
    if _counter_difference_size(legacy_secondary_counter, source_counter):
        raise ValueError("The 96-row CSV contains a participant record absent from the pinned SPSS source")
    if _counter_difference_size(source_counter, legacy_secondary_counter) != 1:
        raise ValueError("The 96-row CSV must omit exactly one SPSS-source participant")
    if source_counter != secondary_counter:
        raise ValueError("The repaired 97-row CSV does not exactly match the pinned SPSS source")

    missing_fingerprint = next(iter((source_counter - legacy_secondary_counter).elements()))
    missing_prepared = prepared.loc[prepared_fingerprints.eq(missing_fingerprint)]
    if len(missing_prepared) != 1:
        raise ValueError("Could not identify one prepared participant omitted from the 96-row CSV")
    missing = missing_prepared.iloc[0]
    observed_waves = [
        wave
        for wave in range(1, 6)
        if any(pd.notna(missing.get(f"{stub}{wave}")) for stub in _LONG_STUBS)
    ]

    return {
        "shared_source_fields": len(shared),
        "source_fields": len(source.columns),
        "source_identifier_fields_excluded": len(source_only),
        "source_native_basmat_fields": source_basmat_fields,
        "retained_visual_recall_fields": 0,
        "source_participants": len(source),
        "legacy_secondary_export_participants": len(legacy_secondary),
        "secondary_export_participants": len(secondary),
        "prepared_participants": len(prepared),
        "source_group_counts": _group_counts(source),
        "legacy_secondary_export_group_counts": _group_counts(legacy_secondary),
        "secondary_export_group_counts": _group_counts(secondary),
        "prepared_group_counts": _group_counts(prepared),
        "prepared_rows_matching_source": len(prepared),
        "prepared_source_value_differences": 0,
        "legacy_secondary_rows_matching_source": len(legacy_secondary),
        "legacy_secondary_rows_not_in_source": 0,
        "source_rows_missing_from_legacy_secondary": 1,
        "secondary_rows_matching_source": len(secondary),
        "secondary_rows_not_in_source": 0,
        "source_rows_missing_from_secondary": 0,
        "secondary_source_value_differences": 0,
        "recovered_in_secondary": {
            "subject_id": str(missing["subject_id"]),
            "readgrp": int(missing["readgrp"]),
            "observed_waves": observed_waves,
        },
    }


def compare_long_with_wide(wide: pd.DataFrame, long: pd.DataFrame) -> dict[str, int]:
    """Confirm that the long file preserves every wide-file source value."""
    expected_ids = set(wide["subject_id"])
    if set(long["subject_id"]) != expected_ids:
        raise ValueError("Wide and long files contain different participant identifiers")
    if not long.groupby("subject_id").size().eq(5).all():
        raise ValueError("The long file must contain exactly five rows per participant")

    differences = 0
    wide_indexed = wide.set_index("subject_id")
    for row in long.itertuples(index=False):
        source = wide_indexed.loc[row.subject_id]
        if int(row.readgrp) != int(source["readgrp"]) or int(row.sex) != int(source["sex"]):
            differences += 1
        for stub in _LONG_STUBS:
            wide_value = source.get(f"{stub}{int(row.time)}", np.nan)
            long_value = getattr(row, stub)
            if not (pd.isna(wide_value) and pd.isna(long_value)) and wide_value != long_value:
                differences += 1
    if differences:
        raise ValueError(f"The long file differs from the prepared wide file in {differences} values")
    return {"long_rows": len(long), "long_rows_matching_wide": len(long), "long_value_differences": 0}


def build_manifest(
    source_payload: bytes,
    legacy_secondary_payload: bytes,
    secondary_payload: bytes,
    wide_payload: bytes,
    long_payload: bytes,
) -> dict[str, Any]:
    """Build the non-identifying, checksum-bound reconciliation manifest."""
    _require_digest(source_payload, SOURCE_SHA256, "SPSS source")
    _require_digest(legacy_secondary_payload, LEGACY_SECONDARY_SHA256, "legacy secondary CSV")
    _require_digest(secondary_payload, SECONDARY_SHA256, "repaired secondary CSV")
    _require_digest(wide_payload, PREPARED_WIDE_SHA256, "prepared wide CSV")
    _require_digest(long_payload, PREPARED_LONG_SHA256, "prepared long CSV")
    if _git_blob_sha1(source_payload) != SOURCE_GIT_BLOB_SHA1:
        raise ValueError("SPSS source Git blob identifier does not match the pinned blob")
    if _git_blob_sha1(legacy_secondary_payload) != LEGACY_SECONDARY_GIT_BLOB_SHA1:
        raise ValueError("Legacy secondary CSV Git blob identifier does not match the pinned blob")
    if _git_blob_sha1(secondary_payload) != SECONDARY_GIT_BLOB_SHA1:
        raise ValueError("Repaired secondary CSV Git blob identifier does not match the pinned blob")

    source = _read_sav(source_payload)
    legacy_secondary = pd.read_csv(io.BytesIO(legacy_secondary_payload), encoding="utf-8-sig")
    secondary = pd.read_csv(io.BytesIO(secondary_payload), encoding="utf-8-sig")
    wide = pd.read_csv(io.BytesIO(wide_payload))
    long = pd.read_csv(io.BytesIO(long_payload))
    comparison = compare_source_frames(source, legacy_secondary, secondary, wide)
    comparison.update(compare_long_with_wide(wide, long))

    if comparison["source_group_counts"] != EXPECTED_GROUP_COUNTS:
        raise ValueError("Pinned SPSS source group counts do not match the published starting groups")
    if comparison["legacy_secondary_export_group_counts"] != EXPECTED_LEGACY_SECONDARY_GROUP_COUNTS:
        raise ValueError("Legacy secondary CSV group counts do not match the reconciled omission")
    if comparison["secondary_export_group_counts"] != EXPECTED_GROUP_COUNTS:
        raise ValueError("Repaired secondary CSV group counts do not match the 97-person source")
    recovered = comparison["recovered_in_secondary"]
    if recovered != {
        "subject_id": EXPECTED_OMITTED_SUBJECT_ID,
        "readgrp": EXPECTED_OMITTED_GROUP,
        "observed_waves": EXPECTED_OMITTED_WAVES,
    }:
        raise ValueError("The participant omitted from the secondary CSV differs from the signed-off record")

    return {
        "schema_version": 3,
        "generated_by": "Codex/GPT-5",
        "audit_date": "2026-08-16",
        "decision": "source_provenance_confirmed",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "path": SOURCE_PATH,
            "git_blob_sha1": SOURCE_GIT_BLOB_SHA1,
            "sha256": SOURCE_SHA256,
        },
        "legacy_secondary_export": {
            "revision": LEGACY_SECONDARY_REVISION,
            "path": LEGACY_SECONDARY_PATH,
            "git_blob_sha1": LEGACY_SECONDARY_GIT_BLOB_SHA1,
            "sha256": LEGACY_SECONDARY_SHA256,
            "status": "incomplete_derivative",
        },
        "secondary_export": {
            "revision": SECONDARY_REVISION,
            "path": SECONDARY_PATH,
            "git_blob_sha1": SECONDARY_GIT_BLOB_SHA1,
            "sha256": SECONDARY_SHA256,
            "status": "complete_derivative",
        },
        "prepared": {
            "wide_path": "data/reading-language-memory/reading_language_memory_data_wide.csv",
            "wide_sha256": PREPARED_WIDE_SHA256,
            "long_path": "data/reading-language-memory/reading_language_memory_data_long.csv",
            "long_sha256": PREPARED_LONG_SHA256,
        },
        "comparison": comparison,
        "conclusion": (
            "The prepared 97-participant extract is an exact non-identifying-field match to the pinned 97-case "
            "SPSS source. The historical 96-row CSV was an incomplete derivative missing one Down-syndrome "
            "participant; the repaired 97-row CSV now exactly matches the source on the same 52 fields. The "
            "retained source has native BASMAT3-BASMAT5 fields and no retained visual-recall fields."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repository", required=True, type=Path, help="Local research-data-analysis checkout")
    parser.add_argument("--prepared-wide", type=Path, default=PREPARED_WIDE_PATH)
    parser.add_argument("--prepared-long", type=Path, default=PREPARED_LONG_PATH)
    parser.add_argument("--manifest", type=Path, default=PROVENANCE_MANIFEST_PATH)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write the reproduced manifest instead of comparing it",
    )
    args = parser.parse_args()

    source_payload = _git_file(args.source_repository, SOURCE_REVISION, SOURCE_PATH)
    legacy_secondary_payload = _git_file(
        args.source_repository,
        LEGACY_SECONDARY_REVISION,
        LEGACY_SECONDARY_PATH,
    )
    secondary_payload = _git_file(args.source_repository, SECONDARY_REVISION, SECONDARY_PATH)
    observed = build_manifest(
        source_payload,
        legacy_secondary_payload,
        secondary_payload,
        args.prepared_wide.read_bytes(),
        args.prepared_long.read_bytes(),
    )
    if args.write_manifest:
        args.manifest.write_text(
            json.dumps(observed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.manifest}")
        return
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError(f"Reproduced audit does not match {args.manifest}")
    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
