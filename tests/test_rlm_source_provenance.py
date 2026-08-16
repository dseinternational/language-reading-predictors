# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the checksum-bound Byrne/RLM source reconciliation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from language_reading_predictors import paths

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit_rlm_source_provenance.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_rlm_source_provenance", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shared = {"readgrp": [1, 2, 1]}
    shared.update(
        {
            column: [index, index + 1, index + 2]
            for index, column in enumerate(
                sorted(audit_column for audit_column in _expected_shared_columns())
            )
            if column != "readgrp"
        }
    )
    source = pd.DataFrame({"name": ["a", "b", "c"], "subno": [1, 2, 3], **shared})
    legacy_secondary = pd.DataFrame(
        {"code": ["x", "y"], "sex": [0, 1], **{key: values[:2] for key, values in shared.items()}}
    )
    secondary = pd.DataFrame({"code": ["x", "y", "z"], "sex": [0, 1, 0], **shared})
    prepared = pd.DataFrame(
        {
            "subject_id": ["ID_A", "ID_B", "ID_C"],
            "sex": [0, 1, 0],
            **shared,
        }
    )
    return source, legacy_secondary, secondary, prepared


def _expected_shared_columns() -> set[str]:
    return {
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


def test_frame_comparison_validates_the_repaired_export(audit):
    source, legacy_secondary, secondary, prepared = _synthetic_frames()

    result = audit.compare_source_frames(source, legacy_secondary, secondary, prepared)

    assert result["prepared_rows_matching_source"] == 3
    assert result["legacy_secondary_rows_matching_source"] == 2
    assert result["secondary_rows_matching_source"] == 3
    assert result["source_rows_missing_from_legacy_secondary"] == 1
    assert result["source_rows_missing_from_secondary"] == 0
    assert result["recovered_in_secondary"]["subject_id"] == "ID_C"
    assert result["prepared_source_value_differences"] == 0
    assert result["source_native_basmat_fields"] == ["basmat3", "basmat4", "basmat5"]
    assert result["retained_visual_recall_fields"] == 0


def test_committed_manifest_is_bound_to_the_prepared_extract(audit):
    manifest = json.loads(audit.PROVENANCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    wide = paths.DATA_DIR / "reading-language-memory" / "reading_language_memory_data_wide.csv"
    long = paths.DATA_DIR / "reading-language-memory" / "reading_language_memory_data_long.csv"

    assert manifest["decision"] == "source_provenance_confirmed"
    assert manifest["source"]["sha256"] == audit.SOURCE_SHA256
    assert manifest["legacy_secondary_export"]["status"] == "incomplete_derivative"
    assert manifest["secondary_export"]["status"] == "complete_derivative"
    assert manifest["secondary_export"]["sha256"] == audit.SECONDARY_SHA256
    assert manifest["prepared"]["wide_sha256"] == hashlib.sha256(wide.read_bytes()).hexdigest()
    assert manifest["prepared"]["long_sha256"] == hashlib.sha256(long.read_bytes()).hexdigest()
    assert manifest["comparison"]["prepared_participants"] == 97
    assert manifest["comparison"]["prepared_source_value_differences"] == 0
    assert manifest["comparison"]["source_rows_missing_from_secondary"] == 0
    assert manifest["comparison"]["source_fields"] == 54
    assert manifest["comparison"]["source_identifier_fields_excluded"] == 2
    assert manifest["comparison"]["source_native_basmat_fields"] == [
        "basmat3",
        "basmat4",
        "basmat5",
    ]
    assert manifest["comparison"]["retained_visual_recall_fields"] == 0
    assert manifest["comparison"]["recovered_in_secondary"] == {
        "observed_waves": [1, 2, 3],
        "readgrp": 1,
        "subject_id": audit.EXPECTED_OMITTED_SUBJECT_ID,
    }
