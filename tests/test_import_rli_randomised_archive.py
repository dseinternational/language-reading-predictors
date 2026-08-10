# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Checksum, ZIP-member, shape and atomic-write tests for the RLI importer."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.statistical_models import itt_missingness as missing

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "import_rli_randomised_archive.py"
)


@pytest.fixture(scope="module")
def importer():
    """Load the importer by path because ``scripts`` is not a Python package."""

    spec = importlib.util.spec_from_file_location("import_rli_randomised_archive", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zip_payload(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


def _archive_payload(*, rows: int = 57) -> bytes:
    group = np.asarray([1] * 29 + [2] * 28)
    included = np.asarray([1] * 28 + [0] + [1] * 26 + [0] * 2)
    post = np.full(57, np.nan)
    post[:28] = np.arange(28) % 20
    post[29:54] = np.arange(25) % 17
    index = np.arange(57)
    frame = pd.DataFrame(
        {
            "group": group,
            "area": 1 + (index % 2),
            "gender": 1 + ((index // 2) % 2),
            "included": included,
            "age_ts": 60 + index,
            "expr_vocab_raw_ts": 6 + (index % 60),
            "recep_vocab_raw_ts": 5 + (index % 58),
            "word_reading_raw_ts": index % 31,
            "letter_sound_raw_ts": index % 33,
            "word_reading_t2": post,
        }
    ).iloc[:rows]
    return frame.to_csv(index=False).encode("utf-8")


def _strict_loader(expected_sha256: str):
    def validate(path: Path):
        return missing.load_randomised_w_archive(
            path,
            expected_sha256=expected_sha256,
            local_wide_path=None,
        )

    return validate


def test_zip_extraction_requires_the_pinned_checksum(importer, monkeypatch):
    csv_payload = b"archive-data"
    zip_payload = _zip_payload(importer._ZIP_MEMBER, csv_payload)
    monkeypatch.setattr(importer, "RLI_ARCHIVE_ZIP_SHA256", _sha256(zip_payload))

    assert importer._archive_csv_from_zip(zip_payload) == csv_payload

    monkeypatch.setattr(importer, "RLI_ARCHIVE_ZIP_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        importer._archive_csv_from_zip(zip_payload)


def test_zip_extraction_requires_the_registered_member(importer, monkeypatch):
    zip_payload = _zip_payload("unexpected/archive.csv", b"archive-data")
    monkeypatch.setattr(importer, "RLI_ARCHIVE_ZIP_SHA256", _sha256(zip_payload))

    with pytest.raises(ValueError, match="has no member"):
        importer._archive_csv_from_zip(zip_payload)


def test_bad_csv_checksum_does_not_replace_an_existing_install(
    importer, monkeypatch, tmp_path
):
    destination = tmp_path / "generated" / "archive.csv"
    destination.parent.mkdir()
    destination.write_bytes(b"known-good-existing-copy")
    monkeypatch.setattr(importer, "RLI_ARCHIVE_CSV_SHA256", "0" * 64)

    with pytest.raises(ValueError, match="CSV checksum mismatch"):
        importer._install(b"different-copy", destination)

    assert destination.read_bytes() == b"known-good-existing-copy"


def test_install_atomically_replaces_then_validates_a_57_row_archive(
    importer, monkeypatch, tmp_path
):
    payload = _archive_payload()
    digest = _sha256(payload)
    destination = tmp_path / "generated" / "archive.csv"
    destination.parent.mkdir()
    destination.write_bytes(b"old-copy")
    real_replace = importer.os.replace
    replacements: list[tuple[Path, Path]] = []

    def replace(source, target):
        source_path = Path(source)
        target_path = Path(target)
        assert source_path.parent == destination.parent
        assert source_path.read_bytes() == payload
        replacements.append((source_path, target_path))
        real_replace(source, target)

    monkeypatch.setattr(importer, "RLI_ARCHIVE_CSV_SHA256", digest)
    monkeypatch.setattr(importer, "load_randomised_w_archive", _strict_loader(digest))
    monkeypatch.setattr(importer.os, "replace", replace)

    importer._install(payload, destination)

    assert len(replacements) == 1
    assert replacements[0][1] == destination
    assert destination.read_bytes() == payload
    assert not list(destination.parent.glob(f".{destination.name}-*"))


def test_install_surfaces_the_57_row_shape_validation(importer, monkeypatch, tmp_path):
    payload = _archive_payload(rows=56)
    digest = _sha256(payload)
    destination = tmp_path / "generated" / "archive.csv"
    destination.parent.mkdir()
    destination.write_bytes(b"known-good-existing-copy")
    monkeypatch.setattr(importer, "RLI_ARCHIVE_CSV_SHA256", digest)
    monkeypatch.setattr(importer, "load_randomised_w_archive", _strict_loader(digest))

    with pytest.raises(ValueError, match="must contain 57 rows"):
        importer._install(payload, destination)

    assert destination.read_bytes() == b"known-good-existing-copy"
    assert not list(destination.parent.glob(f".{destination.name}-*"))
