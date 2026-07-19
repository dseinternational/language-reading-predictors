# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for portable statistical-run provenance and failure records."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from language_reading_predictors.statistical_models import provenance


def test_package_versions_retains_missing_distribution(monkeypatch):
    provenance._cached_package_versions.cache_clear()

    def fake_version(distribution: str) -> str:
        if distribution == "missing-dist":
            raise metadata.PackageNotFoundError(distribution)
        return "1.2.3"

    monkeypatch.setattr(provenance.metadata, "version", fake_version)

    assert provenance.package_versions(
        {"available": "available-dist", "missing": "missing-dist"}
    ) == {"available": "1.2.3", "missing": None}
    provenance._cached_package_versions.cache_clear()


def test_package_versions_caches_each_distribution_set(monkeypatch):
    provenance._cached_package_versions.cache_clear()
    calls: list[str] = []

    def fake_version(distribution: str) -> str:
        calls.append(distribution)
        return "1.2.3"

    monkeypatch.setattr(provenance.metadata, "version", fake_version)

    expected = {"available": "1.2.3"}
    assert provenance.package_versions({"available": "available-dist"}) == expected
    assert provenance.package_versions({"available": "available-dist"}) == expected
    assert calls == ["available-dist"]
    provenance._cached_package_versions.cache_clear()


def test_source_provenance_fails_safely_without_git(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_git_output", lambda *args, **kwargs: None)

    assert provenance.source_provenance(tmp_path) == {
        "repository_root": None,
        "commit": None,
        "branch": None,
        "dirty": None,
    }


def test_source_provenance_distinguishes_clean_checkout(monkeypatch, tmp_path):
    responses = iter([str(tmp_path), "abc123", "main", ""])
    monkeypatch.setattr(
        provenance,
        "_git_output",
        lambda *args, **kwargs: next(responses),
    )

    assert provenance.source_provenance(tmp_path) == {
        "repository_root": str(tmp_path),
        "commit": "abc123",
        "branch": "main",
        "dirty": False,
    }


def test_source_provenance_defaults_to_package_checkout(monkeypatch):
    queried_directories: list[Path] = []

    def fake_git_output(*_args, cwd: Path, **_kwargs):
        queried_directories.append(cwd)
        return None

    monkeypatch.setattr(provenance, "_git_output", fake_git_output)

    provenance.source_provenance()

    assert queried_directories == [Path(provenance.__file__).resolve().parent]


def test_run_provenance_records_invocation_runtime_and_packages(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance.sys, "argv", ["fit_statistical_model.py", "model"])
    monkeypatch.setattr(
        provenance,
        "source_provenance",
        lambda: {"commit": "abc123", "branch": "main", "dirty": False},
    )
    monkeypatch.setattr(provenance, "package_versions", lambda: {"pymc": "5.0"})

    record = provenance.run_provenance()

    assert record["invocation"] == {
        "argv": ["fit_statistical_model.py", "model"],
        "working_directory": str(tmp_path),
    }
    assert record["source"]["commit"] == "abc123"
    assert record["runtime"]["python_version"]
    assert record["runtime"]["python_executable"]
    assert record["packages"] == {"pymc": "5.0"}


def test_write_failure_record_preserves_traceback_and_context(monkeypatch, tmp_path):
    timestamp = datetime(2026, 7, 18, 12, 34, 56, tzinfo=UTC)
    monkeypatch.setattr(provenance, "_utc_now", lambda: timestamp)
    monkeypatch.setattr(
        provenance.uuid,
        "uuid4",
        lambda: uuid.UUID("12345678-90ab-cdef-1234-567890abcdef"),
    )
    monkeypatch.setattr(
        provenance,
        "run_provenance",
        lambda: {"source": {"commit": "abc123"}},
    )

    path = provenance.write_failure_record(
        tmp_path,
        model_id="lrp-rli-itt-001",
        config="reporting",
        error=RuntimeError("sampler failed"),
        traceback_text="Traceback (most recent call last):\nRuntimeError: sampler failed\n",
    )

    assert path == (
        tmp_path
        / "run_metadata"
        / "failures"
        / "lrp-rli-itt-001-20260718T123456Z-1234567890ab.json"
    )
    record = json.loads(path.read_text())
    assert record["model_id"] == "lrp-rli-itt-001"
    assert record["config"] == "reporting"
    assert record["exception"] == {
        "type": "RuntimeError",
        "message": "sampler failed",
    }
    assert record["traceback"].startswith("Traceback")
    assert record["provenance"]["source"]["commit"] == "abc123"


def test_failure_records_do_not_collide_within_one_second(monkeypatch, tmp_path):
    timestamp = datetime(2026, 7, 18, 12, 34, 56, tzinfo=UTC)
    identifiers = iter(
        [
            uuid.UUID("12345678-90ab-cdef-1234-567890abcdef"),
            uuid.UUID("abcdef12-3456-7890-abcd-ef1234567890"),
        ]
    )
    monkeypatch.setattr(provenance, "_utc_now", lambda: timestamp)
    monkeypatch.setattr(provenance.uuid, "uuid4", lambda: next(identifiers))
    monkeypatch.setattr(provenance, "run_provenance", lambda: {})

    paths = [
        provenance.write_failure_record(
            tmp_path,
            model_id="lrp-rli-itt-001",
            config="reporting",
            error=RuntimeError("sampler failed"),
            traceback_text="traceback",
        )
        for _ in range(2)
    ]

    assert paths[0] != paths[1]
    assert all(path.exists() for path in paths)


def test_conda_lock_records_exact_builds_and_skips_broken_metadata(tmp_path):
    conda_meta = tmp_path / "conda-meta"
    conda_meta.mkdir()
    (conda_meta / "numpy.json").write_text(
        json.dumps(
            {
                "name": "numpy",
                "version": "2.4.6",
                "build": "py314h123_0",
                "build_number": 0,
                "channel": "https://conda.anaconda.org/conda-forge",
                "subdir": "osx-arm64",
                "fn": "numpy-2.4.6-py314h123_0.conda",
                "sha256": "abc123",
            }
        ),
        encoding="utf-8",
    )
    (conda_meta / "broken.json").write_text("not json", encoding="utf-8")

    assert provenance._conda_lock_records(tmp_path) == [
        {
            "name": "numpy",
            "version": "2.4.6",
            "build": "py314h123_0",
            "build_number": 0,
            "channel": "https://conda.anaconda.org/conda-forge",
            "subdir": "osx-arm64",
            "fn": "numpy-2.4.6-py314h123_0.conda",
            "sha256": "abc123",
        }
    ]


def test_direct_url_sanitisation_removes_credentials_query_and_fragment():
    raw = json.dumps(
        {
            "url": "https://user:secret@example.test/research.git?token=abc#frag",
            "vcs_info": {"vcs": "git", "commit_id": "deadbeef"},
        }
    )

    assert provenance._sanitise_direct_url(raw) == {
        "url": "https://example.test/research.git",
        "vcs_info": {"vcs": "git", "commit_id": "deadbeef"},
    }


def test_write_environment_lock_is_stable_and_returns_content_hash(
    monkeypatch, tmp_path
):
    lock = {
        "schema_version": 1,
        "conda_packages": [{"name": "numpy", "version": "2.4.6"}],
    }
    monkeypatch.setattr(provenance, "environment_lock", lambda: lock)

    path, digest = provenance.write_environment_lock(tmp_path)
    payload = path.read_bytes()

    assert path.name == "environment-lock.json"
    assert json.loads(payload) == lock
    assert digest == hashlib.sha256(payload).hexdigest()
