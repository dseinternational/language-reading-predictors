# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Portable provenance and failure records for statistical-model runs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


CORE_DISTRIBUTIONS: dict[str, str] = {
    "arviz": "arviz",
    "dse_research_utils": "dse-research-utils",
    "numpy": "numpy",
    "nutpie": "nutpie",
    "pandas": "pandas",
    "pymc": "pymc",
    "pytensor": "pytensor",
    "scipy": "scipy",
    "xarray": "xarray",
}
# Import name to installed-distribution name for the numerical core.


def _utc_now() -> datetime:
    """Return an aware UTC timestamp behind a small, testable boundary."""
    return datetime.now(UTC)


@lru_cache(maxsize=16)
def _cached_package_versions(
    distributions: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str | None], ...]:
    """Resolve one distribution set once for the lifetime of the process."""
    versions: list[tuple[str, str | None]] = []
    for name, distribution in distributions:
        try:
            version = metadata.version(distribution)
        except (metadata.PackageNotFoundError, OSError, ValueError):
            version = None
        versions.append((name, version))
    return tuple(versions)


def package_versions(
    distributions: Mapping[str, str] = CORE_DISTRIBUTIONS,
) -> dict[str, str | None]:
    """Return installed versions, retaining ``None`` when metadata is unavailable."""
    return dict(_cached_package_versions(tuple(distributions.items())))


def _git_output(arguments: list[str], *, cwd: Path) -> str | None:
    """Run one read-only Git query, returning ``None`` outside a usable checkout."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def source_provenance(cwd: str | Path | None = None) -> dict[str, Any]:
    """Describe the package checkout without making Git a runtime requirement."""
    working_directory = Path(__file__).resolve().parent if cwd is None else Path(cwd)
    root_text = _git_output(["rev-parse", "--show-toplevel"], cwd=working_directory)
    if root_text is None:
        return {
            "repository_root": None,
            "commit": None,
            "branch": None,
            "dirty": None,
        }

    repository_root = Path(root_text)
    commit = _git_output(["rev-parse", "HEAD"], cwd=repository_root)
    branch = _git_output(["branch", "--show-current"], cwd=repository_root)
    status = _git_output(["status", "--porcelain"], cwd=repository_root)
    return {
        "repository_root": str(repository_root),
        "commit": commit,
        "branch": branch or None,
        "dirty": None if status is None else bool(status),
    }


def run_provenance() -> dict[str, Any]:
    """Collect enough execution context to interpret a copied model directory."""
    return {
        "recorded_at_utc": _utc_now().isoformat(),
        "invocation": {
            "argv": list(sys.argv),
            "working_directory": str(Path.cwd()),
        },
        "source": source_provenance(),
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": package_versions(),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _conda_lock_records(prefix: Path) -> list[dict[str, Any]]:
    """Read exact conda package builds without requiring the conda executable."""
    records: list[dict[str, Any]] = []
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        record = {
            key: raw.get(key)
            for key in (
                "name",
                "version",
                "build",
                "build_number",
                "channel",
                "subdir",
                "fn",
                "sha256",
                "md5",
            )
            if raw.get(key) is not None
        }
        if record.get("name") and record.get("version"):
            records.append(record)
    return sorted(
        records,
        key=lambda item: (
            str(item.get("name", "")).casefold(),
            str(item.get("version", "")),
            str(item.get("build", "")),
        ),
    )


def _sanitise_direct_url(raw_text: str | None) -> dict[str, Any] | None:
    """Retain reproducible direct-install metadata without credentials or queries."""
    if not raw_text:
        return None
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    url = raw.get("url")
    if not isinstance(url, str):
        return None
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    clean: dict[str, Any] = {
        "url": urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    }
    for key in ("vcs_info", "dir_info", "archive_info"):
        value = raw.get(key)
        if isinstance(value, dict):
            clean[key] = value
    return clean


def _python_distribution_lock_records() -> list[dict[str, Any]]:
    """Return every installed Python distribution and any exact direct source."""
    records: list[dict[str, Any]] = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        record: dict[str, Any] = {
            "name": name,
            "version": distribution.version,
        }
        try:
            direct_url = _sanitise_direct_url(
                distribution.read_text("direct_url.json")
            )
        except (OSError, UnicodeError, ValueError):
            direct_url = None
        if direct_url is not None:
            record["direct_url"] = direct_url
        records.append(record)
    return sorted(
        records,
        key=lambda item: (
            str(item["name"]).casefold(),
            str(item["version"]),
        ),
    )


@lru_cache(maxsize=1)
def environment_lock() -> dict[str, Any]:
    """Snapshot the exact installed conda builds and Python distributions.

    This complements ``environment.yml``: the YAML declares compatible ranges,
    while this record identifies the concrete environment that produced a fit.
    """
    prefix = Path(sys.prefix).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    spec_path = repository_root / "environment.yml"
    project_spec = None
    if spec_path.is_file():
        project_spec = {
            "path": "environment.yml",
            "sha256": _file_sha256(spec_path),
        }
    return {
        "schema_version": 1,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "prefix": str(prefix),
        },
        "project_environment_spec": project_spec,
        "conda_packages": _conda_lock_records(prefix),
        "python_distributions": _python_distribution_lock_records(),
    }


def write_environment_lock(output_dir: str | Path) -> tuple[Path, str]:
    """Write the environment snapshot beside a fit and return path plus SHA-256."""
    path = Path(output_dir) / "environment-lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(environment_lock(), indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_model_id(model_id: str) -> str:
    """Limit diagnostic filenames to predictable, portable characters."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in model_id)


def write_failure_record(
    output_root: str | Path,
    *,
    model_id: str,
    config: str,
    error: BaseException,
    traceback_text: str,
) -> Path:
    """Persist a structured traceback and execution context after a failed fit."""
    timestamp = _utc_now()
    failure_dir = Path(output_root) / "run_metadata" / "failures"
    failure_dir.mkdir(parents=True, exist_ok=True)
    path = failure_dir / (
        f"{_safe_model_id(model_id)}-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"
        f"-{uuid.uuid4().hex[:12]}.json"
    )
    record = {
        "recorded_at_utc": timestamp.isoformat(),
        "model_id": model_id,
        "config": config,
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "traceback": traceback_text,
        "provenance": run_provenance(),
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
