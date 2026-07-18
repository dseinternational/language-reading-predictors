# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Portable provenance and failure records for statistical-model runs."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


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
"""Import name to installed-distribution name for the numerical core."""


def _utc_now() -> datetime:
    """Return an aware UTC timestamp behind a small, testable boundary."""
    return datetime.now(UTC)


def package_versions(
    distributions: Mapping[str, str] = CORE_DISTRIBUTIONS,
) -> dict[str, str | None]:
    """Return installed versions, retaining ``None`` when metadata is unavailable."""
    versions: dict[str, str | None] = {}
    for name, distribution in distributions.items():
        try:
            versions[name] = metadata.version(distribution)
        except (metadata.PackageNotFoundError, OSError, ValueError):
            versions[name] = None
    return versions


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
    """Describe the source checkout without making Git a runtime requirement."""
    working_directory = Path.cwd() if cwd is None else Path(cwd)
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
        f"{_safe_model_id(model_id)}-{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
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
