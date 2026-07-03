# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configurable output-root path resolver (issue #180).

The output *layout* is fixed; only its **root** is configurable, so ephemeral
scratch-disk VM runs can redirect large artefacts (statistical-model ``trace.nc``
files, rendered reports) off the repo disk without breaking the established
relative layout, report rendering, uploads, comparisons, or scripts that read
previous runs.

Precedence for the output root:

1. an explicit CLI ``--output-dir`` for the process (via :func:`set_output_root`);
2. the ``DSE_LRP_OUTPUT_DIR`` environment variable;
3. the repo-local ``<repo>/output`` default.

Paths are exposed as **functions** rather than import-time constants so that a
CLI override applied after import (the common case: a script parses its args,
then builds the pipeline) is still honoured. ``ROOT_DIR`` / ``DATA_DIR`` /
``DOCS_DIR`` are fixed repository locations and stay constants.
"""

from __future__ import annotations

import os
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent  # <repo>/src
ROOT_DIR = _SRC_DIR.parent  # <repo>
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"

OUTPUT_ROOT_ENV_VAR = "DSE_LRP_OUTPUT_DIR"

_cli_override: Path | None = None


def set_output_root(path: str | os.PathLike[str] | None) -> Path:
    """Set (or clear) the process-wide CLI output-root override — highest precedence.

    Pass the parsed ``--output-dir`` value, or ``None`` to clear it. Returns the
    resolved output root. Call once, early in a command, before the pipeline
    constructs any run directory.
    """
    global _cli_override
    _cli_override = Path(path).expanduser().resolve() if path else None
    return output_root()


def output_root() -> Path:
    """Resolve the output root: CLI override, then ``DSE_LRP_OUTPUT_DIR``, then default."""
    if _cli_override is not None:
        return _cli_override
    env = os.environ.get(OUTPUT_ROOT_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return ROOT_DIR / "output"


# Layout below the (configurable) root — unchanged relative structure.
def output_dir() -> Path:
    return output_root()


def gb_models_dir() -> Path:
    return output_root() / "models"


def gb_tuning_dir() -> Path:
    return output_root() / "tuning"


def stat_dir() -> Path:
    return output_root() / "statistical_models"


def stat_models_dir() -> Path:
    return stat_dir() / "models"


def stat_comparison_dir() -> Path:
    return stat_dir() / "comparison"


def is_overridden() -> bool:
    """True when the resolved root differs from the repo-local ``<repo>/output``."""
    return output_root() != (ROOT_DIR / "output")


def describe_output_root() -> str:
    """One-line description of the resolved root and its source, for run logs."""
    if _cli_override is not None:
        source = "--output-dir"
    elif os.environ.get(OUTPUT_ROOT_ENV_VAR):
        source = OUTPUT_ROOT_ENV_VAR
    else:
        source = "repo-local default"
    return f"{output_root()}  (source: {source})"
