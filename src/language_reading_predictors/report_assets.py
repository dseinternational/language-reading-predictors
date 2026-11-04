# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Copy the shared HTML theme and project rules beside generated reports."""

from __future__ import annotations

import shutil
from pathlib import Path


def copy_report_theme(output_dir: str | Path, *, docs_dir: str | Path) -> None:
    """Keep copied model reports independent of the source-tree theme paths."""
    destination = Path(output_dir) / "theme"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("dse.scss", "project.scss"):
        shutil.copyfile(Path(docs_dir) / "theme" / filename, destination / filename)
