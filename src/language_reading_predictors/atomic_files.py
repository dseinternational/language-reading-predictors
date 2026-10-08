# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Atomic file replacement with a caller-selected permission policy.

The shared helper creates, replaces and cleans up temporary files. Callers
provide the serialisation callback. The default keeps private temporary-file
permissions; ``process_default`` uses the mode of a newly opened file. This
replaces one file at a time and provides no locking or multi-file transaction.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from dse_research_utils.storage.files import atomic_write
from dse_research_utils.storage.files import default_file_mode as _shared_default_file_mode

ModePolicy = Literal["private", "process_default"]
"""``private`` keeps ``mkstemp``'s 0600; ``process_default`` restores the umask mode."""

_FALLBACK_FILE_MODE = 0o644


def process_default_file_mode(directory: Path) -> int:
    """The mode a plain ``open(..., "w")`` would leave in ``directory``.

    The shared probe avoids changing the process-wide umask. Failures retain
    this project's ``0644`` fallback, the usual mode under a ``022`` umask.
    """
    try:
        return _shared_default_file_mode(directory)
    except OSError:
        return _FALLBACK_FILE_MODE


def write_atomic(
    path: str | os.PathLike[str],
    write_temporary: Callable[[Path], object],
    *,
    mode: ModePolicy = "private",
) -> None:
    """Replace ``path`` with what ``write_temporary`` writes, in one rename.

    ``mode="process_default"`` chmods the temporary file to the mode a plain
    ``open`` would have produced. Read access depends on the process's umask.
    """
    if mode not in ("private", "process_default"):
        raise ValueError("mode must be 'private' or 'process_default'")

    def write(temporary: Path) -> None:
        if mode == "process_default":
            temporary.chmod(process_default_file_mode(temporary.parent))
        write_temporary(temporary)

    atomic_write(path, write)
