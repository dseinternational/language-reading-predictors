# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Atomic single-file replacement, delegated to the shared library (#662).

``dse_research_utils.storage.files.atomic_write`` owns the temporary-file
creation, the single ``os.replace`` and the failure cleanup. This module keeps
the two things the library deliberately leaves to the consumer:

* **Serialisation.** Every caller still writes its own bytes in the callback, so
  CSV/JSON encoding, index handling and trailing newlines are unchanged.
* **Permissions.** The shared helper creates its temporary file with
  ``mkstemp``'s owner-only ``0600`` and the destination inherits that. Sites that
  previously created their temporary file with a plain ``open`` published a file
  at the process's umask-derived mode (usually ``0644``), and those artefacts are
  read back by report rendering and by the upload script. :func:`write_atomic`
  therefore takes an explicit ``mode`` policy rather than silently narrowing
  them.

The library's contract — visible old-or-new file, no locking, no bundle
transaction — is unchanged; see the 0.14.0 file-and-provenance guide.
"""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from dse_research_utils.storage.files import atomic_write

ModePolicy = Literal["private", "process_default"]
"""``private`` keeps ``mkstemp``'s 0600; ``process_default`` restores the umask mode."""

_FALLBACK_FILE_MODE = 0o644


def process_default_file_mode(directory: Path) -> int:
    """The mode a plain ``open(..., "w")`` would leave in ``directory``.

    Probed rather than derived from ``os.umask``: reading the umask means
    setting it, which is process-wide and not thread-safe. A probe also
    reflects a parent directory's default ACL where one applies. Failures fall
    back to ``0644``, which is what an unset-umask process would produce.
    """
    probe = directory / f".tmp-mode-probe-{uuid.uuid4().hex}"
    try:
        probe.touch()
        try:
            return stat.S_IMODE(probe.stat().st_mode)
        finally:
            probe.unlink(missing_ok=True)
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
    ``open`` would have produced, so a caller that previously wrote through an
    ordinary ``open`` keeps publishing a world-readable artefact.
    """
    if mode not in ("private", "process_default"):
        raise ValueError("mode must be 'private' or 'process_default'")

    def write(temporary: Path) -> None:
        if mode == "process_default":
            temporary.chmod(process_default_file_mode(temporary.parent))
        write_temporary(temporary)

    atomic_write(path, write)
