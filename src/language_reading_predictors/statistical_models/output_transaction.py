# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Transactional publication of one statistical-model output directory.

Fits write every artefact to a hidden sibling directory.  The completed directory
is promoted only after the final report files exist, so a failed refit cannot mix
new partial output with the last successful publication.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path


_STALE_PRIVATE_PATH_AGE_SECONDS = 7 * 24 * 60 * 60


def _remove_private_path(path: Path) -> None:
    """Best-effort cleanup used for staging directories and private backups."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def _remove_stale_private_paths(final_dir: Path, *, now: float | None = None) -> None:
    """Remove week-old staging data and redundant backups for one publication.

    Recent paths may belong to a concurrent fit and are left untouched. Backups are
    removed only when the final publication exists; if it is absent, a backup may
    be the only recoverable successful fit and is therefore preserved.
    """
    current_time = time.time() if now is None else now
    prefixes = [f".{final_dir.name}.staging-"]
    if final_dir.exists():
        prefixes.append(f".{final_dir.name}.backup-")
    try:
        siblings = tuple(final_dir.parent.iterdir())
    except OSError:
        return
    for candidate in siblings:
        if not any(candidate.name.startswith(prefix) for prefix in prefixes):
            continue
        try:
            age_seconds = current_time - candidate.stat().st_mtime
        except OSError:
            continue
        if age_seconds >= _STALE_PRIVATE_PATH_AGE_SECONDS:
            _remove_private_path(candidate)


def _apply_default_directory_mode(directory: Path) -> None:
    """Replace ``mkdtemp``'s 0700 mode with the process's normal directory mode."""
    probe = directory / ".mode-probe"
    probe.mkdir()
    try:
        mode = stat.S_IMODE(probe.stat().st_mode)
    finally:
        probe.rmdir()
    directory.chmod(mode)


@dataclass(slots=True, weakref_slot=True)
class OutputTransaction:
    """A staging directory and the final directory it will replace.

    ``publish`` uses same-filesystem renames.  When a previous publication exists,
    it is first moved to a private backup; if promotion fails, that backup is moved
    back.  The staging directory is never exposed at the final path, and only a
    complete directory is promoted.
    """

    final_dir: Path
    staging_dir: Path
    _published: bool = False
    _finalizer: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.final_dir = self.final_dir.resolve()
        self.staging_dir = self.staging_dir.resolve()
        if self.final_dir.parent != self.staging_dir.parent:
            raise ValueError("staging and final output directories must be siblings")
        self._finalizer = weakref.finalize(
            self, _remove_private_path, self.staging_dir
        )

    @classmethod
    def create(cls, final_dir: str | Path) -> OutputTransaction:
        """Create a hidden same-filesystem staging directory for ``final_dir``."""
        final = Path(final_dir).resolve()
        final.parent.mkdir(parents=True, exist_ok=True)
        _remove_stale_private_paths(final)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{final.name}.staging-",
                dir=final.parent,
            )
        )
        try:
            _apply_default_directory_mode(staging)
        except BaseException:
            _remove_private_path(staging)
            raise
        return cls(final_dir=final, staging_dir=staging)

    @property
    def published(self) -> bool:
        return self._published

    @property
    def output_dir(self) -> Path:
        """Return the writable staging path, or the final path after publication."""
        return self.final_dir if self._published else self.staging_dir

    def publish(self) -> Path:
        """Promote the completed staging directory and return its final path."""
        if self._published:
            return self.final_dir
        if not self.staging_dir.is_dir():
            raise FileNotFoundError(
                f"staging output directory does not exist: {self.staging_dir}"
            )

        backup: Path | None = None
        if self.final_dir.exists():
            backup = self.final_dir.parent / (
                f".{self.final_dir.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(self.final_dir, backup)

        try:
            os.replace(self.staging_dir, self.final_dir)
        except BaseException:
            if backup is not None and backup.exists() and not self.final_dir.exists():
                os.replace(backup, self.final_dir)
            raise

        self._published = True
        self._finalizer.detach()
        if backup is not None:
            _remove_private_path(backup)
        return self.final_dir

    def abandon(self) -> None:
        """Discard an unpublished staging directory without touching published data."""
        if self._published:
            return
        self._finalizer()
