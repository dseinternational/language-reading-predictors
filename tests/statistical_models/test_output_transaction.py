# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for atomic statistical-model output publication."""

from __future__ import annotations

import os
import stat
import time

import pytest

from language_reading_predictors.statistical_models.output_transaction import (
    _STALE_PRIVATE_PATH_AGE_SECONDS,
    OutputTransaction,
)


def test_failed_staging_preserves_the_previous_publication(tmp_path):
    final = tmp_path / "model-reporting"
    final.mkdir()
    (final / "config.json").write_text("old", encoding="utf-8")
    transaction = OutputTransaction.create(final)
    (transaction.staging_dir / "config.json").write_text("partial", encoding="utf-8")

    transaction.abandon()

    assert (final / "config.json").read_text(encoding="utf-8") == "old"
    assert not transaction.staging_dir.exists()


def test_successful_publish_replaces_the_directory_without_stale_files(tmp_path):
    final = tmp_path / "model-reporting"
    final.mkdir()
    (final / "stale.csv").write_text("old", encoding="utf-8")
    transaction = OutputTransaction.create(final)
    (transaction.staging_dir / "config.json").write_text("new", encoding="utf-8")

    published = transaction.publish()

    assert published == final
    assert transaction.output_dir == final
    assert (final / "config.json").read_text(encoding="utf-8") == "new"
    assert not (final / "stale.csv").exists()
    assert not transaction.staging_dir.exists()


def test_failed_promotion_restores_the_previous_publication(monkeypatch, tmp_path):
    final = tmp_path / "model-reporting"
    final.mkdir()
    (final / "config.json").write_text("old", encoding="utf-8")
    transaction = OutputTransaction.create(final)
    (transaction.staging_dir / "config.json").write_text("new", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("promotion failed")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_replace)

    with pytest.raises(OSError, match="promotion failed"):
        transaction.publish()

    assert (final / "config.json").read_text(encoding="utf-8") == "old"
    transaction.abandon()


def test_create_removes_only_week_old_private_siblings(tmp_path):
    final = tmp_path / "model-reporting"
    final.mkdir()
    stale_staging = tmp_path / ".model-reporting.staging-abandoned"
    stale_backup = tmp_path / ".model-reporting.backup-redundant"
    recent_staging = tmp_path / ".model-reporting.staging-active"
    unrelated = tmp_path / ".another-model.staging-abandoned"
    for path in (stale_staging, stale_backup, recent_staging, unrelated):
        path.mkdir()
    stale_time = time.time() - _STALE_PRIVATE_PATH_AGE_SECONDS - 60
    for path in (stale_staging, stale_backup, unrelated):
        os.utime(path, (stale_time, stale_time))

    transaction = OutputTransaction.create(final)

    assert not stale_staging.exists()
    assert not stale_backup.exists()
    assert recent_staging.exists()
    assert unrelated.exists()
    transaction.abandon()


def test_create_preserves_a_stale_backup_when_final_is_missing(tmp_path):
    final = tmp_path / "model-reporting"
    backup = tmp_path / ".model-reporting.backup-recoverable"
    backup.mkdir()
    stale_time = time.time() - _STALE_PRIVATE_PATH_AGE_SECONDS - 60
    os.utime(backup, (stale_time, stale_time))

    transaction = OutputTransaction.create(final)

    assert backup.exists()
    transaction.abandon()


def test_staging_and_published_directory_use_the_normal_umask_mode(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    expected_mode = stat.S_IMODE(control.stat().st_mode)
    final = tmp_path / "model-reporting"

    transaction = OutputTransaction.create(final)

    assert stat.S_IMODE(transaction.staging_dir.stat().st_mode) == expected_mode
    transaction.publish()
    assert stat.S_IMODE(final.stat().st_mode) == expected_mode
