# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for atomic statistical-model output publication."""

from __future__ import annotations

import os

import pytest

from language_reading_predictors.statistical_models.output_transaction import (
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
