# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the configurable output-root resolver (issue #180)."""

from __future__ import annotations

import pytest

from language_reading_predictors import paths


@pytest.fixture(autouse=True)
def _isolate_paths(monkeypatch):
    """Reset the process-global env var + CLI override around each test."""
    monkeypatch.delenv(paths.OUTPUT_ROOT_ENV_VAR, raising=False)
    paths.set_output_root(None)
    yield
    paths.set_output_root(None)


def test_default_output_root_is_repo_local():
    assert paths.output_root() == paths.ROOT_DIR / "output"
    assert not paths.is_overridden()
    assert "repo-local default" in paths.describe_output_root()


def test_layout_is_relative_to_root():
    root = paths.output_root()
    assert paths.gb_models_dir() == root / "models"
    assert paths.gb_tuning_dir() == root / "tuning"
    assert paths.stat_dir() == root / "statistical_models"
    assert paths.stat_models_dir() == root / "statistical_models" / "models"
    assert paths.stat_comparison_dir() == root / "statistical_models" / "comparison"


def test_env_var_redirects_root(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.OUTPUT_ROOT_ENV_VAR, str(tmp_path))
    expected = tmp_path.resolve()
    assert paths.output_root() == expected
    # The relative layout below the root is unchanged.
    assert paths.gb_models_dir() == expected / "models"
    assert paths.gb_tuning_dir() == expected / "tuning"
    assert paths.stat_models_dir() == expected / "statistical_models" / "models"
    assert paths.stat_comparison_dir() == expected / "statistical_models" / "comparison"
    assert paths.is_overridden()
    assert paths.OUTPUT_ROOT_ENV_VAR in paths.describe_output_root()


def test_cli_override_beats_env(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.OUTPUT_ROOT_ENV_VAR, str(tmp_path / "from_env"))
    cli = tmp_path / "from_cli"
    assert paths.set_output_root(cli) == cli.resolve()
    assert paths.output_root() == cli.resolve()
    assert "--output-dir" in paths.describe_output_root()
    # Clearing the CLI override falls back to the env var.
    paths.set_output_root(None)
    assert paths.output_root() == (tmp_path / "from_env").resolve()


def test_fixed_repo_locations_are_absolute():
    for p in (paths.ROOT_DIR, paths.DATA_DIR, paths.DOCS_DIR):
        assert p.is_absolute()
    assert paths.DATA_DIR == paths.ROOT_DIR / "data"
    assert paths.DOCS_DIR == paths.ROOT_DIR / "docs"


def test_plot_helpers_resolve_output_root_at_call_time(monkeypatch, tmp_path):
    # save_figure / display_image must resolve the root at call time (not cache it
    # at import), so a later env var / --output-dir is honoured (#180 review).
    import language_reading_predictors.plot_utils as pu

    seen: dict[str, str] = {}
    monkeypatch.setattr(pu, "_shared_save_figure", lambda fn, root, **kw: seen.update(save=root))
    monkeypatch.setattr(pu, "_shared_display_image", lambda fn, root, **kw: seen.update(disp=root))

    monkeypatch.setenv(paths.OUTPUT_ROOT_ENV_VAR, str(tmp_path))
    pu.save_figure("f.png")
    pu.display_image("f.png")
    assert seen["save"] == str(tmp_path.resolve())
    assert seen["disp"] == str(tmp_path.resolve())

    # An explicit output_dir overrides the resolved root.
    pu.save_figure("f.png", output_dir="/data/explicit")
    assert seen["save"] == "/data/explicit"
