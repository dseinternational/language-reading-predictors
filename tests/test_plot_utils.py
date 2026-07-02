# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`plot_utils` output-root resolution.

Regression guard for the historical bug where the plot output dir resolved to
``src/output`` instead of the repo-root ``output/`` that every other module
writes to. Since #180 the root is resolved via ``paths.output_root()`` **at call
time** (env / ``--output-dir`` aware) rather than a module constant, so this
checks the default root that ``save_figure`` / ``display_image`` pass through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from language_reading_predictors import paths


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv(paths.OUTPUT_ROOT_ENV_VAR, raising=False)
    paths.set_output_root(None)
    yield
    paths.set_output_root(None)


def test_plot_helpers_default_to_repo_root_output(monkeypatch):
    import language_reading_predictors.plot_utils as pu

    seen: dict[str, str] = {}
    monkeypatch.setattr(
        pu, "_shared_save_figure", lambda fn, root, **kw: seen.update(root=root)
    )
    pu.save_figure("fig.png")

    root = Path(seen["root"])
    assert root.name == "output"
    # The repo root is identified by pyproject.toml living next to output/, not a
    # hard-coded absolute path (robust to the checkout location).
    assert (root.parent / "pyproject.toml").exists()
    # Regression guard: the output root must not be nested under src/.
    assert "src" not in root.parts
