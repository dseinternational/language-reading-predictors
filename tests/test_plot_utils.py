# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`plot_utils`.

Covers ``OUTPUT_DIR`` (regression test for a bug where it resolved to
``src/output`` instead of the repo-root ``output/`` that every other module
writes to — ``save_figure`` / ``display_image`` would have silently written
figures to the wrong place).
"""

from __future__ import annotations

from language_reading_predictors.plot_utils import OUTPUT_DIR


def test_output_dir_is_repo_root_output():
    assert OUTPUT_DIR.name == "output"
    # The repo root is identified by pyproject.toml living next to output/,
    # not by a hard-coded absolute path (robust to the checkout location).
    assert (OUTPUT_DIR.parent / "pyproject.toml").exists()
    # Regression guard: OUTPUT_DIR must not be nested under src/.
    assert "src" not in OUTPUT_DIR.parts
