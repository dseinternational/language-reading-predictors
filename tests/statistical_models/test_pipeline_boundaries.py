# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Structural guards for the family-split boundaries (#394 step 5).

The split only stays possible while the dependency edges point one way: the
shared artefact and presentation modules, and the family orchestration modules
under ``pipelines/``, must not reach back into ``pipeline.py``. A back-edge would
reintroduce the import cycle that kept every family inside the monolith, and it
would do so silently — nothing else in the suite would fail. These tests make it
fail here instead.
"""

from __future__ import annotations

import ast
import pathlib

from language_reading_predictors.statistical_models import pipeline
from language_reading_predictors.statistical_models.pipelines import (
    itt as itt_pipeline,
    joint as joint_pipeline,
)

PACKAGE = pathlib.Path(pipeline.__file__).parent
MONOLITH = "language_reading_predictors.statistical_models.pipeline"

# The shared layer ``pipelines/*`` is built on: artefact production, presentation
# and the stage binding. Each was carved out of ``pipeline.py`` and must stay
# below it.
SHARED_MODULES = (
    "figure_artifacts",
    "ppc_artifacts",
    "prior_artifacts",
    "publication",
    "runtime",
)


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Every module name this file imports, module-level or function-local."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_shared_artifact_modules_do_not_import_the_monolith():
    for name in SHARED_MODULES:
        path = PACKAGE / f"{name}.py"
        assert path.exists(), f"{name}.py is missing"
        assert MONOLITH not in _imported_modules(path), (
            f"{name}.py imports pipeline.py; the shared layer must stay below it"
        )


def test_family_pipelines_do_not_import_the_monolith():
    modules = sorted((PACKAGE / "pipelines").glob("*.py"))
    assert len(modules) >= 3, "expected __init__ plus the migrated family modules"
    for path in modules:
        assert MONOLITH not in _imported_modules(path), (
            f"pipelines/{path.name} imports pipeline.py; family modules must not"
        )


def test_pipeline_re_exports_the_migrated_family_entry_points():
    """``pipeline.py`` stays a working facade until every caller has migrated."""
    assert pipeline.fit_itt is itt_pipeline.fit_itt
    assert pipeline.fit_joint is joint_pipeline.fit_joint


def test_migrated_families_are_no_longer_defined_in_the_monolith():
    source = pathlib.Path(pipeline.__file__).read_text(encoding="utf-8")
    for entry in ("def fit_itt(", "def fit_joint(", "def fit_itt_floor_rule("):
        assert entry not in source, f"{entry!r} is back in pipeline.py"
