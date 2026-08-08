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
import importlib
import pathlib

import pytest

from language_reading_predictors.statistical_models import pipeline

PACKAGE = pathlib.Path(pipeline.__file__).parent
MONOLITH = "language_reading_predictors.statistical_models.pipeline"

# The shared layer ``pipelines/*`` is built on: artefact production, presentation,
# the stage binding, the samplers and the posterior summaries. Each was carved out
# of ``pipeline.py`` (or, for ``diagnostics`` and ``reporting``, predates it) and
# must stay below it.
SHARED_MODULES = (
    "adjustment",
    "diagnostics",
    "figure_artifacts",
    "ppc_artifacts",
    "prior_artifacts",
    "publication",
    "reporting",
    "runtime",
)

# Every family that has moved out of the monolith, and the entry points
# ``pipeline.py`` must keep re-exporting for it. Most families have exactly one.
# Mediation is the outlier, with three fit functions and a data-preparation helper
# that ``scripts/regenerate_mediation_calibration.py`` imports by name; adjusted
# and horseshoe each carry a second entry point for the Byrne (RLM) cohort, which
# ``definitions.KINDS`` keys as the same family.
MIGRATED_FAMILIES: dict[str, tuple[str, ...]] = {
    "adjusted": ("fit_adjusted", "fit_rlm_adjusted"),
    "aligned": ("fit_aligned",),
    "block_exposure": ("fit_block_exposure",),
    "concurrent": ("fit_concurrent",),
    "did": ("fit_did",),
    "dose_response": ("fit_dose_response",),
    "gain_factors": ("fit_gain_factors",),
    "horseshoe": ("fit_horseshoe", "fit_rlm_horseshoe"),
    "itt": ("fit_itt",),
    "joint": ("fit_joint",),
    "joint_mechanism": ("fit_joint_mechanism",),
    "level_factors": ("fit_level_factors",),
    "mechanism": ("fit_mechanism",),
    "mediation": (
        "fit_mediation",
        "fit_mediation_multi",
        "fit_mediation_period_stacked",
        "prepare_mediation_data",
    ),
}

MIGRATED_ENTRY_POINTS = sorted(
    (family, entry) for family, entries in MIGRATED_FAMILIES.items() for entry in entries
)


def _package_of(path: pathlib.Path) -> str:
    """The dotted package a source file belongs to, from its directory chain."""
    parts: list[str] = []
    directory = path.parent
    while (directory / "__init__.py").exists():
        parts.append(directory.name)
        directory = directory.parent
    return ".".join(reversed(parts))


def _imported_modules(source: str, package: str) -> set[str]:
    """Every module name ``source`` imports, as absolute dotted names.

    Relative imports are resolved against ``package``: ``from . import pipeline``
    carries no module name at all, and ``from ..pipeline import fit_itt`` carries
    the bare ``pipeline``, so comparing either against an absolute name without
    resolving it would let a back-edge through the guard below. Both
    module-level and function-local imports are collected — the moved figure
    wrappers deliberately import inside their functions.
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                owner = package.split(".")
                prefix = ".".join(owner[: len(owner) - node.level + 1])
                base = f"{prefix}.{base}" if base else prefix
            if not base:
                continue
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _imports_of(path: pathlib.Path) -> set[str]:
    return _imported_modules(path.read_text(encoding="utf-8"), _package_of(path))


SM = "language_reading_predictors.statistical_models"
BACK_EDGES = [
    (SM, f"from {SM} import pipeline"),
    (SM, f"from {SM}.pipeline import fit_itt"),
    (SM, f"import {SM}.pipeline"),
    (SM, "from . import pipeline"),
    (SM, "from .pipeline import fit_itt"),
    (f"{SM}.pipelines", "from .. import pipeline"),
    (f"{SM}.pipelines", "from ..pipeline import fit_itt"),
    (f"{SM}.pipelines", "def fit():\n    from .. import pipeline\n"),
]


@pytest.mark.parametrize("package,source", BACK_EDGES)
def test_the_guard_detects_every_spelling_of_a_back_edge(package, source):
    """The guard is only worth having if no spelling slips past it."""
    assert MONOLITH in _imported_modules(source, package)


@pytest.mark.parametrize(
    "package,source",
    [
        (SM, "from . import runtime"),
        (SM, f"from {SM}.runtime import run_ppc"),
        (f"{SM}.pipelines", "from .itt import write_analysis_audit"),
        (f"{SM}.pipelines", "from ..runtime import require_spec"),
    ],
)
def test_the_guard_does_not_flag_permitted_edges(package, source):
    assert MONOLITH not in _imported_modules(source, package)


def test_shared_artifact_modules_do_not_import_the_monolith():
    for name in SHARED_MODULES:
        path = PACKAGE / f"{name}.py"
        assert path.exists(), f"{name}.py is missing"
        assert MONOLITH not in _imports_of(path), (
            f"{name}.py imports pipeline.py; the shared layer must stay below it"
        )


def test_family_pipelines_do_not_import_the_monolith():
    modules = sorted((PACKAGE / "pipelines").glob("*.py"))
    present = {p.stem for p in modules} - {"__init__"}
    assert present == set(MIGRATED_FAMILIES), (
        "pipelines/ and MIGRATED_FAMILIES disagree; update the guard when a "
        f"family moves. Only in package: {sorted(present - set(MIGRATED_FAMILIES))}; "
        f"only in guard: {sorted(set(MIGRATED_FAMILIES) - present)}"
    )
    for path in modules:
        assert MONOLITH not in _imports_of(path), (
            f"pipelines/{path.name} imports pipeline.py; family modules must not"
        )


@pytest.mark.parametrize("family,entry", MIGRATED_ENTRY_POINTS)
def test_pipeline_re_exports_the_migrated_family_entry_points(family, entry):
    """``pipeline.py`` stays a working facade until every caller has migrated."""
    module = importlib.import_module(
        f"language_reading_predictors.statistical_models.pipelines.{family}"
    )
    assert getattr(pipeline, entry) is getattr(module, entry)


def test_migrated_families_are_no_longer_defined_in_the_monolith():
    source = pathlib.Path(pipeline.__file__).read_text(encoding="utf-8")
    entries = [f"def {entry}(" for _, entry in MIGRATED_ENTRY_POINTS]
    for entry in [*entries, "def fit_itt_floor_rule("]:
        assert entry not in source, f"{entry!r} is back in pipeline.py"
