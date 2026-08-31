# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The shared layer's module boundaries, after the #637 stage 3 split.

``reporting.py`` was 9,417 lines and one of the three dependency hubs the
maintainability review named. It is now four modules — estimands, predictive
checks, run metadata, key findings — plus the convergence gate, with ``reporting``
kept as a **temporary** re-export facade so existing call sites keep working.

Two concrete import cycles came with those hubs and are closed here: ``factories``
imported level-factor policy while level-factor code reached back into
``factories`` for a private helper, and ``reporting`` and ``release`` imported one
another through function-local imports written to hide the fact.

These tests pin the shape, not the contents: which module owns what, that the
facade is complete, and that no module-level cycle returns.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from language_reading_predictors.statistical_models import definitions

PACKAGE = pathlib.Path(definitions.__file__).parent
SM = "language_reading_predictors.statistical_models"

#: The four responsibility modules ``reporting.py`` was split into, plus the gate.
SPLIT_MODULES = (
    "estimands",
    "predictive_checks",
    "run_metadata",
    "key_findings",
    "convergence",
)


def _module_imports(path: pathlib.Path) -> set[str]:
    """Sibling modules imported at **module level** — a local import is not a cycle."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(SM):
            imported.add(node.module[len(SM) + 1 :] or "__init__")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(SM):
                    imported.add(alias.name[len(SM) + 1 :] or "__init__")
    return imported


def _edges() -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.stem.startswith("lrp_"):
            continue
        name = path.stem if path.parent == PACKAGE else f"pipelines.{path.stem}"
        edges[name] = _module_imports(path)
    return edges


def _cycles(edges: dict[str, set[str]]) -> list[list[str]]:
    found: list[list[str]] = []
    done: set[str] = set()

    def walk(node: str, stack: list[str]) -> None:
        if node in stack:
            found.append(stack[stack.index(node) :] + [node])
            return
        if node in done:
            return
        stack.append(node)
        for nxt in sorted(edges.get(node, ())):
            walk(nxt, stack)
        stack.pop()
        done.add(node)

    for start in sorted(edges):
        walk(start, [])
    return found


def test_the_package_has_no_module_level_import_cycle():
    """Both named cycles are closed, and no new one may appear.

    A function-local import that exists only to hide a cycle is the smell this
    replaces: it makes the edge invisible to every tool and to the reader.
    """
    cycles = _cycles(_edges())
    assert cycles == [], [" -> ".join(cycle) for cycle in cycles]


def test_factories_no_longer_imports_level_factor_policy():
    assert "level_factors" not in _module_imports(PACKAGE / "factories.py")


def test_reporting_and_release_no_longer_import_each_other():
    assert "release" not in _module_imports(PACKAGE / "reporting.py")
    assert "reporting" not in _module_imports(PACKAGE / "release.py")
    # ``release`` reaches the gate at its owner, not through a hub.
    assert "convergence" in _module_imports(PACKAGE / "release.py")


@pytest.mark.parametrize("module", SPLIT_MODULES)
def test_each_split_module_exists_and_is_smaller_than_the_hub_it_left(module):
    path = PACKAGE / f"{module}.py"
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").splitlines()) < 4500


def test_the_reporting_facade_re_exports_every_name_the_split_modules_own():
    """A call site that imported it from ``reporting`` must still find it there."""
    import importlib

    from language_reading_predictors.statistical_models import reporting

    for module in SPLIT_MODULES:
        loaded = importlib.import_module(f"{SM}.{module}")
        tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            else:
                continue
            for name in names:
                if not hasattr(loaded, name):
                    continue
                assert hasattr(reporting, name), f"{module}.{name} is not re-exported"


def test_the_facade_defines_nothing_of_its_own():
    """It is a compatibility seam. New behaviour belongs in an owning module."""
    tree = ast.parse((PACKAGE / "reporting.py").read_text(encoding="utf-8"))
    defined = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert defined == [], defined


def test_every_split_module_is_reachable_and_owns_distinct_names():
    """No name may be defined in two of them — the facade would import ambiguously."""
    import importlib

    owners: dict[str, str] = {}
    clashes: list[str] = []
    for module in SPLIT_MODULES:
        importlib.import_module(f"{SM}.{module}")
        tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name in owners:
                clashes.append(f"{node.name}: {owners[node.name]} and {module}")
            owners[node.name] = module
    assert clashes == [], clashes


def test_the_hub_shrank_by_the_amount_it_moved():
    """A guard on the point of the exercise, not on an arbitrary number."""
    facade = len((PACKAGE / "reporting.py").read_text(encoding="utf-8").splitlines())
    parts = sum(
        len((PACKAGE / f"{m}.py").read_text(encoding="utf-8").splitlines())
        for m in SPLIT_MODULES
    )
    assert facade < 600
    assert parts > 8000
    largest = max(
        len((PACKAGE / f"{m}.py").read_text(encoding="utf-8").splitlines())
        for m in SPLIT_MODULES
    )
    assert largest < 4500, largest


def test_the_row_subset_helper_lives_with_the_dataclass_it_rebuilds():
    from language_reading_predictors.statistical_models.preprocessing import _subset

    assert _subset.__module__.endswith("preprocessing")


def test_the_post_phase_labels_constant_has_one_definition():
    from language_reading_predictors.statistical_models import level_factors

    assert level_factors.POST_PHASE_LABELS is definitions.POST_PHASE_LABELS
    source = (PACKAGE / "level_factors.py").read_text(encoding="utf-8")
    assert "POST_PHASE_LABELS: tuple[str, ...] = (" not in source
