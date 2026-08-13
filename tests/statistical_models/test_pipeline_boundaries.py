# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Structural guards for the family-owned pipeline boundaries (#394 and #521).

The split only stays possible while the dependency edges point one way: model
modules call their family orchestration module under ``pipelines/``, which calls
the shared artefact, presentation and lifecycle modules beneath it. The retired
``pipeline.py`` facade must not return, because it would restore a central import
edge across every family.

The tests pin direct model-to-family imports, the family entry points, complete
``ModelSpec.kind`` coverage and the prohibition on family-local posterior
sampling.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

from language_reading_predictors.statistical_models import definitions

PACKAGE = pathlib.Path(definitions.__file__).parent
RETIRED_FACADE = "language_reading_predictors.statistical_models.pipeline"

# The shared layer ``pipelines/*`` is built on: artefact production, presentation,
# the stage binding, the samplers and the posterior summaries. Each was carved out
# of ``pipeline.py`` (or, for ``diagnostics`` and ``reporting``, predates it) and
# must stay below the family pipelines.
SHARED_MODULES = (
    "adjustment",
    "artifacts",
    "diagnostics",
    "figure_artifacts",
    "lcf_inference",
    "lcf_summaries",
    "ppc_artifacts",
    "prior_artifacts",
    "publication",
    "reporting",
    "runtime",
    "stages",
    "subfits",
)

# Every family-owned entry point. Most families have exactly one. Mediation is
# the outlier, with three fit functions and a data-preparation helper used by a
# maintenance script; adjusted, correlated-factor and horseshoe families also
# carry Byrne (RLM) cohort entry points under the same ``ModelSpec.kind``.
FAMILY_ENTRY_POINTS: dict[str, tuple[str, ...]] = {
    "adjusted": ("fit_adjusted", "fit_rlm_adjusted"),
    "aligned": ("fit_aligned",),
    "block_exposure": ("fit_block_exposure",),
    "concurrent": ("fit_concurrent",),
    "corr_factor": ("fit_correlated_factor", "fit_rlm_corr_factor"),
    "did": ("fit_did",),
    "dose_response": ("fit_dose_response",),
    "gain_factors": ("fit_gain_factors",),
    "growth": ("fit_growth",),
    "historical_growth": ("fit_historical_growth",),
    "historical_joint": ("fit_rlm_joint_growth",),
    "horseshoe": ("fit_horseshoe", "fit_rlm_horseshoe"),
    "itt": ("fit_itt",),
    "joint": ("fit_joint",),
    "joint_mechanism": ("fit_joint_mechanism",),
    "lcsm": ("fit_lcsm",),
    "level_factors": ("fit_level_factors",),
    "long_corr_factor": ("fit_longitudinal_corr_factor",),
    "mechanism": ("fit_mechanism",),
    "mediation": (
        "fit_mediation",
        "fit_mediation_multi",
        "fit_mediation_period_stacked",
        "prepare_mediation_data",
    ),
    "survival": ("fit_survival",),
}

DIRECT_ENTRY_POINTS = sorted(
    (family, entry) for family, entries in FAMILY_ENTRY_POINTS.items() for entry in entries
)

# Incremental #521 adoption ledger. Each listed primary entry point must express
# its invariant execution through ``PrimaryFitPlan``; exceptional families are
# added only after their current ordering has been characterised.
PRIMARY_LIFECYCLE_ENTRY_POINTS = (
    ("adjusted", "fit_adjusted"),
    ("adjusted", "fit_rlm_adjusted"),
    ("aligned", "fit_aligned"),
    ("block_exposure", "fit_block_exposure"),
    ("corr_factor", "fit_correlated_factor"),
    ("corr_factor", "fit_rlm_corr_factor"),
    ("did", "fit_did"),
    ("dose_response", "fit_dose_response"),
    ("gain_factors", "fit_gain_factors"),
    ("growth", "fit_growth"),
    ("historical_growth", "fit_historical_growth"),
    ("historical_joint", "fit_rlm_joint_growth"),
    ("horseshoe", "fit_horseshoe"),
    ("horseshoe", "fit_rlm_horseshoe"),
    ("lcsm", "fit_lcsm"),
    ("level_factors", "fit_level_factors"),
    ("mechanism", "fit_mechanism"),
    ("mediation", "fit_mediation"),
    ("mediation", "fit_mediation_multi"),
    ("mediation", "fit_mediation_period_stacked"),
    ("survival", "fit_survival"),
)

# ``mediation_multi`` is a distinct ``ModelSpec.kind`` but the same family module:
# its two-mediator decomposition shares the g-formula machinery with the
# single-mediator fits, so ``pipelines/mediation.py`` owns both entry points.
KIND_MODULES = {"mediation_multi": "mediation"}


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
    assert RETIRED_FACADE in _imported_modules(source, package)


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
    assert RETIRED_FACADE not in _imported_modules(source, package)


def test_source_modules_do_not_import_the_retired_facade():
    """No package source can retain a dependency on the deleted module."""
    sources = sorted(PACKAGE.parent.rglob("*.py"))
    offenders = [path.name for path in sources if RETIRED_FACADE in _imports_of(path)]
    assert not offenders, f"source modules importing the retired facade: {offenders}"


def test_source_docstrings_do_not_name_retired_facade_entry_points():
    """Documentation must point at the family owner, not deleted ``pipeline.fit_*``."""
    sources = sorted(PACKAGE.rglob("*.py"))
    offenders = [
        path.name
        for path in sources
        if "pipeline.fit_" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"source docstrings naming retired entry points: {offenders}"


def test_shared_artifact_modules_do_not_import_the_retired_facade():
    for name in SHARED_MODULES:
        path = PACKAGE / f"{name}.py"
        assert path.exists(), f"{name}.py is missing"
        assert RETIRED_FACADE not in _imports_of(path), (
            f"{name}.py imports the retired facade; the shared layer must stay below families"
        )


def test_family_pipelines_do_not_import_the_retired_facade():
    modules = sorted((PACKAGE / "pipelines").glob("*.py"))
    present = {p.stem for p in modules} - {"__init__"}
    assert present == set(FAMILY_ENTRY_POINTS), (
        "pipelines/ and FAMILY_ENTRY_POINTS disagree; update the guard when a "
        f"family moves. Only in package: {sorted(present - set(FAMILY_ENTRY_POINTS))}; "
        f"only in guard: {sorted(set(FAMILY_ENTRY_POINTS) - present)}"
    )
    for path in modules:
        assert RETIRED_FACADE not in _imports_of(path), (
            f"pipelines/{path.name} imports the retired facade; family modules must not"
        )


@pytest.mark.parametrize("family,entry", DIRECT_ENTRY_POINTS)
def test_family_modules_expose_the_registered_entry_points(family, entry):
    """Each documented entry point is owned by its family module."""
    module = importlib.import_module(
        f"language_reading_predictors.statistical_models.pipelines.{family}"
    )
    assert callable(getattr(module, entry))


def test_registered_model_modules_import_family_pipelines_directly():
    """#521 acceptance: every registered model bypasses the retired facade."""
    paths = {
        model_id: PACKAGE / f"{model_id.replace('-', '_')}.py"
        for model_id in definitions.MODEL_REGISTRY
    }
    missing = [
        str(path.relative_to(PACKAGE)) for path in paths.values() if not path.exists()
    ]
    assert not missing, f"registered models with no module: {missing}"

    imports = {model_id: _imports_of(path) for model_id, path in paths.items()}
    offenders = [
        paths[model_id].name
        for model_id, imported in imports.items()
        if RETIRED_FACADE in imported
    ]
    assert not offenders, f"registered models importing the retired facade: {offenders}"

    missing_direct = []
    for model_id, definition in definitions.MODEL_REGISTRY.items():
        family = KIND_MODULES.get(definition.kind, definition.kind)
        expected = f"{SM}.pipelines.{family}"
        if expected not in imports[model_id]:
            missing_direct.append(f"{paths[model_id].name}: {expected}")
    assert not missing_direct, f"models missing their direct family import: {missing_direct}"


def test_maintenance_scripts_import_family_pipelines_directly():
    """The retired facade is not kept alive by a standalone maintenance caller."""
    scripts = sorted((PACKAGE.parents[2] / "scripts").glob("*.py"))
    offenders = [path.name for path in scripts if RETIRED_FACADE in _imports_of(path)]
    assert not offenders, f"scripts importing the retired facade: {offenders}"


def test_test_modules_do_not_import_the_retired_facade():
    """Tests exercise the owning family module rather than a compatibility seam."""
    this_file = pathlib.Path(__file__).resolve()
    tests = [
        path
        for path in sorted((PACKAGE.parents[2] / "tests").rglob("*.py"))
        if path.resolve() != this_file
    ]
    offenders = [path.name for path in tests if RETIRED_FACADE in _imports_of(path)]
    assert not offenders, f"tests importing the retired facade: {offenders}"


def test_compatibility_facade_is_retired():
    """The aggregate facade has no documented external consumer and stays absent."""
    assert not (PACKAGE / "pipeline.py").exists()


def test_every_registered_family_kind_has_an_orchestration_module():
    """#394 acceptance criterion 2, checked against the authoritative kind list."""
    modules = {p.stem for p in (PACKAGE / "pipelines").glob("*.py")} - {"__init__"}
    missing = {
        kind
        for kind in definitions.KINDS
        if KIND_MODULES.get(kind, kind) not in modules
    }
    assert not missing, f"family kinds with no module under pipelines/: {sorted(missing)}"


@pytest.mark.parametrize("family,entry", PRIMARY_LIFECYCLE_ENTRY_POINTS)
def test_adopted_primary_entry_points_use_the_shared_lifecycle(family, entry):
    """The incremental adoption ledger cannot silently fall back to manual stages."""
    path = PACKAGE / "pipelines" / f"{family}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    assert any(
        isinstance(call.func, ast.Name) and call.func.id == "PrimaryFitPlan"
        for call in calls
    ), f"pipelines/{family}.py::{entry} does not declare a PrimaryFitPlan"
    assert any(
        isinstance(call.func, ast.Attribute) and call.func.attr == "run_primary_fit"
        for call in calls
    ), f"pipelines/{family}.py::{entry} does not call run_primary_fit"


@pytest.mark.parametrize(
    "family,entry",
    [
        ("block_exposure", "fit_block_exposure"),
        ("gain_factors", "fit_gain_factors"),
        ("level_factors", "fit_level_factors"),
    ],
)
def test_late_psense_families_preserve_their_post_trace_artifact_order(family, entry):
    """The overlay and forest still precede these families' late power scaling.

    ``run_primary_fit`` ends with trace persistence for ``family_tail`` psense;
    this structural characterisation pins the explicit family tail that follows.
    """
    path = PACKAGE / "pipelines" / f"{family}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    primary_plan = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "PrimaryFitPlan"
    )
    psense_timing = next(
        keyword.value
        for keyword in primary_plan.keywords
        if keyword.arg == "psense_timing"
    )
    assert isinstance(psense_timing, ast.Constant)
    assert psense_timing.value == "family_tail"

    def _line(attribute: str) -> int:
        return next(
            call.lineno
            for call in calls
            if (
                isinstance(call.func, ast.Attribute) and call.func.attr == attribute
            )
            or (isinstance(call.func, ast.Name) and call.func.id == attribute)
        )

    assert _line("run_primary_fit") < _line("save_prior_posterior_plot")
    assert _line("save_prior_posterior_plot") < _line("save_forest_plot")
    assert _line("save_forest_plot") < _line("run_psense")


def test_did_preserves_cell_ppc_and_post_trace_sensitivity_order():
    """DiD's stratified PPC is pre-gate; its power scaling stays post-trace."""
    path = PACKAGE / "pipelines" / "did.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "fit_did"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    primary_plan = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "PrimaryFitPlan"
    )
    keywords = {keyword.arg: keyword.value for keyword in primary_plan.keywords}

    assert isinstance(keywords["post_ppc_audit"], ast.Name)
    assert keywords["post_ppc_audit"].id == "_write_did_cell_ppc"
    assert isinstance(keywords["psense_timing"], ast.Constant)
    assert keywords["psense_timing"].value == "family_tail"

    def _line(attribute: str) -> int:
        return next(
            call.lineno
            for call in calls
            if (
                isinstance(call.func, ast.Attribute) and call.func.attr == attribute
            )
        )

    assert _line("run_primary_fit") < _line("save_prior_posterior_plot")
    assert _line("save_prior_posterior_plot") < _line("run_psense")


@pytest.mark.parametrize(
    "entry,expected_timing",
    [("fit_adjusted", "after_ppc"), ("fit_rlm_adjusted", "before_ppc")],
)
def test_adjusted_fits_consume_the_gate_and_preserve_psense_timing(
    entry, expected_timing
):
    """Both adjusted ports reuse the returned gate; only RLI is PPC-first."""
    path = PACKAGE / "pipelines" / "adjusted.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry
    )
    assignments = [node for node in ast.walk(function) if isinstance(node, ast.Assign)]
    gate_assignment = next(
        node
        for node in assignments
        if any(isinstance(target, ast.Name) and target.id == "_primary_gate" for target in node.targets)
    )
    assert isinstance(gate_assignment.value, ast.Call)
    assert isinstance(gate_assignment.value.func, ast.Attribute)
    assert gate_assignment.value.func.attr == "run_primary_fit"

    primary_plan = next(
        call
        for call in ast.walk(gate_assignment.value)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "PrimaryFitPlan"
    )
    timings = {
        keyword.arg: keyword.value
        for keyword in primary_plan.keywords
        if keyword.arg == "psense_timing"
    }
    if expected_timing == "before_ppc":
        assert not timings
    else:
        assert isinstance(timings["psense_timing"], ast.Constant)
        assert timings["psense_timing"].value == expected_timing

    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    clean_pass = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "convergence_gate_clean_passed"
    )
    overlay = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "save_prior_posterior_plot"
    )
    assert gate_assignment.lineno < clean_pass.lineno < overlay.lineno


def test_no_family_module_samples_a_posterior_of_its_own():
    """#394 acceptance criterion: one shared runner for every sub-fit.

    A family pipeline declares *which* sub-fits to run — which wave, which
    predictor, which prior width — and delegates the sampling. Two modules in the
    package call ``pm.sample``: ``diagnostics.sample_posterior`` for the primary
    fit and ``subfits.run_subfit`` for every sub-fit. An inline ``pm.sample`` in a
    family module is how three of them drifted apart before design point 5, so it
    fails here now.

    Scoped to ``pipelines/`` deliberately. The post-hoc sweep tools
    (``influence.py`` and the ``scripts/*_prior_sensitivity.py`` runners) sample
    their own refits outside any family fit, with their own provenance
    conventions; bringing them onto the runner is separate work, not something
    this guard should quietly assert is done.
    """
    offenders = [
        path.name
        for path in sorted((PACKAGE / "pipelines").glob("*.py"))
        if "pm.sample(" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"family modules sampling their own posterior: {offenders}; "
        "use subfits.run_subfit (sub-fit) or the shared stages (primary)"
    )
