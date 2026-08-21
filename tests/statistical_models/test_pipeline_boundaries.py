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
    "pooled_levels": ("fit_pooled_levels",),
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
    ("pooled_levels", "fit_pooled_levels"),
    ("adjusted", "fit_adjusted"),
    ("adjusted", "fit_rlm_adjusted"),
    ("aligned", "fit_aligned"),
    ("block_exposure", "fit_block_exposure"),
    ("concurrent", "fit_concurrent"),
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
    ("itt", "fit_itt"),
    ("joint", "fit_joint"),
    ("joint_mechanism", "fit_joint_mechanism"),
    ("lcsm", "fit_lcsm"),
    ("level_factors", "fit_level_factors"),
    ("long_corr_factor", "fit_longitudinal_corr_factor"),
    ("mechanism", "fit_mechanism"),
    ("mediation", "fit_mediation"),
    ("mediation", "fit_mediation_multi"),
    ("mediation", "fit_mediation_period_stacked"),
    ("survival", "fit_survival"),
)

# Public dispatchers whose distinct primary paths must each adopt the lifecycle.
# ITT's ordinary entry point is itself a primary path and its floor helper is the
# second; joint mechanism dispatches entirely to its two design implementations.
PRIMARY_LIFECYCLE_IMPLEMENTATIONS = {
    ("itt", "fit_itt"): ("fit_itt", "fit_itt_floor_rule"),
    ("joint_mechanism", "fit_joint_mechanism"): (
        "_fit_joint_mechanism_levels",
        "_fit_joint_mechanism_transition",
    ),
}

# ``mediation_multi`` is a distinct ``ModelSpec.kind`` but the same family module:
# its two-mediator decomposition shares the g-formula machinery with the
# single-mediator fits, so ``pipelines/mediation.py`` owns both entry points.
KIND_MODULES = {"mediation_multi": "mediation"}

# ``ModelSpec.extra`` is retained only as a strict legacy-declaration adapter, a
# single global sampler-option boundary, and persisted-metadata compatibility /
# provenance. Runtime pipelines and artefact logic consume resolved plans instead.
# This is deliberately function-granular: adding a read anywhere else requires an
# explicit architectural decision rather than quietly growing the seam again.
SPEC_EXTRA_BOUNDARY_FUNCTIONS = {
    ("adjusted.py", "declared_adjusted_settings"),
    ("adjusted.py", "resolve_adjusted_run_plan"),
    ("aligned.py", "declared_aligned_settings"),
    ("block_exposure.py", "declared_block_exposure_settings"),
    ("concurrent.py", "declared_concurrent_settings"),
    ("context.py", "spec_target_accept"),
    ("corr_factor.py", "declared_corr_factor_settings"),
    ("corr_factor.py", "resolve_corr_factor_run_plan"),
    ("did.py", "declared_did_settings"),
    ("dose_response.py", "declared_dose_response_settings"),
    ("gain_factors.py", "declared_gain_factors_settings"),
    ("growth.py", "declared_growth_settings"),
    ("historical_growth.py", "declared_historical_growth_settings"),
    ("historical_joint.py", "declared_historical_joint_settings"),
    ("horseshoe.py", "declared_horseshoe_settings"),
    ("horseshoe.py", "resolve_horseshoe_run_plan"),
    ("influence.py", "summarise_influence_refit"),
    ("itt.py", "declared_itt_settings"),
    ("pooled_levels.py", "resolve_pooled_levels_run_plan"),
    ("itt.py", "declared_settings_dict"),
    ("joint.py", "declared_joint_settings"),
    ("joint_mechanism.py", "declared_joint_mechanism_settings"),
    ("lcsm.py", "declared_lcsm_settings"),
    ("level_factors.py", "declared_level_factors_settings"),
    ("long_corr_factor.py", "declared_long_corr_factor_settings"),
    ("mechanism.py", "declared_mechanism_settings"),
    ("mediation_settings.py", "declared_mediation_multi_settings"),
    ("mediation_settings.py", "declared_mediation_settings"),
    ("reporting.py", "_historical_growth_run_plan"),
    ("reporting.py", "_mediation_run_plan"),
    ("reporting.py", "_reuse_compatibility_contract"),
    ("reporting.py", "write_run_metadata"),
    ("survival.py", "declared_survival_settings"),
}


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


def _spec_extra_functions(path: pathlib.Path) -> set[str]:
    """Functions containing a direct ``spec.extra``/``context.spec.extra`` read."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    functions: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != "extra":
            continue
        owner = node.value
        is_spec = isinstance(owner, ast.Name) and owner.id == "spec"
        is_context_spec = isinstance(owner, ast.Attribute) and owner.attr == "spec"
        if not (is_spec or is_context_spec):
            continue
        current: ast.AST = node
        while current in parents and not isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            current = parents[current]
        functions.add(
            current.name
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef))
            else "<module>"
        )
    return functions


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
        (SM, f"from {SM}.runtime import shared_stages"),
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


def test_spec_extra_reads_stay_inside_explicit_compatibility_boundaries():
    """#521: scientific runtime logic consumes typed, resolved family plans."""
    observed = {
        (path.name, function)
        for path in sorted(PACKAGE.rglob("*.py"))
        for function in _spec_extra_functions(path)
    }
    unexpected = observed - SPEC_EXTRA_BOUNDARY_FUNCTIONS
    assert not unexpected, (
        "spec.extra read escaped its declaration/global-option/provenance boundary: "
        f"{sorted(unexpected)}"
    )

    scripts = sorted((PACKAGE.parents[2] / "scripts").glob("*.py"))
    script_reads = {
        (path.name, function)
        for path in scripts
        for function in _spec_extra_functions(path)
    }
    assert not script_reads, (
        "maintenance scripts must resolve typed plans or call the shared global-option "
        f"boundary, not read spec.extra directly: {sorted(script_reads)}"
    )


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
    implementations = PRIMARY_LIFECYCLE_IMPLEMENTATIONS.get(
        (family, entry), (entry,)
    )
    for implementation in implementations:
        function = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == implementation
        )
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

        assert any(
            isinstance(call.func, ast.Name)
            and call.func.id in {"PrimaryFitPlan", "_jm_primary_fit_plan"}
            for call in calls
        ), (
            f"pipelines/{family}.py::{implementation} does not declare a "
            "PrimaryFitPlan"
        )
        assert any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "run_primary_fit"
            for call in calls
        ), f"pipelines/{family}.py::{implementation} does not call run_primary_fit"


def test_primary_lifecycle_ledger_covers_every_fit_entry_point():
    """Completed adoption makes the former incremental ledger exhaustive."""
    expected = set(DIRECT_ENTRY_POINTS) - {("mediation", "prepare_mediation_data")}
    assert set(PRIMARY_LIFECYCLE_ENTRY_POINTS) == expected


def test_runtime_does_not_offer_partial_primary_lifecycle_escape_hatches():
    """Completed adoption retires wrappers that let a family rebuild half the order."""
    tree = ast.parse((PACKAGE / "runtime.py").read_text(encoding="utf-8"))
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "run_sampling_and_loo" not in functions
    assert "run_ppc" not in functions


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


def test_joint_declares_multi_outcome_ppc_loo_pit_and_family_tail():
    """Joint ITT keeps its per-outcome PPC/LOO-PIT and post-trace sensitivity."""
    path = PACKAGE / "pipelines" / "joint.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "fit_joint"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    primary_plan = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "PrimaryFitPlan"
    )
    keywords = {keyword.arg: keyword.value for keyword in primary_plan.keywords}

    assert isinstance(keywords["custom_posterior_predictive"], ast.Name)
    assert keywords["custom_posterior_predictive"].id == "_run_joint_ppc"
    assert isinstance(keywords["post_extended_audit"], ast.Name)
    assert keywords["post_extended_audit"].id == "_write_joint_loo_pit"
    assert isinstance(keywords["include_loo_pit"], ast.Constant)
    assert keywords["include_loo_pit"].value is False
    assert isinstance(keywords["psense_timing"], ast.Constant)
    assert keywords["psense_timing"].value == "family_tail"

    def _line(attribute: str) -> int:
        return next(
            call.lineno
            for call in calls
            if isinstance(call.func, ast.Attribute) and call.func.attr == attribute
        )

    assert _line("run_primary_fit") < _line("save_prior_posterior_plot")
    assert _line("save_prior_posterior_plot") < _line("run_psense")


def test_longitudinal_factor_declares_stitched_loo_and_pre_trace_sensitivity():
    """LCF keeps exact child LOO after sampling and psense before persistence."""
    path = PACKAGE / "pipelines" / "long_corr_factor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "fit_longitudinal_corr_factor"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    primary_plan = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "PrimaryFitPlan"
    )
    keywords = {keyword.arg: keyword.value for keyword in primary_plan.keywords}

    assert isinstance(keywords["post_sampling_audit"], ast.Name)
    assert keywords["post_sampling_audit"].id == "_stitch_child_loo"
    assert isinstance(keywords["psense_timing"], ast.Constant)
    assert keywords["psense_timing"].value == "before_trace"
    assert isinstance(keywords["prior_predictive_var_names"], ast.Call)

    def _line(name: str) -> int:
        return next(
            call.lineno
            for call in calls
            if (isinstance(call.func, ast.Name) and call.func.id == name)
            or (isinstance(call.func, ast.Attribute) and call.func.attr == name)
        )

    assert _line("run_primary_fit") < _line("write_indicator_prior_check")
    assert _line("write_indicator_prior_check") < _line("save_prior_posterior_plot")


def test_concurrent_declares_anchor_interleave_and_gate_audit():
    """Concurrent retains wave subfits around sampling and its anchor-only labels."""
    path = PACKAGE / "pipelines" / "concurrent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fit_concurrent"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    primary_plan = next(
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "PrimaryFitPlan"
    )
    keywords = {keyword.arg: keyword.value for keyword in primary_plan.keywords}

    assert isinstance(keywords["post_sampling_audit"], ast.Name)
    assert keywords["post_sampling_audit"].id == "_finish_wave_fits"
    assert isinstance(keywords["post_gate_audit"], ast.Name)
    assert keywords["post_gate_audit"].id == "_record_primary_convergence"
    assert isinstance(keywords["summary_header"], ast.Constant)
    assert keywords["summary_header"].value == "Summary diagnostics (primary wave)"
    assert isinstance(keywords["extended_header"], ast.Constant)
    assert keywords["extended_header"].value == "Extended diagnostics (primary wave)"


def test_itt_paths_preserve_prior_plot_ppc_audit_and_late_sensitivity():
    """Ordinary and floor ITT keep their distinct prior/PPC phase contracts."""
    path = PACKAGE / "pipelines" / "itt.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    ordinary_call = next(
        node
        for node in ast.walk(functions["fit_itt"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PrimaryFitPlan"
    )
    ordinary = {keyword.arg: keyword.value for keyword in ordinary_call.keywords}
    assert isinstance(ordinary["post_ppc_audit"], ast.Name)
    assert ordinary["post_ppc_audit"].id == "_write_itt_ppc"
    assert isinstance(ordinary["post_gate_audit"], ast.Name)
    assert ordinary["post_gate_audit"].id == "_plot_prior_after_gate"
    assert isinstance(ordinary["psense_timing"], ast.Constant)
    assert ordinary["psense_timing"].value == "family_tail"

    floor_call = next(
        node
        for node in ast.walk(functions["fit_itt_floor_rule"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PrimaryFitPlan"
    )
    floor = {keyword.arg: keyword.value for keyword in floor_call.keywords}
    assert isinstance(floor["ppc_var_names"], ast.Tuple)
    assert [item.value for item in floor["ppc_var_names"].elts] == ["y_offfloor"]
    assert isinstance(floor["plot_prior_predictive"], ast.Lambda)
    assert isinstance(floor["post_ppc_audit"], ast.Name)
    assert floor["post_ppc_audit"].id == "_write_floor_ppc"
    assert isinstance(floor["psense_timing"], ast.Constant)
    assert floor["psense_timing"].value == "family_tail"


def test_joint_mechanism_declares_per_outcome_predictive_diagnostics():
    """Both designs share the declared per-outcome PPC profile; the per-outcome
    LOO-PIT hook is declared exactly where PSIS-LOO itself is computed (the
    saturated levels design computes neither; 2026-08-21 review, finding 2)."""
    path = PACKAGE / "pipelines" / "joint_mechanism.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_jm_primary_fit_plan"
    )
    primary_plan = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PrimaryFitPlan"
    )
    keywords = {keyword.arg: keyword.value for keyword in primary_plan.keywords}

    assert isinstance(keywords["plot_prior_predictive"], ast.Name)
    assert keywords["plot_prior_predictive"].id == "_plot_prior"
    assert isinstance(keywords["custom_posterior_predictive"], ast.Name)
    assert keywords["custom_posterior_predictive"].id == "_run_ppc"
    loo_pit_hook = keywords["post_extended_audit"]
    assert isinstance(loo_pit_hook, ast.IfExp)
    assert isinstance(loo_pit_hook.test, ast.Name)
    assert loo_pit_hook.test.id == "compute_loo"
    assert isinstance(loo_pit_hook.body, ast.Name)
    assert loo_pit_hook.body.id == "_write_loo_pit"
    assert isinstance(loo_pit_hook.orelse, ast.Constant)
    assert loo_pit_hook.orelse.value is None
    assert isinstance(keywords["extended_term"], ast.Constant)
    assert keywords["extended_term"].value == "delta_ls_decoding"
    assert isinstance(keywords["include_loo_pit"], ast.Constant)
    assert keywords["include_loo_pit"].value is False


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
