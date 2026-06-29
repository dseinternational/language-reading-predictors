# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard the lightweight model registry against drift from the fitted model set.

``statistical_models/definitions.py`` deliberately duplicates the lightweight
metadata (id / kind / outcome) that otherwise lives on each module's ``SPEC``. These
tests assert the two cannot silently diverge.

The primary guard is **import-free**: every model lives in a ``lrp*.py`` module in
the package, so the registry keys must equal that set of module files. A stronger
cross-check against each module's ``SPEC`` (kind and outcome) is included but
*skipped* if the heavy model modules cannot be imported in the current environment
(they pull in PyMC / PyTensor / Numba), so the suite stays green on a registry that
is correct even where the modelling stack is not installable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from language_reading_predictors.statistical_models import definitions
from language_reading_predictors.statistical_models.definitions import (
    KINDS,
    MODEL_REGISTRY,
    OUTCOMES,
    Status,
)

_PKG_DIR = Path(definitions.__file__).resolve().parent


def _module_ids() -> set[str]:
    """The fitted models, derived import-free from the package's ``lrp*.py`` modules.

    Every model module is named exactly for its id (``lrpitt01.py`` ->
    ``"lrpitt01"``, ``lrp72base.py`` -> ``"lrp72base"``); no non-model module in the
    package starts with ``lrp``.
    """
    return {p.stem for p in _PKG_DIR.glob("lrp*.py")}


def test_registry_matches_module_files() -> None:
    reg, files = set(MODEL_REGISTRY), _module_ids()
    assert reg == files, (
        f"registry vs model modules disagree — only in registry: {sorted(reg - files)}; "
        f"only as modules: {sorted(files - reg)}"
    )


def test_kinds_and_outcomes_are_valid() -> None:
    for definition in MODEL_REGISTRY.values():
        assert definition.kind in KINDS, f"{definition.model_id}: bad kind {definition.kind!r}"
        assert isinstance(definition.status, Status)
        if definition.outcome is not None:
            assert definition.outcome in OUTCOMES, (
                f"{definition.model_id}: unknown outcome {definition.outcome!r}"
            )


def test_base_references_resolve() -> None:
    for definition in MODEL_REGISTRY.values():
        if definition.base is not None:
            assert definition.base in MODEL_REGISTRY, (
                f"{definition.model_id}: base {definition.base!r} is not a registered model"
            )


def _fit_models() -> dict:
    """The authoritative ``MODELS`` dict from the fit script (id -> module)."""
    path = _PKG_DIR.parents[2] / "scripts" / "fit_statistical_model.py"
    spec = importlib.util.spec_from_file_location("_fit_statistical_model", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MODELS


def test_registry_agrees_with_specs() -> None:
    """Cross-check kinds / outcomes against each module's SPEC (skips if the
    modelling stack is not importable in this environment)."""
    try:
        fit = _fit_models()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"model modules not importable here ({type(exc).__name__}: {exc})")

    assert set(MODEL_REGISTRY) == set(fit), (
        f"registry vs fit script disagree — only in registry: "
        f"{sorted(set(MODEL_REGISTRY) - set(fit))}; "
        f"only in fit script: {sorted(set(fit) - set(MODEL_REGISTRY))}"
    )
    for model_id, module in fit.items():
        spec = module.SPEC
        definition = MODEL_REGISTRY[model_id]
        assert definition.kind == spec.kind, (
            f"{model_id}: registry kind {definition.kind!r} != SPEC {spec.kind!r}"
        )
        if spec.outcome_symbol is not None:
            assert definition.outcome == spec.outcome_symbol, (
                f"{model_id}: registry outcome {definition.outcome!r} != "
                f"SPEC {spec.outcome_symbol!r}"
            )
