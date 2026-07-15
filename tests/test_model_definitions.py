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
    """The RLI fitted models, derived import-free from the package's modules.

    Since #168 Phase 2 every model module is named in canonical underscore form
    (``lrp_rli_itt_001.py``); its canonical CLI id is that with hyphens. The
    register catalogues the RLI study, so glob the ``lrp_rli_*`` modules (the one
    ``lrp_rlm_*`` historical-growth module is another study, out of scope here).
    """
    return {p.stem.replace("_", "-") for p in _PKG_DIR.glob("lrp_rli_*.py")}


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


def test_modality_contrast_is_not_catalogued_as_generalisation() -> None:
    definition = MODEL_REGISTRY["lrp-rli-itt-016"]
    assert definition.family == "Modality contrast"


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


def _module_spec(module):
    """A module's ``ModelSpec``, from its module-level ``SPEC`` or its lazy
    ``get_spec()`` (e.g. ``lrp-rli-adj-065`` builds its spec lazily so the DAG-only path
    imports without the Bayesian stack)."""
    spec = getattr(module, "SPEC", None)
    if spec is None:
        build = getattr(module, "get_spec", None)
        spec = build() if callable(build) else None
    return spec


def test_registry_agrees_with_specs() -> None:
    """Cross-check kinds / outcomes against each module's SPEC (skips if the
    modelling stack is not importable in this environment).

    The register catalogues this report's RLI study (``study_id == "rli"``); the
    fit script additionally discovers other-study models (e.g. ``lrp-rlm-hg-001``,
    ``study_id == "rlm"``), which are out of the register's scope and excluded."""
    try:
        fit = _fit_models()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"model modules not importable here ({type(exc).__name__}: {exc})")

    rli = {
        mid: mod
        for mid, mod in fit.items()
        if (spec := _module_spec(mod)) is not None and spec.study_id == "rli"
    }

    assert set(MODEL_REGISTRY) == set(rli), (
        f"registry vs fit script disagree — only in registry: "
        f"{sorted(set(MODEL_REGISTRY) - set(rli))}; "
        f"only in fit script (RLI study): {sorted(set(rli) - set(MODEL_REGISTRY))}"
    )
    for model_id, module in rli.items():
        spec = _module_spec(module)
        definition = MODEL_REGISTRY[model_id]
        assert definition.kind == spec.kind, (
            f"{model_id}: registry kind {definition.kind!r} != SPEC {spec.kind!r}"
        )
        if spec.outcome_symbol is not None:
            assert definition.outcome == spec.outcome_symbol, (
                f"{model_id}: registry outcome {definition.outcome!r} != "
                f"SPEC {spec.outcome_symbol!r}"
            )


def test_provenance_keys_do_not_alias_live_models() -> None:
    # Issue #273: pre-#168 provenance ids 70/74/75/76 were reused by live models
    # (gc-070, med-074/075/076). A bare provenance key that equals a live model's
    # legacy alias is ambiguous; qualified keys are fine. Guard against future
    # bare-id reuse.
    from language_reading_predictors.statistical_models.definitions import (
        provenance_alias_collisions,
    )

    collisions = provenance_alias_collisions()
    assert collisions == [], (
        f"provenance keys collide with live legacy aliases: {collisions} — "
        "qualify them (e.g. 'LRP74 [pre-#168 ...]') so they don't read as a live id"
    )
