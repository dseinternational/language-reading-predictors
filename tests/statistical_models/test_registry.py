# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for model auto-discovery (#165) - the registry that replaced the
hand-maintained import block + MODELS dict in ``scripts/fit_statistical_model.py``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.registry import (
    LazyModel,
    discover_models,
)
from language_reading_predictors.statistical_models.run_options import (
    StatisticalRunOptions,
    current_run_options,
)


def _expected_model_modules() -> set[str]:
    """Canonical CLI ids implied by the naming convention: non-package submodules
    whose name carries a number (``lrp_rli_itt_001``, ``lrp_rlm_hg_001`` ...), with
    the module underscores rendered as hyphens (the registry key form since #168
    Phase 2). Infrastructure modules are digit-free, so this stays allowlist-free
    while failing on any missing or unexpected model."""
    import pkgutil

    from language_reading_predictors import statistical_models as _pkg

    return {
        info.name.replace("_", "-")
        for info in pkgutil.iter_modules(_pkg.__path__)
        if not info.ispkg and any(ch.isdigit() for ch in info.name)
    }


def test_discovers_every_model_module():
    # Exact-match against the naming-convention set: dropping several models (or
    # registering an unexpected one) fails the test — stronger than a size floor,
    # and with no hand-maintained allowlist.
    assert set(discover_models()) == _expected_model_modules()


def test_discovers_known_models_across_families():
    models = discover_models()
    # Readable family spot-checks, in canonical CLI form (the exact-set test above
    # is the real guard).
    for mid in (
        "lrp-rli-itt-001",  # ITT suite
        "lrp-rli-gf-001",  # gain factors
        "lrp-rli-lf-001",  # level factors
        "lrp-rli-did-001",  # DiD
        "lrp-rli-al-001",  # aligned
        "lrp-rli-mm-001",  # measurement
        "lrp-rli-hs-001",  # horseshoe
        "lrp-rli-adj-065",  # adjusted, LAZY spec (get_spec, no module-level SPEC)
        "lrp-rli-lcsm-067",  # LCSM
        "lrp-rlm-hg-001",  # historical growth (2nd dataset)
    ):
        assert mid in models, f"{mid} not discovered"
        assert callable(models[mid].fit)


def test_discovery_and_fit_access_do_not_import_model_modules(monkeypatch):
    module_name = (
        "language_reading_predictors.statistical_models.lrp_rli_itt_001"
    )
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    entry = discover_models()["lrp-rli-itt-001"]

    assert isinstance(entry, LazyModel)
    assert callable(entry.fit)
    assert module_name not in sys.modules
    assert entry.SPEC.model_id == "lrp-rli-itt-001"
    assert module_name in sys.modules


def test_lazy_model_scopes_options_to_one_fit(monkeypatch):
    observed = []
    entry = LazyModel(model_id="example", module_name="example.module")
    module = SimpleNamespace(
        fit=lambda config: observed.append((config, current_run_options())) or "fit"
    )
    monkeypatch.setattr(LazyModel, "load", lambda _self: module)

    result = entry.fit(
        "test",
        options=StatisticalRunOptions(target_accept=0.93),
    )

    assert result == "fit"
    assert observed == [("test", StatisticalRunOptions(target_accept=0.93))]
    assert current_run_options() == StatisticalRunOptions()


def test_infrastructure_modules_are_not_registered():
    models = discover_models()
    for infra in (
        "context",
        "factories",
        "pipeline",
        "reporting",
        "diagnostics",
        "priors",
        "measures",
        "preprocessing",
        "datasets",
        "historical",
        "registry",
        "environment",
        "hsgp",
        "likelihood",
    ):
        assert infra not in models


def test_key_matches_spec_model_id_when_present():
    """Where a module exposes a module-level SPEC, its id must equal the key."""
    for mid, mod in discover_models().items():
        spec = getattr(mod, "SPEC", None)
        if isinstance(spec, ModelSpec):
            assert spec.model_id == mid, f"{mid}: SPEC.model_id={spec.model_id!r}"


def test_registry_is_sorted_and_unique():
    keys = list(discover_models())
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))
