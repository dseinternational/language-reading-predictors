# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for model auto-discovery (#165) - the registry that replaced the
hand-maintained import block + MODELS dict in ``scripts/fit_statistical_model.py``.
"""

from __future__ import annotations

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.registry import discover_models


def _expected_model_modules() -> set[str]:
    """Model modules implied by the naming convention: non-package submodules whose
    name carries a number (``lrpitt01``, ``rlmhg01`` ...). Infrastructure modules
    are digit-free, so this stays allowlist-free while failing on any missing or
    unexpected model."""
    import pkgutil

    from language_reading_predictors import statistical_models as _pkg

    return {
        info.name
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
    # Readable family spot-checks (the exact-set test above is the real guard).
    for mid in (
        "lrpitt01",  # ITT suite
        "lrpgf01",  # gain factors
        "lrplf01",  # level factors
        "lrpdid01",  # DiD
        "lrpal01",  # aligned
        "lrpmm01",  # measurement
        "lrphs01",  # horseshoe
        "lrp65",  # adjusted, LAZY spec (get_spec, no module-level SPEC)
        "lrp67",  # LCSM
        "rlmhg01",  # historical growth (2nd dataset)
    ):
        assert mid in models, f"{mid} not discovered"
        assert callable(models[mid].fit)


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
