# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for model auto-discovery (#165) - the registry that replaced the
hand-maintained import block + MODELS dict in ``scripts/fit_statistical_model.py``.
"""

from __future__ import annotations

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.registry import discover_models


def test_discovers_known_models_across_families():
    models = discover_models()
    # A soft floor (the suite is ~87 models); the point is nothing whole-family
    # gets dropped by discovery.
    assert len(models) >= 80
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
