# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the matched cross-cohort replication script (#409)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "exploratory" / "cross_cohort_replication.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("cross_cohort_replication", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_followup_rows(seed: int = 0, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = rng.normal(size=n)
    age = rng.normal(size=n)
    memory = rng.normal(size=n)
    group = np.tile([1, 2, 3], n // 3)
    followup = 0.65 * baseline - 0.40 * age + 0.35 * memory + 0.2 * (group == 2) - 0.1 * (group == 3)
    followup += rng.normal(scale=0.55, size=n)
    return pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "group": group,
            "age_raw": age,
            "memory_raw": memory,
            "reading_baseline_logit": baseline,
            "reading_followup_logit": followup,
        }
    )


def test_followup_estimator_recovers_age_and_memory_directions(mod):
    estimates = mod.estimate_followup_associations(_synthetic_followup_rows())

    assert estimates["age"] < -0.25
    assert estimates["verbal_memory"] > 0.20


def test_stable_estimator_removes_group_by_wave_means(mod):
    rng = np.random.default_rng(1)
    n = 300
    group = np.tile([1, 2, 3], n // 3)
    reading_trait = rng.normal(size=n)
    vocabulary_trait = 0.75 * reading_trait + rng.normal(scale=0.45, size=n)
    data: dict[str, np.ndarray] = {
        "subject_id": np.arange(n),
        "group": group,
    }
    for wave in (1, 2, 3):
        data[f"reading_w{wave}"] = reading_trait + 0.4 * wave + 0.7 * group + rng.normal(scale=0.15, size=n)
        data[f"vocabulary_w{wave}"] = (
            vocabulary_trait - 0.3 * wave - 0.5 * group + rng.normal(scale=0.15, size=n)
        )

    estimate = mod.estimate_stable_correlation(pd.DataFrame(data), (1, 2, 3))

    assert estimate["receptive_vocabulary"] > 0.75


def test_stratified_bootstrap_is_deterministic_and_uses_house_interval(mod):
    rows = _synthetic_followup_rows(n=90)
    first = mod.bootstrap_intervals(
        rows,
        mod.estimate_followup_associations,
        n_bootstrap=80,
        rng=np.random.default_rng(42),
    )
    second = mod.bootstrap_intervals(
        rows,
        mod.estimate_followup_associations,
        n_bootstrap=80,
        rng=np.random.default_rng(42),
    )

    assert first == second
    assert set(first) == {"age", "verbal_memory"}
    for lower, upper, valid in first.values():
        assert lower < upper
        assert valid == 80


def test_cross_cohort_contract_propagates_confirmed_byrne_source(mod):
    assert mod.RLI.source_provenance_confirmed is True
    assert mod.RLM.measures_confirmed is True
    assert mod.RLM.source_provenance_confirmed is True
    assert "96" in mod.RLM.source_note
    assert "97" in mod.RLM.source_note
