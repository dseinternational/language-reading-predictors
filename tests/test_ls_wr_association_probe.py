# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the exploratory LS/WR probe's fit contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "notes"
    / "assets"
    / "202607241000-ls-wr-association-probe.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("ls_wr_association_probe", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_level_loader_routes_complete_hearing_block(mod, monkeypatch):
    captured = {}
    sentinel = object()

    def fake_load(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(mod, "load_and_prepare", fake_load)

    result = mod._load_level_probe_data(
        outcomes=("W", "L"), post_covariates=("erbto", "erbto_missing")
    )

    assert result is sentinel
    assert captured["phase_mode"] == "levels"
    assert captured["baseline_covariates"] == ("blocks", "hs", "hs_missing")
    assert captured["post_covariates"] == ("erbto", "erbto_missing")
    assert captured["pre_required"] == ()


def test_q4_loader_routes_hearing_value_and_missingness(mod, monkeypatch, tmp_path):
    captured = {}

    class StopAfterLoad(Exception):
        pass

    def fake_load(**kwargs):
        captured.update(kwargs)
        raise StopAfterLoad

    monkeypatch.setattr(mod, "load_and_prepare", fake_load)

    with pytest.raises(StopAfterLoad):
        mod.run_q4_word_learning(tmp_path)

    assert captured["post_covariates"] == ("hs", "hs_missing")


def test_informative_covariates_drop_only_fitted_row_constants(mod):
    sub = SimpleNamespace(
        covariates={
            "parent": np.array([-1.0, 0.0, 1.0]),
            "constant_flag": np.zeros(3),
            "varying_flag": np.array([0.0, 0.0, 1.0]),
        }
    )

    assert mod._informative_covariates(
        sub, ("parent", "constant_flag", "varying_flag", "absent")
    ) == ["parent", "varying_flag"]


def test_fit_connects_retained_nuisance_and_omits_constant_alias(mod, monkeypatch):
    sub = SimpleNamespace(
        covariates={
            "parent": np.array([-1.0, 0.0, 1.0]),
            "constant_flag": np.zeros(3),
            "varying_flag": np.array([0.0, 0.0, 1.0]),
        }
    )
    captured = {}

    class FakeModel:
        def __init__(self):
            self.free_RVs = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    built = SimpleNamespace(model=FakeModel())

    def fake_build(_sub, **kwargs):
        captured.update(kwargs)
        return built

    monkeypatch.setattr(mod.F, "build_concurrent_model", fake_build)
    monkeypatch.setattr(mod.pm, "sample", lambda **_kwargs: object())
    monkeypatch.setattr(
        mod,
        "_sampling_diagnostics",
        lambda _trace, _built: {
            "max_rhat": 1.0,
            "min_ess": 1000.0,
            "min_ess_bulk": 1000.0,
            "min_ess_tail": 1000.0,
            "min_bfmi": 0.9,
            "n_div": 0,
            "gate_pass": True,
        },
    )

    _trace, _built, diagnostics = mod._fit(
        sub,
        ["L"],
        ["parent", "constant_flag", "varying_flag"],
        draws=10,
    )

    assert captured["covariates"] == ["parent", "varying_flag"]
    assert diagnostics["effective_covariates"] == "parent+varying_flag"


def test_sampling_diagnostics_are_unrounded_and_cover_full_house_gate(mod, monkeypatch):
    captured = {}

    def fake_summary(*_args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {"r_hat": [1.0099], "ess_bulk": [450.5], "ess_tail": [410.2]}
        )

    monkeypatch.setattr(mod.az, "summary", fake_summary)
    monkeypatch.setattr(
        mod,
        "sampling_quality",
        lambda *_args, **_kwargs: SimpleNamespace(
            max_rhat=1.0099,
            min_ess=410.2,
            min_bfmi=0.301,
            n_divergences=0,
        ),
    )
    built = SimpleNamespace(
        model=SimpleNamespace(free_RVs=[SimpleNamespace(name="theta")])
    )

    diagnostics = mod._sampling_diagnostics(object(), built)

    assert captured["var_names"] == ["theta"]
    assert captured["round_to"] == "none"
    assert captured["kind"] == "diagnostics"
    assert diagnostics == {
        "max_rhat": 1.0099,
        "min_ess": 410.2,
        "min_ess_bulk": 450.5,
        "min_ess_tail": 410.2,
        "min_bfmi": 0.301,
        "n_div": 0,
        "gate_pass": True,
    }
