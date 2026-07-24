# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Safety tests for cross-model PSIS-LOO comparisons."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "compare_statistical_models.py"
)


@pytest.fixture(scope="module")
def cmp_mod():
    spec = importlib.util.spec_from_file_location("compare_statistical_models_loo", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def out_root(cmp_mod, tmp_path):
    cmp_mod._paths.set_output_root(str(tmp_path))
    yield tmp_path
    cmp_mod._paths.set_output_root(None)


def _fake_trace(child_idx=(0, 1), phase_idx=(0, 0)):
    n_obs = len(child_idx)
    constant_data = xr.Dataset(
        {
            "child_idx": ("obs_id", np.asarray(child_idx)),
            "phase_idx": ("obs_id", np.asarray(phase_idx)),
        },
        coords={"obs_id": range(n_obs)},
    )
    posterior = xr.Dataset(
        {"eta": (("chain", "draw", "obs_id"), [[[0.0] * n_obs]])},
        coords={"chain": [0], "draw": [0], "obs_id": range(n_obs)},
    )
    log_likelihood = xr.Dataset(
        {"y": (("chain", "draw", "obs_id"), [[[0.0] * n_obs]])},
        coords={"chain": [0], "draw": [0], "obs_id": range(n_obs)},
    )
    return SimpleNamespace(
        groups=("/posterior", "/constant_data", "/log_likelihood"),
        posterior=posterior,
        constant_data=constant_data,
        log_likelihood=log_likelihood,
    )


def _install_run(cmp_mod, model_id: str, *, passed: bool) -> Path:
    run_dir = Path(cmp_mod._run_dir(model_id, "dev"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trace.nc").touch()
    (run_dir / "diagnostics_summary.json").write_text(
        json.dumps({"passed": passed})
    )
    return run_dir


def test_loo_excludes_gate_failures(cmp_mod, out_root, monkeypatch):
    ids = ["pass-a", "review", "pass-b"]
    for model_id in ids:
        _install_run(cmp_mod, model_id, passed=model_id != "review")
    traces = {model_id: _fake_trace() for model_id in ids}
    monkeypatch.setattr(
        cmp_mod.az,
        "from_netcdf",
        lambda path: traces[Path(path).parent.name.removesuffix("-dev")],
    )
    compared = {}

    def _compare(eligible):
        compared["ids"] = set(eligible)
        return pd.DataFrame({"elpd_loo": [-1.0, -2.0]}, index=list(eligible))

    monkeypatch.setattr(cmp_mod.az, "compare", _compare)
    out = out_root / "comparison.csv"
    assert cmp_mod._loo_compare(ids, "dev", str(out))
    assert compared["ids"] == {"pass-a", "pass-b"}
    written = pd.read_csv(out)
    assert written["comparison_valid"].all()


def test_equal_counts_but_different_row_order_do_not_produce_delta(
    cmp_mod, out_root, monkeypatch
):
    ids = ["model-a", "model-b"]
    for model_id in ids:
        _install_run(cmp_mod, model_id, passed=True)
    traces = {
        "model-a": _fake_trace(child_idx=(0, 1), phase_idx=(0, 0)),
        "model-b": _fake_trace(child_idx=(1, 0), phase_idx=(0, 0)),
    }
    monkeypatch.setattr(
        cmp_mod.az,
        "from_netcdf",
        lambda path: traces[Path(path).parent.name.removesuffix("-dev")],
    )
    monkeypatch.setattr(
        cmp_mod.az,
        "compare",
        lambda _: pytest.fail("az.compare must not run for mismatched rows"),
    )
    monkeypatch.setattr(
        cmp_mod.az,
        "loo",
        lambda _: SimpleNamespace(elpd=-10.0, se=2.0, p=1.5),
    )

    out = out_root / "comparison.csv"
    assert cmp_mod._loo_compare(ids, "dev", str(out))
    written = pd.read_csv(out)
    assert not written["comparison_valid"].any()
    assert "ordered analysis rows differ" in written["comparison_reason"].iloc[0]


def test_unreliable_pareto_k_invalidates_the_comparison(
    cmp_mod, out_root, monkeypatch
):
    """#390 P1: when a fit's persisted Pareto-k is unreliable (> good_k), PSIS-LOO
    and therefore elpd_diff are untrustworthy, so the comparison must fall back to
    per-model elpd_loo marked invalid rather than az.compare deltas."""
    ids = ["model-a", "model-b"]
    for model_id in ids:
        run_dir = _install_run(cmp_mod, model_id, passed=True)
        max_k = 1.2 if model_id == "model-b" else 0.3  # model-b is unreliable
        pd.DataFrame(
            {
                "observation_index": [0, 1],
                "pareto_k": [max_k, 0.1],
                "good_k_threshold": [0.7, 0.7],
            }
        ).to_csv(run_dir / "pareto_k.csv", index=False)
    traces = {model_id: _fake_trace() for model_id in ids}
    monkeypatch.setattr(
        cmp_mod.az,
        "from_netcdf",
        lambda path: traces[Path(path).parent.name.removesuffix("-dev")],
    )
    monkeypatch.setattr(
        cmp_mod.az,
        "compare",
        lambda _: pytest.fail("az.compare must not run when Pareto-k is unreliable"),
    )
    monkeypatch.setattr(
        cmp_mod.az,
        "loo",
        lambda _: SimpleNamespace(elpd=-10.0, se=2.0, p=1.5),
    )

    out = out_root / "comparison.csv"
    assert cmp_mod._loo_compare(ids, "dev", str(out))
    written = pd.read_csv(out)
    assert not written["comparison_valid"].any()
    assert "Pareto-k" in written["comparison_reason"].iloc[0]
    assert "model-b" in written["comparison_reason"].iloc[0]
    assert "elpd_diff" not in written.columns


def test_did_dose_comparison_is_copied_beside_both_reports(
    cmp_mod, out_root, monkeypatch
):
    for model_id in cmp_mod.DID_DOSE_LOO_IDS:
        _install_run(cmp_mod, model_id, passed=True)

    def _write(ids, config, out_path):
        assert ids == cmp_mod.DID_DOSE_LOO_IDS and config == "dev"
        pd.DataFrame({"comparison_valid": [True]}).to_csv(out_path, index=False)
        return True

    monkeypatch.setattr(cmp_mod, "_loo_compare", _write)
    out = out_root / "comparison" / "did_dose_loo_compare.csv"
    out.parent.mkdir()
    assert cmp_mod.did_dose_loo_compare("dev", str(out))
    for model_id in cmp_mod.DID_DOSE_LOO_IDS:
        copied = Path(cmp_mod._run_dir(model_id, "dev")) / out.name
        assert copied.read_text() == out.read_text()
