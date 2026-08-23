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
        json.dumps(
            {
                "passed": passed,
                "checks": {
                    "rhat": True,
                    "ess": True,
                    "divergences": passed,
                    "bfmi": True,
                },
                "divergences": 0 if passed else 1,
                "max_rhat": 1.001,
                "min_ess": 1000.0,
                "bfmi_per_chain": [0.8, 0.9],
            }
        )
    )
    return run_dir


def test_gate_status_rejects_inconsistent_pass_payload(cmp_mod, out_root):
    run_dir = _install_run(cmp_mod, "inconsistent", passed=True)
    (run_dir / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "rhat": True,
                    "ess": True,
                    "divergences": False,
                    "bfmi": True,
                },
                "divergences": 1,
                "max_rhat": 1.001,
                "min_ess": 1000.0,
                "bfmi_per_chain": [0.8, 0.9],
            }
        )
    )
    assert cmp_mod._gate_status("inconsistent", "dev") == "REVIEW"


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


@pytest.mark.parametrize(
    "func_name, ids_attr, out_name",
    [
        (
            "joint_readiness_lxb_w_loo_compare",
            "JOINT_READINESS_LXB_W_LOO_IDS",
            "joint_readiness_lxb_w_loo_compare.csv",
        ),
        (
            "joint_readiness_lxn_w_loo_compare",
            "JOINT_READINESS_LXN_W_LOO_IDS",
            "joint_readiness_lxn_w_loo_compare.csv",
        ),
    ],
)
def test_joint_readiness_comparison_copied_beside_both_reports(
    cmp_mod, out_root, monkeypatch, func_name, ids_attr, out_name
):
    """#404 review: each interaction-vs-baseline comparison must be copied beside
    both paired reports under the generic mechanism_loo_compare.csv name the
    mechanism report partial reads, or the reports silently omit it."""
    ids = getattr(cmp_mod, ids_attr)
    for model_id in ids:
        _install_run(cmp_mod, model_id, passed=True)

    def _write(got_ids, config, out_path):
        assert got_ids == ids and config == "dev"
        pd.DataFrame({"comparison_valid": [True]}).to_csv(out_path, index=False)
        return True

    monkeypatch.setattr(cmp_mod, "_loo_compare", _write)
    out = out_root / "comparison" / out_name
    out.parent.mkdir()
    assert getattr(cmp_mod, func_name)("dev", str(out))
    for model_id in ids:
        copied = Path(cmp_mod._run_dir(model_id, "dev")) / "mechanism_loo_compare.csv"
        assert copied.read_text() == out.read_text()


# ---------------------------------------------------------------------------
# Mechanism forest: fail-closed gating and a defined nonlinear slope estimand
# (#586 finding 6). The forest loaded whatever trace it found, wrote no gate
# column, reported posterior means, and averaged derivatives over a deduplicated
# grid — so ties in the exposure silently reweighted the average.
# ---------------------------------------------------------------------------


def _mech_trace(mech_logit, *, curve=None, beta=None):
    """A mechanism trace carrying ``mech_post_logit`` plus f_mech or beta_mech."""
    n_obs = len(mech_logit)
    constant_data = xr.Dataset(
        {"mech_post_logit": ("obs_id", np.asarray(mech_logit, dtype=float))},
        coords={"obs_id": range(n_obs)},
    )
    data = {}
    if curve is not None:
        # (chain, draw, obs_id) with two identical draws, so quantiles are exact.
        data["f_mech"] = (
            ("chain", "draw", "obs_id"),
            np.asarray([[curve, curve]], dtype=float),
        )
    if beta is not None:
        data["beta_mech"] = (("chain", "draw"), np.asarray([[beta, beta]], dtype=float))
    posterior = xr.Dataset(
        data, coords={"chain": [0], "draw": [0, 1], "obs_id": range(n_obs)}
    )
    return SimpleNamespace(
        groups=("/posterior", "/constant_data"),
        posterior=posterior,
        constant_data=constant_data,
    )


@pytest.mark.parametrize("failing_gate", [False, None])
def test_mechanism_forest_fails_closed_on_the_convergence_gate(
    cmp_mod, out_root, monkeypatch, failing_gate
):
    """A REVIEW or missing gate must abandon the forest, not publish it unmarked.

    ``_gate_status`` states the rule in terms — a REVIEW fit "is not interpretable
    ... a tau/slope from such a run must never enter the comparison forests
    unmarked" — and every other comparison in this script honours it. This one
    loaded every available trace without consulting the gate (#586 finding 6).
    """
    mech_logit = np.linspace(-2.0, 2.0, 6)
    for index, (model_id, _sym) in enumerate(cmp_mod.MECH_IDS):
        if index == 1 and failing_gate is None:
            continue  # no run directory at all -> gate MISSING
        _install_run(
            cmp_mod, model_id, passed=not (index == 1 and failing_gate is False)
        )
    monkeypatch.setattr(
        cmp_mod.az,
        "from_netcdf",
        lambda path: _mech_trace(mech_logit, beta=0.3),
    )
    out = out_root / "mechanism_forest.png"
    assert cmp_mod.mechanism_forest("dev", str(out)) is False
    assert not out.exists()
    assert not out.with_suffix(".csv").exists()


def test_mechanism_forest_records_gate_estimand_and_uses_medians(
    cmp_mod, out_root, monkeypatch
):
    mech_logit = np.linspace(-2.0, 2.0, 6)
    for model_id, _sym in cmp_mod.MECH_IDS:
        _install_run(cmp_mod, model_id, passed=True)
    monkeypatch.setattr(
        cmp_mod.az, "from_netcdf", lambda path: _mech_trace(mech_logit, beta=0.3)
    )
    out = out_root / "mechanism_forest.png"
    assert cmp_mod.mechanism_forest("dev", str(out)) is True

    written = pd.read_csv(out.with_suffix(".csv"))
    assert len(written) == len(cmp_mod.MECH_IDS)
    assert written["converged"].all()
    assert (written["gate_status"] == "PASS").all()
    assert written["estimand"].str.len().gt(0).all()
    # Median-first, the house convention; the column name must say so.
    assert "slope_median" in written.columns
    assert "slope_mean" not in written.columns
    assert np.allclose(written["slope_median"], 0.3)


def test_curve_slope_is_a_fitted_row_average_over_an_irregular_grid(cmp_mod):
    """Ties count once per fitted row, not once per distinct exposure value.

    On a bounded count measure many children share an exposure, so deduplicating
    before averaging reweights the mean toward the sparse tail of the range — where
    an HSGP curve is least constrained. Pinned with a deliberately lopsided grid:
    five rows at the shallow end, one at the steep end.
    """
    # Piecewise-linear curve: slope 1 below 0, slope 11 above it.
    x = np.array([-2.0, -2.0, -2.0, -1.0, 0.0, 1.0])
    curve = np.where(x <= 0.0, x, 11.0 * x)

    slopes = cmp_mod._mechanism_slope_distribution(_mech_trace(x, curve=curve), x)

    # Unique grid is [-2, -1, 0, 1]; np.gradient there gives [1, 1, 6, 11].
    # Deduplicating and averaging equally would give (1 + 1 + 6 + 11) / 4 = 4.75.
    # Weighting by fitted rows gives (1*3 + 1 + 6 + 11) / 6 = 3.5 — the three tied
    # shallow rows now pull the average toward where the data actually are.
    assert np.allclose(slopes, 3.5)
    assert not np.allclose(slopes, 4.75)


def test_mechanism_comparisons_are_copied_beside_both_paired_runs(
    cmp_mod, out_root, monkeypatch
):
    """Model reports render from their own run directory, so a comparison that
    lives only in the shared directory never reaches either report (#586 finding
    13). mech-058/071 and mech-072/172 were the two pairs still missing the copy."""
    for pair, wrapper, expected_note in (
        (cmp_mod.LOO_COMPARE_IDS, cmp_mod.mechanism_loo_compare, "Joint comparison"),
        (cmp_mod.PHONICS_LOO_IDS, cmp_mod.phonics_route_loo_compare, "Nested comparison"),
    ):
        for model_id in pair:
            _install_run(cmp_mod, model_id, passed=True)
        monkeypatch.setattr(
            cmp_mod, "_loo_compare", lambda ids, config, path: (
                pd.DataFrame({"model": list(ids), "elpd_loo": [-1.0, -2.0]}).to_csv(
                    path, index=False
                )
                or True
            )
        )
        out = out_root / "comparison.csv"
        assert wrapper("dev", str(out)) is True
        for model_id in pair:
            beside = Path(cmp_mod._run_dir(model_id, "dev")) / "mechanism_loo_compare.csv"
            assert beside.exists(), model_id
            note = pd.read_csv(beside)["comparison_note"].iloc[0]
            assert expected_note in note
