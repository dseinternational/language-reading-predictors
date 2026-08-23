# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for the gain-factor treatment-prior sweep runner (#391).

The runner's manifest is release-gate evidence — it can lift a withheld
headline — so its bindings mirror the level/did runners', with the gain-factor
twists exercised here: the sweep set is keyed by **model id** (the
taught-vocabulary outcomes each have two registered primaries), the loader
refuses treated-only companions (no ``beta_trt``) and moderation variants
(``beta_trt`` never released as causal), and the identifying sample is the
**period-1** arm split read from ``on_intervention``/``phase_idx``. The shared
attach/rollback discipline itself is covered by
``test_level_prior_sensitivity.py`` and ``test_did_prior_sensitivity.py``; this
file adds the gain-factor loader, its sampling contract, the tier-grid
selection and the same-outcome sibling-model refusal.
"""

from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models.sensitivity import (
    _STANDARD_REQUIRED_COLUMNS,
    STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS,
    STANDARD_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_PROVENANCE_ATTR,
    STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS,
    attach_outcome_bundle,
    load_primary_gf_reference,
    persist_sensitivity_trace,
    sha256_file,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "gf_prior_sensitivity.py"
)
_SPEC = spec_from_file_location("_lrp_gf_prior_sensitivity", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gf_script = module_from_spec(_SPEC)
_SPEC.loader.exec_module(gf_script)

_SAMPLING = {
    "draws": 40,
    "tune": 10,
    "chains": 2,
    "target_accept": 0.95,
    "random_seed": 47,
}
_N_CHILDREN = 8
_N_OBS = 3 * _N_CHILDREN


def _fake_primary(
    root: Path,
    *,
    model_id: str = "lrp-rli-gf-001",
    outcome: str = "W",
    kind: str = "gain_factors",
    run_plan: dict | None = None,
    posterior_vars: tuple[str, ...] = ("alpha", "beta_trt"),
    sampling: dict | None = None,
    constant_vars: tuple[str, ...] = ("on_intervention", "phase_idx"),
) -> Path:
    """A minimal on-disk primary fit satisfying load_primary_gf_reference."""
    sampling = dict(_SAMPLING if sampling is None else sampling)
    if run_plan is None:
        run_plan = {"treated_only": False, "moderation_variant": False}
    directory = root / f"{model_id}-reporting"
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    chains, draws = sampling["chains"], sampling["draws"]
    values = {
        name: (("chain", "draw"), rng.normal(size=(chains, draws)))
        for name in posterior_vars
    }
    posterior = xr.Dataset(
        values,
        coords={"chain": np.arange(chains), "draw": np.arange(draws)},
    )
    # A period-stacked panel: children alternate arms; every phase>=1 row is on
    # intervention, so only the period-1 rows split by arm.
    child_g = np.tile([1.0, 0.0], _N_CHILDREN // 2)
    child_idx = np.repeat(np.arange(_N_CHILDREN), 3)
    phase = np.tile(np.arange(3, dtype=float), _N_CHILDREN)
    trt = ((child_g[child_idx] == 1.0) | (phase >= 1)).astype(float)
    data = {"on_intervention": trt, "phase_idx": phase}
    constant_data = xr.Dataset(
        {name: (("obs_id",), data[name]) for name in constant_vars if name in data},
        coords={"obs_id": np.arange(_N_OBS)},
    )
    xr.DataTree.from_dict(
        {"posterior": posterior, "constant_data": constant_data}
    ).to_netcdf(directory / "trace.nc")
    config = {
        "model_id": model_id,
        "outcome_symbol": outcome,
        "kind": kind,
        "resolved_run_plan": run_plan,
        "data_sha256": "a" * 64,
        "n_obs": _N_OBS,
        "sampling": sampling,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


# --- load_primary_gf_reference ------------------------------------------------


def test_gf_reference_loads_and_binds_period1_arms(tmp_path):
    d = _fake_primary(tmp_path)
    ref = load_primary_gf_reference(d, "lrp-rli-gf-001", config_name="reporting")
    assert ref.model_id == "lrp-rli-gf-001"
    assert ref.outcome == "W"
    assert ref.config_sha256 == sha256_file(d / "config.json")
    assert ref.trace_sha256 == sha256_file(d / "trace.nc")
    # The identifying sample is the period-1 arm split, not the stacked total.
    assert ref.n == _N_OBS
    assert ref.n_intervention == _N_CHILDREN // 2
    assert ref.n_control == _N_CHILDREN // 2


@pytest.mark.parametrize(
    "mutate,match",
    [
        (dict(model_id="lrp-rli-gf-999"), "unsupported gain-factor sensitivity model"),
        (dict(kind="itt"), "primary kind mismatch"),
        (dict(posterior_vars=("alpha",)), "lacks alpha or beta_trt"),
        (dict(run_plan="drop"), "lacks a resolved run plan"),
        (
            dict(run_plan={"treated_only": True, "moderation_variant": False}),
            "treated-only companion",
        ),
        (
            dict(run_plan={"treated_only": False, "moderation_variant": True}),
            "moderation variant",
        ),
        (dict(sampling={**_SAMPLING, "draws": 39}), "do not match config"),
        (
            dict(constant_vars=("phase_idx",)),
            "lacks on_intervention or phase_idx",
        ),
    ],
    ids=[
        "unregistered-model",
        "wrong-kind",
        "missing-beta-trt",
        "missing-run-plan",
        "treated-only",
        "moderation-variant",
        "draws-mismatch",
        "missing-treatment-data",
    ],
)
def test_gf_reference_rejects_identity_mismatches(tmp_path, mutate, match):
    requested = mutate.get("model_id", "lrp-rli-gf-001")
    build_kwargs = {
        k: v
        for k, v in mutate.items()
        if k not in ("sampling", "run_plan", "model_id")
    }
    run_plan = mutate.get("run_plan")
    if isinstance(run_plan, dict):
        build_kwargs["run_plan"] = run_plan
    d = _fake_primary(tmp_path, model_id=requested, **build_kwargs)
    cfg = json.loads((d / "config.json").read_text())
    if run_plan == "drop":
        del cfg["resolved_run_plan"]
    if "sampling" in mutate:
        cfg["sampling"] = mutate["sampling"]
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_primary_gf_reference(d, requested, config_name="reporting")


def test_gf_reference_rejects_wrong_directory_identity(tmp_path):
    d = _fake_primary(tmp_path, model_id="lrp-rli-gf-009", outcome="TR")
    with pytest.raises(ValueError, match="primary model mismatch"):
        load_primary_gf_reference(d, "lrp-rli-gf-001", config_name="reporting")


def test_gf_reference_requires_artefacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_primary_gf_reference(
            tmp_path / "nope", "lrp-rli-gf-001", config_name="reporting"
        )


# --- the gain-factor sampling contract -----------------------------------------


def _reference(tmp_path, **kwargs):
    return load_primary_gf_reference(
        _fake_primary(tmp_path, **kwargs),
        kwargs.get("model_id", "lrp-rli-gf-001"),
        config_name="reporting",
    )


def test_gf_contract_adopts_primary_target_accept(tmp_path):
    """A primary with a registered target_accept override is adopted, not
    refused (mirrors the did contract)."""
    ref = _reference(tmp_path, sampling={**_SAMPLING, "target_accept": 0.97})
    sampling = SimpleNamespace(draws=40, tune=10, chains=2, target_accept=0.95)
    gf_script.assert_gf_sampling_contract(sampling, ref, config="reporting")


def test_gf_contract_rejects_draw_budget_mismatch(tmp_path):
    ref = _reference(tmp_path)
    sampling = SimpleNamespace(draws=39, tune=10, chains=2, target_accept=0.95)
    with pytest.raises(RuntimeError, match="does not match"):
        gf_script.assert_gf_sampling_contract(sampling, ref, config="reporting")


def test_gf_contract_rejects_config_mismatch(tmp_path):
    ref = _reference(tmp_path)
    sampling = SimpleNamespace(draws=40, tune=10, chains=2, target_accept=0.95)
    with pytest.raises(RuntimeError, match="does not match"):
        gf_script.assert_gf_sampling_contract(sampling, ref, config="dev")


# --- grid selection -------------------------------------------------------------


def test_grid_for_follows_the_outcome_tau_tier():
    proximal = SimpleNamespace(outcome_symbol="P")
    distal = SimpleNamespace(outcome_symbol="R")
    assert gf_script._grid_for(proximal) == STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS
    assert gf_script._grid_for(distal) == STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS


# --- attach: model-keyed bundles and the same-outcome sibling refusal -----------


# Attach re-runs the convergence gate on each cell trace (#584 finding 3), so a
# fixture cell must carry enough independent draws — and the ``energy`` BFMI
# needs — to pass it, rather than declaring ``converged=True`` over five draws
# from one chain.
_CELL_SAMPLING = {
    "draws": 400,
    "tune": 100,
    "chains": 4,
    "cores": 1,
    "target_accept": 0.95,
    "random_seed": 1,
    "nuts_sampler": "nutpie",
}


class _TraceLike:
    """Just enough of an InferenceData for persist_sensitivity_trace."""

    def __init__(self, groups: dict[str, xr.Dataset]):
        self._tree = xr.DataTree.from_dict(groups)

    def to_netcdf(self, path):
        self._tree.to_netcdf(path)


def _rows_for(reference, sweep_dir: Path) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(3)
    shape = (_CELL_SAMPLING["chains"], _CELL_SAMPLING["draws"])
    for token, sigma in (("0p25", 0.25), ("0p5", 0.5), ("0p75", 0.75)):
        tau_draws = rng.normal(0.2 + sigma / 10, 0.05, size=shape)
        posterior = xr.Dataset(
            {
                "alpha": (("chain", "draw"), rng.normal(size=shape)),
                "beta_trt": (("chain", "draw"), tau_draws),
            },
            coords={"chain": np.arange(shape[0]), "draw": np.arange(shape[1])},
        )
        provenance = {
            "schema_version": 1,
            "model_kind": "gain_factors",
            "config": "reporting",
            "outcome": reference.outcome,
            "model_id": reference.model_id,
            "focal_term": "beta_trt",
            "sensitivity_axis": "tau",
            "tau_sigma": sigma,
            "primary_model_id": reference.model_id,
            "primary_config_sha256": reference.config_sha256,
            "primary_trace_sha256": reference.trace_sha256,
            "free_variables": ["alpha", "beta_trt"],
            "sampling": dict(_CELL_SAMPLING),
        }
        posterior.attrs[STANDARD_SENSITIVITY_PROVENANCE_ATTR] = json.dumps(
            provenance, sort_keys=True, separators=(",", ":")
        )
        sample_stats = xr.Dataset(
            {
                "diverging": (("chain", "draw"), np.zeros(shape, dtype=bool)),
                "energy": (("chain", "draw"), rng.normal(size=shape) * 5.0 + 100.0),
            },
            coords={"chain": np.arange(shape[0]), "draw": np.arange(shape[1])},
        )
        token_id = reference.model_id.removeprefix("lrp-rli-")
        trace_file, digest = persist_sensitivity_trace(
            _TraceLike({"posterior": posterior, "sample_stats": sample_stats}),
            sensitivity_dir=sweep_dir,
            semantic_file=Path("traces")
            / "gf-reporting"
            / f"trace_{token_id}_tau-{token}.nc",
        )
        row = dict.fromkeys(_STANDARD_REQUIRED_COLUMNS, "")
        row.update(
            config="reporting",
            outcome=reference.outcome,
            n_trials=79,
            sensitivity_axis="tau",
            tau_sigma=sigma,
            converged=True,
            tau_logit_mean=float(tau_draws.mean()),
            n_divergences=0,
            sampling_draws=_CELL_SAMPLING["draws"],
            sampling_tune=_CELL_SAMPLING["tune"],
            sampling_chains=_CELL_SAMPLING["chains"],
            sampling_cores=_CELL_SAMPLING["cores"],
            sampling_target_accept=_CELL_SAMPLING["target_accept"],
            sampling_random_seed=_CELL_SAMPLING["random_seed"],
            sampling_nuts_sampler="nutpie",
            primary_model_id=reference.model_id,
            primary_config_sha256=reference.config_sha256,
            primary_trace_sha256=reference.trace_sha256,
            trace_file=trace_file.as_posix(),
            trace_sha256=digest,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_gf_attach_installs_model_keyed_bundle(tmp_path):
    primary = _fake_primary(tmp_path)
    reference = load_primary_gf_reference(
        primary, "lrp-rli-gf-001", config_name="reporting"
    )
    sweep = tmp_path / "sweep"
    rows = _rows_for(reference, sweep)
    destination = attach_outcome_bundle(
        rows,
        outcome="W",
        primary_dir=primary,
        sensitivity_dir=sweep,
        reference=reference,
    )
    assert destination == primary / STANDARD_SENSITIVITY_FILENAME
    installed = pd.read_csv(destination)
    assert set(installed["primary_model_id"]) == {"lrp-rli-gf-001"}
    for _, row in installed.iterrows():
        target = primary / str(row["trace_file"])
        assert target.is_file() and "/" not in str(row["trace_file"])
        assert sha256_file(target) == str(row["trace_sha256"])


def test_gf_attach_refuses_sibling_tr_model_rows(tmp_path):
    """gf-009 and gf-012 both fit outcome TR: rows sha-bound and labelled for
    the sibling must not attach to this primary, and nothing may be left
    behind."""
    primary = _fake_primary(tmp_path, model_id="lrp-rli-gf-009", outcome="TR")
    reference = load_primary_gf_reference(
        primary, "lrp-rli-gf-009", config_name="reporting"
    )
    sweep = tmp_path / "sweep"
    rows = _rows_for(reference, sweep)
    rows["primary_model_id"] = "lrp-rli-gf-012"
    with pytest.raises(RuntimeError, match="different primary model"):
        attach_outcome_bundle(
            rows,
            outcome="TR",
            primary_dir=primary,
            sensitivity_dir=sweep,
            reference=reference,
        )
    assert not (primary / STANDARD_SENSITIVITY_FILENAME).exists()
    assert not list(primary.glob("*.staging"))
    assert not list(primary.glob("trace_gf-009_tau-*.nc"))
