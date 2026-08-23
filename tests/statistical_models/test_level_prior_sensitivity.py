# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for the level-factor treatment-prior sweep runner (#389 crit. 6).

The runner's manifest is release-gate evidence — it can lift a withheld causal
result — so its bindings are exercised the way the standard/floor runners' are:
primary identity and sampling mismatches must refuse before fitting, and the
attach step must never expose a manifest whose traces are missing, corrupt,
unconverged, sign-unstable or bound to a different primary.
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
    STANDARD_SENSITIVITY_FILENAME,
    load_primary_level_reference,
    sha256_file,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "level_factors_prior_sensitivity.py"
)
_SPEC = spec_from_file_location("_lrp_level_prior_sensitivity", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
level_script = module_from_spec(_SPEC)
_SPEC.loader.exec_module(level_script)

_SAMPLING = {
    "draws": 40,
    "tune": 10,
    "chains": 2,
    "target_accept": 0.95,
    "random_seed": 20260701,
}
_N_OBS = 24  # 6 children x 4 waves


def _fake_primary(
    root: Path,
    *,
    outcome: str = "W",
    model_id: str = "lrp-rli-lf-001",
    kind: str = "level_factors",
    posterior_vars: tuple[str, ...] = ("alpha", "b_grp_time"),
    sampling: dict | None = None,
    focal_term: str | None = "b_grp_time[1]",
) -> Path:
    """A minimal on-disk primary fit satisfying load_primary_level_reference."""
    sampling = dict(_SAMPLING if sampling is None else sampling)
    directory = root / f"{model_id}-reporting"
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    chains, draws = sampling["chains"], sampling["draws"]
    values: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    if "alpha" in posterior_vars:
        values["alpha"] = (("chain", "draw"), rng.normal(size=(chains, draws)))
    if "b_grp_time" in posterior_vars:
        values["b_grp_time"] = (
            ("chain", "draw", "phase"),
            rng.normal(size=(chains, draws, 4)),
        )
    posterior = xr.Dataset(
        values,
        coords={
            "chain": np.arange(chains),
            "draw": np.arange(draws),
            "phase": np.arange(4),
        },
    )
    G = np.tile(np.repeat([1.0, 0.0], 2), _N_OBS // 4)
    constant_data = xr.Dataset(
        {"G": (("obs_id",), G)}, coords={"obs_id": np.arange(_N_OBS)}
    )
    xr.DataTree.from_dict(
        {"posterior": posterior, "constant_data": constant_data}
    ).to_netcdf(directory / "trace.nc")
    config = {
        "model_id": model_id,
        "outcome_symbol": outcome,
        "kind": kind,
        "data_sha256": "a" * 64,
        "n_obs": _N_OBS,
        "sampling": sampling,
        # Every stored level fit records its resolved plan; the attach step binds
        # each cell's focal term to it (#584 finding 3).
        "resolved_run_plan": {"focal_term": focal_term} if focal_term else {},
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


# --- load_primary_level_reference --------------------------------------------


def test_level_reference_loads_and_binds(tmp_path):
    d = _fake_primary(tmp_path)
    ref = load_primary_level_reference(d, "W", config_name="reporting")
    assert ref.model_id == "lrp-rli-lf-001"
    assert ref.config_sha256 == sha256_file(d / "config.json")
    assert ref.trace_sha256 == sha256_file(d / "trace.nc")
    assert ref.n_intervention > 0 and ref.n_control > 0


@pytest.mark.parametrize(
    "mutate,match",
    [
        (dict(model_id="lrp-rli-lf-999"), "primary model mismatch"),
        (dict(kind="itt"), "primary kind mismatch"),
        (dict(posterior_vars=("alpha",)), "lacks alpha or b_grp_time"),
        (dict(sampling={**_SAMPLING, "draws": 39}), "do not match config"),
    ],
    ids=["wrong-model-id", "wrong-kind", "missing-focal-vector", "draws-mismatch"],
)
def test_level_reference_rejects_identity_mismatches(tmp_path, mutate, match):
    d = _fake_primary(tmp_path, **{k: v for k, v in mutate.items() if k != "sampling"})
    if "sampling" in mutate:
        # Rewrite the config's sampling metadata so it disagrees with the trace.
        cfg = json.loads((d / "config.json").read_text())
        cfg["sampling"] = mutate["sampling"]
        (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_primary_level_reference(d, "W", config_name="reporting")


def test_level_reference_requires_artefacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_primary_level_reference(tmp_path / "nope", "W", config_name="reporting")


# --- the sampling contract ----------------------------------------------------


def _reference(tmp_path):
    return load_primary_level_reference(
        _fake_primary(tmp_path), "W", config_name="reporting"
    )


def test_sampling_contract_accepts_matching_preset(tmp_path):
    ref = _reference(tmp_path)
    sampling = SimpleNamespace(draws=40, tune=10, chains=2, target_accept=0.95)
    level_script.assert_primary_sampling_contract(sampling, ref, config="reporting")


def test_sampling_contract_rejects_target_accept_override(tmp_path):
    """A primary fitted with a --target-accept override must not be swept with
    the default preset and attached anyway (review comment on #488)."""
    ref = _reference(tmp_path)
    sampling = SimpleNamespace(draws=40, tune=10, chains=2, target_accept=0.99)
    with pytest.raises(RuntimeError, match="does not match"):
        level_script.assert_primary_sampling_contract(sampling, ref, config="reporting")


def test_sampling_contract_rejects_config_mismatch(tmp_path):
    ref = _reference(tmp_path)
    sampling = SimpleNamespace(draws=40, tune=10, chains=2, target_accept=0.95)
    with pytest.raises(RuntimeError, match="does not match"):
        level_script.assert_primary_sampling_contract(sampling, ref, config="dev")


# --- attach_outcome_bundle ----------------------------------------------------


# A cell trace must now survive the convergence gate being re-run on it (#584
# finding 3), so the fixture samples enough independent draws to clear the ESS
# threshold instead of declaring ``converged=True`` over five draws from one
# chain — which is exactly the fail-open the finding describes.
_CELL_SAMPLING = {
    "draws": 400,
    "tune": 100,
    "chains": 4,
    "cores": 1,
    "target_accept": 0.95,
    "random_seed": 1,
    "nuts_sampler": "nutpie",
}


def _cell_trace(
    sweep_dir: Path,
    token: str,
    *,
    provenance: dict,
    focal_offset: float = 0.0,
    omit_energy: bool = False,
) -> tuple[str, str, dict]:
    """One digest-suffixed, provenance-stamped cell trace (#489 review): the
    attach step now opens each trace and verifies it identifies itself as this
    cell of this primary's sweep, so an arbitrary NetCDF no longer attaches.

    Returns the summary the manifest row must carry as well, so the fixture's
    row is *derived from* its trace rather than asserted beside it."""
    from language_reading_predictors.statistical_models.diagnostics import (
        subfit_convergence,
    )
    from language_reading_predictors.statistical_models.sensitivity import (
        STANDARD_SENSITIVITY_PROVENANCE_ATTR,
    )

    traces = sweep_dir / "traces" / "level-reporting"
    traces.mkdir(parents=True, exist_ok=True)
    tmp = traces / f".tmp-{token}.nc"
    rng = np.random.default_rng(3)
    shape = (_CELL_SAMPLING["chains"], _CELL_SAMPLING["draws"])
    b_grp = rng.normal(size=(*shape, 4)) * 0.1
    # A same-sign focal element across the grid: the bundle is evidence of prior
    # stability, and attach refuses a sign flip.
    b_grp[..., 1] += 0.4 + focal_offset
    posterior = xr.Dataset(
        {
            "alpha_offset": (("chain", "draw"), rng.normal(size=shape)),
            "b_grp_time": (("chain", "draw", "phase"), b_grp),
        },
        coords={
            "chain": np.arange(shape[0]),
            "draw": np.arange(shape[1]),
            "phase": ["t1", "t2", "t3", "t4"],
        },
    )
    posterior.attrs[STANDARD_SENSITIVITY_PROVENANCE_ATTR] = json.dumps(
        provenance, sort_keys=True, separators=(",", ":")
    )
    stats: dict = {"diverging": (("chain", "draw"), np.zeros(shape, dtype=bool))}
    if not omit_energy:
        stats["energy"] = (("chain", "draw"), rng.normal(size=shape) * 5.0 + 100.0)
    sample_stats = xr.Dataset(
        stats, coords={"chain": np.arange(shape[0]), "draw": np.arange(shape[1])}
    )
    tree = xr.DataTree.from_dict({"posterior": posterior, "sample_stats": sample_stats})
    tree.to_netcdf(tmp)
    digest = sha256_file(tmp)
    final = traces / f"trace_W_tau-{token}-{digest[:12]}.nc"
    tmp.rename(final)
    focal = b_grp[..., 1].reshape(-1)
    convergence = subfit_convergence(
        tree, label=token, var_names=["alpha_offset", "b_grp_time"]
    )
    summary = {
        "tau_logit_mean": float(focal.mean()),
        "tau_logit_lo": float(np.quantile(focal, 0.055)),
        "tau_logit_hi": float(np.quantile(focal, 0.945)),
        "pd": float(np.mean(focal > 0.0)),
        "converged": convergence["converged"],
        "max_rhat": convergence["max_rhat"],
        "min_ess": convergence["min_ess"],
        "min_bfmi": convergence["min_bfmi"],
        "n_divergences": convergence["n_divergences"],
    }
    return final.relative_to(sweep_dir).as_posix(), digest, summary


def _rows(
    primary: Path,
    sweep_dir: Path,
    *,
    focal_term: str = "b_grp_time[1]",
    omit_energy: bool = False,
    **overrides,
) -> pd.DataFrame:
    ref = load_primary_level_reference(primary, "W", config_name="reporting")
    rows = []
    for token, sigma in (("0p25", 0.25), ("0p5", 0.5), ("0p75", 0.75)):
        provenance = {
            "schema_version": 1,
            "model_kind": "level_factors",
            "config": "reporting",
            "outcome": "W",
            "focal_term": focal_term,
            "sensitivity_axis": "tau",
            "tau_sigma": sigma,
            "primary_model_id": ref.model_id,
            "primary_config_sha256": ref.config_sha256,
            "primary_trace_sha256": ref.trace_sha256,
            "free_variables": ["alpha_offset", "b_grp_time"],
            "sampling": dict(_CELL_SAMPLING),
        }
        trace_file, digest, summary = _cell_trace(
            sweep_dir,
            token,
            provenance=provenance,
            focal_offset=sigma / 10.0,
            omit_energy=omit_energy,
        )
        row = dict.fromkeys(_STANDARD_REQUIRED_COLUMNS, "")
        row.update(
            config="reporting",
            outcome="W",
            n_trials=79,
            sensitivity_axis="tau",
            tau_sigma=sigma,
            **summary,
            sampling_draws=_CELL_SAMPLING["draws"],
            sampling_tune=_CELL_SAMPLING["tune"],
            sampling_chains=_CELL_SAMPLING["chains"],
            sampling_cores=_CELL_SAMPLING["cores"],
            sampling_target_accept=_CELL_SAMPLING["target_accept"],
            sampling_random_seed=_CELL_SAMPLING["random_seed"],
            sampling_nuts_sampler="nutpie",
            primary_model_id=ref.model_id,
            primary_config_sha256=ref.config_sha256,
            primary_trace_sha256=ref.trace_sha256,
            trace_file=trace_file,
            trace_sha256=digest,
        )
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def _attach(primary, sweep_dir, rows):
    ref = load_primary_level_reference(primary, "W", config_name="reporting")
    return level_script.attach_outcome_bundle(
        rows, outcome="W", primary_dir=primary, sensitivity_dir=sweep_dir, reference=ref
    )


def test_attach_installs_trace_backed_bundle(tmp_path):
    primary = _fake_primary(tmp_path)
    sweep = tmp_path / "sweep"
    rows = _rows(primary, sweep)
    destination = _attach(primary, sweep, rows)
    assert destination == primary / STANDARD_SENSITIVITY_FILENAME
    installed = pd.read_csv(destination)
    # trace_file rewritten to the installed digest-suffixed basename, and each
    # installed trace's content matches its recorded sha256.
    for _, row in installed.iterrows():
        target = primary / str(row["trace_file"])
        assert target.is_file() and "/" not in str(row["trace_file"])
        assert sha256_file(target) == str(row["trace_sha256"])


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("missing_trace", "missing cell trace"),
        ("tampered_trace", "does not match its recorded sha256"),
        ("unconverged", "failed the convergence gate"),
        ("sign_flip", "changes sign"),
        ("stale_primary", "different primary"),
        ("wrong_model", "different primary model"),
        ("missing_column", "required columns"),
    ],
)
def test_attach_refuses_and_rolls_back(tmp_path, corruption, match):
    primary = _fake_primary(tmp_path)
    sweep = tmp_path / "sweep"
    rows = _rows(primary, sweep)
    if corruption == "missing_trace":
        (sweep / rows.at[0, "trace_file"]).unlink()
    elif corruption == "tampered_trace":
        path = sweep / rows.at[0, "trace_file"]
        path.write_bytes(path.read_bytes() + b"x")
    elif corruption == "unconverged":
        rows.at[1, "converged"] = False
    elif corruption == "sign_flip":
        rows.at[2, "tau_logit_mean"] = -0.4
    elif corruption == "stale_primary":
        rows["primary_trace_sha256"] = "b" * 64
    elif corruption == "wrong_model":
        # Rows from a sibling fit of the same outcome (the did family has two
        # swept W fits) must not attach as this primary's evidence.
        rows["primary_model_id"] = "lrp-rli-lf-999"
    elif corruption == "missing_column":
        rows = rows.drop(columns=["kappa_median"])
    with pytest.raises(RuntimeError, match=match):
        _attach(primary, sweep, rows)
    # Nothing exposed: no manifest, no staging file, no installed traces.
    assert not (primary / STANDARD_SENSITIVITY_FILENAME).exists()
    assert not list(primary.glob("*.staging"))
    assert not list(primary.glob("trace_W_tau-*.nc"))


# --- trace-derived evidence (#584 finding 3) ----------------------------------


def test_attach_rejects_a_tampered_focal_summary(tmp_path):
    """The reproduction in the review: editing only the CSV's focal mean used to
    attach, because an *indexed* focal term was never recomputed from its trace."""
    primary = _fake_primary(tmp_path)
    sweep = tmp_path / "sweep"
    rows = _rows(primary, sweep)
    # Same sign as the honest value, so the sign-stability check cannot catch it.
    rows["tau_logit_mean"] = 999.0
    rows["tau_logit_hi"] = 1000.0
    with pytest.raises(RuntimeError, match="does not reproduce its row's focal"):
        _attach(primary, sweep, rows)
    assert not (primary / STANDARD_SENSITIVITY_FILENAME).exists()


def test_attach_rejects_an_interval_that_does_not_bracket_its_mean(tmp_path):
    primary = _fake_primary(tmp_path)
    sweep = tmp_path / "sweep"
    rows = _rows(primary, sweep)
    rows["tau_logit_hi"] = rows["tau_logit_mean"].astype(float) - 0.01
    with pytest.raises(RuntimeError, match="does not bracket the focal mean"):
        _attach(primary, sweep, rows)


def test_attach_rejects_a_tampered_direction_probability(tmp_path):
    primary = _fake_primary(tmp_path)
    sweep = tmp_path / "sweep"
    rows = _rows(primary, sweep)
    rows["pd"] = 0.999
    with pytest.raises(RuntimeError, match="direction probability"):
        _attach(primary, sweep, rows)


def test_attach_rejects_a_false_convergence_claim(tmp_path):
    """A cell whose trace cannot pass the gate is not evidence, whatever its row
    claims. The traces here carry no ``energy``, so BFMI is uncheckable — the
    row's ``converged=True`` is then a claim about a trace that never supported
    it, which is the fail-open the review reproduced."""
    primary = _fake_primary(tmp_path)
    sweep = tmp_path / "sweep"
    rows = _rows(primary, sweep, omit_energy=True, converged=True)
    with pytest.raises(RuntimeError, match="convergence gate"):
        _attach(primary, sweep, rows)


def test_attach_rejects_a_mistyped_convergence_number(tmp_path):
    primary = _fake_primary(tmp_path)
    sweep = tmp_path / "sweep"
    rows = _rows(primary, sweep)
    rows["min_ess"] = 40_000.0
    with pytest.raises(RuntimeError, match="min_ess"):
        _attach(primary, sweep, rows)


def test_attach_rejects_a_focal_element_the_vector_does_not_carry(tmp_path):
    """A coordinate that names no wave is a mislabelled sweep, not evidence."""
    primary = _fake_primary(tmp_path, focal_term="b_grp_time[t9]")
    sweep = tmp_path / "sweep"
    rows = _rows(primary, sweep, focal_term="b_grp_time[t9]")
    with pytest.raises(RuntimeError, match="not a coordinate"):
        _attach(primary, sweep, rows)


def test_attach_rejects_a_sweep_of_a_superseded_parameterisation(tmp_path):
    """#552 renamed the focal term; a pre-#552 grid must not certify a
    t1-centred primary just because the hashes still line up."""
    primary = _fake_primary(tmp_path, focal_term="d_grp_time[t2]")
    sweep = tmp_path / "sweep"
    rows = _rows(primary, sweep)  # cells stamped with the old b_grp_time[1]
    with pytest.raises(RuntimeError, match="resolved run plan"):
        _attach(primary, sweep, rows)

