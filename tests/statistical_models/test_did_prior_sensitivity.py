# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for the did treatment-prior sweep runner (#390).

The runner's manifest is release-gate evidence — it can lift a withheld
headline — so its bindings mirror the level runner's (#488 review), with two
did-specific twists exercised here: the sweep set is keyed by **model id**
(two withheld fits share outcome W, so outcome identity alone cannot bind a
bundle to its primary), and cells *adopt* the primary's recorded
``target_accept`` (did-007's registered spec overrides the preset) rather than
refusing it. The shared attach/rollback discipline itself is covered by
``test_level_prior_sensitivity.py``; this file adds the did loader, the did
sampling contract, and the same-outcome sibling-model refusal.
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
    attach_outcome_bundle,
    did_focal_term,
    load_primary_did_reference,
    persist_sensitivity_trace,
    sha256_file,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "did_prior_sensitivity.py"
)
_SPEC = spec_from_file_location("_lrp_did_prior_sensitivity", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
did_script = module_from_spec(_SPEC)
_SPEC.loader.exec_module(did_script)

_SAMPLING = {
    "draws": 40,
    "tune": 10,
    "chains": 2,
    "target_accept": 0.97,  # a did-007-style registered override
    "random_seed": 20260701,
}
_N_OBS = 24


def _fake_primary(
    root: Path,
    *,
    model_id: str = "lrp-rli-did-001",
    outcome: str = "W",
    kind: str = "did",
    run_plan: dict | None = None,
    posterior_vars: tuple[str, ...] = ("alpha", "tau_t2"),
    sampling: dict | None = None,
) -> Path:
    """A minimal on-disk primary fit satisfying load_primary_did_reference."""
    sampling = dict(_SAMPLING if sampling is None else sampling)
    if run_plan is None:
        run_plan = {"dose": False, "period_varying": False}
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
        "resolved_run_plan": run_plan,
        "data_sha256": "a" * 64,
        "n_obs": _N_OBS,
        "sampling": sampling,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


# --- focal-term derivation ----------------------------------------------------


@pytest.mark.parametrize(
    "plan,expected",
    [
        ({"dose": False, "period_varying": False}, "tau_t2"),
        ({"dose": True, "period_varying": False}, "beta_dose"),
        ({"dose": True, "period_varying": True}, "mu_dose"),
    ],
)
def test_did_focal_term_mirrors_effect_term(plan, expected):
    assert did_focal_term(plan) == expected


# --- load_primary_did_reference -----------------------------------------------


def test_did_reference_loads_and_binds(tmp_path):
    d = _fake_primary(tmp_path)
    ref = load_primary_did_reference(d, "lrp-rli-did-001", config_name="reporting")
    assert ref.model_id == "lrp-rli-did-001"
    assert ref.outcome == "W"
    assert ref.config_sha256 == sha256_file(d / "config.json")
    assert ref.trace_sha256 == sha256_file(d / "trace.nc")
    assert ref.n_intervention > 0 and ref.n_control > 0
    assert ref.sampling["target_accept"] == pytest.approx(0.97)


def test_did_reference_requires_dose_focal_term(tmp_path):
    """A did-007-style primary binds on mu_dose, not tau_t2."""
    plan = {"dose": True, "period_varying": True}
    d = _fake_primary(
        tmp_path,
        model_id="lrp-rli-did-007",
        outcome="L",
        run_plan=plan,
        posterior_vars=("alpha", "mu_dose"),
    )
    ref = load_primary_did_reference(d, "lrp-rli-did-007", config_name="reporting")
    assert ref.outcome == "L"
    # The same trace *without* the focal term must refuse.
    d2 = _fake_primary(
        tmp_path / "b",
        model_id="lrp-rli-did-007",
        outcome="L",
        run_plan=plan,
        posterior_vars=("alpha", "tau_t2"),
    )
    with pytest.raises(ValueError, match="focal term 'mu_dose'"):
        load_primary_did_reference(d2, "lrp-rli-did-007", config_name="reporting")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (dict(model_id="lrp-rli-did-999"), "unsupported did sensitivity model"),
        (dict(kind="itt"), "primary kind mismatch"),
        (dict(posterior_vars=("alpha",)), "lacks alpha or the focal term"),
        (dict(run_plan="drop"), "lacks a resolved run plan"),
        (dict(sampling={**_SAMPLING, "draws": 39}), "do not match config"),
    ],
    ids=[
        "unregistered-model",
        "wrong-kind",
        "missing-focal-term",
        "missing-run-plan",
        "draws-mismatch",
    ],
)
def test_did_reference_rejects_identity_mismatches(tmp_path, mutate, match):
    requested = mutate.get("model_id", "lrp-rli-did-001")
    build_kwargs = {
        k: v for k, v in mutate.items() if k not in ("sampling", "run_plan", "model_id")
    }
    d = _fake_primary(tmp_path, model_id=requested, **build_kwargs)
    cfg = json.loads((d / "config.json").read_text())
    if mutate.get("run_plan") == "drop":
        del cfg["resolved_run_plan"]
    if "sampling" in mutate:
        # Rewrite the config's sampling metadata so it disagrees with the trace.
        cfg["sampling"] = mutate["sampling"]
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_primary_did_reference(d, requested, config_name="reporting")


def test_did_reference_rejects_wrong_directory_identity(tmp_path):
    """A registered id must still match the directory's own config."""
    d = _fake_primary(tmp_path, model_id="lrp-rli-did-013")
    with pytest.raises(ValueError, match="primary model mismatch"):
        load_primary_did_reference(d, "lrp-rli-did-001", config_name="reporting")


def test_did_reference_requires_artefacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_primary_did_reference(
            tmp_path / "nope", "lrp-rli-did-001", config_name="reporting"
        )


# --- the did sampling contract ------------------------------------------------


def _reference(tmp_path, **kwargs):
    return load_primary_did_reference(
        _fake_primary(tmp_path, **kwargs),
        kwargs.get("model_id", "lrp-rli-did-001"),
        config_name="reporting",
    )


def test_did_contract_adopts_primary_target_accept(tmp_path):
    """The preset's 0.95 differs from the primary's registered 0.97 override;
    the did contract accepts (cells then sample at the primary's value)."""
    ref = _reference(tmp_path)
    sampling = SimpleNamespace(draws=40, tune=10, chains=2, target_accept=0.95)
    did_script.assert_did_sampling_contract(sampling, ref, config="reporting")


def test_did_contract_rejects_draw_budget_mismatch(tmp_path):
    ref = _reference(tmp_path)
    sampling = SimpleNamespace(draws=39, tune=10, chains=2, target_accept=0.97)
    with pytest.raises(RuntimeError, match="does not match"):
        did_script.assert_did_sampling_contract(sampling, ref, config="reporting")


def test_did_contract_rejects_config_mismatch(tmp_path):
    ref = _reference(tmp_path)
    sampling = SimpleNamespace(draws=40, tune=10, chains=2, target_accept=0.97)
    with pytest.raises(RuntimeError, match="does not match"):
        did_script.assert_did_sampling_contract(sampling, ref, config="dev")


# --- attach: the same-outcome sibling-model refusal ---------------------------


# Attach re-runs the convergence gate on each cell trace (#584 finding 3), so a
# fixture cell must carry enough independent draws — and the ``energy`` BFMI
# needs — to pass it, rather than declaring ``converged=True`` over five draws
# from one chain.
_CELL_SAMPLING = {
    "draws": 400,
    "tune": 100,
    "chains": 4,
    "cores": 1,
    "target_accept": 0.97,
    "random_seed": 1,
    "nuts_sampler": "nutpie",
}


def _rows_for(
    reference,
    sweep_dir: Path,
    *,
    provenance_overrides: dict | None = None,
    divergences: int = 0,
) -> pd.DataFrame:
    from language_reading_predictors.statistical_models.sensitivity import (
        STANDARD_SENSITIVITY_PROVENANCE_ATTR,
    )

    rows = []
    rng = np.random.default_rng(3)
    shape = (_CELL_SAMPLING["chains"], _CELL_SAMPLING["draws"])
    for token, sigma in (("0p25", 0.25), ("0p5", 0.5), ("0p75", 0.75)):
        tau_draws = rng.normal(0.2 + sigma / 10, 0.05, size=shape)
        posterior = xr.Dataset(
            {
                "alpha_offset": (("chain", "draw"), rng.normal(size=shape)),
                "tau_t2": (("chain", "draw"), tau_draws),
            },
            coords={"chain": np.arange(shape[0]), "draw": np.arange(shape[1])},
        )
        provenance = {
            "schema_version": 1,
            "model_kind": "did",
            "config": "reporting",
            "outcome": reference.outcome,
            "model_id": reference.model_id,
            "focal_term": "tau_t2",
            "sensitivity_axis": "tau",
            "tau_sigma": sigma,
            "primary_model_id": reference.model_id,
            "primary_config_sha256": reference.config_sha256,
            "primary_trace_sha256": reference.trace_sha256,
            # #576 finding 6: cells are bound to the primary's fitted equation,
            # not only to its identity and row counts.
            "primary_run_plan_sha256": reference.run_plan_digest,
            "free_variables": ["alpha_offset", "tau_t2"],
            "sampling": dict(_CELL_SAMPLING),
        }
        provenance.update(provenance_overrides or {})
        posterior.attrs[STANDARD_SENSITIVITY_PROVENANCE_ATTR] = json.dumps(
            provenance, sort_keys=True, separators=(",", ":")
        )
        diverging = np.zeros(shape, dtype=bool)
        diverging.flat[:divergences] = True
        sample_stats = xr.Dataset(
            {
                "diverging": (("chain", "draw"), diverging),
                "energy": (("chain", "draw"), rng.normal(size=shape) * 5.0 + 100.0),
            },
            coords={"chain": np.arange(shape[0]), "draw": np.arange(shape[1])},
        )
        trace_file, digest = persist_sensitivity_trace(
            _TraceLike({"posterior": posterior, "sample_stats": sample_stats}),
            sensitivity_dir=sweep_dir,
            semantic_file=Path("traces")
            / "did-reporting"
            / f"trace_did-001_tau-{token}.nc",
        )
        row = dict.fromkeys(_STANDARD_REQUIRED_COLUMNS, "")
        row.update(
            config="reporting",
            outcome=reference.outcome,
            n_trials=79,
            sensitivity_axis="tau",
            tau_sigma=sigma,
            converged=True,
            # The attach validator recomputes a bare-name focal summary from
            # the trace itself, so the row must carry the true draw mean.
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
            primary_run_plan_sha256=reference.run_plan_digest,
            trace_file=trace_file.as_posix(),
            trace_sha256=digest,
        )
        rows.append(row)
    return pd.DataFrame(rows)


class _TraceLike:
    """Just enough of an InferenceData for persist_sensitivity_trace."""

    def __init__(self, groups: dict[str, xr.Dataset]):
        self._tree = xr.DataTree.from_dict(groups)

    def to_netcdf(self, path):
        self._tree.to_netcdf(path)


def test_did_attach_installs_model_keyed_bundle(tmp_path):
    primary = _fake_primary(tmp_path)
    reference = load_primary_did_reference(
        primary, "lrp-rli-did-001", config_name="reporting"
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
    assert set(installed["primary_model_id"]) == {"lrp-rli-did-001"}
    for _, row in installed.iterrows():
        target = primary / str(row["trace_file"])
        assert target.is_file() and "/" not in str(row["trace_file"])
        assert sha256_file(target) == str(row["trace_sha256"])


def test_did_attach_refuses_sibling_w_model_rows(tmp_path):
    """did-001 and did-013 both fit outcome W: rows sha-bound and labelled for
    the sibling must not attach to this primary, and nothing may be left
    behind."""
    primary = _fake_primary(tmp_path)
    reference = load_primary_did_reference(
        primary, "lrp-rli-did-001", config_name="reporting"
    )
    sweep = tmp_path / "sweep"
    rows = _rows_for(reference, sweep)
    rows["primary_model_id"] = "lrp-rli-did-013"
    with pytest.raises(RuntimeError, match="different primary model"):
        attach_outcome_bundle(
            rows,
            outcome="W",
            primary_dir=primary,
            sensitivity_dir=sweep,
            reference=reference,
        )
    assert not (primary / STANDARD_SENSITIVITY_FILENAME).exists()
    assert not list(primary.glob("*.staging"))
    assert not list(primary.glob("trace_did-001_tau-*.nc"))


# --- deep trace validation (#489 review) --------------------------------------


def test_did_attach_refuses_foreign_provenance_trace(tmp_path):
    """A digest-matching trace whose stamped provenance names a different cell
    (here: another outcome) is not this bundle's evidence — the sha alone only
    proves the file matches the row's own claim."""
    primary = _fake_primary(tmp_path)
    reference = load_primary_did_reference(
        primary, "lrp-rli-did-001", config_name="reporting"
    )
    sweep = tmp_path / "sweep"
    rows = _rows_for(reference, sweep, provenance_overrides={"outcome": "L"})
    with pytest.raises(RuntimeError, match="provenance outcome does not match"):
        attach_outcome_bundle(
            rows,
            outcome="W",
            primary_dir=primary,
            sensitivity_dir=sweep,
            reference=reference,
        )
    assert not (primary / STANDARD_SENSITIVITY_FILENAME).exists()
    assert not list(primary.glob("trace_did-001_tau-*.nc"))


@pytest.mark.parametrize(
    "changed_plan,label",
    [
        ({"dose": False, "period_varying": False, "use_age": False}, "age adjustment"),
        (
            {"dose": False, "period_varying": False, "use_intercept_anchor": False},
            "intercept anchor",
        ),
        (
            {"dose": False, "period_varying": False, "likelihood": "bernoulli_offfloor"},
            "likelihood",
        ),
        (
            {"dose": False, "period_varying": False, "tau_t2_prior_sigma": 1.0},
            "focal prior width",
        ),
    ],
    ids=["use_age", "intercept-anchor", "likelihood", "tau-prior-width"],
)
def test_did_attach_refuses_a_sweep_of_a_different_run_plan(
    tmp_path, changed_plan, label
):
    """#576 finding 6: a non-swept plan field must invalidate the bundle.

    The sweep is generated against one registered declaration and the primary was
    fitted under another. Every pre-#576 binding still matches — same model id, same
    outcome, same data digest, same row count, same arm totals, same config/trace
    hashes — because none of them can see the fitted *equation*. Only the run-plan
    digest can, and without it a primary fitted under an older likelihood, intercept
    anchor, age adjustment or prior width would be released by evidence for a model
    it is not.
    """
    primary = _fake_primary(tmp_path)
    reference = load_primary_did_reference(
        primary, "lrp-rli-did-001", config_name="reporting"
    )
    sweep = tmp_path / "sweep"
    rows = _rows_for(reference, sweep)
    # The cells were swept under a plan that differs only in this one field.
    from language_reading_predictors.statistical_models.sensitivity import (
        did_run_plan_digest,
    )

    other = did_run_plan_digest(changed_plan)
    assert other != reference.run_plan_digest, label
    rows["primary_run_plan_sha256"] = other
    with pytest.raises(RuntimeError, match="primary_run_plan_sha256"):
        attach_outcome_bundle(
            rows,
            outcome="W",
            primary_dir=primary,
            sensitivity_dir=sweep,
            reference=reference,
        )
    assert not (primary / STANDARD_SENSITIVITY_FILENAME).exists()


def test_did_attach_refuses_a_sweep_with_no_run_plan_binding(tmp_path):
    """A bundle produced before run-plan binding cannot certify a bound primary."""
    primary = _fake_primary(tmp_path)
    reference = load_primary_did_reference(
        primary, "lrp-rli-did-001", config_name="reporting"
    )
    sweep = tmp_path / "sweep"
    rows = _rows_for(reference, sweep).drop(columns=["primary_run_plan_sha256"])
    with pytest.raises(RuntimeError, match="carry no primary_run_plan_sha256"):
        attach_outcome_bundle(
            rows,
            outcome="W",
            primary_dir=primary,
            sensitivity_dir=sweep,
            reference=reference,
        )


def test_did_attach_refuses_divergence_mismatch(tmp_path):
    """The trace's own sample_stats must reproduce the row's divergence count:
    a row claiming zero divergences over a diverging trace is not evidence."""
    primary = _fake_primary(tmp_path)
    reference = load_primary_did_reference(
        primary, "lrp-rli-did-001", config_name="reporting"
    )
    sweep = tmp_path / "sweep"
    rows = _rows_for(reference, sweep, divergences=2)
    with pytest.raises(RuntimeError, match="divergence count does not match"):
        attach_outcome_bundle(
            rows,
            outcome="W",
            primary_dir=primary,
            sensitivity_dir=sweep,
            reference=reference,
        )
    assert not (primary / STANDARD_SENSITIVITY_FILENAME).exists()


def test_did_attach_refuses_summary_mismatch(tmp_path):
    """A bare-name focal summary is recomputed from the trace draws; a manifest
    row whose tau_logit_mean does not reproduce is refused."""
    primary = _fake_primary(tmp_path)
    reference = load_primary_did_reference(
        primary, "lrp-rli-did-001", config_name="reporting"
    )
    sweep = tmp_path / "sweep"
    rows = _rows_for(reference, sweep)
    rows.at[0, "tau_logit_mean"] = float(rows.at[0, "tau_logit_mean"]) + 0.01
    with pytest.raises(RuntimeError, match="does not reproduce its row's focal"):
        attach_outcome_bundle(
            rows,
            outcome="W",
            primary_dir=primary,
            sensitivity_dir=sweep,
            reference=reference,
        )
    assert not (primary / STANDARD_SENSITIVITY_FILENAME).exists()


def test_did_attach_failure_preserves_previous_bundle(tmp_path):
    """A failed replacement must restore the previously published evidence, not
    destroy it (#489 review): after a valid attach, a later corrupt attempt
    leaves the old manifest and its installed traces exactly as they were."""
    primary = _fake_primary(tmp_path)
    reference = load_primary_did_reference(
        primary, "lrp-rli-did-001", config_name="reporting"
    )
    sweep_a = tmp_path / "sweep-a"
    destination = attach_outcome_bundle(
        _rows_for(reference, sweep_a),
        outcome="W",
        primary_dir=primary,
        sensitivity_dir=sweep_a,
        reference=reference,
    )
    published = destination.read_bytes()
    published_traces = sorted(primary.glob("trace_did-001_tau-*.nc"))
    assert published_traces

    sweep_b = tmp_path / "sweep-b"
    bad = _rows_for(reference, sweep_b)
    corrupt = sweep_b / str(bad.at[0, "trace_file"])
    corrupt.write_bytes(corrupt.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="does not match its recorded sha256"):
        attach_outcome_bundle(
            bad,
            outcome="W",
            primary_dir=primary,
            sensitivity_dir=sweep_b,
            reference=reference,
        )
    assert destination.read_bytes() == published
    assert sorted(primary.glob("trace_did-001_tau-*.nc")) == published_traces
    for trace in published_traces:
        assert trace.is_file()
    assert not list(primary.glob("*.staging"))
    assert not list(primary.glob("*.restore"))


# --- the runner's grid selection ----------------------------------------------


def test_grid_for_selects_focal_terms_grid():
    tau_plan = SimpleNamespace(effect_term="tau_t2", outcome_symbol="W")
    dose_plan = SimpleNamespace(effect_term="mu_dose", outcome_symbol="L")
    distal_plan = SimpleNamespace(effect_term="tau_t2", outcome_symbol="R")
    assert did_script._grid_for(tau_plan) == (0.25, 0.5, 0.75)
    assert did_script._grid_for(dose_plan) == (0.5, 1.0, 1.5)
    # A distal-tier outcome would ride the distal grid; none is currently swept
    # but the selection must not silently apply the proximal grid to one.
    assert did_script._grid_for(distal_plan) == (0.2, 0.25, 0.3, 0.5)
