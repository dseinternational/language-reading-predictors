# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import influence
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.lrp_rli_itt_012 import (
    SPEC as JOINT_SPEC,
)
from language_reading_predictors.statistical_models.lrp_rli_itt_013 import (
    SPEC as SES_SPEC,
)
from language_reading_predictors.statistical_models.lrp_rli_itt_023 import (
    SPEC as ABILITY_SPEC,
)


@pytest.fixture(autouse=True)
def _register_synthetic_bundle_contract(monkeypatch):
    """Give the synthetic 999 fixture an independent registered-model contract."""
    original = influence._registered_leave_out_free_variable_contract

    def _contract(model_id, config, *, model_output_root):
        if model_id == "lrp-rli-itt-999":
            return ("alpha", "tau")
        return original(model_id, config, model_output_root=model_output_root)

    monkeypatch.setattr(
        influence,
        "_registered_leave_out_free_variable_contract",
        _contract,
    )


def _reference_for_prepared(spec, prepared, tmp_path):
    subject_ids = np.asarray(prepared.subject_ids).astype(str)
    pareto = pd.DataFrame(
        {
            "observation_index": np.arange(prepared.n_obs),
            "subject_id": subject_ids,
            "pareto_k": np.r_[0.8, np.repeat(0.1, prepared.n_obs - 1)],
            "good_k_threshold": 0.7,
        }
    )
    return influence.InfluenceReference(
        model_dir=tmp_path,
        metadata={
            "model_id": spec.model_id,
            "kind": spec.kind,
            "n_obs": prepared.n_obs,
            "n_children": prepared.n_children,
            "data_sha256": prepared.data_sha256,
            "ci_prob": 0.95,
        },
        pareto=pareto,
        flagged=pareto.iloc[[0]].copy(),
        full_summary=pd.DataFrame(),
        primary_hashes={},
    )


@pytest.mark.parametrize(
    ("spec", "expected_full_n", "expected_free"),
    [
        (JOINT_SPEC, 53, {"alpha", "tau", "gamma_own", "gamma_A", "kappa"}),
        (
            SES_SPEC,
            33,
            {
                "alpha",
                "tau",
                "gamma_own",
                "gamma_A",
                "gamma_mumedupost16",
                "gamma_dadedupost16",
                "gamma_agebooks",
                "kappa",
            },
        ),
        (
            ABILITY_SPEC,
            54,
            {"alpha", "tau", "gamma_own", "gamma_A", "gamma_blocks", "kappa"},
        ),
    ],
)
def test_build_influence_model_reconstructs_registered_specs(
    spec, expected_full_n, expected_free, tmp_path
):
    if spec.kind == "joint":
        prepared, _outcomes = influence._prepare_joint(spec)
    else:
        prepared, _adjust_for = influence._prepare_itt(spec)
    assert prepared.n_obs == expected_full_n
    reference = _reference_for_prepared(spec, prepared, tmp_path)

    result = influence.build_influence_model(spec, reference)

    assert result.built.prepared.n_obs == expected_full_n - 1
    assert result.excluded_subject_ids == (str(prepared.subject_ids[0]),)
    assert {rv.name for rv in result.built.model.free_RVs} == expected_free


def _write_reference_files(tmp_path, spec, *, duplicate_subject=False):
    model_dir = tmp_path / f"{spec.model_id}-reporting"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_id": spec.model_id,
                "kind": spec.kind,
                "spec_extra": spec.extra,
                "n_obs": 2,
                "n_children": 2,
                "data_sha256": "abc",
                "sampling": {
                    "draws": 10,
                    "tune": 10,
                    "chains": 2,
                    "target_accept": 0.95,
                },
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "diagnostics_summary.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    _synthetic_trace(n_obs=2, seed=40).to_netcdf(model_dir / "trace.nc")
    pd.DataFrame(
        {
            "observation_index": [0, 1],
            "subject_id": ["A", "A" if duplicate_subject else "B"],
            "pareto_k": [0.71, 0.2],
            "good_k_threshold": [0.7, 0.7],
            # Deliberately stale: the loader must recompute from k and threshold.
            "loo_reliable": [True, True],
        }
    ).to_csv(model_dir / "pareto_k.csv", index=False)
    pd.DataFrame(
        {
            "tau_prob_median": [0.1],
            "prob_ame_pos": [0.9],
        }
    ).to_csv(model_dir / "tau_summary.csv", index=False)
    return model_dir


def test_load_reference_recomputes_flag_from_saved_threshold(tmp_path):
    spec = ModelSpec(
        model_id="lrp-rli-itt-999",
        kind="itt",
        title="test",
        outcome_symbol="W",
        extra={"outcomes": ("W",)},
    )
    _write_reference_files(tmp_path, spec)

    reference = influence.load_influence_reference(
        spec, "reporting", model_output_root=tmp_path
    )

    assert reference.flagged["subject_id"].tolist() == ["A"]
    assert reference.primary_hashes == influence.hash_primary_artifacts(
        reference.model_dir
    )


def test_load_reference_rejects_non_child_level_pareto_rows(tmp_path):
    spec = ModelSpec(
        model_id="lrp-rli-itt-999",
        kind="itt",
        title="test",
        outcome_symbol="W",
        extra={"outcomes": ("W",)},
    )
    _write_reference_files(tmp_path, spec, duplicate_subject=True)

    with pytest.raises(ValueError, match="one Pareto-k point per unique child"):
        influence.load_influence_reference(
            spec, "reporting", model_output_root=tmp_path
        )


def test_load_reference_rejects_corrupt_primary_trace(tmp_path):
    spec = ModelSpec(
        model_id="lrp-rli-itt-999",
        kind="itt",
        title="test",
        outcome_symbol="W",
        extra={"outcomes": ("W",)},
    )
    model_dir = _write_reference_files(tmp_path, spec)
    (model_dir / "trace.nc").write_bytes(b"not a readable NetCDF trace")

    with pytest.raises(ValueError, match="primary trace is unreadable"):
        influence.load_influence_reference(
            spec, "reporting", model_output_root=tmp_path
        )


def _synthetic_trace(*, n_obs: int, seed: int, divergent: bool = False):
    rng = np.random.default_rng(seed)
    n_chains = 4
    n_draws = 500
    treatment = (np.arange(n_obs) % 2).astype(float)
    tau = rng.normal(0.4, 0.05, size=(n_chains, n_draws))
    alpha = rng.normal(-0.5, 0.05, size=(n_chains, n_draws))
    eta = alpha[:, :, None] + tau[:, :, None] * treatment
    posterior = xr.Dataset(
        {
            "tau": (("chain", "draw"), tau),
            "alpha": (("chain", "draw"), alpha),
            "eta": (("chain", "draw", "obs_id"), eta),
        },
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "obs_id": np.arange(n_obs),
        },
    )
    sample_stats = xr.Dataset(
        {
            "diverging": (
                ("chain", "draw"),
                np.full((n_chains, n_draws), divergent, dtype=bool),
            ),
            "energy": (("chain", "draw"), rng.normal(size=(n_chains, n_draws))),
        }
    )
    constant_data = xr.Dataset(
        {"G": (("obs_id",), treatment)}, coords={"obs_id": np.arange(n_obs)}
    )
    return xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "sample_stats": sample_stats,
            "constant_data": constant_data,
        }
    )


def _candidate_bundle(tmp_path, *, divergent: bool = False):
    spec = ModelSpec(
        model_id="lrp-rli-itt-999",
        kind="itt",
        title="test",
        outcome_symbol="W",
        extra={"outcomes": ("W",)},
    )
    model_dir = tmp_path / "models" / "lrp-rli-itt-999-reporting"
    model_dir.mkdir(parents=True)
    primary_trace = _synthetic_trace(n_obs=3, seed=41)
    primary_trace.to_netcdf(model_dir / "trace.nc")
    full = pd.DataFrame(
        [
            influence._report.tau_summary_itt(
                primary_trace,
                ci_prob=0.95,
                G=np.asarray(primary_trace.constant_data["G"].values),
            )
        ]
    )
    full.to_csv(model_dir / "tau_summary.csv", index=False)
    pareto = pd.DataFrame(
        {
            "observation_index": [0, 1, 2],
            "subject_id": ["A", "B", "C"],
            "pareto_k": [0.8, 0.1, 0.2],
            "good_k_threshold": [0.7, 0.7, 0.7],
        }
    )
    pareto.to_csv(model_dir / "pareto_k.csv", index=False)
    metadata = {
        "model_id": spec.model_id,
        "kind": spec.kind,
        "outcome_symbol": spec.outcome_symbol,
        "spec_extra": spec.extra,
        "n_obs": 3,
        "n_children": 3,
        "data_sha256": "abc",
        "ci_prob": 0.95,
        "sampling": {
            "draws": 500,
            "tune": 500,
            "chains": 4,
            "target_accept": 0.95,
            "random_seed": 42,
        },
    }
    (model_dir / "config.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    reference = influence.InfluenceReference(
        model_dir=model_dir,
        metadata=metadata,
        pareto=pareto,
        flagged=pareto.iloc[[0]].copy(),
        full_summary=full,
        primary_hashes=influence.hash_primary_artifacts(model_dir),
    )
    trace = _synthetic_trace(n_obs=2, seed=43, divergent=divergent)
    built = SimpleNamespace(
        model=SimpleNamespace(
            free_RVs=[SimpleNamespace(name="alpha"), SimpleNamespace(name="tau")]
        ),
        prepared=SimpleNamespace(
            G=np.asarray(trace.constant_data["G"].values), n_obs=2
        ),
        extras={},
    )
    influence_build = influence.InfluenceBuild(
        built=built,
        full_subject_ids=np.array(["A", "B", "C"]),
        excluded_subject_ids=("A",),
        data_path="data.csv",
        data_sha256="abc",
    )
    summary = influence.summarise_influence_refit(
        spec,
        reference,
        influence_build,
        trace,
        config="reporting",
        sampling={
            "draws": 500,
            "tune": 500,
            "chains": 4,
            "cores": 1,
            "target_accept": 0.95,
            "random_seed": 42,
        },
    )
    return SimpleNamespace(
        spec=spec,
        trace=trace,
        reference=reference,
        influence_build=influence_build,
        summary=summary,
        metadata=metadata,
    )


def test_writer_installs_hash_bound_trace_and_provenance(tmp_path):
    candidate = _candidate_bundle(tmp_path)

    installed, central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )

    trace_name = str(installed.loc[0, "trace_file"])
    trace_hash = str(installed.loc[0, "sensitivity_trace_sha256"])
    assert installed.loc[0, "excluded_subject_ids"] == "A"
    assert installed.loc[0, "convergence_scope"] == "all_free_variables"
    assert installed.loc[0, "ame_comparison_population"] == "common_retained_children"
    assert installed.loc[0, "shift_decomposition"] == (
        "total_shift=composition_shift+refit_shift"
    )
    assert installed.loc[0, "delta_ame_prob_median_alias"] == (
        "total_shift_ame_prob_median"
    )
    assert installed.loc[0, "total_shift_ame_prob_median"] == pytest.approx(
        installed.loc[0, "composition_shift_ame_prob_median"]
        + installed.loc[0, "refit_shift_ame_prob_median"]
    )
    assert installed.loc[0, "delta_ame_prob_median"] == pytest.approx(
        installed.loc[0, "total_shift_ame_prob_median"]
    )
    assert bool(installed.loc[0, "bundle_validation_passed"])
    assert trace_name == f"{influence.INFLUENCE_TRACE_STEM}-{trace_hash[:16]}.nc"
    assert central_csv.is_file()
    assert report_csv.is_file()
    assert influence.sha256_file(central_csv.parent / trace_name) == trace_hash
    assert influence.sha256_file(candidate.reference.model_dir / trace_name) == trace_hash
    installed_trace = xr.open_datatree(candidate.reference.model_dir / trace_name)
    try:
        assert json.loads(
            installed_trace.posterior.attrs[
                influence.INFLUENCE_FREE_VARIABLES_ATTR
            ]
        ) == ["alpha", "tau"]
        assert json.loads(
            installed_trace.posterior.attrs[influence.INFLUENCE_SAMPLING_ATTR]
        ) == json.loads(str(installed.loc[0, "sampling_json"]))
        assert json.loads(
            installed_trace.posterior.attrs[influence.INFLUENCE_IDENTITY_ATTR]
        ) == json.loads(str(installed.loc[0, "identity_json"]))
    finally:
        installed_trace.close()
    assert (
        candidate.reference.model_dir / "_partials" / "_diagnostics.qmd"
    ).is_file()
    for column, expected in candidate.reference.primary_hashes.items():
        assert installed[column].eq(expected).all()

    status = influence.evaluate_influence_bundle(
        pd.read_csv(report_csv),
        candidate.reference.model_dir,
        candidate.metadata,
        "reporting",
    )
    assert status["ready"] is True
    assert status["max_refit_ame_shift"] == pytest.approx(
        abs(float(installed.loc[0, "refit_shift_ame_prob_median"]))
    )
    assert status["max_composition_ame_shift"] == pytest.approx(
        abs(float(installed.loc[0, "composition_shift_ame_prob_median"]))
    )
    assert status["max_total_ame_shift"] == pytest.approx(
        abs(float(installed.loc[0, "total_shift_ame_prob_median"]))
    )


def test_report_validator_accepts_relative_primary_directory(tmp_path, monkeypatch):
    """Mirror Quarto's report-local ``Path(".")`` execution context."""
    candidate = _candidate_bundle(tmp_path)
    _installed, _central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    monkeypatch.chdir(candidate.reference.model_dir)

    status = influence.evaluate_influence_bundle(
        pd.read_csv(report_csv),
        Path("."),
        candidate.metadata,
        "reporting",
    )

    assert status["ready"] is True


def test_report_validator_rejects_stale_primary_hash(tmp_path):
    candidate = _candidate_bundle(tmp_path)
    _installed, _central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    tau_path = candidate.reference.model_dir / "tau_summary.csv"
    tau_path.write_text(tau_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    status = influence.evaluate_influence_bundle(
        pd.read_csv(report_csv),
        candidate.reference.model_dir,
        candidate.metadata,
        "reporting",
    )

    assert status["ready"] is False
    assert "primary_tau_summary_sha256" in status["reason"]


def test_report_validator_rejects_hash_bound_but_corrupt_trace(tmp_path):
    candidate = _candidate_bundle(tmp_path)
    _installed, _central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    corrupt_temp = candidate.reference.model_dir / "corrupt.tmp.nc"
    corrupt_temp.write_bytes(b"not a readable NetCDF trace")
    corrupt_hash = influence.sha256_file(corrupt_temp)
    corrupt_name = f"{influence.INFLUENCE_TRACE_STEM}-{corrupt_hash[:16]}.nc"
    corrupt_temp.replace(candidate.reference.model_dir / corrupt_name)
    summary = pd.read_csv(report_csv)
    summary["trace_file"] = corrupt_name
    summary["sensitivity_trace_sha256"] = corrupt_hash
    summary.to_csv(report_csv, index=False)

    status = influence.evaluate_influence_bundle(
        summary,
        candidate.reference.model_dir,
        candidate.metadata,
        "reporting",
    )

    assert status["ready"] is False
    assert "unreadable" in status["reason"]


def test_report_validator_rejects_trace_free_variable_contract_drift(tmp_path):
    candidate = _candidate_bundle(tmp_path)
    installed, _central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    source = candidate.reference.model_dir / str(installed.loc[0, "trace_file"])
    drifted = xr.open_datatree(source)
    temporary = candidate.reference.model_dir / "drifted.tmp.nc"
    try:
        drifted.load()
        drifted.posterior.attrs[influence.INFLUENCE_FREE_VARIABLES_ATTR] = (
            json.dumps(["alpha"])
        )
        drifted.to_netcdf(temporary)
    finally:
        drifted.close()
    drifted_hash = influence.sha256_file(temporary)
    drifted_name = f"{influence.INFLUENCE_TRACE_STEM}-{drifted_hash[:16]}.nc"
    temporary.replace(candidate.reference.model_dir / drifted_name)
    summary = pd.read_csv(report_csv)
    summary["trace_file"] = drifted_name
    summary["sensitivity_trace_sha256"] = drifted_hash

    status = influence.evaluate_influence_bundle(
        summary,
        candidate.reference.model_dir,
        candidate.metadata,
        "reporting",
    )

    assert status["ready"] is False
    assert "trace free-variable contract" in status["reason"]


def test_report_validator_rejects_self_consistent_free_variable_subset(tmp_path):
    candidate = _candidate_bundle(tmp_path)
    installed, _central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    source = candidate.reference.model_dir / str(installed.loc[0, "trace_file"])
    drifted = xr.open_datatree(source)
    temporary = candidate.reference.model_dir / "subset-contract.tmp.nc"
    try:
        drifted.load()
        drifted.posterior.attrs[influence.INFLUENCE_FREE_VARIABLES_ATTR] = (
            json.dumps(["alpha"])
        )
        drifted.to_netcdf(temporary)
    finally:
        drifted.close()
    drifted_hash = influence.sha256_file(temporary)
    drifted_name = f"{influence.INFLUENCE_TRACE_STEM}-{drifted_hash[:16]}.nc"
    temporary.replace(candidate.reference.model_dir / drifted_name)

    summary = pd.read_csv(report_csv)
    summary["trace_file"] = drifted_name
    summary["sensitivity_trace_sha256"] = drifted_hash
    summary["free_variables"] = "alpha"
    summary["n_free_variables"] = 1
    free_contract_hash = hashlib.sha256(b"alpha").hexdigest()
    summary["free_variables_sha256"] = free_contract_hash
    tampered_trace = xr.open_datatree(candidate.reference.model_dir / drifted_name)
    try:
        convergence = influence._diag.subfit_convergence(
            tampered_trace,
            label="tampered subset",
            var_names=["alpha"],
        )
    finally:
        tampered_trace.close()
    for column in ("converged", "max_rhat", "min_ess", "min_bfmi", "n_divergences"):
        summary[column] = convergence[column]

    status = influence.evaluate_influence_bundle(
        summary,
        candidate.reference.model_dir,
        candidate.metadata,
        "reporting",
    )

    assert status["ready"] is False
    assert "registered leave-out model" in status["reason"]


def test_report_validator_rejects_trace_identity_contract_drift(tmp_path):
    candidate = _candidate_bundle(tmp_path)
    installed, _central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    source = candidate.reference.model_dir / str(installed.loc[0, "trace_file"])
    drifted = xr.open_datatree(source)
    temporary = candidate.reference.model_dir / "identity-drifted.tmp.nc"
    try:
        drifted.load()
        identity = json.loads(
            drifted.posterior.attrs[influence.INFLUENCE_IDENTITY_ATTR]
        )
        identity["excluded_subject_ids"] = ["B"]
        drifted.posterior.attrs[influence.INFLUENCE_IDENTITY_ATTR] = json.dumps(
            identity, sort_keys=True
        )
        drifted.to_netcdf(temporary)
    finally:
        drifted.close()
    drifted_hash = influence.sha256_file(temporary)
    drifted_name = f"{influence.INFLUENCE_TRACE_STEM}-{drifted_hash[:16]}.nc"
    temporary.replace(candidate.reference.model_dir / drifted_name)
    summary = pd.read_csv(report_csv)
    summary["trace_file"] = drifted_name
    summary["sensitivity_trace_sha256"] = drifted_hash

    status = influence.evaluate_influence_bundle(
        summary,
        candidate.reference.model_dir,
        candidate.metadata,
        "reporting",
    )

    assert status["ready"] is False
    assert "identity contract" in status["reason"]


def test_report_validator_rejects_wrong_trace_observation_count(tmp_path):
    candidate = _candidate_bundle(tmp_path)
    _installed, _central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    wrong_length = _synthetic_trace(n_obs=3, seed=45)
    wrong_length.posterior.attrs.update(dict(candidate.trace.posterior.attrs))
    temporary = candidate.reference.model_dir / "wrong-length.tmp.nc"
    wrong_length.to_netcdf(temporary)
    wrong_hash = influence.sha256_file(temporary)
    wrong_name = f"{influence.INFLUENCE_TRACE_STEM}-{wrong_hash[:16]}.nc"
    temporary.replace(candidate.reference.model_dir / wrong_name)
    summary = pd.read_csv(report_csv)
    summary["trace_file"] = wrong_name
    summary["sensitivity_trace_sha256"] = wrong_hash

    status = influence.evaluate_influence_bundle(
        summary,
        candidate.reference.model_dir,
        candidate.metadata,
        "reporting",
    )

    assert status["ready"] is False
    assert "observation count" in status["reason"]


def test_report_validator_recomputes_saved_ame_shift(tmp_path):
    candidate = _candidate_bundle(tmp_path)
    _installed, _central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    summary = pd.read_csv(report_csv)
    summary["delta_ame_prob_median"] += 0.01

    status = influence.evaluate_influence_bundle(
        summary,
        candidate.reference.model_dir,
        candidate.metadata,
        "reporting",
    )

    assert status["ready"] is False
    assert "total-shift alias" in status["reason"]


@pytest.mark.parametrize(
    ("column", "reason"),
    [
        ("ame_prob_median_full_retained", "bound artefact"),
        ("composition_shift_ame_prob_median", "composition AME shifts"),
        ("refit_shift_ame_prob_median", "common-population refit AME shifts"),
        ("total_shift_ame_prob_median", "total AME shifts"),
    ],
)
def test_report_validator_rejects_tampered_ame_decomposition(
    tmp_path, column, reason
):
    candidate = _candidate_bundle(tmp_path)
    _installed, _central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    summary = pd.read_csv(report_csv)
    summary[column] += 0.01

    status = influence.evaluate_influence_bundle(
        summary,
        candidate.reference.model_dir,
        candidate.metadata,
        "reporting",
    )

    assert status["ready"] is False
    assert reason in status["reason"]


def test_nonconverged_refit_is_preserved_but_does_not_replace_valid_bundle(tmp_path):
    candidate = _candidate_bundle(tmp_path)
    _installed, central_csv, report_csv = influence.write_influence_artifacts(
        candidate.trace,
        candidate.summary,
        candidate.reference,
        "reporting",
        sensitivity_root=tmp_path / "sensitivity",
    )
    report_before = report_csv.read_bytes()
    central_before = central_csv.read_bytes()
    failed_trace = _synthetic_trace(n_obs=2, seed=44, divergent=True)
    failed_summary = influence.summarise_influence_refit(
        candidate.spec,
        candidate.reference,
        candidate.influence_build,
        failed_trace,
        config="reporting",
        sampling={
            "draws": 500,
            "tune": 500,
            "chains": 4,
            "cores": 1,
            "target_accept": 0.95,
            "random_seed": 42,
        },
    )

    with pytest.raises(influence.InfluenceBundleError, match="convergence"):
        influence.write_influence_artifacts(
            failed_trace,
            failed_summary,
            candidate.reference,
            "reporting",
            sensitivity_root=tmp_path / "sensitivity",
        )

    assert report_csv.read_bytes() == report_before
    assert central_csv.read_bytes() == central_before
    assert list(central_csv.parent.glob("influence_sensitivity_failed-*.csv"))
    assert len(list(central_csv.parent.glob(f"{influence.INFLUENCE_TRACE_STEM}-*.nc"))) == 2
