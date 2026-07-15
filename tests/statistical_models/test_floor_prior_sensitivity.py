# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Focused tests for the P/N treatment-prior sensitivity release gate."""

from __future__ import annotations

import json
from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import diagnostics as _diag
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
)
from language_reading_predictors.statistical_models.reporting import (
    rope_summary,
    tau_summary_offfloor,
)
from language_reading_predictors.statistical_models.sensitivity import (
    FLOOR_SENSITIVITY_AGE_ADJUSTMENTS,
    FLOOR_SENSITIVITY_AXIS,
    FLOOR_SENSITIVITY_FILENAME,
    FLOOR_SENSITIVITY_MODEL_IDS,
    FLOOR_SENSITIVITY_PROVENANCE_ATTR,
    FLOOR_SENSITIVITY_SAMPLING_ATTR,
    FLOOR_SENSITIVITY_TAU_SIGMAS,
    PrimaryFloorReference,
    evaluate_floor_sensitivity,
    floor_trace_provenance,
    load_primary_floor_reference,
    sha256_file,
    tau_psense_status,
)

# ``scripts`` is not an installed package, so load the CLI independently of
# whether pytest itself placed the repository root on ``sys.path``.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tau_prior_sensitivity.py"
_SCRIPT_SPEC = spec_from_file_location("_lrp_floor_tau_prior_sensitivity", _SCRIPT_PATH)
if _SCRIPT_SPEC is None or _SCRIPT_SPEC.loader is None:
    raise ImportError(f"cannot load sensitivity runner from {_SCRIPT_PATH}")
sensitivity_script = module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(sensitivity_script)


FIXTURE_CONFIG = "dev"
FIXTURE_SAMPLING = {
    "draws": 500,
    "tune": 500,
    "chains": 2,
    "target_accept": 0.85,
}


def test_floor_only_runs_use_a_separate_default_output_archive():
    assert sensitivity_script._default_output_name(["P", "N"]) == "floor_tau_prior_sensitivity"
    assert sensitivity_script._default_output_name(["P"]) == "floor_tau_prior_sensitivity"
    assert sensitivity_script._default_output_name(list(sensitivity_script.DEFAULT_OUTCOMES)) == "tau_prior_sensitivity"


def _sample_counts(symbol: str) -> tuple[int, int, int]:
    return (41, 24, 17) if symbol == "P" else (36, 21, 15)


def _complete_grid(symbol: str = "P") -> pd.DataFrame:
    n, n_intervention, n_control = _sample_counts(symbol)
    rows = []
    for tau_sigma in FLOOR_SENSITIVITY_TAU_SIGMAS:
        for age_adjusted in FLOOR_SENSITIVITY_AGE_ADJUSTMENTS:
            suffix = f"{tau_sigma:g}-{age_adjusted}"
            rows.append(
                {
                    "config": FIXTURE_CONFIG,
                    "outcome": symbol,
                    "model_id": FLOOR_SENSITIVITY_MODEL_IDS[symbol],
                    "estimand": ("off_floor_risk_difference_given_observed_baseline_floor"),
                    "analysis_subset": "observed_baseline_floor",
                    "likelihood": "bernoulli_offfloor",
                    "sensitivity_axis": FLOOR_SENSITIVITY_AXIS,
                    "tau_sigma": tau_sigma,
                    "age_adjusted": age_adjusted,
                    "use_age_linear": age_adjusted,
                    "use_own_baseline": False,
                    "data_sha256": "a" * 64,
                    "n": n,
                    "n_intervention": n_intervention,
                    "n_control": n_control,
                    "primary_config_sha256": "b" * 64,
                    "primary_trace_sha256": "c" * 64,
                    "primary_sampling_draws": FIXTURE_SAMPLING["draws"],
                    "primary_sampling_tune": FIXTURE_SAMPLING["tune"],
                    "primary_sampling_chains": FIXTURE_SAMPLING["chains"],
                    "primary_sampling_target_accept": FIXTURE_SAMPLING[
                        "target_accept"
                    ],
                    "primary_sampling_random_seed": 47,
                    "sampling_draws": FIXTURE_SAMPLING["draws"],
                    "sampling_tune": FIXTURE_SAMPLING["tune"],
                    "sampling_chains": FIXTURE_SAMPLING["chains"],
                    "sampling_cores": FIXTURE_SAMPLING["chains"],
                    "sampling_target_accept": FIXTURE_SAMPLING["target_accept"],
                    "sampling_random_seed": 20260701,
                    "sampling_nuts_sampler": "nutpie",
                    "risk_difference_median": 0.02 + tau_sigma / 100,
                    "risk_difference_mean": 0.02 + tau_sigma / 100,
                    "risk_difference_lo50": -0.02,
                    "risk_difference_hi50": 0.08,
                    "risk_difference_lo90": -0.08,
                    "risk_difference_hi90": 0.16,
                    "risk_difference_lo": -0.10,
                    "risk_difference_hi": 0.20,
                    "risk_difference_hpdi_lo": -0.11,
                    "risk_difference_hpdi_hi": 0.19,
                    "prob_risk_difference_positive": 0.70,
                    "meaningful_risk_difference": 0.10,
                    "prob_risk_difference_ge_0_10": 0.40,
                    "tau_logit_median": 0.20,
                    "tau_logit_lo": -0.10,
                    "tau_logit_hi": 0.50,
                    "converged": True,
                    "max_rhat": 1.005,
                    "min_ess": 500.0,
                    "min_bfmi": 0.80,
                    "n_divergences": 0,
                    "free_variables": ("alpha|tau|gamma_A" if age_adjusted else "alpha|tau"),
                    "n_free_variables": 3 if age_adjusted else 2,
                    "convergence_scope": "all_free_variables",
                    "trace_file": f"trace-{suffix}.nc",
                    "trace_sha256": "d" * 64,
                }
            )
    return pd.DataFrame(rows)


def _write_trace(
    path: Path,
    variable_names: set[str],
    *,
    n: int | None = None,
    n_intervention: int | None = None,
    chains: int = 2,
    draws: int = 4,
    seed: int = 1,
    sampling_provenance: dict[str, int | float | str] | None = None,
    trace_provenance: dict[str, object] | str | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    shape = (chains, draws)

    def independent_chains(mean: float, sd: float) -> np.ndarray:
        base = rng.normal(mean, sd, size=draws)
        return np.stack([base[rng.permutation(draws)] for _ in range(chains)])

    alpha = independent_chains(-0.8, 0.25)
    tau = independent_chains(0.20 + seed / 5000, 0.22)
    values: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name in sorted(variable_names):
        if name == "alpha":
            draws_for_variable = alpha
        elif name == "tau":
            draws_for_variable = tau
        elif name == "gamma_A":
            draws_for_variable = independent_chains(0.08, 0.12)
        else:
            draws_for_variable = independent_chains(0.0, 0.2)
        values[name] = (("chain", "draw"), draws_for_variable)

    groups: dict[str, xr.Dataset] = {}
    posterior_coords: dict[str, np.ndarray] = {
        "chain": np.arange(chains),
        "draw": np.arange(draws),
    }
    if n is not None:
        if n_intervention is None:
            raise ValueError("n_intervention is required when n is supplied")
        G = np.r_[np.ones(n_intervention), np.zeros(n - n_intervention)]
        age = np.linspace(-1.0, 1.0, n)
        eta = alpha[:, :, None] + tau[:, :, None] * G[None, None, :]
        if "gamma_A" in variable_names:
            gamma_age = values["gamma_A"][1]
            eta = eta + gamma_age[:, :, None] * age[None, None, :]
        values["eta"] = (("chain", "draw", "obs_id"), eta)
        posterior_coords["obs_id"] = np.arange(n)
        groups["constant_data"] = xr.Dataset(
            {"G": (("obs_id",), G)},
            coords={"obs_id": np.arange(n)},
        )

    posterior = xr.Dataset(values, coords=posterior_coords)
    if sampling_provenance is not None:
        posterior.attrs[FLOOR_SENSITIVITY_SAMPLING_ATTR] = json.dumps(
            sampling_provenance,
            sort_keys=True,
            separators=(",", ":"),
        )
    if trace_provenance is not None:
        posterior.attrs[FLOOR_SENSITIVITY_PROVENANCE_ATTR] = (
            trace_provenance
            if isinstance(trace_provenance, str)
            else json.dumps(
                trace_provenance,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    groups["posterior"] = posterior
    if n is not None:
        groups["sample_stats"] = xr.Dataset(
            {
                "diverging": (("chain", "draw"), np.zeros(shape, dtype=bool)),
                "energy": (("chain", "draw"), rng.normal(size=shape)),
            },
            coords={"chain": np.arange(chains), "draw": np.arange(draws)},
        )
    xr.DataTree.from_dict(groups).to_netcdf(path)


def _make_primary_reference(
    model_dir: Path,
    symbol: str = "P",
    *,
    config_name: str = FIXTURE_CONFIG,
) -> PrimaryFloorReference:
    model_dir.mkdir(parents=True, exist_ok=True)
    n, n_intervention, n_control = _sample_counts(symbol)
    data_sha256 = "a" * 64
    sampling = (
        {
            "draws": 6000,
            "tune": 6000,
            "chains": 6,
            "target_accept": 0.95,
        }
        if config_name == "reporting"
        else FIXTURE_SAMPLING
    )
    config = {
        "model_id": FLOOR_SENSITIVITY_MODEL_IDS[symbol],
        "outcome_symbol": symbol,
        "n_obs": n,
        "data_sha256": data_sha256,
        "sampling": {
            **sampling,
            "random_seed": 47,
        },
        "extra": {
            "floor_rule": {
                "outcome": symbol,
                "at_risk_n": n,
                "eligibility_by_arm": [
                    {
                        "arm": "intervention",
                        "n_exploratory_eligible": n_intervention,
                    },
                    {
                        "arm": "control",
                        "n_exploratory_eligible": n_control,
                    },
                ],
            }
        },
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
    _write_trace(
        model_dir / "trace.nc",
        {"alpha", "tau", "gamma_A"},
        chains=int(sampling["chains"]),
        draws=int(sampling["draws"]),
    )
    return load_primary_floor_reference(
        model_dir,
        symbol,
        config_name=config_name,
    )


def _materialise_grid(
    trace_root: Path,
    reference: PrimaryFloorReference,
) -> pd.DataFrame:
    trace_root.mkdir(parents=True, exist_ok=True)
    rows = _complete_grid(reference.outcome)
    for column, value in reference.manifest_values().items():
        rows[column] = value
    rows["sampling_draws"] = reference.sampling["draws"]
    rows["sampling_tune"] = reference.sampling["tune"]
    rows["sampling_chains"] = reference.sampling["chains"]
    rows["sampling_cores"] = reference.sampling["chains"]
    rows["sampling_target_accept"] = reference.sampling["target_accept"]
    for index, row in rows.iterrows():
        declared = {name for name in str(row["free_variables"]).split("|") if name}
        trace_provenance = floor_trace_provenance(row)
        sampling_provenance = trace_provenance["sampling"]
        semantic = f"trace-{row['tau_sigma']:g}-{bool(row['age_adjusted'])}"
        temporary = trace_root / f".{semantic}.nc"
        _write_trace(
            temporary,
            declared,
            n=int(row["n"]),
            n_intervention=int(row["n_intervention"]),
            chains=int(row["sampling_chains"]),
            draws=int(row["sampling_draws"]),
            seed=100 + int(index),
            sampling_provenance=sampling_provenance,
            trace_provenance=trace_provenance,
        )
        trace = az.from_netcdf(temporary)
        try:
            convergence = _diag.subfit_convergence(
                trace,
                label="floor sensitivity test fixture",
                var_names=list(trace_provenance["free_variables"]),
            )
            assert convergence["converged"] is True
            G = np.asarray(trace.constant_data["G"].values, dtype=float)
            summary = tau_summary_offfloor(trace, ci_prob=0.95, G=G)
            magnitude = rope_summary(
                trace,
                G=G,
                n_trials=1,
                delta=0.10,
                ci_prob=0.95,
                varying_term="",
            )
        finally:
            trace.close()
        evidence = {
            "converged": convergence["converged"],
            "max_rhat": convergence["max_rhat"],
            "min_ess": convergence["min_ess"],
            "min_bfmi": convergence["min_bfmi"],
            "n_divergences": convergence["n_divergences"],
            "risk_difference_median": summary["tau_prob_median"],
            "risk_difference_mean": summary["tau_prob_mean"],
            "risk_difference_lo50": summary["tau_prob_lo50"],
            "risk_difference_hi50": summary["tau_prob_hi50"],
            "risk_difference_lo90": summary["tau_prob_lo90"],
            "risk_difference_hi90": summary["tau_prob_hi90"],
            "risk_difference_lo": summary["tau_prob_lo"],
            "risk_difference_hi": summary["tau_prob_hi"],
            "risk_difference_hpdi_lo": summary["tau_prob_hpdi_lo"],
            "risk_difference_hpdi_hi": summary["tau_prob_hpdi_hi"],
            "prob_risk_difference_positive": summary["prob_ame_pos"],
            "prob_risk_difference_ge_0_10": magnitude["prob_benefit_ge_delta"],
            "tau_logit_median": summary["tau_logit_median"],
            "tau_logit_lo": summary["tau_logit_lo"],
            "tau_logit_hi": summary["tau_logit_hi"],
        }
        for column, value in evidence.items():
            rows.at[index, column] = value
        digest = sha256_file(temporary)
        trace_path = trace_root / f"{semantic}-{digest[:12]}.nc"
        temporary.replace(trace_path)
        rows.at[index, "trace_file"] = trace_path.name
        rows.at[index, "trace_sha256"] = digest
    return rows


def _ready_bundle(
    tmp_path: Path,
    symbol: str = "P",
) -> tuple[pd.DataFrame, PrimaryFloorReference, Path]:
    model_dir = (
        tmp_path
        / "primary"
        / f"{FLOOR_SENSITIVITY_MODEL_IDS[symbol]}-{FIXTURE_CONFIG}"
    )
    reference = _make_primary_reference(
        model_dir,
        symbol,
        config_name=FIXTURE_CONFIG,
    )
    trace_root = tmp_path / "sensitivity"
    return _materialise_grid(trace_root, reference), reference, trace_root


def _replace_row_trace(
    grid: pd.DataFrame,
    trace_root: Path,
    row: int,
    *,
    suffix: str,
    mutate,
) -> None:
    """Mutate one valid trace and keep its manifest digest/path internally valid."""
    original = trace_root / str(grid.loc[row, "trace_file"])
    trace = az.from_netcdf(original)
    try:
        mutate(trace)
        temporary = trace_root / f".{suffix}.nc"
        trace.to_netcdf(temporary)
    finally:
        trace.close()
    digest = sha256_file(temporary)
    replacement = trace_root / f"trace-{suffix}-{digest[:12]}.nc"
    temporary.replace(replacement)
    grid.loc[row, "trace_file"] = replacement.name
    grid.loc[row, "trace_sha256"] = digest


@pytest.mark.parametrize(("symbol", "expected_n"), [("P", 41), ("N", 36)])
def test_floor_sensitivity_builder_matches_primary_estimand(symbol: str, expected_n: int):
    prepared = load_and_prepare(
        phase_mode="itt",
        outcomes=(symbol,),
        pre_required=(),
    )

    no_age = sensitivity_script._build_floor_sensitivity_model(
        prepared,
        symbol,
        tau_sigma=1.0,
        age_adjusted=False,
    )
    with_age = sensitivity_script._build_floor_sensitivity_model(
        prepared,
        symbol,
        tau_sigma=1.0,
        age_adjusted=True,
    )

    assert no_age.prepared.n_obs == expected_n
    assert with_age.prepared.n_obs == expected_n
    assert {rv.name for rv in no_age.model.observed_RVs} == {"y_offfloor"}
    assert {rv.name for rv in with_age.model.observed_RVs} == {"y_offfloor"}
    assert "gamma_A" not in {rv.name for rv in no_age.model.free_RVs}
    assert "gamma_A" in {rv.name for rv in with_age.model.free_RVs}
    assert "gamma_own" not in {rv.name for rv in with_age.model.free_RVs}


@pytest.fixture(scope="module")
def floor_prepared_p():
    return load_and_prepare(
        phase_mode="itt",
        outcomes=("P",),
        pre_required=(),
    )


@pytest.mark.parametrize("tau_sigma", FLOOR_SENSITIVITY_TAU_SIGMAS)
@pytest.mark.parametrize("age_adjusted", FLOOR_SENSITIVITY_AGE_ADJUSTMENTS)
def test_floor_sensitivity_builder_honours_prior_and_age_contract(
    floor_prepared_p,
    tau_sigma: float,
    age_adjusted: bool,
):
    built = sensitivity_script._build_floor_sensitivity_model(
        floor_prepared_p,
        "P",
        tau_sigma=tau_sigma,
        age_adjusted=age_adjusted,
    )

    tau = built.model["tau"]
    assert "normal_rv" in str(tau.owner.op)
    assert float(np.asarray(tau.owner.inputs[-1].data)) == pytest.approx(tau_sigma)
    expected_free = {"alpha", "tau"}
    if age_adjusted:
        expected_free.add("gamma_A")
    assert {rv.name for rv in built.model.free_RVs} == expected_free
    assert {rv.name for rv in built.model.observed_RVs} == {"y_offfloor"}


def test_floor_sensitivity_gate_requires_primary_aligned_validated_bundle(tmp_path: Path):
    grid, reference, trace_root = _ready_bundle(tmp_path)

    complete = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )
    assert complete["expected_n"] == 6
    assert complete["observed_n"] == 6
    assert complete["complete"] is True
    assert complete["converged"] is True
    assert complete["primary_aligned"] is True
    assert complete["traces_present"] is True
    assert complete["traces_validated"] is True
    assert complete["ready"] is True
    assert complete["risk_difference_median_min"] == pytest.approx(grid["risk_difference_median"].min())
    assert complete["risk_difference_median_max"] == pytest.approx(grid["risk_difference_median"].max())
    assert complete["prob_meaningful_min"] == pytest.approx(grid["prob_risk_difference_ge_0_10"].min())
    assert complete["prob_meaningful_max"] == pytest.approx(grid["prob_risk_difference_ge_0_10"].max())

    no_primary = evaluate_floor_sensitivity(grid, "P", trace_root=trace_root)
    assert no_primary["complete"] is True
    assert no_primary["primary_aligned"] is False
    assert no_primary["ready"] is False

    incomplete = evaluate_floor_sensitivity(
        grid.iloc[:-1],
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )
    assert incomplete["complete"] is False
    assert incomplete["ready"] is False

    failed = grid.copy()
    failed.loc[0, "converged"] = False
    nonconverged = evaluate_floor_sensitivity(
        failed,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )
    assert nonconverged["complete"] is True
    assert nonconverged["converged"] is False
    assert nonconverged["ready"] is False

    missing_path = trace_root / str(grid.iloc[0]["trace_file"])
    missing_path.unlink()
    missing_trace = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )
    assert missing_trace["complete"] is True
    assert missing_trace["traces_present"] is False
    assert missing_trace["ready"] is False


@pytest.mark.parametrize(
    "mutation",
    ["config", "data_sha256", "n", "primary_config_sha256", "primary_trace_sha256"],
)
def test_floor_sensitivity_gate_rejects_stale_primary_binding(
    tmp_path: Path,
    mutation: str,
):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    if mutation == "config":
        grid["config"] = "test"
    elif mutation == "data_sha256":
        grid["data_sha256"] = "f" * 64
    elif mutation == "n":
        grid["n"] = 42
        grid["n_intervention"] = 25
    elif mutation == "primary_config_sha256":
        grid["primary_config_sha256"] = "e" * 64
    else:
        grid["primary_trace_sha256"] = "e" * 64

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["primary_aligned"] is False
    assert status["ready"] is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("sampling_draws", 501),
        ("sampling_tune", 501),
        ("sampling_chains", 3),
        ("sampling_target_accept", 0.90),
    ],
)
def test_floor_sensitivity_gate_rejects_reduced_or_changed_primary_sampling(
    tmp_path: Path,
    column: str,
    value: int | float,
):
    grid, reference, _trace_root = _ready_bundle(tmp_path)
    grid[column] = value
    if column == "sampling_chains":
        grid["sampling_cores"] = value

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
    )

    assert status["complete"] is True
    assert status["primary_aligned"] is False
    assert status["ready"] is False


@pytest.mark.parametrize("primary_file", ["config.json", "trace.nc"])
def test_floor_sensitivity_gate_compares_current_primary_file_hash(
    tmp_path: Path,
    primary_file: str,
):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    path = reference.model_dir / primary_file
    if primary_file == "config.json":
        config = json.loads(path.read_text(encoding="utf-8"))
        config["title"] = "changed after sensitivity fit"
        path.write_text(json.dumps(config), encoding="utf-8")
    else:
        with path.open("ab") as handle:
            handle.write(b"changed")
    current_reference = load_primary_floor_reference(
        reference.model_dir,
        "P",
        config_name=reference.config_name,
    )

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=current_reference,
        trace_root=trace_root,
    )

    assert status["primary_aligned"] is False
    assert status["ready"] is False


def test_primary_reference_rejects_corrupt_primary_trace(tmp_path: Path):
    model_dir = tmp_path / "primary" / "lrp-rli-itt-009-reporting"
    _make_primary_reference(model_dir, config_name="reporting")
    (model_dir / "trace.nc").write_bytes(b"not a NetCDF file")

    with pytest.raises(ValueError, match="primary trace is not a readable NetCDF"):
        load_primary_floor_reference(
            model_dir,
            "P",
            config_name="reporting",
        )


def test_primary_reference_rejects_missing_variables_and_dimension_drift(
    tmp_path: Path,
):
    model_dir = tmp_path / "primary" / "lrp-rli-itt-009-reporting"
    _make_primary_reference(model_dir, config_name="reporting")
    _write_trace(model_dir / "trace.nc", {"alpha"})
    with pytest.raises(ValueError, match="lacks required variables: tau"):
        load_primary_floor_reference(
            model_dir,
            "P",
            config_name="reporting",
        )

    _write_trace(model_dir / "trace.nc", {"alpha", "tau"})
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sampling"]["draws"] = 5
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="chain/draw dimensions do not match"):
        load_primary_floor_reference(
            model_dir,
            "P",
            config_name="reporting",
        )


def test_floor_sensitivity_gate_rejects_corrupt_trace_even_with_matching_hash(
    tmp_path: Path,
):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    row = grid.index[0]
    old_path = trace_root / str(grid.loc[row, "trace_file"])
    old_path.unlink()
    temporary = trace_root / ".corrupt.nc"
    temporary.write_bytes(b"not a NetCDF file")
    digest = sha256_file(temporary)
    corrupt_path = trace_root / f"trace-corrupt-{digest[:12]}.nc"
    temporary.replace(corrupt_path)
    grid.loc[row, "trace_file"] = corrupt_path.name
    grid.loc[row, "trace_sha256"] = digest

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["traces_present"] is True
    assert status["traces_validated"] is False
    assert "unreadable NetCDF" in " ".join(status["trace_errors"])
    assert status["ready"] is False


def test_floor_sensitivity_gate_rejects_trace_missing_declared_free_variable(
    tmp_path: Path,
):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    row = grid.index[0]
    old_path = trace_root / str(grid.loc[row, "trace_file"])
    old_path.unlink()
    temporary = trace_root / ".missing-tau.nc"
    row_values = grid.loc[row]
    provenance = floor_trace_provenance(row_values)
    _write_trace(
        temporary,
        {"alpha"},
        n=int(row_values["n"]),
        n_intervention=int(row_values["n_intervention"]),
        chains=int(row_values["sampling_chains"]),
        draws=int(row_values["sampling_draws"]),
        sampling_provenance=provenance["sampling"],
        trace_provenance=provenance,
    )
    digest = sha256_file(temporary)
    trace_path = trace_root / f"trace-missing-tau-{digest[:12]}.nc"
    temporary.replace(trace_path)
    grid.loc[row, "trace_file"] = trace_path.name
    grid.loc[row, "trace_sha256"] = digest

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["traces_validated"] is False
    assert "missing posterior variables tau" in " ".join(status["trace_errors"])
    assert status["ready"] is False


def test_floor_sensitivity_gate_validates_trace_sampling_provenance(tmp_path: Path):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    grid["sampling_draws"] = 5

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["complete"] is True
    assert status["traces_validated"] is False
    assert "posterior dimensions do not match sampling provenance" in " ".join(status["trace_errors"])
    assert status["ready"] is False


def test_floor_sensitivity_gate_rejects_same_age_trace_swap(tmp_path: Path):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    same_age = grid.index[~grid["age_adjusted"].astype(bool)].tolist()
    first, second = same_age[:2]
    for column in ("trace_file", "trace_sha256"):
        grid.loc[[first, second], column] = grid.loc[[second, first], column].to_numpy()

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["complete"] is True
    assert status["traces_present"] is True
    assert status["traces_validated"] is False
    assert "trace provenance does not match manifest" in " ".join(status["trace_errors"])
    assert status["ready"] is False


def test_floor_sensitivity_gate_rejects_fabricated_passing_convergence(
    tmp_path: Path,
):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    grid.loc[0, ["converged", "max_rhat", "min_ess", "min_bfmi", "n_divergences"]] = [
        True,
        1.0,
        99999.0,
        1.9,
        0,
    ]

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["converged"] is True
    assert status["traces_validated"] is False
    assert "does not match trace" in " ".join(status["trace_errors"])
    assert status["ready"] is False


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("risk_difference_mean", 0.999),
        ("prob_risk_difference_positive", 0.999),
        ("prob_risk_difference_ge_0_10", 0.999),
        ("tau_logit_median", None),
    ],
)
def test_floor_sensitivity_gate_rejects_fabricated_effect_summary(
    tmp_path: Path,
    column: str,
    replacement: float | None,
):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    grid.loc[0, column] = replacement if replacement is not None else float(grid.loc[0, column]) + 0.01

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["complete"] is True
    assert status["traces_validated"] is False
    assert f"manifest {column} does not match trace" in " ".join(status["trace_errors"])
    assert status["ready"] is False


@pytest.mark.parametrize("provenance", [None, "{not-json"])
def test_floor_sensitivity_gate_rejects_missing_or_malformed_trace_provenance(
    tmp_path: Path,
    provenance: str | None,
):
    grid, reference, trace_root = _ready_bundle(tmp_path)

    def mutate(trace) -> None:
        if provenance is None:
            trace.posterior.attrs.pop(FLOOR_SENSITIVITY_PROVENANCE_ATTR)
        else:
            trace.posterior.attrs[FLOOR_SENSITIVITY_PROVENANCE_ATTR] = provenance

    _replace_row_trace(
        grid,
        trace_root,
        0,
        suffix="bad-provenance",
        mutate=mutate,
    )

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["traces_present"] is True
    assert status["traces_validated"] is False
    assert "missing or malformed trace provenance" in " ".join(status["trace_errors"])
    assert status["ready"] is False


def test_floor_sensitivity_gate_rejects_outcome_model_mismatch(tmp_path: Path):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    grid["model_id"] = FLOOR_SENSITIVITY_MODEL_IDS["N"]

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["complete"] is False
    assert status["ready"] is False


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("risk_difference_median", 1.01),
        ("risk_difference_mean", -1.01),
        ("risk_difference_lo50", -1.01),
        ("risk_difference_hi50", 1.01),
        ("risk_difference_lo90", -1.01),
        ("risk_difference_hi90", 1.01),
        ("risk_difference_lo", -1.01),
        ("risk_difference_hi", 1.01),
        ("risk_difference_hpdi_lo", -1.01),
        ("risk_difference_hpdi_hi", 1.01),
    ],
)
def test_floor_sensitivity_gate_rejects_out_of_range_risk_differences(
    tmp_path: Path,
    column: str,
    invalid_value: float,
):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    grid.loc[0, column] = invalid_value

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["complete"] is False
    assert status["ready"] is False


@pytest.mark.parametrize("age_adjusted", [False, True])
def test_floor_sensitivity_gate_rejects_free_variable_contract_mismatch(
    tmp_path: Path,
    age_adjusted: bool,
):
    grid, reference, trace_root = _ready_bundle(tmp_path)
    row = grid.index[grid["age_adjusted"] == age_adjusted][0]
    if age_adjusted:
        grid.loc[row, "free_variables"] = "alpha|tau"
        grid.loc[row, "n_free_variables"] = 2
    else:
        grid.loc[row, "free_variables"] = "alpha|tau|unexpected"
        grid.loc[row, "n_free_variables"] = 3

    status = evaluate_floor_sensitivity(
        grid,
        "P",
        primary_reference=reference,
        trace_root=trace_root,
    )

    assert status["complete"] is False
    assert status["ready"] is False


def test_tau_psense_status_requires_one_explicit_interpretable_tau_row():
    conflict = pd.DataFrame(
        {"diagnosis": ["potential prior-data conflict", "✓"]},
        index=["tau", "alpha"],
    )
    assert tau_psense_status(conflict) == "conflict"

    no_tau_conflict = conflict.copy()
    no_tau_conflict.loc["tau", "diagnosis"] = "✓"
    no_tau_conflict.loc["alpha", "diagnosis"] = "potential prior-data conflict"
    assert tau_psense_status(no_tau_conflict) == "no_conflict"
    no_tau_conflict.loc["tau", "diagnosis"] = "no prior-data conflict"
    assert tau_psense_status(no_tau_conflict) == "no_conflict"

    assert tau_psense_status(None) == "unavailable"
    assert tau_psense_status(pd.DataFrame()) == "unavailable"
    assert tau_psense_status(pd.DataFrame({"diagnosis": ["✓"]}, index=["alpha"])) == "unavailable"
    assert tau_psense_status(pd.DataFrame({"diagnosis": ["✓", "✓"]}, index=["tau", "tau"])) == "unavailable"
    assert tau_psense_status(pd.DataFrame({"diagnosis": ["ambiguous output"]}, index=["tau"])) == "unavailable"


def test_floor_sensitivity_artifacts_are_transactionally_installed(tmp_path: Path):
    sensitivity_dir = tmp_path / "sensitivity"
    model_root = tmp_path / "models"
    report_dir = model_root / f"lrp-rli-itt-009-{FIXTURE_CONFIG}"
    reference = _make_primary_reference(
        report_dir,
        config_name=FIXTURE_CONFIG,
    )
    rows = _materialise_grid(sensitivity_dir, reference)

    written = sensitivity_script._copy_floor_model_artifacts(
        rows,
        sensitivity_dir=sensitivity_dir,
        model_output_root=model_root,
        config=FIXTURE_CONFIG,
    )

    assert written == [report_dir / FLOOR_SENSITIVITY_FILENAME]
    copied = pd.read_csv(written[0])
    assert len(copied) == 6
    assert all(Path(name).parent == Path(".") for name in copied["trace_file"])
    assert all((report_dir / name).is_file() for name in copied["trace_file"])
    assert all(
        Path(row.trace_file).stem.endswith(f"-{row.trace_sha256[:12]}") for row in copied.itertuples(index=False)
    )
    installed = evaluate_floor_sensitivity(
        copied,
        "P",
        primary_reference=load_primary_floor_reference(
            report_dir,
            "P",
            config_name=FIXTURE_CONFIG,
        ),
        trace_root=report_dir,
    )
    assert installed["ready"] is True


def test_floor_trace_persistence_is_content_addressed_and_immutable(tmp_path: Path):
    sensitivity_dir = tmp_path / "sensitivity"
    semantic = Path("traces/lrp-rli-itt-009-reporting/trace-floor-cell.nc")
    first_source = tmp_path / "first.nc"
    second_source = tmp_path / "second.nc"
    _write_trace(first_source, {"alpha", "tau"}, seed=11)
    _write_trace(second_source, {"alpha", "tau"}, seed=12)

    first_trace = az.from_netcdf(first_source)
    try:
        first_relative, first_digest = sensitivity_script._persist_content_addressed_trace(
            first_trace,
            sensitivity_dir=sensitivity_dir,
            semantic_file=semantic,
        )
    finally:
        first_trace.close()
    first_path = sensitivity_dir / first_relative
    first_bytes = first_path.read_bytes()

    second_trace = az.from_netcdf(second_source)
    try:
        second_relative, second_digest = sensitivity_script._persist_content_addressed_trace(
            second_trace,
            sensitivity_dir=sensitivity_dir,
            semantic_file=semantic,
        )
    finally:
        second_trace.close()

    assert first_relative.stem.endswith(f"-{first_digest[:12]}")
    assert second_relative.stem.endswith(f"-{second_digest[:12]}")
    assert first_relative != second_relative
    assert first_path.read_bytes() == first_bytes
    assert sha256_file(first_path) == first_digest
    assert sha256_file(sensitivity_dir / second_relative) == second_digest


def test_content_addressed_csv_and_atomic_fixed_manifest_publication(
    tmp_path: Path,
):
    sensitivity_dir = tmp_path / "sensitivity"
    frame = pd.DataFrame({"outcome": ["P"], "tau_sigma": [0.5]})
    run_csv = sensitivity_script._write_content_addressed_csv(
        frame,
        sensitivity_dir,
    )
    same_run_csv = sensitivity_script._write_content_addressed_csv(
        frame,
        sensitivity_dir,
    )
    fixed = sensitivity_dir / "tau_prior_sensitivity.csv"
    fixed.write_text("old manifest\n", encoding="utf-8")

    sensitivity_script._publish_csv_atomically(run_csv, fixed)

    assert run_csv == same_run_csv
    assert run_csv.stem.endswith(f"-{sha256_file(run_csv)[:12]}")
    assert fixed.read_bytes() == run_csv.read_bytes()


def _ready_floor_publication_bundle(
    tmp_path: Path,
) -> tuple[pd.DataFrame, dict[str, PrimaryFloorReference], Path]:
    p_grid, p_reference, trace_root = _ready_bundle(tmp_path, "P")
    n_grid, n_reference, n_trace_root = _ready_bundle(tmp_path, "N")
    assert n_trace_root == trace_root
    frame = pd.concat([p_grid, n_grid], ignore_index=True)
    references = {"P": p_reference, "N": n_reference}
    return frame, references, trace_root


def _with_reporting_publication_identity(
    frame: pd.DataFrame,
    references: dict[str, PrimaryFloorReference],
) -> tuple[pd.DataFrame, dict[str, PrimaryFloorReference]]:
    reporting_frame = frame.copy(deep=True)
    reporting_frame["config"] = "reporting"
    reporting_references = {
        symbol: replace(reference, config_name="reporting")
        for symbol, reference in references.items()
    }
    return reporting_frame, reporting_references


def test_floor_fixed_manifest_publishes_after_both_validators_accept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frame, references, sensitivity_dir = _ready_floor_publication_bundle(tmp_path)
    frame, references = _with_reporting_publication_identity(frame, references)
    run_csv = sensitivity_script._write_content_addressed_csv(frame, sensitivity_dir)
    validated: list[str] = []

    def accept_bundle(
        manifest: pd.DataFrame,
        symbol: str,
        *,
        primary_reference: PrimaryFloorReference,
        trace_root: Path,
        require_hash_suffix: bool,
    ) -> dict[str, bool]:
        assert len(manifest) == 12
        assert primary_reference is references[symbol]
        assert trace_root == sensitivity_dir
        assert require_hash_suffix is True
        validated.append(symbol)
        return {"ready": True}

    monkeypatch.setattr(
        sensitivity_script,
        "evaluate_floor_sensitivity",
        accept_bundle,
    )

    published = sensitivity_script._publish_validated_sensitivity_manifest(
        frame,
        run_csv=run_csv,
        sensitivity_dir=sensitivity_dir,
        config="reporting",
        requested_outcomes=["P", "N"],
        primary_floor_references=references,
    )

    assert published == sensitivity_dir / "tau_prior_sensitivity.csv"
    assert published.read_bytes() == run_csv.read_bytes()
    assert validated == ["P", "N"]


def test_invalid_exact_floor_reporting_run_preserves_fixed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frame, references, sensitivity_dir = _ready_floor_publication_bundle(tmp_path)
    frame, references = _with_reporting_publication_identity(frame, references)
    run_csv = sensitivity_script._write_content_addressed_csv(frame, sensitivity_dir)
    fixed = sensitivity_dir / "tau_prior_sensitivity.csv"
    fixed.write_text("old certified floor manifest\n", encoding="utf-8")
    validated: list[str] = []

    def reject_n_bundle(
        _manifest: pd.DataFrame,
        symbol: str,
        **_kwargs,
    ) -> dict[str, bool]:
        validated.append(symbol)
        return {"ready": symbol == "P"}

    monkeypatch.setattr(
        sensitivity_script,
        "evaluate_floor_sensitivity",
        reject_n_bundle,
    )

    with pytest.raises(RuntimeError, match="P/N trace-backed validation failed"):
        sensitivity_script._publish_validated_sensitivity_manifest(
            frame,
            run_csv=run_csv,
            sensitivity_dir=sensitivity_dir,
            config="reporting",
            requested_outcomes=["P", "N"],
            primary_floor_references=references,
        )

    assert fixed.read_text(encoding="utf-8") == "old certified floor manifest\n"
    assert run_csv.is_file()
    assert validated == ["P", "N"]


def test_floor_publication_rejects_a_stale_run_csv_for_a_different_frame(
    tmp_path: Path,
):
    frame, references, sensitivity_dir = _ready_floor_publication_bundle(tmp_path)
    frame, references = _with_reporting_publication_identity(frame, references)
    run_csv = sensitivity_script._write_content_addressed_csv(frame, sensitivity_dir)
    changed = frame.copy(deep=True)
    changed.loc[0, "risk_difference_mean"] += 0.01
    fixed = sensitivity_dir / "tau_prior_sensitivity.csv"
    fixed.write_text("old certified floor manifest\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not exactly represent"):
        sensitivity_script._publish_validated_sensitivity_manifest(
            changed,
            run_csv=run_csv,
            sensitivity_dir=sensitivity_dir,
            config="reporting",
            requested_outcomes=["P", "N"],
            primary_floor_references=references,
        )

    assert fixed.read_text(encoding="utf-8") == "old certified floor manifest\n"
    assert run_csv.is_file()


def test_atomic_manifest_publication_preserves_old_file_on_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "run.csv"
    destination = tmp_path / "tau_prior_sensitivity.csv"
    source.write_text("new manifest\n", encoding="utf-8")
    destination.write_text("old manifest\n", encoding="utf-8")

    def fail_after_partial_copy(_source: Path, temporary: Path) -> None:
        Path(temporary).write_text("partial", encoding="utf-8")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(
        sensitivity_script.shutil,
        "copyfile",
        fail_after_partial_copy,
    )

    with pytest.raises(OSError, match="simulated interrupted copy"):
        sensitivity_script._publish_csv_atomically(source, destination)

    assert destination.read_text(encoding="utf-8") == "old manifest\n"
    assert list(tmp_path.glob(".tau_prior_sensitivity-*")) == []


def test_nonconverged_grid_exits_without_copy_and_preserves_central_traces(
    tmp_path: Path,
):
    sensitivity_dir = tmp_path / "sensitivity"
    model_root = tmp_path / "models"
    report_dir = model_root / f"lrp-rli-itt-009-{FIXTURE_CONFIG}"
    reference = _make_primary_reference(
        report_dir,
        config_name=FIXTURE_CONFIG,
    )
    rows = _materialise_grid(sensitivity_dir, reference)
    rows.loc[0, "converged"] = False
    manifest = report_dir / FLOOR_SENSITIVITY_FILENAME
    manifest.write_text("old manifest\n", encoding="utf-8")
    fixed_central = sensitivity_dir / "tau_prior_sensitivity.csv"
    fixed_central.write_text("old central manifest\n", encoding="utf-8")
    run_csv = sensitivity_script._write_content_addressed_csv(
        rows,
        sensitivity_dir,
    )
    central_traces = [sensitivity_dir / name for name in rows["trace_file"]]

    with pytest.raises(RuntimeError, match="refusing to install"):
        sensitivity_script._copy_floor_model_artifacts(
            rows,
            sensitivity_dir=sensitivity_dir,
            model_output_root=model_root,
            config=FIXTURE_CONFIG,
        )

    assert manifest.read_text(encoding="utf-8") == "old manifest\n"
    assert fixed_central.read_text(encoding="utf-8") == "old central manifest\n"
    assert run_csv.is_file()
    assert all(path.is_file() for path in central_traces)
    assert set(report_dir.glob("trace-*.nc")) == set()


def test_floor_report_prominently_gates_conflict_and_unavailable_psense():
    repo = Path(__file__).resolve().parents[2]
    partial = (repo / "docs/models/_partials/_results_floored.qmd").read_text(encoding="utf-8")

    assert FLOOR_SENSITIVITY_FILENAME in partial
    assert "Release gate failed: prior-data conflict is unresolved" in partial
    assert "Release gate failed: tau power-scaling is unavailable" in partial
    assert "not ready for scientific interpretation" in partial
    assert "all-free-variable convergence gate" in partial
    assert "primary_config_sha256" not in partial
    assert "trace-recomputed all-free-variable convergence gate" in partial
    assert "recomputes convergence, risk differences and posterior probabilities" in partial
    assert "cell/provenance hashes match this report" in partial
    assert "risk_difference_median_min" in partial
