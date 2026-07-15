# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for the trace-backed standard ITT prior-sensitivity archive."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from language_reading_predictors.statistical_models import diagnostics as _diag
from language_reading_predictors.statistical_models import sensitivity as _sensitivity
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.reporting import tau_summary_itt
from language_reading_predictors.statistical_models.sensitivity import (
    STANDARD_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_MODEL_IDS,
    STANDARD_SENSITIVITY_OUTCOMES,
    STANDARD_SENSITIVITY_PROVENANCE_ATTR,
    STANDARD_SENSITIVITY_SAMPLING_ATTR,
    PrimaryStandardReference,
    evaluate_standard_sensitivity,
    sha256_file,
    standard_trace_provenance,
)

# ``scripts`` is not an installed package, so load the CLI independently of
# whether pytest itself placed the repository root on ``sys.path``.
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tau_prior_sensitivity.py"
_SCRIPT_SPEC = spec_from_file_location("_lrp_standard_tau_prior_sensitivity", _SCRIPT_PATH)
if _SCRIPT_SPEC is None or _SCRIPT_SPEC.loader is None:
    raise ImportError(f"cannot load sensitivity runner from {_SCRIPT_PATH}")
sensitivity_script = module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(sensitivity_script)


DISTAL_SIGMAS = (0.2, 0.25, 0.3, 0.5)
PROXIMAL_SIGMAS = (0.25, 0.5, 0.75)
BASELINE_SIGMAS = (0.25, 0.5)
KAPPA_SIGMAS = (25.0, 50.0, 100.0, 200.0)
FIXTURE_CONFIG = "dev"
FIXTURE_SAMPLING = {
    "draws": 500,
    "tune": 500,
    "chains": 2,
    "target_accept": 0.85,
}


def test_cli_rejects_mixed_standard_and_floor_outcomes_before_output_setup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    def unexpected_output_setup(_output_dir) -> None:
        raise AssertionError("output setup must not run for a mixed request")

    monkeypatch.setattr(
        sensitivity_script._paths,
        "set_output_root",
        unexpected_output_setup,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["tau_prior_sensitivity.py", "--outcomes", "P", "R"],
    )

    with pytest.raises(SystemExit) as exc_info:
        sensitivity_script.main()

    assert exc_info.value.code == 2
    assert "cannot be mixed in one run" in capsys.readouterr().err


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _cell_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outcome in STANDARD_SENSITIVITY_OUTCOMES:
        tau_sigmas = DISTAL_SIGMAS if outcome not in {"L", "W"} else PROXIMAL_SIGMAS
        for tau_sigma in tau_sigmas:
            rows.append(
                _base_row(
                    outcome,
                    axis="tau_sigma",
                    tau_sigma=tau_sigma,
                    gamma_own_sigma=0.25,
                    kappa_sigma=50.0,
                    use_precision_terms=True,
                )
            )
    for outcome in ("L", "W"):
        for gamma_own_sigma in BASELINE_SIGMAS:
            rows.append(
                _base_row(
                    outcome,
                    axis="gamma_own_sigma",
                    tau_sigma=0.5,
                    gamma_own_sigma=gamma_own_sigma,
                    kappa_sigma=50.0,
                    use_precision_terms=True,
                )
            )
        rows.append(
            _base_row(
                outcome,
                axis="unadjusted_benchmark",
                tau_sigma=0.5,
                gamma_own_sigma=None,
                kappa_sigma=50.0,
                use_precision_terms=False,
            )
        )
        for kappa_sigma in KAPPA_SIGMAS:
            rows.append(
                _base_row(
                    outcome,
                    axis="kappa_sigma",
                    tau_sigma=0.5,
                    gamma_own_sigma=0.25,
                    kappa_sigma=kappa_sigma,
                    use_precision_terms=True,
                )
            )
    assert len(rows) == 44
    return rows


def _base_row(
    outcome: str,
    *,
    axis: str,
    tau_sigma: float,
    gamma_own_sigma: float | None,
    kappa_sigma: float,
    use_precision_terms: bool,
) -> dict[str, object]:
    free_variables = "alpha|tau|gamma_own|gamma_A|kappa" if use_precision_terms else "alpha|tau|kappa"
    return {
        "config": FIXTURE_CONFIG,
        "outcome": outcome,
        "n_trials": MEASURES[outcome].n_trials,
        "sensitivity_axis": axis,
        "tau_sigma": tau_sigma,
        "gamma_own_sigma": gamma_own_sigma,
        "kappa_sigma": kappa_sigma,
        "use_precision_terms": use_precision_terms,
        "data_sha256": _sha("standard-sensitivity-input"),
        "n": 12,
        "n_intervention": 6,
        "n_control": 6,
        "primary_model_id": STANDARD_SENSITIVITY_MODEL_IDS[outcome],
        "primary_config_sha256": _sha(f"primary-config-{outcome}"),
        "primary_trace_sha256": _sha(f"primary-trace-{outcome}"),
        "primary_sampling_draws": FIXTURE_SAMPLING["draws"],
        "primary_sampling_tune": FIXTURE_SAMPLING["tune"],
        "primary_sampling_chains": FIXTURE_SAMPLING["chains"],
        "primary_sampling_target_accept": FIXTURE_SAMPLING["target_accept"],
        "primary_sampling_random_seed": 47,
        "sampling_draws": FIXTURE_SAMPLING["draws"],
        "sampling_tune": FIXTURE_SAMPLING["tune"],
        "sampling_chains": FIXTURE_SAMPLING["chains"],
        "sampling_cores": FIXTURE_SAMPLING["chains"],
        "sampling_target_accept": FIXTURE_SAMPLING["target_accept"],
        "sampling_random_seed": 20260701,
        "sampling_nuts_sampler": "nutpie",
        "pd": 0.75,
        "tau_logit_mean": 0.20,
        "tau_logit_lo": -0.20,
        "tau_logit_hi": 0.60,
        "ci_width_logit": 0.80,
        "tau_sd_logit": 0.20,
        "kappa_median": 50.0,
        "items_mean": 0.5,
        "items_lo": -0.5,
        "items_hi": 1.5,
        "converged": True,
        "max_rhat": 1.0,
        "min_ess": 800.0,
        "min_bfmi": 1.0,
        "n_divergences": 0,
        "free_variables": free_variables,
        "n_free_variables": len(free_variables.split("|")),
        "convergence_scope": "all_free_variables",
        "trace_file": "placeholder.nc",
        "trace_sha256": "0" * 64,
    }


def _independent_chains(
    rng: np.random.Generator,
    chains: int,
    draws: int,
    *,
    mean: float,
    sd: float,
) -> np.ndarray:
    base = rng.normal(mean, sd, size=draws)
    return np.stack([base[rng.permutation(draws)] for _ in range(chains)])


def _write_cell_trace(path: Path, row: pd.Series, *, seed: int) -> None:
    provenance = standard_trace_provenance(row)
    sampling = provenance["sampling"]
    chains = int(sampling["chains"])
    draws = int(sampling["draws"])
    n = int(row["n"])
    rng = np.random.default_rng(seed)
    alpha = _independent_chains(rng, chains, draws, mean=-0.6, sd=0.25)
    tau = _independent_chains(
        rng,
        chains,
        draws,
        mean=0.18 + seed / 10000,
        sd=0.20,
    )
    kappa = np.exp(_independent_chains(rng, chains, draws, mean=np.log(50.0), sd=0.15))
    G = np.r_[np.ones(n // 2), np.zeros(n - n // 2)]
    values: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "alpha": (("chain", "draw"), alpha),
        "tau": (("chain", "draw"), tau),
        "kappa": (("chain", "draw"), kappa),
    }
    eta = alpha[:, :, None] + tau[:, :, None] * G[None, None, :]
    if bool(row["use_precision_terms"]):
        gamma_own = _independent_chains(rng, chains, draws, mean=1.0, sd=0.15)
        gamma_age = _independent_chains(rng, chains, draws, mean=0.05, sd=0.10)
        own_pre = np.linspace(-0.7, 0.7, n)
        age = np.linspace(-1.0, 1.0, n)
        eta = eta + gamma_own[:, :, None] * own_pre[None, None, :] + gamma_age[:, :, None] * age[None, None, :]
        values["gamma_own"] = (("chain", "draw"), gamma_own)
        values["gamma_A"] = (("chain", "draw"), gamma_age)
    values["eta"] = (("chain", "draw", "obs_id"), eta)

    posterior = xr.Dataset(
        values,
        coords={
            "chain": np.arange(chains),
            "draw": np.arange(draws),
            "obs_id": np.arange(n),
        },
    )
    posterior.attrs[STANDARD_SENSITIVITY_SAMPLING_ATTR] = json.dumps(
        sampling,
        sort_keys=True,
        separators=(",", ":"),
    )
    posterior.attrs[STANDARD_SENSITIVITY_PROVENANCE_ATTR] = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    )
    sample_stats = xr.Dataset(
        {
            "diverging": (
                ("chain", "draw"),
                np.zeros((chains, draws), dtype=bool),
            ),
            "energy": (("chain", "draw"), rng.normal(size=(chains, draws))),
        },
        coords={"chain": np.arange(chains), "draw": np.arange(draws)},
    )
    constant_data = xr.Dataset(
        {"G": (("obs_id",), G)},
        coords={"obs_id": np.arange(n)},
    )
    xr.DataTree.from_dict(
        {
            "posterior": posterior,
            "sample_stats": sample_stats,
            "constant_data": constant_data,
        }
    ).to_netcdf(path)


def _materialise_ready_grid(trace_root: Path) -> pd.DataFrame:
    trace_root.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(_cell_rows())
    evidence_by_precision: dict[bool, dict[str, object]] = {}
    for index, row in rows.iterrows():
        temporary = trace_root / f".standard-{index}.nc"
        precision = bool(row["use_precision_terms"])
        _write_cell_trace(temporary, row, seed=500 if precision else 900)
        if precision not in evidence_by_precision:
            trace = az.from_netcdf(temporary)
            try:
                free_variables = str(row["free_variables"]).split("|")
                convergence = _diag.subfit_convergence(
                    trace,
                    label="standard sensitivity test fixture",
                    var_names=free_variables,
                )
                assert convergence["converged"] is True
                G = np.asarray(trace.constant_data["G"].values, dtype=float)
                summary = tau_summary_itt(trace, ci_prob=0.95, G=G)
                tau_draws = trace.posterior["tau"].values.reshape(-1)
                kappa_draws = trace.posterior["kappa"].values.reshape(-1)
            finally:
                trace.close()
            evidence_by_precision[precision] = {
                "pd": summary["prob_tau_pos"],
                "tau_logit_mean": summary["tau_logit_mean"],
                "tau_logit_lo": summary["tau_logit_lo"],
                "tau_logit_hi": summary["tau_logit_hi"],
                "ci_width_logit": summary["tau_logit_hi"] - summary["tau_logit_lo"],
                "tau_sd_logit": float(np.std(tau_draws)),
                "kappa_median": float(np.median(kappa_draws)),
                "tau_prob_mean": summary["tau_prob_mean"],
                "tau_prob_lo": summary["tau_prob_lo"],
                "tau_prob_hi": summary["tau_prob_hi"],
                **convergence,
            }
        cached = evidence_by_precision[precision]
        n_trials = int(row["n_trials"])
        evidence = {
            **{key: value for key, value in cached.items() if not key.startswith("tau_prob_")},
            "items_mean": float(cached["tau_prob_mean"]) * n_trials,
            "items_lo": float(cached["tau_prob_lo"]) * n_trials,
            "items_hi": float(cached["tau_prob_hi"]) * n_trials,
        }
        for column, value in evidence.items():
            rows.at[index, column] = value
        digest = sha256_file(temporary)
        destination = trace_root / f"standard-{index}-{digest[:12]}.nc"
        temporary.replace(destination)
        rows.at[index, "trace_file"] = destination.name
        rows.at[index, "trace_sha256"] = digest
    primary_sampling = {**FIXTURE_SAMPLING, "random_seed": 47}
    rows.attrs["primary_references"] = {
        outcome: PrimaryStandardReference(
            model_dir=trace_root / f"primary-{outcome}",
            config_name=FIXTURE_CONFIG,
            model_id=STANDARD_SENSITIVITY_MODEL_IDS[outcome],
            outcome=outcome,
            data_sha256=_sha("standard-sensitivity-input"),
            n=12,
            n_intervention=6,
            n_control=6,
            config_sha256=_sha(f"primary-config-{outcome}"),
            trace_sha256=_sha(f"primary-trace-{outcome}"),
            sampling=primary_sampling,
        )
        for outcome in STANDARD_SENSITIVITY_OUTCOMES
    }
    return rows


@pytest.fixture(scope="module")
def standard_bundle(tmp_path_factory: pytest.TempPathFactory):
    trace_root = tmp_path_factory.mktemp("standard-prior-traces")
    return _materialise_ready_grid(trace_root), trace_root


def _evaluate(grid: pd.DataFrame, trace_root: Path) -> dict[str, object]:
    return evaluate_standard_sensitivity(
        grid,
        config_name=FIXTURE_CONFIG,
        primary_references=grid.attrs["primary_references"],
        trace_root=trace_root,
    )


def _copy_fixture_traces(
    grid: pd.DataFrame,
    source_root: Path,
    destination_root: Path,
) -> None:
    for name in grid["trace_file"]:
        shutil.copy2(source_root / str(name), destination_root / str(name))


def _stub_convergence_with_fixture_values(
    monkeypatch: pytest.MonkeyPatch,
    grid: pd.DataFrame,
) -> None:
    values_by_variables = {
        str(row.free_variables): {
            "converged": bool(row.converged),
            "max_rhat": float(row.max_rhat),
            "min_ess": float(row.min_ess),
            "min_bfmi": float(row.min_bfmi),
            "n_divergences": int(row.n_divergences),
        }
        for row in grid.itertuples(index=False)
    }

    def fixture_convergence(_trace, *, label: str, var_names: list[str]):
        del label
        return values_by_variables["|".join(var_names)]

    monkeypatch.setattr(_diag, "subfit_convergence", fixture_convergence)


def _validate_only_selected_traces(
    monkeypatch: pytest.MonkeyPatch,
    selected_names: set[str],
) -> None:
    original = _sensitivity._validate_standard_trace

    def selective_validation(path: Path, row) -> None:
        if path.name in selected_names:
            original(path, row)

    monkeypatch.setattr(
        _sensitivity,
        "_validate_standard_trace",
        selective_validation,
    )


def test_standard_sensitivity_requires_exact_trace_validated_44_cell_grid(
    standard_bundle,
    monkeypatch: pytest.MonkeyPatch,
):
    grid, trace_root = standard_bundle
    _stub_convergence_with_fixture_values(monkeypatch, grid)

    status = _evaluate(grid, trace_root)

    assert status["expected_n"] == 44
    assert status["observed_n"] == 44
    assert status["complete"] is True
    assert status["converged"] is True
    assert status["traces_present"] is True
    assert status["traces_validated"] is True
    assert status["ready"] is True
    assert status["missing_cells"] == []


def test_standard_sensitivity_rejects_missing_and_duplicate_cells(standard_bundle):
    grid, trace_root = standard_bundle

    missing = _evaluate(grid.iloc[:-1].copy(), trace_root)
    duplicate_grid = pd.concat([grid, grid.iloc[[0]]], ignore_index=True)
    duplicate = _evaluate(duplicate_grid, trace_root)

    assert missing["observed_n"] == 43
    assert missing["complete"] is False
    assert missing["ready"] is False
    assert missing["missing_cells"]
    assert duplicate["observed_n"] == 45
    assert duplicate["complete"] is False
    assert duplicate["ready"] is False


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("sensitivity_axis", "wrong_axis"),
        ("tau_sigma", 9.0),
        ("gamma_own_sigma", 9.0),
        ("kappa_sigma", 9.0),
        ("config", "reporting"),
        ("outcome", "P"),
        ("use_precision_terms", False),
    ],
)
def test_standard_sensitivity_rejects_wrong_cell_contract(
    standard_bundle,
    mutation: str,
    value: object,
):
    grid, trace_root = standard_bundle
    changed = grid.copy(deep=True)
    changed.loc[0, mutation] = value

    status = _evaluate(changed, trace_root)

    assert status["complete"] is False
    assert status["ready"] is False


def test_standard_sensitivity_rejects_missing_required_column(standard_bundle):
    grid, trace_root = standard_bundle
    changed = grid.drop(columns="data_sha256")

    status = _evaluate(changed, trace_root)

    assert status["complete"] is False
    assert status["missing_columns"] == ["data_sha256"]
    assert status["ready"] is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("data_sha256", "f" * 64),
        ("sampling_draws", 251),
        ("sampling_tune", 251),
        ("sampling_chains", 5),
        ("sampling_cores", 3),
        ("sampling_target_accept", 0.90),
        ("sampling_random_seed", 20260702),
        ("sampling_nuts_sampler", "pymc"),
    ],
)
def test_standard_sensitivity_rejects_inconsistent_data_or_sampling_provenance(
    standard_bundle,
    column: str,
    value: object,
):
    grid, _trace_root = standard_bundle
    changed = grid.copy(deep=True)
    changed.loc[0, column] = value

    status = evaluate_standard_sensitivity(
        changed,
        config_name=FIXTURE_CONFIG,
        trace_root=None,
    )

    assert status["complete"] is False
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
def test_standard_sensitivity_rejects_changed_primary_sampling_contract(
    standard_bundle,
    column: str,
    value: int | float,
):
    grid, _trace_root = standard_bundle
    changed = grid.copy(deep=True)
    changed[column] = value
    if column == "sampling_chains":
        changed["sampling_cores"] = value

    status = evaluate_standard_sensitivity(
        changed,
        config_name=FIXTURE_CONFIG,
        primary_references=grid.attrs["primary_references"],
    )

    assert status["complete"] is True
    assert status["primary_aligned"] is False
    assert status["ready"] is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("tau_logit_mean", np.nan),
        ("pd", 1.01),
        ("tau_logit_lo", 0.8),
        ("ci_width_logit", 99.0),
        ("tau_sd_logit", -0.1),
        ("kappa_median", 0.0),
        ("items_lo", 171.0),
    ],
)
def test_standard_sensitivity_rejects_nonfinite_or_incoherent_summary(
    standard_bundle,
    column: str,
    value: float,
):
    grid, trace_root = standard_bundle
    changed = grid.copy(deep=True)
    changed.loc[0, column] = value

    status = _evaluate(changed, trace_root)

    assert status["complete"] is False
    assert status["ready"] is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("converged", False),
        ("max_rhat", 1.02),
        ("min_ess", 399.0),
        ("min_bfmi", 0.29),
        ("n_divergences", 1),
    ],
)
def test_standard_sensitivity_rejects_failed_convergence_signal(
    standard_bundle,
    column: str,
    value: object,
):
    grid, _trace_root = standard_bundle
    changed = grid.copy(deep=True)
    changed.loc[0, column] = value

    status = evaluate_standard_sensitivity(
        changed,
        config_name=FIXTURE_CONFIG,
        trace_root=None,
    )

    assert status["converged"] is False
    assert status["ready"] is False


def test_standard_sensitivity_rejects_trace_swap_between_tau_cells(
    standard_bundle,
    monkeypatch: pytest.MonkeyPatch,
):
    grid, trace_root = standard_bundle
    changed = grid.copy(deep=True)
    candidates = changed.index[(changed["outcome"] == "R") & (changed["sensitivity_axis"] == "tau_sigma")].tolist()
    first, second = candidates[:2]
    for column in ("trace_file", "trace_sha256"):
        changed.loc[[first, second], column] = changed.loc[[second, first], column].to_numpy()
    _validate_only_selected_traces(
        monkeypatch,
        set(changed.loc[[first, second], "trace_file"].astype(str)),
    )

    status = _evaluate(changed, trace_root)

    assert status["complete"] is True
    assert status["traces_present"] is True
    assert status["traces_validated"] is False
    assert "trace provenance does not match manifest" in " ".join(status["trace_errors"])
    assert status["ready"] is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("max_rhat", 1.0),
        ("min_ess", 99999.0),
        ("tau_logit_mean", 0.25),
        ("items_mean", 0.75),
    ],
)
def test_standard_sensitivity_rejects_fabricated_trace_evidence(
    standard_bundle,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    value: float,
):
    grid, trace_root = standard_bundle
    changed = grid.copy(deep=True)
    changed.loc[0, column] = value
    _validate_only_selected_traces(
        monkeypatch,
        {str(changed.loc[0, "trace_file"])},
    )

    status = _evaluate(changed, trace_root)

    assert status["complete"] is True
    assert status["traces_validated"] is False
    assert f"manifest {column} does not match trace" in " ".join(status["trace_errors"])
    assert status["ready"] is False


def test_standard_sensitivity_rejects_trace_hash_tampering(
    standard_bundle,
    monkeypatch: pytest.MonkeyPatch,
):
    grid, trace_root = standard_bundle
    changed = grid.copy(deep=True)
    changed.loc[0, "trace_sha256"] = "f" * 64
    _validate_only_selected_traces(monkeypatch, set())

    status = _evaluate(changed, trace_root)

    assert status["complete"] is True
    assert status["traces_present"] is True
    assert status["traces_validated"] is False
    assert "SHA-256 mismatch" in " ".join(status["trace_errors"])
    assert status["ready"] is False


@pytest.mark.parametrize(
    ("config", "requested_outcomes"),
    [
        ("dev", list(STANDARD_SENSITIVITY_OUTCOMES)),
        ("reporting", ["R", "E"]),
    ],
)
def test_subset_or_dev_run_cannot_replace_fixed_standard_manifest(
    standard_bundle,
    tmp_path: Path,
    config: str,
    requested_outcomes: list[str],
):
    grid, trace_root = standard_bundle
    publication_dir = tmp_path / "publication"
    publication_dir.mkdir()
    _copy_fixture_traces(grid, trace_root, publication_dir)
    fixed = publication_dir / STANDARD_SENSITIVITY_FILENAME
    fixed.write_text("old certified manifest\n", encoding="utf-8")
    run_csv = sensitivity_script._write_content_addressed_csv(grid, publication_dir)

    published = sensitivity_script._publish_validated_sensitivity_manifest(
        grid,
        run_csv=run_csv,
        sensitivity_dir=publication_dir,
        config=config,
        requested_outcomes=requested_outcomes,
    )

    assert published is None
    assert fixed.read_text(encoding="utf-8") == "old certified manifest\n"
    assert run_csv.is_file()
    assert all((publication_dir / name).is_file() for name in grid["trace_file"])


def test_invalid_full_run_preserves_fixed_manifest_and_immutable_run_artifacts(
    standard_bundle,
    tmp_path: Path,
):
    grid, trace_root = standard_bundle
    invalid = grid.iloc[:-1].copy()
    publication_dir = tmp_path / "publication"
    publication_dir.mkdir()
    _copy_fixture_traces(invalid, trace_root, publication_dir)
    fixed = publication_dir / STANDARD_SENSITIVITY_FILENAME
    fixed.write_text("old certified manifest\n", encoding="utf-8")
    run_csv = sensitivity_script._write_content_addressed_csv(invalid, publication_dir)

    with pytest.raises(RuntimeError, match="44-cell trace-backed validation failed"):
        sensitivity_script._publish_validated_sensitivity_manifest(
            invalid,
            run_csv=run_csv,
            sensitivity_dir=publication_dir,
            config="reporting",
            requested_outcomes=list(STANDARD_SENSITIVITY_OUTCOMES),
            primary_references=invalid.attrs["primary_references"],
        )

    assert fixed.read_text(encoding="utf-8") == "old certified manifest\n"
    assert run_csv.is_file()
    assert all((publication_dir / name).is_file() for name in invalid["trace_file"])


def test_fixed_standard_manifest_publishes_the_exact_csv_accepted_by_validator(
    standard_bundle,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    grid, trace_root = standard_bundle
    publication_dir = tmp_path / "publication"
    publication_dir.mkdir()
    _copy_fixture_traces(grid, trace_root, publication_dir)
    run_csv = sensitivity_script._write_content_addressed_csv(grid, publication_dir)
    validated: list[pd.DataFrame] = []

    def accept_bundle(
        manifest: pd.DataFrame,
        *,
        config_name: str,
        requested_outcomes,
        primary_references,
        trace_root: Path,
    ) -> dict[str, bool]:
        assert config_name == "reporting"
        assert set(requested_outcomes) == set(STANDARD_SENSITIVITY_OUTCOMES)
        assert primary_references is grid.attrs["primary_references"]
        assert trace_root == publication_dir
        validated.append(manifest)
        return {"ready": True}

    monkeypatch.setattr(
        sensitivity_script,
        "evaluate_standard_sensitivity",
        accept_bundle,
    )

    published = sensitivity_script._publish_validated_sensitivity_manifest(
        grid,
        run_csv=run_csv,
        sensitivity_dir=publication_dir,
        config="reporting",
        requested_outcomes=list(STANDARD_SENSITIVITY_OUTCOMES),
        primary_references=grid.attrs["primary_references"],
    )

    assert published == publication_dir / STANDARD_SENSITIVITY_FILENAME
    assert published.read_bytes() == run_csv.read_bytes()
    assert len(validated) == 1
    pd.testing.assert_frame_equal(validated[0], pd.read_csv(run_csv))


@pytest.mark.parametrize("config", ["dev", "test"])
def test_exact_floor_nonreporting_run_preserves_fixed_floor_manifest(
    tmp_path: Path,
    config: str,
):
    publication_dir = tmp_path / "floor-only"
    publication_dir.mkdir()
    fixed = publication_dir / STANDARD_SENSITIVITY_FILENAME
    fixed.write_text("old certified floor manifest\n", encoding="utf-8")
    frame = pd.DataFrame({"outcome": ["P", "N"], "not_a_standard_grid": [True, True]})
    run_csv = sensitivity_script._write_content_addressed_csv(frame, publication_dir)

    published = sensitivity_script._publish_validated_sensitivity_manifest(
        frame,
        run_csv=run_csv,
        sensitivity_dir=publication_dir,
        config=config,
        requested_outcomes=["P", "N"],
    )

    assert published is None
    assert fixed.read_text(encoding="utf-8") == "old certified floor manifest\n"
    assert run_csv.is_file()


def test_unvalidated_exact_floor_reporting_run_preserves_fixed_floor_manifest(
    tmp_path: Path,
):
    publication_dir = tmp_path / "floor-only"
    publication_dir.mkdir()
    fixed = publication_dir / STANDARD_SENSITIVITY_FILENAME
    fixed.write_text("old certified floor manifest\n", encoding="utf-8")
    frame = pd.DataFrame({"outcome": ["P", "N"], "not_a_floor_grid": [True, True]})
    run_csv = sensitivity_script._write_content_addressed_csv(frame, publication_dir)

    with pytest.raises(RuntimeError, match="P/N trace-backed validation failed"):
        sensitivity_script._publish_validated_sensitivity_manifest(
            frame,
            run_csv=run_csv,
            sensitivity_dir=publication_dir,
            config="reporting",
            requested_outcomes=["P", "N"],
        )

    assert fixed.read_text(encoding="utf-8") == "old certified floor manifest\n"
    assert run_csv.is_file()


def test_partial_floor_run_cannot_replace_fixed_floor_manifest(tmp_path: Path):
    publication_dir = tmp_path / "floor-only"
    publication_dir.mkdir()
    fixed = publication_dir / STANDARD_SENSITIVITY_FILENAME
    fixed.write_text("old complete floor manifest\n", encoding="utf-8")
    frame = pd.DataFrame({"outcome": ["P"], "not_a_standard_grid": [True]})
    run_csv = sensitivity_script._write_content_addressed_csv(frame, publication_dir)

    published = sensitivity_script._publish_validated_sensitivity_manifest(
        frame,
        run_csv=run_csv,
        sensitivity_dir=publication_dir,
        config="reporting",
        requested_outcomes=["P"],
    )

    assert published is None
    assert fixed.read_text(encoding="utf-8") == "old complete floor manifest\n"
    assert run_csv.is_file()
