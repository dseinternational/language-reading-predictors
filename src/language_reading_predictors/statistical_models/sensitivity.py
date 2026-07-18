# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared contracts for trace-backed ITT sensitivity artefacts.

The standard 44-cell prior sweep and the separate floor-rule P/N grids use
these helpers to decide whether their manifests, primary-fit references, and
persisted traces form complete, auditable bundles. Keeping those decisions out
of the fitting script and Quarto templates makes the release gates testable and
prevents partial, stale, mixed, or fabricated CSV/trace bundles from being
mistaken for completed analyses.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

FLOOR_SENSITIVITY_FILENAME = "floor_tau_prior_sensitivity.csv"
FLOOR_SENSITIVITY_AXIS = "floor_tau_sigma_x_age_adjustment"
FLOOR_SENSITIVITY_SAMPLING_ATTR = "floor_sensitivity_sampling_json"
FLOOR_SENSITIVITY_PROVENANCE_ATTR = "floor_sensitivity_provenance_json"
FLOOR_SENSITIVITY_TAU_SIGMAS = (0.5, 1.0, 1.5)
FLOOR_SENSITIVITY_AGE_ADJUSTMENTS = (False, True)
FLOOR_SENSITIVITY_MODEL_IDS = {
    "P": "lrp-rli-itt-009",
    "N": "lrp-rli-itt-011",
}
STANDARD_SENSITIVITY_FILENAME = "tau_prior_sensitivity.csv"
STANDARD_SENSITIVITY_PROVENANCE_ATTR = "standard_sensitivity_provenance_json"
STANDARD_SENSITIVITY_SAMPLING_ATTR = "standard_sensitivity_sampling_json"
STANDARD_SENSITIVITY_OUTCOMES = ("R", "E", "UR", "UE", "T", "F", "L", "W")
STANDARD_SENSITIVITY_DISTAL_OUTCOMES = ("R", "E", "UR", "UE", "T", "F")
STANDARD_SENSITIVITY_PROXIMAL_OUTCOMES = ("L", "W")
STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS = (0.2, 0.25, 0.3, 0.5)
STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS = (0.25, 0.5, 0.75)
STANDARD_SENSITIVITY_GAMMA_OWN_SIGMAS = (0.25, 0.5)
STANDARD_SENSITIVITY_KAPPA_SIGMAS = (25.0, 50.0, 100.0, 200.0)
STANDARD_SENSITIVITY_MODEL_IDS = {
    "UR": "lrp-rli-itt-003",
    "UE": "lrp-rli-itt-004",
    "R": "lrp-rli-itt-005",
    "E": "lrp-rli-itt-006",
    "L": "lrp-rli-itt-007",
    "W": "lrp-rli-itt-010",
    "F": "lrp-rli-itt-025",
    "T": "lrp-rli-itt-026",
}
TAU_PSENSE_STATUSES = ("conflict", "no_conflict", "unavailable")
TauPsenseStatus = Literal["conflict", "no_conflict", "unavailable"]

_PRIMARY_SAMPLING_KEYS = (
    "draws",
    "tune",
    "chains",
    "target_accept",
    "random_seed",
)
_PRIMARY_MATCHED_SENSITIVITY_KEYS = (
    "draws",
    "tune",
    "chains",
    "target_accept",
)
_SENSITIVITY_SAMPLING_COLUMNS = (
    "sampling_draws",
    "sampling_tune",
    "sampling_chains",
    "sampling_cores",
    "sampling_target_accept",
    "sampling_random_seed",
)
_PRIMARY_SAMPLING_COLUMNS = tuple(
    f"primary_sampling_{key}" for key in _PRIMARY_SAMPLING_KEYS
)

_FLOOR_REQUIRED_COLUMNS = {
    "config",
    "outcome",
    "model_id",
    "estimand",
    "analysis_subset",
    "likelihood",
    "sensitivity_axis",
    "tau_sigma",
    "age_adjusted",
    "use_age_linear",
    "use_own_baseline",
    "data_sha256",
    "n",
    "n_intervention",
    "n_control",
    "primary_config_sha256",
    "primary_trace_sha256",
    *_SENSITIVITY_SAMPLING_COLUMNS,
    "sampling_nuts_sampler",
    *_PRIMARY_SAMPLING_COLUMNS,
    "risk_difference_median",
    "risk_difference_mean",
    "risk_difference_lo50",
    "risk_difference_hi50",
    "risk_difference_lo",
    "risk_difference_hi",
    "risk_difference_hpdi_lo",
    "risk_difference_hpdi_hi",
    "prob_risk_difference_positive",
    "meaningful_risk_difference",
    "prob_risk_difference_ge_0_10",
    "tau_logit_median",
    "tau_logit_lo",
    "tau_logit_hi",
    "converged",
    "max_rhat",
    "min_ess",
    "min_bfmi",
    "n_divergences",
    "free_variables",
    "n_free_variables",
    "convergence_scope",
    "trace_file",
    "trace_sha256",
}

_RISK_DIFFERENCE_COLUMNS = (
    "risk_difference_median",
    "risk_difference_mean",
    "risk_difference_lo50",
    "risk_difference_hi50",
    "risk_difference_lo",
    "risk_difference_hi",
    "risk_difference_hpdi_lo",
    "risk_difference_hpdi_hi",
)

_TRACE_SUMMARY_COLUMNS = {
    "risk_difference_median": "tau_prob_median",
    "risk_difference_mean": "tau_prob_mean",
    "risk_difference_lo50": "tau_prob_lo50",
    "risk_difference_hi50": "tau_prob_hi50",
    "risk_difference_lo": "tau_prob_lo",
    "risk_difference_hi": "tau_prob_hi",
    "risk_difference_hpdi_lo": "tau_prob_hpdi_lo",
    "risk_difference_hpdi_hi": "tau_prob_hpdi_hi",
    "prob_risk_difference_positive": "prob_ame_pos",
    "tau_logit_median": "tau_logit_median",
    "tau_logit_lo": "tau_logit_lo",
    "tau_logit_hi": "tau_logit_hi",
}

_TRACE_CONVERGENCE_COLUMNS = (
    "max_rhat",
    "min_ess",
    "min_bfmi",
)

_STANDARD_N_TRIALS = {
    "R": 170,
    "E": 170,
    "UR": 12,
    "UE": 12,
    "T": 32,
    "F": 18,
    "L": 32,
    "W": 79,
}
_STANDARD_REQUIRED_COLUMNS = {
    "config",
    "outcome",
    "n_trials",
    "sensitivity_axis",
    "tau_sigma",
    "gamma_own_sigma",
    "kappa_sigma",
    "use_precision_terms",
    "data_sha256",
    "n",
    "n_intervention",
    "n_control",
    "primary_model_id",
    "primary_config_sha256",
    "primary_trace_sha256",
    *_PRIMARY_SAMPLING_COLUMNS,
    "pd",
    "tau_logit_mean",
    "tau_logit_lo",
    "tau_logit_hi",
    "ci_width_logit",
    "tau_sd_logit",
    "kappa_median",
    "items_mean",
    "items_lo",
    "items_hi",
    "converged",
    "max_rhat",
    "min_ess",
    "min_bfmi",
    "n_divergences",
    "free_variables",
    "n_free_variables",
    "convergence_scope",
    *_SENSITIVITY_SAMPLING_COLUMNS,
    "sampling_nuts_sampler",
    "trace_file",
    "trace_sha256",
}
_STANDARD_SUMMARY_COLUMNS = {
    "pd": "prob_tau_pos",
    "tau_logit_mean": "tau_logit_mean",
    "tau_logit_lo": "tau_logit_lo",
    "tau_logit_hi": "tau_logit_hi",
}

StandardSensitivityCell = tuple[
    str,
    str,
    float,
    float | None,
    float,
    bool,
]


@dataclass(frozen=True)
class PrimaryFloorReference:
    """Immutable identity and sample metadata for the primary floor-rule fit."""

    model_dir: Path
    config_name: str
    model_id: str
    outcome: str
    data_sha256: str
    n: int
    n_intervention: int
    n_control: int
    config_sha256: str
    trace_sha256: str
    sampling: Mapping[str, int | float]

    def manifest_values(self) -> dict[str, Any]:
        """Return the columns copied into every sensitivity-manifest row."""
        values: dict[str, Any] = {
            "config": self.config_name,
            "model_id": self.model_id,
            "outcome": self.outcome,
            "data_sha256": self.data_sha256,
            "n": self.n,
            "n_intervention": self.n_intervention,
            "n_control": self.n_control,
            "primary_config_sha256": self.config_sha256,
            "primary_trace_sha256": self.trace_sha256,
        }
        values.update(
            {
                f"primary_sampling_{key}": self.sampling[key]
                for key in _PRIMARY_SAMPLING_KEYS
            }
        )
        return values


@dataclass(frozen=True)
class PrimaryStandardReference:
    """Current registered primary fit to which a standard sweep is anchored."""

    model_dir: Path
    config_name: str
    model_id: str
    outcome: str
    data_sha256: str
    n: int
    n_intervention: int
    n_control: int
    config_sha256: str
    trace_sha256: str
    sampling: Mapping[str, int | float]

    def manifest_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "outcome": self.outcome,
            "data_sha256": self.data_sha256,
            "n": self.n,
            "n_intervention": self.n_intervention,
            "n_control": self.n_control,
            "primary_model_id": self.model_id,
            "primary_config_sha256": self.config_sha256,
            "primary_trace_sha256": self.trace_sha256,
        }
        values.update(
            {
                f"primary_sampling_{key}": self.sampling[key]
                for key in _PRIMARY_SAMPLING_KEYS
            }
        )
        return values


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _floor_arm_counts(config: Mapping[str, Any]) -> tuple[int, int]:
    """Read the exploratory-eligible counts from primary run metadata."""
    try:
        floor_rule = config["extra"]["floor_rule"]
        eligibility = floor_rule["eligibility_by_arm"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "primary config lacks extra.floor_rule.eligibility_by_arm"
        ) from exc

    counts: dict[str, int] = {}
    for row in eligibility:
        try:
            arm = str(row["arm"])
            count = int(row["n_exploratory_eligible"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("primary floor eligibility metadata is malformed") from exc
        if arm in counts:
            raise ValueError(f"primary floor eligibility repeats arm {arm!r}")
        counts[arm] = count
    if set(counts) != {"intervention", "control"}:
        raise ValueError(
            "primary floor eligibility must contain intervention and control exactly once"
        )
    if counts["intervention"] <= 0 or counts["control"] <= 0:
        raise ValueError("primary floor analysis must retain both randomised arms")
    return counts["intervention"], counts["control"]


def load_primary_floor_reference(
    model_dir: str | Path,
    outcome_symbol: str,
    *,
    config_name: str,
) -> PrimaryFloorReference:
    """Load and hash the primary fit that a sensitivity manifest must certify.

    The report directory itself is authoritative. The returned reference binds
    a grid to its current ``config.json`` and ``trace.nc`` bytes, data digest,
    model/outcome identity, analysis-set size, arm counts, and sampling settings.
    """
    directory = Path(model_dir)
    config_path = directory / "config.json"
    trace_path = directory / "trace.nc"
    if not config_path.is_file():
        raise FileNotFoundError(f"primary config does not exist: {config_path}")
    if not trace_path.is_file():
        raise FileNotFoundError(f"primary trace does not exist: {trace_path}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"primary config is not readable JSON: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("primary config must contain a JSON object")

    expected_model_id = FLOOR_SENSITIVITY_MODEL_IDS.get(outcome_symbol)
    if expected_model_id is None:
        raise ValueError(f"unsupported floored outcome {outcome_symbol!r}")
    if str(config.get("model_id")) != expected_model_id:
        raise ValueError(
            f"primary model mismatch: expected {expected_model_id!r}, "
            f"got {config.get('model_id')!r}"
        )
    if str(config.get("outcome_symbol")) != outcome_symbol:
        raise ValueError(
            f"primary outcome mismatch: expected {outcome_symbol!r}, "
            f"got {config.get('outcome_symbol')!r}"
        )

    data_sha256 = str(config.get("data_sha256", "")).strip().lower()
    if not _is_sha256(data_sha256):
        raise ValueError("primary config lacks a valid data_sha256")
    try:
        n = int(config["n_obs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("primary config lacks a valid n_obs") from exc
    n_intervention, n_control = _floor_arm_counts(config)
    if n <= 0 or n_intervention + n_control != n:
        raise ValueError(
            "primary n_obs does not equal the sum of exploratory-eligible arm counts"
        )
    try:
        at_risk_n = int(config["extra"]["floor_rule"]["at_risk_n"])
        floor_outcome = str(config["extra"]["floor_rule"]["outcome"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("primary config lacks floor-rule identity metadata") from exc
    if at_risk_n != n or floor_outcome != outcome_symbol:
        raise ValueError("primary floor-rule identity or analysis-set size is inconsistent")

    sampling_raw = config.get("sampling")
    if not isinstance(sampling_raw, dict):
        raise ValueError("primary config lacks sampling provenance")
    sampling: dict[str, int | float] = {}
    for key in _PRIMARY_SAMPLING_KEYS:
        if key not in sampling_raw:
            raise ValueError(f"primary sampling metadata lacks {key!r}")
        try:
            value: int | float
            if key == "target_accept":
                value = float(sampling_raw[key])
                if not 0.0 < value <= 1.0:
                    raise ValueError
            else:
                value = int(sampling_raw[key])
                if value <= 0:
                    raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"primary sampling metadata has invalid {key!r}") from exc
        sampling[key] = value

    try:
        posterior_variables, posterior_sizes, _posterior_attrs = _posterior_metadata(
            trace_path
        )
    except Exception as exc:  # noqa: BLE001 - corrupt primary artefact is gate data
        raise ValueError(f"primary trace is not a readable NetCDF: {trace_path}") from exc
    missing_primary_variables = {"alpha", "tau"} - posterior_variables
    if missing_primary_variables:
        raise ValueError(
            "primary trace posterior lacks required variables: "
            + ", ".join(sorted(missing_primary_variables))
        )
    if (
        int(posterior_sizes.get("chain", -1)) != sampling["chains"]
        or int(posterior_sizes.get("draw", -1)) != sampling["draws"]
    ):
        raise ValueError(
            "primary trace posterior chain/draw dimensions do not match config sampling"
        )

    return PrimaryFloorReference(
        model_dir=directory,
        config_name=str(config_name),
        model_id=expected_model_id,
        outcome=outcome_symbol,
        data_sha256=data_sha256,
        n=n,
        n_intervention=n_intervention,
        n_control=n_control,
        config_sha256=sha256_file(config_path),
        trace_sha256=sha256_file(trace_path),
        sampling=sampling,
    )


def load_primary_standard_reference(
    model_dir: str | Path,
    outcome_symbol: str,
    *,
    config_name: str,
) -> PrimaryStandardReference:
    """Load the current registered primary ITT identity for one standard outcome."""
    import arviz as az

    directory = Path(model_dir)
    config_path = directory / "config.json"
    trace_path = directory / "trace.nc"
    if not config_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError(
            f"primary standard fit is incomplete: {directory}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"primary config is not readable JSON: {config_path}") from exc
    expected_model_id = STANDARD_SENSITIVITY_MODEL_IDS.get(outcome_symbol)
    if expected_model_id is None:
        raise ValueError(f"unsupported standard sensitivity outcome {outcome_symbol!r}")
    if str(config.get("model_id")) != expected_model_id:
        raise ValueError(
            f"primary model mismatch for {outcome_symbol}: expected "
            f"{expected_model_id}, got {config.get('model_id')!r}"
        )
    if str(config.get("outcome_symbol")) != outcome_symbol:
        raise ValueError(f"primary outcome mismatch for {outcome_symbol}")
    data_sha256 = str(config.get("data_sha256", "")).strip().lower()
    if not _is_sha256(data_sha256):
        raise ValueError("primary config lacks a valid data_sha256")
    n = _required_int(config.get("n_obs"), "primary n_obs", positive=True)
    sampling_raw = config.get("sampling")
    if not isinstance(sampling_raw, dict):
        raise ValueError("primary config lacks sampling provenance")
    sampling: dict[str, int | float] = {}
    for key in _PRIMARY_SAMPLING_KEYS:
        if key not in sampling_raw:
            raise ValueError(f"primary sampling metadata lacks {key!r}")
        if key == "target_accept":
            value: int | float = _required_float(
                sampling_raw[key], f"primary sampling {key}"
            )
            if not 0.0 < value <= 1.0:
                raise ValueError(f"primary sampling metadata has invalid {key!r}")
        else:
            value = _required_int(
                sampling_raw[key], f"primary sampling {key}", positive=True
            )
        sampling[key] = value

    try:
        trace = az.from_netcdf(trace_path)
    except Exception as exc:  # noqa: BLE001 - corrupt primary artefact is gate data
        raise ValueError(f"primary trace is not a readable NetCDF: {trace_path}") from exc
    try:
        posterior = getattr(trace, "posterior", None)
        if posterior is None or not {"alpha", "tau"}.issubset(posterior.data_vars):
            raise ValueError("primary trace posterior lacks alpha or tau")
        if (
            int(posterior.sizes.get("chain", -1)) != sampling["chains"]
            or int(posterior.sizes.get("draw", -1)) != sampling["draws"]
        ):
            raise ValueError(
                "primary trace posterior chain/draw dimensions do not match config"
            )
        constant_data = getattr(trace, "constant_data", None)
        if constant_data is None or "G" not in constant_data:
            raise ValueError("primary trace constant_data lacks G")
        G = np.asarray(constant_data["G"].values, dtype=float).reshape(-1)
        if G.size != n or not np.isin(G, (0.0, 1.0)).all():
            raise ValueError("primary trace treatment assignment is inconsistent")
        n_intervention = int(np.sum(G == 1.0))
        n_control = int(np.sum(G == 0.0))
        if n_intervention <= 0 or n_control <= 0:
            raise ValueError("primary trace must contain both randomised arms")
    finally:
        close = getattr(trace, "close", None)
        if callable(close):
            close()
    return PrimaryStandardReference(
        model_dir=directory,
        config_name=str(config_name),
        model_id=expected_model_id,
        outcome=outcome_symbol,
        data_sha256=data_sha256,
        n=n,
        n_intervention=n_intervention,
        n_control=n_control,
        config_sha256=sha256_file(config_path),
        trace_sha256=sha256_file(trace_path),
        sampling=sampling,
    )


def load_primary_standard_references(
    model_output_root: str | Path,
    *,
    config_name: str,
) -> dict[str, PrimaryStandardReference]:
    """Load all eight current primary references for the standard sweep."""
    root = Path(model_output_root)
    return {
        outcome: load_primary_standard_reference(
            root / f"{model_id}-{config_name}",
            outcome,
            config_name=config_name,
        )
        for outcome, model_id in STANDARD_SENSITIVITY_MODEL_IDS.items()
    }


def _as_bool(value: Any) -> bool | None:
    """Parse a CSV boolean without treating arbitrary non-empty strings as true."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised == "true":
            return True
        if normalised == "false":
            return False
    return None


def _required_bool(value: Any, label: str) -> bool:
    parsed = _as_bool(value)
    if parsed is None:
        raise ValueError(f"{label} is not a boolean")
    return parsed


def _required_int(value: Any, label: str, *, positive: bool = False) -> int:
    """Parse an integer without silently truncating a fractional CSV value."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} is not an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{label} is not an integer")
    parsed = int(numeric)
    if positive and parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _required_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} is not finite")
    return parsed


def floor_trace_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical identity embedded in one floor-sensitivity trace.

    The identity intentionally repeats the manifest's exact cell, data, primary
    fit, model, free-variable, and sampling metadata. A trace can therefore be
    verified independently of its filename, and a valid trace from another grid
    cell cannot be substituted together with its matching digest.
    """
    age_adjusted = _required_bool(row["age_adjusted"], "age_adjusted")
    free_variables = [
        name.strip()
        for name in str(row["free_variables"]).split("|")
        if name.strip()
    ]
    expected_free_variables = (
        ["alpha", "tau", "gamma_A"] if age_adjusted else ["alpha", "tau"]
    )
    if free_variables != expected_free_variables:
        raise ValueError(
            "free_variables do not match the floor model's ordered free variables"
        )
    n_free_variables = _required_int(
        row["n_free_variables"], "n_free_variables", positive=True
    )
    if n_free_variables != len(free_variables):
        raise ValueError("n_free_variables does not match free_variables")

    sampling = {
        "draws": _required_int(row["sampling_draws"], "sampling_draws", positive=True),
        "tune": _required_int(row["sampling_tune"], "sampling_tune", positive=True),
        "chains": _required_int(
            row["sampling_chains"], "sampling_chains", positive=True
        ),
        "cores": _required_int(row["sampling_cores"], "sampling_cores", positive=True),
        "target_accept": _required_float(
            row["sampling_target_accept"], "sampling_target_accept"
        ),
        "random_seed": _required_int(
            row["sampling_random_seed"], "sampling_random_seed", positive=True
        ),
        "nuts_sampler": str(row["sampling_nuts_sampler"]),
    }
    primary_sampling = {
        "draws": _required_int(
            row["primary_sampling_draws"], "primary_sampling_draws", positive=True
        ),
        "tune": _required_int(
            row["primary_sampling_tune"], "primary_sampling_tune", positive=True
        ),
        "chains": _required_int(
            row["primary_sampling_chains"], "primary_sampling_chains", positive=True
        ),
        "target_accept": _required_float(
            row["primary_sampling_target_accept"],
            "primary_sampling_target_accept",
        ),
        "random_seed": _required_int(
            row["primary_sampling_random_seed"],
            "primary_sampling_random_seed",
            positive=True,
        ),
    }
    return {
        "schema_version": 1,
        "config": str(row["config"]),
        "outcome": str(row["outcome"]),
        "model_id": str(row["model_id"]),
        "estimand": str(row["estimand"]),
        "analysis_subset": str(row["analysis_subset"]),
        "likelihood": str(row["likelihood"]),
        "sensitivity_axis": str(row["sensitivity_axis"]),
        "tau_sigma": _required_float(row["tau_sigma"], "tau_sigma"),
        "age_adjusted": age_adjusted,
        "use_age_linear": _required_bool(row["use_age_linear"], "use_age_linear"),
        "use_own_baseline": _required_bool(
            row["use_own_baseline"], "use_own_baseline"
        ),
        "data_sha256": str(row["data_sha256"]),
        "n": _required_int(row["n"], "n", positive=True),
        "n_intervention": _required_int(
            row["n_intervention"], "n_intervention", positive=True
        ),
        "n_control": _required_int(row["n_control"], "n_control", positive=True),
        "primary_config_sha256": str(row["primary_config_sha256"]),
        "primary_trace_sha256": str(row["primary_trace_sha256"]),
        "primary_sampling": primary_sampling,
        "free_variables": free_variables,
        "sampling": sampling,
    }


def _optional_float(value: Any, label: str) -> float | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return _required_float(value, label)


def _standard_expected_cells() -> set[StandardSensitivityCell]:
    cells: set[StandardSensitivityCell] = set()
    for outcome in STANDARD_SENSITIVITY_DISTAL_OUTCOMES:
        cells.update(
            (outcome, "tau_sigma", sigma, 0.25, 50.0, True)
            for sigma in STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS
        )
    for outcome in STANDARD_SENSITIVITY_PROXIMAL_OUTCOMES:
        cells.update(
            (outcome, "tau_sigma", sigma, 0.25, 50.0, True)
            for sigma in STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS
        )
        cells.update(
            (outcome, "gamma_own_sigma", 0.5, sigma, 50.0, True)
            for sigma in STANDARD_SENSITIVITY_GAMMA_OWN_SIGMAS
        )
        cells.add((outcome, "unadjusted_benchmark", 0.5, None, 50.0, False))
        cells.update(
            (outcome, "kappa_sigma", 0.5, 0.25, sigma, True)
            for sigma in STANDARD_SENSITIVITY_KAPPA_SIGMAS
        )
    return cells


def _standard_cell(row: Mapping[str, Any]) -> StandardSensitivityCell:
    return (
        str(row["outcome"]),
        str(row["sensitivity_axis"]),
        _required_float(row["tau_sigma"], "tau_sigma"),
        _optional_float(row["gamma_own_sigma"], "gamma_own_sigma"),
        _required_float(row["kappa_sigma"], "kappa_sigma"),
        _required_bool(row["use_precision_terms"], "use_precision_terms"),
    )


def standard_trace_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical model/run identity for one standard-sweep trace."""
    cell = _standard_cell(row)
    use_precision_terms = cell[-1]
    free_variables = [
        name.strip()
        for name in str(row["free_variables"]).split("|")
        if name.strip()
    ]
    expected_free_variables = (
        ["alpha", "tau", "gamma_own", "gamma_A", "kappa"]
        if use_precision_terms
        else ["alpha", "tau", "kappa"]
    )
    if free_variables != expected_free_variables:
        raise ValueError(
            "free_variables do not match the standard ITT model's ordered free variables"
        )
    if _required_int(
        row["n_free_variables"], "n_free_variables", positive=True
    ) != len(free_variables):
        raise ValueError("n_free_variables does not match free_variables")
    sampling = {
        "draws": _required_int(row["sampling_draws"], "sampling_draws", positive=True),
        "tune": _required_int(row["sampling_tune"], "sampling_tune", positive=True),
        "chains": _required_int(
            row["sampling_chains"], "sampling_chains", positive=True
        ),
        "cores": _required_int(row["sampling_cores"], "sampling_cores", positive=True),
        "target_accept": _required_float(
            row["sampling_target_accept"], "sampling_target_accept"
        ),
        "random_seed": _required_int(
            row["sampling_random_seed"], "sampling_random_seed", positive=True
        ),
        "nuts_sampler": str(row["sampling_nuts_sampler"]),
    }
    primary_sampling = {
        "draws": _required_int(
            row["primary_sampling_draws"], "primary_sampling_draws", positive=True
        ),
        "tune": _required_int(
            row["primary_sampling_tune"], "primary_sampling_tune", positive=True
        ),
        "chains": _required_int(
            row["primary_sampling_chains"], "primary_sampling_chains", positive=True
        ),
        "target_accept": _required_float(
            row["primary_sampling_target_accept"],
            "primary_sampling_target_accept",
        ),
        "random_seed": _required_int(
            row["primary_sampling_random_seed"],
            "primary_sampling_random_seed",
            positive=True,
        ),
    }
    outcome, axis, tau_sigma, gamma_own_sigma, kappa_sigma, _ = cell
    return {
        "schema_version": 1,
        "config": str(row["config"]),
        "outcome": outcome,
        "model_kind": "itt",
        "likelihood": "beta_binomial",
        "sensitivity_axis": axis,
        "tau_sigma": tau_sigma,
        "gamma_own_sigma": gamma_own_sigma,
        "kappa_sigma": kappa_sigma,
        "use_precision_terms": use_precision_terms,
        "n_trials": _required_int(row["n_trials"], "n_trials", positive=True),
        "data_sha256": str(row["data_sha256"]),
        "n": _required_int(row["n"], "n", positive=True),
        "n_intervention": _required_int(
            row["n_intervention"], "n_intervention", positive=True
        ),
        "n_control": _required_int(row["n_control"], "n_control", positive=True),
        "primary_model_id": str(row["primary_model_id"]),
        "primary_config_sha256": str(row["primary_config_sha256"]),
        "primary_trace_sha256": str(row["primary_trace_sha256"]),
        "primary_sampling": primary_sampling,
        "free_variables": free_variables,
        "sampling": sampling,
    }


def tau_psense_status(psense: pd.DataFrame | None) -> TauPsenseStatus:
    """Classify the explicit, unique ``tau`` power-scaling diagnosis.

    Missing, duplicated, or unrecognised ``tau`` rows are unavailable rather
    than silently treated as no conflict. This is deliberately fail-closed for
    a release gate.
    """
    if psense is None or psense.empty or "diagnosis" not in psense.columns:
        return "unavailable"
    tau_mask = pd.Index(psense.index).astype(str).str.strip().str.casefold() == "tau"
    tau_rows = psense.loc[tau_mask, "diagnosis"]
    if len(tau_rows) != 1:
        return "unavailable"
    diagnosis = str(tau_rows.iloc[0]).strip()
    normalised = diagnosis.casefold()
    if diagnosis == "✓" or normalised in {
        "ok",
        "no concern",
        "no conflict",
        "no prior-data conflict",
    }:
        return "no_conflict"
    if "prior-data conflict" in normalised:
        return "conflict"
    return "unavailable"


def psense_has_prior_data_conflict(psense: pd.DataFrame | None) -> bool:
    """Compatibility wrapper; prefer :func:`tau_psense_status` for release gates."""
    return tau_psense_status(psense) == "conflict"


def _posterior_metadata(
    path: Path,
) -> tuple[set[str], Mapping[str, int], Mapping[str, Any]]:
    """Open one NetCDF trace and return posterior variables, sizes, and attributes."""
    import arviz as az

    try:
        trace = az.from_netcdf(path)
    except Exception as exc:  # noqa: BLE001 - corrupt artefact is validation data
        raise ValueError(f"unreadable NetCDF ({exc})") from exc
    try:
        posterior = getattr(trace, "posterior", None)
        if posterior is None:
            raise ValueError("trace has no posterior group")
        return set(posterior.data_vars), dict(posterior.sizes), dict(posterior.attrs)
    finally:
        close = getattr(trace, "close", None)
        if callable(close):
            close()


def _values_close(recorded: Any, recomputed: Any) -> bool:
    """Compare a CSV round-trip with a deterministic trace recomputation."""
    try:
        recorded_float = float(recorded)
        recomputed_float = float(recomputed)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(recorded_float)
        and np.isfinite(recomputed_float)
        and np.isclose(
            recorded_float,
            recomputed_float,
            rtol=1e-10,
            atol=1e-12,
        )
    )


def _validate_floor_trace(path: Path, row: Mapping[str, Any]) -> None:
    """Recompute the release evidence from one provenance-bound NetCDF trace."""
    import arviz as az

    from language_reading_predictors.statistical_models import diagnostics as _diag
    from language_reading_predictors.statistical_models.reporting import (
        REPORTING_CI_PROB,
        rope_summary,
        tau_summary_offfloor,
    )

    expected_provenance = floor_trace_provenance(row)
    expected_sampling = expected_provenance["sampling"]
    free_variables = expected_provenance["free_variables"]
    try:
        trace = az.from_netcdf(path)
    except Exception as exc:  # noqa: BLE001 - corrupt artefact is validation data
        raise ValueError(f"unreadable NetCDF ({exc})") from exc
    try:
        posterior = getattr(trace, "posterior", None)
        if posterior is None:
            raise ValueError("trace has no posterior group")
        missing_free_variables = sorted(
            set(free_variables) - set(posterior.data_vars)
        )
        if missing_free_variables:
            raise ValueError(
                "missing posterior variables " + ", ".join(missing_free_variables)
            )
        if "eta" not in posterior:
            raise ValueError("missing posterior variable eta")
        if (
            _required_int(posterior.sizes.get("chain", -1), "posterior chain")
            != expected_sampling["chains"]
            or _required_int(posterior.sizes.get("draw", -1), "posterior draw")
            != expected_sampling["draws"]
        ):
            raise ValueError(
                "posterior dimensions do not match sampling provenance"
            )

        try:
            trace_provenance = json.loads(
                str(posterior.attrs[FLOOR_SENSITIVITY_PROVENANCE_ATTR])
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("missing or malformed trace provenance") from exc
        canonical_trace_provenance = json.dumps(
            trace_provenance,
            sort_keys=True,
            separators=(",", ":"),
        )
        canonical_expected_provenance = json.dumps(
            expected_provenance,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical_trace_provenance != canonical_expected_provenance:
            raise ValueError("trace provenance does not match manifest")

        try:
            trace_sampling = json.loads(
                str(posterior.attrs[FLOOR_SENSITIVITY_SAMPLING_ATTR])
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("missing or malformed sampling provenance") from exc
        if trace_sampling != expected_sampling:
            raise ValueError("sampling provenance does not match manifest")

        constant_data = getattr(trace, "constant_data", None)
        if constant_data is None or "G" not in constant_data:
            raise ValueError("trace constant_data lacks treatment assignment G")
        G = np.asarray(constant_data["G"].values, dtype=float).reshape(-1)
        if (
            G.size != expected_provenance["n"]
            or not np.isfinite(G).all()
            or not np.isin(G, (0.0, 1.0)).all()
            or int(np.sum(G == 1.0)) != expected_provenance["n_intervention"]
            or int(np.sum(G == 0.0)) != expected_provenance["n_control"]
        ):
            raise ValueError("trace treatment assignments do not match manifest")

        convergence = _diag.subfit_convergence(
            trace,
            label=(
                f"{expected_provenance['outcome']} floor trace validation "
                f"tau={expected_provenance['tau_sigma']:g} "
                f"age={'on' if expected_provenance['age_adjusted'] else 'off'}"
            ),
            var_names=free_variables,
        )
        if convergence["converged"] is not True:
            raise ValueError("trace does not pass recomputed convergence gate")
        if _required_bool(row["converged"], "converged") is not True:
            raise ValueError("manifest convergence flag does not match trace")
        for column in _TRACE_CONVERGENCE_COLUMNS:
            if not _values_close(row[column], convergence[column]):
                raise ValueError(f"manifest {column} does not match trace")
        if _required_int(row["n_divergences"], "n_divergences") != int(
            convergence["n_divergences"]
        ):
            raise ValueError("manifest n_divergences does not match trace")

        summary = tau_summary_offfloor(trace, ci_prob=REPORTING_CI_PROB, G=G)
        magnitude = rope_summary(
            trace,
            G=G,
            n_trials=1,
            delta=0.10,
            ci_prob=REPORTING_CI_PROB,
            varying_term="",
        )
        for column, summary_key in _TRACE_SUMMARY_COLUMNS.items():
            if not _values_close(row[column], summary[summary_key]):
                raise ValueError(f"manifest {column} does not match trace")
        if not _values_close(
            row["prob_risk_difference_ge_0_10"],
            magnitude["prob_benefit_ge_delta"],
        ):
            raise ValueError(
                "manifest prob_risk_difference_ge_0_10 does not match trace"
            )
    finally:
        close = getattr(trace, "close", None)
        if callable(close):
            close()


def _validate_standard_trace(path: Path, row: Mapping[str, Any]) -> None:
    """Recompute convergence and headline summaries from one standard trace."""
    import arviz as az

    from language_reading_predictors.statistical_models import diagnostics as _diag
    from language_reading_predictors.statistical_models.reporting import (
        REPORTING_CI_PROB,
        tau_summary_itt,
    )

    expected_provenance = standard_trace_provenance(row)
    expected_sampling = expected_provenance["sampling"]
    free_variables = expected_provenance["free_variables"]
    try:
        trace = az.from_netcdf(path)
    except Exception as exc:  # noqa: BLE001 - corrupt artefact is validation data
        raise ValueError(f"unreadable NetCDF ({exc})") from exc
    try:
        posterior = getattr(trace, "posterior", None)
        if posterior is None:
            raise ValueError("trace has no posterior group")
        missing_variables = sorted(
            (set(free_variables) | {"eta"}) - set(posterior.data_vars)
        )
        if missing_variables:
            raise ValueError(
                "missing posterior variables " + ", ".join(missing_variables)
            )
        if (
            _required_int(posterior.sizes.get("chain", -1), "posterior chain")
            != expected_sampling["chains"]
            or _required_int(posterior.sizes.get("draw", -1), "posterior draw")
            != expected_sampling["draws"]
        ):
            raise ValueError("posterior dimensions do not match sampling provenance")

        try:
            trace_provenance = json.loads(
                str(posterior.attrs[STANDARD_SENSITIVITY_PROVENANCE_ATTR])
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("missing or malformed trace provenance") from exc
        if json.dumps(
            trace_provenance, sort_keys=True, separators=(",", ":")
        ) != json.dumps(
            expected_provenance, sort_keys=True, separators=(",", ":")
        ):
            raise ValueError("trace provenance does not match manifest")
        try:
            trace_sampling = json.loads(
                str(posterior.attrs[STANDARD_SENSITIVITY_SAMPLING_ATTR])
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("missing or malformed sampling provenance") from exc
        if trace_sampling != expected_sampling:
            raise ValueError("sampling provenance does not match manifest")

        constant_data = getattr(trace, "constant_data", None)
        if constant_data is None or "G" not in constant_data:
            raise ValueError("trace constant_data lacks treatment assignment G")
        G = np.asarray(constant_data["G"].values, dtype=float).reshape(-1)
        if (
            G.size != expected_provenance["n"]
            or not np.isfinite(G).all()
            or not np.isin(G, (0.0, 1.0)).all()
            or int(np.sum(G == 1.0)) != expected_provenance["n_intervention"]
            or int(np.sum(G == 0.0)) != expected_provenance["n_control"]
        ):
            raise ValueError("trace treatment assignments do not match manifest")

        convergence = _diag.subfit_convergence(
            trace,
            label=(
                f"{expected_provenance['outcome']} standard sensitivity trace "
                f"{expected_provenance['sensitivity_axis']}"
            ),
            var_names=free_variables,
        )
        if convergence["converged"] is not True:
            raise ValueError("trace does not pass recomputed convergence gate")
        if _required_bool(row["converged"], "converged") is not True:
            raise ValueError("manifest convergence flag does not match trace")
        for column in _TRACE_CONVERGENCE_COLUMNS:
            if not _values_close(row[column], convergence[column]):
                raise ValueError(f"manifest {column} does not match trace")
        if _required_int(row["n_divergences"], "n_divergences") != int(
            convergence["n_divergences"]
        ):
            raise ValueError("manifest n_divergences does not match trace")

        summary = tau_summary_itt(trace, ci_prob=REPORTING_CI_PROB, G=G)
        for column, summary_key in _STANDARD_SUMMARY_COLUMNS.items():
            if not _values_close(row[column], summary[summary_key]):
                raise ValueError(f"manifest {column} does not match trace")
        n_trials = expected_provenance["n_trials"]
        derived = {
            "ci_width_logit": summary["tau_logit_hi"] - summary["tau_logit_lo"],
            "tau_sd_logit": float(np.std(np.asarray(posterior["tau"].values))),
            "kappa_median": float(np.median(np.asarray(posterior["kappa"].values))),
            "items_mean": summary["tau_prob_mean"] * n_trials,
            "items_lo": summary["tau_prob_lo"] * n_trials,
            "items_hi": summary["tau_prob_hi"] * n_trials,
        }
        for column, recomputed in derived.items():
            if not _values_close(row[column], recomputed):
                raise ValueError(f"manifest {column} does not match trace")
    finally:
        close = getattr(trace, "close", None)
        if callable(close):
            close()


def _row_values_match_reference(
    rows: pd.DataFrame,
    reference: PrimaryFloorReference | PrimaryStandardReference,
) -> bool:
    expected = reference.manifest_values()
    for column, value in expected.items():
        if isinstance(value, float):
            observed = pd.to_numeric(rows[column], errors="coerce")
            if not np.isfinite(observed).all() or not np.allclose(
                observed.to_numpy(dtype=float), value, rtol=0.0, atol=1e-12
            ):
                return False
        elif isinstance(value, int):
            observed = pd.to_numeric(rows[column], errors="coerce")
            if not np.isfinite(observed).all() or not observed.eq(value).all():
                return False
        elif not rows[column].astype(str).eq(str(value)).all():
            return False
    return True


def _sensitivity_sampling_matches_reference(
    rows: pd.DataFrame,
    reference: PrimaryFloorReference | PrimaryStandardReference,
) -> bool:
    """Require the sensitivity effort to match its primary sampling preset.

    The sensitivity seed is intentionally independent, and ``cores`` is an
    execution detail. Draws, tuning iterations, chains, and target acceptance
    define the inferential sampling contract and must not be silently reduced.
    """
    for key in _PRIMARY_MATCHED_SENSITIVITY_KEYS:
        observed = pd.to_numeric(rows[f"sampling_{key}"], errors="coerce")
        expected = reference.sampling[key]
        if not np.isfinite(observed).all():
            return False
        if key == "target_accept":
            if not np.allclose(
                observed.to_numpy(dtype=float),
                float(expected),
                rtol=0.0,
                atol=1e-12,
            ):
                return False
        elif not observed.eq(int(expected)).all():
            return False
    return True


def evaluate_standard_sensitivity(
    sensitivity: pd.DataFrame | None,
    *,
    config_name: str = "reporting",
    requested_outcomes: Iterable[str] = STANDARD_SENSITIVITY_OUTCOMES,
    primary_references: Mapping[str, PrimaryStandardReference] | None = None,
    trace_root: str | Path | None = None,
    require_hash_suffix: bool = True,
) -> dict[str, Any]:
    """Fail closed unless the standard ITT sweep is the exact trace-backed grid."""
    expected = _standard_expected_cells()
    requested = tuple(str(value) for value in requested_outcomes)
    requested_aligned = bool(
        len(requested) == len(STANDARD_SENSITIVITY_OUTCOMES)
        and set(requested) == set(STANDARD_SENSITIVITY_OUTCOMES)
    )
    result: dict[str, Any] = {
        "expected_n": len(expected),
        "observed_n": 0,
        "complete": False,
        "converged": False,
        "requested_run_aligned": requested_aligned,
        "primary_aligned": False,
        "traces_present": False,
        "traces_validated": False,
        "ready": False,
        "missing_cells": sorted(expected, key=str),
    }
    if sensitivity is None or sensitivity.empty:
        return result
    missing_columns = sorted(_STANDARD_REQUIRED_COLUMNS - set(sensitivity.columns))
    if missing_columns:
        result["missing_columns"] = missing_columns
        return result

    rows = sensitivity.copy()
    result["observed_n"] = int(len(rows))
    row_records = rows.to_dict(orient="records")
    cells: list[StandardSensitivityCell] = []
    cell_errors: list[str] = []
    for index, row in enumerate(row_records):
        try:
            cells.append(_standard_cell(row))
        except (KeyError, TypeError, ValueError) as exc:
            cell_errors.append(f"row {index}: {exc}")
    observed = set(cells)
    result["missing_cells"] = sorted(expected - observed, key=str)
    if cell_errors:
        result["cell_errors"] = cell_errors

    numeric_columns = [
        "n_trials",
        "tau_sigma",
        "kappa_sigma",
        "n",
        "n_intervention",
        "n_control",
        *_PRIMARY_SAMPLING_COLUMNS,
        "pd",
        "tau_logit_mean",
        "tau_logit_lo",
        "tau_logit_hi",
        "ci_width_logit",
        "tau_sd_logit",
        "kappa_median",
        "items_mean",
        "items_lo",
        "items_hi",
        "max_rhat",
        "min_ess",
        "min_bfmi",
        "n_divergences",
        "n_free_variables",
        *_SENSITIVITY_SAMPLING_COLUMNS,
    ]
    numeric = rows[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_complete = bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
    integer_columns = [
        "n_trials",
        "n",
        "n_intervention",
        "n_control",
        "n_divergences",
        "n_free_variables",
        "sampling_draws",
        "sampling_tune",
        "sampling_chains",
        "sampling_cores",
        "sampling_random_seed",
        "primary_sampling_draws",
        "primary_sampling_tune",
        "primary_sampling_chains",
        "primary_sampling_random_seed",
    ]
    integer_values = numeric[integer_columns].to_numpy(dtype=float)
    integer_contract = bool(
        np.isfinite(integer_values).all()
        and np.equal(integer_values, np.floor(integer_values)).all()
    )
    provenance_contract = True
    provenance_errors: list[str] = []
    for index, row in enumerate(row_records):
        try:
            standard_trace_provenance(row)
        except (KeyError, TypeError, ValueError) as exc:
            provenance_contract = False
            provenance_errors.append(f"row {index}: {exc}")
    if provenance_errors:
        result["provenance_errors"] = provenance_errors

    trial_contract = all(
        outcome in _STANDARD_N_TRIALS
        and np.isfinite(observed_trials)
        and observed_trials == _STANDARD_N_TRIALS[outcome]
        for outcome, observed_trials in zip(
            rows["outcome"].astype(str),
            numeric["n_trials"],
            strict=True,
        )
    )
    sampling_consistent = bool(
        all(
            rows[column].nunique(dropna=False) == 1
            for column in (*_SENSITIVITY_SAMPLING_COLUMNS, "sampling_nuts_sampler")
        )
        and (numeric["sampling_draws"] > 0).all()
        and (numeric["sampling_tune"] > 0).all()
        and (numeric["sampling_chains"] > 0).all()
        and (numeric["sampling_cores"] > 0).all()
        and (numeric["sampling_cores"] <= numeric["sampling_chains"]).all()
        and numeric["sampling_target_accept"].gt(0.0).all()
        and numeric["sampling_target_accept"].le(1.0).all()
        and (numeric["sampling_random_seed"] > 0).all()
        and rows["sampling_nuts_sampler"].astype(str).eq("nutpie").all()
    )
    primary_sampling_contract = bool(
        (numeric["primary_sampling_draws"] > 0).all()
        and (numeric["primary_sampling_tune"] > 0).all()
        and (numeric["primary_sampling_chains"] > 0).all()
        and numeric["primary_sampling_target_accept"].gt(0.0).all()
        and numeric["primary_sampling_target_accept"].le(1.0).all()
        and (numeric["primary_sampling_random_seed"] > 0).all()
    )
    hashes_valid = bool(
        rows["data_sha256"].map(_is_sha256).all()
        and rows["data_sha256"].astype(str).nunique() == 1
        and rows["primary_config_sha256"].map(_is_sha256).all()
        and rows["primary_trace_sha256"].map(_is_sha256).all()
        and rows["trace_sha256"].map(_is_sha256).all()
    )
    coherence = bool(
        numeric["pd"].between(0.0, 1.0).all()
        and (numeric["tau_logit_lo"] <= numeric["tau_logit_mean"]).all()
        and (numeric["tau_logit_mean"] <= numeric["tau_logit_hi"]).all()
        and (numeric["items_lo"] <= numeric["items_mean"]).all()
        and (numeric["items_mean"] <= numeric["items_hi"]).all()
        and (numeric["tau_sd_logit"] > 0.0).all()
        and (numeric["kappa_median"] > 0.0).all()
        and np.allclose(
            numeric["ci_width_logit"].to_numpy(dtype=float),
            (
                numeric["tau_logit_hi"] - numeric["tau_logit_lo"]
            ).to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    contract_complete = bool(
        requested_aligned
        and rows["config"].astype(str).eq(str(config_name)).all()
        and set(rows["outcome"].astype(str)) == set(STANDARD_SENSITIVITY_OUTCOMES)
        and all(
            rows.loc[rows["outcome"].astype(str) == outcome, "primary_model_id"]
            .astype(str)
            .eq(model_id)
            .all()
            for outcome, model_id in STANDARD_SENSITIVITY_MODEL_IDS.items()
        )
        and trial_contract
        and rows["convergence_scope"].astype(str).eq("all_free_variables").all()
        and provenance_contract
        and sampling_consistent
        and primary_sampling_contract
        and hashes_valid
        and integer_contract
        and (numeric["n"] > 0).all()
        and (numeric["n_intervention"] > 0).all()
        and (numeric["n_control"] > 0).all()
        and (
            numeric["n_intervention"] + numeric["n_control"] == numeric["n"]
        ).all()
        and coherence
    )
    result["complete"] = bool(
        len(rows) == len(expected)
        and len(cells) == len(expected)
        and len(observed) == len(expected)
        and observed == expected
        and numeric_complete
        and contract_complete
        and rows["trace_file"].astype(str).str.strip().ne("").all()
        and rows["trace_file"].astype(str).nunique() == len(expected)
        and rows["trace_sha256"].astype(str).nunique() == len(expected)
    )

    from language_reading_predictors.statistical_models.diagnostics import (
        BFMI_THRESHOLD,
        ESS_THRESHOLD,
        RHAT_MAX,
    )

    convergence_flags = rows["converged"].map(_as_bool)
    result["converged"] = bool(
        result["complete"]
        and convergence_flags.notna().all()
        and convergence_flags.all()
        and (numeric["max_rhat"] <= RHAT_MAX).all()
        and (numeric["min_ess"] >= ESS_THRESHOLD).all()
        and (numeric["min_bfmi"] >= BFMI_THRESHOLD).all()
        and (numeric["n_divergences"] == 0).all()
    )

    if primary_references is not None and set(primary_references) == set(
        STANDARD_SENSITIVITY_OUTCOMES
    ):
        result["primary_aligned"] = all(
            reference.outcome == outcome
            and reference.model_id == STANDARD_SENSITIVITY_MODEL_IDS[outcome]
            and reference.config_name == str(config_name)
            and _row_values_match_reference(
                rows.loc[rows["outcome"].astype(str) == outcome],
                reference,
            )
            and _sensitivity_sampling_matches_reference(
                rows.loc[rows["outcome"].astype(str) == outcome],
                reference,
            )
            for outcome, reference in primary_references.items()
        )

    trace_paths: list[Path] = []
    if trace_root is not None:
        root = Path(trace_root).resolve()
        for name in rows["trace_file"].astype(str):
            candidate = (root / name).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                trace_paths = []
                break
            trace_paths.append(candidate)
        result["traces_present"] = bool(
            result["complete"]
            and len(trace_paths) == len(expected)
            and all(path.is_file() for path in trace_paths)
        )
    if result["traces_present"]:
        trace_errors: list[str] = []
        for path, recorded_hash, row in zip(
            trace_paths,
            rows["trace_sha256"].astype(str),
            row_records,
            strict=True,
        ):
            actual_hash = sha256_file(path)
            if actual_hash != recorded_hash:
                trace_errors.append(f"{path.name}: SHA-256 mismatch")
                continue
            if require_hash_suffix and not path.stem.endswith(f"-{actual_hash[:12]}"):
                trace_errors.append(f"{path.name}: filename lacks SHA-256 suffix")
                continue
            try:
                _validate_standard_trace(path, row)
            except Exception as exc:  # noqa: BLE001 - malformed trace is gate data
                trace_errors.append(f"{path.name}: {exc}")
        result["traces_validated"] = not trace_errors
        if trace_errors:
            result["trace_errors"] = trace_errors
    result["ready"] = bool(
        result["complete"]
        and result["converged"]
        and result["requested_run_aligned"]
        and result["primary_aligned"]
        and result["traces_present"]
        and result["traces_validated"]
    )
    return result


def evaluate_floor_sensitivity(
    sensitivity: pd.DataFrame | None,
    outcome_symbol: str,
    *,
    primary_reference: PrimaryFloorReference | None = None,
    trace_root: str | Path | None = None,
    require_hash_suffix: bool = True,
    trace_exists: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate the required 3 x 2 floored-outcome treatment-prior grid.

    A grid is ready only when every tau-SD/age-adjustment cell occurs exactly
    once, all results are finite and coherent, every fit passes the shared
    all-free-variable convergence gate, its manifest matches the *current*
    primary config and trace, and every sensitivity trace is readable, matches
    its recorded SHA-256 digest and embedded cell provenance, passes convergence
    when recomputed, and reproduces every reported effect summary from its draws.

    ``trace_exists`` is retained only for compatibility with older callers. It
    can establish presence, but never trace readability, so it cannot by itself
    clear the release gate.
    """
    expected = {
        (tau_sigma, age_adjusted)
        for tau_sigma in FLOOR_SENSITIVITY_TAU_SIGMAS
        for age_adjusted in FLOOR_SENSITIVITY_AGE_ADJUSTMENTS
    }
    result: dict[str, Any] = {
        "expected_n": len(expected),
        "observed_n": 0,
        "complete": False,
        "converged": False,
        "primary_aligned": False,
        "traces_present": False,
        "traces_validated": False,
        "ready": False,
        "missing_cells": sorted(expected),
        "risk_difference_median_min": np.nan,
        "risk_difference_median_max": np.nan,
        "risk_difference_interval_min": np.nan,
        "risk_difference_interval_max": np.nan,
        "prob_positive_min": np.nan,
        "prob_positive_max": np.nan,
        "prob_meaningful_min": np.nan,
        "prob_meaningful_max": np.nan,
    }
    if sensitivity is None or sensitivity.empty:
        return result

    expected_model_id = FLOOR_SENSITIVITY_MODEL_IDS.get(outcome_symbol)
    if expected_model_id is None:
        result["unsupported_outcome"] = outcome_symbol
        return result

    missing_columns = sorted(_FLOOR_REQUIRED_COLUMNS - set(sensitivity.columns))
    if missing_columns:
        result["missing_columns"] = missing_columns
        return result

    rows = sensitivity.loc[
        sensitivity["outcome"].astype(str) == outcome_symbol
    ].copy()
    result["observed_n"] = int(len(rows))
    if rows.empty:
        return result

    rows["tau_sigma"] = pd.to_numeric(rows["tau_sigma"], errors="coerce")
    rows["_age_bool"] = rows["age_adjusted"].map(_as_bool)
    cells = [
        (float(tau_sigma), bool(age_adjusted))
        for tau_sigma, age_adjusted in zip(
            rows["tau_sigma"], rows["_age_bool"], strict=True
        )
        if np.isfinite(tau_sigma) and age_adjusted is not None
    ]
    observed = set(cells)
    result["missing_cells"] = sorted(expected - observed)

    numeric_columns = [
        *_RISK_DIFFERENCE_COLUMNS,
        "tau_logit_median",
        "tau_logit_lo",
        "tau_logit_hi",
        "prob_risk_difference_positive",
        "meaningful_risk_difference",
        "prob_risk_difference_ge_0_10",
        "max_rhat",
        "min_ess",
        "min_bfmi",
        "n_divergences",
        "n_free_variables",
        "n",
        "n_intervention",
        "n_control",
        *_SENSITIVITY_SAMPLING_COLUMNS,
        *_PRIMARY_SAMPLING_COLUMNS,
    ]
    numeric = rows[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric_complete = bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
    free_variable_lists = [
        [name.strip() for name in str(value).split("|") if name.strip()]
        for value in rows["free_variables"]
    ]
    free_variable_contract = all(
        age_adjusted is not None
        and names
        == (["alpha", "tau", "gamma_A"] if age_adjusted else ["alpha", "tau"])
        and len(names) == n_free
        for names, age_adjusted, n_free in zip(
            free_variable_lists,
            rows["_age_bool"],
            numeric["n_free_variables"],
            strict=True,
        )
        if np.isfinite(n_free)
    ) and bool(np.isfinite(numeric["n_free_variables"]).all())
    risk_differences_bounded = bool(
        numeric[list(_RISK_DIFFERENCE_COLUMNS)].ge(-1.0).all().all()
        and numeric[list(_RISK_DIFFERENCE_COLUMNS)].le(1.0).all().all()
    )
    hashes_valid = bool(
        rows["data_sha256"].map(_is_sha256).all()
        and rows["primary_config_sha256"].map(_is_sha256).all()
        and rows["primary_trace_sha256"].map(_is_sha256).all()
        and rows["trace_sha256"].map(_is_sha256).all()
    )
    integer_columns = [
        "n",
        "n_intervention",
        "n_control",
        "n_free_variables",
        "n_divergences",
        "sampling_draws",
        "sampling_tune",
        "sampling_chains",
        "sampling_cores",
        "sampling_random_seed",
        "primary_sampling_draws",
        "primary_sampling_tune",
        "primary_sampling_chains",
        "primary_sampling_random_seed",
    ]
    integer_contract = bool(
        np.equal(
            numeric[integer_columns].to_numpy(dtype=float),
            np.floor(numeric[integer_columns].to_numpy(dtype=float)),
        ).all()
    )
    sampling_contract = bool(
        (numeric["sampling_draws"] > 0).all()
        and (numeric["sampling_tune"] > 0).all()
        and (numeric["sampling_chains"] > 0).all()
        and (numeric["sampling_cores"] > 0).all()
        and numeric["sampling_target_accept"].gt(0.0).all()
        and numeric["sampling_target_accept"].le(1.0).all()
        and (numeric["sampling_random_seed"] > 0).all()
        and (numeric["sampling_cores"] <= numeric["sampling_chains"]).all()
        and all(numeric[column].nunique() == 1 for column in _SENSITIVITY_SAMPLING_COLUMNS)
        and rows["sampling_nuts_sampler"].astype(str).eq("nutpie").all()
    )
    contract_complete = bool(
        rows["model_id"].astype(str).eq(expected_model_id).all()
        and rows["estimand"]
        .astype(str)
        .eq("off_floor_risk_difference_given_observed_baseline_floor")
        .all()
        and rows["analysis_subset"]
        .astype(str)
        .eq("observed_baseline_floor")
        .all()
        and rows["likelihood"].astype(str).eq("bernoulli_offfloor").all()
        and rows["sensitivity_axis"].astype(str).eq(FLOOR_SENSITIVITY_AXIS).all()
        and rows["config"].astype(str).str.strip().ne("").all()
        and rows["config"].astype(str).nunique() == 1
        and rows["use_age_linear"].map(_as_bool).equals(rows["_age_bool"])
        and rows["use_own_baseline"].map(_as_bool).eq(False).all()
        and rows["convergence_scope"]
        .astype(str)
        .eq("all_free_variables")
        .all()
        and free_variable_contract
        and integer_contract
        and hashes_valid
        and sampling_contract
        and (numeric["n_free_variables"] > 0).all()
        and (numeric["n"] > 0).all()
        and numeric["n"].nunique() == 1
        and (numeric["n_intervention"] > 0).all()
        and (numeric["n_control"] > 0).all()
        and (
            numeric["n_intervention"] + numeric["n_control"] == numeric["n"]
        ).all()
        and numeric["n_intervention"].nunique() == 1
        and numeric["n_control"].nunique() == 1
        and (
            numeric["risk_difference_lo"] <= numeric["risk_difference_median"]
        ).all()
        and (
            numeric["risk_difference_median"] <= numeric["risk_difference_hi"]
        ).all()
        and (
            numeric["risk_difference_lo50"] <= numeric["risk_difference_median"]
        ).all()
        and (
            numeric["risk_difference_median"] <= numeric["risk_difference_hi50"]
        ).all()
        and (
            numeric["risk_difference_hpdi_lo"]
            <= numeric["risk_difference_hpdi_hi"]
        ).all()
        and (numeric["tau_logit_lo"] <= numeric["tau_logit_median"]).all()
        and (numeric["tau_logit_median"] <= numeric["tau_logit_hi"]).all()
        and risk_differences_bounded
        and numeric["prob_risk_difference_positive"].between(0.0, 1.0).all()
        and numeric["meaningful_risk_difference"].eq(0.10).all()
        and numeric["prob_risk_difference_ge_0_10"].between(0.0, 1.0).all()
    )
    result["complete"] = bool(
        len(rows) == len(expected)
        and len(cells) == len(expected)
        and len(set(cells)) == len(expected)
        and observed == expected
        and numeric_complete
        and contract_complete
        and rows["trace_file"].astype(str).str.strip().ne("").all()
        and rows["trace_file"].astype(str).nunique() == len(expected)
    )

    from language_reading_predictors.statistical_models.diagnostics import (
        BFMI_THRESHOLD,
        ESS_THRESHOLD,
        RHAT_MAX,
    )

    convergence_flags = rows["converged"].map(_as_bool)
    result["converged"] = bool(
        result["complete"]
        and convergence_flags.notna().all()
        and convergence_flags.all()
        and (numeric["max_rhat"] <= RHAT_MAX).all()
        and (numeric["min_ess"] >= ESS_THRESHOLD).all()
        and (numeric["min_bfmi"] >= BFMI_THRESHOLD).all()
        and (numeric["n_divergences"] == 0).all()
    )

    if primary_reference is not None:
        result["primary_aligned"] = bool(
            primary_reference.outcome == outcome_symbol
            and primary_reference.model_id == expected_model_id
            and _row_values_match_reference(rows, primary_reference)
            and _sensitivity_sampling_matches_reference(rows, primary_reference)
        )

    trace_names = rows["trace_file"].astype(str).tolist()
    trace_paths: list[Path] = []
    if trace_root is not None:
        root = Path(trace_root).resolve()
        for name in trace_names:
            candidate = (root / name).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                trace_paths = []
                break
            trace_paths.append(candidate)
        result["traces_present"] = bool(
            result["complete"]
            and len(trace_paths) == len(expected)
            and all(path.is_file() for path in trace_paths)
        )
    elif trace_exists is not None:
        result["traces_present"] = bool(
            result["complete"] and all(trace_exists(name) for name in trace_names)
        )

    if result["traces_present"] and trace_paths:
        traces_validated = True
        trace_errors: list[str] = []
        for path, recorded_hash, row in zip(
            trace_paths,
            rows["trace_sha256"].astype(str),
            rows.to_dict(orient="records"),
            strict=True,
        ):
            actual_hash = sha256_file(path)
            if actual_hash != recorded_hash:
                traces_validated = False
                trace_errors.append(f"{path.name}: SHA-256 mismatch")
                continue
            if require_hash_suffix and not path.stem.endswith(f"-{actual_hash[:12]}"):
                traces_validated = False
                trace_errors.append(f"{path.name}: filename lacks SHA-256 suffix")
                continue
            try:
                _validate_floor_trace(path, row)
            except Exception as exc:  # noqa: BLE001 - malformed trace is gate data
                traces_validated = False
                trace_errors.append(f"{path.name}: {exc}")
        result["traces_validated"] = traces_validated
        if trace_errors:
            result["trace_errors"] = trace_errors

    result["ready"] = bool(
        result["complete"]
        and result["converged"]
        and result["primary_aligned"]
        and result["traces_present"]
        and result["traces_validated"]
    )

    if numeric_complete:
        result.update(
            risk_difference_median_min=float(numeric["risk_difference_median"].min()),
            risk_difference_median_max=float(numeric["risk_difference_median"].max()),
            risk_difference_interval_min=float(numeric["risk_difference_lo"].min()),
            risk_difference_interval_max=float(numeric["risk_difference_hi"].max()),
            prob_positive_min=float(numeric["prob_risk_difference_positive"].min()),
            prob_positive_max=float(numeric["prob_risk_difference_positive"].max()),
            prob_meaningful_min=float(
                numeric["prob_risk_difference_ge_0_10"].min()
            ),
            prob_meaningful_max=float(
                numeric["prob_risk_difference_ge_0_10"].max()
            ),
        )
    return result
