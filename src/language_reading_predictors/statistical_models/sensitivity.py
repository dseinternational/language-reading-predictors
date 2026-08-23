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
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
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


# --- Level-factor treatment-prior sweep (#389 criterion 6) -------------------
#
# The robustness release gate (#482) extends to ``level_factors`` on the plan's
# focal t2 term (``d_grp_time[t2]`` under the t1-referenced parameterisation,
# #552; ``b_grp_time[1]`` on the free comparator), and its per-fit evidence check
# (``release._standard_sweep_evidence``) accepts a standard-schema
# ``tau_prior_sensitivity.csv`` bound to the fit's own config/trace hashes.
# These constants define the level-family sweep. The default sweep set stays
# the five outcomes the #389 review names (W, L, P, B, N — all proximal-tier),
# but the model-id map covers every registered LF primary (2026-08-20 review,
# finding 7): the gate's withhold remedy names a treatment-prior sweep, so a
# prior-dominant classification on any of the other six outcomes must have a
# runner path. The runner picks the tau grid by outcome tier
# (``measures.is_distal``): the proximal grid for the directly-taught /
# decoding / floored outcomes, the distal grid (which brackets the registered
# 0.3 scale) for R/E/F/T.

LEVEL_SENSITIVITY_OUTCOMES = ("W", "L", "P", "B", "N")
LEVEL_SENSITIVITY_MODEL_IDS = {
    "W": "lrp-rli-lf-001",
    "R": "lrp-rli-lf-002",
    "E": "lrp-rli-lf-003",
    "L": "lrp-rli-lf-004",
    "P": "lrp-rli-lf-005",
    "B": "lrp-rli-lf-006",
    "F": "lrp-rli-lf-007",
    "T": "lrp-rli-lf-008",
    "TR": "lrp-rli-lf-009",
    "TE": "lrp-rli-lf-010",
    "N": "lrp-rli-lf-011",
}

# Balance-prior sweep grid for the level family's ``arm_gap_t1`` term
# (2026-08-20 review, finding 1): the registered cross-coupling scale (0.3), a
# midpoint, and the review's recommended weakly-informative comparator (1.0).
# The balance term is prior-dominated in most reporting fits and trades off
# directly against the released ``d_grp_time[t2]`` contrast, so this axis
# measures how much of the #552 t1-imbalance subtraction the prior is
# determining. Rows go to a separate ``level_arm_gap_prior_sensitivity.csv``
# and are never attached as gate evidence — the gate's evidence contract is
# the treatment-prior (tau) sweep only.
LEVEL_SENSITIVITY_ARM_GAP_SIGMAS = (0.3, 0.5, 1.0)

# Nuisance-scale axes (#584 decision 4). Power scaling flagged both parameters for
# prior-data conflict across the stored suite -- ``sigma_child`` in all eleven fits
# and the dispersion in eight of the nine graded ones -- so each gets a registered
# axis rather than an exploratory one-off. Both grids **include the pre-decision
# scale**, so the sweep answers "did changing this prior move the answer?" directly
# rather than by comparison with a differently-run fit. Like the arm-gap axis they
# write their own CSV and are never gate evidence.
#
# Dispersion is swept on whichever scale the fit declares: under the default
# ``1/sqrt(kappa)`` parameterisation the grid is the registered 0.25, half it (less
# dispersion admitted) and double it (more).
LEVEL_SENSITIVITY_DISPERSION_SIGMAS = (0.125, 0.25, 0.5)

# Child heterogeneity: the pre-decision gain-model scale, the registered level scale
# and a wider one. 0.5 is the comparator -- it is the prior whose 99th percentile
# two of the eleven fitted posteriors exceeded.
LEVEL_SENSITIVITY_SIGMA_CHILD_SIGMAS = (0.5, 1.0, 1.5)

# The gate likewise covers ``did`` on the plan's focal term (``tau_t2``, or a
# dose model's own slope — ``release.causal_term_for`` mirrors
# ``DiDRunPlan.effect_term``). Unlike the ITT/level sweeps the did set is keyed
# by **model id**, not outcome: two withheld fits share outcome W (the primary
# arm-by-wave lrp-rli-did-001 and the varying-crossover lrp-rli-did-013), so an
# outcome-keyed map cannot name them both. The tuple lists the fits whose
# release decision is prior-dominant and therefore needs sweep evidence (#390);
# extend it if power-scaling flags another did fit — lrp-rli-did-101 (the
# independent-prior intercept companion) flags exactly as its anchored parent
# does, which is itself evidence the anchor is not what makes tau_t2
# prior-dominant. ``mu_dose`` rides its own grid around its ``Normal(0, 1)``
# default with the same half/default/1.5x geometry as the proximal tau grid.

DID_SENSITIVITY_MODEL_IDS = (
    "lrp-rli-did-001",
    "lrp-rli-did-003",
    "lrp-rli-did-007",
    "lrp-rli-did-013",
    "lrp-rli-did-101",
)
DID_SENSITIVITY_MU_DOSE_SIGMAS = (0.5, 1.0, 1.5)

# The gate covers ``gain_factors`` on ``beta_trt`` (#391). Like the did sweep the
# set is keyed by **model id**, not outcome: the taught-vocabulary outcomes each
# have two registered primaries (TR: gf-009/gf-012; TE: gf-010/gf-013), so an
# outcome-keyed map cannot name them all. Every non-companion primary is
# sweepable — which fits *need* the evidence is decided by each refit's own
# power-scaling diagnosis (prior-dominant -> sweep or withhold), not by this
# tuple. Treated-only companions (no ``beta_trt``) and moderation variants
# (``beta_trt`` present but never the causal headline — ``release.gate_applies``
# skips them) are not sweepable.
GF_SENSITIVITY_MODEL_IDS = tuple(f"lrp-rli-gf-{n:03d}" for n in range(1, 14))


def load_primary_level_reference(
    model_dir: str | Path,
    outcome_symbol: str,
    *,
    config_name: str,
) -> PrimaryStandardReference:
    """Load the current registered primary level-factor identity for one outcome.

    The level analogue of :func:`load_primary_standard_reference`: same identity
    and binding checks (model id, outcome, data hash, sampling provenance,
    config/trace sha256), with the family's own posterior contract — the per-wave
    arm-gap vector ``b_grp_time`` is present in every parameterisation (free under
    the pre-#552 comparator, a Deterministic levels view under the t1 reference)
    and ``alpha`` is the anchored Deterministic (#389 finding 2) — and arm counts
    read over the level rows, whose t2 subset is what the randomised contrast is
    estimated from.
    """
    import arviz as az

    directory = Path(model_dir)
    config_path = directory / "config.json"
    trace_path = directory / "trace.nc"
    if not config_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError(f"primary level-factor fit is incomplete: {directory}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"primary config is not readable JSON: {config_path}") from exc
    expected_model_id = LEVEL_SENSITIVITY_MODEL_IDS.get(outcome_symbol)
    if expected_model_id is None:
        raise ValueError(
            f"unsupported level sensitivity outcome {outcome_symbol!r}"
        )
    if str(config.get("model_id")) != expected_model_id:
        raise ValueError(
            f"primary model mismatch for {outcome_symbol}: expected "
            f"{expected_model_id}, got {config.get('model_id')!r}"
        )
    if str(config.get("outcome_symbol")) != outcome_symbol:
        raise ValueError(f"primary outcome mismatch for {outcome_symbol}")
    if str(config.get("kind")) != "level_factors":
        raise ValueError(
            f"primary kind mismatch for {outcome_symbol}: {config.get('kind')!r}"
        )
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
        if posterior is None or not {"alpha", "b_grp_time"}.issubset(
            posterior.data_vars
        ):
            raise ValueError("primary trace posterior lacks alpha or b_grp_time")
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


def did_focal_term(resolved_run_plan: Mapping[str, Any]) -> str:
    """The focal coefficient a did fit's release decision turns on.

    Mirrors ``DiDRunPlan.effect_term`` / ``release.causal_term_for`` from the
    *persisted* plan, so the sweep, the fit's own psense emission and the gate
    cannot disagree about which term matters.
    """
    if resolved_run_plan.get("period_varying"):
        return "mu_dose"
    return "beta_dose" if resolved_run_plan.get("dose") else "tau_t2"


def load_primary_did_reference(
    model_dir: str | Path,
    model_id: str,
    *,
    config_name: str,
) -> PrimaryStandardReference:
    """Load the current registered primary did identity for one swept model.

    The did analogue of :func:`load_primary_level_reference`, keyed by **model
    id** because two swept fits share outcome W. Same identity and binding
    checks (model id, kind, data hash, sampling provenance, config/trace
    sha256); the posterior contract requires the *plan's own* focal term
    (``tau_t2``, or a dose model's slope) rather than a family-wide name.
    """
    import arviz as az

    directory = Path(model_dir)
    config_path = directory / "config.json"
    trace_path = directory / "trace.nc"
    if not config_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError(f"primary did fit is incomplete: {directory}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"primary config is not readable JSON: {config_path}") from exc
    if model_id not in DID_SENSITIVITY_MODEL_IDS:
        raise ValueError(f"unsupported did sensitivity model {model_id!r}")
    if str(config.get("model_id")) != model_id:
        raise ValueError(
            f"primary model mismatch: expected {model_id}, got "
            f"{config.get('model_id')!r}"
        )
    if str(config.get("kind")) != "did":
        raise ValueError(
            f"primary kind mismatch for {model_id}: {config.get('kind')!r}"
        )
    outcome = str(config.get("outcome_symbol") or "")
    if not outcome:
        raise ValueError(f"primary config lacks an outcome_symbol: {model_id}")
    plan = config.get("resolved_run_plan")
    if not isinstance(plan, dict):
        raise ValueError(
            f"primary config lacks a resolved run plan: {model_id}; the sweep "
            "cannot determine the focal term"
        )
    focal = did_focal_term(plan)
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
        if posterior is None or not {"alpha", focal}.issubset(posterior.data_vars):
            raise ValueError(
                f"primary trace posterior lacks alpha or the focal term {focal!r}"
            )
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
        model_id=model_id,
        outcome=outcome,
        data_sha256=data_sha256,
        n=n,
        n_intervention=n_intervention,
        n_control=n_control,
        config_sha256=sha256_file(config_path),
        trace_sha256=sha256_file(trace_path),
        sampling=sampling,
    )


def load_primary_gf_reference(
    model_dir: str | Path,
    model_id: str,
    *,
    config_name: str,
) -> PrimaryStandardReference:
    """Load the current registered primary gain-factor identity for one swept model.

    The gain-factor analogue of :func:`load_primary_did_reference`, keyed by
    **model id** because the taught-vocabulary outcomes each have two registered
    primaries. Same identity and binding checks (model id, kind, data hash,
    sampling provenance, config/trace sha256); the posterior contract requires
    ``beta_trt`` (the family's single causal head), and the plan must be a
    gated headline — a treated-only companion has no ``beta_trt`` and a
    moderation variant's is never released as causal, so neither is sweepable.

    ``n_intervention`` / ``n_control`` are the **period-1** arm counts: the
    stacked later periods are all on-intervention, so the period-1 contrast is
    the identifying sample the sweep's evidence speaks for.
    """
    import arviz as az

    directory = Path(model_dir)
    config_path = directory / "config.json"
    trace_path = directory / "trace.nc"
    if not config_path.is_file() or not trace_path.is_file():
        raise FileNotFoundError(f"primary gain-factor fit is incomplete: {directory}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"primary config is not readable JSON: {config_path}") from exc
    if model_id not in GF_SENSITIVITY_MODEL_IDS:
        raise ValueError(f"unsupported gain-factor sensitivity model {model_id!r}")
    if str(config.get("model_id")) != model_id:
        raise ValueError(
            f"primary model mismatch: expected {model_id}, got "
            f"{config.get('model_id')!r}"
        )
    if str(config.get("kind")) != "gain_factors":
        raise ValueError(
            f"primary kind mismatch for {model_id}: {config.get('kind')!r}"
        )
    outcome = str(config.get("outcome_symbol") or "")
    if not outcome:
        raise ValueError(f"primary config lacks an outcome_symbol: {model_id}")
    plan = config.get("resolved_run_plan")
    if not isinstance(plan, dict):
        raise ValueError(
            f"primary config lacks a resolved run plan: {model_id}; the sweep "
            "cannot confirm the fit is a gated headline"
        )
    if bool(plan.get("treated_only", False)):
        raise ValueError(
            f"{model_id} is a treated-only companion: it has no beta_trt to sweep"
        )
    if bool(plan.get("moderation_variant", False)):
        raise ValueError(
            f"{model_id} is a moderation variant: its beta_trt is never released "
            "as causal, so the gate does not demand sweep evidence for it"
        )
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
        if posterior is None or not {"alpha", "beta_trt"}.issubset(
            posterior.data_vars
        ):
            raise ValueError("primary trace posterior lacks alpha or beta_trt")
        if (
            int(posterior.sizes.get("chain", -1)) != sampling["chains"]
            or int(posterior.sizes.get("draw", -1)) != sampling["draws"]
        ):
            raise ValueError(
                "primary trace posterior chain/draw dimensions do not match config"
            )
        constant_data = getattr(trace, "constant_data", None)
        if constant_data is None or not {"on_intervention", "phase_idx"}.issubset(
            constant_data
        ):
            raise ValueError(
                "primary trace constant_data lacks on_intervention or phase_idx"
            )
        trt = np.asarray(
            constant_data["on_intervention"].values, dtype=float
        ).reshape(-1)
        phase = np.asarray(constant_data["phase_idx"].values, dtype=float).reshape(-1)
        if trt.size != n or phase.size != n or not np.isin(trt, (0.0, 1.0)).all():
            raise ValueError("primary trace treatment assignment is inconsistent")
        p1 = phase == 0.0
        n_intervention = int(np.sum(trt[p1] == 1.0))
        n_control = int(np.sum(trt[p1] == 0.0))
        if n_intervention <= 0 or n_control <= 0:
            raise ValueError(
                "primary trace must contain both randomised arms in period 1"
            )
    finally:
        close = getattr(trace, "close", None)
        if callable(close):
            close()
    return PrimaryStandardReference(
        model_dir=directory,
        config_name=str(config_name),
        model_id=model_id,
        outcome=outcome,
        data_sha256=data_sha256,
        n=n,
        n_intervention=n_intervention,
        n_control=n_control,
        config_sha256=sha256_file(config_path),
        trace_sha256=sha256_file(trace_path),
        sampling=sampling,
    )


def assert_primary_sampling_contract(
    sampling: Any,
    reference: PrimaryStandardReference,
    *,
    config: str,
    keys: tuple[str, ...] = _PRIMARY_MATCHED_SENSITIVITY_KEYS,
    label: str | None = None,
) -> None:
    """Fail before fitting if the selected preset differs from the primary fit.

    The primary may have been produced with a supported ``--target-accept``
    override (or a different preset entirely); sweeping it with mismatched
    sampling and attaching the result would bind evidence produced under a
    different contract. Shared by the family sweep runners (#488 review):

    - the level runner matches every key including ``target_accept`` (no level
      primary carries an override, so any difference is a mistake);
    - the did runner matches ``draws``/``tune``/``chains`` here and *adopts* the
      primary's own recorded ``target_accept`` for its cells, because did-007's
      registered spec legitimately overrides the preset (0.97) and a sweep of
      that fit must reproduce its contract, not refuse it.
    """
    for key in keys:
        observed = getattr(sampling, key)
        if not np.isclose(
            float(observed), float(reference.sampling[key]), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"{label or reference.outcome} sensitivity sampling does not "
                f"match its current {config} primary fit"
            )
    if reference.config_name != config:
        raise RuntimeError(
            f"{label or reference.outcome} sensitivity sampling does not "
            f"match its current {config} primary fit"
        )


def _optional_floats(row: Mapping[str, Any], *columns: str) -> tuple[float, ...] | None:
    """The named columns as floats, or ``None`` when any is absent or blank.

    A sweep runner writes every numeric column, but the schema check only proves
    the columns *exist*; a blank is "not recorded", which is a gap in the
    evidence rather than a contradiction of it, so the caller skips that check
    instead of failing the bundle on a formatting difference.
    """
    values: list[float] = []
    for column in columns:
        value = row.get(column)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None
    return tuple(values)


def _primary_plan_focal_term(reference: PrimaryStandardReference) -> str | None:
    """The focal term the primary's own stored run plan names, if it records one.

    Read from the primary directory rather than the manifest so a sweep cannot
    certify itself against a parameterisation the fit no longer uses (#584
    finding 3). Families whose plan has no ``focal_term`` (the ITT plans name
    ``tau`` through other fields) return ``None`` and are unaffected.
    """
    path = reference.model_dir / "config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    plan = config.get("resolved_run_plan")
    if not isinstance(plan, Mapping):
        return None
    focal = plan.get("focal_term")
    return str(focal) if isinstance(focal, str) and focal else None


def _focal_term_draws(posterior, focal: str, *, outcome: str) -> np.ndarray:
    """Flattened draws of ``focal``, which may name one element of a vector.

    ``d_grp_time[t2]`` is resolved against the variable's own coordinate labels
    when it has them (so a wave label cannot silently drift to another wave) and
    otherwise as a positional index, mirroring how the summaries label it.
    """
    base, _, indexed = focal.partition("[")
    values = posterior[base]
    extra = [d for d in values.dims if d not in ("chain", "draw")]
    if not indexed:
        if extra:
            raise RuntimeError(
                f"{outcome}: focal term {focal!r} names a vector without an "
                "element; the row's summary is not a single coefficient"
            )
        return np.asarray(values.values, dtype=float).reshape(-1)
    if not extra:
        raise RuntimeError(
            f"{outcome}: focal term {focal!r} is indexed but {base!r} is a scalar"
        )
    label = indexed.rstrip("]")
    dim = extra[0]
    coords = (
        [str(c) for c in np.asarray(values.coords[dim].values).reshape(-1)]
        if dim in values.coords
        else []
    )
    if label in coords:
        position = coords.index(label)
    elif label.lstrip("-").isdigit():
        position = int(label)
    else:
        raise RuntimeError(
            f"{outcome}: focal element {label!r} is not a coordinate of "
            f"{base!r}'s {dim!r} dimension (labels: {coords})"
        )
    if not 0 <= position < int(values.sizes[dim]):
        raise RuntimeError(
            f"{outcome}: focal element {label!r} is outside {base!r}'s {dim!r} "
            f"dimension of size {int(values.sizes[dim])}"
        )
    return np.asarray(values.isel({dim: position}).values, dtype=float).reshape(-1)


def _verify_cell_convergence(
    trace, row: Mapping[str, Any], *, outcome: str, free_variables: Sequence[str]
) -> None:
    """Re-run the sub-fit convergence gate on the cell's own trace.

    ``attach_outcome_bundle`` refuses a bundle whose cells did not converge, but
    it used to read that verdict from the CSV the same sweep wrote — so an
    unconverged (or simply mislabelled) cell attached as evidence of prior
    stability (#584 finding 3). The recomputation is the same
    ``subfit_convergence`` the runner used, over the free variables the trace
    itself records, and it fails closed: an uncheckable trace is not a pass.
    """
    from language_reading_predictors.statistical_models.diagnostics import (
        subfit_convergence,
    )

    recomputed = subfit_convergence(
        trace, label=f"{outcome} sweep cell", var_names=list(free_variables) or None
    )
    if recomputed["converged"] is not True:
        raise RuntimeError(
            f"{outcome}: cell trace does not pass the convergence gate when it is "
            f"re-run (converged={recomputed['converged']!r}, "
            f"max_rhat={recomputed['max_rhat']!r}, min_ess={recomputed['min_ess']!r}, "
            f"min_bfmi={recomputed['min_bfmi']!r}), whatever its row claims"
        )
    # The recorded numbers must be the recomputed ones. The tolerance is loose
    # enough to absorb an ArviZ point-release difference in the ESS estimator and
    # far tighter than any edit worth making to a published diagnostic.
    for key, column in (
        ("max_rhat", "max_rhat"),
        ("min_ess", "min_ess"),
        ("min_bfmi", "min_bfmi"),
    ):
        claimed = row.get(column)
        if claimed in (None, "") or recomputed[key] is None:
            continue
        if not np.isclose(
            float(claimed), float(recomputed[key]), rtol=1e-3, atol=1e-6
        ):
            raise RuntimeError(
                f"{outcome}: cell row's {column} ({claimed}) is not what its trace "
                f"gives ({recomputed[key]})"
            )


def _validate_cell_trace(
    source: Path, row: Mapping[str, Any], *, reference: PrimaryStandardReference
) -> None:
    """Verify a cell trace *is* the evidence its manifest row claims (#489 review).

    The digest check alone only proves the file matches the row's own recorded
    hash — a circular fact if the row itself is wrong. This opens the trace and
    cross-checks the provenance the runner stamped at fit time against both the
    row and the freshly loaded primary reference: identity (outcome, primary
    model id and artefact hashes, prior scale), the sampling contract
    (draws/tune/chains/target-accept and the posterior's actual dimensions),
    the recorded free variables, the focal term, the row's focal summary and its
    convergence claim.

    **Indexed focal terms are recomputed too** (#584 finding 3). The level
    family's focal term is an element of a vector (``d_grp_time[t2]``), and this
    validator used to check only that the *base* variable existed there — so a
    row whose ``tau_logit_mean`` had been edited, whose coordinate named a
    different wave, or whose ``converged`` flag was simply untrue, attached
    successfully on hashes alone. ``tau_logit_mean`` is the mean of the focal
    coordinate's draws in every family that writes this schema (the items-scale
    marginal lives in the ``items_*`` columns), so it is recomputed from the
    element the provenance names, the interval is required to bracket it, and
    the convergence gate is re-run over the cell's own recorded free variables
    rather than believed. The cell's focal term is additionally bound to the
    primary's stored ``resolved_run_plan``, so a semantically stale grid — a
    pre-#552 ``b_grp_time[1]`` sweep against a t1-centred primary — is refused
    even when every hash matches.
    """
    import arviz as az

    outcome = str(row["outcome"])
    try:
        trace = az.from_netcdf(source)
    except Exception as exc:  # noqa: BLE001 - a corrupt cell trace is gate data
        raise RuntimeError(
            f"{outcome}: cell trace is not a readable NetCDF: {source}"
        ) from exc
    try:
        posterior = getattr(trace, "posterior", None)
        if posterior is None:
            raise RuntimeError(f"{outcome}: cell trace has no posterior group: {source}")
        raw = posterior.attrs.get(STANDARD_SENSITIVITY_PROVENANCE_ATTR)
        if not raw:
            raise RuntimeError(
                f"{outcome}: cell trace carries no sweep provenance: {source}"
            )
        try:
            provenance = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{outcome}: cell trace provenance is not valid JSON: {source}"
            ) from exc
        identity = (
            ("outcome", outcome, False),
            ("primary_model_id", reference.model_id, False),
            (
                "primary_config_sha256",
                str(row["primary_config_sha256"]).strip().lower(),
                True,
            ),
            (
                "primary_trace_sha256",
                str(row["primary_trace_sha256"]).strip().lower(),
                True,
            ),
        )
        for key, expected, lower in identity:
            recorded = str(provenance.get(key, "")).strip()
            if lower:
                recorded = recorded.lower()
            if recorded != expected:
                raise RuntimeError(
                    f"{outcome}: cell trace provenance {key} does not match its "
                    f"row ({recorded!r} != {expected!r})"
                )
        if not np.isclose(
            float(provenance.get("tau_sigma", np.nan)),
            float(row["tau_sigma"]),
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError(
                f"{outcome}: cell trace provenance names a different prior scale "
                "than its row"
            )
        sampling = provenance.get("sampling") or {}
        for key, column in (
            ("draws", "sampling_draws"),
            ("tune", "sampling_tune"),
            ("chains", "sampling_chains"),
            ("target_accept", "sampling_target_accept"),
        ):
            if not np.isclose(
                float(sampling.get(key, np.nan)),
                float(row[column]),
                rtol=0.0,
                atol=1e-9,
            ):
                raise RuntimeError(
                    f"{outcome}: cell trace sampling provenance does not match "
                    f"its row ({key})"
                )
        if int(posterior.sizes.get("chain", -1)) != int(row["sampling_chains"]) or int(
            posterior.sizes.get("draw", -1)
        ) != int(row["sampling_draws"]):
            raise RuntimeError(
                f"{outcome}: cell trace chain/draw dimensions do not match its row"
            )
        free = provenance.get("free_variables") or []
        missing_vars = sorted(set(map(str, free)) - set(map(str, posterior.data_vars)))
        if missing_vars:
            raise RuntimeError(
                f"{outcome}: cell trace posterior lacks its own recorded free "
                f"variables ({missing_vars[:3]})"
            )
        focal = str(provenance.get("focal_term", ""))
        base = focal.split("[", 1)[0]
        if not base or base not in posterior.data_vars:
            raise RuntimeError(
                f"{outcome}: cell trace posterior lacks the focal term {focal!r}"
            )
        planned = _primary_plan_focal_term(reference)
        if planned is not None and focal != planned:
            raise RuntimeError(
                f"{outcome}: cell trace is a sweep of {focal!r}, but the current "
                f"primary's resolved run plan reports {planned!r} as its focal "
                "term; re-run the sweep against the current parameterisation"
            )
        draws = _focal_term_draws(posterior, focal, outcome=outcome)
        recomputed = float(draws.mean())
        if not np.isclose(
            recomputed, float(row["tau_logit_mean"]), rtol=0.0, atol=1e-8
        ):
            raise RuntimeError(
                f"{outcome}: cell trace does not reproduce its row's focal "
                f"summary ({recomputed:.6f} != {float(row['tau_logit_mean']):.6f})"
            )
        bounds = _optional_floats(row, "tau_logit_lo", "tau_logit_hi")
        if bounds is not None:
            lo, hi = bounds
            if not lo <= recomputed <= hi:
                raise RuntimeError(
                    f"{outcome}: cell row's interval [{lo:.6f}, {hi:.6f}] does not "
                    "bracket the focal mean recomputed from its trace "
                    f"({recomputed:.6f})"
                )
        if str(provenance.get("model_kind", "")) == "level_factors" and _optional_floats(
            row, "pd"
        ):
            # The level family's items-scale marginal adds the same focal draw to
            # every fitted row, so ``expit(eta0 + d) - expit(eta0)`` carries the
            # sign of ``d`` in every row and the published direction probability is
            # exactly the focal coordinate's. Recomputing it here is therefore an
            # identity for that family — and the one number a reader takes the
            # direction claim from.
            direction = float(np.mean(draws > 0.0))
            if not np.isclose(direction, float(row["pd"]), rtol=0.0, atol=1e-9):
                raise RuntimeError(
                    f"{outcome}: cell trace does not reproduce its row's direction "
                    f"probability ({direction:.6f} != {float(row['pd']):.6f})"
                )
        sample_stats = getattr(trace, "sample_stats", None)
        if sample_stats is None or "diverging" not in sample_stats:
            raise RuntimeError(
                f"{outcome}: cell trace lacks sample_stats.diverging, so its "
                "convergence claim cannot be re-checked"
            )
        divergences = int(np.asarray(sample_stats["diverging"].values).sum())
        if divergences != int(float(row["n_divergences"])):
            raise RuntimeError(
                f"{outcome}: cell trace divergence count does not match its row "
                f"({divergences} != {row['n_divergences']})"
            )
        _verify_cell_convergence(trace, row, outcome=outcome, free_variables=free)
    finally:
        close = getattr(trace, "close", None)
        if callable(close):
            close()


def attach_outcome_bundle(
    rows: pd.DataFrame,
    *,
    outcome: str,
    primary_dir: Path,
    sensitivity_dir: Path,
    reference: PrimaryStandardReference,
) -> Path:
    """Install one fit's **trace-backed** sweep bundle beside its primary.

    Follows the floor installer's discipline so the report-local manifest is
    never exposed without its evidence: every clause is verified *before* the
    gate's filename exists —

    - the manifest carries the standard sweep's full column set, only this
      outcome's rows, one ``primary_model_id`` (the reference's own), >= 2 focal
      prior scales, every cell converged, and one sign;
    - its ``primary_config_sha256`` / ``primary_trace_sha256`` match the
      **current** primary artefacts (via the freshly loaded ``reference``);
    - each cell trace exists in the sweep directory, its sha256 matches the
      manifest, its **stamped provenance** identifies it as this cell of this
      primary's sweep and reproduces the row's summaries
      (:func:`_validate_cell_trace`, #489 review — the digest alone only
      proves the bytes match the row's own claim), and the copy installed
      beside the fit re-verifies after copy;
    - ``trace_file`` is rewritten to the installed digest-suffixed basename.

    The manifest lands via an atomic rename, and the release gate's own
    evidence check (``release._standard_sweep_evidence``) is run as a final
    assert. Any failure rolls back **this attempt only**: newly installed
    traces are removed and a pre-existing manifest is restored rather than
    destroyed (#489 review), so a failed replacement cannot delete previously
    published evidence.
    """
    from language_reading_predictors.statistical_models.release import (
        _standard_sweep_evidence,
    )

    rows = rows.reset_index(drop=True).copy()
    if rows.empty:
        raise RuntimeError(f"{outcome}: no sweep rows to attach")
    missing = sorted(_STANDARD_REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise RuntimeError(
            f"{outcome}: sweep rows lack required columns: {missing[:4]}"
        )
    if set(rows["outcome"].astype(str)) != {outcome}:
        raise RuntimeError(f"{outcome}: sweep rows mix outcomes")
    # Two did fits share outcome W, so outcome identity alone cannot prove the
    # rows belong to this primary; the model id must match too.
    recorded_models = set(rows["primary_model_id"].astype(str))
    if recorded_models != {reference.model_id}:
        raise RuntimeError(
            f"{outcome}: sweep rows name a different primary model "
            f"({sorted(recorded_models)} != {reference.model_id})"
        )
    if rows["tau_sigma"].nunique() < 2:
        raise RuntimeError(f"{outcome}: fewer than two tau scales")
    if not rows["converged"].astype(bool).all():
        raise RuntimeError(
            f"{outcome}: refusing to attach — one or more cells failed the "
            "convergence gate"
        )
    signs = set(np.sign(rows["tau_logit_mean"].astype(float)).tolist())
    if len(signs) != 1:
        raise RuntimeError(
            f"{outcome}: the effect changes sign across the grid; the bundle is "
            "reportable evidence of instability, not of stability — not attaching"
        )
    for column, expected in (
        ("primary_config_sha256", reference.config_sha256),
        ("primary_trace_sha256", reference.trace_sha256),
    ):
        recorded = {str(v).strip().lower() for v in rows[column]}
        if recorded != {expected}:
            raise RuntimeError(
                f"{outcome}: sweep rows bind to a different primary {column}; "
                "re-run the sweep against the current fit"
            )

    installed_new: list[Path] = []
    staging = primary_dir / (STANDARD_SENSITIVITY_FILENAME + ".staging")
    destination = primary_dir / STANDARD_SENSITIVITY_FILENAME
    # A failed replacement must restore, not destroy, previously published
    # evidence (#489 review): snapshot any existing manifest before starting,
    # and delete only the trace files this attempt itself created — a
    # digest-named file that already exists belongs to the previous bundle.
    previous_manifest = destination.read_bytes() if destination.is_file() else None
    try:
        for index, row in rows.iterrows():
            source = sensitivity_dir / str(row["trace_file"])
            trace_sha256 = str(row["trace_sha256"]).strip().lower()
            if not source.is_file():
                raise RuntimeError(f"{outcome}: missing cell trace {source}")
            if sha256_file(source) != trace_sha256:
                raise RuntimeError(
                    f"{outcome}: cell trace does not match its recorded sha256: "
                    f"{source}"
                )
            # The digest only proves the bytes match the row's own claim;
            # verify the trace's stamped provenance says it is this cell of
            # this primary's sweep, and that it reproduces the row (#489).
            _validate_cell_trace(source, row, reference=reference)
            digest_suffix = f"-{trace_sha256[:12]}"
            name = (
                source.name
                if source.stem.endswith(digest_suffix)
                else f"{source.stem}{digest_suffix}.nc"
            )
            target = primary_dir / name
            if target.exists():
                if sha256_file(target) != trace_sha256:
                    raise RuntimeError(
                        f"{outcome}: an existing installed file blocks this cell "
                        f"trace and does not match its digest: {target}"
                    )
            else:
                shutil.copy2(source, target)
                installed_new.append(target)
                if sha256_file(target) != trace_sha256:
                    raise RuntimeError(
                        f"{outcome}: installed trace changed during copy: {target}"
                    )
            rows.at[index, "trace_file"] = name
        rows.to_csv(staging, index=False)
        os.replace(staging, destination)
        ready, reason = _standard_sweep_evidence(primary_dir, outcome)
        if not ready:
            raise RuntimeError(
                f"{outcome}: installed bundle fails the release gate's own "
                f"evidence check ({reason})"
            )
    except BaseException:
        for target in installed_new:
            target.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)
        if previous_manifest is not None:
            restore = primary_dir / (STANDARD_SENSITIVITY_FILENAME + ".restore")
            restore.write_bytes(previous_manifest)
            os.replace(restore, destination)
        else:
            destination.unlink(missing_ok=True)
        raise
    return destination


def persist_sensitivity_trace(
    trace: Any, *, sensitivity_dir: Path, semantic_file: Path, label: str = "sensitivity"
) -> tuple[Path, str]:
    """Atomically install an immutable cell trace named by its content digest."""
    trace_dir = sensitivity_dir / semantic_file.parent
    trace_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{semantic_file.stem}-", suffix=".nc", dir=trace_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        trace.to_netcdf(temporary)
        digest = sha256_file(temporary)
        destination = trace_dir / f"{semantic_file.stem}-{digest[:12]}.nc"
        if destination.exists():
            if sha256_file(destination) != digest:
                raise RuntimeError(
                    f"{label} trace digest-prefix collision: {destination}"
                )
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        return destination.relative_to(sensitivity_dir), digest
    finally:
        if temporary.exists():
            temporary.unlink()


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
