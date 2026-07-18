# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Direct treatment-effect sensitivity refits for influential ITT children.

The main pipelines persist one Pareto-k value per child for the single-period
``itt`` and ``joint`` families.  A value above ArviZ's reliability threshold
makes the PSIS-LOO approximation unreliable; it does not, by itself, show that
the treatment estimate is unstable.  This module reconstructs the registered
model, removes every flagged child, refits it, and compares the headline
probability-scale average marginal effects with the completed full-data fit.

The implementation deliberately rejects stacked random-intercept families.
Their pointwise LOO unit is a child-by-period row conditional on the fitted child
intercept, so a whole-child deletion is not the sensitivity implied by a
flagged point.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

import dse_research_utils.statistics.models.sampling as _sampling
from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models import diagnostics as _diag
from language_reading_predictors.statistical_models import factories as _factories
from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import (
    declared_settings_dict,
    resolve_itt_run_plan,
)
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES
from language_reading_predictors.statistical_models.preprocessing import (
    _subset_prepared,
    load_and_prepare,
    restrict_to_baseline_floored,
)
from language_reading_predictors.statistical_models.registry import discover_models

INFLUENCE_SENSITIVITY_FILENAME = "influence_sensitivity.csv"
INFLUENCE_TRACE_STEM = "trace_influence_sensitivity"
INFLUENCE_FREE_VARIABLES_ATTR = "influence_free_variables_json"
INFLUENCE_IDENTITY_ATTR = "influence_identity_json"
INFLUENCE_SAMPLING_ATTR = "influence_sampling_json"
INFLUENCE_SUPPORTED_KINDS = frozenset({"itt", "joint"})

PRIMARY_ARTIFACT_HASH_COLUMNS = {
    "primary_config_sha256": "config.json",
    "primary_trace_sha256": "trace.nc",
    "primary_pareto_k_sha256": "pareto_k.csv",
    "primary_tau_summary_sha256": "tau_summary.csv",
}

_PARETO_REQUIRED_COLUMNS = {
    "observation_index",
    "subject_id",
    "pareto_k",
    "good_k_threshold",
}


@dataclass(frozen=True)
class InfluenceReference:
    """Validated artefacts from the completed full-data fit."""

    model_dir: Path
    metadata: dict[str, Any]
    pareto: pd.DataFrame
    flagged: pd.DataFrame
    full_summary: pd.DataFrame
    primary_hashes: dict[str, str]


@dataclass(frozen=True)
class InfluenceBuild:
    """Leave-out model plus provenance from its full-data preparation."""

    built: Any
    full_subject_ids: np.ndarray
    excluded_subject_ids: tuple[str, ...]
    data_path: str
    data_sha256: str


class InfluenceBundleError(RuntimeError):
    """Raised when a central refit cannot safely replace the report-local bundle."""


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artefact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close_trace(trace: Any) -> None:
    """Release lazy NetCDF handles when the backend exposes ``close``."""
    close = getattr(trace, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # Validation has already consumed the trace; a backend close failure
            # must not turn a fail-closed status into an uncaught render error.
            pass


def _open_readable_trace(path: Path, *, label: str):
    """Open a trace and require a non-empty posterior chain-by-draw structure."""
    try:
        trace = az.from_netcdf(path)
    except Exception as exc:
        raise ValueError(f"{label} trace is unreadable: {exc}") from exc
    try:
        posterior = getattr(trace, "posterior", None)
        if posterior is None:
            raise ValueError(f"{label} trace has no posterior group")
        for dimension in ("chain", "draw"):
            if int(posterior.sizes.get(dimension, 0)) <= 0:
                raise ValueError(
                    f"{label} trace has no non-empty posterior {dimension!r} dimension"
                )
    except Exception:
        _close_trace(trace)
        raise
    return trace


def hash_primary_artifacts(model_dir: Path) -> dict[str, str]:
    """Hash every primary artefact that defines the influence-refit reference."""
    hashes: dict[str, str] = {}
    for column, filename in PRIMARY_ARTIFACT_HASH_COLUMNS.items():
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"required completed-fit artefact is missing: {path}")
        if filename == "trace.nc":
            trace = _open_readable_trace(path, label="primary")
            _close_trace(trace)
        hashes[column] = sha256_file(path)
    return hashes


def _atomic_write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Write a CSV by atomic rename so readers never observe a partial bundle index."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy one file to a sibling temporary path and atomically install it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_normalise(value: Any) -> Any:
    """Normalise tuples and other JSON-safe containers for drift comparisons."""
    return json.loads(json.dumps(value, sort_keys=True))


def resolve_registered_spec(model_id: str) -> tuple[ModuleType, ModelSpec]:
    """Return the canonical registered module and spec for an ITT/joint model."""
    canonical = model_id.strip().lower()
    models = discover_models()
    if canonical not in models:
        raise ValueError(f"unknown registered statistical model: {model_id!r}")
    module = models[canonical]
    spec = getattr(module, "SPEC", None)
    if not isinstance(spec, ModelSpec):
        raise ValueError(f"{canonical} does not expose a static ModelSpec")
    if spec.kind not in INFLUENCE_SUPPORTED_KINDS:
        raise ValueError(
            f"influence sensitivity supports only {sorted(INFLUENCE_SUPPORTED_KINDS)}, "
            f"not kind={spec.kind!r}; stacked random-intercept LOO has a different unit"
        )
    return module, spec


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required completed-fit artefact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_influence_reference(
    spec: ModelSpec,
    config: str,
    *,
    model_output_root: Path | None = None,
) -> InfluenceReference:
    """Load and validate the full fit whose unreliable LOO point triggered the refit."""
    root = model_output_root or _paths.stat_models_dir()
    model_dir = root / f"{spec.model_id}-{config}"
    metadata = _read_json(model_dir / "config.json")
    diagnostics = _read_json(model_dir / "diagnostics_summary.json")
    if diagnostics.get("passed") is not True:
        raise ValueError(
            f"the completed {spec.model_id}-{config} fit did not pass its convergence gate"
        )
    if metadata.get("model_id") != spec.model_id or metadata.get("kind") != spec.kind:
        raise ValueError(
            "saved fit identity does not match the current registered specification: "
            f"saved=({metadata.get('model_id')!r}, {metadata.get('kind')!r}), "
            f"current=({spec.model_id!r}, {spec.kind!r})"
        )
    if spec.kind == "itt":
        current_plan = resolve_itt_run_plan(spec).as_dict()
        current_plan.pop("settings_source", None)
        if metadata.get("resolved_run_plan") is not None:
            saved_settings = dict(metadata["resolved_run_plan"])
            saved_settings.pop("settings_source", None)
        else:
            saved_spec = replace(
                spec,
                model_settings=None,
                extra=dict(metadata.get("spec_extra", {})),
            )
            saved_settings = resolve_itt_run_plan(saved_spec).as_dict()
            saved_settings.pop("settings_source", None)
    else:
        current_plan = spec.extra
        saved_settings = metadata.get("spec_extra", {})
    if _json_normalise(saved_settings) != _json_normalise(current_plan):
        raise ValueError(
            f"registered settings for {spec.model_id} have drifted since the completed fit; "
            "refit the full model before running influence sensitivity"
        )

    main_trace = model_dir / "trace.nc"
    if not main_trace.is_file():
        raise FileNotFoundError(f"completed full-data trace is missing: {main_trace}")

    pareto_path = model_dir / "pareto_k.csv"
    if not pareto_path.is_file():
        raise FileNotFoundError(f"pointwise Pareto-k table is missing: {pareto_path}")
    pareto = pd.read_csv(pareto_path)
    missing = sorted(_PARETO_REQUIRED_COLUMNS - set(pareto.columns))
    if missing:
        raise ValueError(f"{pareto_path} is missing required columns: {missing}")
    if pareto.empty:
        raise ValueError(f"{pareto_path} contains no pointwise observations")

    for column in ("observation_index", "pareto_k", "good_k_threshold"):
        pareto[column] = pd.to_numeric(pareto[column], errors="coerce")
    numeric = pareto[["observation_index", "pareto_k", "good_k_threshold"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{pareto_path} contains non-finite pointwise diagnostics")
    indices = pareto["observation_index"].to_numpy(dtype=float)
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError(f"{pareto_path} contains non-integer observation indices")
    if not np.allclose(
        pareto["good_k_threshold"], pareto["good_k_threshold"].iloc[0]
    ):
        raise ValueError(f"{pareto_path} contains inconsistent Pareto-k thresholds")
    if pareto["observation_index"].duplicated().any():
        raise ValueError(f"{pareto_path} contains duplicate observation indices")
    if pareto["subject_id"].astype(str).duplicated().any():
        raise ValueError(
            f"{spec.model_id} does not have one Pareto-k point per unique child; "
            "whole-child influence sensitivity is not valid for this likelihood unit"
        )

    n_obs = int(metadata.get("n_obs", -1))
    n_children = int(metadata.get("n_children", -1))
    if n_obs <= 0 or n_obs != n_children or len(pareto) != n_obs:
        raise ValueError(
            f"{spec.model_id} must have one fitted row and Pareto-k point per child; "
            f"saved n_obs={n_obs}, n_children={n_children}, pareto rows={len(pareto)}"
        )
    observed_indices = set(pareto["observation_index"].astype(int))
    if observed_indices != set(range(n_obs)):
        raise ValueError(f"{pareto_path} observation indices do not cover 0..{n_obs - 1}")

    threshold = float(pareto["good_k_threshold"].iloc[0])
    flagged = pareto.loc[pareto["pareto_k"] > threshold].copy()
    if flagged.empty:
        raise ValueError(
            f"{spec.model_id}-{config} has no Pareto-k point above {threshold:.3f}; "
            "a direct influence refit is not required"
        )

    summary_path = model_dir / "tau_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"full-data treatment summary is missing: {summary_path}")
    full_summary = pd.read_csv(summary_path)
    if full_summary.empty:
        raise ValueError(f"full-data treatment summary is empty: {summary_path}")
    primary_hashes = hash_primary_artifacts(model_dir)

    return InfluenceReference(
        model_dir=model_dir,
        metadata=metadata,
        pareto=pareto,
        flagged=flagged,
        full_summary=full_summary,
        primary_hashes=primary_hashes,
    )


def _prepare_itt(spec: ModelSpec):
    """Mirror the registered single-outcome ITT preparation contract."""
    plan = resolve_itt_run_plan(spec)
    prepared = load_and_prepare(**plan.prepare_kwargs())
    adjust_for = plan.adjust_for
    adjust_for = tuple(c for c in adjust_for if c in prepared.covariates)
    if plan.floor_rule:
        prepared = restrict_to_baseline_floored(prepared, spec.outcome_symbol)
    return prepared, adjust_for, plan


def _prepare_joint(spec: ModelSpec):
    """Mirror the registered joint-model preparation contract."""
    outcomes = tuple(spec.extra.get("outcomes") or ITT_OUTCOMES)
    prepared = load_and_prepare(phase_mode="itt", outcomes=outcomes)
    return prepared, outcomes


def _validate_reference_alignment(prepared, reference: InfluenceReference) -> None:
    """Ensure persisted point indices still identify the same current-data children."""
    expected_n = int(reference.metadata["n_obs"])
    if prepared.n_obs != expected_n or prepared.n_children != expected_n:
        raise ValueError(
            "current prepared sample does not match the completed fit: "
            f"current n={prepared.n_obs}, saved n={expected_n}"
        )
    saved_sha = str(reference.metadata.get("data_sha256", ""))
    if not saved_sha or prepared.data_sha256 != saved_sha:
        raise ValueError(
            "current input data checksum does not match the completed fit; refit the full "
            "model before running influence sensitivity"
        )
    subject_ids = np.asarray(prepared.subject_ids).astype(str)
    by_index = reference.pareto.sort_values("observation_index")
    persisted_ids = by_index["subject_id"].astype(str).to_numpy()
    if not np.array_equal(subject_ids, persisted_ids):
        raise ValueError(
            "current prepared row order/subject identities do not match pareto_k.csv; "
            "refusing to exclude a potentially different child"
        )


def build_influence_model(
    spec: ModelSpec,
    reference: InfluenceReference,
) -> InfluenceBuild:
    """Rebuild ``spec`` after removing every child with unreliable Pareto-k."""
    excluded = tuple(reference.flagged["subject_id"].astype(str).tolist())
    if spec.kind == "itt":
        prepared, adjust_for, plan = _prepare_itt(spec)
        _validate_reference_alignment(prepared, reference)
        full_ids = np.asarray(prepared.subject_ids).astype(str)
        keep = ~np.isin(full_ids, excluded)
        reduced = _subset_prepared(prepared, keep)
        if np.unique(reduced.G).size != 2:
            raise ValueError("excluding the flagged child leaves fewer than two trial arms")
        if plan.floor_rule:
            built = _factories.build_itt_model(
                reduced,
                likelihood="bernoulli_offfloor",
                **plan.factory_kwargs(effective_adjustment=adjust_for),
            )
        else:
            built = _factories.build_itt_model(
                reduced,
                **plan.factory_kwargs(effective_adjustment=adjust_for),
            )
    else:
        prepared, outcomes = _prepare_joint(spec)
        _validate_reference_alignment(prepared, reference)
        full_ids = np.asarray(prepared.subject_ids).astype(str)
        keep = ~np.isin(full_ids, excluded)
        reduced = _subset_prepared(prepared, keep)
        if np.unique(reduced.G).size != 2:
            raise ValueError("excluding the flagged child leaves fewer than two trial arms")
        built = _factories.build_joint_model(
            reduced,
            outcomes=outcomes,
            use_age_gp=spec.extra.get("use_age_gp", False),
            partial_pool_age_gp=spec.extra.get("partial_pool_age_gp", True),
            use_residual_correlation=spec.extra.get(
                "use_residual_correlation", False
            ),
            use_cross_baselines=spec.extra.get("use_cross_baselines", True),
            use_age_linear=spec.extra.get("use_age_linear", False),
        )

    if built.prepared.n_obs != len(full_ids) - len(excluded):
        raise RuntimeError(
            "leave-out build changed the sample by more than the flagged children: "
            f"full={len(full_ids)}, excluded={len(excluded)}, built={built.prepared.n_obs}"
        )
    return InfluenceBuild(
        built=built,
        full_subject_ids=full_ids,
        excluded_subject_ids=excluded,
        data_path=prepared.data_path,
        data_sha256=prepared.data_sha256,
    )


def _sample_influence_model(
    built,
    reference: InfluenceReference,
    config: str,
    *,
    random_seed: int,
    cores: int | None,
):
    """Sample with the completed full fit's saved reporting settings."""
    saved = reference.metadata.get("sampling")
    if not isinstance(saved, dict):
        raise ValueError("completed fit config.json has no sampling settings")
    required = {"draws", "tune", "chains", "target_accept"}
    missing = sorted(required - set(saved))
    if missing:
        raise ValueError(f"completed fit sampling settings are missing: {missing}")
    preset = _sampling.get_sampling_configuration(config, random_seed=random_seed)
    draws = int(saved["draws"])
    tune = int(saved["tune"])
    chains = int(saved["chains"])
    target_accept = float(saved["target_accept"])
    use_cores = int(cores if cores is not None else min(preset.cores, chains))
    with built.model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=use_cores,
            target_accept=target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=random_seed,
            progressbar=False,
        )
    return trace, {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": use_cores,
        "target_accept": target_accept,
        "random_seed": int(random_seed),
    }


def _normalise_summary(summary: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Put single- and multi-outcome summaries on the same AME column names."""
    return _normalise_summary_symbol(summary, spec.outcome_symbol)


def _normalise_summary_symbol(
    summary: pd.DataFrame, outcome_symbol: str | None
) -> pd.DataFrame:
    """Normalise a saved summary when only report metadata is available."""
    out = summary.copy()
    if "outcome" not in out.columns:
        out.insert(0, "outcome", outcome_symbol)
    rename = {
        column: column.replace("tau_prob_", "ame_prob_", 1)
        for column in out.columns
        if column.startswith("tau_prob_")
    }
    out = out.rename(columns=rename)
    required = {"outcome", "ame_prob_median", "prob_ame_pos"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError(f"treatment summary lacks required AME columns: {missing}")
    if out["outcome"].astype(str).duplicated().any():
        raise ValueError("treatment summary contains duplicate outcomes")
    return out


def _suffix_summary_metrics(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Suffix every treatment-summary field except the outcome join key."""
    return frame.rename(
        columns={column: f"{column}_{suffix}" for column in frame if column != "outcome"}
    )


def _trace_treatment_data(
    trace: Any,
    *,
    expected_n: int,
    label: str,
) -> tuple[np.ndarray, Any]:
    """Return a trace-bound two-arm treatment vector with exact observation axes."""
    constant = getattr(trace, "constant_data", None)
    if constant is None or "G" not in constant:
        raise ValueError(f"{label} trace has no constant treatment indicator G")
    treatment = constant["G"]
    G = np.asarray(treatment.values, dtype=float)
    if (
        G.ndim != 1
        or tuple(treatment.dims) != ("obs_id",)
        or int(treatment.sizes.get("obs_id", -1)) != expected_n
    ):
        raise ValueError(
            f"{label} trace treatment indicator does not match the expected "
            "observation count"
        )
    if not np.isin(G, (0.0, 1.0)).all() or np.unique(G).size != 2:
        raise ValueError(f"{label} trace treatment indicator is not a two-arm design")
    posterior = trace.posterior
    if "eta" not in posterior or int(posterior["eta"].sizes.get("obs_id", -1)) != (
        expected_n
    ):
        raise ValueError(
            f"{label} trace eta does not match the expected observation count"
        )
    return G, constant


def _trace_itt_moderators(
    trace: Any,
    constant: Any,
    *,
    expected_n: int,
    label: str,
) -> list[tuple[str, np.ndarray]]:
    """Recover the trace-bound ITT treatment moderators used in its AME."""
    moderators: list[tuple[str, np.ndarray]] = []
    if "gamma_tau_int" in trace.posterior:
        if "z_tau_moderator" not in constant:
            raise ValueError(f"{label} moderated trace lacks z_tau_moderator")
        moderator_data = constant["z_tau_moderator"]
        moderator = np.asarray(moderator_data.values, dtype=float)
        if (
            moderator.ndim != 1
            or tuple(moderator_data.dims) != ("obs_id",)
            or int(moderator_data.sizes.get("obs_id", -1)) != expected_n
            or not np.isfinite(moderator).all()
        ):
            raise ValueError(
                f"{label} trace treatment moderator does not match the expected "
                "observation count"
            )
        moderators.append(("gamma_tau_int", moderator))
    return moderators


def summarise_influence_refit(
    spec: ModelSpec,
    reference: InfluenceReference,
    influence_build: InfluenceBuild,
    trace,
    *,
    config: str,
    sampling: dict[str, Any],
) -> pd.DataFrame:
    """Decompose full-versus-leave-out AMEs and attach convergence/provenance."""
    built = influence_build.built
    ci_prob = float(reference.metadata.get("ci_prob", 0.95))
    if spec.kind == "joint":
        outcomes = list(trace.posterior["outcome"].values.astype(str))
        refit = _report.tau_summary_joint(
            trace, outcomes, ci_prob=ci_prob, G=built.prepared.G
        )
    else:
        moderators = built.extras.get("tau_interaction_moderators", [])
        summary = _report.tau_summary_itt(
            trace,
            ci_prob=ci_prob,
            G=built.prepared.G,
            moderators=moderators,
        )
        refit = pd.DataFrame([summary])

    full = _normalise_summary(reference.full_summary, spec)
    refit = _normalise_summary(refit, spec)
    excluded = set(influence_build.excluded_subject_ids)
    full_subject_ids = np.asarray(influence_build.full_subject_ids).astype(str)
    retained_mask = ~np.isin(full_subject_ids, list(excluded))
    if int((~retained_mask).sum()) != len(excluded):
        raise ValueError("excluded children do not identify unique full-sample rows")
    if int(retained_mask.sum()) != int(built.prepared.n_obs):
        raise ValueError("retained primary rows do not match the leave-out sample size")

    primary_trace = _open_readable_trace(
        reference.model_dir / "trace.nc", label="primary"
    )
    try:
        primary_G, primary_constant = _trace_treatment_data(
            primary_trace,
            expected_n=len(full_subject_ids),
            label="primary",
        )
        if spec.kind == "joint":
            primary_retained = _report.tau_summary_joint(
                primary_trace,
                outcomes,
                ci_prob=ci_prob,
                G=primary_G,
                row_mask=retained_mask,
            )
        else:
            primary_moderators = _trace_itt_moderators(
                primary_trace,
                primary_constant,
                expected_n=len(full_subject_ids),
                label="primary",
            )
            primary_retained = pd.DataFrame(
                [
                    _report.tau_summary_itt(
                        primary_trace,
                        ci_prob=ci_prob,
                        G=primary_G,
                        moderators=primary_moderators,
                        row_mask=retained_mask,
                    )
                ]
            )
    finally:
        _close_trace(primary_trace)
    primary_retained = _normalise_summary(primary_retained, spec)
    outcome_sets = [
        set(frame["outcome"].astype(str))
        for frame in (full, primary_retained, refit)
    ]
    if not outcome_sets[0] == outcome_sets[1] == outcome_sets[2]:
        raise ValueError(
            "full, retained-population, and influence-refit summaries cover "
            "different outcomes"
        )
    merged = _suffix_summary_metrics(full, "full").merge(
        _suffix_summary_metrics(primary_retained, "full_retained"), on="outcome"
    )
    merged = merged.merge(
        _suffix_summary_metrics(refit, "without_flagged"), on="outcome"
    )
    merged["composition_shift_ame_prob_median"] = (
        merged["ame_prob_median_full_retained"]
        - merged["ame_prob_median_full"]
    )
    merged["refit_shift_ame_prob_median"] = (
        merged["ame_prob_median_without_flagged"]
        - merged["ame_prob_median_full_retained"]
    )
    merged["total_shift_ame_prob_median"] = (
        merged["ame_prob_median_without_flagged"]
        - merged["ame_prob_median_full"]
    )
    # Backward-compatible alias: this has always meant the total full-sample to
    # leave-out-sample contrast, not the common-population refit component.
    merged["delta_ame_prob_median"] = merged["total_shift_ame_prob_median"]

    free_names = [rv.name for rv in built.model.free_RVs]
    convergence = _diag.subfit_convergence(
        trace,
        label=f"{spec.model_id} influence sensitivity",
        var_names=free_names,
    )
    free_contract = "|".join(free_names)
    sampling_json = json.dumps(_json_normalise(sampling), sort_keys=True)
    identity = {
        "model_id": spec.model_id,
        "model_kind": spec.kind,
        "config": config,
        "data_sha256": influence_build.data_sha256,
        "excluded_subject_ids": sorted(influence_build.excluded_subject_ids),
        "n_without_flagged": int(built.prepared.n_obs),
        **reference.primary_hashes,
    }
    identity_json = json.dumps(_json_normalise(identity), sort_keys=True)
    trace.posterior.attrs[INFLUENCE_FREE_VARIABLES_ATTR] = json.dumps(free_names)
    trace.posterior.attrs[INFLUENCE_IDENTITY_ATTR] = identity_json
    trace.posterior.attrs[INFLUENCE_SAMPLING_ATTR] = sampling_json
    flagged = reference.flagged.sort_values("pareto_k", ascending=False)
    provenance: dict[str, Any] = {
        "model_id": spec.model_id,
        "model_kind": spec.kind,
        "config": config,
        "sensitivity_method": "direct_leave_out_refit_excluding_all_psis_flagged_children",
        "loo_unit": "child",
        "ame_comparison_population": "common_retained_children",
        "shift_decomposition": "total_shift=composition_shift+refit_shift",
        "delta_ame_prob_median_alias": "total_shift_ame_prob_median",
        "excluded_subject_id": ";".join(influence_build.excluded_subject_ids),
        "excluded_subject_ids": ";".join(influence_build.excluded_subject_ids),
        "n_excluded_children": len(influence_build.excluded_subject_ids),
        "excluded_observation_indices": ";".join(
            str(int(value)) for value in flagged["observation_index"]
        ),
        "reference_pareto_k": float(flagged["pareto_k"].max()),
        "reference_pareto_k_values": ";".join(
            f"{float(value):.17g}" for value in flagged["pareto_k"]
        ),
        "good_k_threshold": float(flagged["good_k_threshold"].iloc[0]),
        "n_full": len(influence_build.full_subject_ids),
        "n_without_flagged": int(built.prepared.n_obs),
        "data_path": influence_build.data_path,
        "data_sha256": influence_build.data_sha256,
        "spec_extra_json": json.dumps(_json_normalise(spec.extra), sort_keys=True),
        "model_settings_json": json.dumps(
            _json_normalise(declared_settings_dict(spec)), sort_keys=True
        ),
        "resolved_run_plan_json": (
            json.dumps(
                _json_normalise(resolve_itt_run_plan(spec).as_dict()),
                sort_keys=True,
            )
            if spec.kind == "itt"
            else ""
        ),
        "reference_config_file": "config.json",
        "reference_pareto_file": "pareto_k.csv",
        "reference_trace_file": "trace.nc",
        "reference_tau_summary_file": "tau_summary.csv",
        "free_variables": free_contract,
        "free_variables_sha256": hashlib.sha256(free_contract.encode("utf-8")).hexdigest(),
        "n_free_variables": len(free_names),
        "identity_json": identity_json,
        "sampling_json": sampling_json,
        "convergence_scope": "all_free_variables",
        "ci_prob": ci_prob,
        "created_utc": datetime.now(UTC).isoformat(),
        **{f"sampling_{key}": value for key, value in sampling.items()},
        **reference.primary_hashes,
        **convergence,
    }
    for column, value in provenance.items():
        merged[column] = value

    lead = [
        "model_id",
        "model_kind",
        "config",
        "outcome",
        "ame_prob_median_full",
        "ame_prob_median_full_retained",
        "ame_prob_median_without_flagged",
        "refit_shift_ame_prob_median",
        "composition_shift_ame_prob_median",
        "total_shift_ame_prob_median",
        "delta_ame_prob_median",
        "prob_ame_pos_full",
        "prob_ame_pos_full_retained",
        "prob_ame_pos_without_flagged",
        "excluded_subject_ids",
        "reference_pareto_k",
        "good_k_threshold",
        "n_full",
        "n_without_flagged",
        "converged",
        "max_rhat",
        "min_ess",
        "min_bfmi",
        "n_divergences",
    ]
    remaining = [column for column in merged.columns if column not in lead]
    return merged.loc[:, [*lead, *remaining]]


def _one_value(frame: pd.DataFrame, column: str) -> Any:
    """Return a required bundle-wide scalar, rejecting mixed rows."""
    if column not in frame.columns:
        raise ValueError(f"influence summary is missing required column {column!r}")
    values = frame[column]
    if values.isna().any() or values.nunique(dropna=False) != 1:
        raise ValueError(f"influence summary has mixed or missing {column!r} values")
    return values.iloc[0]


def _compare_summary_metrics(
    declared: pd.DataFrame,
    recomputed: pd.DataFrame,
    *,
    declared_suffix: str,
) -> None:
    """Require saved AME medians/probabilities to match their bound artefact."""
    declared_by_outcome = declared.set_index(declared["outcome"].astype(str))
    recomputed_by_outcome = recomputed.set_index(recomputed["outcome"].astype(str))
    if declared_by_outcome.index.duplicated().any():
        raise ValueError("influence bundle contains duplicate outcome rows")
    if recomputed_by_outcome.index.duplicated().any():
        raise ValueError("bound treatment summary contains duplicate outcome rows")
    if set(declared_by_outcome.index) != set(recomputed_by_outcome.index):
        raise ValueError("influence bundle and bound treatment summary cover different outcomes")
    for metric in ("ame_prob_median", "prob_ame_pos"):
        column = f"{metric}_{declared_suffix}"
        if column not in declared_by_outcome.columns:
            raise ValueError(f"influence summary is missing {column!r}")
        left = pd.to_numeric(declared_by_outcome[column], errors="coerce").to_numpy()
        right = pd.to_numeric(
            recomputed_by_outcome.loc[declared_by_outcome.index, metric],
            errors="coerce",
        ).to_numpy()
        if not np.isfinite(left).all() or not np.allclose(
            left, right, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"saved {column} values do not match their bound artefact")


def _registered_leave_out_free_variable_contract(
    model_id: str,
    config: str,
    *,
    model_output_root: Path,
) -> tuple[str, ...]:
    """Reconstruct the registered leave-out model's exact ordered free RVs.

    A candidate CSV and trace both carry a free-variable declaration, but those
    two declarations can be edited together.  They therefore cannot define the
    scope of their own convergence gate.  Rebuilding the current registered
    specification against the current primary artefacts supplies an independent
    contract: every free random variable that PyMC creates, in construction
    order, must be declared, present in the trace, and included in convergence
    validation.
    """
    _module, spec = resolve_registered_spec(model_id)
    reference = load_influence_reference(
        spec,
        config,
        model_output_root=model_output_root,
    )
    influence_build = build_influence_model(spec, reference)
    names = tuple(rv.name for rv in influence_build.built.model.free_RVs)
    if not names or len(names) != len(set(names)):
        raise ValueError(
            "registered leave-out model has an empty or duplicate free-variable contract"
        )
    return names


def evaluate_influence_bundle(
    summary: pd.DataFrame | None,
    primary_dir: Path,
    report_config: dict[str, Any],
    expected_config: str,
    *,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed validation used both before installation and during rendering.

    Validation binds the summary to the current primary config, trace, Pareto-k
    table and treatment summary; verifies the content-addressed sensitivity trace;
    reconstructs its headline AMEs; and recomputes convergence over the exact
    free-variable contract saved by the PyMC build.
    """
    # Quarto executes a copied report from inside its model directory and passes
    # ``Path(".")`` here.  Resolve that directory once so reconstructing the
    # registered leave-out contract uses the model directory's actual parent,
    # rather than accidentally looking for ``<model>/<model>/config.json``.
    primary_dir = Path(primary_dir).resolve()
    if trace_path is not None:
        trace_path = Path(trace_path).resolve()

    result: dict[str, Any] = {
        "ready": False,
        "reason": "influence_sensitivity.csv is absent",
        "max_ame_shift": np.nan,
        "max_refit_ame_shift": np.nan,
        "max_composition_ame_shift": np.nan,
        "max_total_ame_shift": np.nan,
        "recomputed_convergence": None,
    }
    primary_trace = None
    sensitivity_trace = None
    if summary is None or summary.empty:
        return result

    try:
        model_id = str(_one_value(summary, "model_id"))
        model_kind = str(_one_value(summary, "model_kind"))
        if model_kind not in INFLUENCE_SUPPORTED_KINDS:
            raise ValueError(f"unsupported model kind in influence bundle: {model_kind!r}")
        if model_id != str(report_config.get("model_id")):
            raise ValueError("influence model id does not match current primary config")
        if model_kind != str(report_config.get("kind")):
            raise ValueError("influence model kind does not match current primary config")
        if str(_one_value(summary, "config")) != expected_config:
            raise ValueError("influence sampling config does not match the report directory")
        if (
            str(_one_value(summary, "sensitivity_method"))
            != "direct_leave_out_refit_excluding_all_psis_flagged_children"
        ):
            raise ValueError("influence sensitivity method is not the direct leave-out contract")
        if str(_one_value(summary, "loo_unit")) != "child":
            raise ValueError("influence sensitivity is not bound to the child-level LOO unit")
        if (
            str(_one_value(summary, "ame_comparison_population"))
            != "common_retained_children"
        ):
            raise ValueError(
                "influence AME comparison is not standardised to retained children"
            )
        if (
            str(_one_value(summary, "shift_decomposition"))
            != "total_shift=composition_shift+refit_shift"
        ):
            raise ValueError("influence AME shift decomposition contract is invalid")
        if (
            str(_one_value(summary, "delta_ame_prob_median_alias"))
            != "total_shift_ame_prob_median"
        ):
            raise ValueError("legacy influence AME delta alias is ambiguous")
        if str(_one_value(summary, "convergence_scope")) != "all_free_variables":
            raise ValueError("influence convergence scope is not all free variables")
        if str(_one_value(summary, "data_sha256")) != str(
            report_config.get("data_sha256")
        ):
            raise ValueError("influence data checksum does not match current primary config")
        if int(_one_value(summary, "n_full")) != int(report_config.get("n_obs", -1)):
            raise ValueError("influence full-sample count does not match current primary config")

        current_primary_hashes = hash_primary_artifacts(primary_dir)
        for column, current_hash in current_primary_hashes.items():
            if str(_one_value(summary, column)) != current_hash:
                raise ValueError(f"stale or mixed influence bundle: {column} does not match")

        pareto = pd.read_csv(primary_dir / "pareto_k.csv")
        missing = sorted(_PARETO_REQUIRED_COLUMNS - set(pareto.columns))
        if missing:
            raise ValueError(f"current pareto_k.csv is missing columns: {missing}")
        n_full = int(report_config.get("n_obs", -1))
        observation_indices = pd.to_numeric(
            pareto["observation_index"], errors="coerce"
        )
        pareto_k = pd.to_numeric(pareto["pareto_k"], errors="coerce")
        thresholds = pd.to_numeric(pareto["good_k_threshold"], errors="coerce")
        if (
            not np.isfinite(observation_indices).all()
            or not np.isfinite(pareto_k).all()
            or not np.isfinite(thresholds).all()
            or thresholds.nunique() != 1
        ):
            raise ValueError("current Pareto-k table has invalid values or thresholds")
        if not np.equal(observation_indices, np.floor(observation_indices)).all():
            raise ValueError("current Pareto-k table has non-integer observation indices")
        integer_indices = observation_indices.astype(int)
        if (
            len(pareto) != n_full
            or set(integer_indices) != set(range(n_full))
            or integer_indices.duplicated().any()
            or pareto["subject_id"].astype(str).duplicated().any()
        ):
            raise ValueError(
                "current Pareto-k rows do not map one-to-one onto the primary children"
            )
        flagged = pareto.loc[pareto_k > thresholds]
        current_excluded = set(flagged["subject_id"].astype(str))
        declared_excluded = {
            value
            for value in str(_one_value(summary, "excluded_subject_ids")).split(";")
            if value
        }
        if not current_excluded or declared_excluded != current_excluded:
            raise ValueError("excluded children do not match the current Pareto-k flags")
        if int(_one_value(summary, "n_excluded_children")) != len(current_excluded):
            raise ValueError("excluded-child count does not match the current Pareto-k flags")
        retained_mask = np.ones(n_full, dtype=bool)
        retained_mask[integer_indices.loc[flagged.index].to_numpy()] = False
        expected_without = int(retained_mask.sum())
        if int(_one_value(summary, "n_without_flagged")) != expected_without:
            raise ValueError("leave-out sample count does not match the current flags")
        current_threshold = float(thresholds.iloc[0])
        if not np.isclose(
            float(_one_value(summary, "good_k_threshold")),
            current_threshold,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("saved Pareto-k threshold does not match the current table")
        if not np.isclose(
            float(_one_value(summary, "reference_pareto_k")),
            float(pareto_k.loc[flagged.index].max()),
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("saved maximum Pareto-k does not match the current table")

        expected_identity = {
            "model_id": model_id,
            "model_kind": model_kind,
            "config": expected_config,
            "data_sha256": str(report_config.get("data_sha256")),
            "excluded_subject_ids": sorted(current_excluded),
            "n_without_flagged": expected_without,
            **current_primary_hashes,
        }
        try:
            declared_identity = json.loads(str(_one_value(summary, "identity_json")))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "influence summary has no readable trace identity contract"
            ) from exc
        if declared_identity != expected_identity:
            raise ValueError(
                "influence trace identity contract does not match the current analysis"
            )

        primary_summary = _normalise_summary_symbol(
            pd.read_csv(primary_dir / "tau_summary.csv"),
            str(report_config.get("outcome_symbol") or ""),
        )
        _compare_summary_metrics(summary, primary_summary, declared_suffix="full")
        ci_prob = float(_one_value(summary, "ci_prob"))
        if not np.isclose(
            ci_prob,
            float(report_config.get("ci_prob", 0.95)),
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError("influence credible-interval coverage does not match primary fit")
        primary_trace = _open_readable_trace(
            primary_dir / "trace.nc", label="primary"
        )
        primary_G, primary_constant = _trace_treatment_data(
            primary_trace,
            expected_n=n_full,
            label="primary",
        )
        primary_outcomes = list(primary_summary["outcome"].astype(str))
        if model_kind == "joint":
            recomputed_primary_full = _report.tau_summary_joint(
                primary_trace,
                primary_outcomes,
                ci_prob=ci_prob,
                G=primary_G,
            )
            recomputed_primary_retained = _report.tau_summary_joint(
                primary_trace,
                primary_outcomes,
                ci_prob=ci_prob,
                G=primary_G,
                row_mask=retained_mask,
            )
        else:
            primary_moderators = _trace_itt_moderators(
                primary_trace,
                primary_constant,
                expected_n=n_full,
                label="primary",
            )
            recomputed_primary_full = pd.DataFrame(
                [
                    _report.tau_summary_itt(
                        primary_trace,
                        ci_prob=ci_prob,
                        G=primary_G,
                        moderators=primary_moderators,
                    )
                ]
            )
            recomputed_primary_retained = pd.DataFrame(
                [
                    _report.tau_summary_itt(
                        primary_trace,
                        ci_prob=ci_prob,
                        G=primary_G,
                        moderators=primary_moderators,
                        row_mask=retained_mask,
                    )
                ]
            )
        recomputed_primary_full = _normalise_summary_symbol(
            recomputed_primary_full,
            str(report_config.get("outcome_symbol") or ""),
        )
        recomputed_primary_retained = _normalise_summary_symbol(
            recomputed_primary_retained,
            str(report_config.get("outcome_symbol") or ""),
        )
        _compare_summary_metrics(
            _suffix_summary_metrics(primary_summary, "bound_primary"),
            recomputed_primary_full,
            declared_suffix="bound_primary",
        )
        _compare_summary_metrics(
            summary,
            recomputed_primary_retained,
            declared_suffix="full_retained",
        )

        trace_filename = str(_one_value(summary, "trace_file"))
        if Path(trace_filename).name != trace_filename:
            raise ValueError("sensitivity trace_file must be a report-local basename")
        declared_trace_hash = str(_one_value(summary, "sensitivity_trace_sha256"))
        expected_suffix = f"-{declared_trace_hash[:16]}.nc"
        if not trace_filename.startswith(f"{INFLUENCE_TRACE_STEM}-") or not trace_filename.endswith(
            expected_suffix
        ):
            raise ValueError("sensitivity trace filename is not hash-suffixed")
        resolved_trace = trace_path or (primary_dir / trace_filename)
        if not resolved_trace.is_file():
            raise FileNotFoundError(f"sensitivity trace is missing: {resolved_trace}")
        if sha256_file(resolved_trace) != declared_trace_hash:
            raise ValueError("sensitivity trace SHA-256 does not match the summary")
        sensitivity_trace = _open_readable_trace(
            resolved_trace, label="sensitivity"
        )

        free_contract = str(_one_value(summary, "free_variables"))
        free_names = free_contract.split("|") if free_contract else []
        if not free_names or len(free_names) != len(set(free_names)):
            raise ValueError("free-variable contract is empty or contains duplicates")
        registered_free_names = _registered_leave_out_free_variable_contract(
            model_id,
            expected_config,
            model_output_root=primary_dir.parent,
        )
        if tuple(free_names) != registered_free_names:
            raise ValueError(
                "declared free-variable contract does not match the current "
                "registered leave-out model"
            )
        if len(free_names) != int(_one_value(summary, "n_free_variables")):
            raise ValueError("free-variable count does not match the declared contract")
        contract_hash = hashlib.sha256(free_contract.encode("utf-8")).hexdigest()
        if str(_one_value(summary, "free_variables_sha256")) != contract_hash:
            raise ValueError("free-variable contract SHA-256 does not match")
        try:
            trace_free_names = json.loads(
                str(
                    sensitivity_trace.posterior.attrs[
                        INFLUENCE_FREE_VARIABLES_ATTR
                    ]
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "sensitivity trace has no readable free-variable contract attribute"
            ) from exc
        if trace_free_names != free_names:
            raise ValueError(
                "sensitivity trace free-variable contract does not match the summary"
            )
        missing_free = sorted(set(free_names) - set(sensitivity_trace.posterior.data_vars))
        if missing_free:
            raise ValueError(f"sensitivity trace is missing free variables: {missing_free}")
        try:
            trace_identity = json.loads(
                str(sensitivity_trace.posterior.attrs[INFLUENCE_IDENTITY_ATTR])
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "sensitivity trace has no readable identity contract attribute"
            ) from exc
        if trace_identity != declared_identity:
            raise ValueError(
                "sensitivity trace identity contract does not match the summary"
            )
        sampling_json = str(_one_value(summary, "sampling_json"))
        try:
            declared_sampling = json.loads(sampling_json)
            trace_sampling = json.loads(
                str(sensitivity_trace.posterior.attrs[INFLUENCE_SAMPLING_ATTR])
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "sensitivity trace or summary has no readable sampling contract"
            ) from exc
        if not isinstance(declared_sampling, dict) or trace_sampling != declared_sampling:
            raise ValueError(
                "sensitivity trace sampling contract does not match the summary"
            )
        sampling = report_config.get("sampling")
        if not isinstance(sampling, dict):
            raise ValueError("current primary config has no sampling contract")
        for key in ("draws", "tune", "chains", "target_accept"):
            declared = _one_value(summary, f"sampling_{key}")
            expected = sampling.get(key)
            if key == "target_accept":
                matches = np.isclose(float(declared), float(expected), rtol=0, atol=1e-12)
            else:
                matches = int(declared) == int(expected)
            if not matches:
                raise ValueError(f"sensitivity sampling_{key} does not match the primary fit")
        if sensitivity_trace.posterior.sizes.get("chain") != int(sampling["chains"]):
            raise ValueError("sensitivity trace chain count does not match its contract")
        if sensitivity_trace.posterior.sizes.get("draw") != int(sampling["draws"]):
            raise ValueError("sensitivity trace draw count does not match its contract")

        G, constant = _trace_treatment_data(
            sensitivity_trace,
            expected_n=expected_without,
            label="sensitivity",
        )

        recomputed = _diag.subfit_convergence(
            sensitivity_trace,
            label=f"{model_id} report influence validation",
            var_names=free_names,
        )
        result["recomputed_convergence"] = recomputed
        if recomputed.get("converged") is not True:
            raise ValueError("sensitivity trace fails the recomputed convergence gate")
        if (
            recomputed["max_rhat"] > _diag.RHAT_MAX
            or recomputed["min_ess"] < _diag.ESS_THRESHOLD
            or recomputed["min_bfmi"] < _diag.BFMI_THRESHOLD
            or recomputed["n_divergences"] != 0
        ):
            raise ValueError("sensitivity trace violates an explicit convergence threshold")
        converged_values = summary["converged"].astype(str).str.lower()
        if not converged_values.eq("true").all():
            raise ValueError("saved influence convergence flag is not true for every row")
        for column in ("max_rhat", "min_ess", "min_bfmi"):
            recorded = pd.to_numeric(summary[column], errors="coerce").to_numpy()
            if not np.isfinite(recorded).all() or not np.allclose(
                recorded,
                float(recomputed[column]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(f"saved {column} does not match the sensitivity trace")
        recorded_divergences = pd.to_numeric(
            summary["n_divergences"], errors="coerce"
        ).to_numpy()
        if not np.equal(recorded_divergences, recomputed["n_divergences"]).all():
            raise ValueError("saved divergence count does not match the sensitivity trace")

        if model_kind == "joint":
            recomputed_summary = _report.tau_summary_joint(
                sensitivity_trace, primary_outcomes, ci_prob=ci_prob, G=G
            )
        else:
            moderators = _trace_itt_moderators(
                sensitivity_trace,
                constant,
                expected_n=expected_without,
                label="sensitivity",
            )
            recomputed_summary = pd.DataFrame(
                [
                    _report.tau_summary_itt(
                        sensitivity_trace,
                        ci_prob=ci_prob,
                        G=G,
                        moderators=moderators,
                    )
                ]
            )
        recomputed_summary = _normalise_summary_symbol(
            recomputed_summary, str(report_config.get("outcome_symbol") or "")
        )
        _compare_summary_metrics(
            summary, recomputed_summary, declared_suffix="without_flagged"
        )

        full_ame = pd.to_numeric(summary["ame_prob_median_full"], errors="coerce")
        retained_ame = pd.to_numeric(
            summary["ame_prob_median_full_retained"], errors="coerce"
        )
        leave_out_ame = pd.to_numeric(
            summary["ame_prob_median_without_flagged"], errors="coerce"
        )
        composition_shift = pd.to_numeric(
            summary["composition_shift_ame_prob_median"], errors="coerce"
        )
        refit_shift = pd.to_numeric(
            summary["refit_shift_ame_prob_median"], errors="coerce"
        )
        total_shift = pd.to_numeric(
            summary["total_shift_ame_prob_median"], errors="coerce"
        )
        legacy_delta = pd.to_numeric(
            summary["delta_ame_prob_median"], errors="coerce"
        )
        vectors = (
            full_ame,
            retained_ame,
            leave_out_ame,
            composition_shift,
            refit_shift,
            total_shift,
            legacy_delta,
        )
        if not all(np.isfinite(values).all() for values in vectors):
            raise ValueError("influence AME decomposition contains non-finite values")
        expected_composition = retained_ame - full_ame
        expected_refit = leave_out_ame - retained_ame
        expected_total = leave_out_ame - full_ame
        if not np.allclose(
            composition_shift, expected_composition, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                "saved composition AME shifts do not equal retained minus full AMEs"
            )
        if not np.allclose(
            refit_shift, expected_refit, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                "saved common-population refit AME shifts do not equal leave-out "
                "minus retained-primary AMEs"
            )
        if not np.allclose(total_shift, expected_total, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "saved total AME shifts do not equal leave-out minus full AMEs"
            )
        if not np.allclose(
            total_shift,
            composition_shift + refit_shift,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("saved AME shift components do not sum to the total shift")
        if not np.allclose(legacy_delta, total_shift, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "legacy delta_ame_prob_median is not the declared total-shift alias"
            )
        max_refit = float(refit_shift.abs().max())
        max_composition = float(composition_shift.abs().max())
        max_total = float(total_shift.abs().max())
        result.update(
            ready=True,
            reason="",
            max_ame_shift=max_total,
            max_refit_ame_shift=max_refit,
            max_composition_ame_shift=max_composition,
            max_total_ame_shift=max_total,
            trace_file=trace_filename,
            sensitivity_trace_sha256=declared_trace_hash,
        )
    except Exception as exc:
        result["reason"] = str(exc)
    finally:
        if primary_trace is not None:
            _close_trace(primary_trace)
        if sensitivity_trace is not None:
            _close_trace(sensitivity_trace)
    return result


def write_influence_artifacts(
    trace,
    summary: pd.DataFrame,
    reference: InfluenceReference,
    config: str,
    *,
    sensitivity_root: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path]:
    """Persist a central run, then atomically install a validated report bundle.

    The main fit copies shared Quarto partials into its self-contained output
    directory.  An influence sensitivity normally runs later, so refresh the one
    shared partial that reads these new artefacts before the report is re-rendered.
    A failed or stale candidate remains centrally auditable but never replaces a
    previously valid report-local CSV; that CSV is the bundle index and is installed
    atomically only after the content-addressed trace and partial are in place.
    """
    root = sensitivity_root or (_paths.stat_dir() / "influence_sensitivity")
    central_dir = root / f"{reference.metadata['model_id']}-{config}"
    central_dir.mkdir(parents=True, exist_ok=True)
    temporary_trace = central_dir / f".{INFLUENCE_TRACE_STEM}.{uuid.uuid4().hex}.tmp.nc"
    try:
        trace.to_netcdf(temporary_trace)
        trace_hash = sha256_file(temporary_trace)
        trace_filename = f"{INFLUENCE_TRACE_STEM}-{trace_hash[:16]}.nc"
        central_trace = central_dir / trace_filename
        if central_trace.is_file() and sha256_file(central_trace) == trace_hash:
            temporary_trace.unlink()
        else:
            os.replace(temporary_trace, central_trace)
    finally:
        if temporary_trace.exists():
            temporary_trace.unlink()

    bound_summary = summary.copy()
    bound_summary["trace_file"] = trace_filename
    bound_summary["sensitivity_trace_sha256"] = trace_hash
    report_config = _read_json(reference.model_dir / "config.json")
    validation = evaluate_influence_bundle(
        bound_summary,
        reference.model_dir,
        report_config,
        config,
        trace_path=central_trace,
    )
    bound_summary["bundle_validation_passed"] = bool(validation["ready"])
    bound_summary["bundle_validation_reason"] = str(validation["reason"])
    run_token = trace_hash[:16]
    run_csv = central_dir / f"influence_sensitivity-{run_token}.csv"
    _atomic_write_csv(bound_summary, run_csv)
    if not validation["ready"]:
        failed_csv = central_dir / f"influence_sensitivity_failed-{run_token}.csv"
        _atomic_write_csv(bound_summary, failed_csv)
        raise InfluenceBundleError(
            "influence refit was preserved centrally but not installed beside the "
            f"report: {validation['reason']}"
        )

    central_csv = central_dir / INFLUENCE_SENSITIVITY_FILENAME
    _atomic_write_csv(bound_summary, central_csv)
    report_trace = reference.model_dir / trace_filename
    report_csv = reference.model_dir / INFLUENCE_SENSITIVITY_FILENAME
    # Install the immutable trace first. If anything below fails, the old CSV still
    # points to its old immutable trace and remains a complete bundle.
    _atomic_copy(central_trace, report_trace)
    if sha256_file(report_trace) != trace_hash:
        raise InfluenceBundleError("report-local sensitivity trace failed SHA-256 verification")
    source_partial = _paths.DOCS_DIR / "models" / "_partials" / "_diagnostics.qmd"
    if not source_partial.is_file():
        raise FileNotFoundError(f"shared diagnostic report partial is missing: {source_partial}")
    report_partial = reference.model_dir / "_partials" / "_diagnostics.qmd"
    _atomic_copy(source_partial, report_partial)
    # The CSV is the mutable bundle index and must be the final atomic operation.
    _atomic_write_csv(bound_summary, report_csv)
    return bound_summary, central_csv, report_csv


def run_influence_sensitivity(
    model_id: str,
    config: str,
    *,
    random_seed: int = 20260715,
    cores: int | None = None,
    model_output_root: Path | None = None,
    sensitivity_root: Path | None = None,
) -> pd.DataFrame:
    """Run one validated direct leave-all-flagged-children-out sensitivity refit."""
    _module, spec = resolve_registered_spec(model_id)
    reference = load_influence_reference(
        spec, config, model_output_root=model_output_root
    )
    influence_build = build_influence_model(spec, reference)
    trace, sampling = _sample_influence_model(
        influence_build.built,
        reference,
        config,
        random_seed=random_seed,
        cores=cores,
    )
    summary = summarise_influence_refit(
        spec,
        reference,
        influence_build,
        trace,
        config=config,
        sampling=sampling,
    )
    installed_summary, _central_csv, _report_csv = write_influence_artifacts(
        trace,
        summary,
        reference,
        config,
        sensitivity_root=sensitivity_root,
    )
    return installed_summary
