# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trace-backed release evidence for the phoneme-blending response link.

Phoneme blending (``B``) has ten three-alternative forced-choice items.  The
ordinary logit mean used by the main ITT model permits expected scores below
chance, whereas the registered robustness companion constrains the mean to
``1/3 + 2/3 * expit(eta)``.  Because that modelling choice changes the scientific
conclusion, neither fit is sufficient release evidence on its own.

This module validates the two completed reporting fits, recomputes their
items-scale estimands and convergence checks from the saved traces, verifies
identical fitted children and treatment assignments, and writes a
content-addressed two-trace bundle.  Reports consume only the small installed CSV.
The full trace recomputation runs at build time, and the central archive manifest
is written only after it passes; every later check (report render, key findings,
``release.evaluate_publication``) byte-binds the installed CSV to that manifest
and re-hashes both current fit directories, so a stale or edited pair cannot be
quoted anywhere without failing closed (2026-08-20 ITT review, finding 1).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models import diagnostics as _diag
from language_reading_predictors.statistical_models import reporting as _report
from language_reading_predictors.statistical_models.sensitivity import sha256_file

BLENDING_SENSITIVITY_FILENAME = "blending_link_sensitivity.csv"
BLENDING_SENSITIVITY_SCHEMA_VERSION = 2
BLENDING_PRIMARY_MODEL_ID = "lrp-rli-itt-008"
BLENDING_COMPANION_MODEL_ID = "lrp-rli-itt-108"
BLENDING_LINK_MODELS: tuple[tuple[str, str], ...] = (
    (BLENDING_PRIMARY_MODEL_ID, "logit"),
    (BLENDING_COMPANION_MODEL_ID, "three_choice_guessing_floor"),
)
_FREE_VARIABLES = ("alpha", "tau", "gamma_own", "gamma_A", "kappa")
_SUMMARY_COLUMNS = {
    "effect_items_median": "tau_prob_median",
    "effect_items_lo": "tau_prob_lo",
    "effect_items_hi": "tau_prob_hi",
    "prob_effect_positive": "prob_ame_pos",
}

# Every scientific CSV or PNG that ``_results_itt.qmd`` can expose for the two
# graded phoneme-blending fits, plus the single-fit prior pushforward rendered by
# ``_priors.qmd`` beside the trace-recomputed paired pushforwards.  The paired
# summary stores a complete SHA-256 map for this exact set.  Local report generation
# and the public-release validator re-hash both current fit directories, so changing
# a table or figure after the trace-backed pair is built closes the release gate
# rather than publishing a stale mixture of artefacts.  The paired summary itself is
# intentionally absent: it is the manifest carrying these hashes.
BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS: tuple[str, ...] = (
    "analysis_set.csv",
    "attrition_bounds.csv",
    "tau_summary.csv",
    "tau_forest.png",
    "rope_summary.csv",
    "rope_sensitivity.csv",
    "rope_summary.png",
    "rope_benefit_curve.png",
    "predicted_scores.csv",
    "predicted_scores.png",
    "predicted_effect.png",
    "icon_array.png",
    "arm_overlap_mean.png",
    "arm_overlap_predictive.png",
    "prior_pushforward.csv",
)
_SCIENTIFIC_ARTIFACT_HASH_COLUMN = "scientific_artifacts_sha256"


@dataclass(frozen=True, slots=True)
class _FitRecord:
    """Validated inputs and recomputed evidence from one completed fit."""

    model_dir: Path
    config: dict[str, Any]
    model_id: str
    score_mean_link: str
    config_name: str
    config_sha256: str
    trace_sha256: str
    row_map_sha256: str
    data_sha256: str
    environment_lock_sha256: str
    source_commit: str
    n_obs: int
    subject_ids: tuple[str, ...]
    treatment: tuple[int, ...]
    sampling: dict[str, Any]
    summary: dict[str, float]
    rope: dict[str, float]
    prior: dict[str, float]
    convergence: dict[str, Any]
    loo_elpd: float
    loo_p: float
    pareto_k_max: float
    good_k_threshold: float
    loo_reliable: bool
    loo_i: np.ndarray
    pareto_k: np.ndarray


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _one_csv_row(path: Path, *, label: str) -> dict[str, Any]:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label} is not readable: {path}") from exc
    if len(frame) != 1:
        raise ValueError(f"{label} must contain exactly one row")
    return frame.iloc[0].to_dict()


def _scientific_artifact_hashes(directory: Path) -> dict[str, str]:
    """Hash the complete scientific-artefact surface rendered by the ITT partial."""

    missing = [
        name
        for name in BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS
        if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "fit is missing scientific report artefacts: " + ", ".join(missing)
        )
    return {
        name: sha256_file(directory / name)
        for name in BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS
    }


def _encode_scientific_artifact_hashes(hashes: Mapping[str, str]) -> str:
    """Return the canonical JSON representation stored in the paired CSV."""

    return json.dumps(dict(hashes), sort_keys=True, separators=(",", ":"))


def _decode_scientific_artifact_hashes(value: Any) -> dict[str, str]:
    """Validate and decode one complete per-fit scientific-artefact hash map."""

    try:
        decoded = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("scientific artefact hash map is not valid JSON") from exc
    if not isinstance(decoded, dict) or set(decoded) != set(
        BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS
    ):
        raise ValueError(
            "scientific artefact hash map does not match the ITT report contract"
        )
    malformed = [
        name
        for name, digest in decoded.items()
        if not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in digest)
    ]
    if malformed:
        raise ValueError(
            "scientific artefact hash map contains malformed SHA-256 values: "
            + ", ".join(sorted(malformed))
        )
    return {str(name): str(digest).lower() for name, digest in decoded.items()}


def _values_match(recorded: Any, recomputed: Any) -> bool:
    try:
        left = float(recorded)
        right = float(recomputed)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(left)
        and np.isfinite(right)
        and np.isclose(left, right, rtol=1e-9, atol=1e-11)
    )


def _recorded_bool(value: Any) -> bool | None:
    """Parse a manifest boolean strictly; unrecognised text is ``None`` (fail closed).

    ``bool("False")`` is ``True``, so the previous ``bool(row[...])`` idiom was
    correct only while pandas parsed the column to a boolean dtype — a quoted or
    mixed-case value would have failed open on ``converged`` and false-alarmed on
    ``loo_reliable`` (2026-08-20 ITT review, finding 6).
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _load_row_map(
    path: Path,
    *,
    n_obs: int,
) -> tuple[tuple[str, ...], np.ndarray, float]:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Pareto-k table is not readable: {path}") from exc
    required = {
        "observation_index",
        "subject_id",
        "pareto_k",
        "good_k_threshold",
        "loo_reliable",
    }
    if not required.issubset(frame.columns) or len(frame) != n_obs:
        raise ValueError("Pareto-k table does not map every fitted observation")
    index = pd.to_numeric(frame["observation_index"], errors="coerce")
    if (
        not np.isfinite(index).all()
        or not np.equal(index, np.floor(index)).all()
        or index.astype(int).duplicated().any()
        or set(index.astype(int)) != set(range(n_obs))
    ):
        raise ValueError("Pareto-k observation indices are not a complete row map")
    ordered = frame.assign(_index=index.astype(int)).sort_values("_index")
    subjects = ordered["subject_id"].astype(str)
    if subjects.str.strip().eq("").any() or subjects.duplicated().any():
        raise ValueError("Pareto-k subject identities are blank or duplicated")
    pareto_k = pd.to_numeric(ordered["pareto_k"], errors="coerce").to_numpy(
        dtype=float
    )
    thresholds = pd.to_numeric(
        ordered["good_k_threshold"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(pareto_k).all()
        or not np.isfinite(thresholds).all()
        or not np.allclose(thresholds, thresholds[0], rtol=0.0, atol=1e-12)
        or thresholds[0] <= 0.0
    ):
        raise ValueError("Pareto-k values or reliability threshold are malformed")
    reliable_text = ordered["loo_reliable"].map(
        lambda value: str(value).strip().lower()
    )
    if not reliable_text.isin({"true", "false"}).all():
        raise ValueError("Pareto-k reliability flags must be true or false")
    reliable = reliable_text.eq("true").to_numpy(dtype=bool)
    if not np.array_equal(reliable, pareto_k <= thresholds[0]):
        raise ValueError("Pareto-k reliability flags do not match their threshold")
    return tuple(subjects), pareto_k, float(thresholds[0])


def _config_link(config: Mapping[str, Any]) -> str:
    plan = config.get("resolved_run_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("config lacks a resolved ITT run plan")
    link = str(plan.get("score_mean_link", ""))
    settings = config.get("model_settings")
    if not isinstance(settings, Mapping) or settings.get("score_mean_link") != link:
        raise ValueError("declared and resolved score-mean links do not agree")
    return link


def _load_fit_record(
    model_dir: str | Path,
    *,
    expected_model_id: str,
    expected_link: str,
    config_name: str,
) -> _FitRecord:
    """Validate one completed fit and recompute its released estimand from trace."""

    directory = Path(model_dir)
    required = {
        name: directory / name
        for name in (
            "config.json",
            "diagnostics_summary.json",
            "diagnostics.csv",
            "pareto_k.csv",
            "tau_summary.csv",
            "rope_summary.csv",
            "prior_pushforward.csv",
            "trace.nc",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{expected_model_id} is missing required artefacts: {', '.join(missing)}"
        )
    # These are not needed to recompute the paired estimand, but they are the
    # scientific tables and figures the report can publish.  Requiring the full
    # surface here makes a plotting or finalisation failure a release failure.
    _scientific_artifact_hashes(directory)

    config = _read_json(required["config.json"], label="fit config")
    if config.get("model_id") != expected_model_id:
        raise ValueError(
            f"model identity mismatch: expected {expected_model_id}, "
            f"got {config.get('model_id')!r}"
        )
    if config.get("kind") != "itt" or config.get("outcome_symbol") != "B":
        raise ValueError(f"{expected_model_id} is not the registered B ITT fit")
    link = _config_link(config)
    if link != expected_link:
        raise ValueError(
            f"{expected_model_id} declares {link!r}, expected {expected_link!r}"
        )
    if directory.name != f"{expected_model_id}-{config_name}":
        raise ValueError(
            f"fit directory {directory.name!r} does not match config {config_name!r}"
        )
    data_sha = str(config.get("data_sha256", ""))
    environment_sha = str(config.get("environment_lock_sha256", ""))
    if len(data_sha) != 64 or len(environment_sha) != 64:
        raise ValueError("fit config lacks data/environment SHA-256 provenance")
    try:
        n_obs = int(config["n_obs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fit config lacks a valid n_obs") from exc
    if n_obs <= 0:
        raise ValueError("fit config n_obs must be positive")
    sampling = config.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("fit config lacks sampling provenance")
    source = (config.get("provenance") or {}).get("source") or {}
    source_commit = str(source.get("commit", ""))
    if len(source_commit) != 40 or source.get("dirty") is not False:
        raise ValueError("fit must come from a clean, recorded source commit")

    diagnostic_names = pd.read_csv(required["diagnostics.csv"], index_col=0).index
    if tuple(str(name) for name in diagnostic_names) != _FREE_VARIABLES:
        raise ValueError(
            f"{expected_model_id} diagnostics do not cover the fixed B free-variable set"
        )
    saved_gate = _read_json(
        required["diagnostics_summary.json"], label="diagnostics summary"
    )
    if not _report.convergence_gate_clean_passed(saved_gate):
        raise ValueError(f"{expected_model_id} did not pass its saved clean gate")

    saved_summary = _one_csv_row(required["tau_summary.csv"], label="tau summary")
    saved_rope = _one_csv_row(required["rope_summary.csv"], label="ROPE summary")
    saved_prior = _one_csv_row(
        required["prior_pushforward.csv"], label="prior pushforward"
    )
    subject_ids, saved_pareto_k, good_k_threshold = _load_row_map(
        required["pareto_k.csv"], n_obs=n_obs
    )

    try:
        trace = az.from_netcdf(required["trace.nc"])
    except Exception as exc:  # noqa: BLE001 - corrupt evidence is validation data
        raise ValueError(f"trace is unreadable: {required['trace.nc']}") from exc
    try:
        posterior = getattr(trace, "posterior", None)
        if posterior is None or not set(_FREE_VARIABLES).issubset(posterior.data_vars):
            raise ValueError("trace posterior lacks one or more required free variables")
        if (
            int(posterior.sizes.get("chain", -1)) != int(sampling.get("chains", -2))
            or int(posterior.sizes.get("draw", -1)) != int(sampling.get("draws", -2))
        ):
            raise ValueError("trace chain/draw dimensions do not match config")
        constant = getattr(trace, "constant_data", None)
        if constant is None or "G" not in constant:
            raise ValueError("trace constant_data lacks treatment assignment G")
        treatment_array = np.asarray(constant["G"].values, dtype=float).reshape(-1)
        if (
            treatment_array.size != n_obs
            or not np.isin(treatment_array, (0.0, 1.0)).all()
            or not np.any(treatment_array == 0.0)
            or not np.any(treatment_array == 1.0)
        ):
            raise ValueError("trace treatment assignment is malformed")
        convergence = _diag.subfit_convergence(
            trace,
            label=f"{expected_model_id} link-sensitivity validation",
            var_names=list(_FREE_VARIABLES),
        )
        if convergence.get("converged") is not True:
            raise ValueError("trace does not pass the recomputed all-variable gate")
        summary = _report.tau_summary_itt(
            trace,
            ci_prob=float(config.get("ci_prob", 0.89)),
            G=treatment_array,
            score_mean_link=link,
        )
        rope = _report.rope_summary(
            trace,
            G=treatment_array,
            n_trials=10,
            delta=1.0,
            ci_prob=float(config.get("ci_prob", 0.89)),
            score_mean_link=link,
        )
        prior = _report.prior_pushforward(
            trace,
            G=treatment_array,
            n_trials=10,
            ci_prob=float(config.get("ci_prob", 0.89)),
            score_mean_link=link,
        )
        for saved, recomputed, label in (
            (saved_summary, summary, "tau summary"),
            (saved_rope, rope, "ROPE summary"),
            (saved_prior, prior, "prior pushforward"),
        ):
            common = set(saved) & set(recomputed)
            mismatched = [
                name
                for name in common
                if isinstance(recomputed[name], (int, float, np.integer, np.floating))
                and not _values_match(saved[name], recomputed[name])
            ]
            if mismatched:
                raise ValueError(
                    f"saved {label} does not match trace: {', '.join(sorted(mismatched))}"
                )
        loo = az.loo(trace, pointwise=True)
        loo_i = np.asarray(loo.elpd_i.values, dtype=float).reshape(-1)
        pareto_k = np.asarray(loo.pareto_k.values, dtype=float).reshape(-1)
        if loo_i.size != n_obs or pareto_k.size != n_obs:
            raise ValueError("trace pointwise LOO does not align with fitted rows")
        if not np.allclose(
            saved_pareto_k,
            pareto_k,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError("saved Pareto-k row map does not match the trace")
    finally:
        close = getattr(trace, "close", None)
        if callable(close):
            close()

    return _FitRecord(
        model_dir=directory,
        config=config,
        model_id=expected_model_id,
        score_mean_link=link,
        config_name=config_name,
        config_sha256=sha256_file(required["config.json"]),
        trace_sha256=sha256_file(required["trace.nc"]),
        row_map_sha256=sha256_file(required["pareto_k.csv"]),
        data_sha256=data_sha,
        environment_lock_sha256=environment_sha,
        source_commit=source_commit,
        n_obs=n_obs,
        subject_ids=subject_ids,
        treatment=tuple(treatment_array.astype(int)),
        sampling=dict(sampling),
        summary={
            name: float(value)
            for name, value in summary.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        rope={
            name: float(value)
            for name, value in rope.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        prior={
            name: float(value)
            for name, value in prior.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        convergence=dict(convergence),
        loo_elpd=float(loo.elpd),
        loo_p=float(loo.p),
        pareto_k_max=float(np.max(pareto_k)),
        good_k_threshold=good_k_threshold,
        loo_reliable=bool(np.all(pareto_k <= good_k_threshold)),
        loo_i=loo_i,
        pareto_k=pareto_k,
    )


def _check_pair(primary: _FitRecord, companion: _FitRecord) -> None:
    """Require the two traces to differ only in the declared score-mean link."""

    shared = (
        "config_name",
        "data_sha256",
        "environment_lock_sha256",
        "source_commit",
        "n_obs",
        "subject_ids",
        "treatment",
        "sampling",
    )
    drift = [name for name in shared if getattr(primary, name) != getattr(companion, name)]
    if drift:
        raise ValueError(
            "B link fits are not a like-for-like pair; mismatched " + ", ".join(drift)
        )
    if primary.trace_sha256 == companion.trace_sha256:
        raise ValueError("B link fits unexpectedly point to identical trace bytes")
    settings_a = dict(primary.config["model_settings"])
    settings_b = dict(companion.config["model_settings"])
    settings_a.pop("score_mean_link", None)
    settings_b.pop("score_mean_link", None)
    if settings_a != settings_b:
        raise ValueError("B link fits change settings other than score_mean_link")
    plan_a = dict(primary.config["resolved_run_plan"])
    plan_b = dict(companion.config["resolved_run_plan"])
    for plan in (plan_a, plan_b):
        for name in (
            "model_id",
            "score_mean_link",
            "required_link_companion_model_id",
        ):
            plan.pop(name, None)
    if plan_a != plan_b:
        raise ValueError("B link fits resolve to different run plans beyond the link")


def _record_row(
    record: _FitRecord,
    *,
    trace_file: str,
    row_map_file: str,
) -> dict[str, Any]:
    n_intervention = int(sum(record.treatment))
    row: dict[str, Any] = {
        "schema_version": BLENDING_SENSITIVITY_SCHEMA_VERSION,
        "config": record.config_name,
        "model_id": record.model_id,
        "outcome": "B",
        "score_mean_link": record.score_mean_link,
        "sensitivity_of": BLENDING_PRIMARY_MODEL_ID,
        "data_sha256": record.data_sha256,
        "environment_lock_sha256": record.environment_lock_sha256,
        "source_commit": record.source_commit,
        "config_sha256": record.config_sha256,
        "trace_file": trace_file,
        "trace_sha256": record.trace_sha256,
        "row_map_file": row_map_file,
        "row_map_sha256": record.row_map_sha256,
        _SCIENTIFIC_ARTIFACT_HASH_COLUMN: _encode_scientific_artifact_hashes(
            _scientific_artifact_hashes(record.model_dir)
        ),
        "n": record.n_obs,
        "n_intervention": n_intervention,
        "n_control": record.n_obs - n_intervention,
        "subject_order_sha256": _text_sha256("\n".join(record.subject_ids)),
        "treatment_order_sha256": _text_sha256("".join(map(str, record.treatment))),
        "sampling_draws": int(record.sampling["draws"]),
        "sampling_tune": int(record.sampling["tune"]),
        "sampling_chains": int(record.sampling["chains"]),
        "sampling_target_accept": float(record.sampling["target_accept"]),
        "sampling_random_seed": int(record.sampling["random_seed"]),
        "ci_prob": float(record.config.get("ci_prob", 0.89)),
        "converged": True,
        "max_rhat": float(record.convergence["max_rhat"]),
        "min_ess": float(record.convergence["min_ess"]),
        "min_bfmi": float(record.convergence["min_bfmi"]),
        "n_divergences": int(record.convergence["n_divergences"]),
        "loo_elpd": record.loo_elpd,
        "loo_p": record.loo_p,
        "pareto_k_max": record.pareto_k_max,
        "good_k_threshold": record.good_k_threshold,
        "loo_reliable": record.loo_reliable,
    }
    for output_name, summary_name in _SUMMARY_COLUMNS.items():
        value = record.summary[summary_name]
        if output_name.startswith("effect_items_"):
            value *= 10.0
        row[output_name] = value
    row.update(
        prob_meaningful_benefit=float(record.rope["prob_benefit_ge_delta"]),
        prob_practically_negligible=float(record.rope["prob_in_rope"]),
        prior_effect_items_median=float(record.prior["prior_items_median"]),
        prior_effect_items_lo=float(record.prior["prior_items_lo"]),
        prior_effect_items_hi=float(record.prior["prior_items_hi"]),
    )
    return row


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _install_content_addressed_copy(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    """Atomically install one immutable evidence file after hashing both copies."""

    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}-",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise ValueError(f"archived copy failed its SHA-256 check: {source}")
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def build_blending_link_sensitivity(
    model_output_root: str | Path,
    archive_root: str | Path,
    *,
    config_name: str = "reporting",
    install_report_copies: bool = True,
) -> pd.DataFrame:
    """Validate, archive, and install the mandatory two-link B comparison."""

    models = Path(model_output_root)
    archive = Path(archive_root)
    records = [
        _load_fit_record(
            models / f"{model_id}-{config_name}",
            expected_model_id=model_id,
            expected_link=link,
            config_name=config_name,
        )
        for model_id, link in BLENDING_LINK_MODELS
    ]
    primary, companion = records
    _check_pair(primary, companion)
    loo_difference = companion.loo_i - primary.loo_i
    delta_elpd = float(np.sum(loo_difference))
    delta_se = float(np.sqrt(len(loo_difference) * np.var(loo_difference, ddof=1)))

    archive.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record in records:
        trace_name = f"{record.model_id}-{record.trace_sha256[:16]}.nc"
        _install_content_addressed_copy(
            record.model_dir / "trace.nc",
            archive / trace_name,
            expected_sha256=record.trace_sha256,
        )
        row_map_name = (
            f"{record.model_id}-rows-{record.row_map_sha256[:16]}.csv"
        )
        _install_content_addressed_copy(
            record.model_dir / "pareto_k.csv",
            archive / row_map_name,
            expected_sha256=record.row_map_sha256,
        )
        row = _record_row(
            record,
            trace_file=trace_name,
            row_map_file=row_map_name,
        )
        row["guessing_floor_minus_logit_elpd"] = delta_elpd
        row["guessing_floor_minus_logit_elpd_se"] = delta_se
        rows.append(row)
    summary = pd.DataFrame(rows)

    status = evaluate_blending_link_sensitivity(
        summary,
        trace_root=archive,
        primary_model_dirs={record.model_id: record.model_dir for record in records},
    )
    if not status["release_ready"]:
        raise ValueError(f"new B link-sensitivity archive failed validation: {status}")
    # Only a fully validated bundle may replace the previous manifest: writing it
    # before validation left a failed rebuild squatting on the last good archive
    # (2026-08-20 ITT review, finding 6). The content-addressed traces installed
    # above are additive, so validating against ``trace_root=archive`` first is
    # safe either way.
    _atomic_write_csv(summary, archive / BLENDING_SENSITIVITY_FILENAME)
    if install_report_copies:
        for record in records:
            _atomic_write_csv(
                summary,
                record.model_dir / BLENDING_SENSITIVITY_FILENAME,
            )
    return summary


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validate_archive_trace(
    path: Path,
    row: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Recompute every published paired-sensitivity quantity from one trace."""

    if not path.is_file() or sha256_file(path) != str(row["trace_sha256"]):
        raise ValueError("trace is missing or its SHA-256 does not match")
    try:
        trace = az.from_netcdf(path)
    except Exception as exc:  # noqa: BLE001 - corrupt evidence is validation data
        raise ValueError("trace is not readable NetCDF") from exc
    try:
        posterior = getattr(trace, "posterior", None)
        if posterior is None or not set(_FREE_VARIABLES).issubset(posterior.data_vars):
            raise ValueError("trace lacks required posterior variables")
        if (
            int(posterior.sizes.get("chain", -1)) != int(row["sampling_chains"])
            or int(posterior.sizes.get("draw", -1)) != int(row["sampling_draws"])
        ):
            raise ValueError("trace dimensions do not match the manifest")
        constant = getattr(trace, "constant_data", None)
        if constant is None or "G" not in constant:
            raise ValueError("trace constant_data lacks G")
        treatment = np.asarray(constant["G"].values, dtype=float).reshape(-1)
        if (
            treatment.size != int(row["n"])
            or int(np.sum(treatment == 1.0)) != int(row["n_intervention"])
            or int(np.sum(treatment == 0.0)) != int(row["n_control"])
            or _text_sha256("".join(map(str, treatment.astype(int))))
            != str(row["treatment_order_sha256"])
        ):
            raise ValueError("trace treatment assignments do not match the manifest")
        convergence = _diag.subfit_convergence(
            trace,
            label=f"{row['model_id']} archived B-link validation",
            var_names=list(_FREE_VARIABLES),
        )
        if convergence.get("converged") is not True:
            raise ValueError("trace fails the recomputed convergence gate")
        if _recorded_bool(row["converged"]) is not True:
            raise ValueError("manifest convergence flag does not match the trace")
        for column in ("max_rhat", "min_ess", "min_bfmi"):
            if not _values_match(row[column], convergence[column]):
                raise ValueError(f"trace does not reproduce {column}")
        if int(row["n_divergences"]) != int(convergence["n_divergences"]):
            raise ValueError("trace does not reproduce n_divergences")
        summary = _report.tau_summary_itt(
            trace,
            ci_prob=float(row["ci_prob"]),
            G=treatment,
            score_mean_link=str(row["score_mean_link"]),
        )
        for output_name, summary_name in _SUMMARY_COLUMNS.items():
            value = summary[summary_name]
            if output_name.startswith("effect_items_"):
                value *= 10.0
            if not _values_match(row[output_name], value):
                raise ValueError(f"trace does not reproduce {output_name}")
        rope = _report.rope_summary(
            trace,
            G=treatment,
            n_trials=10,
            delta=1.0,
            ci_prob=float(row["ci_prob"]),
            score_mean_link=str(row["score_mean_link"]),
        )
        for column, key in (
            ("prob_meaningful_benefit", "prob_benefit_ge_delta"),
            ("prob_practically_negligible", "prob_in_rope"),
        ):
            if not _values_match(row[column], rope[key]):
                raise ValueError(f"trace does not reproduce {column}")
        prior = _report.prior_pushforward(
            trace,
            G=treatment,
            n_trials=10,
            ci_prob=float(row["ci_prob"]),
            score_mean_link=str(row["score_mean_link"]),
        )
        for column, key in (
            ("prior_effect_items_median", "prior_items_median"),
            ("prior_effect_items_lo", "prior_items_lo"),
            ("prior_effect_items_hi", "prior_items_hi"),
        ):
            if not _values_match(row[column], prior[key]):
                raise ValueError(f"trace does not reproduce {column}")
        loo = az.loo(trace, pointwise=True)
        loo_i = np.asarray(loo.elpd_i.values, dtype=float).reshape(-1)
        pareto_k = np.asarray(loo.pareto_k.values, dtype=float).reshape(-1)
        if loo_i.size != int(row["n"]) or pareto_k.size != int(row["n"]):
            raise ValueError("trace pointwise LOO does not align with the manifest")
        for column, value in (
            ("loo_elpd", loo.elpd),
            ("loo_p", loo.p),
            ("pareto_k_max", np.max(pareto_k)),
        ):
            if not _values_match(row[column], value):
                raise ValueError(f"trace does not reproduce {column}")
    finally:
        close = getattr(trace, "close", None)
        if callable(close):
            close()
    return {"loo_i": loo_i, "pareto_k": pareto_k}


def _validate_archive_row_map(
    path: Path,
    row: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Validate the archived child/observation identity map and Pareto values."""

    if not path.is_file() or sha256_file(path) != str(row["row_map_sha256"]):
        raise ValueError("row map is missing or its SHA-256 does not match")
    subjects, pareto_k, threshold = _load_row_map(path, n_obs=int(row["n"]))
    if _text_sha256("\n".join(subjects)) != str(row["subject_order_sha256"]):
        raise ValueError("row map does not reproduce the fitted child order")
    if not _values_match(row["pareto_k_max"], np.max(pareto_k)):
        raise ValueError("row map does not reproduce pareto_k_max")
    if not _values_match(row["good_k_threshold"], threshold):
        raise ValueError("row map does not reproduce good_k_threshold")
    if bool(np.all(pareto_k <= threshold)) is not _recorded_bool(row["loo_reliable"]):
        raise ValueError("row map does not reproduce loo_reliable")
    return {"pareto_k": pareto_k}


def evaluate_blending_link_sensitivity(
    summary: pd.DataFrame | None,
    *,
    trace_root: str | Path,
    primary_model_dirs: Mapping[str, str | Path] | None = None,
    trace_validator: Callable[
        [Path, Mapping[str, Any]], Mapping[str, np.ndarray] | None
    ] = _validate_archive_trace,
    row_map_validator: Callable[
        [Path, Mapping[str, Any]], Mapping[str, np.ndarray] | None
    ] = _validate_archive_row_map,
) -> dict[str, Any]:
    """Fail-closed validator used by report generation and public release."""

    status: dict[str, Any] = {
        "ready": False,
        "complete": False,
        "paired": False,
        "traces_validated": False,
        "row_maps_validated": False,
        "scientific_artifacts_bound": False,
        "scientific_artifacts_current": False,
        "primary_fits_current": False,
        "archive_ready": False,
        "release_ready": False,
        "reason": "",
    }
    if summary is None or summary.empty:
        status["reason"] = "the B link-sensitivity summary is missing"
        return status
    required = {
        "schema_version",
        "config",
        "model_id",
        "outcome",
        "score_mean_link",
        "sensitivity_of",
        "data_sha256",
        "environment_lock_sha256",
        "source_commit",
        "config_sha256",
        "trace_file",
        "trace_sha256",
        "row_map_file",
        "row_map_sha256",
        _SCIENTIFIC_ARTIFACT_HASH_COLUMN,
        "n",
        "n_intervention",
        "n_control",
        "subject_order_sha256",
        "treatment_order_sha256",
        "sampling_draws",
        "sampling_tune",
        "sampling_chains",
        "sampling_target_accept",
        "sampling_random_seed",
        "ci_prob",
        "converged",
        "effect_items_median",
        "effect_items_lo",
        "effect_items_hi",
        "prob_effect_positive",
        "prob_meaningful_benefit",
        "prob_practically_negligible",
        "prior_effect_items_median",
        "prior_effect_items_lo",
        "prior_effect_items_hi",
        "max_rhat",
        "min_ess",
        "min_bfmi",
        "n_divergences",
        "loo_elpd",
        "loo_p",
        "pareto_k_max",
        "good_k_threshold",
        "loo_reliable",
        "guessing_floor_minus_logit_elpd",
        "guessing_floor_minus_logit_elpd_se",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        status["reason"] = "summary lacks columns: " + ", ".join(missing)
        return status
    expected = dict(BLENDING_LINK_MODELS)
    rows = summary[summary["model_id"].isin(expected)].copy()
    if (
        len(summary) != 2
        or len(rows) != 2
        or rows["model_id"].duplicated().any()
    ):
        status["reason"] = "summary must contain exactly the 008 and 108 fits"
        return status
    rows = rows.set_index("model_id").loc[list(expected)].reset_index()
    status["complete"] = bool(
        rows["outcome"].astype(str).eq("B").all()
        and rows["sensitivity_of"].astype(str).eq(BLENDING_PRIMARY_MODEL_ID).all()
        and all(
            str(row.score_mean_link) == expected[str(row.model_id)]
            for row in rows.itertuples()
        )
        and pd.to_numeric(rows["schema_version"], errors="coerce")
        .eq(BLENDING_SENSITIVITY_SCHEMA_VERSION)
        .all()
        and rows["converged"]
        .map(lambda value: str(value).strip().lower() == "true")
        .all()
    )
    shared = (
        "config",
        "data_sha256",
        "environment_lock_sha256",
        "source_commit",
        "n",
        "n_intervention",
        "n_control",
        "subject_order_sha256",
        "treatment_order_sha256",
        "sampling_draws",
        "sampling_tune",
        "sampling_chains",
        "sampling_target_accept",
        "sampling_random_seed",
        "ci_prob",
    )
    status["paired"] = all(rows[column].astype(str).nunique() == 1 for column in shared)
    if not status["complete"] or not status["paired"]:
        status["reason"] = "summary is incomplete or the two fits are not paired"
        return status
    numeric_columns = (
        "n",
        "n_intervention",
        "n_control",
        "sampling_draws",
        "sampling_tune",
        "sampling_chains",
        "sampling_target_accept",
        "sampling_random_seed",
        "ci_prob",
        "effect_items_median",
        "effect_items_lo",
        "effect_items_hi",
        "prob_effect_positive",
        "prob_meaningful_benefit",
        "prob_practically_negligible",
        "prior_effect_items_median",
        "prior_effect_items_lo",
        "prior_effect_items_hi",
        "max_rhat",
        "min_ess",
        "min_bfmi",
        "n_divergences",
        "loo_elpd",
        "loo_p",
        "pareto_k_max",
        "good_k_threshold",
        "guessing_floor_minus_logit_elpd",
        "guessing_floor_minus_logit_elpd_se",
    )
    numeric = rows[list(numeric_columns)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        status["reason"] = "summary contains missing or non-finite release quantities"
        return status
    probability_columns = (
        "prob_effect_positive",
        "prob_meaningful_benefit",
        "prob_practically_negligible",
    )
    if not (
        numeric[list(probability_columns)].ge(0.0).all().all()
        and numeric[list(probability_columns)].le(1.0).all().all()
        and (numeric["effect_items_lo"] <= numeric["effect_items_median"]).all()
        and (numeric["effect_items_median"] <= numeric["effect_items_hi"]).all()
        and (numeric["prior_effect_items_lo"] <= numeric["prior_effect_items_median"])
        .all()
        and (numeric["prior_effect_items_median"] <= numeric["prior_effect_items_hi"])
        .all()
        and (numeric["n_intervention"] + numeric["n_control"])
        .eq(numeric["n"])
        .all()
        and numeric["n_intervention"].gt(0).all()
        and numeric["n_control"].gt(0).all()
        and numeric["sampling_draws"].gt(0).all()
        and numeric["sampling_tune"].gt(0).all()
        and numeric["sampling_chains"].gt(0).all()
        and numeric["sampling_random_seed"].ge(0).all()
        and numeric["sampling_target_accept"].gt(0.0).all()
        and numeric["sampling_target_accept"].le(1.0).all()
        and numeric["ci_prob"].gt(0.0).all()
        and numeric["ci_prob"].lt(1.0).all()
        and numeric["max_rhat"].le(1.01).all()
        and numeric["min_ess"].ge(400.0).all()
        and numeric["min_bfmi"].ge(0.3).all()
        and numeric["n_divergences"].eq(0).all()
        and numeric["good_k_threshold"].gt(0.0).all()
        and numeric["guessing_floor_minus_logit_elpd_se"].ge(0.0).all()
    ):
        status["reason"] = "summary release quantities are internally incoherent"
        return status
    integer_columns = (
        "n",
        "n_intervention",
        "n_control",
        "sampling_draws",
        "sampling_tune",
        "sampling_chains",
        "sampling_random_seed",
        "n_divergences",
    )
    if not np.equal(
        numeric[list(integer_columns)].to_numpy(dtype=float),
        np.floor(numeric[list(integer_columns)].to_numpy(dtype=float)),
    ).all():
        status["reason"] = "summary count and sampling fields must be integers"
        return status
    loo_flags = rows["loo_reliable"].map(
        lambda value: str(value).strip().lower()
    )
    if not loo_flags.isin({"true", "false"}).all() or not np.array_equal(
        loo_flags.eq("true").to_numpy(dtype=bool),
        (numeric["pareto_k_max"] <= numeric["good_k_threshold"]).to_numpy(
            dtype=bool
        ),
    ):
        status["reason"] = "summary LOO reliability flags are inconsistent"
        return status
    hash_columns = (
        "data_sha256",
        "environment_lock_sha256",
        "config_sha256",
        "trace_sha256",
        "row_map_sha256",
        "subject_order_sha256",
        "treatment_order_sha256",
    )
    if not all(
        rows[column]
        .astype(str)
        .str.fullmatch(r"[0-9a-f]{64}", case=False)
        .all()
        for column in hash_columns
    ) or not rows["source_commit"].astype(str).str.fullmatch(
        r"[0-9a-f]{40}", case=False
    ).all():
        status["reason"] = "summary contains malformed provenance hashes"
        return status
    for row in rows.to_dict(orient="records"):
        expected_trace = f"{row['model_id']}-{str(row['trace_sha256'])[:16]}.nc"
        expected_row_map = (
            f"{row['model_id']}-rows-{str(row['row_map_sha256'])[:16]}.csv"
        )
        if str(row["trace_file"]) != expected_trace or str(
            row["row_map_file"]
        ) != expected_row_map:
            status["reason"] = "evidence filenames are not content-addressed"
            return status
    if rows["trace_sha256"].astype(str).nunique() != 2:
        status["reason"] = "paired link fits cannot share identical trace bytes"
        return status
    try:
        scientific_artifact_hashes = {
            str(row["model_id"]): _decode_scientific_artifact_hashes(
                row[_SCIENTIFIC_ARTIFACT_HASH_COLUMN]
            )
            for row in rows.to_dict(orient="records")
        }
    except ValueError as exc:
        status["reason"] = str(exc)
        return status
    status["scientific_artifacts_bound"] = True
    indexed_numeric = numeric.set_axis(rows["model_id"].astype(str)).copy()
    scalar_delta_elpd = float(
        indexed_numeric.loc[BLENDING_COMPANION_MODEL_ID, "loo_elpd"]
        - indexed_numeric.loc[BLENDING_PRIMARY_MODEL_ID, "loo_elpd"]
    )
    if not all(
        _values_match(value, scalar_delta_elpd)
        for value in numeric["guessing_floor_minus_logit_elpd"]
    ):
        status["reason"] = "summary LOO difference does not match its two fit rows"
        return status
    try:
        trace_results: dict[str, Mapping[str, np.ndarray]] = {}
        row_map_results: dict[str, Mapping[str, np.ndarray]] = {}
        for row in rows.to_dict(orient="records"):
            result = trace_validator(
                Path(trace_root) / str(row["trace_file"]), row
            )
            if result is not None:
                trace_results[str(row["model_id"])] = result
            row_map_result = row_map_validator(
                Path(trace_root) / str(row["row_map_file"]), row
            )
            if row_map_result is not None:
                row_map_results[str(row["model_id"])] = row_map_result
        status["traces_validated"] = True
        status["row_maps_validated"] = True
        if len(trace_results) == 2 and len(row_map_results) == 2:
            for model_id in expected:
                if not np.allclose(
                    np.asarray(trace_results[model_id]["pareto_k"], dtype=float),
                    np.asarray(row_map_results[model_id]["pareto_k"], dtype=float),
                    rtol=1e-9,
                    atol=1e-11,
                ):
                    raise ValueError(
                        f"{model_id} archived row map does not match its trace"
                    )
        if len(trace_results) == 2:
            primary_loo = np.asarray(
                trace_results[BLENDING_PRIMARY_MODEL_ID]["loo_i"], dtype=float
            )
            companion_loo = np.asarray(
                trace_results[BLENDING_COMPANION_MODEL_ID]["loo_i"], dtype=float
            )
            if primary_loo.shape != companion_loo.shape:
                raise ValueError("paired pointwise LOO arrays have different shapes")
            loo_difference = companion_loo - primary_loo
            delta_elpd = float(np.sum(loo_difference))
            delta_se = float(
                np.sqrt(len(loo_difference) * np.var(loo_difference, ddof=1))
            )
            for column, value in (
                ("guessing_floor_minus_logit_elpd", delta_elpd),
                ("guessing_floor_minus_logit_elpd_se", delta_se),
            ):
                if not all(_values_match(recorded, value) for recorded in rows[column]):
                    raise ValueError(f"paired traces do not reproduce {column}")
        if primary_model_dirs is not None:
            for row in rows.to_dict(orient="records"):
                model_id = str(row["model_id"])
                directory = Path(primary_model_dirs[model_id])
                if sha256_file(directory / "config.json") != str(row["config_sha256"]):
                    raise ValueError(f"{model_id} config has changed")
                if sha256_file(directory / "trace.nc") != str(row["trace_sha256"]):
                    raise ValueError(f"{model_id} trace has changed")
                if sha256_file(directory / "pareto_k.csv") != str(
                    row["row_map_sha256"]
                ):
                    raise ValueError(f"{model_id} row map has changed")
                for name, expected_sha256 in scientific_artifact_hashes[
                    model_id
                ].items():
                    path = directory / name
                    if not path.is_file():
                        raise ValueError(
                            f"{model_id} scientific report artefact is missing: {name}"
                        )
                    if sha256_file(path) != expected_sha256:
                        raise ValueError(
                            f"{model_id} scientific report artefact has changed: {name}"
                        )
            status["scientific_artifacts_current"] = True
            status["primary_fits_current"] = True
    except (OSError, KeyError, TypeError, ValueError) as exc:
        status["reason"] = str(exc)
        return status
    status["archive_ready"] = bool(
        status["complete"]
        and status["paired"]
        and status["traces_validated"]
        and status["row_maps_validated"]
        and status["scientific_artifacts_bound"]
    )
    status["release_ready"] = bool(
        status["archive_ready"]
        and status["primary_fits_current"]
        and status["scientific_artifacts_current"]
    )
    status["ready"] = (
        status["release_ready"]
        if primary_model_dirs is not None
        else status["archive_ready"]
    )
    status["ready_scope"] = (
        "release" if primary_model_dirs is not None else "archive"
    )
    return status


def evaluate_local_blending_link_sensitivity(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    archive_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Cheap freshness check for a report-local installed sensitivity summary.

    The local report does not reopen two large traces, but it hashes both current
    fit directories' config, trace, row-map and rendered scientific-artefact bytes,
    applies the complete pair schema, and **byte-binds the installed summary to the
    central archive manifest** — the copy written only after the full
    trace-recomputing validation passed at build time — so an edited installed CSV
    cannot quote numbers the build never validated (2026-08-20 ITT review,
    finding 1). ``archive_dir`` names the central archive directory and defaults
    to ``<statistical-model-root>/blending_link_sensitivity``, discovered two
    levels above the fit directory; pass it explicitly for non-standard layouts.

    ``output_dir`` is resolved before any sibling lookup: the report partials call
    this with the relative ``Path(".")``, whose ``.parent`` is itself, which made
    the paired-fit lookup miss and wrongly withhold a valid pair at render time
    (2026-08-20 ITT review — the live bug behind the empty rendered 008/108
    results sections).
    """

    directory = Path(output_dir).resolve()
    config_path = directory / "config.json"
    try:
        current = dict(config) if config is not None else _read_json(
            config_path, label="fit config"
        )
        model_id = str(current.get("model_id"))
        if model_id not in dict(BLENDING_LINK_MODELS):
            return {"required": False, "ready": True, "reason": "not a B-link fit"}
        summary_path = directory / BLENDING_SENSITIVITY_FILENAME
        if not summary_path.is_file():
            return {
                "required": True,
                "ready": False,
                "reason": "mandatory trace-backed B link sensitivity is missing",
            }
        archive_manifest = (
            Path(archive_dir).resolve()
            if archive_dir is not None
            else directory.parent.parent / "blending_link_sensitivity"
        ) / BLENDING_SENSITIVITY_FILENAME
        if not archive_manifest.is_file():
            raise ValueError(
                "central B link archive manifest is missing "
                f"({archive_manifest}); rebuild it with "
                "scripts/blending_link_sensitivity.py"
            )
        if sha256_file(summary_path) != sha256_file(archive_manifest):
            raise ValueError(
                "installed B link summary does not match the validated central "
                "archive manifest, so its numbers are not the build-validated ones"
            )
        summary = pd.read_csv(summary_path)
        config_names = summary.get("config")
        if config_names is None or config_names.astype(str).nunique() != 1:
            raise ValueError("installed B link summary has no single fit configuration")
        config_name = str(config_names.iloc[0])
        model_dirs = {
            paired_model_id: (
                directory
                if paired_model_id == model_id
                else directory.parent / f"{paired_model_id}-{config_name}"
            )
            for paired_model_id, _link in BLENDING_LINK_MODELS
        }
        status = evaluate_blending_link_sensitivity(
            summary,
            trace_root=directory,
            primary_model_dirs=model_dirs,
            trace_validator=lambda _path, _row: None,
            row_map_validator=lambda _path, _row: None,
        )
        if not status.get("release_ready"):
            raise ValueError(str(status.get("reason") or "installed B link pair is stale"))
    except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        return {"required": True, "ready": False, "reason": str(exc)}
    return {
        "required": True,
        "ready": True,
        "reason": "",
        "summary": summary,
        "summary_sha256": sha256_file(summary_path),
    }



# --- The level family's registered link pair (#584 decision 2) ----------------
#
# The ITT machinery above is a *public-release* apparatus: it recomputes both
# estimands from their traces, content-addresses a two-trace bundle and byte-binds
# every consumer to a central archive manifest. The level family's pair is held to
# the same **policy** -- neither link releases without the other -- through a
# deliberately smaller, local check: both fits present under the same config, both
# the registered pair under opposite links, both clean on their own stored gates,
# and fitted on identical rows. It reads stored artefacts rather than reopening
# traces, so it is a pair-readiness check, not the archive-grade validation, and it
# says so in what it returns. Promoting the level pair to the archive path is
# tracked separately; until then a level B fit publishes only beside its twin.

def _level_pair_card(directory: Path, model_id: str) -> dict[str, Any]:
    """One side of the pair: its link, its card and its gate verdict."""
    config = _read_json(directory / "config.json", label=f"{model_id} config")
    if str(config.get("model_id")) != model_id:
        raise ValueError(
            f"{directory.name} holds {config.get('model_id')!r}, not {model_id}"
        )
    if str(config.get("kind")) != "level_factors":
        raise ValueError(f"{model_id} is not a level-factor fit")
    if str(config.get("outcome_symbol")) != "B":
        raise ValueError(f"{model_id} is not a phoneme-blending fit")
    gate = _read_json(
        directory / "diagnostics_summary.json", label=f"{model_id} diagnostics"
    )
    if not _report.convergence_gate_clean_passed(gate):
        raise ValueError(f"{model_id} did not pass its saved clean convergence gate")
    rope = _one_csv_row(directory / "rope_summary.csv", label=f"{model_id} ROPE summary")
    plan = config.get("resolved_run_plan") or {}
    identity = config.get("fitted_data_identity") or {}
    return {
        "model_id": model_id,
        "score_mean_link": str(plan.get("score_mean_link", "logit")),
        "config_name": str(config.get("config_name", "")),
        "data_sha256": str(config.get("data_sha256", "")),
        "fitted_rows_digest": str(identity.get("digest", "")),
        "n_obs": config.get("n_obs"),
        "items_median": _finite_float(
            rope.get("items_median"), label=f"{model_id} items_median"
        ),
        "items_lo": _finite_float(rope.get("items_lo"), label=f"{model_id} items_lo"),
        "items_hi": _finite_float(rope.get("items_hi"), label=f"{model_id} items_hi"),
        "pd": _finite_float(rope.get("pd"), label=f"{model_id} pd"),
    }


def evaluate_level_blending_link_pair(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Is this level-factor B fit releasable beside its opposite-link twin?

    Returns ``{"required", "ready", "reason", "cards"}``. ``required`` is read from
    the fit's own resolved plan **and** from the registered pair, so a B level fit
    outside the pair fails closed rather than publishing unpaired, exactly as the
    ITT gate treats an unregistered B ITT fit.

    The two cards must come from the same data (``data_sha256``), the same fitted
    rows (``fitted_data_identity.digest``) and the same sampling configuration; both
    must have passed their own stored convergence gate; and the two must genuinely
    differ in link. Anything else is not a pair, and a reader shown one card without
    the other would take a below-chance-permitting estimate for the whole answer.
    """

    from language_reading_predictors.statistical_models.level_factors import (
        LEVEL_BLENDING_COMPANION_MODEL_ID,
        LEVEL_BLENDING_PRIMARY_MODEL_ID,
    )

    registered = (LEVEL_BLENDING_PRIMARY_MODEL_ID, LEVEL_BLENDING_COMPANION_MODEL_ID)
    directory = Path(output_dir).resolve()
    try:
        current = (
            dict(config)
            if config is not None
            else _read_json(directory / "config.json", label="fit config")
        )
        if str(current.get("kind") or "") != "level_factors":
            return {"required": False, "ready": True, "reason": "not a level fit"}
        plan = current.get("resolved_run_plan") or {}
        model_id = str(current.get("model_id") or "")
        required = bool(plan.get("link_sensitivity_required_for_release")) or (
            model_id in registered
        )
        if not required:
            return {"required": False, "ready": True, "reason": "no link pairing"}
        if model_id not in registered:
            return {
                "required": True,
                "ready": False,
                "reason": (
                    f"{model_id} declares a mandatory response-link pairing but is "
                    "not one of the registered level blending fits "
                    f"({registered[0]} + {registered[1]}); register the pair before "
                    "releasing"
                ),
            }
        companion_id = next(m for m in registered if m != model_id)
        config_name = str(current.get("config_name") or "")
        if not config_name:
            raise ValueError("fit config lacks config_name")
        companion_dir = directory.parent / f"{companion_id}-{config_name}"
        if not (companion_dir / "config.json").is_file():
            raise ValueError(
                f"the paired {companion_id} fit is not present beside this one "
                f"({companion_dir}); fit the pair before releasing either side"
            )
        cards = {
            model_id: _level_pair_card(directory, model_id),
            companion_id: _level_pair_card(companion_dir, companion_id),
        }
        links = {card["score_mean_link"] for card in cards.values()}
        if links != {"logit", "three_choice_guessing_floor"}:
            raise ValueError(
                "the two fits do not carry opposite score-mean links "
                f"({sorted(links)}), so they are not a link-sensitivity pair"
            )
        for field, label in (
            ("data_sha256", "dataset"),
            ("fitted_rows_digest", "fitted rows"),
            ("config_name", "sampling configuration"),
            ("n_obs", "row count"),
        ):
            values = {card[field] for card in cards.values()}
            if len(values) != 1:
                raise ValueError(
                    f"the paired fits do not share a {label} "
                    f"({sorted(map(str, values))})"
                )
            recorded = next(iter(values))
            if recorded is None or not str(recorded).strip():
                # Agreeing on "unrecorded" is not agreement: an unstamped digest
                # would let two different row sets pass as one pair.
                raise ValueError(
                    f"the paired fits do not record a {label}, so the pairing "
                    "cannot be verified"
                )
    except (OSError, KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        return {"required": True, "ready": False, "reason": str(exc)}
    return {"required": True, "ready": True, "reason": "", "cards": cards}


__all__ = [
    "BLENDING_COMPANION_MODEL_ID",
    "BLENDING_LINK_MODELS",
    "BLENDING_PRIMARY_MODEL_ID",
    "BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS",
    "BLENDING_SENSITIVITY_FILENAME",
    "build_blending_link_sensitivity",
    "evaluate_blending_link_sensitivity",
    "evaluate_level_blending_link_pair",
    "evaluate_local_blending_link_sensitivity",
]
