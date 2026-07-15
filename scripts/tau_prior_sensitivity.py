# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prior-sensitivity and unadjusted benchmarks for the ITT suite (#141, #341).

Refits representative single-outcome ITT models across a grid of tau prior SDs and
reports whether the headline conclusion is stable — posterior direction
(``pd = P(AME > 0)``), the logit and items-scale effect, and the interval width —
rather than a binary significant/not read.

The two-tier proposal keeps the wider ``Normal(0, 0.5)`` for proximal outcomes and
tightens the broad standardised-transfer (distal) outcomes to ``Normal(0, 0.3)``.
This sweep audits that post-data regularisation choice: the distal outcomes should be stable
across defensible SDs and proximal L/W should keep their direction. The default
now covers **every distal member** — the clearly-null R/E and the *borderline*
UR/UE/T/F — so the certifying sweep is not limited to the null outcomes (issue
#267): the printed table lets a reviewer check whether any evidence-ladder
boundary moves between SD 0.3 and 0.5 for the borderline members. It separately
compares the own-baseline precision prior ``Normal(1, 0.25)`` with
``Normal(1, 0.5)`` for the headline L/W outcomes and fits an unadjusted
randomised-arm benchmark (no baseline or age precision terms). It also varies
the Beta-Binomial concentration prior across ``HalfNormal(25)``, ``HalfNormal(50)``,
``HalfNormal(100)`` and ``HalfNormal(200)`` for L/W, spanning stronger
overdispersion through a much more permissive near-Binomial region.

The floored P/N outcomes use a separate, estimand-matched release grid rather
than the graded-outcome sweep: the observed-baseline-floor subset, a Bernoulli
off-floor likelihood, tau SDs 0.5/1.0/1.5, and linear age adjustment on/off.
Every floor-grid trace is persisted and its risk-difference summary is copied
beside an existing matching model report.  Those report-local artefacts are the
release gate when power-scaling diagnoses a possible prior-data conflict.

Usage::

    python scripts/tau_prior_sensitivity.py                 # dev config (fast)
    python scripts/tau_prior_sensitivity.py --config test   # more draws
    python scripts/tau_prior_sensitivity.py --config reporting --outcomes P N \
      --out-dir output/statistical_models/floor_tau_prior_sensitivity

The default outcome set is the 44-cell standard R/E/UR/UE/T/F/L/W sweep. Only
its exact reporting run can replace
``output/statistical_models/tau_prior_sensitivity/tau_prior_sensitivity.csv``.
P/N are an explicit separate run and default to
``output/statistical_models/floor_tau_prior_sensitivity/``, so they cannot mix
with or overwrite the standard archive. Every attempt retains an immutable
content-addressed manifest. Before running either sweep, fit the corresponding
primary ITT models with the same sampling configuration; the sensitivity run
is bound to those models' current config, trace, data, analysis-set counts, and
sampling metadata and fails closed when a primary artefact is absent or stale.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

import dse_research_utils.statistics.models.sampling as _sampling
from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.factories import build_itt_model
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.sensitivity import (
    FLOOR_SENSITIVITY_AGE_ADJUSTMENTS,
    FLOOR_SENSITIVITY_AXIS,
    FLOOR_SENSITIVITY_FILENAME,
    FLOOR_SENSITIVITY_MODEL_IDS,
    FLOOR_SENSITIVITY_PROVENANCE_ATTR,
    FLOOR_SENSITIVITY_SAMPLING_ATTR,
    FLOOR_SENSITIVITY_TAU_SIGMAS,
    PrimaryFloorReference,
    PrimaryStandardReference,
    STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS,
    STANDARD_SENSITIVITY_FILENAME,
    STANDARD_SENSITIVITY_GAMMA_OWN_SIGMAS,
    STANDARD_SENSITIVITY_KAPPA_SIGMAS,
    STANDARD_SENSITIVITY_MODEL_IDS,
    STANDARD_SENSITIVITY_OUTCOMES,
    STANDARD_SENSITIVITY_PROVENANCE_ATTR,
    STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS,
    STANDARD_SENSITIVITY_SAMPLING_ATTR,
    evaluate_floor_sensitivity,
    evaluate_standard_sensitivity,
    floor_trace_provenance,
    load_primary_floor_reference,
    load_primary_standard_reference,
    sha256_file,
    standard_trace_provenance,
)

# The default sweep grid: distal outcomes get the sub-0.5 SDs the tier proposes,
# proximal outcomes bracket the 0.5 default. Both include the *other* tier's
# anchor so the comparison is symmetric.
DISTAL_SIGMAS = STANDARD_SENSITIVITY_DISTAL_TAU_SIGMAS
PROXIMAL_SIGMAS = STANDARD_SENSITIVITY_PROXIMAL_TAU_SIGMAS
# Every distal member (clearly-null R/E and borderline UR/UE/T/F) plus the L/W
# proximal anchors — so the certifying sweep is not limited to the null outcomes
# (issue #267).
DEFAULT_OUTCOMES = STANDARD_SENSITIVITY_OUTCOMES
BASELINE_SENSITIVITY_OUTCOMES = ("L", "W")
BASELINE_SIGMAS = STANDARD_SENSITIVITY_GAMMA_OWN_SIGMAS
CONCENTRATION_SENSITIVITY_OUTCOMES = ("L", "W")
KAPPA_SIGMAS = STANDARD_SENSITIVITY_KAPPA_SIGMAS


def _default_output_name(outcomes: list[str] | tuple[str, ...]) -> str:
    """Keep floor-only manifests separate from the standard sensitivity archive."""
    floor_only = bool(outcomes) and set(outcomes).issubset(FLOOR_SENSITIVITY_MODEL_IDS)
    return "floor_tau_prior_sensitivity" if floor_only else "tau_prior_sensitivity"


def _validate_requested_outcome_mode(outcomes: list[str] | tuple[str, ...]) -> None:
    """Reject a standard/floor mixture before any data or artefact work starts."""
    requested = set(str(outcome) for outcome in outcomes)
    standard = requested.intersection(STANDARD_SENSITIVITY_OUTCOMES)
    floor = requested.intersection(FLOOR_SENSITIVITY_MODEL_IDS)
    if standard and floor:
        raise ValueError(
            "standard graded-outcome and floor-rule sensitivity outcomes cannot "
            "be mixed in one run; run R/E/UR/UE/T/F/L/W and P/N separately"
        )


def _assert_primary_sampling_contract(
    sampling,
    reference: PrimaryFloorReference | PrimaryStandardReference,
    *,
    config: str,
) -> None:
    """Fail before fitting if the selected preset differs from the primary fit."""
    matched = {
        "draws": sampling.draws,
        "tune": sampling.tune,
        "chains": sampling.chains,
        "target_accept": sampling.target_accept,
    }
    if reference.config_name != config or any(
        not np.isclose(
            float(observed),
            float(reference.sampling[key]),
            rtol=0.0,
            atol=1e-12,
        )
        for key, observed in matched.items()
    ):
        raise RuntimeError(
            f"{reference.outcome} sensitivity sampling does not match its "
            f"current {config} primary fit"
        )


def _sigmas_for(symbol: str) -> tuple[float, ...]:
    from language_reading_predictors.statistical_models.measures import is_distal

    return DISTAL_SIGMAS if is_distal(symbol) else PROXIMAL_SIGMAS


def _adopted_tau_sigma(symbol: str) -> float:
    from language_reading_predictors.statistical_models.measures import is_distal

    return 0.3 if is_distal(symbol) else 0.5


def _floor_trace_file(
    symbol: str,
    tau_sigma: float,
    age_adjusted: bool,
    config: str,
) -> Path:
    """Return the sensitivity-root-relative trace path for one floor-grid cell."""
    sigma_token = f"{tau_sigma:g}".replace(".", "p")
    age_token = "on" if age_adjusted else "off"
    model_id = FLOOR_SENSITIVITY_MODEL_IDS[symbol]
    return (
        Path("traces")
        / f"{model_id}-{config}"
        / f"trace_floor_tau-sd-{sigma_token}_age-{age_token}.nc"
    )


def _standard_trace_file(row: dict, config: str) -> Path:
    """Return a unique semantic path before adding the content digest."""

    def token(value) -> str:
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            return "none"
        return f"{float(value):g}".replace(".", "p")

    precision = "on" if bool(row["use_precision_terms"]) else "off"
    filename = (
        f"trace_{row['outcome']}_{row['sensitivity_axis']}"
        f"_tau-{token(row['tau_sigma'])}"
        f"_gamma-own-{token(row['gamma_own_sigma'])}"
        f"_kappa-{token(row['kappa_sigma'])}"
        f"_precision-{precision}.nc"
    )
    return Path("traces") / f"standard-{config}" / filename


def _persist_content_addressed_trace(
    trace,
    *,
    sensitivity_dir: Path,
    semantic_file: Path,
) -> tuple[Path, str]:
    """Atomically install an immutable trace whose filename carries its digest."""
    trace_dir = sensitivity_dir / semantic_file.parent
    trace_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{semantic_file.stem}-",
        suffix=".nc",
        dir=trace_dir,
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
                    "floor sensitivity trace digest-prefix collision: "
                    f"{destination}"
                )
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        return destination.relative_to(sensitivity_dir), digest
    finally:
        temporary.unlink(missing_ok=True)


def _write_content_addressed_csv(frame: pd.DataFrame, directory: Path) -> Path:
    """Persist every attempted sweep manifest without replacing the last good one."""
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".tau-prior-sensitivity-",
        suffix=".csv",
        dir=directory,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        digest = sha256_file(temporary)
        destination = directory / f"tau_prior_sensitivity-{digest[:12]}.csv"
        if destination.exists():
            if sha256_file(destination) != digest:
                raise RuntimeError(
                    "tau-prior manifest digest-prefix collision: "
                    f"{destination}"
                )
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _read_bound_run_manifest(frame: pd.DataFrame, run_csv: Path) -> pd.DataFrame:
    """Read and bind the exact immutable CSV that publication would expose."""
    source = Path(run_csv)
    if not source.is_file():
        raise RuntimeError(f"sensitivity run manifest does not exist: {source}")
    digest = sha256_file(source)
    if not source.stem.endswith(f"-{digest[:12]}"):
        raise RuntimeError(
            "sensitivity run manifest is not content-addressed by its current digest"
        )
    try:
        recorded = pd.read_csv(source)
        expected = pd.read_csv(io.StringIO(frame.to_csv(index=False)))
        pd.testing.assert_frame_equal(
            recorded,
            expected,
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except Exception as exc:  # noqa: BLE001 - malformed release input fails closed
        raise RuntimeError(
            "sensitivity run manifest does not exactly represent the validated frame"
        ) from exc
    return recorded


def _publish_csv_atomically(source: Path, destination: Path) -> None:
    """Replace a fixed manifest only after its complete bundle has validated."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != sha256_file(source):
            raise RuntimeError(f"manifest changed during atomic publication: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_validated_sensitivity_manifest(
    frame: pd.DataFrame,
    *,
    run_csv: Path,
    sensitivity_dir: Path,
    config: str,
    requested_outcomes: list[str] | tuple[str, ...],
    primary_references: dict[str, PrimaryStandardReference] | None = None,
    primary_floor_references: dict[str, PrimaryFloorReference] | None = None,
) -> Path | None:
    """Publish only an exact, trace-backed reporting sensitivity sweep."""
    requested = tuple(str(outcome) for outcome in requested_outcomes)
    fixed = sensitivity_dir / STANDARD_SENSITIVITY_FILENAME
    floor_only = bool(requested) and set(requested).issubset(FLOOR_SENSITIVITY_MODEL_IDS)
    if floor_only:
        exact_floor = bool(
            len(requested) == len(FLOOR_SENSITIVITY_MODEL_IDS)
            and set(requested) == set(FLOOR_SENSITIVITY_MODEL_IDS)
        )
        if not exact_floor or config != "reporting":
            return None
        manifest = _read_bound_run_manifest(frame, run_csv)
        references = primary_floor_references or {}
        observed_outcomes = (
            set(manifest["outcome"].astype(str))
            if "outcome" in manifest.columns
            else set()
        )
        reporting_identity = bool(
            "config" in manifest.columns
            and manifest["config"].astype(str).eq(config).all()
            and all(reference.config_name == config for reference in references.values())
        )
        statuses = {
            symbol: evaluate_floor_sensitivity(
                manifest,
                symbol,
                primary_reference=references.get(symbol),
                trace_root=sensitivity_dir,
                require_hash_suffix=True,
            )
            for symbol in FLOOR_SENSITIVITY_MODEL_IDS
        }
        if (
            observed_outcomes != set(FLOOR_SENSITIVITY_MODEL_IDS)
            or set(references) != set(FLOOR_SENSITIVITY_MODEL_IDS)
            or not reporting_identity
            or not all(status["ready"] for status in statuses.values())
        ):
            raise RuntimeError(
                "refusing to publish the fixed floor sensitivity manifest: "
                f"P/N trace-backed validation failed ({statuses})"
            )
        _publish_csv_atomically(run_csv, fixed)
        return fixed

    exact_standard = bool(
        len(requested) == len(STANDARD_SENSITIVITY_OUTCOMES)
        and set(requested) == set(STANDARD_SENSITIVITY_OUTCOMES)
    )
    if not exact_standard or config != "reporting":
        return None
    manifest = _read_bound_run_manifest(frame, run_csv)
    status = evaluate_standard_sensitivity(
        manifest,
        config_name=config,
        requested_outcomes=requested,
        primary_references=primary_references,
        trace_root=sensitivity_dir,
    )
    if not status["ready"]:
        raise RuntimeError(
            "refusing to publish the fixed standard sensitivity manifest: "
            f"44-cell trace-backed validation failed ({status})"
        )
    _publish_csv_atomically(run_csv, fixed)
    return fixed


def _build_floor_sensitivity_model(
    prepared,
    symbol: str,
    *,
    tau_sigma: float,
    age_adjusted: bool,
):
    """Build one estimand-matched P/N sensitivity model without sampling it."""
    from language_reading_predictors.statistical_models.preprocessing import (
        restrict_to_baseline_floored,
    )

    at_risk = restrict_to_baseline_floored(prepared, symbol)
    return build_itt_model(
        at_risk,
        outcome_symbol=symbol,
        cross_symbols=(),
        use_age_linear=age_adjusted,
        use_own_baseline=False,
        likelihood="bernoulli_offfloor",
        tau_sigma=tau_sigma,
    )


def _fit_floor_one(
    prepared,
    symbol: str,
    tau_sigma: float,
    age_adjusted: bool,
    sampling,
    config: str,
    *,
    sensitivity_dir: Path,
    primary_reference: PrimaryFloorReference,
) -> dict:
    """Fit and persist one cell of the floored-outcome release grid."""
    from language_reading_predictors.statistical_models import diagnostics as _diag
    from language_reading_predictors.statistical_models.reporting import (
        rope_summary,
        tau_summary_offfloor,
    )

    _assert_primary_sampling_contract(
        sampling,
        primary_reference,
        config=config,
    )
    built = _build_floor_sensitivity_model(
        prepared,
        symbol,
        tau_sigma=tau_sigma,
        age_adjusted=age_adjusted,
    )
    with built.model:
        trace = pm.sample(
            draws=sampling.draws,
            tune=sampling.tune,
            chains=sampling.chains,
            cores=sampling.cores,
            target_accept=sampling.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=sampling.random_seed,
            progressbar=False,
        )

    free_names = [rv.name for rv in built.model.free_RVs]
    convergence = _diag.subfit_convergence(
        trace,
        label=f"{symbol} floor tau={tau_sigma:g} age={'on' if age_adjusted else 'off'}",
        var_names=free_names,
    )
    summary = tau_summary_offfloor(
        trace,
        ci_prob=0.95,
        G=built.prepared.G,
    )
    magnitude = rope_summary(
        trace,
        G=built.prepared.G,
        n_trials=1,
        delta=0.10,
        ci_prob=0.95,
        varying_term="",
    )

    n = int(built.prepared.n_obs)
    n_intervention = int(np.sum(built.prepared.G == 1))
    n_control = int(np.sum(built.prepared.G == 0))
    data_sha256 = str(getattr(built.prepared, "data_sha256", ""))
    if (
        data_sha256 != primary_reference.data_sha256
        or n != primary_reference.n
        or n_intervention != primary_reference.n_intervention
        or n_control != primary_reference.n_control
    ):
        raise RuntimeError(
            f"{symbol} floor sensitivity no longer matches its primary analysis "
            "data or arm counts"
        )

    row = {
        **primary_reference.manifest_values(),
        "estimand": "off_floor_risk_difference_given_observed_baseline_floor",
        "analysis_subset": "observed_baseline_floor",
        "likelihood": "bernoulli_offfloor",
        "sensitivity_axis": FLOOR_SENSITIVITY_AXIS,
        "tau_sigma": tau_sigma,
        "age_adjusted": age_adjusted,
        "use_age_linear": age_adjusted,
        "use_own_baseline": False,
        "sampling_draws": sampling.draws,
        "sampling_tune": sampling.tune,
        "sampling_chains": sampling.chains,
        "sampling_cores": sampling.cores,
        "sampling_target_accept": sampling.target_accept,
        "sampling_random_seed": sampling.random_seed,
        "sampling_nuts_sampler": "nutpie",
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
        "meaningful_risk_difference": 0.10,
        "prob_risk_difference_ge_0_10": magnitude["prob_benefit_ge_delta"],
        "tau_logit_median": summary["tau_logit_median"],
        "tau_logit_lo": summary["tau_logit_lo"],
        "tau_logit_hi": summary["tau_logit_hi"],
        "free_variables": "|".join(free_names),
        "n_free_variables": len(free_names),
        "convergence_scope": "all_free_variables",
        "converged": convergence["converged"],
        "max_rhat": convergence["max_rhat"],
        "min_ess": convergence["min_ess"],
        "min_bfmi": convergence["min_bfmi"],
        "n_divergences": convergence["n_divergences"],
    }
    provenance = floor_trace_provenance(row)
    trace.posterior.attrs[FLOOR_SENSITIVITY_SAMPLING_ATTR] = json.dumps(
        provenance["sampling"],
        sort_keys=True,
        separators=(",", ":"),
    )
    trace.posterior.attrs[FLOOR_SENSITIVITY_PROVENANCE_ATTR] = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    )
    trace_file, trace_sha256 = _persist_content_addressed_trace(
        trace,
        sensitivity_dir=sensitivity_dir,
        semantic_file=_floor_trace_file(symbol, tau_sigma, age_adjusted, config),
    )
    row.update(
        trace_file=trace_file.as_posix(),
        trace_sha256=trace_sha256,
    )
    return row


def _copy_floor_model_artifacts(
    sensitivity: pd.DataFrame,
    *,
    sensitivity_dir: Path,
    model_output_root: Path,
    config: str,
) -> list[Path]:
    """Install only complete, primary-aligned P/N bundles beside model reports.

    All central bundles are validated before any report-local file is copied.
    Trace filenames include their content digest and are installed before an
    atomic manifest replacement, so an old manifest can never silently point to
    a partly replaced set of traces.
    """
    if sensitivity.empty or "outcome" not in sensitivity.columns:
        return []

    validated: list[tuple[str, Path, pd.DataFrame, PrimaryFloorReference]] = []
    for symbol, model_id in FLOOR_SENSITIVITY_MODEL_IDS.items():
        rows = sensitivity.loc[sensitivity["outcome"] == symbol].copy()
        if rows.empty:
            continue
        model_dir = model_output_root / f"{model_id}-{config}"
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"cannot install {symbol} floor sensitivity: primary model "
                f"directory does not exist: {model_dir}"
            )
        reference = load_primary_floor_reference(
            model_dir,
            symbol,
            config_name=config,
        )
        status = evaluate_floor_sensitivity(
            rows,
            symbol,
            primary_reference=reference,
            trace_root=sensitivity_dir,
            require_hash_suffix=True,
        )
        if not status["ready"]:
            raise RuntimeError(
                f"refusing to install {symbol} floor sensitivity: central bundle "
                f"failed validation ({status})"
            )
        validated.append((symbol, model_dir, rows, reference))

    plans: list[tuple[Path, Path, pd.DataFrame, list[str]]] = []
    stage_dirs: list[Path] = []
    try:
        # Stage and revalidate every bundle before exposing any report-local file.
        for symbol, model_dir, rows, reference in validated:
            stage_dir = Path(
                tempfile.mkdtemp(
                    prefix=".floor-sensitivity-stage-",
                    dir=model_dir,
                )
            )
            stage_dirs.append(stage_dir)
            installed_names: list[str] = []
            for index, row in rows.iterrows():
                source = sensitivity_dir / str(row["trace_file"])
                trace_sha256 = str(row["trace_sha256"])
                digest_suffix = f"-{trace_sha256[:12]}"
                destination_name = (
                    source.name
                    if source.stem.endswith(digest_suffix)
                    else f"{source.stem}{digest_suffix}.nc"
                )
                staged_trace = stage_dir / destination_name
                shutil.copy2(source, staged_trace)
                if sha256_file(staged_trace) != trace_sha256:
                    raise RuntimeError(
                        f"staged floor sensitivity trace changed during copy: {source}"
                    )
                rows.at[index, "trace_file"] = destination_name
                installed_names.append(destination_name)

            staged_status = evaluate_floor_sensitivity(
                rows,
                symbol,
                primary_reference=reference,
                trace_root=stage_dir,
                require_hash_suffix=True,
            )
            if not staged_status["ready"]:
                raise RuntimeError(
                    f"refusing to install {symbol} floor sensitivity: staged bundle "
                    f"failed validation ({staged_status})"
                )
            staged_csv = stage_dir / FLOOR_SENSITIVITY_FILENAME
            rows.to_csv(staged_csv, index=False)
            plans.append((stage_dir, model_dir, rows, installed_names))

        # Install every immutable trace first. Existing manifests continue to
        # describe their old bundles until the final atomic CSV replacements.
        for stage_dir, model_dir, _rows, installed_names in plans:
            for name in installed_names:
                os.replace(stage_dir / name, model_dir / name)

        written: list[Path] = []
        for stage_dir, model_dir, _rows, _installed_names in plans:
            csv_path = model_dir / FLOOR_SENSITIVITY_FILENAME
            os.replace(stage_dir / FLOOR_SENSITIVITY_FILENAME, csv_path)
            written.append(csv_path)
        return written
    finally:
        for stage_dir in stage_dirs:
            shutil.rmtree(stage_dir, ignore_errors=True)


def _fit_one(
    prepared,
    symbol: str,
    tau_sigma: float,
    sampling,
    config: str,
    *,
    sensitivity_dir: Path,
    primary_reference: PrimaryStandardReference,
    gamma_own_sigma: float | None = 0.25,
    kappa_sigma: float | None = 50.0,
    use_precision_terms: bool = True,
    sensitivity_axis: str = "tau_sigma",
) -> dict:
    from language_reading_predictors.statistical_models import diagnostics as _diag
    from language_reading_predictors.statistical_models.reporting import tau_summary_itt

    _assert_primary_sampling_contract(
        sampling,
        primary_reference,
        config=config,
    )
    built = build_itt_model(
        prepared,
        outcome_symbol=symbol,
        cross_symbols=(),
        use_age_linear=use_precision_terms,
        use_own_baseline=use_precision_terms,
        tau_sigma=tau_sigma,
        gamma_own_sigma=gamma_own_sigma,
        kappa_sigma=kappa_sigma,
    )
    with built.model:
        trace = pm.sample(
            draws=sampling.draws,
            tune=sampling.tune,
            chains=sampling.chains,
            cores=sampling.cores,
            target_accept=sampling.target_accept,
            nuts_sampler="nutpie",
            return_inferencedata=True,
            random_seed=sampling.random_seed,
            progressbar=False,
        )
    s = tau_summary_itt(trace, ci_prob=0.95, G=built.prepared.G)
    n_trials = MEASURES[symbol].n_trials
    tau_draws = trace.posterior["tau"].stack(sample=("chain", "draw")).values
    kappa_draws = (
        trace.posterior["kappa"].stack(sample=("chain", "draw")).values
        if "kappa" in trace.posterior
        else np.array([np.nan])
    )
    free_names = [rv.name for rv in built.model.free_RVs]
    convergence = _diag.subfit_convergence(
        trace,
        label=f"{symbol} {sensitivity_axis}",
        var_names=free_names,
    )
    G = np.asarray(built.prepared.G)
    n = int(built.prepared.n_obs)
    n_intervention = int(np.sum(G == 1))
    n_control = int(np.sum(G == 0))
    data_sha256 = str(getattr(built.prepared, "data_sha256", ""))
    if (
        primary_reference.config_name != config
        or primary_reference.outcome != symbol
        or data_sha256 != primary_reference.data_sha256
        or n != primary_reference.n
        or n_intervention != primary_reference.n_intervention
        or n_control != primary_reference.n_control
    ):
        raise RuntimeError(
            f"{symbol} standard sensitivity does not match its current primary "
            "ITT data, sample, arm counts, or config"
        )
    row = {
        **primary_reference.manifest_values(),
        "config": config,
        "outcome": symbol,
        "n_trials": n_trials,
        "sensitivity_axis": sensitivity_axis,
        "tau_sigma": tau_sigma,
        "gamma_own_sigma": gamma_own_sigma,
        "kappa_sigma": kappa_sigma,
        "use_precision_terms": use_precision_terms,
        "data_sha256": data_sha256,
        "n": n,
        "n_intervention": n_intervention,
        "n_control": n_control,
        "pd": s["prob_tau_pos"],
        "tau_logit_mean": s["tau_logit_mean"],
        "tau_logit_lo": s["tau_logit_lo"],
        "tau_logit_hi": s["tau_logit_hi"],
        "ci_width_logit": s["tau_logit_hi"] - s["tau_logit_lo"],
        "tau_sd_logit": float(np.std(tau_draws)),
        "kappa_median": float(np.nanmedian(kappa_draws)),
        "items_mean": s["tau_prob_mean"] * n_trials,
        "items_lo": s["tau_prob_lo"] * n_trials,
        "items_hi": s["tau_prob_hi"] * n_trials,
        "converged": convergence["converged"],
        "max_rhat": convergence["max_rhat"],
        "min_ess": convergence["min_ess"],
        "min_bfmi": convergence["min_bfmi"],
        "n_divergences": convergence["n_divergences"],
        "free_variables": "|".join(free_names),
        "n_free_variables": len(free_names),
        "convergence_scope": "all_free_variables",
        "sampling_draws": sampling.draws,
        "sampling_tune": sampling.tune,
        "sampling_chains": sampling.chains,
        "sampling_cores": sampling.cores,
        "sampling_target_accept": sampling.target_accept,
        "sampling_random_seed": sampling.random_seed,
        "sampling_nuts_sampler": "nutpie",
    }
    provenance = standard_trace_provenance(row)
    trace.posterior.attrs[STANDARD_SENSITIVITY_SAMPLING_ATTR] = json.dumps(
        provenance["sampling"], sort_keys=True, separators=(",", ":")
    )
    trace.posterior.attrs[STANDARD_SENSITIVITY_PROVENANCE_ATTR] = json.dumps(
        provenance, sort_keys=True, separators=(",", ":")
    )
    trace_file, trace_sha256 = _persist_content_addressed_trace(
        trace,
        sensitivity_dir=sensitivity_dir,
        semantic_file=_standard_trace_file(row, config),
    )
    row.update(trace_file=trace_file.as_posix(), trace_sha256=trace_sha256)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="dev", help="sampling preset (dev/test/reporting)")
    ap.add_argument("--outcomes", nargs="+", default=list(DEFAULT_OUTCOMES))
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output dir (default: <output-root>/statistical_models/"
            "floor_tau_prior_sensitivity for floor-only runs; "
            "tau_prior_sensitivity otherwise)."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root for this run (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR); the relative layout is unchanged. Default: "
            "repo-local output/."
        ),
    )
    args = ap.parse_args()

    try:
        _validate_requested_outcome_mode(args.outcomes)
    except ValueError as exc:
        ap.error(str(exc))

    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")
    args.out_dir = args.out_dir or os.path.join(
        str(_paths.stat_dir()), _default_output_name(args.outcomes)
    )
    sensitivity_dir = Path(args.out_dir)

    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
        restrict_to_baseline_floored,
    )

    sampling = _sampling.get_sampling_configuration(args.config, random_seed=20260701)
    # Prepare each outcome separately, matching its registered single-outcome
    # model. A shared eight-outcome frame would both omit taught outcomes (UR/UE)
    # and impose an unintended cross-outcome complete-case restriction.
    prepared_by_symbol = {
        symbol: load_and_prepare(
            phase_mode="itt",
            outcomes=(symbol,),
            # P/N need the t1 value to determine subgroup eligibility, but a
            # missing value must remain visible rather than being silently
            # dropped by preprocessing.  The explicit restriction below keeps
            # only observed baseline zeros, exactly matching the main fit.
            pre_required=() if symbol in FLOOR_SENSITIVITY_MODEL_IDS else None,
        )
        for symbol in args.outcomes
    }
    loaded_counts = ", ".join(
        f"{symbol}={prepared.n_obs}"
        for symbol, prepared in prepared_by_symbol.items()
    )
    print(
        f"Loaded separate outcome frames ({loaded_counts}); "
        f"config={args.config} (draws={sampling.draws}, tune={sampling.tune}, "
        f"chains={sampling.chains})"
    )

    model_output_root = Path(_paths.stat_dir()) / "models"
    primary_standard_references: dict[str, PrimaryStandardReference] = {}
    for symbol in args.outcomes:
        model_id = STANDARD_SENSITIVITY_MODEL_IDS.get(symbol)
        if model_id is None:
            continue
        primary_standard_references[symbol] = load_primary_standard_reference(
            model_output_root / f"{model_id}-{args.config}",
            symbol,
            config_name=args.config,
        )
    primary_floor_references: dict[str, PrimaryFloorReference] = {}
    for symbol in args.outcomes:
        model_id = FLOOR_SENSITIVITY_MODEL_IDS.get(symbol)
        if model_id is None:
            continue
        model_dir = model_output_root / f"{model_id}-{args.config}"
        reference = load_primary_floor_reference(
            model_dir,
            symbol,
            config_name=args.config,
        )
        at_risk = restrict_to_baseline_floored(prepared_by_symbol[symbol], symbol)
        observed_counts = (
            int(at_risk.n_obs),
            int(np.sum(at_risk.G == 1)),
            int(np.sum(at_risk.G == 0)),
        )
        expected_counts = (
            reference.n,
            reference.n_intervention,
            reference.n_control,
        )
        if (
            str(getattr(at_risk, "data_sha256", "")) != reference.data_sha256
            or observed_counts != expected_counts
        ):
            raise RuntimeError(
                f"{symbol} sensitivity input does not match the current primary "
                "report's data digest or analysis-set counts"
            )
        primary_floor_references[symbol] = reference

    rows = []
    for symbol in args.outcomes:
        if symbol in FLOOR_SENSITIVITY_MODEL_IDS:
            for sigma in FLOOR_SENSITIVITY_TAU_SIGMAS:
                for age_adjusted in FLOOR_SENSITIVITY_AGE_ADJUSTMENTS:
                    print(
                        f"  fitting {symbol} off-floor  tau ~ Normal(0, {sigma:g}); "
                        f"age={'on' if age_adjusted else 'off'} ...",
                        flush=True,
                    )
                    rows.append(
                        _fit_floor_one(
                            prepared_by_symbol[symbol],
                            symbol,
                            sigma,
                            age_adjusted,
                            sampling,
                            args.config,
                            sensitivity_dir=sensitivity_dir,
                            primary_reference=primary_floor_references[symbol],
                        )
                    )
            continue
        for sigma in _sigmas_for(symbol):
            print(f"  fitting {symbol}  tau ~ Normal(0, {sigma}) ...", flush=True)
            rows.append(
                _fit_one(
                    prepared_by_symbol[symbol],
                    symbol,
                    sigma,
                    sampling,
                    args.config,
                    sensitivity_dir=sensitivity_dir,
                    primary_reference=primary_standard_references[symbol],
                    sensitivity_axis="tau_sigma",
                )
            )

    # Own-baseline sensitivity is a separate axis rather than a Cartesian product
    # with every tau value: it answers the prior-audit question without multiplying
    # the already large sweep. Restrict it to the two headline reading anchors.
    for symbol in BASELINE_SENSITIVITY_OUTCOMES:
        if symbol not in args.outcomes:
            continue
        tau_sigma = _adopted_tau_sigma(symbol)
        for baseline_sigma in BASELINE_SIGMAS:
            print(
                f"  fitting {symbol}  gamma_own ~ Normal(1, {baseline_sigma}) ...",
                flush=True,
            )
            rows.append(
                _fit_one(
                    prepared_by_symbol[symbol],
                    symbol,
                    tau_sigma,
                    sampling,
                    args.config,
                    sensitivity_dir=sensitivity_dir,
                    primary_reference=primary_standard_references[symbol],
                    gamma_own_sigma=baseline_sigma,
                    sensitivity_axis="gamma_own_sigma",
                )
            )

        print(f"  fitting {symbol}  unadjusted randomised-arm benchmark ...", flush=True)
        rows.append(
            _fit_one(
                prepared_by_symbol[symbol],
                symbol,
                tau_sigma,
                sampling,
                args.config,
                sensitivity_dir=sensitivity_dir,
                primary_reference=primary_standard_references[symbol],
                gamma_own_sigma=None,
                use_precision_terms=False,
                sensitivity_axis="unadjusted_benchmark",
            )
        )

    # Dispersion-prior sensitivity is kept as its own axis. The larger scales are
    # important for the 79- and 32-item headline tests because the adopted
    # HalfNormal(50) puts little mass in the near-Binomial region for high
    # denominators; a stable treatment contrast should not depend materially on
    # that restriction.
    for symbol in CONCENTRATION_SENSITIVITY_OUTCOMES:
        if symbol not in args.outcomes:
            continue
        tau_sigma = _adopted_tau_sigma(symbol)
        for kappa_sigma in KAPPA_SIGMAS:
            print(
                f"  fitting {symbol}  kappa ~ HalfNormal({kappa_sigma:g}) ...",
                flush=True,
            )
            rows.append(
                _fit_one(
                    prepared_by_symbol[symbol],
                    symbol,
                    tau_sigma,
                    sampling,
                    args.config,
                    sensitivity_dir=sensitivity_dir,
                    primary_reference=primary_standard_references[symbol],
                    kappa_sigma=kappa_sigma,
                    sensitivity_axis="kappa_sigma",
                )
            )

    df = pd.DataFrame(rows)
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    run_csv = _write_content_addressed_csv(df, sensitivity_dir)
    copied = _copy_floor_model_artifacts(
        df,
        sensitivity_dir=sensitivity_dir,
        model_output_root=model_output_root,
        config=args.config,
    )
    published_csv = _publish_validated_sensitivity_manifest(
        df,
        run_csv=run_csv,
        sensitivity_dir=sensitivity_dir,
        config=args.config,
        requested_outcomes=args.outcomes,
        primary_references=primary_standard_references or None,
        primary_floor_references=primary_floor_references or None,
    )

    show = df.copy()
    for c in (
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
        "risk_difference_median",
        "risk_difference_lo",
        "risk_difference_hi",
        "prob_risk_difference_positive",
        "prob_risk_difference_ge_0_10",
    ):
        if c in show.columns:
            show[c] = show[c].round(3)
    print("\n=== tau prior sensitivity ===")
    print(show.to_string(index=False))
    print(f"\nWrote immutable run manifest: {run_csv}")
    if published_csv is not None:
        print(f"Published validated manifest: {published_csv}")
    else:
        print(
            "Did not replace the fixed manifest: certification requires the exact "
            "44-cell standard reporting sweep or the exact P/N reporting sweep."
        )
    for path in copied:
        print(f"Wrote report-local floor sensitivity: {path}")


if __name__ == "__main__":
    main()
