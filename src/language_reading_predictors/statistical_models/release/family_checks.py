# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-family release checks over a stored fit directory.

The mediation t3 sensitivity, the joint-mechanism per-wave bundle, the
concurrent published-fit contract, the adjusted SES pairing, the gain family's
period-1 marginal, the growth influence refits and the ITT missingness
sensitivity. Each returns the failures it found; ``publication`` decides.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
from language_reading_predictors.statistical_models.release.base import (
    GROWTH_INFLUENCE_TRACE_FILENAME,
    JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS,
    MEDIATION_T3_TRACE_FILENAME,
    _read_csv,
    _read_json,
    _stored_bool,
)

def _mediation_t3_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Fail-closed computation and artefact checks for the mediation t3 fit.

    Single-mediator fits without a longitudinal primary estimand always run
    this temporal-ordering sensitivity — the interventional companions included,
    since #585 finding 2 made them run the same fit under their own labels
    (#631 finding 9 closed the release-side exemption that outlived it).  Its
    posterior bypasses the primary ``diagnostics_summary.json`` gate, so release
    requires a checked, converged provenance row, a concordant summary table and
    the persisted sub-fit trace.  Already-longitudinal primaries and the
    period-stacked entry point (whose ``extra`` carries no estimand) do not run
    it and stay exempt.

    The two returned tuples preserve the release-stage contract. A present but
    failed or unchecked convergence verdict is a computation failure; a missing,
    unreadable or internally inconsistent output is an artefact failure. Losing a
    trace after a clean fit must not be reported as evidence that sampling failed.
    """
    if config.get("kind") != "mediation":
        return (), ()
    extra = config.get("extra") or {}
    if not isinstance(extra, Mapping):
        return (), ("config.json (mediation t3 configuration is unreadable)",)
    required = (
        extra.get("estimand") in ("natural", "interventional")
        and extra.get("outcome_time") is None
    )
    if not required:
        return (), ()

    computation_failures: list[str] = []
    artifact_failures: list[str] = []
    summary = _read_csv(output_dir, "mediation_summary_t3.csv")
    if summary is None or summary.empty:
        artifact_failures.append("mediation_summary_t3.csv")
    else:
        if "converged" not in summary.columns:
            artifact_failures.append("mediation_summary_t3.csv (no convergence column)")
        elif not bool(
            summary["converged"]
            .map(lambda value: str(value).strip().casefold() in {"true", "1", "yes"})
            .all()
        ):
            computation_failures.append(
                "mediation t3 sensitivity summary convergence failed or was unchecked"
            )
        trace_files = (
            set(summary["trace_file"].dropna().astype(str))
            if "trace_file" in summary.columns
            else set()
        )
        if trace_files != {MEDIATION_T3_TRACE_FILENAME}:
            artifact_failures.append("mediation_summary_t3.csv (invalid trace binding)")

    provenance = _read_csv(output_dir, "subfit_provenance.csv")
    model_id = str(config.get("model_id") or "")
    if provenance is None or provenance.empty or "label" not in provenance.columns:
        artifact_failures.append("subfit_provenance.csv")
    else:
        rows = provenance.loc[
            provenance["label"].astype(str) == f"{model_id} t3 sensitivity"
        ]
        if len(rows) != 1:
            artifact_failures.append(
                "subfit_provenance.csv (no unique mediation t3 row)"
            )
        else:
            row = rows.iloc[0]
            if str(row.get("role", "")).strip() != "sensitivity":
                artifact_failures.append(
                    "subfit_provenance.csv (invalid mediation t3 role)"
                )
            if "converged" not in provenance.columns:
                artifact_failures.append(
                    "subfit_provenance.csv (no convergence column)"
                )
            elif str(row.get("converged", "")).strip().casefold() not in {
                "true",
                "1",
                "yes",
            }:
                computation_failures.append(
                    "mediation t3 sensitivity provenance failed or was unchecked"
                )
            if str(row.get("trace_file", "")).strip() != MEDIATION_T3_TRACE_FILENAME:
                artifact_failures.append(
                    "subfit_provenance.csv (invalid mediation t3 trace binding)"
                )

    if not (output_dir / MEDIATION_T3_TRACE_FILENAME).is_file():
        artifact_failures.append(MEDIATION_T3_TRACE_FILENAME)
    return tuple(computation_failures), tuple(artifact_failures)


def _joint_mechanism_wave_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Fail-closed bundle check for every wave a joint-mechanism levels fit publishes.

    The levels design publishes one posterior per timepoint. Before the 2026-08-23
    follow-up review only the wave hosting the fit-level artefacts passed through the
    full lifecycle, so ``release_decision.json`` could say "ok" while three of the
    four published posteriors had no persisted trace, no informative predictive check
    and no recorded power-scaling result — and the fit-level gate had never seen them.

    Returns ``(computation, artefact, qualification)``:

    * **computation** — a wave whose convergence verdict failed or could not be taken.
      Withholds, exactly as the primary gate does.
    * **artefact** — a missing or internally inconsistent bundle: an absent trace,
      predictive or power-scaling file, a slope table naming waves the diagnostics
      table does not, or a non-hosting wave with no matching sub-fit provenance row.
      Withholds.
    * **qualification** — the predeclared predictive-adequacy rule
      (:data:`JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS`). Attaches a note; does not
      withhold, because substantive misfit is a finding about the model rather than a
      computational failure.
    """
    if config.get("kind") != "joint_mechanism":
        return (), (), ()
    extra = config.get("extra") or {}
    if not isinstance(extra, Mapping):
        return (), ("config.json (joint-mechanism configuration is unreadable)",), ()
    if str(extra.get("design", "")) != "levels":
        return (), (), ()

    computation: list[str] = []
    artefacts: list[str] = []
    qualifications: list[str] = []

    diagnostics = _read_csv(output_dir, "joint_mechanism_fit_diagnostics.csv")
    required_columns = (
        "wave",
        "role",
        "converged",
        "trace_file",
        "marginal_ppc_file",
        "psense_file",
    )
    if diagnostics is None or diagnostics.empty:
        return (), ("joint_mechanism_fit_diagnostics.csv",), ()
    missing_columns = [c for c in required_columns if c not in diagnostics.columns]
    if missing_columns:
        return (
            (),
            (
                "joint_mechanism_fit_diagnostics.csv (no "
                f"{', '.join(missing_columns)} column)",
            ),
            (),
        )

    if int((diagnostics["role"].astype(str).str.strip() == "anchor").sum()) != 1:
        artefacts.append(
            "joint_mechanism_fit_diagnostics.csv (no unique artefact-hosting wave)"
        )

    provenance = _read_csv(output_dir, "subfit_provenance.csv")
    model_id = str(config.get("model_id") or "")
    for _, row in diagnostics.iterrows():
        wave = str(row["wave"]).strip()
        if _stored_bool(row.get("converged")) is not True:
            computation.append(
                f"joint-mechanism wave {wave} failed or was not convergence-checked"
            )
        for column in ("trace_file", "marginal_ppc_file", "psense_file"):
            filename = str(row.get(column) or "").strip()
            if not filename:
                artefacts.append(
                    f"joint_mechanism_fit_diagnostics.csv (wave {wave} declares no "
                    f"{column})"
                )
            elif not (output_dir / filename).is_file():
                artefacts.append(filename)
        if str(row["role"]).strip() == "anchor":
            continue
        # A non-hosting wave is a sub-fit, and a published sub-fit estimate is only
        # auditable through its provenance row: which rows it was fitted to, at what
        # sampling settings, scanning which parameters, backed by which trace.
        if provenance is None or "label" not in provenance.columns:
            artefacts.append("subfit_provenance.csv")
            continue
        rows = provenance.loc[
            provenance["label"].astype(str) == f"{model_id} wave {wave}"
        ]
        if len(rows) != 1:
            artefacts.append(f"subfit_provenance.csv (no unique {wave} row)")
            continue
        record = rows.iloc[0]
        if str(record.get("role", "")).strip() != "wave":
            artefacts.append(f"subfit_provenance.csv (invalid {wave} role)")
        if str(record.get("trace_file", "")).strip() != str(row["trace_file"]).strip():
            artefacts.append(f"subfit_provenance.csv (invalid {wave} trace binding)")
        if _stored_bool(record.get("converged")) is not True:
            computation.append(
                f"joint-mechanism wave {wave} provenance failed or was unchecked"
            )

    published = set(diagnostics["wave"].astype(str).str.strip())
    slopes = _read_csv(output_dir, "joint_mechanism_slopes.csv")
    if slopes is None or "wave" not in slopes.columns:
        artefacts.append("joint_mechanism_slopes.csv")
    else:
        reported = set(slopes["wave"].astype(str).str.strip())
        if reported != published:
            artefacts.append(
                "joint_mechanism_slopes.csv (waves do not match "
                "joint_mechanism_fit_diagnostics.csv)"
            )

    qualifications.extend(
        _joint_mechanism_coverage_qualifications(output_dir, diagnostics)
    )
    return tuple(computation), tuple(sorted(set(artefacts))), tuple(qualifications)


def _concurrent_published_fit_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Fail-closed convergence check for every fit a concurrent run publishes.

    The concurrent family publishes one adjusted posterior per wave plus a
    single-skill posterior per wave-by-predictor cell, but only the anchor wave
    passes through the fit-level gate. The pipeline computed
    ``all_published_fits_converged`` and nothing ever read it back (#631
    finding 6), so ``release_decision.json`` could say "ok" while displayed rows
    carried ``converged=False`` — the same class of defect the #591
    joint-mechanism remediation closed for its wave sub-fits.

    A missing or column-incomplete ``concurrent_fit_diagnostics.csv`` is an
    artefact failure; a published row whose verdict failed or was never taken is
    a computation failure, exactly as the primary gate treats the anchor.
    """
    if config.get("kind") != "concurrent":
        return (), ()
    diagnostics = _read_csv(output_dir, "concurrent_fit_diagnostics.csv")
    if diagnostics is None or diagnostics.empty:
        return (), ("concurrent_fit_diagnostics.csv",)
    required_columns = ("timepoint", "fit_kind", "predictor", "converged")
    missing_columns = [c for c in required_columns if c not in diagnostics.columns]
    if missing_columns:
        return (
            (),
            (
                "concurrent_fit_diagnostics.csv (no "
                f"{', '.join(missing_columns)} column)",
            ),
        )
    computation: list[str] = []
    for _, row in diagnostics.iterrows():
        if _stored_bool(row.get("converged")) is not True:
            computation.append(
                f"concurrent published fit t{str(row['timepoint']).strip()} "
                f"{str(row['fit_kind']).strip()} {str(row['predictor']).strip()} "
                "failed or was not convergence-checked"
            )
    return tuple(computation), ()


def _adjusted_ses_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Fail-closed check for the SES refit an RLI adjusted fit promises.

    The resolved recipe declares the SES complete-case sensitivity as a required
    check, but the pipeline catches every exception in that leg and continues,
    recording the failure only as ``extra["ses_error"]`` — which nothing read
    back, so the fit could publish with the section silently absent (#631
    finding 7). Scope: the ``ses_error`` key is written (null on success)
    exactly by the RLI entry point that promises the refit; the Byrne/RLM
    adjusted fits carry no such key and stay exempt.

    A recorded error or a missing/invalid ``ses_sensitivity.csv`` is an artefact
    failure; a present summary whose convergence failed or was unchecked is a
    computation failure — the same boundary the mediation t3 check draws.
    """
    if config.get("kind") != "adjusted":
        return (), ()
    extra = config.get("extra") or {}
    if not isinstance(extra, Mapping):
        return (), ("config.json (adjusted configuration is unreadable)",)
    if "ses_error" not in extra:
        return (), ()
    ses_error = extra.get("ses_error")
    if ses_error:
        return (
            (),
            (f"ses_sensitivity.csv (SES sensitivity refit failed: {ses_error})",),
        )
    summary = _read_csv(output_dir, "ses_sensitivity.csv")
    if summary is None or summary.empty:
        return (), ("ses_sensitivity.csv",)
    if "converged" not in summary.columns:
        return (), ("ses_sensitivity.csv (no convergence column)",)
    if not all(
        _stored_bool(value) is True for value in summary["converged"].tolist()
    ):
        return (
            ("adjusted SES sensitivity convergence failed or was unchecked",),
            (),
        )
    return (), ()


def _gain_period1_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Fail-closed check for the mandatory gain-factors period-1-only refit.

    The recipe declares the period-1-only refit sensitivity mandatory for every
    model of record (`period1_sensitivity_required`), and the pipeline records
    its convergence verdict and headline shift — but release evaluation consumed
    none of it (#631 finding 8), so a fit whose mandatory sensitivity failed or
    never ran still published. Opt-in on the recorded plan flag, so stored fits
    predating the field re-decide unchanged (the growth-influence precedent).

    Returns ``(computation, artefacts, robustness)``:

    * **artefacts** — a missing or malformed ``period1_sensitivity.csv`` (it must
      carry exactly one ``primary_period_stacked`` and one ``period1_only`` row
      with the headline columns) or a missing persisted sub-fit trace.
    * **computation** — a ``period1_only`` convergence verdict that failed or was
      never taken.
    * **robustness** — material disagreement between the stacked primary and the
      period-1-only refit. The documented rule (#631 finding 8) mirrors the
      growth-influence policy: the fit is withheld at the robustness stage when
      the two ``beta_trt`` posterior medians disagree in sign or the two 89%
      intervals fail to overlap; anything milder is left to the report's own
      side-by-side table.
    """
    if config.get("kind") != "gain_factors":
        return (), (), ()
    plan = config.get("resolved_run_plan") or {}
    if not isinstance(plan, Mapping) or not plan.get("period1_sensitivity_required"):
        return (), (), ()

    summary = _read_csv(output_dir, "period1_sensitivity.csv")
    if summary is None or summary.empty:
        return (), ("period1_sensitivity.csv",), ()
    required_columns = (
        "fit",
        "beta_trt_median",
        "beta_trt_lo",
        "beta_trt_hi",
        "converged",
    )
    missing_columns = [c for c in required_columns if c not in summary.columns]
    if missing_columns:
        return (
            (),
            (
                "period1_sensitivity.csv (no "
                f"{', '.join(missing_columns)} column)",
            ),
            (),
        )
    fits = summary["fit"].astype(str).str.strip()
    primary_rows = summary.loc[fits == "primary_period_stacked"]
    refit_rows = summary.loc[fits == "period1_only"]
    if len(primary_rows) != 1 or len(refit_rows) != 1:
        return (
            (),
            (
                "period1_sensitivity.csv (no unique primary_period_stacked / "
                "period1_only row pair)",
            ),
            (),
        )
    artefacts: list[str] = []
    if not (output_dir / "trace_period1_only.nc").is_file():
        artefacts.append("trace_period1_only.nc")

    computation: list[str] = []
    refit = refit_rows.iloc[0]
    if _stored_bool(refit.get("converged")) is not True:
        computation.append(
            "the mandatory gain-factors period-1-only refit sensitivity failed "
            "or was not convergence-checked"
        )

    robustness: list[str] = []
    primary = primary_rows.iloc[0]
    try:
        medians = (
            float(primary["beta_trt_median"]),
            float(refit["beta_trt_median"]),
        )
        intervals = (
            (float(primary["beta_trt_lo"]), float(primary["beta_trt_hi"])),
            (float(refit["beta_trt_lo"]), float(refit["beta_trt_hi"])),
        )
    except (TypeError, ValueError):
        artefacts.append("period1_sensitivity.csv (non-numeric headline columns)")
    else:
        if not all(np.isfinite(v) for v in (*medians, *intervals[0], *intervals[1])):
            artefacts.append("period1_sensitivity.csv (non-finite headline columns)")
        else:
            direction_stable = np.sign(medians[0]) == np.sign(medians[1])
            overlap = max(intervals[0][0], intervals[1][0]) <= min(
                intervals[0][1], intervals[1][1]
            )
            if not (direction_stable and overlap):
                robustness.append(
                    "the period-1-only refit materially disagrees with the "
                    "stacked primary (beta_trt direction or 89% interval overlap)"
                )
    return tuple(computation), tuple(artefacts), tuple(robustness)


def _joint_mechanism_coverage_qualifications(
    output_dir: Path, diagnostics: pd.DataFrame
) -> list[str]:
    """Apply the predeclared new-child coverage floors to every published wave."""
    notes: list[str] = []
    for _, row in diagnostics.iterrows():
        wave = str(row["wave"]).strip()
        filename = str(row.get("marginal_ppc_file") or "").strip()
        if not filename:
            continue
        coverage = _read_csv(output_dir, filename)
        if coverage is None or not {"level_pct", "coverage"} <= set(coverage.columns):
            continue
        for _, entry in coverage.iterrows():
            level = pd.to_numeric(entry.get("level_pct"), errors="coerce")
            value = pd.to_numeric(entry.get("coverage"), errors="coerce")
            floor = JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS.get(
                int(level) if pd.notna(level) else -1
            )
            if floor is None or pd.isna(value) or float(value) >= floor:
                continue
            stored = entry.get("outcome")
            # The pooled row leaves ``outcome`` null; a per-outcome row names it.
            outcome = "all" if pd.isna(stored) else str(stored).strip() or "all"
            notes.append(
                f"new-child predictive coverage at wave {wave} ({outcome}) is "
                f"{float(value):.2f} at the {int(level)}% level, below the "
                f"predeclared floor of {floor:.2f}"
            )
    return notes


_MISSINGNESS_DIAGNOSTIC_FIELDS = (
    "max_rhat",
    "min_ess",
    "min_bfmi",
    "n_divergences",
)


_MISSINGNESS_RHAT_MAX = 1.01


_MISSINGNESS_ESS_MIN = 400.0


_MISSINGNESS_BFMI_MIN = 0.3


def _missingness_diagnostics(record: Mapping[str, Any]) -> dict[str, float | int] | None:
    """Read the four unrounded missingness-subfit gate signals from one record."""

    values: dict[str, float | int] = {}
    for name in _MISSINGNESS_DIAGNOSTIC_FIELDS:
        value = pd.to_numeric(record.get(name), errors="coerce")
        if pd.isna(value) or not np.isfinite(float(value)):
            return None
        if name == "n_divergences":
            integer = int(value)
            if float(value) != float(integer) or integer < 0:
                return None
            values[name] = integer
        else:
            values[name] = float(value)
    return values


def _missingness_diagnostics_pass(values: Mapping[str, float | int]) -> bool:
    """Apply the same unrounded release thresholds as the primary trace gate."""

    return bool(
        float(values["max_rhat"]) <= _MISSINGNESS_RHAT_MAX
        and float(values["min_ess"]) >= _MISSINGNESS_ESS_MIN
        and float(values["min_bfmi"]) >= _MISSINGNESS_BFMI_MIN
        and int(values["n_divergences"]) == 0
    )


def _growth_influence_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Fail closed when a required growth influence refit is absent or unstable."""
    plan = config.get("resolved_run_plan") or {}
    if config.get("kind") != "growth":
        return (), (), ()
    if not isinstance(plan, Mapping):
        return (
            (),
            ("config.json (growth influence configuration is unreadable)",),
            (),
        )
    if not plan.get("observation_influence_sensitivity", False):
        return (), (), ()

    computation_failures: list[str] = []
    artifact_failures: list[str] = []
    robustness_failures: list[str] = []
    pareto = _read_csv(output_dir, "pareto_k.csv")
    pareto_columns = {
        "observation_index",
        "subject_id",
        "wave",
        "outcome",
        "pareto_k",
        "good_k_threshold",
        "loo_reliable",
    }
    if pareto is None or pareto.empty or not pareto_columns.issubset(pareto.columns):
        return (), ("pareto_k.csv (invalid growth observation-cell map)",), ()

    reliable = pareto["loo_reliable"].map(_stored_bool)
    numeric = pareto[
        ["observation_index", "wave", "pareto_k", "good_k_threshold"]
    ].apply(pd.to_numeric, errors="coerce")
    indices = numeric["observation_index"]
    expected_reliable = numeric["pareto_k"] <= numeric["good_k_threshold"]
    numeric_valid = bool(np.isfinite(numeric.to_numpy(dtype=float)).all())
    if numeric_valid:
        integer_indices = indices.to_numpy(dtype=int)
        numeric_valid = bool(
            np.array_equal(indices.to_numpy(dtype=float), integer_indices)
            and set(integer_indices) == set(range(len(indices)))
            and np.array_equal(
                numeric["wave"].to_numpy(dtype=float),
                numeric["wave"].to_numpy(dtype=int),
            )
        )
    if (
        reliable.isna().any()
        or not numeric_valid
        or indices.duplicated().any()
        or pareto[["subject_id", "outcome"]].isna().any().any()
        or not np.array_equal(
            reliable.to_numpy(dtype=bool), expected_reliable.to_numpy()
        )
    ):
        return (), ("pareto_k.csv (internally inconsistent growth diagnostics)",), ()

    flagged = pareto.loc[~reliable.to_numpy(dtype=bool)]
    if flagged.empty:
        return (), (), ()

    summary = _read_csv(output_dir, "growth_influence_sensitivity.csv")
    summary_columns = {
        "coefficient",
        "outcome",
        "n_excluded_cells",
        "n_excluded_children",
        "n_fully_excluded_children",
        "sensitivity_converged",
        "primary_median",
        "primary_lo89",
        "primary_hi89",
        "sensitivity_median",
        "sensitivity_lo89",
        "sensitivity_hi89",
        "median_direction_stable",
        "intervals_overlap",
    }
    if summary is None or summary.empty or not summary_columns.issubset(summary.columns):
        artifact_failures.append("growth_influence_sensitivity.csv")
        summary_verdict: bool | None = None
    else:
        outcomes = set(pareto["outcome"].astype(str))
        expected_rows = {
            (coefficient, outcome)
            for coefficient in ("gamma", "delta")
            for outcome in outcomes
        }
        actual_rows = set(
            summary[["coefficient", "outcome"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if (
            summary.duplicated(subset=["coefficient", "outcome"]).any()
            or actual_rows != expected_rows
        ):
            artifact_failures.append(
                "growth_influence_sensitivity.csv (invalid coefficient rows)"
            )
        counts = pd.to_numeric(summary["n_excluded_cells"], errors="coerce")
        children = pd.to_numeric(summary["n_excluded_children"], errors="coerce")
        fully_excluded = pd.to_numeric(
            summary["n_fully_excluded_children"], errors="coerce"
        )
        expected_children = flagged["subject_id"].astype(str).nunique()
        # A child is *fully* excluded only when every one of its observed cells is
        # unreliable — matching the writer, which keeps a child whose retained-cell
        # count is non-zero and counts the rest under ``all_observed_cells_high_pareto``
        # (``growth._exclude_cells``). Grouping ``~reliable`` with ``.all()`` asks
        # exactly that. The previous form, ``~reliable.groupby(...).all()``, negated
        # *after* the reduction, so it flagged children with **any** unreliable cell —
        # numerically identical to ``expected_children`` above, making the check both
        # redundant and unsatisfiable for any fit with a partially-excluded child.
        none_reliable_by_child = (~reliable).groupby(
            pareto["subject_id"].astype(str)
        ).all()
        expected_fully_excluded = int(none_reliable_by_child.sum())
        if (
            counts.isna().any()
            or not (counts == len(flagged)).all()
            or children.isna().any()
            or not (children == expected_children).all()
            or fully_excluded.isna().any()
            or not (fully_excluded == expected_fully_excluded).all()
        ):
            artifact_failures.append(
                "growth_influence_sensitivity.csv (excluded-cell map mismatch)"
            )

        stability_numeric = summary[
            [
                "primary_median",
                "primary_lo89",
                "primary_hi89",
                "sensitivity_median",
                "sensitivity_lo89",
                "sensitivity_hi89",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        direction_stable = summary["median_direction_stable"].map(_stored_bool)
        intervals_overlap = summary["intervals_overlap"].map(_stored_bool)
        stability_values_valid = bool(
            np.isfinite(stability_numeric.to_numpy(dtype=float)).all()
            and not direction_stable.isna().any()
            and not intervals_overlap.isna().any()
            and (
                stability_numeric["primary_lo89"]
                <= stability_numeric["primary_hi89"]
            ).all()
            and (
                stability_numeric["primary_lo89"]
                <= stability_numeric["primary_median"]
            ).all()
            and (
                stability_numeric["primary_median"]
                <= stability_numeric["primary_hi89"]
            ).all()
            and (
                stability_numeric["sensitivity_lo89"]
                <= stability_numeric["sensitivity_hi89"]
            ).all()
            and (
                stability_numeric["sensitivity_lo89"]
                <= stability_numeric["sensitivity_median"]
            ).all()
            and (
                stability_numeric["sensitivity_median"]
                <= stability_numeric["sensitivity_hi89"]
            ).all()
        )
        if not stability_values_valid:
            artifact_failures.append(
                "growth_influence_sensitivity.csv (invalid coefficient stability values)"
            )
        else:
            expected_direction_stable = (
                np.sign(stability_numeric["primary_median"])
                == np.sign(stability_numeric["sensitivity_median"])
            )
            expected_intervals_overlap = (
                np.maximum(
                    stability_numeric["primary_lo89"],
                    stability_numeric["sensitivity_lo89"],
                )
                <= np.minimum(
                    stability_numeric["primary_hi89"],
                    stability_numeric["sensitivity_hi89"],
                )
            )
            if not (
                np.array_equal(
                    direction_stable.to_numpy(dtype=bool),
                    expected_direction_stable.to_numpy(dtype=bool),
                )
                and np.array_equal(
                    intervals_overlap.to_numpy(dtype=bool),
                    expected_intervals_overlap.to_numpy(dtype=bool),
                )
            ):
                artifact_failures.append(
                    "growth_influence_sensitivity.csv "
                    "(coefficient stability verdict mismatch)"
                )
            elif not (
                direction_stable.to_numpy(dtype=bool).all()
                and intervals_overlap.to_numpy(dtype=bool).all()
            ):
                robustness_failures.append(
                    "growth observation-cell influence sensitivity did not preserve "
                    "every coefficient's median direction with overlapping 89% intervals"
                )
        declared = summary["sensitivity_converged"].map(_stored_bool)
        if declared.isna().any():
            artifact_failures.append(
                "growth_influence_sensitivity.csv (invalid convergence verdict)"
            )
            summary_verdict = None
        elif declared.nunique() != 1:
            artifact_failures.append(
                "growth_influence_sensitivity.csv (inconsistent convergence verdict)"
            )
            summary_verdict = None
        else:
            summary_verdict = bool(declared.iloc[0])
            if not summary_verdict:
                computation_failures.append(
                    "growth observation-cell influence sensitivity failed its "
                    "convergence gate"
                )

    provenance = _read_csv(output_dir, "subfit_provenance.csv")
    model_id = str(config.get("model_id") or "")
    label = f"{model_id} high-Pareto observation-cell exclusion"
    provenance_row: pd.Series | None = None
    provenance_verdict: bool | None = None
    if provenance is None or provenance.empty or "label" not in provenance.columns:
        artifact_failures.append("subfit_provenance.csv")
    else:
        rows = provenance.loc[provenance["label"].astype(str) == label]
        if len(rows) != 1:
            artifact_failures.append(
                "subfit_provenance.csv (no unique growth influence row)"
            )
        else:
            provenance_row = rows.iloc[0]
            if str(provenance_row.get("role", "")).strip() != "sensitivity":
                artifact_failures.append(
                    "subfit_provenance.csv (invalid growth influence role)"
                )
            if (
                str(provenance_row.get("trace_file", "")).strip()
                != GROWTH_INFLUENCE_TRACE_FILENAME
            ):
                artifact_failures.append(
                    "subfit_provenance.csv (invalid growth influence trace binding)"
                )
            values = _missingness_diagnostics(provenance_row)
            declared = _stored_bool(provenance_row.get("converged"))
            if values is None or declared is None:
                artifact_failures.append(
                    "subfit_provenance.csv (invalid growth influence diagnostics)"
                )
            else:
                provenance_verdict = declared
                passed = _missingness_diagnostics_pass(values)
                if declared != passed:
                    artifact_failures.append(
                        "subfit_provenance.csv (growth influence verdict mismatch)"
                    )
                if not passed:
                    computation_failures.append(
                        "growth observation-cell influence sensitivity failed its "
                        "convergence gate"
                    )

    trace_path = output_dir / GROWTH_INFLUENCE_TRACE_FILENAME
    if not trace_path.is_file():
        artifact_failures.append(GROWTH_INFLUENCE_TRACE_FILENAME)
    elif provenance_row is not None:
        from language_reading_predictors.statistical_models.sensitivity import (
            sha256_file,
        )

        recorded = str(provenance_row.get("trace_sha256", "")).strip().lower()
        if len(recorded) != 64 or recorded != sha256_file(trace_path):
            artifact_failures.append(
                "subfit_provenance.csv (growth influence trace hash mismatch)"
            )

    # The growth pipeline records this verdict inside ``config["extra"]``
    # (``pipelines.growth`` builds it as part of the spec's extra payload), so read
    # there as well as at the top level. Looking only at the top level made the
    # verdict unconditionally "missing" for every growth fit that ran the influence
    # sensitivity, withholding a fit whose sensitivity had in fact converged.
    influence_extra = config.get("extra")
    if not isinstance(influence_extra, Mapping):
        influence_extra = {}
    metadata_verdict = _stored_bool(
        config.get(
            "observation_influence_converged",
            influence_extra.get("observation_influence_converged"),
        )
    )
    if metadata_verdict is None:
        artifact_failures.append("config.json (growth influence verdict is missing)")
    elif not metadata_verdict:
        computation_failures.append(
            "growth observation-cell influence sensitivity failed its convergence gate"
        )
    stored_verdicts = {
        verdict
        for verdict in (summary_verdict, provenance_verdict, metadata_verdict)
        if verdict is not None
    }
    if len(stored_verdicts) > 1:
        artifact_failures.append(
            "growth influence convergence verdicts disagree across artifacts"
        )
    return (
        tuple(dict.fromkeys(computation_failures)),
        tuple(dict.fromkeys(artifact_failures)),
        tuple(dict.fromkeys(robustness_failures)),
    )


def _missingness_diagnostics_match(
    left: Mapping[str, float | int], right: Mapping[str, float | int]
) -> bool:
    """Whether two serialisations carry the same unrounded gate evidence."""

    return bool(
        int(left["n_divergences"]) == int(right["n_divergences"])
        and all(
            np.isclose(
                float(left[name]),
                float(right[name]),
                rtol=1e-10,
                atol=1e-12,
            )
            for name in ("max_rhat", "min_ess", "min_bfmi")
        )
    )


def _trailing_size(group: Any, name: str) -> int | None:
    """Size of a variable's last (non chain/draw) dimension, or ``None``."""
    try:
        array = group[name]
    except Exception:  # pragma: no cover - defensive
        return None
    dims = [d for d in getattr(array, "dims", ()) if d not in ("chain", "draw")]
    if not dims:
        return None
    try:
        return int(array.sizes[dims[-1]])
    except Exception:  # pragma: no cover - defensive
        return None


def _missingness_design_dimension_error(
    trace: Any,
    *,
    expected_targets: int | None,
    expected_observations: int | None,
) -> str | None:
    """Check the persisted trace actually carries the registered design."""
    if expected_targets is not None:
        for name in ("p0_target", "p1_target"):
            size = _trailing_size(trace["prior"], name)
            if size is None:
                return f"the /prior group's {name} has no target dimension"
            if size != expected_targets:
                return (
                    f"the /prior group's {name} covers {size} target profiles, "
                    f"not the registered {expected_targets}"
                )
    if expected_observations is not None:
        size = _trailing_size(trace["prior_predictive"], "y_post")
        if size is None:
            return "the /prior_predictive group's y_post has no observation dimension"
        if size != expected_observations:
            return (
                f"the /prior_predictive group's y_post covers {size} observations, "
                f"not the registered {expected_observations}"
            )
    return None


def _missingness_trace_diagnostics(
    trace_path: Path,
    *,
    expected_targets: int | None = None,
    expected_observations: int | None = None,
) -> tuple[dict[str, float | int] | None, str | None]:
    """Recompute the mandatory subfit gate from its persisted NetCDF trace.

    ``expected_targets`` / ``expected_observations`` are the registered design
    dimensions — 57 randomised target profiles and 53 observed word-reading rows.
    Checking them closes the gap the 2026-08-22 ITT audit found (finding 8):
    fresh generation verifies the target count, the likelihood rows and the
    arm / missingness masks, but stored evaluation verified only that the trace
    carried groups and variables *named* ``p0_target`` / ``p1_target`` /
    ``y_post`` — so a trace holding a single target and a single observation
    qualified. Names are not a design.
    """

    if not trace_path.is_file():
        return None, "missing"
    trace = None
    try:
        import arviz as az

        from language_reading_predictors.statistical_models.diagnostics import (
            subfit_convergence,
        )

        trace = az.from_netcdf(trace_path)
        groups = {
            str(group).strip("/")
            for group in getattr(trace, "groups", ())
            if str(group).strip("/")
        }
        required_groups = {"prior", "prior_predictive"}
        if not required_groups.issubset(groups):
            missing = ", ".join(
                f"/{group}" for group in sorted(required_groups - groups)
            )
            return None, f"missing required trace group(s): {missing}"
        prior_vars = set(getattr(trace["prior"], "data_vars", {}))
        prior_predictive_vars = set(
            getattr(trace["prior_predictive"], "data_vars", {})
        )
        if not {"p0_target", "p1_target"}.issubset(prior_vars):
            return None, "the /prior group lacks the registered target probabilities"
        if "y_post" not in prior_predictive_vars:
            return None, "the /prior_predictive group lacks the registered outcome"
        dimension_error = _missingness_design_dimension_error(
            trace,
            expected_targets=expected_targets,
            expected_observations=expected_observations,
        )
        if dimension_error is not None:
            return None, dimension_error
        verdict = subfit_convergence(
            trace,
            label="ITT screening-baseline missingness release check",
            var_names=[
                "alpha",
                "tau",
                "beta_screening_age",
                "beta_screening_word",
                "kappa",
            ],
        )
    except Exception as exc:  # noqa: BLE001 - unreadable trace is an artefact failure
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        with suppress(Exception):
            if trace is not None:
                trace.close()
    diagnostics = _missingness_diagnostics(verdict)
    if diagnostics is None:
        return None, "the trace sampling-quality signals could not be computed"
    return diagnostics, None


def _itt_missingness_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require the trace-bound all-57 W sensitivity declared by the ITT plan."""

    registered_primary = (
        config.get("model_id") == "lrp-rli-itt-010"
        and config.get("kind") == "itt"
        and config.get("outcome_symbol") == "W"
    )
    if not registered_primary:
        return (), ()
    plan = config.get("resolved_run_plan") or {}
    if not isinstance(plan, Mapping):
        return (), ("config.json (ITT run plan is unreadable)",)
    if not bool(plan.get("missingness_sensitivity_required_for_release")):
        return (), ("config.json (word-reading missingness sensitivity is undeclared)",)

    from language_reading_predictors.statistical_models.itt_missingness import (
        MISSINGNESS_PROVENANCE_FILENAME,
        MISSINGNESS_PPC_FILENAME,
        MISSINGNESS_PRIOR_FILENAME,
        MISSINGNESS_PRIOR_DRAWS,
        MISSINGNESS_SCENARIOS,
        MISSINGNESS_SUBFIT_LABEL,
        MISSINGNESS_SUMMARY_FILENAME,
        MISSINGNESS_TRACE_FILENAME,
        OBSERVED_CONTROL_N,
        OBSERVED_INTERVENTION_N,
        LOST_TO_FOLLOW_UP_N,
        WITHIN_ARCHIVE_W_MISSING_N,
        RLI_ARCHIVE_DOI,
        RLI_ARCHIVE_CSV_SHA256,
        RLI_LOCAL_WIDE_SHA256,
        RLI_RECONCILIATION_DIGEST,
        RANDOMISED_CONTROL_N,
        RANDOMISED_INTERVENTION_N,
        RANDOMISED_N,
        WORD_READING_N,
        DEFAULT_DELTA_ITEMS,
        SCREENING_ALPHA_SIGMA,
        SCREENING_COVARIATES,
        sha256_file,
        validate_missingness_prior_check,
        validate_missingness_summary,
    )

    computation_failures: list[str] = []
    artifact_failures: list[str] = []
    stored_diagnostics: list[
        tuple[str, dict[str, float | int], bool | None]
    ] = []
    saved_missingness_plan = plan.get("missingness_plan") or {}
    expected_plan = {
        "source_csv_sha256": RLI_ARCHIVE_CSV_SHA256,
        "source_doi": RLI_ARCHIVE_DOI,
        "local_wide_sha256": RLI_LOCAL_WIDE_SHA256,
        "reconciliation_digest": RLI_RECONCILIATION_DIGEST,
        "screening_covariates": list(SCREENING_COVARIATES),
        "randomised_n": RANDOMISED_N,
        "randomised_intervention_n": RANDOMISED_INTERVENTION_N,
        "randomised_control_n": RANDOMISED_CONTROL_N,
        "observed_intervention_n": OBSERVED_INTERVENTION_N,
        "observed_control_n": OBSERVED_CONTROL_N,
        "lost_to_follow_up_n": LOST_TO_FOLLOW_UP_N,
        "within_archive_w_missing_n": WITHIN_ARCHIVE_W_MISSING_N,
        "word_reading_n": WORD_READING_N,
        "delta_items": list(DEFAULT_DELTA_ITEMS),
        "scenarios": list(MISSINGNESS_SCENARIOS),
        "common_estimand_class": "common_profile_standardisation",
        "completion_estimand_class": "randomised_arm_factual_completion",
        "intercept_prior_anchor": "mean_all_57_screening_word_reading_logit",
        "intercept_prior_sigma": SCREENING_ALPHA_SIGMA,
        "prior_predictive_draws": MISSINGNESS_PRIOR_DRAWS,
        "trace_filename": MISSINGNESS_TRACE_FILENAME,
        "summary_filename": MISSINGNESS_SUMMARY_FILENAME,
        "ppc_filename": MISSINGNESS_PPC_FILENAME,
        "prior_check_filename": MISSINGNESS_PRIOR_FILENAME,
        "provenance_filename": MISSINGNESS_PROVENANCE_FILENAME,
    }
    if not isinstance(saved_missingness_plan, Mapping) or any(
        saved_missingness_plan.get(key) != value for key, value in expected_plan.items()
    ):
        artifact_failures.append("config.json (invalid word-reading missingness plan)")
    trace_path = output_dir / MISSINGNESS_TRACE_FILENAME
    summary = _read_csv(output_dir, MISSINGNESS_SUMMARY_FILENAME)
    if summary is None or summary.empty:
        artifact_failures.append(MISSINGNESS_SUMMARY_FILENAME)
    else:
        for error in validate_missingness_summary(
            summary,
            trace_path=trace_path,
            require_converged=False,
        ):
            artifact_failures.append(f"{MISSINGNESS_SUMMARY_FILENAME} ({error})")
        required_diagnostic_columns = {
            *_MISSINGNESS_DIAGNOSTIC_FIELDS,
            "converged",
        }
        if not required_diagnostic_columns.issubset(summary.columns):
            artifact_failures.append(
                f"{MISSINGNESS_SUMMARY_FILENAME} (missing raw subfit diagnostics)"
            )
        else:
            summary_values = _missingness_diagnostics(summary.iloc[0])
            rows_agree = summary_values is not None and all(
                (values := _missingness_diagnostics(row)) is not None
                and _missingness_diagnostics_match(values, summary_values)
                for _, row in summary.iterrows()
            )
            declared = {_stored_bool(value) for value in summary["converged"]}
            if not rows_agree or len(declared) != 1 or None in declared:
                artifact_failures.append(
                    f"{MISSINGNESS_SUMMARY_FILENAME} "
                    "(inconsistent or invalid raw subfit diagnostics)"
                )
            else:
                stored_diagnostics.append(
                    (
                        MISSINGNESS_SUMMARY_FILENAME,
                        summary_values,
                        next(iter(declared)),
                    )
                )

    provenance_payload, provenance_error = _read_json(
        output_dir / MISSINGNESS_PROVENANCE_FILENAME
    )
    if provenance_error is not None or not isinstance(provenance_payload, Mapping):
        artifact_failures.append(MISSINGNESS_PROVENANCE_FILENAME)
    else:
        source = provenance_payload.get("source") or {}
        analysis = provenance_payload.get("analysis") or {}
        trace = provenance_payload.get("trace") or {}
        outputs = provenance_payload.get("outputs") or {}
        if (
            not isinstance(source, Mapping)
            or source.get("csv_sha256") != RLI_ARCHIVE_CSV_SHA256
            or source.get("local_wide_sha256") != RLI_LOCAL_WIDE_SHA256
            or source.get("reconciled_included_n") != 54
            or source.get("reconciliation_digest") != RLI_RECONCILIATION_DIGEST
        ):
            artifact_failures.append(
                f"{MISSINGNESS_PROVENANCE_FILENAME} (invalid source binding)"
            )
        if (
            not isinstance(analysis, Mapping)
            or analysis.get("observed_outcome_n") != 53
            or analysis.get("target_profile_n") != RANDOMISED_N
            or analysis.get("randomised_by_arm")
            != {"intervention": RANDOMISED_INTERVENTION_N, "control": RANDOMISED_CONTROL_N}
            or analysis.get("observed_outcome_by_arm")
            != {"intervention": OBSERVED_INTERVENTION_N, "control": OBSERVED_CONTROL_N}
            or analysis.get("lost_to_follow_up_n") != LOST_TO_FOLLOW_UP_N
            or analysis.get("within_archive_word_reading_missing_n")
            != WITHIN_ARCHIVE_W_MISSING_N
            or analysis.get("screening_covariates") != list(SCREENING_COVARIATES)
            or analysis.get("delta_items_grid") != list(DEFAULT_DELTA_ITEMS)
        ):
            artifact_failures.append(
                f"{MISSINGNESS_PROVENANCE_FILENAME} (invalid analysis contract)"
            )
        # The recorded design (2026-08-22 ITT audit, finding 8). Absent on fits
        # written before the block existed, which therefore re-decide exactly as
        # before; present, it must agree with the registered trial contract
        # rather than merely be well-formed. Counts alone cannot establish that
        # two runs completed the same profiles, so the digest must be there too.
        recorded_design = analysis.get("design") if isinstance(analysis, Mapping) else None
        if isinstance(recorded_design, Mapping):
            expected_design = {
                "target_profile_n": RANDOMISED_N,
                "observed_outcome_n": OBSERVED_INTERVENTION_N + OBSERVED_CONTROL_N,
                "target_by_arm": {
                    "intervention": RANDOMISED_INTERVENTION_N,
                    "control": RANDOMISED_CONTROL_N,
                },
                "target_observed_by_arm": {
                    "intervention": OBSERVED_INTERVENTION_N,
                    "control": OBSERVED_CONTROL_N,
                },
                "covariate_names": list(SCREENING_COVARIATES),
            }
            disagreeing = sorted(
                key
                for key, value in expected_design.items()
                if recorded_design.get(key) != value
            )
            if disagreeing:
                artifact_failures.append(
                    f"{MISSINGNESS_PROVENANCE_FILENAME} (recorded design disagrees "
                    f"with the registered trial contract: {', '.join(disagreeing)})"
                )
            if not str(recorded_design.get("target_design_sha256") or ""):
                artifact_failures.append(
                    f"{MISSINGNESS_PROVENANCE_FILENAME} "
                    "(recorded design carries no digest)"
                )
        actual_trace_sha256 = sha256_file(trace_path) if trace_path.is_file() else None
        if (
            not isinstance(trace, Mapping)
            or trace.get("file") != MISSINGNESS_TRACE_FILENAME
            or trace.get("sha256") != actual_trace_sha256
        ):
            artifact_failures.append(
                f"{MISSINGNESS_PROVENANCE_FILENAME} (invalid trace binding)"
            )
        if isinstance(trace, Mapping):
            trace_values = _missingness_diagnostics(trace)
            trace_declared = _stored_bool(trace.get("converged"))
            if trace_values is None or trace_declared is None:
                artifact_failures.append(
                    f"{MISSINGNESS_PROVENANCE_FILENAME} "
                    "(invalid raw subfit diagnostics)"
                )
            else:
                stored_diagnostics.append(
                    (
                        MISSINGNESS_PROVENANCE_FILENAME,
                        trace_values,
                        trace_declared,
                    )
                )
        summary_path = output_dir / MISSINGNESS_SUMMARY_FILENAME
        ppc_path = output_dir / MISSINGNESS_PPC_FILENAME
        prior_path = output_dir / MISSINGNESS_PRIOR_FILENAME
        if (
            not isinstance(outputs, Mapping)
            or outputs.get("summary_file") != MISSINGNESS_SUMMARY_FILENAME
            or outputs.get("summary_sha256")
            != (sha256_file(summary_path) if summary_path.is_file() else None)
            or outputs.get("ppc_file") != MISSINGNESS_PPC_FILENAME
            or outputs.get("ppc_sha256")
            != (sha256_file(ppc_path) if ppc_path.is_file() else None)
            or outputs.get("prior_check_file") != MISSINGNESS_PRIOR_FILENAME
            or outputs.get("prior_check_sha256")
            != (sha256_file(prior_path) if prior_path.is_file() else None)
        ):
            artifact_failures.append(
                f"{MISSINGNESS_PROVENANCE_FILENAME} (invalid output binding)"
            )

    prior_check = _read_csv(output_dir, MISSINGNESS_PRIOR_FILENAME)
    if prior_check is None or prior_check.empty:
        artifact_failures.append(MISSINGNESS_PRIOR_FILENAME)
    else:
        for error in validate_missingness_prior_check(prior_check):
            artifact_failures.append(f"{MISSINGNESS_PRIOR_FILENAME} ({error})")

    bounds = _read_csv(output_dir, "attrition_bounds.csv")
    required_bounds = {
        "outcome",
        "observed_intervention_n",
        "observed_control_n",
        "missing_intervention_n",
        "missing_control_n",
        "n_trials",
    }
    if bounds is None or len(bounds) != 1 or not required_bounds.issubset(bounds.columns):
        artifact_failures.append("attrition_bounds.csv")
    else:
        row = bounds.iloc[0]
        numeric_contract = {
            "observed_intervention_n": OBSERVED_INTERVENTION_N,
            "observed_control_n": OBSERVED_CONTROL_N,
            "missing_intervention_n": 1,
            "missing_control_n": 3,
            "n_trials": WORD_READING_N,
        }
        if str(row.get("outcome")) != "W" or any(
            not np.isclose(
                float(pd.to_numeric(row.get(key), errors="coerce")),
                float(value),
            )
            for key, value in numeric_contract.items()
        ):
            artifact_failures.append("attrition_bounds.csv (invalid W count contract)")

    ppc = _read_csv(output_dir, MISSINGNESS_PPC_FILENAME)
    required_ppc = {
        "arm",
        "n",
        "observed_mean_items",
        "posterior_predictive_mean_items",
        "mean_absolute_prediction_error_items",
        "coverage_50",
        "coverage_89",
    }
    if ppc is None or len(ppc) != 3 or not required_ppc.issubset(ppc.columns):
        artifact_failures.append(MISSINGNESS_PPC_FILENAME)
    else:
        expected_n = {"all": 53, "intervention": 28, "control": 25}
        observed_n = dict(
            zip(
                ppc["arm"].astype(str),
                pd.to_numeric(ppc["n"], errors="coerce"),
                strict=True,
            )
        )
        numeric_ppc = ppc[list(required_ppc - {"arm"})].apply(
            pd.to_numeric, errors="coerce"
        )
        if observed_n != expected_n or not np.isfinite(
            numeric_ppc.to_numpy(dtype=float)
        ).all():
            artifact_failures.append(f"{MISSINGNESS_PPC_FILENAME} (invalid values)")
        elif not (
            numeric_ppc["coverage_50"].between(0.0, 1.0).all()
            and numeric_ppc["coverage_89"].between(0.0, 1.0).all()
        ):
            artifact_failures.append(f"{MISSINGNESS_PPC_FILENAME} (invalid coverage)")

    subfits = _read_csv(output_dir, "subfit_provenance.csv")
    if subfits is None or subfits.empty or "label" not in subfits.columns:
        artifact_failures.append("subfit_provenance.csv")
    else:
        rows = subfits.loc[subfits["label"].astype(str) == MISSINGNESS_SUBFIT_LABEL]
        if len(rows) != 1:
            artifact_failures.append(
                "subfit_provenance.csv (no unique ITT missingness row)"
            )
        else:
            row = rows.iloc[0]
            if str(row.get("role", "")).strip() != "sensitivity":
                artifact_failures.append(
                    "subfit_provenance.csv (invalid ITT missingness role)"
                )
            if str(row.get("trace_file", "")).strip() != MISSINGNESS_TRACE_FILENAME:
                artifact_failures.append(
                    "subfit_provenance.csv (invalid ITT missingness trace binding)"
                )
            subfit_values = _missingness_diagnostics(row)
            subfit_declared = _stored_bool(row.get("converged"))
            if subfit_values is None or subfit_declared is None:
                artifact_failures.append(
                    "subfit_provenance.csv (invalid raw ITT missingness diagnostics)"
                )
            else:
                stored_diagnostics.append(
                    ("subfit_provenance.csv", subfit_values, subfit_declared)
                )
            n_obs = pd.to_numeric(row.get("n_obs"), errors="coerce")
            n_children = pd.to_numeric(row.get("n_children"), errors="coerce")
            if not (
                pd.notna(n_obs)
                and pd.notna(n_children)
                and float(n_obs) == 53.0
                and float(n_children) == 53.0
                and bool(str(row.get("data_digest", "")).strip())
            ):
                artifact_failures.append(
                    "subfit_provenance.csv (invalid ITT missingness data identity)"
                )
    if not trace_path.is_file():
        artifact_failures.append(MISSINGNESS_TRACE_FILENAME)
    trace_diagnostics, trace_diagnostics_error = _missingness_trace_diagnostics(
        trace_path,
        expected_targets=RANDOMISED_N,
        expected_observations=OBSERVED_INTERVENTION_N + OBSERVED_CONTROL_N,
    )
    if trace_diagnostics_error is not None or trace_diagnostics is None:
        if trace_path.is_file():
            artifact_failures.append(
                f"{MISSINGNESS_TRACE_FILENAME} ({trace_diagnostics_error})"
            )
    else:
        trace_passed = _missingness_diagnostics_pass(trace_diagnostics)
        if not trace_passed:
            computation_failures.append(
                "ITT screening-baseline missingness sub-fit failed the raw "
                "sampling-quality thresholds"
            )
        for label, values, declared in stored_diagnostics:
            if not _missingness_diagnostics_match(values, trace_diagnostics):
                artifact_failures.append(
                    f"{label} (raw subfit diagnostics do not match the trace)"
                )
            elif declared != _missingness_diagnostics_pass(values):
                artifact_failures.append(
                    f"{label} (stored convergence verdict contradicts raw diagnostics)"
                )
    return tuple(computation_failures), tuple(artifact_failures)
