# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and run plan for historical-cohort growth models.

The registered ``kind="historical_growth"`` models fit one bounded Byrne
reading-language-memory measure at a time over a complete-case core window and
an optional available-case extension.  This module replaces the family's
free-form ``ModelSpec.extra`` boundary with immutable settings and a validated
plan resolved before an output transaction is opened or study data are loaded
(#394 pillar 4).

The migration is behaviour-preserving: selected rows, the Beta-Binomial
likelihood, priors, fitted equation, diagnostic variables, PSIS-LOO policy and
published tables remain unchanged for all nine registered models.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.datasets import resolve_dataset

__all__ = [
    "HistoricalGrowthModelSettings",
    "HistoricalGrowthRunPlan",
    "declared_historical_growth_settings",
    "evaluate_historical_growth_influence_bundle",
    "exclude_historical_growth_observations",
    "historical_growth_influence_summary",
    "historical_growth_pareto_table",
    "resolve_historical_growth_run_plan",
]


_DEFAULT_MEASURE = "basread"
_DEFAULT_WAVES = (1, 2, 3)
_LEGACY_KEYS = frozenset(
    {
        "study_id",
        "measure",
        "waves",
        "extension_waves",
        "eta_prior_sigma",
        "sigma_subject_prior_sigma",
        "kappa_prior_sigma",
        # Global sampler setting resolved by ``make_context``, not this family.
        "target_accept",
    }
)
_GROWTH_DETERMINISTICS = (
    "growth_first_next_items",
    "growth_next_last_items",
    "growth_first_last_items",
)


def _non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _wave_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of positive integers, got {value!r}")
    out = tuple(value)
    if any(isinstance(wave, bool) or not isinstance(wave, int) for wave in out):
        raise TypeError(f"{name} must contain positive integers, got {out!r}")
    if any(wave <= 0 for wave in out):
        raise ValueError(f"{name} must contain positive integers, got {out!r}")
    if tuple(sorted(set(out))) != out:
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    return out


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return out


@dataclass(frozen=True, slots=True)
class HistoricalGrowthModelSettings:
    """Immutable declaration for one historical-cohort growth model."""

    measure: str = _DEFAULT_MEASURE
    waves: tuple[int, ...] = _DEFAULT_WAVES
    extension_waves: tuple[int, ...] = ()
    eta_prior_sigma: float = 1.5
    sigma_subject_prior_sigma: float = 1.0
    kappa_prior_sigma: float = 50.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "measure", _non_empty_string(self.measure, name="measure"))
        object.__setattr__(self, "waves", _wave_tuple(self.waves, name="waves"))
        if len(self.waves) < 2:
            raise ValueError("historical_growth waves must contain at least two waves")
        object.__setattr__(
            self,
            "extension_waves",
            _wave_tuple(self.extension_waves, name="extension_waves"),
        )
        overlap = sorted(set(self.waves) & set(self.extension_waves))
        if overlap:
            raise ValueError(f"extension_waves overlap the complete-case core waves: {overlap}")
        for name in (
            "eta_prior_sigma",
            "sigma_subject_prior_sigma",
            "kappa_prior_sigma",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name=name))

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
        spec_study_id: str,
        outcome_symbol: str | None,
    ) -> HistoricalGrowthModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown historical_growth setting(s): "
                f"{', '.join(unknown)}. Declare HistoricalGrowthModelSettings so "
                "misspellings fail fast."
            )
        legacy_study_id = extra.get("study_id", spec_study_id)
        if not isinstance(legacy_study_id, str) or not legacy_study_id:
            raise TypeError("study_id must be a non-empty string")
        if legacy_study_id != spec_study_id:
            raise ValueError(
                f"{model_id}: extra study_id={legacy_study_id!r} contradicts "
                f"ModelSpec.study_id={spec_study_id!r}"
            )
        return cls(
            measure=extra.get("measure", outcome_symbol or _DEFAULT_MEASURE),
            waves=extra.get("waves", _DEFAULT_WAVES),
            extension_waves=extra.get("extension_waves", ()),
            eta_prior_sigma=extra.get("eta_prior_sigma", 1.5),
            sigma_subject_prior_sigma=extra.get("sigma_subject_prior_sigma", 1.0),
            kappa_prior_sigma=extra.get("kappa_prior_sigma", 50.0),
        )


@dataclass(frozen=True, slots=True)
class HistoricalGrowthRunPlan:
    """Concrete, validated instructions consumed by the complete family fit."""

    model_id: str
    settings_source: str
    study_id: str
    measure: str
    waves: tuple[int, ...]
    extension_waves: tuple[int, ...]
    complete_case: bool
    likelihood: str
    observation_node: str
    eta_prior_sigma: float
    sigma_subject_prior_sigma: float
    kappa_prior_sigma: float
    compute_loo: bool
    loo_unit: str
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``load_longitudinal_panel``."""
        return {
            "waves": self.waves,
            "complete_case": self.complete_case,
            "extension_waves": self.extension_waves,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_historical_growth_model``."""
        return {
            "measure": self.measure,
            "eta_prior_sigma": self.eta_prior_sigma,
            "sigma_subject_prior_sigma": self.sigma_subject_prior_sigma,
            "kappa_prior_sigma": self.kappa_prior_sigma,
        }

    def diagnostic_vars(self, available_vars: Collection[str]) -> list[str]:
        """Curated diagnostics, preserving the factory's conditional deterministics."""
        return [
            "eta_cell",
            "sigma_subject",
            "kappa",
            *(name for name in _GROWTH_DETERMINISTICS if name in available_vars),
        ]

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language account generated from the validated run plan."""
        core_waves = ", ".join(str(wave) for wave in self.waves)
        extension = ", ".join(str(wave) for wave in self.extension_waves) if self.extension_waves else "none"
        return (
            "Note: Generated from the validated historical-growth run plan; "
            "template drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Measure: `{self.measure}`. Complete-case core waves: {core_waves}. "
            f"Available-case extension waves: {extension}. Likelihood: "
            "Beta-Binomial bounded counts with group-by-wave means, "
            "group-specific child-level scales and group-specific "
            "overdispersion.\n\n"
            "## Uncertainty and checks\n\n"
            "Interpret the posterior only after the convergence gate, PSIS-LOO, "
            "posterior-predictive checks and prior-sensitivity diagnostics pass. "
            "The saved `config.json` contains the same resolved run plan in "
            "machine-readable form.\n"
        )


def declared_historical_growth_settings(
    spec: ModelSpec,
) -> tuple[HistoricalGrowthModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: historical_growth settings cannot be split "
                "between model_settings and extra"
            )
        if not isinstance(settings, HistoricalGrowthModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='historical_growth' requires "
                "HistoricalGrowthModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return (
        HistoricalGrowthModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
            spec_study_id=spec.study_id,
            outcome_symbol=spec.outcome_symbol,
        ),
        "legacy_extra",
    )


def resolve_historical_growth_run_plan(spec: ModelSpec) -> HistoricalGrowthRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "historical_growth":
        raise ValueError(f"{spec.model_id}: expected kind 'historical_growth', got {spec.kind!r}")
    if not isinstance(spec.study_id, str) or not spec.study_id:
        raise TypeError(f"{spec.model_id}: study_id must be a non-empty string")

    settings, source = declared_historical_growth_settings(spec)
    _dataset, catalogue = resolve_dataset(spec.study_id)
    if settings.measure not in catalogue:
        raise ValueError(
            f"{spec.model_id}: unregistered {spec.study_id!r} measure symbol: {settings.measure}"
        )
    if spec.outcome_symbol is not None and spec.outcome_symbol != settings.measure:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol={spec.outcome_symbol!r} contradicts "
            f"historical_growth measure={settings.measure!r}"
        )

    return HistoricalGrowthRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        measure=settings.measure,
        waves=settings.waves,
        extension_waves=settings.extension_waves,
        complete_case=True,
        likelihood="beta_binomial",
        observation_node="score",
        eta_prior_sigma=settings.eta_prior_sigma,
        sigma_subject_prior_sigma=settings.sigma_subject_prior_sigma,
        kappa_prior_sigma=settings.kappa_prior_sigma,
        compute_loo=True,
        loo_unit="observation_row",
        design=(
            "Descriptive Beta-Binomial group-by-wave growth model for one bounded "
            "measure in a historical cohort. Supported group-wave cells have separate "
            "means; child-level heterogeneity and overdispersion are group-specific."
        ),
        estimand=(
            "The headline quantities are within-group changes in expected item score "
            "over supported wave intervals. Group-by-wave expected levels and "
            "between-group contrasts over the common observation window are secondary "
            "descriptive summaries."
        ),
        causal_status=(
            "Descriptive only: cohort group is observational, no coefficient is a "
            "treatment effect, and between-group differences must not be read causally."
        ),
        analysis_population=(
            "Children observed on the selected measure at every complete-case core "
            "wave. Retained children contribute extension-wave rows when the measure "
            "is observed there."
        ),
        missing_data_assumption=(
            "Complete-case selection defines the core cohort; extension waves are "
            "available-case among that retained cohort. Later-wave summaries therefore "
            "describe an attrition-selected observed tail, not automatically all "
            "recruited children."
        ),
    )


def historical_growth_pareto_table(
    panel: Any,
    loo: Any,
    *,
    measure: str,
) -> pd.DataFrame:
    """Map historical-growth Pareto-k values to their likelihood rows.

    The factory passes ``panel.long`` to the likelihood without reshaping, so
    pointwise PSIS-LOO and this table have the same row order. The explicit map
    prevents a high-k child-wave observation from being mistaken for a
    whole-child leave-out unit.
    """
    if measure not in panel.measures:
        raise KeyError(f"measure {measure!r} is not in panel {panel.measures!r}")
    if loo is None or getattr(loo, "pareto_k", None) is None:
        raise ValueError(
            "historical-growth influence sensitivity requires pointwise PSIS-LOO"
        )

    pareto_k = np.asarray(loo.pareto_k, dtype=float).ravel()
    if len(panel.long) != len(pareto_k):
        raise ValueError(
            "historical-growth rows do not align with pointwise Pareto-k values: "
            f"{len(panel.long)} rows versus {len(pareto_k)} diagnostics"
        )
    threshold = float(getattr(loo, "good_k", 0.7) or 0.7)
    dataset = panel.dataset
    frame = panel.long.reset_index(drop=True)
    out = pd.DataFrame(
        {
            "observation_index": np.arange(len(frame), dtype=int),
            "subject_id": frame[dataset.subject_col].to_numpy(),
            "wave": frame[dataset.wave_col].to_numpy(dtype=int),
            "group_code": frame[dataset.group_col].to_numpy(dtype=int),
            "group_label": frame[panel.group_label_col].to_numpy(),
            "outcome": measure,
            "score": frame[measure].to_numpy(dtype=int),
            "pareto_k": pareto_k,
            "good_k_threshold": threshold,
            "loo_reliable": pareto_k <= threshold,
        }
    )
    return out.sort_values("pareto_k", ascending=False).reset_index(drop=True)


def exclude_historical_growth_observations(
    panel: Any,
    observation_indices: Any,
) -> Any:
    """Return ``panel`` with selected likelihood rows removed.

    A child remains in the sensitivity panel through every unflagged row. If all
    of a child's rows are removed, the child is removed as well so the refit does
    not retain an unconstrained random intercept. This is a coefficient-stability
    refit, not exact LOO and not a new-child predictive calculation.
    """
    raw = np.asarray(observation_indices)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(
            "observation_indices must be a non-empty one-dimensional array"
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError("observation_indices must contain integers")
    indices = raw.astype(int)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("observation_indices must be unique")
    unknown = sorted(set(indices) - set(range(len(panel.long))))
    if unknown:
        raise IndexError(
            f"historical-growth observation index out of range: {unknown}"
        )

    dataset = panel.dataset
    subject_col = dataset.subject_col
    wave_col = dataset.wave_col
    long = panel.long.drop(index=indices).reset_index(drop=True)
    if long.empty:
        raise ValueError("observation exclusion would leave no fitted rows")

    subject_ids = long[subject_col].drop_duplicates().tolist()
    fully_excluded = len(set(panel.subject_ids) - set(subject_ids))
    group_codes = sorted(int(code) for code in long[dataset.group_col].unique())
    group_labels = [dataset.group_labels[code] for code in group_codes]

    counts: dict[str, np.ndarray] = {}
    obs_mask: dict[str, np.ndarray] = {}
    for measure in panel.measures:
        wide = long.pivot_table(
            index=subject_col,
            columns=wave_col,
            values=measure,
            aggfunc="first",
        ).reindex(index=subject_ids, columns=list(panel.waves))
        values = wide.to_numpy(dtype=float)
        counts[measure] = values
        obs_mask[measure] = np.isfinite(values)

    return replace(
        panel,
        long=long,
        subject_ids=subject_ids,
        group_codes=group_codes,
        group_labels=group_labels,
        counts=counts,
        obs_mask=obs_mask,
        n_subjects=len(subject_ids),
        dropped_subjects=panel.dropped_subjects + fully_excluded,
    )


def historical_growth_influence_summary(
    primary_trace: Any,
    sensitivity_trace: Any,
    *,
    primary_panel: Any,
    sensitivity_panel: Any,
    measure: str,
    excluded_rows: pd.DataFrame,
    sensitivity_converged: bool | None,
) -> pd.DataFrame:
    """Compare reported growth quantities before and after row exclusion.

    These are two separately sampled posteriors, so the table compares their
    marginal summaries and the shift in medians. It does not treat draws from
    the two fits as paired or claim a posterior distribution for their
    difference.
    """
    required = {"subject_id", "observation_index", "pareto_k"}
    missing = required - set(excluded_rows.columns)
    if missing:
        raise ValueError(
            "excluded_rows lacks required columns: " + ", ".join(sorted(missing))
        )
    if excluded_rows.empty:
        raise ValueError(
            "historical-growth influence summary requires an excluded row"
        )

    from language_reading_predictors.statistical_models import historical

    keys = ["quantity", "label", "readgrp_label", "window"]
    statistics = ["n_subjects", "q50", "q_lo", "q_hi", "p_gt_0"]

    def _summary(trace: Any, panel: Any, prefix: str) -> pd.DataFrame:
        frame = historical.growth_summary(trace, panel, measure)
        return frame[keys + statistics].rename(
            columns={column: f"{prefix}_{column}" for column in statistics}
        )

    primary = _summary(primary_trace, primary_panel, "primary")
    sensitivity = _summary(sensitivity_trace, sensitivity_panel, "sensitivity")
    out = primary.merge(sensitivity, on=keys, how="outer", validate="one_to_one")
    out["median_shift"] = out["sensitivity_q50"] - out["primary_q50"]
    out["median_direction_stable"] = np.sign(out["primary_q50"]) == np.sign(
        out["sensitivity_q50"]
    )
    out["intervals_overlap"] = np.maximum(
        out["primary_q_lo"], out["sensitivity_q_lo"]
    ) <= np.minimum(out["primary_q_hi"], out["sensitivity_q_hi"])
    out["n_excluded_rows"] = int(len(excluded_rows))
    out["n_excluded_children"] = int(excluded_rows["subject_id"].nunique())
    out["n_fully_excluded_children"] = int(
        len(set(primary_panel.subject_ids) - set(sensitivity_panel.subject_ids))
    )
    out["max_excluded_pareto_k"] = float(excluded_rows["pareto_k"].max())
    out["sensitivity_converged"] = sensitivity_converged
    return out


def evaluate_historical_growth_influence_bundle(
    summary: pd.DataFrame | None,
    primary_dir: Path,
    report_config: Mapping[str, Any],
    expected_config: str,
) -> dict[str, Any]:
    """Fail closed when a report-local influence bundle is stale or partial."""
    from language_reading_predictors.statistical_models.sensitivity import (
        sha256_file,
    )

    result: dict[str, Any] = {
        "ready": False,
        "reason": "historical_growth_influence_sensitivity.csv is absent",
        "max_median_shift": float("nan"),
    }
    if summary is None or summary.empty:
        return result

    primary_dir = Path(primary_dir).resolve()
    required = {
        "model_id",
        "config",
        "median_shift",
        "median_direction_stable",
        "intervals_overlap",
        "n_excluded_rows",
        "max_excluded_pareto_k",
        "sensitivity_converged",
        "primary_config_sha256",
        "primary_trace_sha256",
        "primary_pareto_k_sha256",
        "sensitivity_trace_file",
        "sensitivity_trace_sha256",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        result["reason"] = "missing columns: " + ", ".join(missing)
        return result

    def _one(column: str) -> Any:
        values = summary[column].drop_duplicates()
        if len(values) != 1:
            raise ValueError(f"{column} is not constant across the bundle")
        return values.iloc[0]

    def _all_true(column: str) -> bool:
        return set(summary[column].astype(str).str.strip().str.lower()) == {"true"}

    try:
        if str(_one("model_id")) != str(report_config.get("model_id")):
            raise ValueError("model id does not match the report")
        if str(report_config.get("kind")) != "historical_growth":
            raise ValueError("report is not a historical-growth model")
        if str(_one("config")) != expected_config:
            raise ValueError("sampling config does not match the report directory")

        artefacts = {
            "primary_config_sha256": primary_dir / "config.json",
            "primary_trace_sha256": primary_dir / "trace.nc",
            "primary_pareto_k_sha256": primary_dir / "pareto_k.csv",
        }
        for column, path in artefacts.items():
            if not path.is_file() or str(_one(column)) != sha256_file(path):
                raise ValueError(f"stale or mixed bundle: {column} does not match")

        pareto = pd.read_csv(primary_dir / "pareto_k.csv")
        pareto_required = {
            "observation_index",
            "subject_id",
            "pareto_k",
            "good_k_threshold",
        }
        if not pareto_required.issubset(pareto.columns):
            raise ValueError("current Pareto-k table lacks its row mapping")
        observation_indices = pd.to_numeric(
            pareto["observation_index"], errors="coerce"
        )
        values = pd.to_numeric(pareto["pareto_k"], errors="coerce")
        thresholds = pd.to_numeric(
            pareto["good_k_threshold"], errors="coerce"
        )
        expected_n = int(report_config.get("n_obs", -1))
        if (
            len(pareto) != expected_n
            or not np.isfinite(observation_indices).all()
            or set(observation_indices.astype(int)) != set(range(expected_n))
            or observation_indices.astype(int).duplicated().any()
            or not np.isfinite(values).all()
            or not np.isfinite(thresholds).all()
            or thresholds.nunique() != 1
        ):
            raise ValueError("current Pareto-k table is not a complete row map")
        flagged = pareto.loc[values > thresholds]
        if flagged.empty or int(_one("n_excluded_rows")) != len(flagged):
            raise ValueError("excluded-row count does not match current Pareto-k flags")
        if not np.isclose(
            float(_one("max_excluded_pareto_k")),
            float(values.loc[flagged.index].max()),
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError("saved maximum Pareto-k does not match the current table")

        trace_name = str(_one("sensitivity_trace_file"))
        if Path(trace_name).name != trace_name:
            raise ValueError("sensitivity trace path is not a report-local filename")
        sensitivity_trace = primary_dir / trace_name
        if (
            not sensitivity_trace.is_file()
            or str(_one("sensitivity_trace_sha256"))
            != sha256_file(sensitivity_trace)
        ):
            raise ValueError("sensitivity trace is absent or hash-mismatched")

        provenance_path = primary_dir / "historical_growth_influence_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "completed":
            raise ValueError("influence provenance does not record a completed fit")
        if provenance.get("model_id") != report_config.get("model_id"):
            raise ValueError("influence provenance model id does not match the report")
        if provenance.get("config") != expected_config:
            raise ValueError("influence provenance config does not match the report")
        for column, path in artefacts.items():
            if provenance.get(column) != sha256_file(path):
                raise ValueError(f"influence provenance {column} does not match")
        flagged_indices = sorted(
            int(value) for value in flagged["observation_index"]
        )
        if provenance.get("flagged_observation_indices") != flagged_indices:
            raise ValueError("influence provenance flags do not match current Pareto-k")
        if provenance.get("sensitivity_trace_sha256") != sha256_file(
            sensitivity_trace
        ):
            raise ValueError("influence provenance is not bound to the sensitivity trace")
        summary_path = primary_dir / "historical_growth_influence_sensitivity.csv"
        if provenance.get("sensitivity_summary_sha256") != sha256_file(summary_path):
            raise ValueError("influence provenance is not bound to the summary")
        if provenance.get("convergence", {}).get("converged") is not True:
            raise ValueError("influence provenance does not pass convergence")
        if not _all_true("sensitivity_converged"):
            raise ValueError("sensitivity summary does not pass convergence")
        if not _all_true("median_direction_stable"):
            raise ValueError("one or more growth medians changed direction")
        if not _all_true("intervals_overlap"):
            raise ValueError("one or more primary and sensitivity intervals do not overlap")

        result.update(
            ready=True,
            reason="trace-bound row-exclusion sensitivity passed",
            max_median_shift=float(
                pd.to_numeric(summary["median_shift"], errors="raise").abs().max()
            ),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        result["reason"] = str(exc)
    return result
