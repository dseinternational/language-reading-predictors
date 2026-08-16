# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for the latent growth-curve family.

Mirrors the ITT / gain-factor / level-factor / DiD / concurrent / aligned run-plan
pattern for the joint multivariate latent growth-curve (``kind="growth"``) models. A
model module declares its settings; the plan is resolved and **validated before any
data are loaded or an output directory is reset**, then drives data preparation,
factory construction and the ``config.json`` / ``model_recipe.md`` audit trail. This
removes the untyped ``spec.extra`` boundary (where a misspelled key silently
defaulted) and records the resolved design, estimand, causal status, analysis
population and missing-data assumption alongside every fit.

The growth design characterises one or more measures' within-child trajectories
(linear in standardised age) and asks whether a **baseline** ability proxy predicts
trajectory shape: ``gamma`` on the growth *rate* and ``delta`` on the level at mean
age. The RLI models use four trial waves and a multivariate outcome; the Byrne/RLM
port uses its paper-compatible first three waves and BAS word reading. Every term is
an **adjusted, latent-general-ability-confounded association**, never causal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.context import ModelSpec

# The complete, closed set of legacy ``spec.extra`` keys the growth family
# understands. Anything else is a typo and must fail before a fit starts.
_LEGACY_KEYS = frozenset(
    {
        "outcomes",
        "baseline_covariate",
        "use_shared_factor",
        "use_random_slope",
        "age_ability_interaction",
        "waves",
        "baseline_scale",
        "min_outcome_waves",
        "adjust_for_group",
        "observation_influence_sensitivity",
        # Sampler knob, not a model setting: ``target_accept`` is resolved centrally by
        # ``context.make_context`` (CLI override > spec default > preset) and is never
        # read by this family's settings. Listed so a legitimate per-model declaration
        # is not rejected as a misspelling by the strict unknown-key check.
        "target_accept",
    }
)

_DEFAULT_OUTCOMES = ("R", "E", "T", "W", "L")


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} must contain non-empty strings, got {item!r}")
    return out


@dataclass(frozen=True, slots=True)
class GrowthModelSettings:
    """Immutable settings declared by a single latent growth-curve model module.

    Defaults encode the primary five-measure growth-curve read on non-verbal ability
    (``blocks``), without the shared growth-tempo factor or the age x ability term.
    """

    outcomes: tuple[str, ...] = _DEFAULT_OUTCOMES
    baseline_covariate: str = "blocks"
    use_shared_factor: bool = False
    use_random_slope: bool = True
    age_ability_interaction: bool = False
    waves: tuple[int, ...] | None = None
    baseline_scale: Literal["raw", "logit_safe"] = "raw"
    min_outcome_waves: int = 1
    adjust_for_group: bool = False
    observation_influence_sensitivity: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcomes", _tuple_of_strings(self.outcomes, name="outcomes")
        )
        if not self.outcomes:
            raise ValueError("outcomes must list at least one measure")
        if not isinstance(self.baseline_covariate, str) or not self.baseline_covariate:
            raise TypeError("baseline_covariate must be a non-empty string")
        if self.waves is not None:
            if isinstance(self.waves, (str, bytes)) or not hasattr(
                self.waves, "__iter__"
            ):
                raise TypeError("waves must be a sequence of integers or None")
            waves = tuple(self.waves)
            if len(waves) < 2:
                raise ValueError("waves must contain at least two assessment waves")
            if any(isinstance(wave, bool) or not isinstance(wave, int) for wave in waves):
                raise TypeError("waves must contain integers")
            if tuple(sorted(set(waves))) != waves:
                raise ValueError("waves must be unique and strictly increasing")
            object.__setattr__(self, "waves", waves)
        if self.baseline_scale not in {"raw", "logit_safe"}:
            raise ValueError("baseline_scale must be 'raw' or 'logit_safe'")
        if (
            isinstance(self.min_outcome_waves, bool)
            or not isinstance(self.min_outcome_waves, int)
            or self.min_outcome_waves < 1
        ):
            raise TypeError("min_outcome_waves must be a positive integer")
        for flag in (
            "use_shared_factor",
            "use_random_slope",
            "age_ability_interaction",
            "adjust_for_group",
            "observation_influence_sensitivity",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be bool")

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> GrowthModelSettings:
        """Strictly translate the former ``spec.extra`` dictionary boundary.

        Rejects unknown keys so a misspelling fails before data loading rather than
        silently taking a default."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown growth setting(s): {', '.join(unknown)}. "
                "Declare GrowthModelSettings so misspellings fail fast."
            )
        # Pass raw values through so __post_init__ is the single validation/coercion
        # point; pre-coercing here would silently reshape misshaped legacy settings.
        return cls(
            outcomes=extra.get("outcomes", _DEFAULT_OUTCOMES),
            baseline_covariate=extra.get("baseline_covariate", "blocks"),
            use_shared_factor=extra.get("use_shared_factor", False),
            use_random_slope=extra.get("use_random_slope", True),
            age_ability_interaction=extra.get("age_ability_interaction", False),
            waves=extra.get("waves"),
            baseline_scale=extra.get("baseline_scale", "raw"),
            min_outcome_waves=extra.get("min_outcome_waves", 1),
            adjust_for_group=extra.get("adjust_for_group", False),
            observation_influence_sensitivity=extra.get(
                "observation_influence_sensitivity", False
            ),
        )


@dataclass(frozen=True, slots=True)
class GrowthRunPlan:
    """Concrete, validated instructions consumed by preparation and modelling."""

    model_id: str
    settings_source: str
    study_id: Literal["rli", "rlm"]
    outcomes: tuple[str, ...]
    baseline_covariate: str
    use_shared_factor: bool
    use_random_slope: bool
    age_ability_interaction: bool
    waves: tuple[int, ...]
    baseline_scale: Literal["raw", "logit_safe"]
    min_outcome_waves: int
    adjust_for_group: bool
    observation_influence_sensitivity: bool
    # Recorded audit metadata (#394 pillar 4).
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_wave_panel`` from the resolved plan.

        The growth family loads the wave panel of its measures with the single
        t1-only baseline covariate (non-verbal ability)."""
        return {
            "outcomes": self.outcomes,
            "baseline_covariates": (self.baseline_covariate,),
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_growth_model`` for this plan."""
        return {
            "baseline_covariate": self.baseline_covariate,
            "use_shared_factor": self.use_shared_factor,
            "use_random_slope": self.use_random_slope,
            "age_ability_interaction": self.age_ability_interaction,
            "adjust_for_group": self.adjust_for_group,
        }

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        outcomes = ", ".join(self.outcomes)
        return (
            "Note: Generated from the validated latent growth-curve run plan; "
            "template drafted by an LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Measures: {outcomes}. Baseline covariate: `{self.baseline_covariate}`. "
            f"Assessment waves: {', '.join(map(str, self.waves))}. Baseline scale: "
            f"{self.baseline_scale}. Minimum observed outcome waves per child: "
            f"{self.min_outcome_waves}. Reading-group nuisance trajectories: "
            f"{self.adjust_for_group}. "
            "High-Pareto observation-cell sensitivity: "
            f"{self.observation_influence_sensitivity}. "
            f"Child random slope: {self.use_random_slope}. Shared growth-tempo "
            f"factor: {self.use_shared_factor}. Age x ability "
            f"interaction: {self.age_ability_interaction}.\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate and posterior-predictive checks pass. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_growth_settings(spec: ModelSpec) -> tuple[GrowthModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: growth settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, GrowthModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='growth' requires GrowthModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return (
        GrowthModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def resolve_growth_run_plan(spec: ModelSpec) -> GrowthRunPlan:
    """Resolve and validate a latent growth-curve spec before any data are loaded."""
    if spec.kind != "growth":
        raise ValueError(f"{spec.model_id}: expected kind 'growth', got {spec.kind!r}")

    settings, source = declared_growth_settings(spec)

    if spec.study_id not in {"rli", "rlm"}:
        raise ValueError(
            f"{spec.model_id}: growth supports study_id 'rli' or 'rlm', got "
            f"{spec.study_id!r}"
        )
    study_id: Literal["rli", "rlm"] = spec.study_id
    waves = settings.waves or ((1, 2, 3, 4) if study_id == "rli" else (1, 2, 3))

    if settings.min_outcome_waves > len(waves):
        raise ValueError(
            f"{spec.model_id}: min_outcome_waves cannot exceed the number of waves"
        )

    if study_id == "rli":
        if waves != (1, 2, 3, 4):
            raise ValueError(
                f"{spec.model_id}: the RLI growth port requires waves (1, 2, 3, 4)"
            )
        if settings.baseline_scale != "raw":
            raise ValueError(
                f"{spec.model_id}: the RLI growth port requires baseline_scale='raw'"
            )
        if settings.min_outcome_waves != 1:
            raise ValueError(
                f"{spec.model_id}: the RLI growth port requires min_outcome_waves=1"
            )
        if settings.adjust_for_group:
            raise ValueError(
                f"{spec.model_id}: the RLI growth port does not use group nuisance "
                "trajectories"
            )
    else:
        from language_reading_predictors.statistical_models.datasets import (
            resolve_dataset,
        )

        _dataset, catalogue = resolve_dataset("rlm")
        requested = (*settings.outcomes, settings.baseline_covariate)
        unknown = sorted(set(requested) - set(catalogue))
        if unknown:
            raise ValueError(
                f"{spec.model_id}: unregistered RLM measure(s): {', '.join(unknown)}"
            )
        unresolved = [
            symbol
            for symbol in requested
            if not catalogue[symbol].n_trials_confirmed
            or not catalogue[symbol].instrument_identity_confirmed
        ]
        if unresolved:
            raise ValueError(
                f"{spec.model_id}: RLM growth requires confirmed denominators and "
                f"instrument identities; unresolved: {', '.join(dict.fromkeys(unresolved))}"
            )
        if settings.baseline_covariate in settings.outcomes:
            raise ValueError(
                f"{spec.model_id}: the baseline ability measure cannot also be a "
                "trajectory outcome"
            )
        if waves != (1, 2, 3):
            raise ValueError(
                f"{spec.model_id}: the primary Byrne growth port requires the "
                "paper-compatible waves (1, 2, 3)"
            )
        if settings.baseline_scale != "logit_safe":
            raise ValueError(
                f"{spec.model_id}: bounded RLM baseline measures require "
                "baseline_scale='logit_safe'"
            )
        if settings.min_outcome_waves < 2:
            raise ValueError(
                f"{spec.model_id}: RLM growth requires at least two observed outcome "
                "waves per child"
            )
        if not settings.adjust_for_group:
            raise ValueError(
                f"{spec.model_id}: pooled RLM growth requires reading-group nuisance "
                "trajectories"
            )
        if settings.use_shared_factor and len(settings.outcomes) < 2:
            raise ValueError(
                f"{spec.model_id}: a shared growth-tempo factor needs at least two "
                "trajectory outcomes"
            )
        if settings.use_random_slope:
            raise ValueError(
                f"{spec.model_id}: the three-wave single-outcome Byrne port uses "
                "random intercepts only; a child random slope is not supported by "
                "this panel"
            )

    if study_id == "rli":
        design = (
            "Joint multivariate latent growth-curve model on the logit scale: each "
            "measure's within-child trajectory across the four waves (linear in "
            "standardised age), with a per-measure child-level random intercept and "
            "slope; optionally a rank-1 shared growth-tempo factor coupling the slopes."
        )
        estimand = (
            "gamma (baseline non-verbal ability -> growth RATE) is the headline Q5 "
            "association; delta is the association with baseline LEVEL. Both are "
            "adjusted, latent-general-ability-confounded associations, never causal "
            "(block design is an off-DAG ability proxy)."
        )
        causal_status = (
            "Associational only: every non-randomised term is a latent-general-"
            "ability-confounded adjusted association under the locked DAG, never a "
            "causal effect."
        )
        analysis_population = (
            "Available-case children with wave-panel trajectories (about 54). The "
            "associations describe this observed cohort, not a randomised contrast."
        )
    else:
        design = (
            "Single-outcome latent growth curve over the Byrne paper's waves 1-3, "
            "linear in standardised age. Reading-group-specific nuisance intercepts "
            "and slopes absorb the three cohorts' different trajectories; the "
            "baseline-ability associations are shared across groups."
        )
        estimand = (
            "gamma is the common within-reading-group association between a 1-SD "
            "higher wave-1 BAS similarities score and the BAS word-reading growth "
            "rate; delta is its association with reading level at the pooled mean age."
        )
        causal_status = (
            "Descriptive association only: reading group and baseline verbal reasoning "
            "were not randomised, latent general ability remains incompletely measured, "
            "and the reading-matched cohort was selected on the outcome."
        )
        analysis_population = (
            "Children in the three historical reading groups with wave-1 BAS "
            "similarities observed and BAS word reading observed at two or more of "
            "waves 1-3."
        )
    missing_data_assumption = (
        "Masked outcome cells are assumed ignorable given the modelled trajectory "
        "and baseline covariate."
        if study_id == "rli"
        else "Available-case baseline selection plus masked outcome cells under "
        "ignorable missingness: retained missing wave scores are assumed ignorable "
        "given the modelled trajectory, reading group and baseline covariate."
    )

    return GrowthRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=study_id,
        outcomes=settings.outcomes,
        baseline_covariate=settings.baseline_covariate,
        use_shared_factor=settings.use_shared_factor,
        use_random_slope=settings.use_random_slope,
        age_ability_interaction=settings.age_ability_interaction,
        waves=waves,
        baseline_scale=settings.baseline_scale,
        min_outcome_waves=settings.min_outcome_waves,
        adjust_for_group=settings.adjust_for_group,
        observation_influence_sensitivity=settings.observation_influence_sensitivity,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )


def growth_observation_index(panel: Any) -> pd.DataFrame:
    """Map the growth likelihood's flattened cells back to child, wave and outcome.

    :func:`build_growth_model` flattens the ``(child, wave, outcome)`` mask in
    NumPy C order. PSIS-LOO returns one Pareto-k value in that same order. Keeping
    this mapping in the family layer makes the diagnostic auditable without
    mistaking one child-wave cell for a whole-child leave-out unit.
    """
    mask = np.stack([panel.obs_mask[symbol] for symbol in panel.outcomes], axis=2)
    child_idx, wave_idx, outcome_idx = np.nonzero(mask)
    outcomes = np.asarray(panel.outcomes, dtype=object)
    return pd.DataFrame(
        {
            "observation_index": np.arange(len(child_idx), dtype=int),
            "child_index": child_idx.astype(int),
            "subject_id": np.asarray(panel.subject_ids)[child_idx],
            "wave": np.asarray(panel.waves)[wave_idx].astype(int),
            "outcome": outcomes[outcome_idx],
        }
    )


def growth_pareto_table(panel: Any, loo: Any) -> pd.DataFrame:
    """Return mapped pointwise Pareto-k diagnostics for a growth likelihood."""
    if loo is None or getattr(loo, "pareto_k", None) is None:
        raise ValueError("growth influence sensitivity requires pointwise PSIS-LOO")
    mapping = growth_observation_index(panel)
    pareto_k = np.asarray(loo.pareto_k, dtype=float).ravel()
    if len(mapping) != len(pareto_k):
        raise ValueError(
            "growth observation map does not align with pointwise Pareto-k values: "
            f"{len(mapping)} cells versus {len(pareto_k)} diagnostics"
        )
    threshold = float(getattr(loo, "good_k", 0.7) or 0.7)
    out = mapping.copy()
    out["pareto_k"] = pareto_k
    out["good_k_threshold"] = threshold
    out["loo_reliable"] = pareto_k <= threshold
    return out.sort_values("pareto_k", ascending=False).reset_index(drop=True)


def exclude_growth_observation_cells(panel: Any, observation_indices: Any) -> Any:
    """Return a panel with selected flattened likelihood cells masked out.

    A child stays in the sensitivity model through any unflagged wave. If every
    observed outcome for one child is flagged, that child is removed rather than
    retaining an unconstrained random intercept. This is a coefficient-stability
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

    def _integer(value: Any) -> int:
        return int(np.asarray(value).item())

    mapping = growth_observation_index(panel)
    cells = {
        _integer(row.observation_index): row
        for row in mapping.itertuples(index=False)
    }
    unknown = sorted(set(indices) - set(cells))
    if unknown:
        raise IndexError(f"growth observation index out of range: {unknown}")

    masks = {
        name: np.array(value, dtype=bool, copy=True)
        for name, value in panel.obs_mask.items()
    }
    counts = {
        name: np.array(value, dtype=float, copy=True)
        for name, value in panel.counts.items()
    }
    logits = {
        name: np.array(value, dtype=float, copy=True)
        for name, value in panel.logit.items()
    }
    wave_lookup = {int(wave): index for index, wave in enumerate(panel.waves)}
    for index in indices:
        cell = cells[int(index)]
        child = _integer(cell.child_index)
        wave = wave_lookup[_integer(cell.wave)]
        outcome = str(cell.outcome)
        masks[outcome][child, wave] = False
        counts[outcome][child, wave] = np.nan
        logits[outcome][child, wave] = np.nan

    retained = np.stack([masks[symbol] for symbol in panel.outcomes], axis=2)
    keep = retained.sum(axis=(1, 2)) > 0
    n_fully_excluded = int((~keep).sum())
    dropped_by_reason = dict(panel.dropped_by_reason)
    if n_fully_excluded:
        dropped_by_reason["all_observed_cells_high_pareto"] = n_fully_excluded

    def _slice_dict(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(value)[keep].copy() for name, value in values.items()
        }

    return replace(
        panel,
        subject_ids=np.asarray(panel.subject_ids)[keep].copy(),
        n_children=int(keep.sum()),
        counts=_slice_dict(counts),
        obs_mask=_slice_dict(masks),
        logit=_slice_dict(logits),
        age_months=np.asarray(panel.age_months)[keep].copy(),
        age_std=np.asarray(panel.age_std)[keep].copy(),
        dose=np.asarray(panel.dose)[keep].copy(),
        dose_std=np.asarray(panel.dose_std)[keep].copy(),
        baseline=_slice_dict(panel.baseline),
        baseline_raw=_slice_dict(panel.baseline_raw),
        group=(
            np.asarray(panel.group)[keep].copy()
            if panel.group is not None
            else None
        ),
        child_covariates=_slice_dict(panel.child_covariates),
        wave_covariates=_slice_dict(panel.wave_covariates),
        excluded_children=panel.excluded_children + n_fully_excluded,
        dropped_by_reason=dropped_by_reason,
    )


def growth_influence_summary(
    primary_trace: Any,
    sensitivity_trace: Any,
    *,
    excluded_cells: pd.DataFrame,
    sensitivity_converged: bool | None,
    n_fully_excluded_children: int = 0,
) -> pd.DataFrame:
    """Compare ``gamma`` and ``delta`` before and after flagged-cell exclusion.

    The two posteriors are separate fits, so their draws are not paired. The table
    therefore reports each marginal posterior and a shift in medians; it does not
    manufacture a posterior distribution for their difference.
    """
    required = {"subject_id", "wave", "outcome", "observation_index", "pareto_k"}
    if not required.issubset(excluded_cells.columns):
        missing = ", ".join(sorted(required - set(excluded_cells.columns)))
        raise ValueError(f"excluded_cells lacks required columns: {missing}")
    if excluded_cells.empty:
        raise ValueError(
            "growth influence summary requires at least one excluded cell"
        )
    if n_fully_excluded_children < 0:
        raise ValueError("n_fully_excluded_children cannot be negative")

    def _draws(trace: Any, coefficient: str, outcome: str) -> np.ndarray:
        try:
            values = trace.posterior[coefficient].sel(outcome=outcome)
        except (AttributeError, KeyError) as exc:
            raise KeyError(
                f"trace lacks {coefficient!r} for outcome {outcome!r}"
            ) from exc
        return np.asarray(values, dtype=float).ravel()

    outcomes = [
        str(value) for value in primary_trace.posterior["gamma"].outcome.values
    ]
    rows: list[dict[str, Any]] = []
    for coefficient in ("gamma", "delta"):
        for outcome in outcomes:
            primary = _draws(primary_trace, coefficient, outcome)
            sensitivity = _draws(sensitivity_trace, coefficient, outcome)
            primary_lo, primary_hi = np.quantile(primary, (0.055, 0.945))
            sensitivity_lo, sensitivity_hi = np.quantile(
                sensitivity, (0.055, 0.945)
            )
            primary_median = float(np.median(primary))
            sensitivity_median = float(np.median(sensitivity))
            rows.append(
                {
                    "coefficient": coefficient,
                    "outcome": outcome,
                    "primary_median": primary_median,
                    "primary_lo89": float(primary_lo),
                    "primary_hi89": float(primary_hi),
                    "primary_prob_positive": float(np.mean(primary > 0)),
                    "sensitivity_median": sensitivity_median,
                    "sensitivity_lo89": float(sensitivity_lo),
                    "sensitivity_hi89": float(sensitivity_hi),
                    "sensitivity_prob_positive": float(np.mean(sensitivity > 0)),
                    "median_shift": sensitivity_median - primary_median,
                    "median_direction_stable": bool(
                        np.sign(primary_median) == np.sign(sensitivity_median)
                    ),
                    "intervals_overlap": bool(
                        max(primary_lo, sensitivity_lo)
                        <= min(primary_hi, sensitivity_hi)
                    ),
                    "n_excluded_cells": int(len(excluded_cells)),
                    "n_excluded_children": int(
                        excluded_cells["subject_id"].nunique()
                    ),
                    "n_fully_excluded_children": int(n_fully_excluded_children),
                    "sensitivity_converged": sensitivity_converged,
                }
            )
    return pd.DataFrame(rows)
