# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Denominator-free participant Bayesian bootstrap for Byrne growth.

This module supports a non-registered robustness analysis for the three Byrne
measures whose score definitions remain unresolved. It puts independent
Dirichlet(1, ..., 1) weights on the retained participants within each reading
group, then computes weighted paired raw-score changes. The same participant
weights are reused across intervals within a draw; extension-wave estimates
renormalise them over children observed at both endpoints.

The result is a posterior over the empirical participant distribution, not a
latent trajectory model and not evidence about an instrument ceiling. It can
therefore test the Phase-A descriptive growth read-out without introducing a
denominator, but it cannot repair denominator-dependent measurement, adjusted,
or horseshoe models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.preprocessing import (
    LongitudinalPanel,
)
from language_reading_predictors.statistical_models.rlm_sensitivity_contract import (
    MAX_MEDIAN_RANGE_FRACTION,
)

BOOTSTRAP_VARIANT = "participant_bayesian_bootstrap"
MONTE_CARLO_TOLERANCE_FRACTION = 0.005

_KEY_COLUMNS = ["quantity", "label", "readgrp_label"]
_QUANTILE_COLUMNS = ("q_lo", "q50", "q_hi")


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _summarise(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap draws must be a non-empty finite vector")
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "q_lo": float(np.quantile(values, 0.055)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "q_hi": float(np.quantile(values, 0.945)),
        "p_gt_0": float(np.mean(values > 0)),
    }


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "readgrp_label" in out:
        out["readgrp_label"] = out["readgrp_label"].fillna("")
    return out


def _supported_waves(
    panel: LongitudinalPanel,
    *,
    measure: str,
    group_label: str,
) -> list[int]:
    part = panel.long[
        (panel.long[panel.group_label_col] == group_label)
        & panel.long[measure].notna()
    ]
    return sorted(int(wave) for wave in part[panel.dataset.wave_col].unique())


def _common_waves(panel: LongitudinalPanel, *, measure: str) -> list[int]:
    common = set(panel.all_waves)
    for label in panel.group_labels:
        common &= set(
            _supported_waves(panel, measure=measure, group_label=label)
        )
    return sorted(common)


def _weighted_interval(
    weights: np.ndarray,
    wide: pd.DataFrame,
    *,
    start: int,
    end: int,
) -> tuple[np.ndarray, int]:
    start_values = wide[start].to_numpy(dtype=float)
    end_values = wide[end].to_numpy(dtype=float)
    paired = np.isfinite(start_values) & np.isfinite(end_values)
    n_subjects = int(paired.sum())
    if n_subjects == 0:
        raise ValueError(f"No participants observed at waves {start} and {end}")
    paired_weights = weights[:, paired]
    denominator = paired_weights.sum(axis=1)
    if np.any(denominator <= 0):
        raise RuntimeError("Bayesian-bootstrap paired weights sum to zero")
    change = end_values[paired] - start_values[paired]
    return (paired_weights @ change) / denominator, n_subjects


def participant_bayesian_bootstrap_growth(
    panel: LongitudinalPanel,
    *,
    measure: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Return posterior summaries of paired raw-score growth.

    Independent participant-weight vectors are drawn within each reading group.
    A group's vector is shared by all its interval calculations in a draw, which
    preserves the longitudinal dependence between reported quantities. Missing
    extension-wave endpoints are handled by renormalising over the paired subset,
    matching the registered historical-growth reporting population.
    """

    draws = _positive_integer(draws, name="draws")
    seed = _positive_integer(seed, name="seed")
    if measure not in panel.measures:
        raise KeyError(f"measure {measure!r} not in panel {panel.measures!r}")

    subject_col = panel.dataset.subject_col
    wave_col = panel.dataset.wave_col
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    total_growth: dict[str, np.ndarray] = {}
    common = _common_waves(panel, measure=measure)

    for label in panel.group_labels:
        part = panel.long[panel.long[panel.group_label_col] == label]
        wide = part.pivot(index=subject_col, columns=wave_col, values=measure)
        wide = wide.reindex(columns=list(panel.all_waves))
        if wide.index.has_duplicates:
            raise ValueError(f"Duplicate participant rows for group {label!r}")
        n_group = len(wide)
        if n_group < 2:
            raise ValueError(
                f"Bayesian bootstrap requires at least two participants in {label!r}"
            )
        weights = rng.dirichlet(np.ones(n_group), size=draws)
        waves = _supported_waves(panel, measure=measure, group_label=label)
        intervals = [
            (waves[index], waves[index + 1])
            for index in range(len(waves) - 1)
        ]
        if len(waves) > 2:
            intervals.append((waves[0], waves[-1]))

        interval_draws: dict[tuple[int, int], np.ndarray] = {}
        for start, end in intervals:
            values, n_subjects = _weighted_interval(
                weights,
                wide,
                start=start,
                end=end,
            )
            interval_draws[(start, end)] = values
            rows.append(
                {
                    "measure": measure,
                    "variant": BOOTSTRAP_VARIANT,
                    "likelihood": "none",
                    "denominator": pd.NA,
                    "quantity": f"growth_{start}_{end}_items",
                    "label": f"Wave {start} to wave {end}",
                    "readgrp_label": label,
                    "window": (
                        "core"
                        if start in panel.waves and end in panel.waves
                        else "extension"
                    ),
                    "n_subjects": n_subjects,
                    "bootstrap_draws": draws,
                    "bootstrap_seed": seed,
                    **_summarise(values),
                }
            )

        if len(common) >= 2:
            start, end = common[0], common[-1]
            total_growth[label], _n_subjects = _weighted_interval(
                weights,
                wide,
                start=start,
                end=end,
            )

    if len(common) >= 2:
        start, end = common[0], common[-1]
        for first in range(len(panel.group_labels)):
            for second in range(first + 1, len(panel.group_labels)):
                a = panel.group_labels[first]
                b = panel.group_labels[second]
                values = total_growth[b] - total_growth[a]
                rows.append(
                    {
                        "measure": measure,
                        "variant": BOOTSTRAP_VARIANT,
                        "likelihood": "none",
                        "denominator": pd.NA,
                        "quantity": f"total_growth_{b}_minus_{a}",
                        "label": (
                            f"Total growth (waves {start}-{end}): {b} minus {a}"
                        ),
                        "readgrp_label": "",
                        "window": (
                            "core"
                            if start in panel.waves and end in panel.waves
                            else "extension"
                        ),
                        "n_subjects": pd.NA,
                        "bootstrap_draws": draws,
                        "bootstrap_seed": seed,
                        **_summarise(values),
                    }
                )

    return pd.DataFrame(rows)


def monte_carlo_stability(
    primary: pd.DataFrame,
    replicate: pd.DataFrame,
    *,
    observed_maximum: int,
    tolerance_fraction: float = MONTE_CARLO_TOLERANCE_FRACTION,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare independent bootstrap simulations and apply a numerical gate."""

    observed_maximum = _positive_integer(
        observed_maximum, name="observed_maximum"
    )
    if not np.isfinite(tolerance_fraction) or tolerance_fraction <= 0:
        raise ValueError("tolerance_fraction must be positive and finite")
    required = {*_KEY_COLUMNS, *_QUANTILE_COLUMNS}
    for name, frame in (("primary", primary), ("replicate", replicate)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} table missing columns: {missing}")
        if frame.duplicated(_KEY_COLUMNS).any():
            raise ValueError(f"{name} table has duplicate estimand rows")

    left = _normalise_keys(primary)[[*_KEY_COLUMNS, *_QUANTILE_COLUMNS]]
    right = _normalise_keys(replicate)[[*_KEY_COLUMNS, *_QUANTILE_COLUMNS]]
    merged = left.merge(
        right,
        on=_KEY_COLUMNS,
        how="outer",
        suffixes=("_primary", "_replicate"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("primary and replicate bootstrap estimands differ")
    merged = merged.drop(columns="_merge")
    difference_columns = []
    for column in _QUANTILE_COLUMNS:
        difference = f"absolute_{column}_difference"
        merged[difference] = (
            merged[f"{column}_primary"] - merged[f"{column}_replicate"]
        ).abs()
        difference_columns.append(difference)
    merged["maximum_quantile_difference"] = merged[difference_columns].max(axis=1)
    merged["maximum_quantile_difference_fraction_observed_max"] = (
        merged["maximum_quantile_difference"] / observed_maximum
    )
    maximum_fraction = float(
        merged["maximum_quantile_difference_fraction_observed_max"].max()
    )
    passed = maximum_fraction <= tolerance_fraction
    return merged, {
        "status": "pass" if passed else "no_go",
        "maximum_quantile_difference_fraction_observed_max": maximum_fraction,
        "maximum_allowed_fraction_observed_max": tolerance_fraction,
        "interpretation": (
            "Numerical reproducibility of two independent Bayesian-bootstrap "
            "simulations; this is not a scientific robustness verdict."
        ),
    }


def compare_bootstrap_with_likelihoods(
    bootstrap: pd.DataFrame,
    likelihood_growth: pd.DataFrame,
    *,
    observed_maximum: int,
    expected_likelihood_variants: tuple[str, ...],
    likelihood_reference_passed: bool,
    monte_carlo_passed: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply the pre-specified five-method empirical robustness rule."""

    observed_maximum = _positive_integer(
        observed_maximum, name="observed_maximum"
    )
    required = {"variant", *_KEY_COLUMNS, *_QUANTILE_COLUMNS}
    for name, frame in (
        ("bootstrap", bootstrap),
        ("likelihood_growth", likelihood_growth),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} table missing columns: {missing}")

    boot = _normalise_keys(bootstrap)
    reference = _normalise_keys(likelihood_growth)
    if set(boot["variant"]) != {BOOTSTRAP_VARIANT}:
        raise ValueError("bootstrap table must contain only the bootstrap variant")
    observed_variants = tuple(dict.fromkeys(reference["variant"].astype(str)))
    if set(observed_variants) != set(expected_likelihood_variants):
        raise ValueError(
            "likelihood variants do not match the pre-specified reference set"
        )
    if boot.duplicated(_KEY_COLUMNS).any():
        raise ValueError("bootstrap table has duplicate estimand rows")
    if reference.duplicated(["variant", *_KEY_COLUMNS]).any():
        raise ValueError("likelihood table has duplicate variant-estimand rows")

    boot_keys = set(map(tuple, boot[_KEY_COLUMNS].itertuples(index=False, name=None)))
    reference_keys = set(
        map(tuple, reference[_KEY_COLUMNS].itertuples(index=False, name=None))
    )
    if boot_keys != reference_keys:
        raise ValueError("bootstrap and likelihood estimand sets differ")

    rows: list[dict[str, object]] = []
    for key, boot_part in boot.groupby(_KEY_COLUMNS, dropna=False, sort=False):
        reference_part = reference[
            (reference["quantity"] == key[0])
            & (reference["label"] == key[1])
            & (reference["readgrp_label"] == key[2])
        ]
        if len(reference_part) != len(expected_likelihood_variants):
            raise ValueError(f"incomplete likelihood set for estimand {key!r}")
        combined = pd.concat([reference_part, boot_part], ignore_index=True)
        medians = combined["q50"].to_numpy(dtype=float)
        lower = combined["q_lo"].to_numpy(dtype=float)
        upper = combined["q_hi"].to_numpy(dtype=float)
        median_range = float(np.ptp(medians))
        all_nonnegative = bool(np.all(medians >= 0))
        all_nonpositive = bool(np.all(medians <= 0))
        baseline = reference_part[
            reference_part["variant"] == "beta_binomial_1x"
        ]
        if len(baseline) != 1:
            raise ValueError("reference lacks exactly one beta_binomial_1x row")
        bootstrap_median = float(boot_part.iloc[0]["q50"])
        baseline_median = float(baseline.iloc[0]["q50"])
        rows.append(
            {
                "quantity": key[0],
                "label": key[1],
                "readgrp_label": key[2],
                "n_methods": len(combined),
                "bootstrap_median": bootstrap_median,
                "beta_binomial_1x_median": baseline_median,
                "bootstrap_minus_beta_binomial_1x": (
                    bootstrap_median - baseline_median
                ),
                "absolute_bootstrap_difference_fraction_observed_max": (
                    abs(bootstrap_median - baseline_median) / observed_maximum
                ),
                "combined_median_min": float(np.min(medians)),
                "combined_median_max": float(np.max(medians)),
                "combined_median_range": median_range,
                "combined_median_range_fraction_observed_max": (
                    median_range / observed_maximum
                ),
                "combined_median_direction_stable": (
                    all_nonnegative or all_nonpositive
                ),
                "combined_89_interval_overlap": bool(
                    float(np.max(lower)) <= float(np.min(upper))
                ),
            }
        )

    comparison = pd.DataFrame(rows)
    direction_stable = bool(comparison["combined_median_direction_stable"].all())
    intervals_overlap = bool(comparison["combined_89_interval_overlap"].all())
    maximum_fraction = float(
        comparison["combined_median_range_fraction_observed_max"].max()
    )
    passed = bool(
        likelihood_reference_passed
        and monte_carlo_passed
        and direction_stable
        and intervals_overlap
        and maximum_fraction <= MAX_MEDIAN_RANGE_FRACTION
    )
    return comparison, {
        "status": "pass" if passed else "no_go",
        "likelihood_reference_passed": bool(likelihood_reference_passed),
        "monte_carlo_stability_passed": bool(monte_carlo_passed),
        "all_five_method_median_directions_stable": direction_stable,
        "all_five_method_89_intervals_overlap": intervals_overlap,
        "maximum_five_method_median_range_fraction_observed_max": (
            maximum_fraction
        ),
        "maximum_allowed_median_range_fraction_observed_max": (
            MAX_MEDIAN_RANGE_FRACTION
        ),
        "interpretation": (
            "Empirical Phase-A growth robustness only. A pass does not identify "
            "an instrument ceiling, validate a score definition, repair other "
            "model families, or clear the publication gate."
        ),
    }
