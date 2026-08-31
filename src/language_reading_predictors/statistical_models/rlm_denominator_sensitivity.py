# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Likelihood sensitivity helpers for the three unresolved Byrne score ceilings.

This is deliberately not a registered model family.  It supports a traceable
stress test of the historical-growth estimands under three increasingly wide
Beta-Binomial denominators and a denominator-free Negative-Binomial count
likelihood.  The command-line harness is
``scripts/rlm_denominator_sensitivity.py``.

The stress-test denominators are mathematical perturbations of the observed
sample maximum, not claims about an administered test form.  Consequently this
module cannot confirm an instrument ceiling or make an existing fit publishable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
import pymc as pm

from language_reading_predictors.statistical_models.factories import (
    BuiltModel,
    build_historical_growth_model,
)
from language_reading_predictors.statistical_models.fitted_payloads import EmptyPayload
from language_reading_predictors.statistical_models.preprocessing import LongitudinalPanel
from language_reading_predictors.statistical_models.rlm_sensitivity_contract import (
    DENOMINATOR_FACTORS,
    MAX_MEDIAN_RANGE_FRACTION,
)
from language_reading_predictors.statistical_models import priors as _priors

SensitivityLikelihood = Literal["beta_binomial", "negative_binomial"]


@dataclass(frozen=True, slots=True)
class SensitivityVariant:
    """One pre-specified likelihood perturbation."""

    name: str
    likelihood: SensitivityLikelihood
    denominator: int | None
    denominator_factor: int | None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready description."""

        return {
            "name": self.name,
            "likelihood": self.likelihood,
            "denominator": self.denominator,
            "denominator_factor": self.denominator_factor,
        }


def denominator_grid(observed_maximum: int) -> tuple[int, ...]:
    """Return the fixed 1x/2x/4x stress grid for an observed integer maximum."""

    if isinstance(observed_maximum, bool) or not isinstance(observed_maximum, int):
        raise TypeError("observed_maximum must be a positive integer")
    if observed_maximum <= 0:
        raise ValueError("observed_maximum must be a positive integer")
    return tuple(observed_maximum * factor for factor in DENOMINATOR_FACTORS)


def sensitivity_variants(observed_maximum: int) -> tuple[SensitivityVariant, ...]:
    """Return the complete, ordered sensitivity set for one measure."""

    bounded = tuple(
        SensitivityVariant(
            name=f"beta_binomial_{factor}x",
            likelihood="beta_binomial",
            denominator=denominator,
            denominator_factor=factor,
        )
        for factor, denominator in zip(
            DENOMINATOR_FACTORS,
            denominator_grid(observed_maximum),
            strict=True,
        )
    )
    return (
        *bounded,
        SensitivityVariant(
            name="negative_binomial",
            likelihood="negative_binomial",
            denominator=None,
            denominator_factor=None,
        ),
    )


def panel_with_denominator(
    panel: LongitudinalPanel,
    *,
    measure: str,
    denominator: int,
) -> LongitudinalPanel:
    """Return a shallow panel copy carrying one stress-test denominator."""

    if measure not in panel.measures:
        raise KeyError(f"measure {measure!r} not in panel {panel.measures!r}")
    observed = pd.to_numeric(panel.long[measure], errors="coerce").dropna()
    observed_maximum = int(observed.max())
    if denominator < observed_maximum:
        raise ValueError(
            f"denominator {denominator} is below observed {measure} maximum "
            f"{observed_maximum}"
        )
    return replace(panel, n_trials={**panel.n_trials, measure: int(denominator)})


def _historical_indices(
    panel: LongitudinalPanel,
    measure: str,
) -> dict[str, Any]:
    """Resolve the common historical-growth indexing contract."""

    df = panel.long
    dataset = panel.dataset
    subject_col = dataset.subject_col
    wave_col = dataset.wave_col
    group_col = dataset.group_col
    group_index = {code: index for index, code in enumerate(panel.group_codes)}
    subject_index = {
        subject: index for index, subject in enumerate(panel.subject_ids)
    }
    cells = panel.cells(measure)
    cell_index = {cell: index for index, cell in enumerate(cells)}
    group_idx = df[group_col].map(group_index).to_numpy(dtype=int)
    obs_cell_idx = np.asarray(
        [
            cell_index[(int(group), int(wave))]
            for group, wave in zip(df[group_col], df[wave_col], strict=True)
        ],
        dtype=int,
    )
    subject_idx = df[subject_col].map(subject_index).to_numpy(dtype=int)
    subject_group = (
        df.drop_duplicates(subject_col)
        .set_index(subject_col)
        .loc[panel.subject_ids, group_col]
        .map(group_index)
        .to_numpy(dtype=int)
    )
    common_waves = [
        wave
        for wave in sorted({wave for _group, wave in cells})
        if all((group, wave) in cell_index for group in panel.group_codes)
    ]
    return {
        "cells": cells,
        "cell_index": cell_index,
        "cell_labels": [
            f"{panel.group_labels[group_index[group]]} | wave {wave}"
            for group, wave in cells
        ],
        "group_idx": group_idx,
        "obs_cell_idx": obs_cell_idx,
        "subject_idx": subject_idx,
        "subject_group": subject_group,
        "common_waves": common_waves,
        "observed": df[measure].to_numpy(dtype=int),
    }


def build_negative_binomial_historical_growth_model(
    panel: LongitudinalPanel,
    *,
    measure: str,
    eta_prior_mu: float = math.log(10.0),
    eta_prior_sigma: float = 1.25,
    sigma_subject_prior_sigma: float = 0.75,
    alpha_prior_sigma: float = 20.0,
) -> BuiltModel[EmptyPayload]:
    """Build the denominator-free count-likelihood sensitivity model.

    The mean structure matches the registered historical-growth family, but the
    log link replaces its bounded-score logit and the observation model is a
    Negative Binomial.  This treats scores as non-negative counts and permits
    overdispersion; it deliberately does not pretend that the scale is unbounded
    in reality.  Posterior-predictive tail behaviour therefore remains a required
    diagnostic rather than a reason to promote this prototype automatically.
    """

    if measure not in panel.measures:
        raise KeyError(f"measure {measure!r} not in panel {panel.measures!r}")
    for name, value in (
        ("eta_prior_mu", eta_prior_mu),
        ("eta_prior_sigma", eta_prior_sigma),
        ("sigma_subject_prior_sigma", sigma_subject_prior_sigma),
        ("alpha_prior_sigma", alpha_prior_sigma),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if eta_prior_sigma <= 0 or sigma_subject_prior_sigma <= 0 or alpha_prior_sigma <= 0:
        raise ValueError("all scale parameters must be positive")

    index = _historical_indices(panel, measure)
    n_groups = len(panel.group_codes)
    coords = {
        "group": panel.group_labels,
        "cell": index["cell_labels"],
        "subject": [str(subject) for subject in panel.subject_ids],
        "obs": np.arange(len(panel.long)),
    }

    def cell_positions(wave: int) -> list[int]:
        return [
            index["cell_index"][(group, wave)] for group in panel.group_codes
        ]

    with pm.Model(coords=coords) as model:
        eta_cell = _priors.declare(
                       pm.Normal(
                                   "eta_cell",
                                   mu=eta_prior_mu,
                                   sigma=eta_prior_sigma,
                                   dims="cell",
                               ),
                       role="nuisance",
                       rationale=(
                           "Group-by-wave population level per cell/measure on the logit scale "
                           "(Normal(0, 1.5)); the fitted cells (mean_items) and growth "
                           "intervals are deterministics of it — descriptive, not a treatment "
                           "effect."
                       ),
                   )
        sigma_subject = _priors.declare(
                            pm.HalfNormal(
                                        "sigma_subject",
                                        sigma=sigma_subject_prior_sigma,
                                        dims="group",
                                    ),
                            role="nuisance",
                            rationale=(
                                "Group-indexed between-subject random-intercept SD (HalfNormal(1)); "
                                "between-child heterogeneity that differs by cohort group."
                            ),
                        )
        z_subject = _priors.declare(
                        pm.Normal("z_subject", mu=0.0, sigma=1.0, dims="subject"),
                        role="nuisance",
                        rationale=(
                            "Non-centred standard-normal per-subject offsets (Normal(0, 1)); "
                            "group-centred and scaled by sigma_subject to form the subject "
                            "random effects."
                        ),
                    )
        z_group_mean = pm.math.stack(
            [
                z_subject[index["subject_group"] == group].mean()
                for group in range(n_groups)
            ]
        )
        subject_offset = pm.Deterministic(
            "subject_offset",
            (
                z_subject
                - z_group_mean[index["subject_group"]]
            )
            * sigma_subject[index["subject_group"]],
            dims="subject",
        )
        alpha = pm.HalfNormal("alpha", sigma=alpha_prior_sigma, dims="group")

        eta_obs = (
            eta_cell[index["obs_cell_idx"]]
            + subject_offset[index["subject_idx"]]
        )
        fitted_mean = pm.math.exp(eta_obs)
        pm.NegativeBinomial(
            "score",
            mu=fitted_mean,
            alpha=alpha[index["group_idx"]],
            observed=index["observed"],
            dims="obs",
        )
        pm.Deterministic("fitted_mean_items_obs", fitted_mean, dims="obs")
        mean_items = pm.Deterministic(
            "mean_items", pm.math.exp(eta_cell), dims="cell"
        )

        common_waves = index["common_waves"]
        if len(common_waves) >= 2:
            pm.Deterministic(
                "growth_first_next_items",
                mean_items[cell_positions(common_waves[1])]
                - mean_items[cell_positions(common_waves[0])],
                dims="group",
            )
            pm.Deterministic(
                "growth_first_last_items",
                mean_items[cell_positions(common_waves[-1])]
                - mean_items[cell_positions(common_waves[0])],
                dims="group",
            )
        if len(common_waves) >= 3:
            pm.Deterministic(
                "growth_next_last_items",
                mean_items[cell_positions(common_waves[-1])]
                - mean_items[cell_positions(common_waves[1])],
                dims="group",
            )

    return BuiltModel(model=model, prepared=panel, payload=EmptyPayload())


def build_sensitivity_model(
    panel: LongitudinalPanel,
    *,
    measure: str,
    variant: SensitivityVariant,
) -> BuiltModel[EmptyPayload]:
    """Build one pre-specified sensitivity variant."""

    if variant.likelihood == "negative_binomial":
        if variant.denominator is not None:
            raise ValueError("negative_binomial must not declare a denominator")
        return build_negative_binomial_historical_growth_model(
            panel,
            measure=measure,
        )
    if variant.denominator is None:
        raise ValueError("beta_binomial requires a denominator")
    bounded_panel = panel_with_denominator(
        panel,
        measure=measure,
        denominator=variant.denominator,
    )
    return build_historical_growth_model(
        bounded_panel,
        measure=measure,
        eta_prior_sigma=1.5,
        sigma_subject_prior_sigma=1.0,
        # Matches the registered specs' reviewed dispersion-scale prior
        # (2026-08-21 review, finding 8) so this sensitivity varies the
        # denominator alone.
        dispersion_prior_sigma=0.25,
    )


def aggregate_sensitivity(
    growth: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    observed_maximum: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare variants and apply the pre-specified empirical robustness rule."""

    required_growth = {
        "variant",
        "quantity",
        "label",
        "readgrp_label",
        "q_lo",
        "q50",
        "q_hi",
    }
    missing = sorted(required_growth - set(growth.columns))
    if missing:
        raise ValueError(f"growth table missing columns: {missing}")
    if growth.empty:
        raise ValueError("growth table is empty")
    if "converged" not in diagnostics or diagnostics.empty:
        raise ValueError("diagnostics must contain at least one convergence verdict")

    rows: list[dict[str, object]] = []
    key_columns = ["quantity", "label", "readgrp_label"]
    for key, part in growth.groupby(key_columns, dropna=False, sort=False):
        medians = part["q50"].to_numpy(dtype=float)
        lower = part["q_lo"].to_numpy(dtype=float)
        upper = part["q_hi"].to_numpy(dtype=float)
        median_range = float(np.ptp(medians))
        all_nonnegative = bool(np.all(medians >= 0))
        all_nonpositive = bool(np.all(medians <= 0))
        rows.append(
            {
                "quantity": key[0],
                "label": key[1],
                "readgrp_label": key[2],
                "n_variants": int(len(part)),
                "median_min": float(np.min(medians)),
                "median_max": float(np.max(medians)),
                "median_range": median_range,
                "median_range_fraction_observed_max": (
                    median_range / observed_maximum
                ),
                "median_direction_stable": all_nonnegative or all_nonpositive,
                "joint_89_interval_overlap": bool(
                    float(np.max(lower)) <= float(np.min(upper))
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    all_converged = bool(diagnostics["converged"].eq(True).all())  # noqa: E712
    direction_stable = bool(comparison["median_direction_stable"].all())
    intervals_overlap = bool(comparison["joint_89_interval_overlap"].all())
    maximum_fraction = float(
        comparison["median_range_fraction_observed_max"].max()
    )
    passed = bool(
        all_converged
        and direction_stable
        and intervals_overlap
        and maximum_fraction <= MAX_MEDIAN_RANGE_FRACTION
    )
    decision = {
        "status": "pass" if passed else "no_go",
        "all_variants_converged": all_converged,
        "all_median_directions_stable": direction_stable,
        "all_joint_89_intervals_overlap": intervals_overlap,
        "maximum_median_range_fraction_observed_max": maximum_fraction,
        "maximum_allowed_median_range_fraction_observed_max": (
            MAX_MEDIAN_RANGE_FRACTION
        ),
        "interpretation": (
            "Empirical likelihood robustness only. A pass does not identify an "
            "instrument ceiling and does not clear the publication gate."
        ),
    }
    return comparison, decision
