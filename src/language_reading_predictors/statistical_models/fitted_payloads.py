# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed, realised payloads carried from model factories to fitted consumers.

The payloads contain only design values that a factory actually used and that a
downstream pipeline, report, influence analysis or exact refit must reuse. Keeping
these values typed and row-aligned prevents consumers from silently recomputing a
different standardisation, mask or HSGP boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from language_reading_predictors.statistical_models.likelihood import ScoreMeanLink
from language_reading_predictors.statistical_models.preprocessing import Standardiser


class FittedPayload:
    """Marker base for every payload accepted by ``BuiltModel``."""


@dataclass(frozen=True)
class EmptyPayload(FittedPayload):
    """A model whose downstream consumers need no extra realised design values."""


TreatmentModerator = tuple[str, np.ndarray]


@dataclass(frozen=True)
class IttPayload(FittedPayload):
    """Realised treatment-moderator design for a single-outcome ITT fit."""

    tau_interaction_moderators: tuple[TreatmentModerator, ...]
    score_mean_link: ScoreMeanLink


@dataclass(frozen=True)
class JointPayload(FittedPayload):
    """Fitted dependence and observation-unit metadata for a joint ITT fit."""

    joint_dependence: Literal[
        "residual_correlated", "factorised_outcome_marginals"
    ]
    loo_unit: Literal["child"]
    outcomes: tuple[str, ...]


@dataclass(frozen=True)
class JointMechanismPayload(FittedPayload):
    """Realised design metadata for a joint mechanism fit."""

    design: Literal["levels", "transition"]
    joint_dependence: Literal[
        "lkj_residual_within_wave", "lkj_child_intercept"
    ]
    likelihood: Literal["binomial", "beta_binomial"]
    loo_unit: Literal["child"]
    outcomes: tuple[str, ...]
    mechanism_symbol: str
    contrast: tuple[str, str]
    adjust_for: tuple[str, ...]


@dataclass(frozen=True)
class MechanismDesign:
    """Data-derived mechanism design quantities pinned for exact refits.

    A leave-one-out refit would otherwise rederive the exposure and moderator
    standardisation and the HSGP boundary from one fewer row. Replaying this
    payload keeps the basis weights and held-out density on the full fit's design.
    """

    mech_scaler: Standardiser
    hsgp_L: float | None
    moderator_scaler: Standardiser | None = None

    def require_moderator_scaler(self) -> Standardiser:
        if self.moderator_scaler is None:
            raise ValueError(
                "frozen design carries no moderator scaler, but the model builds a "
                "moderator term; the design was captured from a different model"
            )
        return self.moderator_scaler

    def hsgp_c_for(self, x: np.ndarray) -> float:
        """Return the boundary factor reproducing ``hsgp_L`` on ``x``."""
        if self.hsgp_L is None:
            raise ValueError("frozen design carries no HSGP boundary to reproduce")
        support = float(max(abs(np.min(x)), abs(np.max(x))))
        if not np.isfinite(support) or support <= 0:
            raise ValueError("cannot reproduce an HSGP boundary on degenerate support")
        return self.hsgp_L / support


@dataclass(frozen=True)
class MechanismPayload(FittedPayload):
    """The realised standardisation and HSGP design of a mechanism fit."""

    design: MechanismDesign


@dataclass(frozen=True)
class DidDosePayload(FittedPayload):
    """Exact fitted rows and treatment-dose design for a dose DiD fit."""

    design: Literal["dose_intensive_margin"]
    dose_scaler: Standardiser
    age_t1_scaler: Standardiser
    analysis_row_ids: np.ndarray
    raw_attend: np.ndarray
    dose_treated_std: np.ndarray
    treated: np.ndarray


@dataclass(frozen=True)
class DidArmWavePayload(FittedPayload):
    """Exact fitted rows and baseline anchor for an arm-by-wave DiD fit."""

    design: Literal["arm_by_wave"]
    alpha_anchor: float | None
    age_t1_scaler: Standardiser
    analysis_row_ids: np.ndarray
    waves: tuple[int, ...]


@dataclass(frozen=True)
class GainFactorsPayload(FittedPayload):
    """Realised treatment-interaction moderators for a gain-factor fit."""

    trt_interaction_moderators: tuple[TreatmentModerator, ...]


@dataclass(frozen=True)
class LevelFactorsPayload(FittedPayload):
    """Outcome-informed intercept anchor, arm-gap parameterisation (``"t1"``
    balance term + changes, or the ``"free"`` per-timepoint comparator, #552) and
    score-mean link (#584 decision 2) of a level-factor fit.

    ``alpha_anchor`` is on the linear-predictor scale, so it is already mapped
    through ``score_mean_link``: the two fields belong together and reading either
    without the other misplaces the anchor by about 1.1 logits on the blending pair.
    """

    alpha_anchor: float
    arm_gap_reference: str = "t1"
    score_mean_link: str = "logit"


@dataclass(frozen=True)
class LongCorrFactorPayload(FittedPayload):
    """Exact missingness-pattern and measurement design of an LCF fit."""

    z_nodes: tuple[str, ...]
    child_of_node: dict[str, np.ndarray]
    cell_indices_of_node: dict[str, np.ndarray]
    observed_z_of_node: dict[str, np.ndarray]
    domains: dict[str, tuple[str, ...]]
    domain_of: dict[str, str]
    indicators: tuple[str, ...]
    cell_names: tuple[str, ...]
    standardisers: dict[str, tuple[float, float]]
    waves: tuple[int, ...]
    n_children: int
    n_used_children: int
    invariance: str


@dataclass(frozen=True)
class ScreeningWordReadingPayload(FittedPayload):
    """Source and target-population metadata for the ITT missingness model."""

    source_sha256: str
    target_n: int
    observed_n: int
