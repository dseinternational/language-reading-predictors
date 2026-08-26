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
class AlignedPayload(FittedPayload):
    """Score-mean link of an onset-aligned fit (#619).

    Carries what the factory *built* rather than what the module declared, so the
    cohort marginal and its prior pushforward map through the same link the
    likelihood used. Reading the declared setting instead would let a floor-link
    posterior publish ordinary-link items.
    """

    score_mean_link: ScoreMeanLink = "logit"


@dataclass(frozen=True)
class MediationPayload(FittedPayload):
    """Score-mean link of a single-mediator mediation fit's OUTCOME leg (#619).

    Carries what the factory built, so the g-formula's counterfactual simulation
    accumulates each ``E[Y(g, M(g'))]`` on the response scale the outcome likelihood
    used. The mediator leg has its own measure and is unaffected.
    """

    score_mean_link: ScoreMeanLink = "logit"


@dataclass(frozen=True)
class ConcurrentPayload(FittedPayload):
    """Score-mean link of one wave's concurrent-associations fit (#619).

    Carries what the factory *built*, so every wave's marginals map through the link
    the likelihood used. The concurrent family fits one model per wave, so the link
    must reach each sub-fit rather than only the primary.
    """

    score_mean_link: ScoreMeanLink = "logit"


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
class DoseResponsePayload(FittedPayload):
    """Realised dose design of a ``dose_response`` fit (#587 findings 2, 3, 13).

    The family reports an **intensive-margin** dose association among on-intervention
    rows, so every consumer has to reuse the same treated-row standardisation, the
    same treated mask and the same per-phase support bounds the factory fitted. The
    audit found three ways that went wrong when they were recomputed downstream: the
    reported ``+1 SD`` contrast was averaged over rows with no treated support, the
    loader's pre-mask scaler disagreed with the fitted rows, and the prior pushforward
    used a scalar slope where the posterior used a phase-indexed one.

    ``dose_scaler`` standardises sessions **over the fitted treated rows only**, so a
    unit of the fitted dose is one treated-row SD of sessions rather than one SD of a
    distribution dominated by structural zeros. ``treated`` and ``raw_attend`` are
    row-aligned to the fitted rows; ``phase_support`` carries the per-phase observed
    session quartiles and bounds behind the reported contrast.
    """

    design: Literal["dose_intensive_margin"]
    dose_scaler: Standardiser
    treated: np.ndarray
    raw_attend: np.ndarray
    dose_between: np.ndarray
    dose_within: np.ndarray
    phase_support: tuple[tuple[float, float, float, float], ...]
    decompose_between_within: bool
    #: The score-mean link the factory built (#619). The family's focal estimand is
    #: the natural-scale treated-row dose marginal, so every consumer that maps eta
    #: onto items must use this rather than assume the ordinary inverse logit.
    score_mean_link: ScoreMeanLink = "logit"


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
    """Exact fitted rows, baseline anchor and score-mean link of an arm-by-wave DiD fit.

    ``alpha_anchor`` is on the linear-predictor scale, so it is already mapped back
    through ``score_mean_link`` (#576 finding 2): the two fields belong together, and
    reading either without the other misplaces the anchor by about 1.1 logits on the
    phoneme-blending pair.
    """

    design: Literal["arm_by_wave"]
    alpha_anchor: float | None
    age_t1_scaler: Standardiser
    analysis_row_ids: np.ndarray
    waves: tuple[int, ...]
    score_mean_link: str = "logit"


@dataclass(frozen=True)
class GainFactorsPayload(FittedPayload):
    """Realised treatment-interaction moderators and score-mean link of a gain fit.

    ``score_mean_link`` is what the factory *built*, not what the module declared,
    so every natural-scale summary the pipeline derives (the treatment marginal, the
    association marginals, the ROPE, the prior pushforward and the predicted scores)
    reads the same link the likelihood used. Reading the declared setting instead
    would let a floor-link posterior publish ordinary-link items (#596).
    """

    trt_interaction_moderators: tuple[TreatmentModerator, ...]
    score_mean_link: str = "logit"
    #: The adjusters the built model actually carries, after the factory's final
    #: analysis mask (#575 finding 1). The loader filters constants on its own
    #: complete frame, but the factory's focal-outcome / treated-only masks can
    #: make a previously varying indicator constant — an exact intercept alias.
    #: The pipeline records THESE, not the loader-time set, as the effective
    #: adjustment, together with what the final mask removed.
    effective_adjust_for: tuple[str, ...] = ()
    post_mask_dropped_adjusters: tuple[str, ...] = ()
    #: Realised per-period, per-arm fitted-row support (#575 finding 5):
    #: ``(phase, arm_label, n_rows, n_children)`` for every fitted cell, written
    #: to ``analysis_support.csv`` by the pipeline.
    period_arm_support: tuple[tuple[int, str, int, int], ...] = ()


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
