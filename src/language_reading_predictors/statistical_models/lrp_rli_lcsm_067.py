# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP67 - latent change-score model: what within-child changes predict reading change?

The within-child, longitudinal complement to **LRP65** (between-child predictors
of word-reading gain). Where LRP65 asks "which baseline skills predict a child's
overall reading gain", LRP67 unrolls that question over all four waves: for each
wave-to-wave transition, does a child's prior-wave **letter-sound** and
**vocabulary** standing predict their subsequent **reading** change?

Model (full McArdle latent change-score with process noise, on the logit scale;
see :func:`factories.build_lcsm_model`):

- A latent logit true-score ``x_m[i, t]`` per measure ``m``, child ``i``, wave ``t``.
- ``x_m[i, t] = x_m[i, t-1] + Delta_m[i, t]``, the change being a structured mean
  plus a per-measure dynamic disturbance (process noise).
- Reading change is coupled to prior-wave levels::

      mean_Delta_W = a_W + b_W * x_W[t-1] + g_L * x_L[t-1] + g_E * x_E[t-1]
                   + d_age_W * age[t-1]

  ``g_L`` / ``g_E`` are the headline coefficients.
- Coupling coefficients are **time-invariant** (pooled across the 3 transitions) -
  a deliberate constraint at n~54.
- Observed counts enter via a masked Beta-Binomial; ``kappa`` is measurement
  overdispersion, distinct from the dynamic process noise.

Symbol mapping (codebase ``MEASURES`` vs the meeting handoff's R / L / V):

    handoff R (reading)        -> W  (ewrswr,  denominator 79)
    handoff L (letter-sounds)  -> L  (yarclet, denominator 32)
    handoff V (vocabulary)     -> E  (eowpvt,  denominator 170)

``R`` is **not** reused here: in ``MEASURES`` it is receptive vocabulary (rowpvt).
So the handoff's headline "L -> R over time" is ``g_L`` (prior L -> reading change)
in this module's notation.

Exploratory at n~54: intervals will be wide. The deliverable is whether the
*direction* of ``g_L`` (letter-sounds -> reading change) agrees with LRP65
(between-child) as a triangulation point, not any single coefficient. The RI-CLPM
companion was dropped as not estimable at this sample size; see the dated note.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_lcsm

SPEC = ModelSpec(
    model_id="lrp-rli-lcsm-067",
    kind="lcsm",
    title=(
        "Latent change-score model: prior letter-sounds (L) and vocabulary (E) "
        "as within-child predictors of reading (W) change"
    ),
    outcome_symbol="W",
    extra={
        # Measure symbols (W reading, L letter-sounds, E expressive vocab).
        "outcomes": ["W", "L", "E"],
        # Regularising prior SD on the cross-couplings into the reading change.
        # Reconciled 0.5 -> 0.3 to match the shared association scale (gamma_cross)
        # — prior-critical-review 2026-07-07, recommendation 3.
        "coupling_prior_sigma": 0.3,
        # Full McArdle form: per-measure dynamic disturbances on the change scores.
        # Set False (or shared_process_noise=True) as a sampling fallback at n~54.
        "use_process_noise": True,
        "shared_process_noise": False,
    },
)


def fit(config: str = "dev"):
    return fit_lcsm(SPEC, config=config)
