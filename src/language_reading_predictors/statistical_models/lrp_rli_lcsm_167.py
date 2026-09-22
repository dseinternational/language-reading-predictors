# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LCSM-167 - LCSM-067 with arm x window change intercepts.

The transition-specific companion of **LCSM-067**: the same three processes
(``W`` word reading, ``L`` letter sounds, ``E`` expressive vocabulary), the same
couplings, priors and process noise, with one change. LCSM-067 gives each
measure a single change intercept pooled over all three transitions; this model
gives each measure one intercept per arm and transition::

    mean_Delta_W = a_W[arm, w] + b_W * x_W[t-1] + g_L * x_L[t-1] + g_E * x_E[t-1]
                 + d_age_W * age[t-1]

Why it is needed. The three transitions are unequal in length: about 7.1, 8.4 and
5.4 months on average, with every child's interval within a transition equal to
whole-month rounding. The last transition is the shortest and starts from the
highest levels of every measure, so under a pooled intercept a smaller change
there can load onto the prior-level couplings (``g_L``, ``g_E``), the self-feedback
``b_W`` and the age term. Treatment status also differs by transition (the
wait-list arm is untreated in the first). The arm x window intercepts absorb both
the mean change of each transition and each arm's treatment schedule, as in
LCSM-081/082/091/181. Decision and measured interval lengths:
``notes/202609212100-assessment-interval-lengths.md``.

Reading. Compare ``g_L`` / ``g_E`` / ``d_age`` with LCSM-067's. Agreement means the
pooled intercept was not driving them; a shift quantifies how much it was. Every
coupling remains an adjusted association (latent general ability is not
blocked). The window-1 arm contrast in ``itt_window1_contrast.csv`` is the
built-in consistency check against the available-case modified ITT estimates.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.lcsm import LcsmModelSettings
from language_reading_predictors.statistical_models.pipelines.lcsm import fit_lcsm

SPEC = ModelSpec(
    model_id="lrp-rli-lcsm-167",
    kind="lcsm",
    title=(
        "Latent change-score model with arm x window change intercepts: prior "
        "letter-sounds (L) and vocabulary (E) as predictors of reading (W) change"
    ),
    outcome_symbol="W",
    model_settings=LcsmModelSettings(
        # As LCSM-067: W reading, L letter-sounds, E expressive vocabulary, with
        # the default couplings (every non-reading measure into the reading change).
        outcomes=("W", "L", "E"),
        # The one difference from LCSM-067: per-arm, per-transition change
        # intercepts in place of one pooled intercept per measure, absorbing the
        # unequal transition lengths and the crossover schedule.
        arm_window_intercepts=True,
        coupling_prior_sigma=0.3,
        use_process_noise=True,
        shared_process_noise=False,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_lcsm(SPEC, config=config)
