# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP77a - Ability-adjusted sensitivity fit for LRP77 (tests the no-g->dose assumption).

Identical to :mod:`lrp77` (period-varying dose-response on word reading) but with
the **baseline-skill cluster** added to the adjustment set: letter-sound
knowledge (L), expressive vocabulary (E) and phoneme blending (B), each as its
baseline (pre) logit. These are the reflective indicators of latent general
ability ``g`` in shared DAG v5 (alongside the already-adjusted baseline reading
W_pre).

Rationale: DAG v5 omits ``g -> dose``, but if abler children attended more
(Frank 2012), ``g`` is a common cause of dose and outcome and ``{G, W_pre, A}``
no longer identifies the dose effect. Conditioning on the baseline-skill cluster
blocks that path. **Read LRP77a against LRP77:** if the dose slope survives, the
no-``g``->dose assumption is defensible here; if it collapses, the dose signal is
substantially ability-confounded and the LRP77 estimate should be downgraded.

Requiring L/E/B baselines complete-cases slightly fewer rows than LRP77. Only
*pre*-dose (baseline) skills are conditioned on - never anything downstream of
dose, which would block the path being estimated.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp77a",
    kind="dose_response",
    title="Dose-response (ability-adjusted sensitivity) - LRP77 + baseline-skill cluster",
    outcome_symbol="W",
    adjustment=["G", "A", "W_pre", "L_pre", "E_pre", "B_pre"],
    extra={
        "adjust_baseline_symbol": "W",
        "dose_covariate": "attend",
        "dose_stage_covariate": "attend_cumul",
        "period_varying_dose": True,
        "use_subject_random_intercept": True,
        "ability_adjust_symbols": ("L", "E", "B"),
        "outcomes": ("W", "L", "E", "B"),
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
