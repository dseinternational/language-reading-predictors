# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP77a - Baseline-ability sensitivity fit for LRP77 (probes the GA -> IS path).

Identical to :mod:`lrp-rli-dose-077` (period-varying dose-response on word reading) but with
the **baseline-skill cluster** added to the adjustment set: letter-sound
knowledge (L), expressive vocabulary (E) and phoneme blending (B). These are the
reflective indicators of latent general ability ``g`` in the revised DAG
(alongside the already-adjusted baseline reading W_pre).

Rationale: the revised DAG carries ``GA -> IS`` (latent), so if abler children
attended more (Frank 2012), ``g`` is a common cause of attendance and outcome and
the measured adjustment set does not identify an attendance effect. Conditioning
on the baseline-skill cluster probes that path with observed proxies.
**Read LRP77a against LRP77:** if the slope survives, the assumption that ability
does not drive attendance is more defensible in this sample; if it collapses, the
LRP77 signal is substantially ability-confounded and should be downgraded. Neither
outcome *proves* the latent path is closed — these are noisy proxies for ``g``,
not ``g``.

Each adjuster is that child's **verified t1 value, broadcast across all three
transitions** (``ability_baseline_wave="t1"``, the default). Until #587 this fit
used each transition's own starting wave instead — t1 skills in period 1, but t2
skills in period 2 and t3 skills in period 3. Those later values are downstream of
earlier intervention and attendance, so the model conditioned on a
treatment-affected time-varying covariate (Robins, Hernan & Brumback 2000) rather
than on a baseline: it could block part of the very path it was measuring, and it
was not the pre-randomisation ability adjustment it was published as. The repair
changes 98 of the 156 fitted rows' ability values.
``ability_baseline_wave="transition_start"`` retains the old behaviour as an
explicitly labelled comparator; it must not be presented as a baseline sensitivity.

A child with no verified t1 ability row is dropped rather than given a later,
treatment-affected substitute.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.dose_response import (
    DoseResponseModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp-rli-dose-177",
    kind="dose_response",
    title="Dose-response (ability-adjusted sensitivity) - LRP77 + baseline-skill cluster",
    outcome_symbol="W",
    adjustment=["G", "A", "W_pre", "L_pre", "E_pre", "B_pre"],
    model_settings=DoseResponseModelSettings(
        adjust_baseline_symbol="W",
        dose_covariate="attend",
        # No cumulative-dose (attend_cumul) control — IS collider (#269).
        period_varying_dose=True,
        use_subject_random_intercept=True,
        ability_adjust_symbols=("L", "E", "B"),
        outcomes=("W", "L", "E", "B"),
    ),
    extra={
        # The worst of the dose family at the reporting preset's 0.95 — 23 divergences,
        # unsurprising given the extra baseline-skill cluster on the same funnelled
        # dose-slope geometry. 0.99 clears it (0 divergences, R-hat 1.002, min ESS
        # 3,300). See notes/202608050649-reporting-refit-predictive-checks.md.
        "target_accept": 0.99,
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
