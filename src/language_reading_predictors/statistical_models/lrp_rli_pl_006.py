# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPPL006 - wave-pooled level association: speech production (deapp_c) -> word reading (W).

Primary (#553). The between/within split extended to speech production accuracy
(``deapp_c``), a raw score with no confirmed maximum, so — as for phonological
memory in ``lrp-rli-pl-005`` — it enters as a **standardised raw-score covariate**
(``mechanism_is_covariate``) with ``require_observed=("deapp_c",)`` dropping the
mean-imputed child-wave rows (imputation plus an indicator is an adjuster policy,
never acceptable for the exposure itself; the dropped count is reported). The
raw-units SD is recorded beside the fit. ``deapp_c`` is measured at every wave,
so the within-child term is meaningful. No own-baseline term (the levels
estimand). Speech production has no mechanism-family fit as an exposure; it
appears only as an adjuster (gamma +0.10/SD in ``lrp-rli-mech-058``).

This is the fit where the split changes what the notes can say: ``pl-001``'s
speech adjuster (gamma +0.26/SD, P = 0.999) is the strongest level association
in question 7 of the question-organised report, and the report currently
*infers* "stable between-child correlate" from the level-versus-gain contrast;
this model tests it directly.

Adjustment set: hearing (``hs`` / ``hs_missing``), phonological memory (``erbto``
/ ``erbto_missing``), the t1 block-design ability proxy, linear age at the wave.
No ``attend`` (transition covariate; omitted as in ``pl-001``).

Association only: exposure and outcome are measured at the same wave.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.pooled_levels import (
    fit_pooled_levels,
)
from language_reading_predictors.statistical_models.pooled_levels import (
    PooledLevelsModelSettings,
)

SPEC = ModelSpec(
    model_id="lrp-rli-pl-006",
    kind="pooled_levels",
    title="Wave-pooled level association: speech production (deapp_c) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="deapp_c",
    model_settings=PooledLevelsModelSettings(
        adjust_for=("hs", "hs_missing", "erbto", "erbto_missing"),
        mechanism_is_covariate=True,
        require_observed=("deapp_c",),
        ability_covariate="blocks",
        use_wave_intercepts=True,
        decompose_between_within=True,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev"):
    return fit_pooled_levels(SPEC, config=config)
