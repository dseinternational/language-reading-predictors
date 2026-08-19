# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPPL005 - wave-pooled level association: phonological memory (erbto) -> word reading (W).

Primary (#553). The between/within split extended to word/nonword repetition. The
exposure is the ERB total (``erbto``), an integer score whose documented test
maximum is recorded nowhere in the repo — registering it as a bounded-count
``Measure`` would fabricate a denominator (``lrp_rli_mech_090.py`` records the
reasoning) — so it enters as a **standardised raw-score covariate**
(``mechanism_is_covariate``): ``beta_between`` / ``beta_within`` are per +1 SD of
the observed ERB total (the raw-units SD is recorded beside the fit as
``mechanism_exposure_sd_raw``). ``require_observed=("erbto",)`` drops the
mean-imputed child-wave rows: imputation plus an indicator is an adjuster policy,
never acceptable for the exposure itself; the dropped count is reported. The
Mundlak split is unchanged, and ``erbto`` is measured at every wave, so the
within-child term is meaningful. No own-baseline term (the levels estimand; the
transition estimand is ``lrp-rli-mech-090``, +0.10 log-odds per SD, P = 0.95).

Adjustment set mirrors ``lrp-rli-mech-090`` minus the own baseline: hearing
(``hs`` / ``hs_missing``), the t1 block-design ability proxy, linear age at the
wave. No ``attend`` (transition covariate; omitted as in ``pl-001``).

What the split settles: whether phonological memory's level association with
word reading (``ca-001`` -0.04 at t3; ``hs-002`` 0.67) is trait-level covariation
or tracks within-child change. ``erbto`` is measured with error; non-negligible
error attenuates both slopes toward zero.

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
    model_id="lrp-rli-pl-005",
    kind="pooled_levels",
    title="Wave-pooled level association: phonological memory (erbto) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="erbto",
    model_settings=PooledLevelsModelSettings(
        adjust_for=("hs", "hs_missing"),
        mechanism_is_covariate=True,
        require_observed=("erbto",),
        ability_covariate="blocks",
        use_wave_intercepts=True,
        decompose_between_within=True,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev"):
    return fit_pooled_levels(SPEC, config=config)
