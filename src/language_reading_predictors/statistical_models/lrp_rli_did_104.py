# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID104 - baseline-allocation prior sensitivity for LRPDID01 (word reading, W).

The arm-by-wave design estimates a free arm gap at each wave, so ``tau_t2`` is the
covariate-adjusted t2 arm-gap **level** — not the differenced quantity
``tau_t2 - arm_gap_t1``. Under randomisation the gap level is a perfectly valid
causal contrast, and the family's estimand sign-off (2026-08-24, #576 finding 4;
``notes/202608241100-did-t2-estimand-signoff.md``) keeps it. What the sign-off also
records is that the baseline adjustment which *does* happen is **soft and
prior-weighted**: a realised t1 imbalance is split between the tightly regularised
``arm_gap_t1`` (Normal(0, 0.3)) and the arm-mean of the shared child random
intercepts (HalfNormal(0.5)), and whatever the intercepts absorb is netted out of all
three wave gaps ANCOVA-style. ``tau_t2`` therefore sits somewhere between the
unadjusted t2 gap and a fully baseline-corrected one, with the mix set by those two
prior widths and nothing else.

This companion is the **estimand-matched** prior sensitivity for that mix. The
registered treatment-prior sweep (``scripts/did_prior_sensitivity.py``) varies
``tau_t2``'s own prior, which answers a different question — how much of the estimate
the effect prior supplies — and leaves the allocation untouched. Here ``tau_t2`` keeps
its tier default and the two allocation widths move instead: ``arm_gap_t1`` widens
from 0.3 to 1.0, so the baseline gap is free to take the realised imbalance rather
than being shrunk into the intercepts, and ``sigma_child`` widens from 0.5 to 1.0 so
the intercepts are equally free to take it.

How to read the comparison: if ``tau_t2`` here matches LRPDID01 to within Monte-Carlo
error, the reported contrast does not depend on how the realised t1 imbalance was
allocated, and the gap-level estimand is behaving like a baseline-referenced one. A
material shift says the opposite — that the published number is partly a function of
the ``arm_gap_t1`` prior — and would be the case for re-parameterising the family on
a t1-referenced gap change, as the level family did in #552.

Reading rules are LRPDID01's, and no term here is a new causal quantity.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.did import DiDModelSettings
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-104",
    kind="did",
    title=(
        "Baseline-allocation prior sensitivity for the word-reading arm-by-wave "
        "contrasts (EWRSWR) (W)"
    ),
    outcome_symbol="W",
    family="did",
    design="waitlist-crossover arm-by-wave levels, wide baseline-allocation priors",
    estimand_type="mixed",
    causal_status="t2 randomised; t3 a randomised treatment-schedule contrast",
    model_settings=DiDModelSettings(
        # Identical to LRPDID01 except the two baseline-allocation widths.
        outcomes=("W",),
        waves=(0, 1, 2),
        use_child_re=True,
        use_age=True,
        dose=False,
        arm_gap_t1_prior_sigma=1.0,
        sigma_child_prior_sigma=1.0,
    ),
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
