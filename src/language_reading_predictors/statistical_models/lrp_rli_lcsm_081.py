# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LCSM-081 - reading -> taught vocabulary: the lagged reverse-coupling headline.

The founding RLI hypothesis as a direct test on the adopted time-lagged DAG
(#250, Option A): does a child's prior-wave **word-reading** standing predict
their subsequent change in the block-1 **taught** vocabulary measures, over and
above the taught measures' own carry-over, the intervention windows, and
everything measured the lagged graph names? Design + verified adjustment-set
derivations: ``notes/202607141030-time-lagged-model-designs.md``.

Model (the LRP67 McArdle latent change-score scaffold, generalised; see
:func:`factories.build_lcsm_model`): processes ``TE`` / ``TR`` / ``W``, change
equations::

    Delta_TE = a_TE[arm, w] + b_TE * x_TE + g_W_TE * x_W + g_TR_TE * x_TR
             + d_age * age + (hs / rw / sp covariate block)
    Delta_TR = a_TR[arm, w] + b_TR * x_TR + g_W_TR * x_W
             + d_age * age + (same covariate block)
    Delta_W  = a_W[arm, w]  + b_W * x_W + d_age * age

``g_W_TE`` and ``g_W_TR`` are the headline coefficients (prior *level* ->
subsequent *change*, pooled over the three transitions). With the ``TR``
process plus the prior-wave ``hs`` / ``erbto`` (RW) / ``deapp_c`` (SP)
covariates, the conditioning set equals the verified sufficient measured
backdoor set for each coupling (latent ``GA`` aside) - so both are **adjusted
associations**, never causal. The **arm x window** change intercepts are
mandatory, not polish: the waitlist crossover makes the randomised arm a
confounder of every transition-2+ coupling (the waitlist arm's block-1
catch-up lands exactly in window 2). Their window-1 cell contrast is a
randomised latent ITT contrast, written to ``itt_window1_contrast.csv`` as a
consistency check against LRP-RLI-ITT-001/002. Sessions stay out (ID-3
collider).

Pre-specified reading (design note): the RLI hypothesis is supported in this
cohort if both couplings are positive with ``P(g > 0) >= 0.9`` and this model
is not LOO-worse than the no-reverse-coupling comparator **LCSM-181**; weaker
patterns are reported as-is.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_lcsm

SPEC = ModelSpec(
    model_id="lrp-rli-lcsm-081",
    kind="lcsm",
    title=(
        "Lagged reverse coupling: prior word reading (W) predicting taught "
        "expressive (TE) and receptive (TR) vocabulary change"
    ),
    outcome_symbol="TE",
    extra={
        # Measure symbols (TE/TR taught vocabulary, W word reading).
        "outcomes": ["TE", "TR", "W"],
        # Target -> sources: prior W into both taught-vocabulary changes, with
        # prior TR as the DAG-named measured confounder of the TE coupling.
        "couplings": {"TE": ["W", "TR"], "TR": ["W"]},
        # Crossover-aware arm x window change intercepts (mandatory; see module
        # docstring) with the window-1 ITT consistency contrast.
        "arm_window_intercepts": True,
        # Shared adjuster block on the taught-vocabulary change equations:
        # hearing (time-invariant dummies) + prior-wave phonological memory
        # (erbto = RW) and speech production (deapp_c = SP), missing-indicator
        # policy throughout. Completes the verified measured backdoor set.
        "covariate_block": [
            "hs", "hs_missing",
            "erbto", "erbto_missing",
            "deapp_c", "deapp_c_missing",
        ],
        "covariate_targets": ["TE", "TR"],
        # Shared association scale (prior-critical-review 2026-07-07, rec. 3).
        "coupling_prior_sigma": 0.3,
        "use_process_noise": True,
        "shared_process_noise": False,
    },
)


def fit(config: str = "dev"):
    return fit_lcsm(SPEC, config=config)
