# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LCSM-181 - no-reverse-coupling LOO comparator for LCSM-081.

Identical to :mod:`lrp_rli_lcsm_081` minus the two reverse couplings
(``g_W_TE``, ``g_W_TR``): the taught-vocabulary changes keep their carry-over,
the ``g_TR_TE`` confounder coupling, the arm x window intercepts and the
``hs`` / ``erbto`` / ``deapp_c`` adjuster block, but prior word reading no
longer enters. Same outcomes, same observed cells, so PSIS-LOO against 081 is
the pre-specified "does the reverse edge earn its place predictively" readout
(design: ``notes/202607141030-time-lagged-model-designs.md``).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_lcsm

SPEC = ModelSpec(
    model_id="lrp-rli-lcsm-181",
    kind="lcsm",
    title=(
        "No-reverse-coupling comparator: taught-vocabulary change without the "
        "prior word-reading terms (LOO baseline for LCSM-081)"
    ),
    outcome_symbol="TE",
    extra={
        "outcomes": ["TE", "TR", "W"],
        # As LCSM-081 but with the W -> TE / W -> TR reverse couplings removed;
        # the TR -> TE adjuster coupling stays (it is a confounder term, not
        # the hypothesis under test).
        "couplings": {"TE": ["TR"]},
        "arm_window_intercepts": True,
        "covariate_block": [
            "hs", "hs_missing",
            "erbto", "erbto_missing",
            "deapp_c", "deapp_c_missing",
        ],
        "covariate_targets": ["TE", "TR"],
        "coupling_prior_sigma": 0.3,
        "use_process_noise": True,
        "shared_process_noise": False,
    },
)


def fit(config: str = "dev"):
    return fit_lcsm(SPEC, config=config)
