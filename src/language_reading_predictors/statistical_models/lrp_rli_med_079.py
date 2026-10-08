# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP79 - NEGATIVE-CONTROL mediator: does word reading route through grammar (T)?

A calibration model for the causal-vs-associational question about the letter-sound
route ([LRP59](lrp_rli_med_059)). The DAG **severs grammar from word reading**: the
receptive-grammar node `RG` (TROG, our `T`) has *no directed path* to `WR` — on the
simple-view logic, grammar loads on reading *comprehension*, not word-level
recognition. So any indirect effect this g-formula reports through `T` **cannot be a
causal route** — it can only be residual confounding (latent general ability `GA`,
shared with reading) that survived the adjustment set. `T` is therefore a **negative
control**: it estimates how large a spurious mediated association the *same* machinery
and adjustment set manufacture for a mediator the DAG says is causally inert for `WR`.

Interpretation:

- The residual mediator -> outcome association ``b_M`` is the main
  negative-control readout. A large value is consistent with residual
  confounding or model misspecification under the stated DAG. A near-zero
  value does not establish that the GA backdoor is closed for grammar or
  letter sounds. The measures can differ in their relation with ability,
  measurement error and precision.

Design uses the same mediation family with mediator T: phase 0 only, mediator
`T_t2` (Beta-Binomial on `T_t1`), outcome `W_t2`, adjustment
{G, A, E, R, W_pre, T_t1}. All ID-2 caveats apply; nothing here is a causal route.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mediation import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-079",
    kind="mediation",
    title=(
        "Negative-control mediator: apparent word-reading (W) route through grammar "
        "(T), a DAG-severed path — calibrates residual GA confounding"
    ),
    outcome_symbol="W",
    mechanism_symbol="T",  # the negative-control mediator (grammar, DAG-severed from WR)
    adjustment=["G", "A", "E", "R", "W_pre", "T_t1"],
    model_settings=MediationModelSettings(),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mediation(SPEC, config=config)
