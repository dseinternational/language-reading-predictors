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

- The residual **mediator -> outcome association** ``b_M`` (grammar -> reading, after
  {A, E, R, W_pre}) is the readout of interest. Grammar is strongly *correlated* with
  DS word reading (Byrne, MacDonald & Buckley 2002: age-partialled TROG-reading
  r ~ 0.54-0.63, *stronger* than vocabulary), so a large residual ``b_M`` here would
  show the adjustment set does **not** close the `GA` back-door — calibrating how much
  of LRP59's letter-sound ``b_M`` could likewise be confounding. A residual ``b_M``
  near zero shows the adjustment substantially closes it, so LRP59's clearly-positive
  ``b_M`` reflects more than the generic ability bleed-through that inflates *every*
  skill-reading correlation.
- The full **NIE through `T`** is reported too, but read it with care: the ITT effect
  on grammar is near zero (`tau_T` ~ 0), so a small NIE is partly "the intervention
  barely moves `T`", not only "`T` does not predict reading". The ``b_M`` comparison is
  the cleaner negative-control quantity.

Design mirrors LRP59 exactly, swapping the mediator L -> T: phase 0 only, mediator
`T_t2` (Beta-Binomial on `T_t1`), outcome `W_t2`, adjustment
{G, A, E, R, W_pre, T_t1}. All ID-2 caveats apply; nothing here is a causal route.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

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
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
