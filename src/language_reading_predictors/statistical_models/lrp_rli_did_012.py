# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID12 - waitlist-crossover / difference-in-differences ITT effect on nonword reading (NWR) (N), off-floor.

A within-person crossover (difference-in-differences) read on nonword reading -
a within-person triangulation of the randomised floor-rule ITT (LRPITT11). Stacks
the two early periods (P1 = t1->t2, P2 = t2->t3) for both arms, with each child as
their own control (a child random intercept) and the immediate arm anchoring the
time/maturation trend.

Nonword reading is heavily floored, so the observation is a Bernoulli on the binary
off-floor indicator (period post > 0), not the graded Beta-Binomial count.

ESTIMAND - off-floor PREVALENCE, not floor exit (#257 review, decision (a)). This
models ``Pr(post > 0)`` at each period end, over *every* child in the period, NOT
the transition ``Pr(post > 0 | pre = 0)`` that #119 defines as the ITT floor-rule
PRIMARY. Those differ materially: a child already off the floor at period start who
stays off counts the same as one who moved off, so prevalence blends "already off +
stayed" with "came off". We use prevalence deliberately, because the DiD's job is a
within-person REPLICATION of the randomised effect and prevalence keeps that
identification clean: the transition estimand's floored risk set is, for the
immediate arm's P2, selected by that arm's own P1 treatment (a post-treatment
selection) that would contaminate the maturation-trend anchor. The transition
primary is carried cleanly by the ITT sibling LRPITT11 on the single randomised
t1->t2 phase; this model complements it on the prevalence scale.

No own-baseline term. Unlike the graded family this model does NOT condition on the
period-start score: for the immediate arm's P2 that score is post-P1-treatment, so
``gamma_own`` would adjust a treatment-affected variable and a child intercept does
not restore the total-effect reading (Rosenbaum 1984, doi:10.2307/2981697). The
prevalence DiD is identified by the period x treated structure plus the child
intercept. (A consequence: a missing period-start score no longer drops the row -
recovering the nonword P1 observations the pre-requirement previously discarded.)

The DiD contrast ``delta`` is the within-person effect on the LOG-ODDS of being off
the floor at period end. Its marginal (``delta_items``, n_trials = 1) is a
model-implied off-floor RISK DIFFERENCE from toggling ``Treated`` at the fitted
covariates - NOT a probability-scale DiD cross-difference. Parallel trends is
imposed on the LOG-ODDS scale (the predictor is additive in logit p); equal
log-odds trends do not imply equal probability trends when baseline risks differ
(Puhani 2012, doi:10.1016/j.econlet.2011.11.025). Sign convention: positive =>
intervention helps (raises Pr(off-floor)). Flagged for review sign-off (#226).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-012",
    kind="did",
    title="Waitlist-crossover (DiD) ITT effect of the intervention on nonword reading (NWR) (N), off-floor",
    outcome_symbol="N",
    extra={
        "outcomes": ("N",),
        "periods": (0, 1),
        "use_child_re": True,
        "use_age": True,
        "dose": False,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
