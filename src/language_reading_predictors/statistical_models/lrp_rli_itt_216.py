# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT216 - dependence-aware companion of LRPITT16 (taught expressive versus taught receptive).

The registered dependence-model sensitivity for LRPITT16 (#551): the same
two-outcome joint Beta-Binomial available-case modified ITT fit — same outcomes,
own-baseline and linear-age precision terms, same contrast — with the per-child
**LKJ residual-correlation block switched on** (``use_residual_correlation=True``,
``joint_structure="residual_correlated"``). With the block off the two outcomes
share no parameter, so the parent's likelihood and priors factorise and its
contrast ``AME[TE] - AME[TR]`` is the difference of two a-posteriori
independent quantities: its interval omits the within-child covariance that the
same 54 children supplying both outcomes induce. A positive residual correlation
(the plausible direction) makes the factorised interval too wide, a negative one
too narrow; the point estimate is unaffected either way. This companion estimates
that covariance — a per-child bivariate-normal offset ``u_i`` with
``Sigma = diag(sigma) Corr diag(sigma)``, ``Corr ~ LKJ(eta = 4)``,
``sigma_k ~ HalfNormal(0.5)``, non-centred through ``pm.LKJCholeskyCov`` — and
publishes the contrast as a posterior difference under it. It is **not** a
replacement for the parent: the parent's point estimate stands, and this fit
answers only whether the parent's interval was too wide or too narrow.

Read it beside the parent: per-outcome ``tau`` and the contrast median should
agree to Monte Carlo error; the contrast's 89% interval and P(> 0) may move, and
``u_corr`` / ``sigma_outcome`` show how much the block is informed by the data
(the April 2026 ten-outcome fit found a 10 × 10 block prior-dominated at n = 53,
which is why the block is off by default and why ``lrp-rli-itt-012`` is out of
scope here). The earlier two-outcome attempt mixed poorly because the per-outcome
residual SD sat at its zero boundary; under the unrounded house gate a repeat of
that geometry withholds the result rather than qualifying it, in which case the
follow-up is a paired bootstrap / randomisation sensitivity outside the pipeline.

Sign convention and estimand are the parent's: ``tau`` is the coefficient on the
intervention indicator (positive => the intervention raised that outcome); the
reported contrast is ``AME[TE] - AME[TR]`` on the proportion-correct
scale.
"""

from dataclasses import replace

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.lrp_rli_itt_016 import (
    SPEC as _PARENT,
)
from language_reading_predictors.statistical_models.pipelines.joint import fit_joint

# Identical to the parent except the dependence block and the note that describes
# it: ``replace`` on the parent's frozen settings guarantees the outcomes, precision
# terms, LOO unit and contrast metadata cannot drift apart (#551).
_PARENT_SETTINGS = _PARENT.model_settings
SPEC = ModelSpec(
    model_id="lrp-rli-itt-216",
    kind="joint",
    title=(
        "Available-case modified ITT estimate: taught expressive (TE) versus "
        "taught receptive (TR) vocabulary contrast, block 1 — LKJ "
        "residual-correlation sensitivity companion of lrp-rli-itt-016"
    ),
    model_settings=replace(
        _PARENT_SETTINGS,
        use_residual_correlation=True,
        joint_structure="residual_correlated",
        contrast=replace(
            _PARENT_SETTINGS.contrast,
            dependence_note=(
                "Dependence-aware sensitivity companion of lrp-rli-itt-016: the "
                "per-child LKJ residual-correlation block is on, so this contrast "
                "is a posterior difference that carries the estimated within-child "
                "covariance between the two outcomes. Read it beside the parent's "
                "factorised interval — the point estimate should agree; the "
                "interval and P(> 0) may move — and check u_corr / sigma_outcome "
                "for how far the block is informed by the data rather than its "
                "prior."
            ),
        ),
    ),
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
