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
too narrow. This companion estimates
that covariance — a per-child bivariate-normal offset ``u_i`` with
``Sigma = diag(sigma) Corr diag(sigma)``, ``Corr ~ LKJ(eta = 4)``,
``sigma_k ~ HalfNormal(0.5)``, non-centred through ``pm.LKJCholeskyCov`` — and
publishes the contrast as a posterior difference under it. It is **not** a
replacement for the parent: the parent remains the model of record, and this fit
answers whether the parent's interval was too wide or too narrow.

Read it beside the parent, and read ``dependence_identification.csv`` first.
**This fit's correlation posterior is its correlation prior**: posterior SD
1.008 times the prior SD read from its own persisted prior group (2026-08-22 ITT
audit, finding 3). At n = 53 the data say nothing about the within-child
correlation, so the interval this companion publishes carries the LKJ prior's
implied correction and not a measured covariance — ``release_decision.json`` and
the findings box now attach that qualifier. The per-outcome residual SDs *are*
informed (posterior SD roughly a third of the prior's), so the block is not
uniformly uninformative; the table reports each parameter separately for that
reason. The April 2026 ten-outcome fit found the same thing more severely for a
10 x 10 block, which is why the block is off by default and why
``lrp-rli-itt-012`` is out of scope here.

Point estimates are *not* guaranteed to be invariant. Adding a logistic-normal
per-child offset changes the marginal likelihood and re-estimates ``alpha``,
``tau``, the baseline slopes and ``kappa`` jointly, and the logit link is
nonlinear, so the earlier claim that "the point estimate is unaffected either
way" held only as an empirical observation about small shifts. Measured, the
parent-to-companion contrast medians move by 0.0001-0.0011 on the
proportion-correct scale — negligible in substance, but not zero by construction.

The AME this fit reports is **conditional on the fitted children's residuals**:
``_joint_ame_draws`` reads the stored ``eta``, which already contains ``u_i``, and
nets out only the treatment term. That is deliberate — it keeps the estimand as
close as possible to the parent's, which has no random effect at all — and is not
a new-child population marginal. Integrating fresh residuals instead was tried and
moves the medians by less than 0.00012.

The earlier two-outcome attempt mixed poorly because the per-outcome
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
            # This fit IS the dependence model, so it must not name a companion
            # (resolve_joint_run_plan rejects a correlated fit that does).
            dependence_companion=None,
            dependence_note=(
                "Dependence-aware sensitivity companion of lrp-rli-itt-016: the "
                "per-child LKJ residual-correlation block is on, so this contrast "
                "is a posterior difference that carries the estimated within-child "
                "covariance between the two outcomes. Read it beside the parent's "
                "factorised interval — this fit's per-child "
                "logistic-normal offset makes its average marginal effect a "
                "latent-conditional estimand rather than the parent's, so "
                "agreement of the point estimates is an empirical finding "
                "(medians move by 0.0001-0.0011 on the proportion-correct "
                "scale), not a mathematical invariant; the interval and "
                "P(> 0) may move — and check u_corr / sigma_outcome "
                "for how far the block is informed by the data rather than its "
                "prior."
            ),
        ),
    ),
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
