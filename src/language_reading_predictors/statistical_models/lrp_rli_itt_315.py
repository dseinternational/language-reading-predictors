# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT315 - dependence-aware companion of LRPITT15b/115 (receptive taught versus not-taught).

Statistical interpretation corrected by a LLM-based AI tool (Codex/GPT-6).

The registered dependence-model sensitivity for LRPITT15b/115 (#551): the same
two-outcome joint Beta-Binomial available-case modified ITT fit — same outcomes,
own-baseline and linear-age precision terms, same contrast — with the per-child
**LKJ residual-correlation block switched on** (``use_residual_correlation=True``,
``joint_structure="residual_correlated"``). With the block off the two outcomes
share no parameter, so the parent's likelihood and priors factorise and its
contrast ``AME[TR] - AME[UR]`` is the difference of two a-posteriori
independent quantities: its interval omits the within-child covariance that the
same 54 children supplying both outcomes induce. Holding marginal variances fixed, the identity
``Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)`` relates covariance to the
variance of a difference. It does not determine equal-tailed interval widths.
The companion adds a bivariate-normal child offset with
``Sigma = diag(sigma) Corr diag(sigma)``, ``Corr ~ LKJ(eta = 4)`` and
``sigma_k ~ HalfNormal(0.5)``. Its residual correlation and posterior covariance
between outcome effects are different quantities. Read the paired effect draws
and marginal variances to assess the contrast. The parent remains the model
of record.

The earlier report of wider contrast intervals did not establish which model
component caused the change. Interval widths cannot supply a covariance
decomposition. Regenerate the paired-draw moments before making that comparison.

Read ``dependence_identification.csv`` beside the prior and posterior overlays.
The previously reported posterior-to-prior SD ratio of 1.002 describes similar
spread only. It does not establish equal distributions or the absence of
information about correlation. Location, shape and sign probabilities may change.
Check the prior source and sensitivity of the named contrast as well as spread.

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
reported contrast is ``AME[TR] - AME[UR]`` on the proportion-correct
scale.
"""

from dataclasses import replace
from language_reading_predictors.statistical_models.joint import (
    JointModelSettings,
)

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.lrp_rli_itt_115 import (
    SPEC as _PARENT,
)
from language_reading_predictors.statistical_models.pipelines.joint import fit_joint

# Identical to the parent except the dependence block and the note that describes
# it: ``replace`` on the parent's frozen settings guarantees the outcomes, precision
# terms, LOO unit and contrast metadata cannot drift apart (#551).
_PARENT_SETTINGS = _PARENT.model_settings
# Not an ``assert``: this companion is defined by reusing its parent's
# settings object, and ``-O`` would remove the one check that the parent
# still declares them typed (#637).
if not isinstance(_PARENT_SETTINGS, JointModelSettings):
    raise TypeError(
        f"{_PARENT.model_id} must declare JointModelSettings for this companion to "
        f"reuse; got {type(_PARENT_SETTINGS).__name__}"
    )
_PARENT_CONTRAST = _PARENT_SETTINGS.contrast
if _PARENT_CONTRAST is None:
    raise ValueError(f"{_PARENT.model_id} declares no contrast for this companion to reuse")
SPEC = ModelSpec(
    model_id="lrp-rli-itt-315",
    kind="joint",
    title=(
        "Available-case modified ITT estimate: receptive taught-versus-not-taught "
        "vocabulary contrast, block 1 — LKJ residual-correlation sensitivity "
        "companion of lrp-rli-itt-115"
    ),
    model_settings=replace(
        _PARENT_SETTINGS,
        use_residual_correlation=True,
        joint_structure="residual_correlated",
        contrast=replace(
            _PARENT_CONTRAST,
            # This fit IS the dependence model, so it must not name a companion
            # (resolve_joint_run_plan rejects a correlated fit that does).
            dependence_companion=None,
            dependence_note=(
                "Dependence-aware sensitivity companion of lrp-rli-itt-115: the "
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


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_joint(SPEC, config=config)
