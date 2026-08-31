# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT22 - ability-adjusted available-case modified ITT estimate for expressive vocabulary (E).

Ability-robustness companion to LRPITT06, part of the LRPITT17-24 ability-adjusted
family (parallel to the SES-adjusted LRPITT13/13b). Adds the baseline nonverbal
cognitive ability measure - block design (``blocks``), recorded at t1 only - as a
linear precision covariate, on top of the own baseline and linear age the uniform
LRPITT spec already carries.

Block design is a child trait measured *before* randomisation, so - like SES - it
cannot confound the available-case modified ITT estimate; randomisation balances it across arms in
expectation. This adjustment is a precision / chance-imbalance robustness check, not
confounding control (the immediate-intervention arm started ~0.27 SD higher in block
design, and ability is prognostic of the outcomes - most strongly for vocabulary).
Holding age in makes the block-design term read as ability *for age* - the general
latent ability construct (raw block design conflates ability with maturation).

Block design is complete for all 54 children, so no rows drop: LRPITT22 vs LRPITT06
is a same-sample adjusted-vs-unadjusted contrast and no matched comparator is needed.

Sign convention: positive tau => intervention helps.

Dispersion prior (2026-08-22 ITT audit, finding 5). This model samples the
Beta-Binomial dispersion as ``1 / sqrt(kappa) ~ HalfNormal(0.25)`` rather than the
suite default ``kappa ~ HalfNormal(50)``. EOWPVT has a 170-item ceiling, and at
that denominator the default prior *enforces* a floor on over-dispersion: variance
inflation over Binomial is ``(n + kappa) / (1 + kappa)``, so its median kappa of
about 33.7 already implies 5.9x, and coming within 10% of Binomial needs
``kappa > 1689``, which has effectively zero prior mass. The registered
``kappa_sigma`` sweep cannot relax that either — even ``HalfNormal(200)`` gives the
near-Binomial region 0.000 mass — so the prior *family*, not its scale, was the
binding constraint.

The sweep in ``output/statistical_models/dispersion_prior_sensitivity/`` showed the
constraint was real for E specifically. Freed of it the concentration posterior
moves from 126 to 475, variance inflation falls from 2.33x to 1.36x, and 15% of
the posterior sits in the near-Binomial region the default excluded a priori.
Predictive calibration improves at both levels: 72.2% of observations fell inside
a nominal **50%** interval under the default against 61.1% here, and 96.3% inside
a nominal 90% against 94.4%. The treatment effect is unchanged either way (the AME
median moves by 0.06 items on a +/-3-item interval), so this is a calibration fix,
not a change of result. R and EI were also swept and their priors do not bind, so
they keep the suite default.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipelines.itt import fit_itt

# Baseline nonverbal cognitive ability (block design), measured at t1 only.
ABILITY_ADJUSTER = (V.BLOCKS,)

SPEC = ModelSpec(
    model_id="lrp-rli-itt-022",
    kind="itt",
    title=(
        "Ability-adjusted available-case modified ITT estimate of the assigned-arm "
        "contrast in expressive vocabulary (EOWPVT) (E)"
    ),
    outcome_symbol="E",
    adjustment=list(ABILITY_ADJUSTER),
    model_settings=IttModelSettings(
        adjust_for=ABILITY_ADJUSTER,
        kappa_prior_family="halfnormal_inverse_sqrt",
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_itt(SPEC, config=config)
