# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID105 - dispersion-prior sensitivity for LRPDID10 (basic concept knowledge, F).

The **low-denominator** half of the family's dispersion check (#576 material
qualification 2); LRPDID106 is the high-denominator half.

Every graded DiD fit takes ``kappa ~ HalfNormal(50)`` on the Beta-Binomial
concentration. A half-normal on the concentration cannot reach the near-Binomial
limit ``kappa >> n``, so for a long test it imposes a *floor* on the estimated
over-dispersion: at ``n = 170`` the prior median already implies roughly 5.9 times
Binomial variance, and the model cannot conclude that a child's score is close to
Binomial even if the data say so. The same prior on an 18-item test is far more
permissive, because a modest ``kappa`` is already large relative to ``n``.

This companion refits LRPDID10 with the dispersion-scale parameterisation the ITT and
level families use — ``1 / sqrt(kappa) ~ HalfNormal(0.25)``, with ``kappa`` retained
as a Deterministic — which *can* reach the near-Binomial limit. Nothing else changes.

Read the pair LRPDID105/LRPDID106 together. If the arm gaps are stable at both
denominators, the concentration prior is a nuisance choice the conclusions do not
turn on. If the high-denominator fit moves and the low-denominator one does not, the
prior's ceiling — not the data — was setting the dispersion, and the family default
should be revisited rather than the individual result reinterpreted.

Reading rules are LRPDID10's; the dispersion prior changes no term's causal status.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.did import DiDModelSettings
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-105",
    kind="did",
    title=(
        "Dispersion-prior sensitivity for the basic-concepts arm-by-wave contrasts "
        "(CELF) (F)"
    ),
    outcome_symbol="F",
    family="did",
    design="waitlist-crossover arm-by-wave levels, inverse-sqrt dispersion prior",
    estimand_type="mixed",
    causal_status="t2 randomised; t3 a randomised treatment-schedule contrast",
    model_settings=DiDModelSettings(
        # Identical to LRPDID10 except kappa_prior_family.
        outcomes=("F",),
        waves=(0, 1, 2),
        use_child_re=True,
        use_age=True,
        dose=False,
        kappa_prior_family="halfnormal_inverse_sqrt",
    ),
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
