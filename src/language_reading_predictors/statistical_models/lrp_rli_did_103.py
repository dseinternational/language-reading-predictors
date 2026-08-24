# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID103 - guessing-floor response-link companion to LRPDID03 (phoneme blending, B).

Phoneme blending is a ten-item, **three-alternative forced-choice** test, so a child
answering at random scores about 3.3 of 10. LRPDID03 fits the ordinary Beta-Binomial
inverse-logit score mean, which places no floor on the fitted mean at all: it can put
posterior mass on expected scores below chance, which no mechanism of the test can
produce. This companion is identical in every respect except the score-mean link,
which maps the mean onto ``[1/3, 1]``::

    mu = 1/3 + (2/3) * inverse_logit(eta)

The methodology has required that pairing for any headline ``B`` interpretation since
the ITT suite adopted it, and it is material rather than cosmetic: on the ITT side the
same sensitivity roughly **halved** the item-scale estimate. Until #576 the DiD family
had no version of it, so LRPDID03 could publish an unqualified ``B`` headline. The ITT
companion (LRPITT108) does not stand in for this one: the arm-by-wave likelihood is a
longitudinal random-intercept model over t1/t2/t3, so t1 and t3 data inform the t2
posterior and the two fits' response-link sensitivities are not interchangeable.

**Neither fit releases alone.** ``release.evaluate_publication`` requires the current,
converged, provenance-matched twin beside whichever side is being published, and the
resolved run plan of both records the requirement. The empirical-Bayes intercept anchor
is inverted through the same link, so it still locates the *linear predictor* rather
than a raw observed logit — for a pooled t1 blending proportion near 0.49 those differ
by more than a logit unit.

Reading rules are LRPDID03's. ``tau_t2`` is the randomised immediate-versus-not-yet
assignment contrast at t2; ``arm_gap_t1`` is a pre-randomisation balance quantity;
``arm_gap_t3`` is the randomised early-start-versus-delayed-start schedule contrast and
``delta_crossover`` the change between the two, neither of them mechanism-identified.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.did import DiDModelSettings
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-103",
    kind="did",
    title=(
        "Guessing-floor response-link sensitivity for the phoneme-blending "
        "arm-by-wave contrasts (B)"
    ),
    outcome_symbol="B",
    family="did",
    design="waitlist-crossover arm-by-wave levels, three-choice guessing-floor link",
    estimand_type="mixed",
    causal_status="t2 randomised; t3 a randomised treatment-schedule contrast",
    model_settings=DiDModelSettings(
        # Identical to LRPDID03 except score_mean_link.
        outcomes=("B",),
        waves=(0, 1, 2),
        use_child_re=True,
        use_age=True,
        dose=False,
        score_mean_link="three_choice_guessing_floor",
    ),
)


def fit(config: str = "dev"):
    return fit_did(SPEC, config=config)
