# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP52d - Period-1 ITT for word reading (W) entering both group and dose.

The #104 Phase-2 ITT restatement. The period-1 ITT is already identified by
randomisation regardless of pooling, so the new value is narrow: make the
**dose-absorbs-group** finding from the Phase-1 GB diagnostic (Q3) explicit. This
is the LRP52 ITT model for W with the period-1 dose (``attend`` at t1) added as a
single linear adjuster alongside the randomised group term, via ``fit_itt`` +
``extra["adjust_for"]`` - no new machinery.

Read against LRP52 (group only):

- ``tau`` is the coefficient on ``G``. **G is coded G=1 = waitlist control,
  G=0 = immediate intervention** (``preprocessing.py`` maps dataset group 1
  [Initial intervention] -> 0 and group 2 [Wait] -> 1; note its inline comment
  is inverted). So a **negative ``tau`` means the immediate arm scores higher**
  (treatment benefit), and ``P(tau > 0)`` is P(waitlist higher), *not*
  P(treatment helps). This is the randomised, primary causal estimand.
- ``gamma_attend`` is the period-1 dose slope. In period 1 ``attend`` and ``G``
  are near-collinear (controls have ``attend == 0``), so entering both splits the
  one randomised contrast between them: the continuous dose carries the signal
  the binary indicator otherwise would. Report the two side by side as the
  mechanism (dose) vs the estimand (assignment), not as independent effects.

Caveat: the period-1 dose is observational within the assignment; this cell is an
interpretive restatement of the randomised effect, not a second causal estimand.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp52d",
    kind="itt",
    title="ITT (W) with period-1 dose - group tau vs dose restatement",
    outcome_symbol="W",
    extra={
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_varying_tau": False,
        "adjust_for": ("attend",),
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
