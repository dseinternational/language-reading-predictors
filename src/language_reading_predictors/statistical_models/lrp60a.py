# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP60a - matched complete-case comparator to LRP60.

LRP60 adjusts the word-reading ITT for SES (`MUMEDUPOST16`, `DADEDUPOST16`,
`AGEBOOKS`), but requesting those covariates triggers complete-case dropping in
`load_and_prepare`, so it runs on only ~33 of 54 children. A naive LRP52-vs-LRP60
comparison therefore confounds two things: the **SES adjustment** and the **sample
change** (the 21 dropped children).

LRP60a is the **unadjusted** word-reading ITT fit on the **exact same complete-case
subset** LRP60 uses. It adjusts for nothing (`adjust_for=()`) but restricts to the
SES complete cases (`restrict_complete=SES_ADJUSTERS`), so its `tau` is comparable
with the sample held fixed:

- **LRP60a vs LRP52** isolates the dropped-children / selection effect.
- **LRP60 vs LRP60a** isolates the SES adjustment, sample held fixed.

All other settings match LRP60 so the *only* difference from LRP60 is the
adjustment. Phase-0 (randomised) window only; n is small and intervals wide — this
is a robustness verdict, not a new headline.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.lrp60 import SES_ADJUSTERS
from language_reading_predictors.statistical_models.pipeline import fit_itt


SPEC = ModelSpec(
    model_id="lrp60a",
    kind="itt",
    title="Unadjusted ITT on the SES complete-case subset (matched comparator to LRP60)",
    outcome_symbol="W",
    adjustment=[],
    extra={
        # Match LRP60 exactly except the adjustment: same GP/tau toggles, no
        # adjusters, but restricted to the same SES complete-case rows.
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_varying_tau": False,
        "adjust_for": (),
        "restrict_complete": SES_ADJUSTERS,
        "variant_of": "lrp60",
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
