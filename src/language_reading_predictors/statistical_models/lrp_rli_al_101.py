# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPAL01d - aligned-40-week dose sensitivity variant for word reading (W).

The LRPAL01 per-protocol aligned ANCOVA with the cumulative sessions delivered
over the aligned window added as a covariate (``use_dose=True``).

Dose is a **collider** on the DAG -- a descendant of both group (the immediate
arm accrues sessions earlier) and ability (more able / available children attend
more) -- so conditioning on it can open a back-door. This is therefore a
**sensitivity** model, *not* a primary adjustment set: read ``gamma_dose`` as a
within-arm dose-response association expected to be weak / inconclusive once the
onset baseline, age-at-onset and ability are already in the model (cf. the
Phase-0b dose checks), never as a causal dose effect. As with LRPAL01, the cohort
term is non-randomised and every coefficient is an association.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_aligned

SPEC = ModelSpec(
    model_id="lrp-rli-al-101",
    kind="aligned",
    title="Aligned-40-week dose sensitivity variant for word reading (W)",
    outcome_symbol="W",
    extra={
        "ability_covariate": V.BLOCKS,
        "use_cohort": True,
        "use_dose": True,
    },
)


def fit(config: str = "dev"):
    return fit_aligned(SPEC, config=config)
