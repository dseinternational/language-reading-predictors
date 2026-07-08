# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF01b - gain factors for word reading (W), treated-only ("gains while on
intervention").

The LRPGF01 model restricted to on-intervention period rows (excluding the
waitlist arm's untreated period 1). Every remaining row is on intervention, so the
treatment indicator and its interactions are constant and are dropped: this is the
factor-association model *among the treated*, not a treatment-effect model. The
remaining factors (own baseline, age, ability, skills L/R, age x ability) are
adjusted associations under the DAG. Compare its factor associations with LRPGF01.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-101",
    kind="gain_factors",
    title="Gain factors for word reading (W), treated-only (gains while on intervention)",
    outcome_symbol="W",
    extra={
        "skill_symbols": ("L", "R"),
        "ability_covariate": V.BLOCKS,
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": True,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
