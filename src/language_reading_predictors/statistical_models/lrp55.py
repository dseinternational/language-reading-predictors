# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP55 - Joint ITT model over eight outcomes with LKJ(4) correlation."""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES
from language_reading_predictors.statistical_models.pipeline import fit_joint

SPEC = ModelSpec(
    model_id="lrp55",
    kind="joint",
    title="Joint outcome ITT model (W, R, E, L, P, B, F, T)",
    # Age GP and LKJ residual correlation are both OFF by default after the
    # 2026-04-18 LRP55 diagnostics (see
    # notes/202604181600-lrp52-58-findings.md,
    # notes/202604181630-lrp55-lkj-drop-and-comparison-fix.md and
    # notes/202604181700-lrp55-age-gp-drop.md). The LKJ block was
    # prior-dominated; the age-GP amplitudes were the source of the
    # remaining ~8 % divergent transitions and LOO did not prefer the
    # GP-on variant. Toggle either flag via the extra dict for explicit
    # sensitivity fits.
    extra={
        "outcomes": ITT_OUTCOMES,
        "use_age_gp": False,
        "partial_pool_age_gp": True,
        "use_residual_correlation": False,
    },
)


def fit(config: str = "dev"):
    return fit_joint(SPEC, config=config)
