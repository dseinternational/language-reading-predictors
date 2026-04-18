# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP54 - ITT analysis of expressive vocabulary (E)."""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp54",
    kind="itt",
    title="ITT effect of group assignment on expressive vocabulary (E)",
    outcome_symbol="E",
    # HSGPs disabled by default; see notes/202604181445-lrp52-gp-sensitivity.md
    # for the rationale. Re-enable via the extra dict if LRP54 PPC shows
    # nonlinear tension in age or own baseline.
    extra={"use_age_gp": False, "use_own_baseline_gp": False, "use_varying_tau": False},
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
