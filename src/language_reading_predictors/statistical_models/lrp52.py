# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP52 - ITT analysis of word reading (W)."""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp52",
    kind="itt",
    title="ITT effect of group assignment on word reading (W)",
    outcome_symbol="W",
    # HSGPs on age and own-baseline are disabled by default after the 2026-04-18
    # LRP52 sensitivity fit (notes/202604181445-lrp52-gp-sensitivity.md):
    # LOO prefers no-GP, τ is unchanged, and the GP-on variant produces ~1 %
    # divergences from the eta -> basis-weight funnel. Flags are kept so the
    # fallback can be re-enabled per-outcome if nonlinear tension appears.
    extra={"use_age_gp": False, "use_own_baseline_gp": False, "use_varying_tau": False},
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
