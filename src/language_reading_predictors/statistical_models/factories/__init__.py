# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Model construction, one module per family.

**A re-export facade since #637 stage 3.** The 8,506-line ``factories.py`` this
replaces was one of the three dependency hubs the maintainability review named. It
is now twenty modules: :mod:`~factories.base` for the pieces more than one family
needs, and one per family for the rest. Every family module depends only on
``base``; nothing crosses between families.

Two helpers moved into ``base`` to make that true: ``_bivariate_lkj_residual``,
which the joint-mechanism design reuses from the joint family, and
``_resolve_adjusted_predictor``, which the horseshoe family reuses from the
adjusted one.

Every name is re-exported here so existing call sites keep working. #637 asks for
exactly that ("split factories by family behind temporary re-exports"), and
*temporary* is the operative word: new code should import from the owning module.
"""

from __future__ import annotations

# Pass-throughs the pre-split module also exposed: several call sites and tests
# reach these on ``factories`` rather than at their owner.
from language_reading_predictors.statistical_models.fitted_payloads import (  # noqa: F401
    MechanismDesign,
)
from language_reading_predictors.statistical_models.preprocessing import (  # noqa: F401
    _subset,
)

from language_reading_predictors.statistical_models.factories.adjusted import (  # noqa: F401
    build_adjusted_model,
    build_rlm_adjusted_model,
    build_rlm_transition_adjusted_model,
)

from language_reading_predictors.statistical_models.factories.aligned import (  # noqa: F401
    build_aligned_model,
)

from language_reading_predictors.statistical_models.factories.base import (  # noqa: F401
    _t1_language_composite,
    BuiltModel,
    PayloadT,
    RequiredPayloadT,
    _LCF_DOMAINS,
    _MECH_HSGP_C,
    _MECH_HSGP_M,
    _add_child_random_intercept,
    _alpha_sigma_for,
    _bivariate_lkj_residual,
    _broadcast_phase_zero,
    _broadcast_phase_zero_optional,
    _interaction_product,
    _resolve_adjusted_predictor,
    _rlm_dispersion_kappa,
    _rlm_group_nuisance,
    _scalar_prior,
    _standardise_child_baseline,
    _tau_sigma_for,
    default_of,
)

from language_reading_predictors.statistical_models.factories.block_exposure import (  # noqa: F401
    build_block_exposure_model,
)

from language_reading_predictors.statistical_models.factories.concurrent import (  # noqa: F401
    build_concurrent_model,
    build_rlm_concurrent_model,
)

from language_reading_predictors.statistical_models.factories.corr_factor import (  # noqa: F401
    build_correlated_factor_model,
    build_rlm_corr_factor_model,
)

from language_reading_predictors.statistical_models.factories.did import (  # noqa: F401
    build_did_model,
)

from language_reading_predictors.statistical_models.factories.dose_response import (  # noqa: F401
    build_dose_response_model,
)

from language_reading_predictors.statistical_models.factories.gain_factors import (  # noqa: F401
    build_gain_factors_model,
)

from language_reading_predictors.statistical_models.factories.growth import (  # noqa: F401
    build_growth_model,
)

from language_reading_predictors.statistical_models.factories.historical import (  # noqa: F401
    _map_panel_rows,
    build_historical_growth_model,
    build_rlm_joint_growth_model,
)

from language_reading_predictors.statistical_models.factories.horseshoe import (  # noqa: F401
    _build_horseshoe_betas,
    _resolve_level_predictor,
    build_horseshoe_model,
    build_rlm_horseshoe_model,
)

from language_reading_predictors.statistical_models.factories.itt import (  # noqa: F401
    build_itt_model,
)

from language_reading_predictors.statistical_models.factories.joint import (  # noqa: F401
    build_joint_model,
)

from language_reading_predictors.statistical_models.factories.joint_mechanism import (  # noqa: F401
    _add_decoding_contrast_deterministics,
    build_joint_mechanism_model,
)

from language_reading_predictors.statistical_models.factories.lcsm import (  # noqa: F401
    build_lcsm_model,
)

from language_reading_predictors.statistical_models.factories.level_factors import (  # noqa: F401
    build_level_factors_model,
)

from language_reading_predictors.statistical_models.factories.long_corr_factor import (  # noqa: F401
    build_longitudinal_corr_factor_model,
)

from language_reading_predictors.statistical_models.factories.mechanism import (  # noqa: F401
    build_mechanism_model,
)

from language_reading_predictors.statistical_models.factories.mediation import (  # noqa: F401
    MediationData,
    PeriodStackedMediationData,
    TwoMediatorData,
    _add_cross_baselines,
    _baseline_confounder_value,
    _build_outcome_leg,
    _build_route_composite,
    _build_route_composite_model,
    _cross_baseline_arrays,
    build_mediation_model,
    build_period_stacked_mediation_model,
    build_two_mediator_model,
)

__all__ = [
    "BuiltModel",
    "MechanismDesign",
    "MediationData",
    "PayloadT",
    "PeriodStackedMediationData",
    "RequiredPayloadT",
    "TwoMediatorData",
    "build_adjusted_model",
    "build_aligned_model",
    "build_block_exposure_model",
    "build_concurrent_model",
    "build_correlated_factor_model",
    "build_did_model",
    "build_dose_response_model",
    "build_gain_factors_model",
    "build_growth_model",
    "build_historical_growth_model",
    "build_horseshoe_model",
    "build_itt_model",
    "build_joint_mechanism_model",
    "build_joint_model",
    "build_lcsm_model",
    "build_level_factors_model",
    "build_longitudinal_corr_factor_model",
    "build_mechanism_model",
    "build_mediation_model",
    "build_period_stacked_mediation_model",
    "build_rlm_adjusted_model",
    "build_rlm_concurrent_model",
    "build_rlm_corr_factor_model",
    "build_rlm_horseshoe_model",
    "build_rlm_joint_growth_model",
    "build_rlm_transition_adjusted_model",
    "build_two_mediator_model",
    "default_of",
]
