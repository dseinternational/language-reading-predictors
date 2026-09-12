# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compatibility imports for older analysis scripts and notebooks.

New code imports shared helpers from ``posteriors`` and family calculations
from ``summaries.<family>``.
"""

from language_reading_predictors.statistical_models.posteriors import (
    REPORTING_CI_PROB as REPORTING_CI_PROB,
    band50 as band50,
    derived_mc_diagnostics as derived_mc_diagnostics,
    loo_delta as loo_delta,
    beta_summary as beta_summary,
    coef_row as coef_row,
)

from language_reading_predictors.statistical_models.summaries.block_exposure import (
    block_exposure_summary as block_exposure_summary,
)

from language_reading_predictors.statistical_models.summaries.concurrent import (
    ConcurrentTerm as ConcurrentTerm,
    concurrent_marginals as concurrent_marginals,
)

from language_reading_predictors.statistical_models.summaries.dependence import (
    DEPENDENCE_PRIOR_DOMINATED_RATIO as DEPENDENCE_PRIOR_DOMINATED_RATIO,
    DEPENDENCE_INFORMED_RATIO as DEPENDENCE_INFORMED_RATIO,
    _dependence_verdict as _dependence_verdict,
    dependence_identification_summary as dependence_identification_summary,
)

from language_reading_predictors.statistical_models.summaries.did import (
    did_summary as did_summary,
    did_cell_ppc as did_cell_ppc,
    did_within_child_ppc as did_within_child_ppc,
)

from language_reading_predictors.statistical_models.summaries.factors import (
    factor_summary as factor_summary,
    AssociationTerm as AssociationTerm,
    association_marginals as association_marginals,
)

from language_reading_predictors.statistical_models.summaries.gain_factors import (
    treatment_marginal_effect as treatment_marginal_effect,
)

from language_reading_predictors.statistical_models.summaries.growth import (
    growth_association_summary as growth_association_summary,
)

from language_reading_predictors.statistical_models.summaries.horseshoe import (
    horseshoe_ranking as horseshoe_ranking,
)

from language_reading_predictors.statistical_models.summaries.itt import (
    _itt_ame_draws as _itt_ame_draws,
    tau_summary_itt as tau_summary_itt,
    tau_summary_offfloor as tau_summary_offfloor,
    offfloor_mover_table as offfloor_mover_table,
    tau_moderation_summary as tau_moderation_summary,
)

from language_reading_predictors.statistical_models.summaries.joint import (
    _joint_observed_row_masks as _joint_observed_row_masks,
    _joint_ame_draws as _joint_ame_draws,
    tau_summary_joint as tau_summary_joint,
    joint_treatment_marginals as joint_treatment_marginals,
    gamma_interaction_summary as gamma_interaction_summary,
    tau_contrast_matrix as tau_contrast_matrix,
    tau_difference_summary as tau_difference_summary,
)

from language_reading_predictors.statistical_models.summaries.level_factors import (
    level_t2_marginal_effect as level_t2_marginal_effect,
    level_window_comparator_cards as level_window_comparator_cards,
)

from language_reading_predictors.statistical_models.summaries.long_corr_factor import (
    _factor_corr_draws as _factor_corr_draws,
    longitudinal_factor_correlations as longitudinal_factor_correlations,
    longitudinal_conditional_slopes as longitudinal_conditional_slopes,
    disattenuation_crosscheck as disattenuation_crosscheck,
)

from language_reading_predictors.statistical_models.summaries.readiness import (
    _KNEE_MIN_INCREASING as _KNEE_MIN_INCREASING,
    _KNEE_MIN_CURVATURE as _KNEE_MIN_CURVATURE,
    _readiness_knee as _readiness_knee,
    readiness_threshold as readiness_threshold,
)

from language_reading_predictors.statistical_models.summaries.rope import (
    rope_markdown as rope_markdown,
    drop_retired_90_band as drop_retired_90_band,
    rope_summary as rope_summary,
    rope_sensitivity as rope_sensitivity,
    rope_sensitivity_markdown as rope_sensitivity_markdown,
)
