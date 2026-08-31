# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Whether a fit may publish, and why.

**A re-export facade since #637 stage 3c.** The 4,130-line ``release.py`` this
replaces was the third of the dependency hubs the maintainability review named. It
is now six modules, in the order the decision is actually made:

* :mod:`~release.base` - the readers, filenames and thresholds every check shares.
* :mod:`~release.robustness` - the treatment-effect gate: whether a causal headline
  survives its prior-sensitivity and floor-grid evidence.
* :mod:`~release.blending` - the phoneme-blending response-link pair gates.
* :mod:`~release.family_checks` - per-family checks over a stored fit directory.
* :mod:`~release.dependence` - joint dependence pairing and its measured consequence.
* :mod:`~release.publication` - the ordered decision that reads all of the above.

The edges run one way: each check reads ``base``, only ``publication`` reads the
checks, and the checks do not read each other. That was not true before the split -
the shared readers lived beside the decision, so every check depended on the module
that depended on it.

Every name is re-exported here so existing call sites keep working; it is a
temporary compatibility seam, not an architecture.
"""

from __future__ import annotations

from language_reading_predictors.statistical_models.release.base import (  # noqa: F401
    GATED_KINDS,
    GROWTH_INFLUENCE_TRACE_FILENAME,
    JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS,
    MEDIATION_T3_TRACE_FILENAME,
    PSENSE_THRESHOLD,
    PublicationStatus,
    RELEASE_DECISION_FILENAME,
    ReleaseStage,
    ReleaseStatus,
    TauSensitivityClass,
    _HISTORICAL_JOINT_PRIOR_BINDING,
    _HISTORICAL_JOINT_PRIOR_SENSITIVE,
    _JOINT_PAIR_BINDING,
    _config_name,
    _finite,
    _load_config,
    _model_tier,
    _plan,
    _read_csv,
    _read_json,
    _stored_bool,
)

from language_reading_predictors.statistical_models.release.robustness import (  # noqa: F401
    ReleaseDecision,
    _AME_CORRELATION_NOISE,
    _CLASS_SEVERITY,
    _CONTRAST_DIRECTION_SHIFT,
    _PRIOR_ATTENUATION_NOTE,
    _QUALIFY_NOTE,
    _WITHHOLD_TIERS,
    _element_rows,
    _floor_decision,
    _gain_offfloor_decision,
    _standard_sweep_evidence,
    _tau_row,
    causal_term_for,
    classify_tau_sensitivity,
    evaluate_itt_release,
    evaluate_release,
    gate_applies,
    sweep_sign_column,
)

from language_reading_predictors.statistical_models.release.blending import (  # noqa: F401
    _BLENDING_PAIR_GATES,
    _aligned_blending_pair_release_failures,
    _blending_pair_release_failures,
    _concurrent_blending_pair_release_failures,
    _did_blending_pair_release_failures,
    _dose_blending_pair_release_failures,
    _gain_blending_pair_release_failures,
    _joint_blending_scope_note,
    _level_blending_pair_release_failures,
    _mediation_blending_pair_release_failures,
)

from language_reading_predictors.statistical_models.release.family_checks import (  # noqa: F401
    _MISSINGNESS_BFMI_MIN,
    _MISSINGNESS_DIAGNOSTIC_FIELDS,
    _MISSINGNESS_ESS_MIN,
    _MISSINGNESS_RHAT_MAX,
    _adjusted_ses_release_failures,
    _concurrent_published_fit_release_failures,
    _gain_period1_release_failures,
    _growth_influence_release_failures,
    _itt_missingness_release_failures,
    _joint_mechanism_coverage_qualifications,
    _joint_mechanism_wave_release_failures,
    _mediation_t3_release_failures,
    _missingness_design_dimension_error,
    _missingness_diagnostics,
    _missingness_diagnostics_match,
    _missingness_diagnostics_pass,
    _missingness_trace_diagnostics,
    _trailing_size,
)

from language_reading_predictors.statistical_models.release.dependence import (  # noqa: F401
    _dependence_identification_note,
    _historical_joint_prior_companion_qualifications,
    _historical_joint_prior_sensitivity,
    _historical_joint_resolvability_change,
    _joint_contrast_consequence,
    _joint_dependence_companion_note,
    _joint_marginal_widths,
    _joint_width_channels,
    _required_dependence_companion,
)

from language_reading_predictors.statistical_models.release.publication import (  # noqa: F401
    JOINT_MECHANISM_WAVE_MARGINAL_PPC,
    JOINT_MECHANISM_WAVE_PSENSE,
    JOINT_MECHANISM_WAVE_TRACE,
    ReleaseEvaluation,
    _CORE_ARTIFACTS_BASE,
    _DIAGNOSTIC_CONFIGS,
    _PUBLICATION_CONFIGS,
    _core_artifact_failures,
    _prior_evidence_qualifications,
    _publication_input_failures,
    _recorded_required_artifacts,
    _robustness_decision,
    _sampling_preset_qualification,
    evaluate_publication,
    write_release_decision,
)

__all__ = [
    "GATED_KINDS",
    "GROWTH_INFLUENCE_TRACE_FILENAME",
    "JOINT_MECHANISM_MARGINAL_COVERAGE_FLOORS",
    "JOINT_MECHANISM_WAVE_MARGINAL_PPC",
    "JOINT_MECHANISM_WAVE_PSENSE",
    "JOINT_MECHANISM_WAVE_TRACE",
    "MEDIATION_T3_TRACE_FILENAME",
    "PSENSE_THRESHOLD",
    "PublicationStatus",
    "RELEASE_DECISION_FILENAME",
    "ReleaseDecision",
    "ReleaseEvaluation",
    "ReleaseStage",
    "ReleaseStatus",
    "TauSensitivityClass",
    "causal_term_for",
    "classify_tau_sensitivity",
    "evaluate_itt_release",
    "evaluate_publication",
    "evaluate_release",
    "gate_applies",
    "sweep_sign_column",
    "write_release_decision",
]
