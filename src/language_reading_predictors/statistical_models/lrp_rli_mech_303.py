# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP303 - phase-stability sensitivity for the L -> R linear slope (#604).

The vocabulary channel of the #604 phase-stability pair. Companion to **LRP97**,
identical in rows, outcome, baseline, adjustment set and priors, differing only in
that the exposure slope is allowed to vary by period transition, partially pooled
toward a shared mean (``beta_mech_phase = mu_mech + sigma_mech_phase * z_phase``).
``lrp-rli-mech-097`` is the pooled comparator for the nested PSIS-LOO test.

**Why this channel too.** LRP97 is a Tier-1 *negative control*: under the revised DAG
receptive vocabulary is not a causal descendant of letter-sound knowledge, so any
adjusted ``L -> R`` association is purely backdoor. Pooling three treatment histories
matters at least as much for a backdoor association as for a substantive one -
period-varying confounding is precisely the thing a stable pooled slope could be
hiding - and the vocabulary models have the family's widest denominators (170 items),
where a slope difference translates into the largest item-scale difference.

Everything said about LRP302 applies here: partial pooling rather than three free
slopes at ~52 rows per period; the per-period exposure support is published beside
the slopes; and a period difference is evidence against pooling, not evidence about
mechanism change over time. Every per-period slope is an adjusted association, and on
this outcome the pooled slope was expected to be near zero in the first place.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-303",
    kind="mechanism",
    title="Phase-varying slope: letter sounds (L) -> receptive vocabulary (R)",
    outcome_symbol="R",
    mechanism_symbol="L",
    # The non-centred per-period slope vector is a small hierarchical funnel:
    # ``sigma_mech_phase`` is informed by three periods, so its lower tail pinches the
    # geometry and the preset 0.95 leaves a handful of divergences. 0.995 clears them
    # (verified at rep-lite: 8 divergences at 0.90, 2 at 0.95, 0 at 0.995). This is a
    # *raise* over the preset for a specific, diagnosed geometry — never a blanket
    # escalation, which can lower an in-module default and manufacture a false pass.
    target_accept=0.995,
    adjustment=["G", "A", "R_pre"],
    model_settings=MechanismModelSettings(
        adjust_baseline_symbol="R",
        outcomes=("R", "L"),
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        linear_mechanism=True,
        phase_varying_slope=True,
        use_age_gp=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mechanism(SPEC, config=config)
