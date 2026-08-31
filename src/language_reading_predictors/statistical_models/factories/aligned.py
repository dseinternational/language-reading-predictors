# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Onset-aligned per-protocol model construction.

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from typing import TYPE_CHECKING

import numpy as np
import pymc as pm

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    AlignedPayload,
)
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    beta_binomial_from_score_mean_link,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _alpha_sigma_for,
    _scalar_prior,
)

def build_aligned_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    ability_covariate: str | None = None,
    use_cohort: bool = True,
    use_dose: bool = False,
    likelihood: str = "beta_binomial",
    # #619: the phoneme-blending response link. ``"logit"`` is the ordinary
    # Beta-Binomial inverse-logit mean; ``"three_choice_guessing_floor"`` maps it
    # onto [1/3, 1] for the ten three-alternative forced-choice blending items,
    # whose expected score cannot fall below chance. B only, graded only, and
    # released only beside its paired ordinary-link fit.
    score_mean_link: ScoreMeanLink = "logit",
) -> BuiltModel[AlignedPayload]:
    """Per-protocol onset-aligned single-gain ANCOVA (LRPAL).

    A cross-sectional Beta-Binomial ANCOVA of the aligned post-score on its own
    onset baseline, age-at-onset and cognitive ability, plus -- optionally -- the
    cohort indicator (immediate vs wait-list) and the cumulative session dose.
    One row per child (``phase_mode="aligned"``), so there is **no child random
    intercept**.

    The cohort term (``beta_cohort``) is **not** a randomised effect: it contrasts
    the two arms at their own onset-aligned endpoints, confounded by age-at-onset
    and cohort/timing -- report it as a per-protocol association, never as the ITT
    treatment effect. ``use_dose`` adds a within-arm cumulative-session covariate,
    a collider descendant of group and ability -- a sensitivity variant, not the
    primary adjustment set.

    Under ``likelihood="bernoulli_offfloor"`` the ``own`` baseline term is the
    **binary off-floor-at-onset indicator** (raw 0/1, ``gamma_own_offfloor ~
    Normal(0, 1)``), not the graded onset logit: the Normal(1, 0.25) "post-logit
    tracks pre-logit 1:1" calibration is a graded test-retest fact that does not
    transfer to a Bernoulli off-floor outcome, and the onset logit of a heavily
    floored measure is a near-degenerate spike — the #391 finding-2 decision,
    adopted for this family by the 2026-08-21 aligned review (finding 2).
    """
    if prepared.phase_mode != "aligned":
        raise ValueError("build_aligned_model requires phase_mode='aligned'")
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    off_floor = likelihood == "bernoulli_offfloor"
    own = outcome_symbol
    if own not in prepared.post_counts or own not in prepared.pre_logit:
        raise KeyError(f"Outcome {own!r} needs pre+post scores in prepared data")
    if off_floor and own not in prepared.pre_counts:
        raise KeyError(
            f"Outcome {own!r} needs raw onset counts (pre_counts) for the "
            "off-floor-at-onset indicator; reload with load_and_prepare_aligned"
        )
    if ability_covariate is not None and ability_covariate not in prepared.covariates:
        raise KeyError(f"ability_covariate {ability_covariate!r} not in prepared.covariates")
    if use_dose and "dose" not in prepared.covariates:
        raise KeyError(
            "use_dose=True requires a 'dose' covariate "
            "(load with load_and_prepare_aligned(include_dose=True))"
        )

    keep = ~np.isnan(prepared.post_counts[own]) & ~np.isnan(prepared.pre_logit[own])
    if off_floor:
        keep = keep & ~np.isnan(np.asarray(prepared.pre_counts[own], dtype=float))
    prepared = _subset(prepared, keep)

    post = prepared.post_counts[own].astype(np.int64)
    cohort = prepared.G.astype(float)
    # Enter the own baseline on the *raw* logit scale, like every sibling factory
    # (ITT, mechanism, gain/level-factors, DiD): the ``gamma_own ~ Normal(1, 0.5)``
    # prior encodes "logit-post ≈ logit-pre" (a slope near 1 in logit units), which
    # only holds on the raw logit scale. Standardising the baseline here (as before)
    # left that prior mean of 1 meaning "1 logit per SD of baseline logit" — an
    # unintended, measure-dependent prior for this precision term.
    own_pre_logit = prepared.pre_logit[own]

    coords = {"obs_id": np.arange(prepared.n_obs)}
    with pm.Model(coords=coords) as model:
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")

        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
        if off_floor:
            # Binary off-floor-at-onset indicator (#391 finding 2, adopted for the
            # aligned floor rule by the 2026-08-21 review, finding 2): with most
            # children at the onset floor, the graded logit is a spike at the
            # Haldane floor value and the Normal(1, 0.25) tracking prior turns
            # into a strongly pessimistic implied intercept for at-floor children.
            own_off = (
                np.asarray(prepared.pre_counts[own], dtype=float) > 0
            ).astype(float)
            own_off_d = pm.Data("own_offfloor_pre", own_off, dims="obs_id")
            gamma_own_off = _priors.gamma_own_offfloor_prior().to_pymc(
                "gamma_own_offfloor"
            )
            eta = alpha + gamma_own_off * own_off_d + gamma_A * A_std_d
        else:
            own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")
            gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
            eta = alpha + gamma_own * own_pre_d + gamma_A * A_std_d

        if use_cohort:
            cohort_d = pm.Data("cohort", cohort, dims="obs_id")
            # The cohort association deliberately keeps the untiered proximal
            # tau scale (Normal(0, 0.5)) for every outcome, including the
            # distal-tier ones whose ITT tau is Normal(0, 0.3): the tier exists
            # to keep a *causal* prior's item-scale implications plausible,
            # whereas this term is a non-gated per-protocol association where
            # the wider prior is the conservative (less informative) choice —
            # and the family's psense diagnostics are clean under it
            # (2026-08-21 aligned review, finding 6). The intercept stays
            # tiered via _alpha_sigma_for because its item-scale argument is
            # about the level, not the contrast.
            beta_cohort = _priors.tau_prior().to_pymc(
                "beta_cohort",
                role="association",
                rationale=(
                    "Per-protocol cohort contrast (immediate versus wait-list at "
                    "aligned endpoints) carried on the treatment prior "
                    "tau ~ Normal(0, 0.5). NOT randomised: confounded by "
                    "age-at-onset and cohort timing, so no term in this family is "
                    "flagged causal."
                ),
            )
            eta = eta + beta_cohort * cohort_d
        if ability_covariate is not None:
            ability_d = pm.Data(
                f"{ability_covariate}_std",
                prepared.covariates[ability_covariate], dims="obs_id",
            )
            gamma_ability = _priors.gamma_cross_prior().to_pymc("gamma_ability")
            eta = eta + gamma_ability * ability_d
        if use_dose:
            dose_d = pm.Data("dose_std", prepared.covariates["dose"], dims="obs_id")
            gamma_dose = _priors.gamma_cross_prior().to_pymc("gamma_dose")
            eta = eta + gamma_dose * dose_d

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        if likelihood == "beta_binomial":
            kappa = _scalar_prior("kappa", _priors.kappa_prior)
            beta_binomial_from_score_mean_link(
                "y_post", eta, n_trials=prepared.n_trials[own], kappa=kappa,
                score_mean_link=score_mean_link,
                observed=post, dims="obs_id",
            )
        else:  # bernoulli_offfloor: exploratory estimand for floored outcomes (e.g. P)
            pm.Bernoulli(
                "y_offfloor", logit_p=eta,
                observed=(post > 0).astype(np.int64), dims="obs_id",
            )

    return BuiltModel(
        model=model,
        prepared=prepared,
        payload=AlignedPayload(score_mean_link=score_mean_link),
    )
