# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Block-design exposure model construction.

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from typing import TYPE_CHECKING, Iterable

import numpy as np
import pymc as pm

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    EmptyPayload,
)
from language_reading_predictors.statistical_models.likelihood import (
    beta_binomial_from_logit,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _add_child_random_intercept,
    _alpha_sigma_for,
    _scalar_prior,
    _tau_sigma_for,
)

def build_block_exposure_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    ability_covariate: str | None = None,
    adjust_for: Iterable[str] = (),
    use_child_re: bool = True,
    likelihood: str = "beta_binomial",
    sigma_child_prior_sigma: float = 0.5,
    # #382 recommendation 4: optional one-off wider prior on the focal
    # block-active exposure effect (LRPBX103). None keeps the outcome-tier
    # default (UE2/UR2 are distal, 0.3); the companion sets 0.5 to confirm the
    # distal tightening is not attenuating a real transfer effect.
    delta_prior_sigma: float | None = None,
) -> BuiltModel[EmptyPayload]:
    """Block-2 taught-vocabulary staggered block-active exposure model (LRPBX, #228 item 5).

    Block 2 of the taught-vocabulary programme (TE2/TR2 taught, UE2/UR2 not-taught)
    is introduced in phase 2, so it has **no t1 baseline and no randomised
    contrast**. Identification borrows the *staggered* block-2 teaching: the
    immediate arm is taught block 2 in phase 2 (measured at t3) while the wait-list
    arm is still on block 1; the wait-list arm reaches block 2 in phase 3 (measured
    at t4). This is a staggered-adoption / event-study design over the per-timepoint
    levels frame (``phase_mode="levels"``; t1 rows self-drop, block 2 being NaN
    there). Linear predictor (logit scale):

        eta = alpha + alpha_time[t]              # secular trend, per timepoint
            + delta * exposed_{i,t}              # <-- focal block-active exposure effect
            + gamma_A * A_std_t                  # age at t (precision)
            [ + gamma_ability * z(ability) ]     # cognitive-ability precision
            [ + sum_c gamma_c * z(c) ]           # revised-DAG adjusters (hs / erbto / deapp_c)
            [ + u_child_i ]                       # each child their own control

    where ``exposed_{i,t} = 1`` once child *i* has been taught block 2 by timepoint
    *t* — immediate arm (``G == 1``) from t3 (phase >= 2), wait-list arm (``G == 0``)
    from t4 (phase >= 3). The switch-on differs between arms only at t3 (immediate
    exposed, wait-list not), which is the identifying cell; both are exposed by t4
    and neither at t2.

    **Estimand and status.** ``delta`` is the shift in the logit of the block-2
    taught-vocabulary level attributable to block 2 being *actively taught*. It is
    an **association (parallel-trends)**, not a randomised effect: it is causal only
    if block-2 trajectories would have been parallel across arms absent block-2
    teaching. Over t2->t3 the wait-list arm is on **block 1**, not idle, so the
    contrast is "block-2-active *vs block-1-active*", never treated-vs-untreated.
    The child random intercept absorbs stable between-child (hence between-arm)
    differences and ``alpha_time`` absorbs the shared maturation trend; what remains
    confounded is age-at-block-2 (immediate children reach block 2 younger; linear
    ``gamma_A`` is only a precision term) and any arm-specific slope difference.
    Report ``delta`` accordingly (``Status.ASSOCIATION``). The taught outcomes
    (TE2/TR2) should show a positive ``delta`` and the not-taught comparators
    (UE2/UR2) should not — a within-family placebo / fidelity check.

    Structural siblings: the levels frame, ``alpha_time`` and the child intercept
    are the level-factors model's (``build_level_factors_model``); the ``eta_base``
    + ``delta * exposed`` split and the ``tau_prior(sigma=_tau_sigma_for(...))``
    effect prior are the DiD model's (``build_did_model``). ``delta`` carries the
    treatment-tier prior (it is the focal effect, mirroring the DiD ``delta`` /
    level-factors group term), not the tighter ``gamma_cross`` adjuster prior.

    ``adjust_for`` (default ()): revised-DAG confounders that are not bounded-count
    measures — hearing status (``hs`` / ``hs_missing``), speech production
    (``deapp_c`` / ``deapp_c_missing``, expressive outcomes) and phonological memory
    (``erbto`` / ``erbto_missing``) — each a linear ``gamma_{c}`` term with the
    cross-coupling prior, mirroring the gain/level-factor adjuster path (#247). No
    measure-skill adjusters and no own-baseline term: a levels model conditioning on
    a contemporaneous evolving skill would condition on a post-treatment
    mediator/collider, and a block-2 own-baseline does not exist (NaN at t1).
    """
    if prepared.phase_mode != "levels":
        raise ValueError("build_block_exposure_model requires phase_mode='levels'")
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    own = outcome_symbol
    if own not in prepared.post_counts:
        raise KeyError(f"Outcome {own!r} missing from prepared data (post_counts)")
    if ability_covariate is not None and ability_covariate not in prepared.covariates:
        raise KeyError(f"ability_covariate {ability_covariate!r} not in prepared.covariates")
    adjust_for = tuple(adjust_for)
    for c in adjust_for:
        if c not in prepared.covariates:
            raise KeyError(f"Adjuster covariate {c!r} not loaded in prepared data")

    keep = ~np.isnan(prepared.post_counts[own])
    for c in adjust_for:
        keep = keep & ~np.isnan(prepared.covariates[c])
    prepared = _subset(prepared, keep)

    post = prepared.post_counts[own].astype(np.int64)
    # Staggered block-2 exposure: immediate arm (G==1) taught block 2 from t3
    # (phase >= 2), wait-list arm (G==0) from t4 (phase >= 3). Derived from the
    # design (G, phase), like build_did_model's ``treated``.
    exposed = (
        ((prepared.G == 1) & (prepared.phase >= 2))
        | ((prepared.G == 0) & (prepared.phase >= 3))
    ).astype(float)
    ability = prepared.covariates[ability_covariate] if ability_covariate is not None else None

    # Only the phases this fit actually observes get a wave intercept (#631
    # finding 17). Block-2 outcomes are absent at t1, so those rows self-drop and
    # a full-width phase vector would carry an element the likelihood never sees —
    # which leaves a translation ridge even under a zero-sum constraint, because
    # the data-free element can absorb the compensating shift.
    observed_phases = np.unique(np.asarray(prepared.phase).astype(np.int64))
    phase_position = {int(p): i for i, p in enumerate(observed_phases)}
    phase_idx = np.array(
        [phase_position[int(p)] for p in np.asarray(prepared.phase)], dtype=np.int64
    )
    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": observed_phases,
        "child": np.arange(prepared.n_children),
    }
    with pm.Model(coords=coords) as model:
        phase_d = pm.Data("phase_idx", phase_idx, dims="obs_id")
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        exposed_d = pm.Data("exposed", exposed, dims="obs_id")
        adjust_d = {
            c: pm.Data(f"{c}_adj", prepared.covariates[c], dims="obs_id")
            for c in adjust_for
        }

        # Own-baseline-free level model. A free ``alpha`` beside a free per-phase
        # ``alpha_time`` left only their sums in the likelihood, so the split
        # between them was decided by the priors rather than the data (#631
        # finding 17) — the same translation ridge #389 finding 2 removed from
        # build_level_factors_model. ``alpha`` now carries the absolute level
        # (this family has no untreated t1 wave to anchor on, unlike the level
        # factors) and ``alpha_time`` is an exact zero-sum deviation vector over
        # the observed phases, so both are identified.
        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
        alpha_time = pm.ZeroSumNormal("alpha_time", sigma=0.5, dims="phase")
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
        eta = alpha + alpha_time[phase_d] + gamma_A * A_std_d

        if ability is not None:
            ability_d = pm.Data(f"{ability_covariate}_std", ability, dims="obs_id")
            gamma_ability = _priors.gamma_cross_prior().to_pymc("gamma_ability")
            eta = eta + gamma_ability * ability_d

        # Raw-covariate adjusters (revised-DAG exogenous confounders HS/SP/RW): linear
        # gamma terms, mirroring the gain/level-factor adjuster path (#247).
        for c in adjust_for:
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}")
            eta = eta + gamma_c * adjust_d[c]

        if use_child_re:
            child_idx_d = pm.Data(
                "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
            )
            eta = _add_child_random_intercept(
                eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
            )

        # Linear predictor without the exposure term, so the pipeline can read the
        # un-exposed baseline for the average-marginal-effect translation (as DiD).
        eta_base = pm.Deterministic("eta_base", eta, dims="obs_id")
        delta = _priors.tau_prior(
            sigma=_tau_sigma_for(outcome_symbol, delta_prior_sigma)
        ).to_pymc("delta")
        eta_full = pm.Deterministic("eta", eta_base + delta * exposed_d, dims="obs_id")

        if likelihood == "beta_binomial":
            kappa = _scalar_prior("kappa", _priors.kappa_prior)
            beta_binomial_from_logit(
                "y_post", eta_full, n_trials=prepared.n_trials[own], kappa=kappa,
                observed=post, dims="obs_id",
            )
        else:  # bernoulli_offfloor (not expected for block-2 taught; kept for parity)
            pm.Bernoulli(
                "y_offfloor", logit_p=eta_full,
                observed=(post > 0).astype(np.int64), dims="obs_id",
            )

    return BuiltModel(model=model, prepared=prepared, payload=EmptyPayload())
