# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DAG-focused gain-factor (ANCOVA) model construction.

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from dataclasses import replace
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pymc as pm

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    GainFactorsPayload,
)
from language_reading_predictors.statistical_models.likelihood import (
    SCORE_MEAN_LINKS,
    ScoreMeanLink,
    beta_binomial_from_score_mean_link,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
    filter_informative_covariates,
    standardise,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _add_child_random_intercept,
    _alpha_sigma_for,
    _interaction_product,
    _rlm_dispersion_kappa,
    _scalar_prior,
    _tau_sigma_for,
)

def build_gain_factors_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    skill_symbols: Iterable[str] = (),
    ability_covariate: str | None = None,
    adjust_for: Iterable[str] = (),
    interactions: Iterable[tuple[str, str]] = (),
    treated_only: bool = False,
    likelihood: str = "beta_binomial",
    # #596: the phoneme-blending response link. ``"logit"`` is the ordinary
    # Beta-Binomial inverse-logit mean; ``"three_choice_guessing_floor"`` maps it
    # onto [1/3, 1] for the ten three-alternative forced-choice blending items,
    # whose expected score cannot fall below chance. B only, graded only, and
    # released only beside its paired ordinary-link fit.
    score_mean_link: ScoreMeanLink = "logit",
    use_subject_random_intercept: bool = True,
    sigma_child_prior_sigma: float = 0.5,
    trt_prior_sigma: float | None = None,
    # #575 finding 10a: the dispersion-prior family for the graded likelihood,
    # mirroring build_itt_model. "halfnormal_concentration" is the registered
    # HalfNormal(50) on kappa; "halfnormal_inverse_sqrt" puts the half-normal on
    # 1/sqrt(kappa) so the near-Binomial limit is reachable. kappa_sigma
    # overrides the chosen family's scale for the sensitivity sweep.
    kappa_prior_family: str = "halfnormal_concentration",
    kappa_sigma: float | None = None,
    # #575 finding 10b: the graded own-baseline slope's prior scale, exposed for
    # the required 0.25-vs-0.5 sensitivity. Ignored on the off-floor path, which
    # carries its own indicator prior.
    gamma_own_prior_sigma: float = 0.25,
) -> BuiltModel[GainFactorsPayload]:
    """Gain-factors model (LRPGF): what is associated with how much children gain.

    Repeated measures over the three period transitions (``phase_mode="all"``):
    the outcome is the period post-count given its pre-count (an ANCOVA "gain").
    Linear predictor (logit scale):

        eta = alpha + alpha_phase[p]
            + beta_trt * OnIntervention           # causal (ITT) — period-1 contrast ~ tau
            + gamma_own * logit(own_pre)          # own baseline (precision)
            + gamma_A * A_std                      # age (precision)
            + gamma_ability * z(ability)           # observed GA handle (blocks)
            + sum_s gamma_s * logit(skill_pre_s)   # upstream DAG skills (adjusted assoc.)
            + sum interactions                     # focal, pre-specified
            + u_child[i]                           # partial GA repair

    ``OnIntervention`` is derived from the data: the immediate arm (``G == 1``) is
    on from period 1; the waitlist (``G == 0``) is off in period 1 only and on once
    it crosses over (``phase >= 1``). Its coefficient is identified almost entirely
    by the period-1 (randomised) contrast, so ``beta_trt`` reproduces the ITT
    ``tau`` (a verification anchor).

    ``treated_only=True`` excludes the waitlist arm's untreated period 1 ("gains
    while on intervention"). Every remaining row is then on-intervention, so the
    treatment term and any treatment interaction are constant — they are dropped
    automatically (the model becomes the factor-association model among the
    treated).

    ``skill_symbols`` are the outcome's measured, repeated-available DAG-upstream
    skills (e.g. L, R for word reading), entered as their period baseline logit.
    ``ability_covariate`` is a ``prepared.covariates`` key (``blocks``).
    ``interactions`` is a set of ``(term_a, term_b)`` pairs over the controlled
    vocabulary ``{"trt", "age", "ability", "own", <skill symbols>}``; each adds a
    ``gamma_int_<a>_<b>`` coefficient on the product of the two standardised terms.
    All non-causal coefficients are adjusted associations under the DAG.

    Under ``likelihood="bernoulli_offfloor"`` the ``own`` term — main effect and any
    interaction naming it — is the **binary off-floor-at-pre indicator** (raw 0/1),
    not the graded pre logit (#391 finding 2 decision, 2026-07-22); see the
    own-baseline block below.

    ``adjust_for`` (default ()): revised-DAG confounders that are not bounded-count
    measures and so cannot enter via ``skill_symbols`` — hearing status (``hs`` /
    ``hs_missing``), speech production (``deapp_c`` / ``deapp_c_missing``) and
    phonological memory (``erbto`` / ``erbto_missing``) (#247). Each must be a key in
    ``prepared.covariates`` (the pipeline requests them via ``covariates=`` and
    standardises the continuous ones / adds missing-indicators). They enter as linear
    ``gamma_{c}`` terms with the regularising cross-coupling prior, exactly as in
    ``build_mechanism_model`` (#245/#258) — reused, not duplicated. These are
    exogenous, non-treatment-affected confounders (``IG`` has no edge to ``HS``,
    ``SP`` or ``RW``), so conditioning on them does not block the randomised
    ``beta_trt`` contrast; like every non-causal term they are adjusted associations.
    """
    if prepared.phase_mode != "all":
        raise ValueError("build_gain_factors_model requires phase_mode='all'")
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    if score_mean_link not in SCORE_MEAN_LINKS:
        raise ValueError(
            f"score_mean_link must be one of {SCORE_MEAN_LINKS}, "
            f"got {score_mean_link!r}"
        )
    if score_mean_link == "three_choice_guessing_floor" and outcome_symbol != "B":
        raise ValueError(
            "three_choice_guessing_floor is only valid for phoneme blending (B), "
            f"got {outcome_symbol!r}"
        )
    if likelihood != "beta_binomial" and score_mean_link != "logit":
        raise ValueError(
            "score_mean_link applies to the graded Beta-Binomial mean; the "
            f"{likelihood!r} branch has no score mean to map"
        )
    # Treatment-prior sweep hook (#391): the gf sensitivity runner refits the
    # registered primary with only the beta_trt prior scale moved across its
    # grid. A treated-only fit has no beta_trt, so scaling its prior is a
    # caller error, not a silent no-op.
    if trt_prior_sigma is not None and treated_only:
        raise ValueError(
            "trt_prior_sigma is meaningless for a treated-only fit: the "
            "treatment indicator is constant and beta_trt is not built"
        )
    if kappa_prior_family not in ("halfnormal_concentration", "halfnormal_inverse_sqrt"):
        raise ValueError(
            "kappa_prior_family must be 'halfnormal_concentration' or "
            f"'halfnormal_inverse_sqrt', got {kappa_prior_family!r}"
        )
    if not gamma_own_prior_sigma > 0:
        raise ValueError(
            f"gamma_own_prior_sigma must be positive, got {gamma_own_prior_sigma!r}"
        )
    own = outcome_symbol
    if own not in prepared.post_counts or own not in prepared.pre_logit:
        raise KeyError(f"Outcome {own!r} needs pre+post scores in prepared data")
    skill_symbols = tuple(skill_symbols)
    for s in skill_symbols:
        if s not in prepared.pre_logit:
            raise KeyError(f"Skill {s!r} has no baseline in prepared data")
    if ability_covariate is not None and ability_covariate not in prepared.covariates:
        raise KeyError(f"ability_covariate {ability_covariate!r} not in prepared.covariates")
    adjust_for = tuple(adjust_for)
    for c in adjust_for:
        if c not in prepared.covariates:
            raise KeyError(f"Adjuster covariate {c!r} not loaded in prepared data")

    valid_terms = {"trt", "age", "own", *skill_symbols}
    if ability_covariate is not None:
        valid_terms.add("ability")
    interactions = tuple(tuple(p) for p in interactions)
    for pair in interactions:
        for k in pair:
            if k not in valid_terms:
                raise KeyError(f"interaction term {k!r} not available; have {sorted(valid_terms)}")

    # Drop rows missing the outcome post, the own baseline, any skill baseline, or
    # any raw-covariate adjuster.
    keep = ~np.isnan(prepared.post_counts[own]) & ~np.isnan(prepared.pre_logit[own])
    if likelihood == "bernoulli_offfloor":
        # The off-floor path replaces the graded own baseline with the binary
        # off-floor-at-pre indicator, which needs the raw pre count.
        keep = keep & ~np.isnan(prepared.pre_counts[own])
    for s in skill_symbols:
        keep = keep & ~np.isnan(prepared.pre_logit[s])
    for c in adjust_for:
        keep = keep & ~np.isnan(prepared.covariates[c])
    on_intervention = (prepared.G == 1) | (prepared.phase >= 1)
    if treated_only:
        keep = keep & on_intervention
    prepared = _subset(prepared, keep)

    # Re-filter the adjusters on the FINAL fitted rows (#575 finding 1). The
    # loader removes constants on its complete frame, but the focal-outcome and
    # treated-only masks above can make a previously varying indicator constant —
    # an exact intercept alias that is not data-identified, perturbs the prior
    # geometry and would be falsely recorded as an informative adjuster (the
    # audit's concrete case: erbto_missing in gf-005/105/205). Dropped names are
    # recorded on the payload so the effective adjustment set in config.json
    # describes the model the posterior actually contains.
    _, _effective_adjust, _post_mask_dropped = filter_informative_covariates(
        prepared, adjust_for
    )
    absent = sorted(set(_post_mask_dropped) - set(prepared.covariates))
    if absent:
        # A name that is not even loaded is a caller error, not a masked-away
        # constant; keep that loud rather than folding it into the record.
        raise KeyError(
            f"Adjuster covariate(s) {', '.join(absent)} not loaded in prepared data"
        )
    adjust_for = _effective_adjust
    if _post_mask_dropped:
        # Reflect the drop on the returned frame too, so ``dropped_constant`` in
        # the effective-adjustment record (which reads
        # ``built.prepared.dropped_covariates``) names it without a second
        # bookkeeping path. Non-adjuster covariates (ability) are untouched.
        prepared = replace(
            prepared,
            covariates={
                k: v
                for k, v in prepared.covariates.items()
                if k not in _post_mask_dropped
            },
            covariate_scalers={
                k: v
                for k, v in prepared.covariate_scalers.items()
                if k not in _post_mask_dropped
            },
            covariate_time={
                k: v
                for k, v in prepared.covariate_time.items()
                if k not in _post_mask_dropped
            },
            dropped_covariates=tuple(
                dict.fromkeys((*prepared.dropped_covariates, *_post_mask_dropped))
            ),
        )

    post = prepared.post_counts[own].astype(np.int64)
    trt = ((prepared.G == 1) | (prepared.phase >= 1)).astype(float)
    # In treated_only the treatment indicator is constant -> not identified; drop
    # it and any interaction involving it.
    include_trt = not treated_only

    # #575 finding 5: the randomised headline needs period-1 support in BOTH
    # arms after the final analysis mask — a one-arm period 1 would leave
    # beta_trt identified only by post-crossover model structure while the
    # report still called it randomised. Realised per-period, per-arm support is
    # recorded on the payload for analysis_support.csv either way.
    _p1 = prepared.phase == 0
    if include_trt:
        n_p1_immediate = int(np.sum(_p1 & (prepared.G == 1)))
        n_p1_waitlist = int(np.sum(_p1 & (prepared.G == 0)))
        if n_p1_immediate == 0 or n_p1_waitlist == 0:
            raise ValueError(
                "build_gain_factors_model requires both randomised arms in "
                "period 1 after the final analysis mask; got "
                f"{n_p1_immediate} immediate and {n_p1_waitlist} wait-list "
                "period-1 rows"
            )
    _support: list[tuple[int, str, int, int]] = []
    for _phase_value in sorted({int(p) for p in prepared.phase}):
        for _g_value, _arm in ((1, "immediate"), (0, "waitlist")):
            _cell = (prepared.phase == _phase_value) & (prepared.G == _g_value)
            _n_rows = int(np.sum(_cell))
            if _n_rows == 0:
                continue
            _n_children = int(np.unique(prepared.child_idx[_cell]).size)
            _support.append((_phase_value, _arm, _n_rows, _n_children))
    active_interactions = [
        pair for pair in interactions if include_trt or "trt" not in pair
    ]

    # Standardise the interaction-term components on the *kept* rows (used for the
    # interaction products and AME moderators). Main effects are entered on their
    # natural scales (raw logit baselines; age uses ``prepared.A_std``).
    # Re-standardise the ability covariate here too: treated_only (…b) variants drop
    # the untreated period-1 rows, so the load-time scaler (over all periods) would
    # otherwise mislabel the “per 1 SD” unit for the treated-only fit.
    # Age is intentionally NOT re-standardised: it is deliberately kept on the shared
    # load-time (all-period) scale so ``gamma_A`` and the age-moderation unit stay
    # directly comparable between each treated-only (…b) variant and its full
    # sibling — the small age-distribution shift from dropping period-1 rows is
    # accepted in exchange for that cross-variant comparability (issue #273).
    term_vecs: dict[str, np.ndarray] = {"trt": trt, "age": prepared.A_std}
    if ability_covariate is not None:
        term_vecs["ability"], _ = standardise(prepared.covariates[ability_covariate])
    # "own" on the off-floor path is the binary off-floor-at-pre indicator (raw
    # 0/1), NOT the standardised pre logit: the pre logit of a heavily-floored
    # measure is a near-degenerate spike, so the indicator is the honest
    # functional form for both the main effect and any declared interaction on
    # ``own`` (#391 finding 2 decision, 2026-07-22).
    if likelihood == "bernoulli_offfloor":
        term_vecs["own"] = (prepared.pre_counts[own] > 0).astype(float)
    else:
        term_vecs["own"], _ = standardise(prepared.pre_logit[own])
    for s in skill_symbols:
        term_vecs[s], _ = standardise(prepared.pre_logit[s])

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }
    with pm.Model(coords=coords) as model:
        phase_d = pm.Data("phase_idx", prepared.phase.astype(np.int64), dims="obs_id")
        child_idx_d = pm.Data("child_idx", prepared.child_idx.astype(np.int64), dims="obs_id")
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        # Own baseline is a precision term for the graded likelihood only; the off-floor
        # (Bernoulli) path drops it (A4 — see below), so its data node is not built.
        own_pre_d = (
            pm.Data("own_pre_logit", prepared.pre_logit[own], dims="obs_id")
            if likelihood != "bernoulli_offfloor"
            else None
        )
        trt_d = pm.Data("on_intervention", trt, dims="obs_id") if include_trt else None
        ability_d = (
            pm.Data(f"{ability_covariate}_std", term_vecs["ability"], dims="obs_id")
            if ability_covariate is not None
            else None
        )
        skill_d = {
            s: pm.Data(f"{s}_pre_logit", prepared.pre_logit[s], dims="obs_id")
            for s in skill_symbols
        }
        adjust_d = {
            c: pm.Data(f"{c}_adj", prepared.covariates[c], dims="obs_id")
            for c in adjust_for
        }
        int_d = {
            pair: pm.Data(f"int_{pair[0]}_{pair[1]}", _interaction_product(term_vecs, *pair), dims="obs_id")
            for pair in active_interactions
        }

        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
        alpha_phase = pm.Normal("alpha_phase", mu=0.0, sigma=0.5, dims="phase")
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")

        eta = alpha + alpha_phase[phase_d] + gamma_A * A_std_d
        # Own-baseline term. On the graded path this is the usual precision slope on
        # the raw pre logit. On the off-floor (Bernoulli) path the graded ``gamma_own``
        # is dropped (A4, 2026-07-13): its Normal(1, 0.25) "post tracks pre 1:1" prior
        # is calibrated to graded test-retest reliability and does not transfer to a
        # binary indicator, and the standardised pre logit of a heavily-floored
        # measure is a near-degenerate spike. In its place the model ALWAYS carries
        # the **binary off-floor-at-pre indicator** as the baseline main effect
        # (#391 finding 2 decision, 2026-07-22): the period-1 control data flatly
        # contradict a flat-in-baseline control arm (2/17 at-floor vs 7/8 off-floor
        # moved off the floor at post), so omitting it either misfits (no own term)
        # or lets a declared interaction on ``own`` absorb the main effect
        # (hierarchy violation — the pre-decision defect). ``term_vecs["own"]`` is
        # this same indicator on the off-floor path, so any interaction on ``own``
        # shares its functional form and hierarchy holds by construction.
        if own_pre_d is not None:
            gamma_own = _priors.gamma_own_prior(
                sigma=gamma_own_prior_sigma
            ).to_pymc("gamma_own")
            eta = eta + gamma_own * own_pre_d
        else:
            own_ff_d = pm.Data("own_pre_offfloor", term_vecs["own"], dims="obs_id")
            gamma_own_ff = _priors.gamma_own_offfloor_prior().to_pymc("gamma_own_offfloor")
            eta = eta + gamma_own_ff * own_ff_d

        if include_trt:
            beta_trt = _priors.tau_prior(
                sigma=(
                    _tau_sigma_for(outcome_symbol)
                    if trt_prior_sigma is None
                    else float(trt_prior_sigma)
                )
            ).to_pymc("beta_trt")
            eta = eta + beta_trt * trt_d
        if ability_d is not None:
            gamma_ability = _priors.gamma_cross_prior().to_pymc("gamma_ability")
            eta = eta + gamma_ability * ability_d
        for s in skill_symbols:
            gamma_s = _priors.gamma_cross_prior().to_pymc(f"gamma_{s}")
            eta = eta + gamma_s * skill_d[s]
        # Raw-covariate adjusters (revised-DAG confounders that are not bounded-count
        # measures): hearing (hs/hs_missing), speech (deapp_c), phonological memory
        # (erbto). Linear gamma terms, mirroring build_mechanism_model's adjust_for
        # path (#245/#258, #247).
        for c in adjust_for:
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}")
            eta = eta + gamma_c * adjust_d[c]
        for pair in active_interactions:
            gi = _priors.gamma_cross_prior().to_pymc(f"gamma_int_{pair[0]}_{pair[1]}")
            eta = eta + gi * int_d[pair]

        if use_subject_random_intercept:
            eta = _add_child_random_intercept(
                eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
            )

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        if likelihood == "beta_binomial":
            if kappa_prior_family == "halfnormal_inverse_sqrt":
                # Dispersion-scale parameterisation so the near-Binomial limit
                # is reachable (#575 finding 10a; same constructor as
                # build_itt_model and the RLM historical families).
                kappa = _rlm_dispersion_kappa(
                    float(_priors.inv_sqrt_kappa_prior().sigma)
                    if kappa_sigma is None
                    else kappa_sigma
                )
            elif kappa_sigma is not None:
                kappa = _priors.kappa_prior(sigma=kappa_sigma).to_pymc("kappa")
            else:
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

    # Expose the treatment×covariate interaction moderators so the pipeline's
    # average-marginal-effect report can net out the *full* per-row treatment
    # contribution ``beta_trt + Σ_k gamma_int_trt_k · z_k`` — not just
    # ``beta_trt`` (issue: gain-family AME ignored the fitted trt interactions).
    # Each entry is ``(gamma_int coefficient name, standardised moderator vector)``
    # for a fitted interaction with ``trt`` as one member; the moderator is the
    # *other* member's term vector, exactly as multiplied into ``eta``. Only
    # populated when the treatment term is present (``include_trt``).
    trt_moderators: list[tuple[str, np.ndarray]] = []
    if include_trt:
        for pair in active_interactions:
            if "trt" not in pair:
                continue
            other = pair[0] if pair[1] == "trt" else pair[1]
            trt_moderators.append(
                (f"gamma_int_{pair[0]}_{pair[1]}", np.asarray(term_vecs[other], dtype=float))
            )

    return BuiltModel(
        model=model,
        prepared=prepared,
        payload=GainFactorsPayload(
            trt_interaction_moderators=tuple(trt_moderators),
            score_mean_link=score_mean_link,
            effective_adjust_for=tuple(adjust_for),
            post_mask_dropped_adjusters=tuple(_post_mask_dropped),
            period_arm_support=tuple(_support),
        ),
    )
