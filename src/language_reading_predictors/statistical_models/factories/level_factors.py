# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DAG-focused level-factor model construction.

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from typing import TYPE_CHECKING, Iterable

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    LevelFactorsPayload,
)
from language_reading_predictors.statistical_models.definitions import (
    POST_PHASE_LABELS,
)
from language_reading_predictors.statistical_models.likelihood import (
    SCORE_MEAN_LINKS,
    ScoreMeanLink,
    beta_binomial_from_score_mean_link,
    invert_score_mean_link,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _add_child_random_intercept,
    _alpha_sigma_for,
    _rlm_dispersion_kappa,
    _tau_sigma_for,
)

def build_level_factors_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    ability_covariate: str | None = None,
    adjust_for: Iterable[str] = (),
    group_by_time: bool = True,
    ability_by_time: bool = True,
    group_ability: bool = True,
    likelihood: str = "beta_binomial",
    # #584 decision 2: the phoneme-blending response link. ``"logit"`` is the
    # ordinary Beta-Binomial inverse-logit mean; ``"three_choice_guessing_floor"``
    # maps it onto [1/3, 1] for the ten three-alternative forced-choice blending
    # items, whose expected score cannot fall below chance. B only, graded only,
    # and released only beside its paired ordinary-link fit.
    score_mean_link: ScoreMeanLink = "logit",
    # #584 decision 3: the post-t1 waves this fit carries, i.e. the coordinates of
    # ``d_grp_time``. The four-wave model of record passes the family default; the
    # randomised-window comparator passes ``("t2",)`` and its panel holds t1/t2 rows
    # only, so no post-crossover observation can inform the reported t2 change.
    post_phase_labels: tuple[str, ...] = POST_PHASE_LABELS,
    # #552: how the per-timepoint arm coefficients are parameterised. ``"t1"``
    # (default) centres them on the pre-randomisation t1 gap -- ``arm_gap_t1``
    # (a balance nuisance term on the cross-coupling prior, the DiD idiom) plus
    # ``d_grp_time[t]`` changes (outcome-tier tau prior) for t2..t4, with the
    # per-wave levels view ``b_grp_time`` kept as a Deterministic; ``"free"``
    # keeps one free tau-prior coefficient per timepoint (the pre-#552 comparator).
    arm_gap_reference: str = "t1",
    use_subject_random_intercept: bool = True,
    # #584 decision 4: a levels model has no own-baseline term, so this intercept
    # carries the entire between-child spread in level rather than the residual a
    # gain model leaves. The gain-model scale of 0.5 asserts a middle-95% child
    # range of 0.18 to 0.45 on a mid-difficulty measure -- narrower than the tests
    # resolve, and past its own 99th percentile for two of the eleven fits.
    sigma_child_prior_sigma: float = 1.0,
    # #584 decision 4: the dispersion prior. ``halfnormal_inverse_sqrt`` puts the
    # half-normal on ``1/sqrt(kappa)``, where the near-Binomial limit is zero and
    # therefore reachable; ``halfnormal_concentration`` is the pre-decision prior,
    # kept for the comparator and the sweep axis.
    kappa_prior_family: str = "halfnormal_inverse_sqrt",
    kappa_prior_sigma: float | None = None,
    # #389 finding 2: the zero-sum wave-deviation scale. Sized so the largest
    # observed wave deviation from the across-wave mean level (about +/-0.85
    # logits, phoneme segmenting) sits within ~1.3 marginal prior SD
    # (ZeroSumNormal marginal SD = sigma * sqrt(3)/2 for four waves), rather
    # than at ~1.9 SD as the former free Normal(0, 0.5) implied.
    alpha_time_prior_sigma: float = 0.75,
    # Sensitivity override for the focal randomised contrast's prior scale
    # (the #382 idiom): None keeps the outcome-tier default from
    # ``_tau_sigma_for``; the level-factor treatment-prior sweep passes explicit
    # grid values.
    tau_prior_sigma: float | None = None,
    # Sensitivity override for the ``arm_gap_t1`` balance-term prior scale
    # (2026-08-20 level-factors review, finding 1): the balance term is
    # prior-dominated in most reporting fits and trades off directly against
    # ``d_grp_time[t2]``, so its prior scale needs its own sweep axis. None
    # keeps the registered cross-coupling default (Normal(0, 0.3)). Only
    # meaningful under the t1-referenced parameterisation, which is the only
    # one with a balance term.
    arm_gap_prior_sigma: float | None = None,
) -> BuiltModel[LevelFactorsPayload]:
    """Level-factors model (LRPLF): what is associated with achievement levels.

    Repeated measures over the four timepoints (``phase_mode="levels"``): the
    outcome is the score *level* at each timepoint (no own baseline / not
    autoregressive). Linear predictor (logit scale):

        eta = alpha + alpha_time[t]
            + b_grp[t] * group            # group x time (trajectory divergence)
            + gamma_A * A_std_t            # age at t (precision)
            + g_ability[t] * z(ability)    # ability x time (observed GA handle)
            + gamma_grp_ability * group * z(ability)   # group x ability
            + u_child[i]                   # partial GA repair

    **Intercepts (#389 finding 2).** ``alpha`` is a Deterministic — a pooled,
    arm-blind empirical-Bayes anchor at the observed **pre-randomisation t1**
    logit (Haldane-smoothed, the DiD ``alpha_anchor`` idiom, #390/#481) plus a
    free zero-centred ``alpha_offset`` at the outcome-tier ``alpha`` scale —
    and ``alpha_time`` is an exact **zero-sum** wave-deviation vector. The
    former parameterisation (free global ``alpha`` centred at logit zero plus
    four free ``alpha_time`` elements) was doubly defective: only the sums
    ``alpha + alpha_time[t]`` were likelihood-identified (posterior
    correlations between ``alpha`` and ``alpha_time`` elements ran -0.62 to
    -0.96), and logit-zero centring implied prior-predictive scores near half
    the instrument maximum against observed means far below it (W: 39.1 items
    implied vs 11.8 observed). The zero-sum constraint removes the translation
    ridge exactly; the t1 anchor recentres the level while using only
    pre-randomisation data, deliberately under-centring the across-wave mean
    by the (smaller) growth increment rather than importing treatment-affected
    waves into the prior. The anchor is recorded in the fitted payload
    and the priors table labels ``alpha_offset`` as empirical Bayes.

    **Level-model caveat (baked into the parameterisation + report):** after t2
    the waitlist crosses over, so the group effect across the four timepoints is
    *not* an available-case modified ITT estimate. The focal ``group x time`` interaction is therefore
    modelled as a per-timepoint group effect ``b_grp[t]`` (dims ``phase`` = the
    timepoint index) — read as trajectory divergence — and the **clean randomised
    contrast lives only at t2**. ``ability x time`` is likewise a
    per-timepoint ability effect ``g_ability[t]``. Set ``group_by_time`` /
    ``ability_by_time`` False to collapse either to a single time-invariant
    coefficient. Only the randomised contrast is causal; all other terms are
    adjusted associations under the DAG.

    **Arm-gap reference (#552).** A levels model conditions on nothing at
    baseline while adjusting for age and ability at every wave, so a free
    per-timepoint vector carries the covariate-adjusted *chance* t1 arm gap
    straight into the t2 contrast (in these data the adjusted t1 gap is negative
    on every outcome). Under ``arm_gap_reference="t1"`` the vector is therefore
    written as a balance term plus changes::

        b_grp[t] = arm_gap_t1 + d_grp_time[t]      (t = t2, t3, t4)
        b_grp[t1] = arm_gap_t1

    with ``arm_gap_t1`` on the cross-coupling prior (a nuisance balance quantity,
    the DiD family's idiom — never an effect) and ``d_grp_time`` (dims
    ``post_phase`` = ``t2``/``t3``/``t4``) on the outcome-tier tau prior. The
    randomised contrast is the **t2 change** ``d_grp_time[t2]`` — a
    difference-in-differences of adjusted levels; ``d_grp_time[t3]`` / ``[t4]``
    are randomised early-start-versus-delayed-start schedule contrasts, not
    treated-versus-untreated effects and carrying no mechanistic reading (#631
    finding 13). ``b_grp_time`` is kept as a Deterministic so
    the per-wave levels view (and the pre-#552 raw t2 gap, for the side-by-side
    comparison) is still in the trace. ``arm_gap_reference="free"`` keeps the
    former free ``b_grp_time`` vector as an explicit comparator.

    ``adjust_for`` (default ()): revised-DAG confounders that are not bounded-count
    measures — hearing status (``hs`` / ``hs_missing``), speech production
    (``deapp_c`` / ``deapp_c_missing``) and phonological memory (``erbto`` /
    ``erbto_missing``) (#247). Each enters as a linear ``gamma_{c}`` term with the
    cross-coupling prior, reusing ``build_mechanism_model``'s idiom. These are
    exogenous, **non**-treatment-affected roots/upstream nodes (``IG`` has no edge to
    ``HS``/``SP``/``RW``), so they do not sit on the causal path from group and their
    adjustment does not block the randomised t2 contrast. Note the level model takes
    **no** measure-skill adjusters (unlike the gain factory's ``skill_symbols``): a
    levels model conditioning on another evolving skill's *contemporaneous* level
    would condition on a post-treatment mediator/collider and bias the group×time
    trajectory it exists to estimate.
    """
    if prepared.phase_mode != "levels":
        raise ValueError("build_level_factors_model requires phase_mode='levels'")
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    if arm_gap_reference not in ("t1", "free"):
        raise ValueError(
            f"arm_gap_reference must be 't1' or 'free', got {arm_gap_reference!r}"
        )
    if arm_gap_reference == "t1" and not group_by_time:
        raise ValueError(
            "arm_gap_reference='t1' requires group_by_time=True (a pooled group "
            "coefficient has no t1 gap to centre on)"
        )
    if arm_gap_prior_sigma is not None and not (
        group_by_time and arm_gap_reference == "t1"
    ):
        # Only the t1-referenced parameterisation has an arm_gap_t1 term; silently
        # ignoring the override on a free/pooled build would report a sensitivity
        # cell that never varied anything.
        raise ValueError(
            "arm_gap_prior_sigma requires group_by_time=True and "
            "arm_gap_reference='t1' (no balance term to re-prior otherwise)"
        )
    own = outcome_symbol
    if own not in prepared.post_counts:
        raise KeyError(f"Outcome {own!r} missing from prepared data (post_counts)")
    if ability_covariate is not None and ability_covariate not in prepared.covariates:
        raise KeyError(f"ability_covariate {ability_covariate!r} not in prepared.covariates")
    if group_ability and ability_covariate is None:
        raise ValueError("group_ability interaction requires an ability_covariate")
    # Mirrors build_itt_model's contract: the guessing floor is a property of the
    # blending instrument, not a switch, and it is only meaningful for a graded
    # score mean (the off-floor branch models a binary indicator, which has no
    # chance floor to respect).
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
    adjust_for = tuple(adjust_for)
    for c in adjust_for:
        if c not in prepared.covariates:
            raise KeyError(f"Adjuster covariate {c!r} not loaded in prepared data")

    keep = ~np.isnan(prepared.post_counts[own])
    for c in adjust_for:
        keep = keep & ~np.isnan(prepared.covariates[c])
    prepared = _subset(prepared, keep)

    post = prepared.post_counts[own].astype(np.int64)
    G_f = prepared.G.astype(float)
    ability = prepared.covariates[ability_covariate] if ability_covariate is not None else None

    # Pooled, arm-blind empirical-Bayes intercept anchor from the observed
    # PRE-randomisation t1 rows only (#389 finding 2; the DiD alpha_anchor idiom).
    # Haldane smoothing keeps it finite at all-zero / all-max t1 scores.
    t1 = post[prepared.phase == 0]
    if not t1.size:
        raise ValueError(f"Cannot anchor {own}: no observed t1 outcome values")
    if likelihood == "bernoulli_offfloor":
        movers = int(np.sum(t1 > 0))
        alpha_anchor = float(np.log((movers + 0.5) / (t1.size - movers + 0.5)))
    else:
        successes = float(np.sum(t1))
        failures = float(t1.size * prepared.n_trials[own] - successes)
        # The anchor locates the intercept prior on the LINEAR PREDICTOR, so under a
        # non-identity score-mean link the observed proportion must be mapped back
        # through the link first (#584 decision 2). Anchoring a guessing-floor fit on
        # the raw observed logit would defeat the point of the empirical-Bayes anchor
        # (#389 finding 2): for blending, logit(0.49) = -0.03 against the -1.16 the
        # floor link needs. ``invert_score_mean_link`` raises if the pooled t1 score
        # sits at or below chance, where no linear predictor could produce it.
        proportion = (successes + 0.5) / (successes + failures + 1.0)
        unit = float(invert_score_mean_link(proportion, score_mean_link))
        alpha_anchor = float(np.log(unit / (1.0 - unit)))

    t1_referenced = group_by_time and arm_gap_reference == "t1"
    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }
    if t1_referenced:
        if prepared.n_phases != len(post_phase_labels) + 1:
            raise ValueError(
                "arm_gap_reference='t1' expects a levels panel of "
                f"t1 + {tuple(post_phase_labels)}, got {prepared.n_phases} phases"
            )
        coords["post_phase"] = list(post_phase_labels)
    with pm.Model(coords=coords) as model:
        phase_d = pm.Data("phase_idx", prepared.phase.astype(np.int64), dims="obs_id")
        child_idx_d = pm.Data("child_idx", prepared.child_idx.astype(np.int64), dims="obs_id")
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", G_f, dims="obs_id")
        ability_d = (
            pm.Data(f"{ability_covariate}_std", ability, dims="obs_id")
            if ability is not None
            else None
        )
        adjust_d = {
            c: pm.Data(f"{c}_adj", prepared.covariates[c], dims="obs_id")
            for c in adjust_for
        }

        # Identified, recentred intercepts (#389 finding 2; see the docstring).
        # The former free alpha + free four-element alpha_time pair carried a
        # translation ridge (only the sums identified) and a logit-zero centre
        # far from this population; alpha is now anchored at the pooled
        # pre-randomisation t1 level and alpha_time is an exact zero-sum
        # wave-deviation vector, so both are identified and the anchor uses no
        # treatment-affected data. The #273 "small global offset" reading this
        # replaces is recorded in the git history of that decision.
        alpha_offset = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha_offset")
        alpha = pm.Deterministic("alpha", alpha_anchor + alpha_offset)
        alpha_time = pm.ZeroSumNormal(
            "alpha_time", sigma=alpha_time_prior_sigma, dims="phase"
        )
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
        eta = alpha + alpha_time[phase_d] + gamma_A * A_std_d

        # group x time (or single group main effect).
        _tau_sigma = _tau_sigma_for(outcome_symbol, tau_prior_sigma)
        if t1_referenced:
            # #552: balance term + changes. ``arm_gap_t1`` is the adjusted
            # pre-randomisation arm gap (the DiD ``arm_gap_t1`` idiom: the
            # cross-coupling prior, a nuisance balance quantity); ``d_grp_time``
            # carries the change from t1 at each later wave on the outcome-tier
            # tau prior, so the prior sits on the randomised t2 *difference*
            # directly rather than on two raw gaps whose difference it would be.
            arm_gap_t1 = (
                _priors.gamma_cross_prior()
                if arm_gap_prior_sigma is None
                else _priors.gamma_cross_prior(sigma=arm_gap_prior_sigma)
            ).to_pymc("arm_gap_t1")
            d_grp = _priors.tau_prior(sigma=_tau_sigma).to_pymc(
                "d_grp_time", dims="post_phase"
            )
            b_grp = pm.Deterministic(
                "b_grp_time",
                pt.concatenate([pt.stack([arm_gap_t1]), arm_gap_t1 + d_grp]),
                dims="phase",
            )
            eta = eta + b_grp[phase_d] * G_d
        elif group_by_time:
            b_grp = _priors.tau_prior(sigma=_tau_sigma).to_pymc(
                "b_grp_time", dims="phase"
            )
            eta = eta + b_grp[phase_d] * G_d
        else:
            beta_grp = _priors.tau_prior(sigma=_tau_sigma).to_pymc("beta_grp")
            eta = eta + beta_grp * G_d

        # ability main / ability x time.
        if ability_d is not None:
            if ability_by_time:
                g_ab = _priors.gamma_cross_prior().to_pymc("gamma_ability_time", dims="phase")
                eta = eta + g_ab[phase_d] * ability_d
            else:
                gamma_ability = _priors.gamma_cross_prior().to_pymc("gamma_ability")
                eta = eta + gamma_ability * ability_d

        # group x ability cross term.
        if group_ability:
            ga_prod = pm.Data("int_group_ability", G_f * np.asarray(ability, dtype=float), dims="obs_id")
            gamma_grp_ability = _priors.gamma_cross_prior().to_pymc("gamma_grp_ability")
            eta = eta + gamma_grp_ability * ga_prod

        # Raw-covariate adjusters (revised-DAG exogenous confounders HS/SP/RW): linear
        # gamma terms, mirroring build_mechanism_model's adjust_for path (#247).
        for c in adjust_for:
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}")
            eta = eta + gamma_c * adjust_d[c]

        if use_subject_random_intercept:
            eta = _add_child_random_intercept(
                eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
            )

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        if likelihood == "beta_binomial":
            if kappa_prior_family == "halfnormal_inverse_sqrt":
                # Same constructor the ITT high-denominator fits and the RLM
                # historical families use, so the three cannot drift (#584
                # decision 4). ``kappa`` survives as the Deterministic the reports
                # speak in; ``inv_sqrt_kappa`` is what is sampled.
                kappa = _rlm_dispersion_kappa(
                    float(_priors.inv_sqrt_kappa_prior().sigma)
                    if kappa_prior_sigma is None
                    else kappa_prior_sigma
                )
            elif kappa_prior_family == "halfnormal_concentration":
                kappa = (
                    _priors.kappa_prior()
                    if kappa_prior_sigma is None
                    else _priors.kappa_prior(sigma=kappa_prior_sigma)
                ).to_pymc("kappa")
            else:
                raise ValueError(
                    "kappa_prior_family must be 'halfnormal_concentration' or "
                    f"'halfnormal_inverse_sqrt', got {kappa_prior_family!r}"
                )
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
        payload=LevelFactorsPayload(
            alpha_anchor=alpha_anchor,
            arm_gap_reference=arm_gap_reference,
            score_mean_link=score_mean_link
        ),
    )
