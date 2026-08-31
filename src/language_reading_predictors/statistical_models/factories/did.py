# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Waitlist-crossover difference-in-differences model construction.

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    DidArmWavePayload,
    DidDosePayload,
)
from language_reading_predictors.statistical_models.likelihood import (
    SCORE_MEAN_LINKS,
    ScoreMeanLink,
    beta_binomial_from_logit,
    beta_binomial_from_score_mean_link,
    invert_score_mean_link,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
    standardise,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _add_child_random_intercept,
    _alpha_sigma_for,
    _broadcast_phase_zero,
    _rlm_dispersion_kappa,
    _scalar_prior,
    _standardise_child_baseline,
    _tau_sigma_for,
)

def build_did_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    waves: Iterable[int] = (0, 1, 2),
    periods: Iterable[int] = (0, 1),
    use_child_re: bool = True,
    use_age: bool = True,
    dose: bool = False,
    period_varying_dose: bool = False,
    use_varying_delta: bool = False,
    # #390 P1 condition 1: False replaces the arm-by-wave empirical-Bayes
    # pooled-t1 anchor with a genuinely independent zero-centred alpha at the
    # outcome-tier scale (LRPDID101). Dose models already build a free alpha.
    use_intercept_anchor: bool = True,
    likelihood: str = "beta_binomial",
    # #382 recommendation 3: optional one-off wider prior on the single causal
    # term (LRPDID102). None keeps the outcome-tier default; arm_gap_t3 keeps
    # the tier scale either way, so the companion isolates tau_t2's prior.
    tau_t2_prior_sigma: float | None = None,
    # #390: the dose models' focal-slope analogue for the treatment-prior sweep
    # (mu_dose when period-varying, beta_dose otherwise). None keeps the
    # Normal(0, 1) default; the per-period deviation scale is not swept.
    dose_slope_prior_sigma: float | None = None,
    # #576 finding 2: the phoneme-blending response link on the graded
    # arm-by-wave score mean. ``"three_choice_guessing_floor"`` maps the mean
    # onto [1/3, 1]; the empirical-Bayes intercept anchor is inverted through
    # the same link so it still locates the *linear predictor*.
    score_mean_link: ScoreMeanLink = "logit",
    # #576 finding 4: the two widths that set how a realised t1 imbalance is
    # allocated between the regularised baseline gap and the shared child
    # intercepts. None keeps the shared defaults.
    arm_gap_t1_prior_sigma: float | None = None,
    sigma_child_prior_sigma: float | None = None,
    # #576 material qualification 2: the dispersion prior's shape. The default
    # concentration prior cannot reach the near-Binomial limit at a high
    # denominator; ``"halfnormal_inverse_sqrt"`` can.
    kappa_prior_family: str = "halfnormal_concentration",
    kappa_prior_sigma: float | None = None,
) -> BuiltModel[DidDosePayload | DidArmWavePayload]:
    """Waitlist-crossover triangulation with explicit arm-by-wave contrasts.

    Binary models use the t1--t3 levels frame.  They estimate the arm gap at
    every wave separately: ``tau_t2`` is the clean randomised t2 contrast,
    ``arm_gap_t3`` is also identified by the original randomisation, but of a
    different exposure -- assignment to the early-start versus delayed-start
    treatment schedule (both arms treated by t3) -- and ``delta_crossover =
    tau_t2 - arm_gap_t3`` is the change between those two randomised regime
    contrasts, never an identified catch-up.  This avoids
    the legacy restriction that forced those distinct quantities to equal one
    common-current-treatment coefficient.  The t2 period-start outcome is never
    conditioned on because it is treatment-affected in the immediate arm.
    ``tau_t2`` is the t2 arm-gap *level*, not the differenced ``tau_t2 -
    arm_gap_t1``: with free per-wave gaps, the shared child random intercept and
    the tight ``arm_gap_t1`` prior give a partial, prior-weighted baseline
    adjustment rather than exact differencing (the level-factor family's
    t1-referenced ``d_grp_time[t2]``, #552, is the gap-change estimand).

    Dose variants retain P1/P2 transition rows because sessions are interval
    exposures.  Because ``treated = (immediate arm) OR (period 2)`` saturates the
    arm-by-period cell design, their ``theta_treated`` at the mean treated dose is
    the crossover *cell* contrast, not an isolated treatment-presence effect.
    They adjust for randomised arm plus the shared t1 outcome and age, and enter
    sessions centred and scaled among treated rows.  Dose coefficients remain
    observational.
    """
    own = outcome_symbol
    if own not in prepared.post_counts:
        raise KeyError(f"Outcome {own!r} missing from prepared data")
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    if likelihood == "bernoulli_offfloor" and dose:
        raise ValueError(
            "bernoulli_offfloor is the binary prevalence estimand; use dose=False"
        )
    if period_varying_dose and not dose:
        raise ValueError("period_varying_dose=True requires dose=True")
    if dose_slope_prior_sigma is not None and not dose:
        raise ValueError(
            "dose_slope_prior_sigma applies to the dose models' focal slope; an "
            "arm-by-wave model has no dose term (sweep tau_t2_prior_sigma instead)"
        )
    if not use_intercept_anchor and dose:
        raise ValueError(
            "use_intercept_anchor=False is the arm-by-wave independent-prior "
            "sensitivity; the dose models already build a free intercept"
        )
    if use_varying_delta and dose:
        raise ValueError("use_varying_delta is unavailable for dose models")
    if use_varying_delta and not use_child_re:
        raise ValueError("use_varying_delta=True requires use_child_re=True")
    # bool is an int subclass, so an unguarded numeric check turned a typo'd flag
    # into a silent prior change (#576 lower-severity 5). Belt-and-braces for direct
    # callers; ``DiDModelSettings`` rejects the same values before any I/O.
    for _name, _width in (
        ("tau_t2_prior_sigma", tau_t2_prior_sigma),
        ("dose_slope_prior_sigma", dose_slope_prior_sigma),
        ("arm_gap_t1_prior_sigma", arm_gap_t1_prior_sigma),
        ("sigma_child_prior_sigma", sigma_child_prior_sigma),
        ("kappa_prior_sigma", kappa_prior_sigma),
    ):
        if _width is None:
            continue
        if isinstance(_width, bool) or not isinstance(_width, (int, float)):
            raise TypeError(
                f"{_name} must be a number when set, got {_width!r}; bool is not a "
                "prior width"
            )
        if not np.isfinite(float(_width)) or float(_width) <= 0.0:
            raise ValueError(f"{_name} must be finite and positive when set")
    if score_mean_link not in SCORE_MEAN_LINKS:
        raise ValueError(
            f"score_mean_link must be one of {list(SCORE_MEAN_LINKS)}, "
            f"got {score_mean_link!r}"
        )
    if score_mean_link != "logit":
        if outcome_symbol != "B":
            raise ValueError(
                "three_choice_guessing_floor is only valid for phoneme blending (B), "
                f"got {outcome_symbol!r}"
            )
        if dose or likelihood != "beta_binomial":
            raise ValueError(
                "score_mean_link applies to the graded arm-by-wave Beta-Binomial "
                "score mean; this branch has no score mean to map"
            )
    if kappa_prior_family not in ("halfnormal_concentration", "halfnormal_inverse_sqrt"):
        raise ValueError(
            "kappa_prior_family must be 'halfnormal_concentration' or "
            f"'halfnormal_inverse_sqrt', got {kappa_prior_family!r}"
        )
    if arm_gap_t1_prior_sigma is not None and dose:
        raise ValueError("arm_gap_t1_prior_sigma applies to the arm-by-wave baseline gap")
    if sigma_child_prior_sigma is not None and not use_child_re:
        raise ValueError("sigma_child_prior_sigma requires use_child_re=True")

    if dose:
        if prepared.phase_mode != "all":
            raise ValueError("DiD dose variants require phase_mode='all'")
        if own not in prepared.pre_logit:
            raise KeyError(f"Dose model needs the t1 baseline for outcome {own!r}")
        if "attend" not in prepared.covariates:
            raise KeyError("dose=True requires the 'attend' covariate")
        periods = tuple(int(p) for p in periods)
        if periods != (0, 1):
            raise ValueError(
                "DiD dose variants require periods=(0, 1); "
                f"got {periods}."
            )

        baseline_t1_all = _broadcast_phase_zero(
            prepared, prepared.pre_logit[own], label=f"{own} t1 baseline"
        )
        age_t1_all, age_scaler = _standardise_child_baseline(
            prepared, prepared.A_months, label="t1 age"
        )
        # #390 P3: a deliberate design restriction and a missing-data exclusion
        # are different facts about a row; subset in two labelled steps so the
        # persisted ``dropped_by_reason`` partitions them instead of folding
        # both into one opaque count.
        in_design = np.isin(prepared.phase, periods)
        baseline_t1_all = baseline_t1_all[in_design]
        age_t1_all = age_t1_all[in_design]
        prepared = _subset(prepared, in_design, reason="design_excluded")
        observed = (
            np.isfinite(prepared.post_counts[own])
            & np.isfinite(prepared.covariates["attend"])
            & np.isfinite(baseline_t1_all)
        )
        baseline_t1 = baseline_t1_all[observed]
        age_t1 = age_t1_all[observed]
        prepared = _subset(prepared, observed, reason="missing_data")
        post = prepared.post_counts[own].astype(np.int64)
        is_p2 = (prepared.phase == 1).astype(float)
        G_f = prepared.G.astype(float)
        treated = ((prepared.G == 1) | (prepared.phase == 1)).astype(float)

        loaded_attend = np.asarray(prepared.covariates["attend"], dtype=float)
        loaded_scaler = prepared.covariate_scalers["attend"]
        raw_attend = loaded_attend * loaded_scaler.sd + loaded_scaler.mean
        treated_z, dose_scaler = standardise(raw_attend[treated == 1])
        dose_centered = np.where(
            treated == 1,
            (raw_attend - dose_scaler.mean) / dose_scaler.sd,
            0.0,
        )
        if not np.allclose(dose_centered[treated == 1], treated_z):
            raise AssertionError("Treated-centred dose standardisation drifted")

        obs_ids = np.asarray(
            [
                f"{subject}|P{int(phase) + 1}"
                for subject, phase in zip(
                    prepared.subject_ids, prepared.phase, strict=True
                )
            ]
        )
        coords: dict[str, Any] = {
            "obs_id": obs_ids,
            "child": np.arange(prepared.n_children),
        }
        if period_varying_dose:
            coords["dose_phase"] = ["P1", "P2"]

        with pm.Model(coords=coords) as model:
            period_d = pm.Data("period", is_p2, dims="obs_id")
            G_d = pm.Data("G", G_f, dims="obs_id")
            treated_d = pm.Data("treated", treated, dims="obs_id")
            baseline_t1_d = pm.Data(
                "baseline_t1_logit", baseline_t1, dims="obs_id"
            )
            dose_d = pm.Data("dose_treated_std", dose_centered, dims="obs_id")

            alpha = _priors.alpha_prior(
                sigma=_alpha_sigma_for(outcome_symbol)
            ).to_pymc("alpha")
            beta_period = _priors.tau_prior().to_pymc("beta_period")
            beta_group = _priors.gamma_cross_prior().to_pymc("beta_group")
            theta_treated = _priors.tau_prior(
                sigma=_tau_sigma_for(own)
            ).to_pymc("theta_treated")
            gamma_t1 = _priors.gamma_own_prior().to_pymc("gamma_t1")
            eta = (
                alpha
                + beta_period * period_d
                + beta_group * G_d
                + theta_treated * treated_d
                + gamma_t1 * baseline_t1_d
            )
            if use_age:
                age_t1_d = pm.Data("A_t1_std", age_t1, dims="obs_id")
                gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
                eta = eta + gamma_A * age_t1_d
            if use_child_re:
                child_idx_d = pm.Data(
                    "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
                )
                eta = _add_child_random_intercept(
                    eta, child_idx_d, sigma_prior_sigma=0.5
                )

            eta_base = pm.Deterministic("eta_base", eta, dims="obs_id")
            _dose_slope_prior = (
                _priors.beta_mech_prior()
                if dose_slope_prior_sigma is None
                else _priors.beta_mech_prior(sigma=float(dose_slope_prior_sigma))
            )
            if period_varying_dose:
                dose_phase_idx = pm.Data(
                    "dose_phase_idx", prepared.phase.astype(np.int64), dims="obs_id"
                )
                mu_dose = _dose_slope_prior.to_pymc("mu_dose")
                sigma_dose = _priors.sigma_dose_phase_prior().to_pymc(
                    "sigma_dose"
                )
                beta_dose_phase = pm.Deterministic(
                    "beta_dose_phase",
                    mu_dose
                    + sigma_dose
                    * pm.Normal(
                        "beta_dose_phase_raw", 0.0, 1.0, dims="dose_phase"
                    ),
                    dims="dose_phase",
                )
                eta_full = eta_base + beta_dose_phase[dose_phase_idx] * dose_d
            else:
                beta_dose = _dose_slope_prior.to_pymc("beta_dose")
                eta_full = eta_base + beta_dose * dose_d

            eta_full = pm.Deterministic("eta", eta_full, dims="obs_id")
            kappa = _scalar_prior("kappa", _priors.kappa_prior)
            beta_binomial_from_logit(
                "y_post",
                eta_full,
                n_trials=prepared.n_trials[own],
                kappa=kappa,
                observed=post,
                dims="obs_id",
            )

        return BuiltModel(
            model=model,
            prepared=prepared,
            payload=DidDosePayload(
                design="dose_intensive_margin",
                dose_scaler=dose_scaler,
                age_t1_scaler=age_scaler,
                analysis_row_ids=obs_ids,
                raw_attend=raw_attend,
                dose_treated_std=dose_centered,
                treated=treated,
            ),
        )

    if prepared.phase_mode != "levels":
        raise ValueError("Binary DiD triangulation requires phase_mode='levels'")
    waves = tuple(int(w) for w in waves)
    if waves != (0, 1, 2):
        raise ValueError(
            "Binary DiD triangulation requires waves=(0, 1, 2); "
            f"got {waves}."
        )

    age_t1_all, age_scaler = _standardise_child_baseline(
        prepared, prepared.A_months, label="t1 age"
    )
    # #390 P3: partition the exclusions — rows outside the modelled waves leave
    # by design (the levels frame carries t4, which this model does not fit);
    # rows with an unobserved outcome leave as missing data.
    in_design = np.isin(prepared.phase, waves)
    age_t1_all = age_t1_all[in_design]
    prepared = _subset(prepared, in_design, reason="design_excluded")
    observed = np.isfinite(prepared.post_counts[own])
    age_t1 = age_t1_all[observed]
    prepared = _subset(prepared, observed, reason="missing_data")
    post = prepared.post_counts[own].astype(np.int64)
    n_trials = prepared.n_trials[own]

    # Every arm-by-wave cell the model estimates a coefficient for must contain
    # both arms, or that coefficient is prior-only — a parameter reported as an
    # estimate with no data behind it. This used to surface, if at all, during
    # reporting, long after the fit (#576 lower-severity 6). Current data fill
    # every cell; the check is here so a future delivery that empties one fails
    # loudly at build time instead.
    for _wave_code, _wave_label in ((0, "t1"), (1, "t2"), (2, "t3")):
        _rows = prepared.phase == _wave_code
        _arms = set(np.unique(prepared.G[_rows]).astype(int).tolist())
        if _arms != {0, 1}:
            raise ValueError(
                f"the arm-by-wave DiD requires both arms at every wave: {_wave_label} "
                f"has {sorted(_arms) or 'no rows'} after the missing-outcome mask, so "
                f"its arm-gap coefficient would be prior-only"
            )

    alpha_anchor: float | None = None
    if use_intercept_anchor:
        t1 = post[prepared.phase == 0]
        if not t1.size:
            raise ValueError(f"Cannot anchor {own}: no observed t1 outcome values")
        if likelihood == "bernoulli_offfloor":
            movers = int(np.sum(t1 > 0))
            alpha_anchor = float(
                np.log((movers + 0.5) / (t1.size - movers + 0.5))
            )
        else:
            successes = float(np.sum(t1))
            failures = float(t1.size * n_trials - successes)
            # The anchor locates the intercept prior on the LINEAR PREDICTOR, so
            # under a non-identity score-mean link the observed proportion must be
            # mapped back through the link first — the same discipline the level
            # family adopted in #584 decision 2. Anchoring a guessing-floor fit on
            # the raw observed logit would put the prior roughly one logit unit
            # away from the value the floor link needs.
            proportion = (successes + 0.5) / (successes + failures + 1.0)
            unit = float(invert_score_mean_link(proportion, score_mean_link))
            alpha_anchor = float(np.log(unit / (1.0 - unit)))

    obs_ids = np.asarray(
        [
            f"{subject}|t{int(wave) + 1}"
            for subject, wave in zip(
                prepared.subject_ids, prepared.phase, strict=True
            )
        ]
    )
    coords: dict[str, Any] = {
        "obs_id": obs_ids,
        "child": np.arange(prepared.n_children),
        "wave": ["t1", "t2", "t3"],
        "post_wave": ["t2", "t3"],
    }
    waitlist_subjects = np.unique(
        prepared.subject_ids[(prepared.G == 0) & (prepared.phase == 2)]
    )
    if use_varying_delta:
        if not waitlist_subjects.size:
            raise ValueError("Crossover heterogeneity requires waitlist children")
        coords["waitlist_child"] = waitlist_subjects.astype(str)

    with pm.Model(coords=coords) as model:
        wave_d = pm.Data(
            "wave_idx", prepared.phase.astype(np.int64), dims="obs_id"
        )
        G_d = pm.Data("G", prepared.G.astype(float), dims="obs_id")
        child_idx_d = pm.Data(
            "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
        )

        if use_intercept_anchor:
            alpha_offset = _priors.alpha_prior(
                sigma=_alpha_sigma_for(outcome_symbol)
            ).to_pymc("alpha_offset")
            alpha = pm.Deterministic("alpha", alpha_anchor + alpha_offset)
        else:
            # The independent-prior sensitivity (#390 P1 condition 1): the same
            # zero-centred tier-scale prior the dose variants use, with no
            # outcome-informed location at all.
            alpha = _priors.alpha_prior(
                sigma=_alpha_sigma_for(outcome_symbol)
            ).to_pymc("alpha")
        beta_period = _priors.tau_prior().to_pymc(
            "beta_period", dims="post_wave"
        )
        wave_offset = pt.concatenate(
            [pt.zeros((1,), dtype=beta_period.dtype), beta_period]
        )
        arm_gap_t1 = (
            _priors.gamma_cross_prior()
            if arm_gap_t1_prior_sigma is None
            else _priors.gamma_cross_prior(sigma=float(arm_gap_t1_prior_sigma))
        ).to_pymc("arm_gap_t1")
        tau_t2 = _priors.tau_prior(
            sigma=(
                _tau_sigma_for(own)
                if tau_t2_prior_sigma is None
                else float(tau_t2_prior_sigma)
            )
        ).to_pymc("tau_t2")
        arm_gap_t3 = _priors.tau_prior(sigma=_tau_sigma_for(own)).to_pymc(
            "arm_gap_t3"
        )
        arm_gap_wave = pm.Deterministic(
            "arm_gap_wave",
            pt.stack([arm_gap_t1, tau_t2, arm_gap_t3]),
            dims="wave",
        )
        delta_crossover = pm.Deterministic(
            "delta_crossover", tau_t2 - arm_gap_t3
        )

        eta_base = alpha + wave_offset[wave_d]
        if use_age:
            age_t1_d = pm.Data("A_t1_std", age_t1, dims="obs_id")
            gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
            eta_base = eta_base + gamma_A * age_t1_d
        if use_child_re:
            eta_base = _add_child_random_intercept(
                eta_base,
                child_idx_d,
                sigma_prior_sigma=(
                    0.5
                    if sigma_child_prior_sigma is None
                    else float(sigma_child_prior_sigma)
                ),
            )
        eta_base = pm.Deterministic("eta_base", eta_base, dims="obs_id")
        eta_full = eta_base + arm_gap_wave[wave_d] * G_d

        if use_varying_delta:
            waitlist_index = {s: i for i, s in enumerate(waitlist_subjects)}
            safe_idx = np.asarray(
                [waitlist_index.get(s, 0) for s in prepared.subject_ids],
                dtype=np.int64,
            )
            waitlist_t3 = ((prepared.G == 0) & (prepared.phase == 2)).astype(float)
            waitlist_idx_d = pm.Data(
                "waitlist_crossover_idx", safe_idx, dims="obs_id"
            )
            waitlist_t3_d = pm.Data(
                "waitlist_t3", waitlist_t3, dims="obs_id"
            )
            sigma_delta = _priors.sigma_delta_prior().to_pymc("sigma_delta")
            v_delta = pm.Deterministic(
                "v_delta",
                sigma_delta
                * pm.Normal(
                    "v_delta_raw", 0.0, 1.0, dims="waitlist_child"
                ),
                dims="waitlist_child",
            )
            pm.Deterministic(
                "delta_crossover_i",
                delta_crossover + v_delta,
                dims="waitlist_child",
            )
            eta_full = eta_full + v_delta[waitlist_idx_d] * waitlist_t3_d

        eta_full = pm.Deterministic("eta", eta_full, dims="obs_id")
        if likelihood == "beta_binomial":
            if kappa_prior_family == "halfnormal_inverse_sqrt":
                # The registered dispersion sensitivity (#576 material
                # qualification 2): a HalfNormal on the concentration cannot
                # reach the near-Binomial limit for a long test, so it imposes a
                # floor on the estimated over-dispersion. The dispersion-scale
                # parameterisation can conclude there is none.
                kappa = _rlm_dispersion_kappa(
                    float(_priors.inv_sqrt_kappa_prior().sigma)
                    if kappa_prior_sigma is None
                    else float(kappa_prior_sigma)
                )
            else:
                kappa = (
                    _priors.kappa_prior()
                    if kappa_prior_sigma is None
                    else _priors.kappa_prior(sigma=float(kappa_prior_sigma))
                ).to_pymc("kappa")
            beta_binomial_from_score_mean_link(
                "y_post",
                eta_full,
                n_trials=n_trials,
                kappa=kappa,
                score_mean_link=score_mean_link,
                observed=post,
                dims="obs_id",
            )
        else:
            pm.Bernoulli(
                "y_offfloor",
                logit_p=eta_full,
                observed=(post > 0).astype(np.int64),
                dims="obs_id",
            )

    return BuiltModel(
        model=model,
        prepared=prepared,
        payload=DidArmWavePayload(
            design="arm_by_wave",
            alpha_anchor=alpha_anchor,
            age_t1_scaler=age_scaler,
            analysis_row_ids=obs_ids,
            waves=waves,
            score_mean_link=score_mean_link,
        ),
    )
