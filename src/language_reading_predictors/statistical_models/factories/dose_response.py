# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dose-response model construction.

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
    DoseResponsePayload,
)
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    beta_binomial_from_score_mean_link,
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
    _broadcast_phase_zero_optional,
    _tau_sigma_for,
)

def build_dose_response_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    adjust_baseline_symbol: str = "W",
    dose_covariate: str = "attend",
    # Default OFF (issue #269): conditioning on cumulative prior dose (a running sum
    # of the IS collider) reopens the latent-GA backdoor, so the headline fits do not
    # adjust it. Set it explicitly only for the flagged collider-sensitivity fit.
    dose_stage_covariate: str | None = None,
    period_varying_dose: bool = True,
    use_subject_random_intercept: bool = True,
    adjust_group: bool = True,
    adjust_age: bool = True,
    ability_adjust_symbols: Iterable[str] = (),
    ability_baseline_wave: str = "t1",
    decompose_between_within: bool = True,
    sigma_child_prior_sigma: float = 0.5,
    # #619: the phoneme-blending response link. B only, and released only beside its
    # paired ordinary-link fit. The family's focal estimand is a natural-scale dose
    # marginal, so it inherits the link exactly as a treatment contrast does.
    score_mean_link: ScoreMeanLink = "logit",
) -> BuiltModel[DoseResponsePayload]:
    """Period-resolved dose-response on the outcome post-score (#104 Phase 2, #587).

    Outcome-generic: the target is ``outcome_symbol`` (default ``"W"``, word
    reading — its lead use in LRP77; reused for letter sounds ``"L"`` in LRP86)
    and the autoregressive baseline is ``adjust_baseline_symbol`` (default the
    same measure).

    Estimand: how the **intensity** of intervention attendance relates to the
    outcome's **conditional change** — the outcome post-count modelled
    Beta-Binomial conditional on its own baseline logit — among rows that are on
    the intervention, and whether that slope **varies by period**.

    Uses all three phase transitions (``prepared.phase_mode == "all"``). The
    linear predictor is

        eta = alpha + alpha_phase[p]              # reference-coded, alpha_phase[0] = 0
            + theta_treated * treated             # extensive margin (see below)
            + beta_arm_late * G * 1{p >= 1}       # post-crossover arm / order only
            + gamma_own * logit(outcome_pre)      # autoregression / RTM
            + gamma_A * z(age)                    # maturation precision covariate
            + u_child[child]                      # subject random intercept
            + dose_terms                          # intensive margin (see below)
            + gamma_dose_stage * z(attend_cumul)  # flagged collider sensitivity
            + sum_s gamma_s_pre * logit(s_pre)    # optional ability adjusters

    **Separating the two margins (#587 finding 2).** Before this repair a single
    ``beta_dose_phase[p] * z(attend)`` term carried both margins at once. In
    period 1 every waitlist child has exactly zero sessions and every
    immediate-arm child has 45–91, so arm and dose correlate at 0.970 and the
    period-1 "dose" slope was, arithmetically, the randomised treatment contrast
    divided by the mean dose — not an intensity effect at all. Sessions now enter
    **centred and standardised over the fitted on-intervention rows only**, with
    every untreated row contributing exactly zero to every dose term, and a
    separate ``theta_treated`` indicator carries the extensive margin. Because
    ``treated`` and ``G`` are identical within period 1, assigned arm is admitted
    only from period 2 onward (``beta_arm_late``), where both arms are on the
    intervention and arm reads as intervention order; fitting both in period 1
    would be exact collinearity, and pretending otherwise is what let unmodelled
    arm differences load onto the dose slopes.

    **Between versus within child.** A lone dose coefficient over a child random
    intercept returns a precision-weighted *blend* of the between-child and
    within-child associations, which answer different questions — the random
    intercept does not make it a within-child quantity. With
    ``decompose_between_within`` (the default) the treated-centred dose is split
    Mundlak-style into each child's study-average attendance
    (``beta_dose_between``) and their deviation from it (the period slopes), so
    neither is blended into the other.

    **Ability adjusters.** ``ability_baseline_wave="t1"`` (the default)
    broadcasts each child's verified pre-randomisation value across all three
    transitions. ``"transition_start"`` reproduces the pre-#587 behaviour, which
    used t2 skills in period 2 and t3 skills in period 3 — values downstream of
    earlier intervention and dose, so a treatment-affected time-varying covariate
    rather than a baseline. It is retained only as a labelled comparator.

    Causal note (revised DAG): the DAG carries ``A -> IS``, ``GA -> IS`` and
    ``IG -> IS``, so age, latent general ability and assigned group are all
    causes of attendance as well as of the outcomes. Age and arm are adjusted;
    latent ability is not closed by conditioning on measured baselines, which is
    why ``ability_adjust_symbols`` is a *sensitivity*, never a proof. Every dose
    coefficient here is an adjusted association. The single exception is
    ``theta_treated`` read in period 1, which is a randomised contrast.

    Parameters mirror :func:`build_mechanism_model`'s backbone options
    (``use_subject_random_intercept``, ``adjust_baseline_symbol``). The arm and
    age covariates are toggled by ``adjust_group`` / ``adjust_age``;
    ``dose_stage_covariate=None`` drops the cumulative-dose control.
    """
    if prepared.phase_mode != "all":
        raise ValueError("Dose-response factory requires phase_mode='all'")
    if outcome_symbol not in prepared.pre_logit:
        raise KeyError(f"Outcome {outcome_symbol!r} missing from prepared data")
    if adjust_baseline_symbol not in prepared.pre_logit:
        raise KeyError(
            f"Baseline {adjust_baseline_symbol!r} missing from prepared data"
        )
    if dose_covariate not in prepared.covariates:
        raise KeyError(
            f"Dose covariate {dose_covariate!r} missing from prepared.covariates; "
            "pass it via load_and_prepare(covariates=...)"
        )
    if dose_stage_covariate is not None and dose_stage_covariate not in prepared.covariates:
        raise KeyError(
            f"Dose-stage covariate {dose_stage_covariate!r} missing from "
            "prepared.covariates"
        )
    ability_adjust_symbols = tuple(ability_adjust_symbols)
    for s in ability_adjust_symbols:
        if s not in prepared.pre_logit:
            raise KeyError(
                f"Ability-adjuster {s!r} has no pre-score; add it to "
                "load_and_prepare(outcomes=...)"
            )
    if ability_baseline_wave not in {"t1", "transition_start"}:
        raise ValueError(
            "ability_baseline_wave must be 't1' or 'transition_start'; got "
            f"{ability_baseline_wave!r}"
        )

    # Ability adjusters are resolved to their t1 value *before* the outcome mask, so
    # the phase-zero row a child needs for the broadcast is still present even when
    # that child's period-1 outcome is missing (#587 finding 1).
    ability_values: dict[str, np.ndarray] = {}
    for s in ability_adjust_symbols:
        if ability_baseline_wave == "t1":
            ability_values[s] = _broadcast_phase_zero_optional(
                prepared, prepared.pre_logit[s], label=f"{s} t1 baseline"
            )
        else:
            ability_values[s] = np.asarray(prepared.pre_logit[s], dtype=float)

    outcome_post = prepared.post_counts[outcome_symbol]
    keep = ~np.isnan(outcome_post)
    if ability_values:
        # A child with no verified t1 ability row cannot enter a baseline-ability
        # sensitivity; drop those rows explicitly rather than silently substituting a
        # later, treatment-affected wave.
        has_ability = np.ones(prepared.n_obs, dtype=bool)
        for values in ability_values.values():
            has_ability &= np.isfinite(values)
        keep = keep & has_ability
    prepared = _subset(prepared, keep)
    ability_values = {s: v[keep] for s, v in ability_values.items()}

    outcome_post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N_outcome = prepared.n_trials[outcome_symbol]
    own_pre_logit = prepared.pre_logit[adjust_baseline_symbol]
    phase_idx = np.asarray(prepared.phase, dtype=int)

    # ---- Dose design: extensive vs intensive margin (#587 findings 2, 3, 13) ----
    # Recover raw sessions from the loader's standardisation, then re-standardise over
    # the *fitted* on-intervention rows. Doing it here — after the outcome mask — is
    # what makes the recorded "standardised over the fitted rows" claim true; the
    # loader scaler is defined over the pre-mask row set.
    loader_scaler = prepared.covariate_scalers[dose_covariate]
    raw_dose = (
        np.asarray(prepared.covariates[dose_covariate], dtype=float) * loader_scaler.sd
        + loader_scaler.mean
    )
    treated = (raw_dose > 0.0).astype(float)
    if not treated.any():
        raise ValueError(
            "Dose-response factory found no on-intervention row: the intensive-margin "
            "estimand is undefined without one."
        )
    treated_mask = treated == 1.0
    dose_treated_z, dose_scaler = standardise(raw_dose[treated_mask])
    dose_c = np.zeros_like(raw_dose)
    dose_c[treated_mask] = dose_treated_z

    # Mundlak split of the treated-centred dose. The child mean is taken over that
    # child's treated rows only, and both components are zeroed on untreated rows so
    # the extensive margin stays entirely in ``theta_treated``.
    child_idx = np.asarray(prepared.child_idx, dtype=int)
    dose_between = np.zeros_like(dose_c)
    for child in np.unique(child_idx[treated_mask]):
        rows = (child_idx == child) & treated_mask
        dose_between[rows] = dose_c[rows].mean()
    dose_within = np.where(treated_mask, dose_c - dose_between, 0.0)

    # Per-phase observed session support behind the reported items-scale contrast.
    phase_support: list[tuple[float, float, float, float]] = []
    for p in range(prepared.n_phases):
        rows = treated_mask & (phase_idx == p)
        if rows.any():
            q1, q3 = np.percentile(raw_dose[rows], [25.0, 75.0])
            phase_support.append(
                (float(raw_dose[rows].min()), float(q1), float(q3), float(raw_dose[rows].max()))
            )
        else:
            phase_support.append((float("nan"),) * 4)

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "phase_later": np.arange(1, prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }

    with pm.Model(coords=coords) as model:
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", prepared.G.astype(float), dims="obs_id")
        own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")
        phase_d = pm.Data("phase_idx", phase_idx.astype(np.int64), dims="obs_id")
        child_idx_d = pm.Data(
            "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
        )
        # Whole-child LOO map (#587 finding 4). ``diagnostics`` picks this up and
        # aggregates the pointwise likelihood within child, because a transition row's
        # own baseline IS the previous transition's fitted outcome — leaving one row
        # out would leave its held-out score in the next row's design matrix.
        pm.Data("loo_child_idx", prepared.child_idx.astype(np.int64), dims="obs_id")
        treated_d = pm.Data("treated", treated, dims="obs_id")
        late_d = pm.Data("late_phase", (phase_idx >= 1).astype(float), dims="obs_id")
        dose_between_d = pm.Data(
            f"{dose_covariate}_child_mean_std", dose_between, dims="obs_id"
        )
        dose_within_d = pm.Data(
            f"{dose_covariate}_within_dev_std", dose_within, dims="obs_id"
        )
        dose_d = pm.Data(f"{dose_covariate}_treated_std", dose_c, dims="obs_id")
        dose_stage_d = None
        if dose_stage_covariate is not None:
            dose_stage_d = pm.Data(
                f"{dose_stage_covariate}_std",
                prepared.covariates[dose_stage_covariate],
                dims="obs_id",
            )
        ability_data: dict[str, pt.TensorVariable] = {}
        for s in ability_adjust_symbols:
            ability_data[s] = pm.Data(
                f"{s}_pre_logit", ability_values[s], dims="obs_id"
            )

        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
        # Reference-coded phase intercepts (#587 finding 11): a grand intercept plus
        # three unconstrained phase indicators is a rank-3 design in four columns, so
        # the nuisance split was prior-identified only. Period 1 is the reference and
        # the later periods carry free deviations from it.
        alpha_phase_free = _priors.declare(
                               pm.Normal(
                                           "alpha_phase_free", mu=0.0, sigma=0.5, dims="phase_later"
                                       ),
                               role="nuisance",
                               rationale=(
                                   "Reference-coded period intercept deviations from period 1 "
                                   "(alpha_phase[1] = 0 exactly), so the intercept design has full "
                                   "rank."
                               ),
                           )
        alpha_phase = pm.Deterministic(
            "alpha_phase",
            pt.concatenate([pt.zeros(1), alpha_phase_free]),
            dims="phase",
        )
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")

        eta = alpha + alpha_phase[phase_d] + gamma_own * own_pre_d

        # Extensive margin: on the intervention this period versus not. In period 1
        # this is exactly the randomised arm contrast (every immediate-arm child
        # attended, every waitlist child attended zero sessions).
        theta_treated = _priors.tau_prior(
            sigma=_tau_sigma_for(outcome_symbol)
        ).to_pymc("theta_treated")
        eta = eta + theta_treated * treated_d

        if adjust_group:
            # Arm enters only from period 2, where both arms are on the intervention
            # and it reads as intervention order / treatment history. In period 1 it
            # would be exactly collinear with ``treated``.
            beta_arm_late = _priors.gamma_cross_prior().to_pymc("beta_arm_late")
            eta = eta + beta_arm_late * G_d * late_d
        if adjust_age:
            gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
            eta = eta + gamma_A * A_std_d

        if use_subject_random_intercept:
            eta = _add_child_random_intercept(
                eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
            )

        # Intensive margin. The dose is centred and standardised over the fitted
        # on-intervention rows, so every slope is per 1 SD of *treated* sessions and
        # every untreated row contributes exactly zero here.
        if decompose_between_within:
            beta_dose_between = _priors.beta_mech_prior().to_pymc("beta_dose_between")
            eta = eta + beta_dose_between * dose_between_d
            dose_slope_target = dose_within_d
        else:
            dose_slope_target = dose_d

        if period_varying_dose:
            mu_dose = _priors.beta_mech_prior().to_pymc("mu_dose")
            sigma_dose = _priors.sigma_dose_phase_prior().to_pymc("sigma_dose")
            beta_dose_phase_raw = _priors.declare(
                                      pm.Normal(
                                                      "beta_dose_phase_raw", mu=0.0, sigma=1.0, dims="phase"
                                                  ),
                                      role="nuisance",
                                      rationale=(
                                          "Standard-normal non-centred period-dose offset; scaled by "
                                          "sigma_dose."
                                      ),
                                  )
            beta_dose_phase = pm.Deterministic(
                "beta_dose_phase", mu_dose + sigma_dose * beta_dose_phase_raw, dims="phase"
            )
            eta = eta + beta_dose_phase[phase_d] * dose_slope_target
        else:
            beta_dose = _priors.beta_mech_prior().to_pymc("beta_dose")
            eta = eta + beta_dose * dose_slope_target

        # Dose-stage control (prior cumulative dose), so a dose-stage effect is
        # not misread as a period effect.
        if dose_stage_d is not None:
            gamma_dose_stage = _priors.gamma_cross_prior().to_pymc("gamma_dose_stage")
            eta = eta + gamma_dose_stage * dose_stage_d

        # Baseline-skill (ability) adjusters - the no-g->dose sensitivity fit.
        for s in ability_adjust_symbols:
            gamma_s = _priors.gamma_cross_prior().to_pymc(f"gamma_{s}_pre")
            eta = eta + gamma_s * ability_data[s]

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        kappa = _priors.kappa_prior().to_pymc("kappa")

        beta_binomial_from_score_mean_link(
            "y_post",
            eta,
            n_trials=N_outcome,
            kappa=kappa,
            score_mean_link=score_mean_link,
            observed=outcome_post,
            dims="obs_id",
        )

    return BuiltModel(
        model=model,
        prepared=prepared,
        payload=DoseResponsePayload(
            design="dose_intensive_margin",
            dose_scaler=dose_scaler,
            treated=treated_mask,
            raw_attend=raw_dose,
            dose_between=dose_between,
            dose_within=dose_within,
            phase_support=tuple(phase_support),
            decompose_between_within=decompose_between_within,
            score_mean_link=score_mean_link,
        ),
    )
