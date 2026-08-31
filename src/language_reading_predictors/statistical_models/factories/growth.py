# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Latent growth-curve model construction.

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


from language_reading_predictors.statistical_models.fitted_payloads import (
    EmptyPayload,
)
from language_reading_predictors.statistical_models.preprocessing import (
    WavePanel,
    standardise,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
)
from language_reading_predictors.statistical_models.invariants import (
    require_value,
)
from language_reading_predictors.statistical_models import priors as _priors

def build_growth_model(
    panel: WavePanel,
    *,
    baseline_covariate: str = "blocks",
    use_shared_factor: bool = False,
    use_random_slope: bool = True,
    age_ability_interaction: bool = False,
    adjust_for_group: bool = False,
    intercept_prior_sigma: float = 1.5,
    slope_prior_sigma: float = 0.5,
    assoc_prior_sigma: float = 0.3,
    re_intercept_prior_sigma: float = 0.5,
    re_slope_prior_sigma: float = 0.5,
    loading_prior_sigma: float = 0.5,
    kappa_prior_sigma: float = 50.0,
) -> BuiltModel[EmptyPayload, WavePanel]:
    """Joint multivariate latent growth-curve model (LRP69/70) on the logit scale.

    Characterises each measure's within-child trajectory across the waves and asks
    whether a **baseline** covariate (``blocks``, the t1-only WPPSI Block Design
    non-verbal score) predicts trajectory *shape*. For measure ``k``, child ``i``,
    wave ``t`` (with ``a`` = standardised age)::

        theta[i,t,k]   = intercept[i,k] + slope[i,k] * a[i,t]
        intercept[i,k] = alpha_k + delta_k * z(blocks_i) + sigma0_k * z0[i,k]
        slope[i,k]     = beta_k  + gamma_k * z(blocks_i) + [loading_k * G_i]
                                 + sigma1_k * z1[i,k]
        y[i,t,k] ~ BetaBinomial(N_k, mu = sigmoid(theta[i,t,k]), kappa_k)

    Growth is **linear in standardised age** — the identifiable choice at four
    waves. ``gamma_k`` (baseline non-verbal ability -> growth *rate*) is the
    headline Q5 estimand; ``delta_k`` is the association with the level at the
    pooled-mean (mid-study) age (``a`` is standardised over all child-wave cells,
    so the entry-level association is ``delta_k + gamma_k * E[a at t1]``). Both are
    **adjusted / GA-confounded associations, never causal** (block design is an
    off-DAG ability proxy; see ``notes/202606231600-dag-revision-consolidated.md``).

    The child-level random intercept and slope are **independent per measure** —
    the within-measure intercept-slope correlation is deliberately omitted at
    n~54, mirroring the joint ITT model's disabled LKJ residual correlation (found
    prior-dominated at this sample size, ``notes/202604181600-lrp52-58-findings.md``).
    Everything is non-centred for sampling.

    ``use_shared_factor`` adds a rank-1 shared child-level growth-tempo factor
    ``G_i ~ Normal(0, 1)`` loading (positively, for identification) on every
    measure's slope — the genuinely *joint* layer (LRP70): does a common
    developmental tempo couple the measures, and (read out post-hoc) does baseline
    non-verbal ability predict it? ``LOO(LRP69 vs LRP70)`` shows whether the factor
    earns its keep. The core LRP69 keeps ``use_shared_factor=False``.

    ``adjust_for_group`` is the historical-cohort port: it gives each observed
    reading group its own population intercept, age slope, random-effect scales and
    Beta-Binomial concentration while keeping ``delta`` and ``gamma`` as common
    within-group ability associations. It is nuisance adjustment, not a group-effect
    estimand. The RLI models leave it off and retain their original parameterisation.

    ``use_random_slope=False`` retains the child random intercept but removes the
    child-specific slope residual. The three-wave, single-outcome Byrne port uses
    this reduced form because a separate latent slope for every child is weakly
    supported and produced unreliable observation-level PSIS-LOO diagnostics. The
    fixed baseline-ability association with growth rate (``gamma``) remains.

    ``age_ability_interaction`` (LRP85, #228 item 10) adds a child-level **baseline
    (t1) age** moderator ``age0`` (standardised across children — distinct from the
    within-child ``age_std`` time axis) to the slope, with its own main effect
    ``gamma_age`` and, headline, an ``age0 × ability`` interaction ``gamma_int``:
    positive ``gamma_int_k`` = older-and-more-able children grow faster on measure k
    than age and ability predict separately (the gain factors' ``gamma_int_A_ability``
    brought onto the growth rate). Default off, so LRP69/70 are unaffected. Still an
    adjusted, GA-confounded association.

    Observed counts enter via a **masked** Beta-Binomial (the LRP55 flattened-mask
    idiom): only the unmasked cells in ``panel.obs_mask`` are observed, so a child
    missing one score still contributes its other waves. The intervention-dose
    covariate is **omitted** (the locked DAG's ``IS`` collider, as in
    :func:`build_lcsm_model`).
    """
    OUT = tuple(panel.outcomes)
    K = len(OUT)
    N = panel.n_children
    T = panel.n_waves
    if T < 2:
        raise ValueError("growth model needs at least two waves")
    if baseline_covariate not in panel.baseline:
        raise KeyError(
            f"baseline_covariate {baseline_covariate!r} not loaded; pass "
            f"baseline_covariates=({baseline_covariate!r},) to load_wave_panel."
        )
    if adjust_for_group and panel.group is None:
        raise ValueError("adjust_for_group=True requires a group code for every child")

    # Observed counts / mask / denominators stacked as (N, T, K) in OUT order.
    counts_int = np.stack(
        [np.nan_to_num(panel.counts[s], nan=0.0).astype(np.int64) for s in OUT],
        axis=2,
    )
    mask = np.stack([panel.obs_mask[s] for s in OUT], axis=2)  # (N, T, K) bool
    n_trials_vec = np.array([panel.n_trials[s] for s in OUT], dtype=int)  # (K,)
    zb = np.asarray(panel.baseline[baseline_covariate], dtype=float)  # (N,) standardised

    # Intercept anchor: grand-mean observed logit per measure (the intercept is the
    # logit level at mean age, age_std = 0). The historical-cohort port anchors each
    # group separately. Guard the all-NaN case loudly.
    missing = [s for s in OUT if not np.isfinite(panel.logit[s]).any()]
    if missing:
        raise ValueError(
            "growth intercept anchor is undefined (no observed value) for: "
            f"{', '.join(missing)}."
        )
    group_values: np.ndarray | None = None
    group_idx: np.ndarray | None = None
    if adjust_for_group:
        panel_group = require_value(panel.group, "panel.group")
        group_values = np.asarray(sorted(set(panel_group.astype(int))), dtype=int)
        group_lookup = {int(code): index for index, code in enumerate(group_values)}
        group_idx = np.asarray(
            [group_lookup[int(code)] for code in panel_group], dtype=np.int64
        )
        intercept_anchor = np.array(
            [
                [np.nanmean(panel.logit[s][group_idx == group_index]) for s in OUT]
                for group_index in range(len(group_values))
            ],
            dtype=float,
        )
        if not np.isfinite(intercept_anchor).all():
            raise ValueError(
                "growth intercept anchor is undefined for at least one group/outcome cell"
            )
    else:
        intercept_anchor = np.array(
            [np.nanmean(panel.logit[s]) for s in OUT], dtype=float
        )

    coords = {"child": np.arange(N), "wave": panel.waves, "outcome": list(OUT)}
    if group_values is not None:
        coords["reading_group"] = group_values

    from dse_research_utils.math.constants import EPSILON  # local import

    with pm.Model(coords=coords) as model:
        age = pm.Data("age_std", panel.age_std, dims=("child", "wave"))
        baseline_name = "blocks_std" if not adjust_for_group else "baseline_std"
        blocks = pm.Data(baseline_name, zb, dims="child")
        group_data = (
            pm.Data("group_idx", group_idx, dims="child")
            if group_idx is not None
            else None
        )

        # Population growth parameters (per measure).
        population_dims = (
            ("reading_group", "outcome") if adjust_for_group else "outcome"
        )
        alpha = _priors.declare(
            pm.Normal(
                "alpha",
                mu=intercept_anchor,
                sigma=intercept_prior_sigma,
                dims=population_dims,
            ),
            role="nuisance",
            panel="alpha",
            rationale=(
                "Per-measure intercept on the logit scale, its mean anchored on the "
                "grand mean observed logit across all waves (not a baseline wave). "
                "Empirical Bayes: the prior mean is computed from the same observed "
                "outcomes that enter the likelihood, so this prior is not "
                "independent of the data and the reported prior-predictive "
                "distribution is partly data-informed."
            ),
        )
        beta = _priors.declare(
            pm.Normal("beta", mu=0.0, sigma=slope_prior_sigma, dims=population_dims),
            role="association",
            rationale=(
                "Per-measure population mean growth rate (slope on standardised "
                "age); a descriptive maturational trend, not a causal or "
                "adjusted-coupling term."
            ),
        )
        # Baseline non-verbal ability -> trajectory shape (the Q5 estimands):
        # delta on the baseline level, gamma on the growth rate (headline).
        delta = _priors.declare(
            pm.Normal("delta", mu=0.0, sigma=assoc_prior_sigma, dims="outcome"),
            role="association",
            panel="predictor_slope",
            rationale=(
                "Baseline non-verbal ability on the baseline *level* "
                "(Normal(0, 0.3)); an adjusted, latent-GA-confounded association, "
                "never causal. Shares a name with the ITT family's randomised "
                "``delta`` and is a different quantity."
            ),
        )
        gamma = _priors.declare(
            pm.Normal("gamma", mu=0.0, sigma=assoc_prior_sigma, dims="outcome"),
            role="association",
            panel="predictor_slope",
            rationale=(
                "Baseline non-verbal ability on the growth *rate* (Normal(0, 0.3)) "
                "— this family's headline shape estimand; an adjusted, "
                "latent-GA-confounded association, never causal."
            ),
        )
        # Child-level random intercept + slope (independent per measure).
        sigma_intercept = _priors.declare(
            pm.HalfNormal(
                "sigma_intercept", sigma=re_intercept_prior_sigma, dims=population_dims
            ),
            role="nuisance",
            rationale=(
                "Child random-intercept SD per measure (HalfNormal(0.5)); the "
                "between-child spread of starting level that ``z_intercept`` scales."
            ),
        )
        sigma_slope = (
            _priors.declare(
                pm.HalfNormal(
                    "sigma_slope", sigma=re_slope_prior_sigma, dims=population_dims
                ),
                role="nuisance",
                rationale=(
                    "Child random-slope SD per measure (HalfNormal(0.5)); the "
                    "between-child spread of growth rate that ``z_slope`` scales."
                ),
            )
            if use_random_slope
            else None
        )
        z_intercept = _priors.declare(
            pm.Normal("z_intercept", 0.0, 1.0, dims=("child", "outcome")),
            role="nuisance",
            rationale=(
                "Non-centred standard-normal per-child, per-measure intercept "
                "offsets (Normal(0, 1)); scaled by the random-intercept SD to form "
                "the child-by-measure growth intercepts."
            ),
        )
        z_slope = (
            _priors.declare(
                pm.Normal("z_slope", 0.0, 1.0, dims=("child", "outcome")),
                role="nuisance",
                rationale=(
                    "Non-centred standard-normal per-child, per-measure slope "
                    "offsets (Normal(0, 1)); scaled by the random-slope SD to form "
                    "the child-by-measure growth slopes."
                ),
            )
            if use_random_slope
            else None
        )
        kappa = _priors.declare(
            pm.HalfNormal("kappa", sigma=kappa_prior_sigma, dims=population_dims),
            role="nuisance",
            panel="kappa",
            rationale="Beta-binomial concentration kappa ~ HalfNormal(50).",
        )

        # child x outcome intercepts and slopes (non-centred).
        if group_data is not None:
            alpha_child = alpha[group_data]
            beta_child = beta[group_data]
            sigma_intercept_child = sigma_intercept[group_data]
            sigma_slope_child = (
                sigma_slope[group_data] if sigma_slope is not None else None
            )
        else:
            alpha_child = alpha[None, :]
            beta_child = beta[None, :]
            sigma_intercept_child = sigma_intercept[None, :]
            sigma_slope_child = (
                sigma_slope[None, :] if sigma_slope is not None else None
            )
        intercept = pm.Deterministic(
            "intercept",
            alpha_child
            + delta[None, :] * blocks[:, None]
            + sigma_intercept_child * z_intercept,
            dims=("child", "outcome"),
        )
        slope_mean = beta_child + gamma[None, :] * blocks[:, None]
        if age_ability_interaction:
            # Child-level baseline (t1) age, standardised ACROSS children — distinct
            # from the within-child ``age_std`` time axis the slope multiplies. Its
            # interaction with ability is the #228 item-10 estimand: older-and-more-
            # able children grow faster than age and ability predict separately,
            # bringing the gain factors' ``gamma_int_A_ability`` onto the growth rate.
            # ``gamma_int`` is on unit-scaled age0 × unit-scaled ability, matching the
            # gain-factor interaction's scale. Missing baseline age -> 0 (the mean).
            a0 = np.asarray(panel.age_std[:, 0], dtype=float)
            # Standardise across children with the shared helper (nanstd ddof=1,
            # matching every other standardised term; it raises on a degenerate
            # zero-variance axis rather than silently falling back to sd=1 and
            # fitting a flat interaction).
            age0_z, _ = standardise(a0)
            age0_np = np.where(np.isfinite(age0_z), age0_z, 0.0)
            age0 = pm.Data("age0_std", age0_np, dims="child")
            gamma_age = _priors.declare(
                pm.Normal("gamma_age", 0.0, assoc_prior_sigma, dims="outcome"),
                role="association",
                panel="predictor_slope",
                rationale=(
                    "Baseline (t1) age main effect on the growth rate "
                    "(gamma_age * age0); an adjusted, GA-confounded association, "
                    "not a cross-baseline coupling."
                ),
            )
            gamma_int = _priors.declare(
                pm.Normal("gamma_int", 0.0, assoc_prior_sigma, dims="outcome"),
                role="association",
                panel="predictor_slope",
                rationale=(
                    "Baseline age x ability interaction on the growth rate (the "
                    "#228 item-10 headline); an adjusted, GA-confounded "
                    "association, not a cross-baseline coupling."
                ),
            )
            slope_mean = (
                slope_mean
                + gamma_age[None, :] * age0[:, None]
                + gamma_int[None, :] * age0[:, None] * blocks[:, None]
            )
        if use_shared_factor:
            # Rank-1 shared child-level growth-tempo factor: positive loadings so
            # G is a common "faster growth on every measure" tempo (identification).
            G = _priors.declare(
                pm.Normal("G_tempo", 0.0, 1.0, dims="child"),
                role="nuisance",
                rationale=(
                    "Shared child-level growth-tempo factor scores (Normal(0, 1)); "
                    "a rank-1 latent 'faster growth on every measure' tempo whose "
                    "reported quantity is the per-measure loading, not the scores "
                    "themselves."
                ),
            )
            loading = _priors.declare(
                pm.HalfNormal("loading", sigma=loading_prior_sigma, dims="outcome"),
                role="association",
                rationale=(
                    "Positive indicator loading (HalfNormal(0.5)); maps each "
                    "standardised test to its unit-variance domain factor."
                ),
            )
            slope_mean = slope_mean + loading[None, :] * G[:, None]
        if sigma_slope_child is not None and z_slope is not None:
            slope_mean = slope_mean + sigma_slope_child * z_slope
        slope = pm.Deterministic(
            "slope", slope_mean, dims=("child", "outcome")
        )

        # Latent logit trajectory (linear in standardised age).
        theta = pm.Deterministic(
            "theta",
            intercept[:, None, :] + slope[:, None, :] * age[:, :, None],
            dims=("child", "wave", "outcome"),
        )

        # Masked Beta-Binomial observation (LRP55 flattened-mask idiom).
        mu = pm.math.sigmoid(theta)
        mu_clip = pm.math.clip(mu, EPSILON, 1 - EPSILON)
        kappa_child = kappa[group_data] if group_data is not None else kappa[None, :]
        alpha_bb = (mu_clip * kappa_child[:, None, :]).reshape((-1,))
        beta_bb = ((1 - mu_clip) * kappa_child[:, None, :]).reshape((-1,))
        idx_i, idx_t, idx_k = np.nonzero(mask)
        lin = np.ravel_multi_index((idx_i, idx_t, idx_k), (N, T, K))
        # Persist each flattened cell's outcome position so predictive checks can
        # select one measure. Without it a consumer must re-derive the mask order,
        # and a pooled overlay across measures with different maxima has no
        # interpretable predictive distribution (issue #208 / the joint family's
        # ``y_post_cell_outcome`` idiom).
        pm.Data("y_obs_cell_outcome", idx_k.astype("int64"), dims="y_obs_cell")
        pm.BetaBinomial(
            "y_obs",
            n=n_trials_vec[idx_k],
            alpha=alpha_bb[lin],
            beta=beta_bb[lin],
            observed=counts_int[idx_i, idx_t, idx_k],
        )

    return BuiltModel(model=model, prepared=panel, payload=EmptyPayload())
