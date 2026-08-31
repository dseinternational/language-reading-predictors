# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Latent change-score model construction.

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models.fitted_payloads import (
    EmptyPayload,
)
from language_reading_predictors.statistical_models.preprocessing import (
    WavePanel,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
)
from language_reading_predictors.statistical_models import priors as _priors

def build_lcsm_model(
    panel: WavePanel,
    *,
    reading_symbol: str = "W",
    couplings: dict[str, tuple[str, ...]] | None = None,
    lagged_change_couplings: dict[str, tuple[str, ...]] | None = None,
    arm_window_intercepts: bool = False,
    covariate_block: tuple[str, ...] = (),
    covariate_targets: tuple[str, ...] = (),
    coupling_prior_sigma: float = 0.3,
    self_prior_mu: float = -0.3,
    self_prior_sigma: float = 0.2,
    intercept_prior_sigma: float = 1.5,
    covariate_prior_sigma: float = 0.3,
    use_process_noise: bool = True,
    shared_process_noise: bool = False,
    sigma_proc_prior_sigma: float = 0.5,
    sigma_init_prior_sigma: float = 1.0,
    kappa_prior_sigma: float = 50.0,
) -> BuiltModel[EmptyPayload, WavePanel]:
    """Full coupled latent change-score model (LRP67 + the lagged suite) on the logit scale.

    A latent logit true-score ``x_m[i, t]`` is modelled for each measure ``m``
    (default ``W`` reading, ``L`` letter-sounds, ``E`` expressive vocabulary),
    child ``i`` and wave ``t``. The within-child trajectory follows a McArdle
    latent change-score recursion with **process noise**::

        x_m[i, 1] = mu1_m + sigma1_m * z1_m[i]                      (non-centred)
        x_m[i, t] = x_m[i, t-1] + Delta_m[i, t]
        Delta_m[i, t] = mean_Delta_m[i, t] + sigma_proc_m * zproc_m[i, t]

    ``couplings`` maps each **target** measure to the source measures whose
    prior-wave levels enter its change equation (prior *level* -> subsequent
    *change*, pooled over transitions). The default — ``None`` — reproduces
    LRP67 exactly: every non-``reading_symbol`` measure couples into the
    reading change::

        mean_Delta_W = a_W + b_W * x_W[t-1]
                     + sum_{c != W} g_c * x_c[t-1]
                     + d_age_W * age[t-1]

    With a single target the coupling parameters keep LRP67's ``g_{src}``
    names; with multiple targets they are ``g_{src}_{tgt}`` (the lagged
    reverse-coupling models LCSM-081/082, #250). Uncoupled measures get a
    self-proportional change plus age only.

    ``lagged_change_couplings`` maps each target measure to the source measures
    whose **previous transition's latent change** enters its change equation
    (prior *change* -> subsequent *change*; #229 specification 2, LCSM-091)::

        mean_Delta_tgt[t] += sum_c h_c * Delta_c[t-1]

    where ``Delta_c[t-1] = x_c[t-1] - x_c[t-2]`` is the realised latent change
    (structured mean plus process noise) of source ``c`` over the previous
    transition. The ``h`` terms enter for transitions ``k >= 1`` only — the
    first transition has no prior change — so with four waves they are
    identified from just **two** transitions, both post-crossover; any model
    using them must therefore also set ``arm_window_intercepts=True`` (the
    transition-2+ arm-confounding result, ``notes/202607141030-time-lagged-model-designs.md``).
    Naming mirrors the level couplings: ``h_{src}`` with a single lag target,
    ``h_{src}_{tgt}`` otherwise.

    ``arm_window_intercepts`` replaces the pooled per-measure change intercept
    with **arm x window cells** ``a_change[arm, trans, outcome]`` — required
    for any model pooling couplings across transitions, because the waitlist
    crossover makes the randomised arm a confounder of every transition-2+
    coupling (verified d-separation, ``notes/202607141030-time-lagged-model-designs.md``).
    The window-1 cell contrast is exposed as the deterministic
    ``itt_w1_contrast`` (immediate - waitlist, per outcome): a randomised
    latent randomised contrast on the change scale, reported as a consistency check
    against the available-case modified ITT suite. The intervention-dose covariate stays **omitted**:
    it is the locked DAG's ``IS`` collider, so conditioning on it would reopen
    the latent-``GA`` backdoor onto the couplings (ID-3); the arm x window
    cells derive from randomised ``IG`` + design timing instead.

    ``covariate_block`` names adjuster covariates on the panel — time-invariant
    (``panel.child_covariates``, e.g. ``hs``/``hs_missing``) or per-wave
    (``panel.wave_covariates``, e.g. ``erbto``/``deapp_c`` + indicators), the
    latter read at the transition's **prior** wave (the lagged DAG's
    parents-at-the-prior-wave rule). Each covariate gets one slope
    ``b_{name}`` **shared** across the ``covariate_targets`` change equations —
    the recommended parameter-sparing default at n~54.

    All change coefficients are **time-invariant** (pooled across the 3
    transitions) — a deliberate constraint at n~54. Everything is non-centred
    for sampling.

    The observed counts enter via a **masked** Beta-Binomial (the LRP55
    flattened-mask idiom): ``mu = sigmoid(x_m[i, t])`` is the logit mean and
    ``kappa_m`` the dispersion (measurement overdispersion, distinct from the
    dynamic ``sigma_proc``). Only the unmasked cells in ``panel.obs_mask`` are
    observed, so a child missing one score still contributes its other waves.

    The ``use_process_noise`` / ``shared_process_noise`` / ``*_prior_sigma``
    knobs implement the fallback ladder for sampling trouble (tighten priors;
    share one process-noise sd; drop process noise entirely).
    """
    OUT = tuple(panel.outcomes)
    if reading_symbol not in OUT:
        raise KeyError(
            f"reading_symbol {reading_symbol!r} not in panel.outcomes {OUT}"
        )
    K = len(OUT)
    N = panel.n_children
    T = panel.n_waves
    if T < 2:
        raise ValueError("LCSM needs at least two waves")
    jidx = {s: i for i, s in enumerate(OUT)}

    if couplings is None:
        couplings = {reading_symbol: tuple(s for s in OUT if s != reading_symbol)}
    couplings = {tgt: tuple(srcs) for tgt, srcs in couplings.items()}
    for tgt, srcs in couplings.items():
        unknown = [s for s in (tgt, *srcs) if s not in OUT]
        if unknown:
            raise KeyError(
                f"coupling symbols {unknown} not in panel.outcomes {OUT}"
            )
        if tgt in srcs:
            raise ValueError(
                f"target {tgt!r} cannot couple to itself (that is b_self)"
            )

    lagged_change_couplings = {
        tgt: tuple(srcs) for tgt, srcs in (lagged_change_couplings or {}).items()
    }
    for tgt, srcs in lagged_change_couplings.items():
        unknown = [s for s in (tgt, *srcs) if s not in OUT]
        if unknown:
            raise KeyError(
                f"lagged-change coupling symbols {unknown} not in panel.outcomes {OUT}"
            )
        if tgt in srcs:
            raise ValueError(
                f"target {tgt!r} cannot lag-couple to its own change "
                "(an AR on changes is not a #229 estimand)"
            )
    if lagged_change_couplings:
        if T < 3:
            raise ValueError(
                "lagged change-on-change couplings need at least three waves "
                "(a prior transition)"
            )
        if not arm_window_intercepts:
            raise ValueError(
                "lagged_change_couplings requires arm_window_intercepts=True: the "
                "h terms are identified entirely from post-crossover transitions, "
                "where the randomised arm confounds every pooled coupling "
                "(notes/202607141030-time-lagged-model-designs.md)"
            )

    if arm_window_intercepts and panel.group is None:
        raise ValueError(
            "arm_window_intercepts=True needs a panel with a group column"
        )
    covariate_block = tuple(covariate_block)
    covariate_targets = tuple(covariate_targets)
    if bool(covariate_block) != bool(covariate_targets):
        raise ValueError(
            "covariate_block and covariate_targets must be given together"
        )
    unknown_tgt = [s for s in covariate_targets if s not in OUT]
    if unknown_tgt:
        raise KeyError(
            f"covariate_targets {unknown_tgt} not in panel.outcomes {OUT}"
        )
    cov_arrays: dict[str, np.ndarray] = {}
    cov_is_wave: dict[str, bool] = {}
    for name in covariate_block:
        if name in panel.wave_covariates:
            cov_arrays[name] = panel.wave_covariates[name]
            cov_is_wave[name] = True
        elif name in panel.child_covariates:
            cov_arrays[name] = panel.child_covariates[name]
            cov_is_wave[name] = False
        else:
            raise KeyError(
                f"covariate {name!r} not on the panel; request it via "
                "load_wave_panel(wave_covariates=..., include_hearing=...)"
            )

    # Observed counts / mask / denominators stacked as (N, T, K) in OUT order.
    counts_int = np.stack(
        [np.nan_to_num(panel.counts[s], nan=0.0).astype(np.int64) for s in OUT],
        axis=2,
    )
    mask = np.stack([panel.obs_mask[s] for s in OUT], axis=2)  # (N, T, K) bool
    n_trials_vec = np.array([panel.n_trials[s] for s in OUT], dtype=int)  # (K,)
    # Observed wave-1 mean logit anchors the initial-latent prior mean. Guard the
    # all-NaN case loudly: an outcome with no observed wave-1 value would make
    # np.nanmean return NaN, which would silently poison mu1's prior mean and
    # surface only as an opaque sampler failure.
    missing_w1 = [s for s in OUT if not np.isfinite(panel.logit[s][:, 0]).any()]
    if missing_w1:
        raise ValueError(
            "LCSM wave-1 anchor is undefined (no observed first-wave value) for: "
            f"{', '.join(missing_w1)}. Drop the outcome or choose a panel with "
            "wave-1 observations."
        )
    w1_anchor = np.array(
        [np.nanmean(panel.logit[s][:, 0]) for s in OUT], dtype=float
    )

    coords = {
        "child": np.arange(N),
        "wave": panel.waves,
        "trans": panel.waves[1:],  # transitions into waves 2..T
        "outcome": list(OUT),
    }
    if arm_window_intercepts:
        coords["arm"] = ["immediate", "waitlist"]
        # Row index into the arm dimension per child (group 1 -> 0, group 2 -> 1).
        arm_idx = (np.asarray(panel.group) == 2).astype(int)

    from dse_research_utils.math.constants import EPSILON  # local import

    with pm.Model(coords=coords) as model:
        age = pm.Data("age_std", panel.age_std, dims=("child", "wave"))
        cov_data: dict[str, pt.TensorVariable] = {}
        for name in covariate_block:
            dims = ("child", "wave") if cov_is_wave[name] else ("child",)
            cov_data[name] = pm.Data(f"cov_{name}", cov_arrays[name], dims=dims)

        # Structural parameters (time-invariant, pooled over transitions).
        mu1 = _priors.declare(
                  pm.Normal("mu1", mu=w1_anchor, sigma=1.0, dims="outcome"),
                  role="nuisance",
                  rationale=(
                      "Initial latent level, its mean anchored on the observed wave-1 "
                      "mean logit per outcome. Empirical Bayes: the prior mean is "
                      "computed from the same observed outcomes that enter the "
                      "likelihood, so this prior is not independent of the data and the "
                      "reported prior-predictive distribution is partly data-informed."
                  ),
              )
        sigma1 = _priors.declare(
                     pm.HalfNormal("sigma1", sigma=sigma_init_prior_sigma, dims="outcome"),
                     role="nuisance",
                     rationale=(
                         "SD of the initial latent level (HalfNormal(1)); the between-child "
                         "spread at wave 1 that ``z1`` scales."
                     ),
                 )
        if arm_window_intercepts:
            a_change = _priors.declare(
                           pm.Normal(
                                           "a_change",
                                           mu=0.0,
                                           sigma=intercept_prior_sigma,
                                           dims=("arm", "trans", "outcome"),
                                       ),
                           role="nuisance",
                           rationale=(
                               "Per-measure change-score intercept (Normal(0, 1.5)); the mean "
                               "annual change each measure makes before any coupling term, "
                               "absorbed rather than reported."
                           ),
                       )
            # Window-1 randomised contrast on the latent change scale
            # (immediate - waitlist), the built-in ITT-suite consistency check.
            pm.Deterministic(
                "itt_w1_contrast",
                a_change[0, 0, :] - a_change[1, 0, :],
                dims="outcome",
            )
        else:
            a_change = _priors.declare(
                           pm.Normal(
                                           "a_change", mu=0.0, sigma=intercept_prior_sigma, dims="outcome"
                                       ),
                           role="nuisance",
                           rationale=(
                               "Per-measure change-score intercept (Normal(0, 1.5)); the mean "
                               "annual change each measure makes before any coupling term, "
                               "absorbed rather than reported."
                           ),
                       )
        # Self-feedback of the proportional change-score recursion: the level AR(1)
        # coefficient is phi = 1 + b_self, so the old ``mu=0`` centred phi on a unit
        # root (random walk) with ~50% prior mass on explosive phi > 1. A
        # proportional-change LCSM instead expects mean-reversion toward an asymptote
        # (negative self-feedback), so b_self is centred at -0.3 (phi ~ 0.7) with a
        # tighter sd: Normal(-0.3, 0.2) puts ~7% mass on explosive phi > 1 (vs ~50%),
        # taming the heavy-tailed geometry that drives divergences at n~54 (review
        # finding A3, 2026-07-13). Still weakly-informative — the data can pull b_self
        # back toward 0 given signal.
        b_self = _priors.declare(
                     pm.Normal(
                                 "b_self", mu=self_prior_mu, sigma=self_prior_sigma, dims="outcome"
                             ),
                     role="precision",
                     rationale=(
                         "Within-measure self-feedback of the change-score recursion (level "
                         "AR(1): phi = 1 + b_self; Normal(-0.3, 0.2)); centred at -0.3 (phi "
                         "~ 0.7) so trajectories mean-revert rather than random-walk — a "
                         "precision own-dynamics term (the LCSM analogue of the own-baseline "
                         "gamma_own), not one of the reported cross-skill couplings. This "
                         "descriptive LCSM has no randomised effect, so 'precision' here "
                         "means an own-dynamics term that supports, but is not, a reported "
                         "coupling."
                     ),
                 )
        d_age = _priors.declare(
                    pm.Normal("d_age", mu=0.0, sigma=covariate_prior_sigma, dims="outcome"),
                    role="precision",
                    rationale=(
                        "Linear age coupling on the change score (Normal(0, 0.3)); a "
                        "precision covariate that sharpens the reported couplings without "
                        "licensing a causal reading of its own."
                    ),
                )
        # Headline cross-couplings (prior source level -> target change). With a
        # single target the parameters keep LRP67's ``g_{src}`` names; with
        # multiple targets the target joins the name (``g_{src}_{tgt}``).
        single_target = len(couplings) == 1
        g_par: dict[tuple[str, str], pt.TensorVariable] = {}
        for tgt, srcs in couplings.items():
            for src in srcs:
                pname = f"g_{src}" if single_target else f"g_{src}_{tgt}"
                g_par[(src, tgt)] = _priors.declare(
                    pm.Normal(pname, mu=0.0, sigma=coupling_prior_sigma),
                    role="association",
                    panel="gamma_cross",
                    rationale=(
                        "Cross-measure coupling of the change-score recursion: the "
                        "effect of one measure's level on another's subsequent "
                        "change (Normal(0, 0.3)). An adjusted association, and the "
                        "quantity this family reports."
                    ),
                )
        # Lagged change-on-change couplings (prior source *change* -> target
        # change; #229 spec 2). Same naming rule and regularising scale as the
        # level couplings.
        single_lag_target = len(lagged_change_couplings) == 1
        h_par: dict[tuple[str, str], pt.TensorVariable] = {}
        for tgt, srcs in lagged_change_couplings.items():
            for src in srcs:
                pname = f"h_{src}" if single_lag_target else f"h_{src}_{tgt}"
                h_par[(src, tgt)] = pm.Normal(
                    pname, mu=0.0, sigma=coupling_prior_sigma
                )
        # Adjuster-covariate slopes, shared across the covariate_targets
        # equations (the parameter-sparing default at n~54).
        b_cov = {
            name: pm.Normal(f"b_{name}", mu=0.0, sigma=covariate_prior_sigma)
            for name in covariate_block
        }
        kappa = _priors.declare(
                    pm.HalfNormal("kappa", sigma=kappa_prior_sigma, dims="outcome"),
                    role="nuisance",
                    panel="kappa",
                    rationale=(
                        "Beta-binomial concentration kappa ~ HalfNormal(50)."
                    ),
                )

        sigma_proc: dict[str, pt.TensorVariable] = {}
        zproc: dict[str, pt.TensorVariable] = {}
        if use_process_noise:
            if shared_process_noise:
                sp = _priors.declare(
                         pm.HalfNormal("sigma_proc", sigma=sigma_proc_prior_sigma),
                         role="nuisance",
                         rationale=(
                             "Process-noise SD of the latent change score (HalfNormal(0.5)); the "
                             "wave-to-wave innovation the ``zproc`` offsets scale."
                         ),
                     )
                sigma_proc = {s: sp for s in OUT}
            else:
                spv = _priors.declare(
                          pm.HalfNormal(
                                              "sigma_proc", sigma=sigma_proc_prior_sigma, dims="outcome"
                                          ),
                          role="nuisance",
                          rationale=(
                              "Process-noise SD of the latent change score (HalfNormal(0.5)); the "
                              "wave-to-wave innovation the ``zproc`` offsets scale."
                          ),
                      )
                sigma_proc = {s: spv[jidx[s]] for s in OUT}
            zproc = {
                s: _priors.declare(
                       pm.Normal(f"zproc_{s}", 0.0, 1.0, dims=("child", "trans")),
                       role="nuisance",
                       rationale=(
                           "Non-centred standard-normal per-child, per-transition offsets "
                           "(Normal(0, 1)); scaled by sigma_proc to form the latent process "
                           "noise."
                       ),
                   )
                for s in OUT
            }

        # Cross-process covariance is deliberately omitted: the initial statuses
        # (z1_*) and the per-transition process noises (zproc_*) are modelled as
        # independent across the W/L/E processes. An LKJ-correlated initial-status
        # block (as the growth model uses) is not reliably estimable at n ~ 54
        # here, so this is a small-n fallback — it may attenuate the coupling
        # coefficients g_L / g_E, which is accepted and flagged (issue #273).
        # Initial latent (wave index 0), non-centred.
        x: dict[str, list[pt.TensorVariable]] = {}
        for s in OUT:
            z1 = _priors.declare(
                     pm.Normal(f"z1_{s}", 0.0, 1.0, dims="child"),
                     role="nuisance",
                     rationale=(
                         "Non-centred standard-normal per-child offsets (Normal(0, 1)); "
                         "scaled by sigma1 to form each child's initial latent level."
                     ),
                 )
            x[s] = [
                pm.Deterministic(
                    f"x1_{s}", mu1[jidx[s]] + sigma1[jidx[s]] * z1, dims="child"
                )
            ]

        # Latent change-score recursion over transitions (t = 1 .. T-1).
        # ``delta_hist[s][k]`` is measure ``s``'s realised latent change on
        # transition ``k`` (structured mean plus process noise) — the lagged
        # change-on-change regressor for transition ``k + 1``.
        delta_hist: dict[str, list[pt.TensorVariable]] = {s: [] for s in OUT}
        for k in range(T - 1):
            t = k + 1
            prev = {s: x[s][t - 1] for s in OUT}
            for s in OUT:
                if arm_window_intercepts:
                    m = a_change[arm_idx, k, jidx[s]] + b_self[jidx[s]] * prev[s]
                else:
                    m = a_change[jidx[s]] + b_self[jidx[s]] * prev[s]
                m = m + d_age[jidx[s]] * age[:, t - 1]
                for src in couplings.get(s, ()):
                    m = m + g_par[(src, s)] * prev[src]
                # Prior-transition latent change of the lag sources; no term on
                # the first transition (k = 0), which has no prior change.
                if k >= 1:
                    for src in lagged_change_couplings.get(s, ()):
                        m = m + h_par[(src, s)] * delta_hist[src][k - 1]
                if s in covariate_targets:
                    for name in covariate_block:
                        v = cov_data[name]
                        # Per-wave states are read at the prior wave (the lagged
                        # DAG's parents-at-the-prior-wave rule).
                        m = m + b_cov[name] * (v[:, t - 1] if cov_is_wave[name] else v)
                delta = m
                if use_process_noise:
                    delta = delta + sigma_proc[s] * zproc[s][:, k]
                delta_hist[s].append(delta)
                x[s].append(prev[s] + delta)

        # Stack latent to (child, wave, outcome) for reporting + likelihood.
        X = pt.stack([pt.stack(x[s], axis=1) for s in OUT], axis=2)
        X = pm.Deterministic("x_latent", X, dims=("child", "wave", "outcome"))

        # Masked Beta-Binomial observation (LRP55 flattened-mask idiom).
        mu = pm.math.sigmoid(X)
        mu_clip = pm.math.clip(mu, EPSILON, 1 - EPSILON)
        alpha_bb = (mu_clip * kappa[None, None, :]).reshape((-1,))
        beta_bb = ((1 - mu_clip) * kappa[None, None, :]).reshape((-1,))
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
