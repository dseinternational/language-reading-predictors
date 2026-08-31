# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Joint-mechanism model construction (levels and transition designs).

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
    JointMechanismPayload,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _bivariate_lkj_residual,
)

def _add_decoding_contrast_deterministics(
    *,
    beta_mech: pt.TensorVariable,
    outcome_symbols: tuple[str, ...],
    contrast: tuple[str, str],
    corr: pt.TensorVariable | None,
    sigmas: pt.TensorVariable | None,
    conditional_slope: bool,
) -> None:
    """Register the two identified within-model contrasts on ``beta_mech``.

    ``delta_ls_decoding = beta_mech[contrast[0]] - beta_mech[contrast[1]]`` (default
    ``N - W``) is the decoding-specificity difference. Because both slopes come from
    one posterior, its interval carries the true cross-outcome covariance rather than
    the paired-draws convolution that ``mech-096`` / ``mech-101`` can only bound.

    ``rho_outcome`` — the off-diagonal of the dependence block — is registered
    whenever one exists. It is the quantity that makes the block auditable: a
    correlation whose interval sits on zero says the joint fit is buying nothing over
    two separate fits, and the report must be able to show that either way.

    With ``conditional_slope`` the **conditional** slope is registered too. Writing
    the focal outcome ``f`` (default
    ``W``) and the held-fixed outcome ``h`` (default ``N``) as latent logits with
    residual covariance Sigma,

        E[y_f | z, y_h] = z (beta_f - (Sigma_fh / Sigma_hh) beta_h) + (Sigma_fh / Sigma_hh) y_h

    so the exposure coefficient *holding the other outcome fixed* is

        beta_mech_focal_given_held = beta_f - rho (sigma_f / sigma_h) beta_h

    and ``share_retained = beta_mech_focal_given_held / beta_f`` is the identified
    counterpart of the review's product-of-marginals ratio (the ``ca-010`` /
    ``ca-011`` paired-draws quantity).

    ``share_retained`` is a **conditional-to-marginal slope ratio, not a bounded
    pathway share**. It is unbounded: negative under suppression or a sign reversal,
    above one under amplification, and unstable wherever the denominator ``beta_f``
    approaches zero. It does not identify a mediated fraction — this is an
    observational model with no mediation identification — so the pipeline governs it
    with an explicit denominator-stability rule, publishes the probability mass below
    zero, inside [0, 1] and above one, and never publishes its mean (2026-08-23
    follow-up review, finding 5). The retained variable name is the machine key; the
    scientific label is the conditional-to-marginal slope ratio.

    Note the estimands differ subtly and deliberately: ``ca-011`` conditions on the
    *observed* nonword count (measurement error and all), whereas this conditions on
    the *latent* nonword logit. Partialling the latent skill is the cleaner reading of
    "holding decoding fixed". Classical additive measurement-error intuition suggests
    it should retain *less*, but that ordering is **not** guaranteed across two
    nonlinear models with different likelihoods, different missing-data handling,
    floor compression and possibly non-classical measurement error — so the two must
    not be presented as bracketing the answer.
    """
    hi = outcome_symbols.index(contrast[0])
    lo = outcome_symbols.index(contrast[1])
    pm.Deterministic("delta_ls_decoding", beta_mech[hi] - beta_mech[lo])
    if corr is None or sigmas is None:
        return
    # Focal = the outcome whose slope is partialled (W); held fixed = the other (N).
    focal, held = lo, hi
    rho = pm.Deterministic("rho_outcome", corr[focal, held])
    if not conditional_slope:
        return
    ratio = rho * sigmas[focal] / sigmas[held]
    pm.Deterministic("beta_held_on_focal", ratio)
    conditional = beta_mech[focal] - ratio * beta_mech[held]
    pm.Deterministic("beta_mech_focal_given_held", conditional)
    pm.Deterministic("share_retained", conditional / beta_mech[focal])


def build_joint_mechanism_model(
    prepared: PreparedData,
    *,
    design: str = "levels",
    mechanism_symbol: str = "L",
    outcome_symbols: Iterable[str] = ("W", "N"),
    contrast: tuple[str, str] = ("N", "W"),
    adjust_for: Iterable[str] = (),
    confounder_symbols: Iterable[str] = ("G", "A"),
    include_group: bool = True,
    predictor_slope_sigma: float = 0.3,
    residual_lkj_eta: float = 2.0,
    residual_sd_sigma: float = 1.0,
    child_lkj_eta: float = 2.0,
    sigma_child_prior_sigma: float = 0.5,
) -> BuiltModel[JointMechanismPayload]:
    """Bivariate mechanism: one standardised exposure -> two outcomes fitted jointly
    with an **LKJ cross-outcome dependence block**, in either of two designs.

    Frank's #421 Tier-3 (1) build. Two quantities that the suite currently reports as
    *product-of-marginals sensitivities* — the decoding-specificity contrast
    ``Delta = beta(LS->N) - beta(LS->W)`` (``notes/202607172358``) and the
    share-retained ratio for "does the letter-sound association survive holding
    decoding fixed" (``notes/202607241000`` Q2) — are assembled by pairing draws from
    *separate* fits under a working-independence assumption. The fits share children,
    so the true joint posterior has a cross-outcome covariance the pairing ignores.
    Fitting ``W`` and ``N`` in one model with an explicit dependence block makes both
    quantities **within-model deterministics**.

    The dependence block is what does the work. A single scalar child intercept with a
    fixed loading of 1 on both logits (this factory's first cut, #427 review) permits
    only the *same* stable shift in both outcomes: conditional on it the two likelihood
    legs still factorise, so it yields no outcome-specific child effect, no residual
    correlation, and no conditional slope. The LKJ block gives all three.

    Designs
    -------
    ``design="levels"`` (``jm-001``) — the per-wave levels/concurrent design #421
    specifies. Expects a **single-wave** subset of the ``phase_mode="levels"`` frame
    (the pipeline slices ``prepared.phase == wave`` and calls once per wave), so there
    is one row per child. Each outcome's *level* is regressed on the standardised
    same-wave letter-sound logit, age, a group nuisance and the trait covariates
    (including the ``hs_missing`` indicator — the missing-indicator policy pairs it
    with the filled ``hs``), matched term-for-term to ``ca-010`` / ``ca-011`` —
    covariate set, Normal(0, 0.3) slope prior and the wide Normal(0, 1) group
    nuisance alike — so the identified share-retained replaces their paired-draws
    ratio like for like. The likelihood is **Binomial**,
    not Beta-Binomial: the bivariate residual already models extra-binomial variance,
    and carrying ``kappa`` as well would leave two overdispersion mechanisms competing
    on the same row — the route by which the ITT joint's LKJ block went
    prior-dominated in 2026-04. Here the residual *is* the overdispersion, and its
    correlation is the estimand.

    ``design="transition"`` (``jm-002``) — the phase-stacked ANCOVA companion, matched
    term-for-term to ``mech-096`` / ``mech-101`` (own baseline, phase intercepts,
    the same {G, A, HS, IS, SP} adjustment set) so the Tier-1 Delta is re-reported on
    the *same parameterisation* it was originally computed on. Here the dependence
    block is a **bivariate child random intercept** (child-level covariance, three
    rows per child) and the Beta-Binomial ``kappa`` is retained for within-child
    overdispersion — the two are at different levels, so both are identified.

    Every slope is an **adjusted association**: latent general ability is unobserved
    and neither dependence block stands in for it. The contrast is a Campbell-Fiske
    convergent/discriminant argument, never identification of a causal decoding
    effect — and that argument is itself conditional on cross-instrument measurement
    invariance the model does not impose: if both outcomes load on one general
    ability with unequal loadings, the latent-scale slopes stay proportional to those
    loadings and their difference is non-zero with no causal letter-sound route at all
    (2026-08-23 follow-up review, finding 3). Each outcome keeps its own denominator
    (79 items for ``W``, 6 for ``N``); the flattened-cell likelihood never pools them.
    """
    from dse_research_utils.math.constants import EPSILON
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    if design not in {"levels", "transition"}:
        raise ValueError(
            f"joint mechanism design must be 'levels' or 'transition'; got {design!r}"
        )
    outcome_symbols = tuple(outcome_symbols)
    confounder_symbols = tuple(confounder_symbols)
    adjust_for = tuple(adjust_for)
    K = len(outcome_symbols)
    if K != 2:
        raise ValueError("joint mechanism model expects exactly two outcomes")
    contrast = tuple(contrast)
    for s in contrast:
        if s not in outcome_symbols:
            raise ValueError(f"contrast outcome {s!r} not in outcome_symbols")
    # A duplicate or short contrast — ``("N", "N")`` — silently produced a
    # ``delta_ls_decoding`` identically zero and a share-retained partialling the
    # focal outcome against itself. The registered typed run plans already reject
    # it, but this is a public factory boundary and must enforce the same
    # invariant (2026-08-23 follow-up review, robustness gap 6).
    if len(contrast) != 2 or set(contrast) != set(outcome_symbols):
        raise ValueError(
            "joint mechanism contrast must name each outcome exactly once; got "
            f"{contrast!r} for outcomes {outcome_symbols!r}"
        )
    for s in outcome_symbols:
        if s not in prepared.post_counts:
            raise KeyError(f"Outcome {s!r} missing a post score in prepared data")
    if mechanism_symbol not in prepared.post_counts:
        raise KeyError(f"Mechanism {mechanism_symbol!r} missing from prepared data")

    expected_mode = "levels" if design == "levels" else "all"
    if prepared.phase_mode != expected_mode:
        raise ValueError(
            f"joint mechanism design={design!r} requires phase_mode="
            f"{expected_mode!r}; got {prepared.phase_mode!r}"
        )
    if design == "transition":
        for s in outcome_symbols:
            if s not in prepared.pre_logit:
                raise KeyError(f"Outcome baseline {s!r}_pre missing from prepared data")

    # Shared exposure: standardised letter-sound post logit, identical for both legs,
    # so the two slopes sit on one commensurate logit-per-SD-of-exposure scale. Rows
    # with a missing exposure are DROPPED rather than mean-imputed — imputing the
    # focal exposure shrinks its realised variance and biases both slopes toward zero,
    # which would corrupt the very contrast the model exists to estimate.
    exposure_ok = ~np.isnan(prepared.post_counts[mechanism_symbol])
    # A row observing neither outcome contributes no likelihood cell.
    any_outcome = np.zeros(prepared.n_obs, dtype=bool)
    for s in outcome_symbols:
        any_outcome |= ~np.isnan(prepared.post_counts[s])
    keep = exposure_ok & any_outcome
    if not keep.all():
        prepared = _subset(prepared, keep)
    if prepared.n_obs == 0:
        raise ValueError(
            "joint mechanism model has no usable rows: no observation has both the "
            f"exposure {mechanism_symbol!r} and at least one of {outcome_symbols}."
        )
    # Re-check adjuster variance on the FINAL fitted rows (2026-08-23 joint audit,
    # lower-priority robustness correction). The loader screens for constant columns
    # on its prepared frame, but the wave subset above (levels design) and this
    # exposure/outcome mask both change the design afterwards, and no second check
    # was applied. A constant adjuster is collinear with the intercept, so its
    # coefficient is unidentified and its prior is what the posterior reports. Every
    # registered jm-001 / jm-002 adjuster varies on the current data, so this closes
    # a latent defect; it fails loudly rather than dropping the column, because
    # silently dropping it would make the recorded ``effective_adjustment`` wrong.
    for name in adjust_for:
        values = np.asarray(prepared.covariates[name], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and np.ptp(finite) == 0.0:
            raise ValueError(
                f"joint mechanism adjuster {name!r} is constant on the "
                f"{prepared.n_obs} fitted rows (value {float(finite[0]):g}), so its "
                "coefficient is not identified. Drop it from adjust_for for this "
                "design, or widen the analysis rows."
            )

    if design == "levels" and prepared.n_obs != prepared.n_children:
        raise ValueError(
            "joint mechanism design='levels' expects one row per child (a single "
            f"wave); got {prepared.n_obs} rows over {prepared.n_children} children "
            "— pass a single-wave subset."
        )

    N_obs = prepared.n_obs
    z_L = standardise(
        logit_safe(
            prepared.post_counts[mechanism_symbol],
            prepared.n_trials[mechanism_symbol],
        )
    )[0]

    # Per-outcome observation mask + flattened observed cells (robust to
    # outcome-specific post missingness), mirroring build_joint_model.
    mask = np.stack(
        [~np.isnan(prepared.post_counts[s]) for s in outcome_symbols], axis=1
    )
    post_counts_int = np.stack(
        [
            np.nan_to_num(prepared.post_counts[s], nan=0.0).astype(np.int64)
            for s in outcome_symbols
        ],
        axis=1,
    )
    n_trials_vec = np.array([prepared.n_trials[s] for s in outcome_symbols], dtype=int)
    idx_row, idx_col = np.nonzero(mask)

    coords = {
        "obs_id": np.arange(N_obs),
        "outcome": list(outcome_symbols),
        # Second outcome axis for outcome x outcome quantities: PyMC requires
        # distinct dim names per axis, so the correlation matrix cannot reuse
        # "outcome" for both.
        "outcome2": list(outcome_symbols),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
        "cell": np.arange(idx_row.size),
    }

    with pm.Model(coords=coords) as model:
        G_d = pm.Data("G", prepared.G.astype(float), dims="obs_id")
        A_std_d = pm.Data(
            "A_std", np.nan_to_num(np.asarray(prepared.A_std, dtype=float)),
            dims="obs_id",
        )
        z_L_d = pm.Data("z_mech_logit", z_L, dims="obs_id")
        child_idx_d = pm.Data(
            "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
        )
        adjust_data = {
            c: pm.Data(
                f"{c}_adj",
                np.nan_to_num(np.asarray(prepared.covariates[c], dtype=float)),
                dims="obs_id",
            )
            for c in adjust_for
            if c in prepared.covariates
        }

        alpha = _priors.alpha_prior().to_pymc("alpha", dims="outcome")
        if design == "levels":
            # Matched to ca-010 / ca-011: the same regularising association prior on
            # the letter-sound slope, so the identified share-retained is comparable
            # with the paired-draws ratio it replaces.
            beta_mech = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                "beta_mech", dims="outcome"
            )
        else:
            # Matched to mech-096 / mech-101, whose Delta this design re-reports.
            beta_mech = _priors.beta_mech_prior().to_pymc("beta_mech", dims="outcome")

        eta = alpha[None, :] + beta_mech[None, :] * pt.shape_padright(z_L_d)

        if design == "transition":
            pre_logit = np.stack(
                [prepared.pre_logit[s] for s in outcome_symbols], axis=1
            )
            pre_logit_d = pm.Data(
                "pre_logit", pre_logit, dims=("obs_id", "outcome")
            )
            gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own", dims="outcome")
            phase_d = pm.Data(
                "phase_idx", prepared.phase.astype(np.int64), dims="obs_id"
            )
            alpha_phase = _priors.declare(
                              pm.Normal(
                                              "alpha_phase", mu=0.0, sigma=0.5, dims=("phase", "outcome")
                                          ),
                              role="nuisance",
                              rationale=(
                                  "Per-phase intercept offset alpha_phase ~ Normal(0, 0.5)."
                              ),
                          )
            eta = eta + gamma_own[None, :] * pre_logit_d + alpha_phase[phase_d]

        if include_group or "G" in confounder_symbols:
            # Group is a NON-INTERPRETABLE nuisance in the levels design (it only
            # absorbs arm composition at the wave), on the same deliberately wide
            # Normal(0, 1) build_concurrent_model uses — matching ca-010 / ca-011
            # term-for-term is the design's warrant, and the first cut's tau_prior
            # halved the width (2026-08-21 joint-mechanism review, finding 1). In
            # the transition design it is the same tau-scaled arm term the
            # mechanism family carries. Named per design so no consumer reads one
            # as the other.
            if design == "levels":
                beta_G = _priors.declare(
                             pm.Normal(
                                                 "beta_group_nuisance", mu=0.0, sigma=1.0, dims="outcome"
                                             ),
                             role="nuisance",
                             rationale=(
                                 "Non-interpretable group-composition nuisance dummy (Normal(0, 1)) "
                                 "held outside the horseshoe / adjustment set to absorb cohort "
                                 "composition (reference = largest group); never a ranked predictor "
                                 "slope or a group-effect estimate."
                             ),
                         )
            else:
                beta_G = _priors.tau_prior().to_pymc(
                    "beta_G",
                    dims="outcome",
                    role="association",
                    rationale=(
                        "Randomised arm entered as an adjusted-association "
                        "covariate on the treatment prior tau ~ Normal(0, 0.5); "
                        "this design's deliverable is the conditional slope and "
                        "its ratio, not an arm effect."
                    ),
                )
            eta = eta + beta_G[None, :] * pt.shape_padright(G_d)

        if "A" in confounder_symbols:
            gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A", dims="outcome")
            eta = eta + gamma_A[None, :] * pt.shape_padright(A_std_d)

        for c, cov_d in adjust_data.items():
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}", dims="outcome")
            eta = eta + gamma_c[None, :] * pt.shape_padright(cov_d)

        # --- Cross-outcome dependence block ---------------------------------------
        if design == "levels":
            # One row per child, so the residual IS the child effect at this wave and
            # its off-diagonal is the *within-wave* residual correlation #421 asks for.
            u, corr, sigmas = _bivariate_lkj_residual(
                "u_resid",
                n_outcomes=K,
                row_dim="obs_id",
                lkj_eta=residual_lkj_eta,
                sd_sigma=residual_sd_sigma,
            )
        else:
            # Three rows per child: the covariance lives at the CHILD level and the
            # Beta-Binomial kappa keeps the within-child overdispersion.
            u_child, corr, sigmas = _bivariate_lkj_residual(
                "u_child",
                n_outcomes=K,
                row_dim="child",
                lkj_eta=child_lkj_eta,
                sd_sigma=sigma_child_prior_sigma,
            )
            u = u_child[child_idx_d]
        eta = eta + u

        eta = pm.Deterministic("eta", eta, dims=("obs_id", "outcome"))

        _add_decoding_contrast_deterministics(
            beta_mech=beta_mech,
            outcome_symbols=outcome_symbols,
            contrast=contrast,
            corr=corr,
            sigmas=sigmas,
            # The conditional slope is a *within-wave, same-row* partialling, so it is
            # only meaningful where the covariance block sits on the observation row.
            # In the transition design the block is a between-child intercept
            # covariance, which answers a different question ("children who run high
            # on W also run high on N"), not "holding this child's decoding fixed at
            # this wave" — so that design reports rho and Delta, and no share-retained.
            conditional_slope=design == "levels",
        )

        # Record the flattened cell mapping so the diagnostics can select one outcome
        # for predictive checks (incompatible denominators are never pooled).
        pm.Data("y_post_cell_row", idx_row.astype("int64"), dims="cell")
        pm.Data("y_post_cell_outcome", idx_col.astype("int64"), dims="cell")
        if design == "transition":
            # Cell -> child map for genuine leave-one-child-out PSIS-LOO. Without
            # it the shared aggregation falls back to ``y_post_cell_row``, whose
            # rows here are child-by-transition rows — a defensible unit, but not
            # the ``loo_unit="child"`` the run plan and recipe declare (2026-08-21
            # joint-mechanism review, finding 3).
            pm.Data(
                "loo_child_idx",
                prepared.child_idx.astype("int64")[idx_row],
                dims="cell",
            )

        mu = pm.math.clip(pm.math.sigmoid(eta), EPSILON, 1 - EPSILON)
        if design == "levels":
            pm.Binomial(
                "y_post",
                n=n_trials_vec[idx_col],
                p=mu[idx_row, idx_col],
                observed=post_counts_int[idx_row, idx_col],
                dims="cell",
            )
        else:
            kappa = _priors.kappa_prior().to_pymc("kappa", dims="outcome")
            alpha_bb = mu * kappa[None, :]
            beta_bb = (1 - mu) * kappa[None, :]
            pm.BetaBinomial(
                "y_post",
                n=n_trials_vec[idx_col],
                alpha=alpha_bb[idx_row, idx_col],
                beta=beta_bb[idx_row, idx_col],
                observed=post_counts_int[idx_row, idx_col],
                dims="cell",
            )

    return BuiltModel(
        model=model,
        prepared=prepared,
        payload=JointMechanismPayload(
            design=design,
            joint_dependence=(
                "lkj_residual_within_wave"
                if design == "levels"
                else "lkj_child_intercept"
            ),
            likelihood="binomial" if design == "levels" else "beta_binomial",
            loo_unit="child",
            outcomes=outcome_symbols,
            mechanism_symbol=mechanism_symbol,
            contrast=contrast,
            adjust_for=tuple(adjust_data),
        ),
    )
