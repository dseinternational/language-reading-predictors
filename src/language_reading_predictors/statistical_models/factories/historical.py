# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Historical-cohort growth and joint-growth model construction.

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


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    EmptyPayload,
)
from language_reading_predictors.statistical_models.preprocessing import (
    LongitudinalPanel,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
)
from language_reading_predictors.statistical_models.invariants import (
    require_value,
)

def _map_panel_rows(values, index: dict, *, what: str) -> np.ndarray:
    """Map tidy-row keys to dense model indices, refusing keys the panel lacks.

    The previous ``Series.map(index).to_numpy(dtype=int)`` turned an unknown key
    into NaN and then cast it to int — undefined behaviour that is silently 0 on
    arm64 and INT64_MIN on x86-64, so a row naming a subject outside
    ``panel.subject_ids`` sampled from subject 0 on one platform and from
    out-of-bounds memory on the other (a test fixture did exactly that, and CI
    failed with a Beta-Binomial domain error while the same test passed on macOS,
    2026-08-22). Failing loudly is the only portable behaviour.
    """
    mapped = np.array([index.get(value, -1) for value in values], dtype=np.int64)
    if mapped.size and (mapped < 0).any():
        unknown = sorted({str(value) for value in values if value not in index})
        raise ValueError(
            f"panel rows name {what}(s) the panel does not index: {unknown}; "
            f"the tidy frame and the panel's {what} list must agree"
        )
    return mapped


def build_historical_growth_model(
    panel: LongitudinalPanel,
    *,
    measure: str = "basread",
    eta_prior_sigma: float = 1.5,
    # HalfNormal(1.0) since #383: the 0.5 scale was in prior-data conflict with
    # the Down-syndrome verbal/reading heterogeneity (posteriors 1.25-1.39, at
    # the HalfNormal(0.5) 99th percentile). Every registered consumer sets the
    # value explicitly in its spec; this default matches the reviewed choice.
    sigma_subject_prior_sigma: float = 1.0,
    # 1/sqrt(kappa) ~ HalfNormal(0.25) since the 2026-08-21 review (finding 8):
    # a HalfNormal on kappa itself cannot reach the near-Binomial limit, which is
    # the answer the data prefer for most of these measures. See
    # ``priors.inv_sqrt_kappa_prior`` for the calibration.
    dispersion_prior_sigma: float = 0.25,
) -> BuiltModel[EmptyPayload]:
    """Descriptive group-by-wave growth model for a historical cohort.

    Beta-Binomial on a bounded count with a population level per **supported
    (group, wave) cell** and a non-centred, group-centred per-subject random
    intercept, with the random-effect scales indexed by group (#338)::

        score_it ~ BetaBinomial(n, p_it, kappa[group_i])
        logit(p_it) = eta_cell[cell(group_i, wave_t)] + subject_offset_i
        subject_offset_i = (z_i - mean(z, group_i)) * sigma_subject[group_i]
        kappa[g] = 1 / inv_sqrt_kappa[g]^2,  inv_sqrt_kappa ~ HalfNormal(0.25)

    ``eta_cell`` ranges over the cells that actually carry data, so a ragged
    follow-up window (the Byrne wave-5 Down-syndrome-only extension, #338) adds
    no prior-only parameters. Group-indexed ``sigma_subject`` / ``kappa`` let
    between-child heterogeneity and overdispersion differ by cohort group
    (follow-up-plan decision 7). This is **descriptive natural-history**
    evidence, not an intervention-effect model: ``group`` carries no treatment
    semantics, there is no baseline-as-precision term and no adjustment set.
    Deterministics expose the per-cell expected item score and within-group
    interval growth over the **common window** (the waves every group supports);
    the ragged extension intervals are summarised downstream
    (:mod:`historical`) on common-subject cells. Ported from the standalone
    ``lrp-rlm-hg-001`` script (#163) onto the shared pipeline (#165).
    """
    df = panel.long
    dataset = panel.dataset
    subj, wave_c, grp = dataset.subject_col, dataset.wave_col, dataset.group_col
    if measure not in panel.n_trials:
        raise KeyError(f"measure {measure!r} not in panel (have {panel.measures}).")
    n_trials = int(panel.n_trials[measure])

    group_codes = panel.group_codes
    group_labels = panel.group_labels
    subject_ids = panel.subject_ids
    group_index = {code: i for i, code in enumerate(group_codes)}
    subject_index = {s: i for i, s in enumerate(subject_ids)}

    cells = panel.cells(measure)
    cell_index = {cell: i for i, cell in enumerate(cells)}
    cell_labels = [
        f"{group_labels[group_index[g]]} | wave {w}" for g, w in cells
    ]
    # Waves supported in *every* group - the window where between-group
    # quantities are defined (the Byrne common window is w1-w4; wave 5 is a
    # Down-syndrome-only extension).
    common_waves = [
        w
        for w in sorted({w for _g, w in cells})
        if all((g, w) in cell_index for g in group_codes)
    ]

    group_idx = _map_panel_rows(df[grp].tolist(), group_index, what="group code")
    obs_cell_idx = np.array(
        [
            cell_index[(int(g), int(w))]
            for g, w in zip(df[grp], df[wave_c], strict=True)
        ],
        dtype=int,
    )
    subject_idx = _map_panel_rows(df[subj].tolist(), subject_index, what="subject")
    observed = df[measure].to_numpy(dtype=int)
    subject_group = (
        df.drop_duplicates(subj)
        .set_index(subj)
        .loc[subject_ids, grp]
        .map(group_index)
        .to_numpy(dtype=int)
    )

    coords = {
        "group": group_labels,
        "cell": cell_labels,
        "subject": [str(s) for s in subject_ids],
        "obs": np.arange(len(df)),
    }

    def _cell_pos(wave: int) -> list[int]:
        """Positions of ``(group, wave)`` in the cell vector, in group order."""
        return [cell_index[(g, wave)] for g in group_codes]

    with pm.Model(coords=coords) as model:
        eta_cell = _priors.declare(
                       pm.Normal(
                                   "eta_cell", mu=0.0, sigma=eta_prior_sigma, dims="cell"
                               ),
                       role="nuisance",
                       rationale=(
                           "Group-by-wave population level per cell/measure on the logit scale "
                           "(Normal(0, 1.5)); the fitted cells (mean_items) and growth "
                           "intervals are deterministics of it — descriptive, not a treatment "
                           "effect."
                       ),
                   )
        sigma_subject = _priors.declare(
                            pm.HalfNormal(
                                        "sigma_subject", sigma=sigma_subject_prior_sigma, dims="group"
                                    ),
                            role="nuisance",
                            rationale=(
                                "Group-indexed between-subject random-intercept SD (HalfNormal(1)); "
                                "between-child heterogeneity that differs by cohort group."
                            ),
                        )
        z_subject = _priors.declare(
                        pm.Normal("z_subject", mu=0.0, sigma=1.0, dims="subject"),
                        role="nuisance",
                        rationale=(
                            "Non-centred standard-normal per-subject offsets (Normal(0, 1)); "
                            "group-centred and scaled by sigma_subject to form the subject "
                            "random effects."
                        ),
                    )
        # Group-centre the subject offsets for identifiability against
        # ``eta_cell`` (the group-by-wave level absorbs the group mean).
        z_group_mean = pm.math.stack(
            [z_subject[subject_group == g].mean() for g in range(len(group_codes))]
        )
        subject_offset = pm.Deterministic(
            "subject_offset",
            (z_subject - z_group_mean[subject_group])
            * sigma_subject[subject_group],
            dims="subject",
        )
        # Sample the DISPERSION, publish the concentration. ``u = 1/sqrt(kappa)``
        # puts the no-extra-Binomial-dispersion limit at u = 0, where a HalfNormal
        # has its mode, instead of in a tail the old HalfNormal(50) on kappa gave
        # probability 0.001 (2026-08-21 review, finding 8). The 1e-6 floor keeps
        # kappa finite and the gradient smooth as u -> 0; at the fitted range
        # (kappa 28-121, u 0.09-0.19) it shifts kappa by under 0.01%.
        inv_sqrt_kappa = _priors.inv_sqrt_kappa_prior(
            sigma=dispersion_prior_sigma
        ).to_pymc("inv_sqrt_kappa", dims="group")
        kappa = pm.Deterministic(
            "kappa", 1.0 / (inv_sqrt_kappa**2 + 1e-6), dims="group"
        )

        eta_obs = eta_cell[obs_cell_idx] + subject_offset[subject_idx]
        p_obs = pm.math.sigmoid(eta_obs)
        kappa_obs = kappa[group_idx]
        pm.BetaBinomial(
            "score",
            n=n_trials,
            alpha=p_obs * kappa_obs,
            beta=(1.0 - p_obs) * kappa_obs,
            observed=observed,
            dims="obs",
        )
        pm.Deterministic("fitted_mean_items_obs", n_trials * p_obs, dims="obs")

        mean_items = pm.Deterministic(
            "mean_items",
            n_trials * pm.math.sigmoid(eta_cell),
            dims="cell",
        )
        # Within-group interval growth (items) over the common window:
        # first->second and second->last common wave, plus first->last.
        if len(common_waves) >= 2:
            pm.Deterministic(
                "growth_first_next_items",
                mean_items[_cell_pos(common_waves[1])]
                - mean_items[_cell_pos(common_waves[0])],
                dims="group",
            )
        if len(common_waves) >= 3:
            pm.Deterministic(
                "growth_next_last_items",
                mean_items[_cell_pos(common_waves[-1])]
                - mean_items[_cell_pos(common_waves[1])],
                dims="group",
            )
        if len(common_waves) >= 2:
            pm.Deterministic(
                "growth_first_last_items",
                mean_items[_cell_pos(common_waves[-1])]
                - mean_items[_cell_pos(common_waves[0])],
                dims="group",
            )

    return BuiltModel(
        model=model, prepared=panel, payload=EmptyPayload()
    )


def build_rlm_joint_growth_model(
    panel: LongitudinalPanel,
    *,
    measures: tuple[str, ...] = ("basread", "bpvs", "basdig"),
    eta_prior_sigma: float = 1.5,
    # HalfNormal(1.0) since #383: the 0.5 scale was in prior-data conflict with
    # the Down-syndrome verbal/reading heterogeneity (posteriors 1.25-1.39, at
    # the HalfNormal(0.5) 99th percentile). Every registered consumer sets the
    # value explicitly in its spec; this default matches the reviewed choice.
    sigma_subject_prior_sigma: float = 1.0,
    # See build_historical_growth_model: dispersion-scale prior (2026-08-21
    # review, finding 8).
    dispersion_prior_sigma: float = 0.25,
    lkj_eta: float = 2.0,
    within_correlation: bool = False,
    sigma_within_prior_sigma: float = 0.5,
    within_lkj_eta: float = 2.0,
) -> BuiltModel[EmptyPayload]:
    """Byrne joint correlated group-by-wave growth models (#338/#409).

    The multivariate extension of :func:`build_historical_growth_model`: each
    measure keeps its own supported-cell population grid, group-indexed
    subject-intercept SD and group-indexed overdispersion, and the per-child
    stable offsets are **correlated across measures** through an LKJ prior::

        score_imt ~ BetaBinomial(n_m, p_imt, kappa[m, group_i])
        kappa[m, g] = 1 / inv_sqrt_kappa[m, g]^2,  inv_sqrt_kappa ~ HalfNormal(0.25)
        logit(p_imt) = eta_cell[m, cell(group_i, wave_t)] + u_im
        (u_i1..u_iM) ~ MVN(0, diag(sigma[m, group_i]) R diag(sigma[m, group_i]))

    The headline deterministic is ``measure_corr`` (R) - how the children's
    stable levels on the measures move together within group, the Byrne
    "reading-language-memory coupling" question in its most parsimonious
    between-child form. The correlation matrix is shared across groups (an
    explicit assumption, stated in the report); the scales are group-indexed
    per the #338 heterogeneity decision. Descriptive natural-history only -
    ``readgrp`` is a cohort factor and nothing here is causal.

    When ``within_correlation=True``, a second LKJ-correlated latent deviation
    is added for each child-wave row. The deviations are double-centred so they
    average to zero within child and within group-by-wave cell, separating them
    from the stable child offsets and population cell means. This path requires
    a balanced panel and reports ``within_corr`` on the latent logit scale. Its
    likelihood is Binomial rather than Beta-Binomial: the logistic-normal
    residual supplies the extra-Binomial variance. Fitting both residual scales
    made the correlation prior-dominated in the development probe, matching the
    registered joint-mechanism precedent.

    The panel must be loaded with all ``measures`` (complete-case core plus
    extension rows require every measure observed on a row), so all measures
    share one supported-cell set.
    """
    df = panel.long
    dataset = panel.dataset
    subj, wave_c, grp = dataset.subject_col, dataset.wave_col, dataset.group_col
    for m in measures:
        if m not in panel.n_trials:
            raise KeyError(f"measure {m!r} not in panel (have {panel.measures}).")
    M = len(measures)

    group_codes = panel.group_codes
    group_labels = panel.group_labels
    subject_ids = panel.subject_ids
    group_index = {code: i for i, code in enumerate(group_codes)}
    subject_index = {s: i for i, s in enumerate(subject_ids)}

    cells = panel.cells(measures[0])
    for m in measures[1:]:
        if panel.cells(m) != cells:
            raise ValueError(
                f"measure {m!r} has a different supported-cell set than "
                f"{measures[0]!r}; the joint model needs one shared cell set."
            )
    cell_index = {cell: i for i, cell in enumerate(cells)}
    cell_labels = [
        f"{group_labels[group_index[g]]} | wave {w}" for g, w in cells
    ]
    common_waves = [
        w
        for w in sorted({w for _g, w in cells})
        if all((g, w) in cell_index for g in group_codes)
    ]

    group_idx = _map_panel_rows(df[grp].tolist(), group_index, what="group code")
    obs_cell_idx = np.array(
        [
            cell_index[(int(g), int(w))]
            for g, w in zip(df[grp], df[wave_c], strict=True)
        ],
        dtype=int,
    )
    subject_idx = _map_panel_rows(df[subj].tolist(), subject_index, what="subject")
    subject_group = (
        df.drop_duplicates(subj)
        .set_index(subj)
        .loc[subject_ids, grp]
        .map(group_index)
        .to_numpy(dtype=int)
    )
    if within_correlation:
        if df.duplicated([subj, wave_c]).any():
            raise ValueError(
                "within_correlation requires exactly one row per child and wave"
            )
        subject_wave_sets = {
            tuple(sorted(int(wave) for wave in frame[wave_c]))
            for _subject, frame in df.groupby(subj, sort=False)
        }
        if len(subject_wave_sets) != 1:
            raise ValueError(
                "within_correlation requires a balanced panel with the same waves "
                "for every child"
            )

    coords = {
        "measure": list(measures),
        "measure_b": list(measures),
        "group": group_labels,
        "cell": cell_labels,
        "subject": [str(s) for s in subject_ids],
        "obs": np.arange(len(df)),
    }

    def _cell_pos(wave: int) -> list[int]:
        return [cell_index[(g, wave)] for g in group_codes]

    with pm.Model(coords=coords) as model:
        eta_cell = _priors.declare(
                       pm.Normal(
                                   "eta_cell", mu=0.0, sigma=eta_prior_sigma, dims=("measure", "cell")
                               ),
                       role="nuisance",
                       rationale=(
                           "Group-by-wave population level per cell/measure on the logit scale "
                           "(Normal(0, 1.5)); the fitted cells (mean_items) and growth "
                           "intervals are deterministics of it — descriptive, not a treatment "
                           "effect."
                       ),
                   )
        sigma_subject = _priors.declare(
                            pm.HalfNormal(
                                        "sigma_subject",
                                        sigma=sigma_subject_prior_sigma,
                                        dims=("measure", "group"),
                                    ),
                            role="nuisance",
                            rationale=(
                                "Group-indexed between-subject random-intercept SD (HalfNormal(1)); "
                                "between-child heterogeneity that differs by cohort group."
                            ),
                        )
        kappa = None
        if not within_correlation:
            # Dispersion-scale prior, as in build_historical_growth_model.
            inv_sqrt_kappa = _priors.inv_sqrt_kappa_prior(
                sigma=dispersion_prior_sigma
            ).to_pymc("inv_sqrt_kappa", dims=("measure", "group"))
            kappa = pm.Deterministic(
                "kappa",
                1.0 / (inv_sqrt_kappa**2 + 1e-6),
                dims=("measure", "group"),
            )

        # Correlated per-child stable offsets across measures. The environment's
        # LKJCorr returns the CHOLESKY FACTOR L, not the correlation matrix R,
        # so R = L @ L.T (see the PyMC-version note in the repo memory /
        # LKJCholeskyCov discussion in mm-001: LKJCorr avoids the unused sd
        # scales of LKJCholeskyCov in a correlation-only role).
        chol = _priors.declare(
                   pm.LKJCorr("measure_corr_chol", n=M, eta=lkj_eta),
                   role="association",
                   rationale=(
                       "LKJ(eta=2) prior on the Cholesky factor of the between-child "
                       "cross-measure correlation (LKJCorrRV(<constant>, 2)); R = chol @ "
                       "chol.T is the headline reading-language-memory coupling estimand."
                   ),
               )
        measure_corr = pm.Deterministic(
            "measure_corr", chol @ chol.T, dims=("measure", "measure_b")
        )
        iu, ju = np.triu_indices(M, k=1)
        if len(iu):
            pm.Deterministic(
                "measure_corr_pairs",
                pt.stack(
                    [measure_corr[i, j] for i, j in zip(iu, ju, strict=True)]
                ),
            )

        z_subject = _priors.declare(
                        pm.Normal(
                                    "z_subject", mu=0.0, sigma=1.0, dims=("subject", "measure")
                                ),
                        role="nuisance",
                        rationale=(
                            "Non-centred standard-normal per-subject offsets (Normal(0, 1)); "
                            "group-centred and scaled by sigma_subject to form the subject "
                            "random effects."
                        ),
                    )
        corr_z = z_subject @ chol.T  # rows ~ MVN(0, R)
        # Group-centre per (group, measure) for identifiability against the
        # per-measure group-by-wave grids (same device as the hg factory).
        z_group_mean = pt.stack(
            [
                corr_z[subject_group == g].mean(axis=0)
                for g in range(len(group_codes))
            ]
        )
        centred = corr_z - z_group_mean[subject_group]
        subject_offset = pm.Deterministic(
            "subject_offset",
            centred * sigma_subject.T[subject_group],
            dims=("subject", "measure"),
        )

        within_offset = None
        if within_correlation:
            sigma_within = _priors.declare(
                               pm.HalfNormal(
                                               "sigma_within",
                                               sigma=sigma_within_prior_sigma,
                                               dims="measure",
                                           ),
                               role="nuisance",
                               rationale=(
                                   "Scale of the wave-specific within-child departure on the logit "
                                   "scale (HalfNormal(0.5)). This model's likelihood is Binomial "
                                   "rather than Beta-Binomial, so this term carries ALL extra-Binomial "
                                   "variance — true within-child fluctuation and measurement noise "
                                   "together — and the double sum-to-zero centring makes the realised "
                                   "departure SD smaller than this parameter."
                               ),
                           )
            within_chol = _priors.declare(
                              pm.LKJCorr(
                                              "within_corr_chol", n=M, eta=within_lkj_eta
                                          ),
                              role="association",
                              rationale=(
                                  "LKJ prior on the Cholesky factor of the WITHIN-child cross-measure "
                                  "correlation of wave-specific departures (LKJCorrRV(<constant>, "
                                  "2)); within_corr = chol @ chol.T is the headline estimand of the "
                                  "within-child companion. Interpretable only for a measure pair "
                                  "whose residual scales are resolvable."
                              ),
                          )
            within_corr = pm.Deterministic(
                "within_corr",
                within_chol @ within_chol.T,
                dims=("measure", "measure_b"),
            )
            if len(iu):
                pm.Deterministic(
                    "within_corr_pairs",
                    pt.stack(
                        [
                            within_corr[i, j]
                            for i, j in zip(iu, ju, strict=True)
                        ]
                    ),
                )
            z_within = _priors.declare(
                           pm.Normal(
                                           "z_within", mu=0.0, sigma=1.0, dims=("obs", "measure")
                                       ),
                           role="nuisance",
                           rationale=(
                               "Non-centred standard-normal per-row, per-measure within-child "
                               "offsets (Normal(0, 1)); correlated through within_corr_chol, "
                               "double-centred within child and within group-by-wave cell, and "
                               "scaled by sigma_within."
                           ),
                       )
            raw_within = z_within @ within_chol.T
            subject_means = pt.stack(
                [
                    raw_within[subject_idx == s].mean(axis=0)
                    for s in range(len(subject_ids))
                ]
            )
            centred_on_subject = raw_within - subject_means[subject_idx]
            cell_means = pt.stack(
                [
                    centred_on_subject[obs_cell_idx == c].mean(axis=0)
                    for c in range(len(cells))
                ]
            )
            centred_within = centred_on_subject - cell_means[obs_cell_idx]
            within_offset = pm.Deterministic(
                "within_offset",
                centred_within * sigma_within,
                dims=("obs", "measure"),
            )

        for mi, m in enumerate(measures):
            n_trials = int(panel.n_trials[m])
            observed = df[m].to_numpy(dtype=int)
            eta_obs = (
                eta_cell[mi, obs_cell_idx] + subject_offset[subject_idx, mi]
            )
            if within_offset is not None:
                eta_obs = eta_obs + within_offset[:, mi]
            p_obs = pm.math.sigmoid(eta_obs)
            if within_correlation:
                pm.Binomial(
                    f"score_{m}",
                    n=n_trials,
                    p=p_obs,
                    observed=observed,
                    dims="obs",
                )
            else:
                kappa = require_value(kappa, "the dispersion concentration")
                kappa_obs = kappa[mi, group_idx]
                pm.BetaBinomial(
                    f"score_{m}",
                    n=n_trials,
                    alpha=p_obs * kappa_obs,
                    beta=(1.0 - p_obs) * kappa_obs,
                    observed=observed,
                    dims="obs",
                )
            pm.Deterministic(
                f"fitted_mean_items_obs_{m}", n_trials * p_obs, dims="obs"
            )
            pm.Deterministic(
                f"mean_items_{m}",
                n_trials * pm.math.sigmoid(eta_cell[mi]),
                dims="cell",
            )
            if len(common_waves) >= 2:
                # The **median-child** growth over the common window: built from
                # ``eta_cell`` with the child offset at zero, so it is the change
                # for a mid-group child, not the change in the cell average. The
                # joint family's published growth
                # (``posterior_growth_summary_{m}``) is the matched-children
                # average taken from ``fitted_mean_items_obs_{m}`` instead, and no
                # historical-joint reporting path reads this node — it is retained
                # for parity with the single-measure ``historical_growth`` family,
                # which does summarise it. Do not quote it as this family's growth
                # result (2026-08-24 historical-joint review).
                mean_items_m = n_trials * pm.math.sigmoid(eta_cell[mi])
                pm.Deterministic(
                    f"growth_first_last_items_{m}",
                    mean_items_m[_cell_pos(common_waves[-1])]
                    - mean_items_m[_cell_pos(common_waves[0])],
                    dims="group",
                )

    return BuiltModel(model=model, prepared=panel, payload=EmptyPayload())
