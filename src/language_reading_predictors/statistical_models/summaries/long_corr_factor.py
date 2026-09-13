# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Long corr factor calculations and summaries."""

from __future__ import annotations

from language_reading_predictors.statistical_models.posteriors import REPORTING_CI_PROB

import numpy as np
import pandas as pd
import xarray as xr
from language_reading_predictors.statistical_models.posteriors import (
    band50,
)


def _factor_corr_draws(trace: xr.DataTree, group: str = "posterior") -> tuple:
    """Return ``(corr, waves, domains)`` from a longitudinal-CFA ``factor_corr`` node.

    ``corr`` is a numpy array of shape ``(S, T, D, D)`` (sample × wave × domain ×
    domain), ``waves`` the wave labels, ``domains`` the domain names.
    """
    post = getattr(trace, group)
    fc = post["factor_corr"].stack(sample=("chain", "draw"))
    fc = fc.transpose("sample", "wave", "domain", "domain_b")
    corr = np.asarray(fc.values)  # (S, T, D, D)
    waves = [w.item() if hasattr(w, "item") else w for w in fc.coords["wave"].values]
    domains = [str(d) for d in fc.coords["domain"].values]
    return corr, waves, domains


def longitudinal_factor_correlations(
    trace: xr.DataTree, *, ci_prob: float = REPORTING_CI_PROB, group: str = "posterior"
) -> pd.DataFrame:
    """Per-wave latent factor correlations (the #313 headline).

    One row per (wave, unique off-diagonal domain pair): the posterior median/mean and
    equal-tailed ``ci_prob`` interval (plus an inner 50 % band) of the within-wave latent
    correlation, and ``prob_pos`` = ``P(rho > 0)``. These are model-based latent-domain
    descriptive associations, with indicator-specific residual variation represented
    separately; they are never causal.
    """
    corr, waves, domains = _factor_corr_draws(trace, group)
    D = len(domains)
    lo_q = (1 - ci_prob) / 2
    rows: list[dict] = []
    for w_i, w in enumerate(waves):
        for i in range(D):
            for j in range(i + 1, D):
                d = corr[:, w_i, i, j]
                lo50, hi50 = band50(d)
                rows.append(
                    {
                        "wave": w,
                        "domain_i": domains[i],
                        "domain_j": domains[j],
                        "median": float(np.median(d)),
                        "mean": float(np.mean(d)),
                        "sd": float(np.std(d)),
                        "lo": float(np.quantile(d, lo_q)),
                        "hi": float(np.quantile(d, 1 - lo_q)),
                        "lo50": lo50,
                        "hi50": hi50,
                        "prob_pos": float(np.mean(d > 0)),
                    }
                )
    return pd.DataFrame(rows)


def longitudinal_conditional_slopes(
    trace: xr.DataTree, *, ci_prob: float = REPORTING_CI_PROB, group: str = "posterior"
) -> pd.DataFrame:
    """Per-wave conditional (partial) latent slopes among the domain factors.

    For each wave and each ordered pair ``(target, predictor)`` the partial
    regression coefficient of the (unit-variance) target factor on the predictor
    factor **controlling for every other factor**, derived per draw from the
    within-wave latent correlation matrix (the multiple-regression coefficient
    ``beta = R[pred, pred]^-1 R[pred, target]``). This is a latent-factor companion
    to the concurrent family's mutually-adjusted observed-score slopes (#312), not the
    same estimand or a guaranteed correction of it: an **adjusted association**, not a
    causal effect. With two predictors the coefficient is a partial slope; with one it
    coincides with the pairwise correlation.
    """
    corr, waves, domains = _factor_corr_draws(trace, group)
    S, T, D, _ = corr.shape
    lo_q = (1 - ci_prob) / 2
    rows: list[dict] = []
    for w_i, w in enumerate(waves):
        R = corr[:, w_i]  # (S, D, D)
        for a in range(D):
            preds = [k for k in range(D) if k != a]
            R_pp = R[:, preds][:, :, preds]  # (S, P, P)
            r_pa = R[:, preds, a]  # (S, P)
            beta = np.linalg.solve(R_pp, r_pa[..., None])[..., 0]  # (S, P)
            for bi, b in enumerate(preds):
                d = beta[:, bi]
                lo50, hi50 = band50(d)
                rows.append(
                    {
                        "wave": w,
                        "target": domains[a],
                        "predictor": domains[b],
                        "median": float(np.median(d)),
                        "mean": float(np.mean(d)),
                        "sd": float(np.std(d)),
                        "lo": float(np.quantile(d, lo_q)),
                        "hi": float(np.quantile(d, 1 - lo_q)),
                        "lo50": lo50,
                        "hi50": hi50,
                        "prob_pos": float(np.mean(d > 0)),
                    }
                )
    return pd.DataFrame(rows)


def disattenuation_crosscheck(latent_df: pd.DataFrame, observed_df: pd.DataFrame) -> pd.DataFrame:
    """Merge latent factor correlations with observed indicator correlations.

    ``latent_df`` is :func:`longitudinal_factor_correlations` output; ``observed_df``
    carries the raw same-wave observed correlation (``observed_corr``) for each
    ``(wave, domain_i, domain_j)`` — the mean pairwise correlation between the two
    domains' standardised indicators. ``gap`` is ``|latent| - |observed|`` and
    ``latent_ge_observed`` records its direction (with a small numerical tolerance).
    This is a descriptive model check, not an acceptance gate: the latent factor and
    the mean indicator-pair correlation are different estimands, so factor aggregation,
    the loading structure, residual structure and sampling uncertainty can all break a
    simple attenuation ordering even when measurement error is present.
    """
    merged = latent_df.merge(observed_df, on=["wave", "domain_i", "domain_j"], how="left")
    lat = merged["mean"].abs()
    obs = merged["observed_corr"].abs()
    merged["gap"] = lat - obs
    # A small tolerance absorbs Monte-Carlo noise around a zero gap. A missing
    # observed comparator (a wave/pair with too few pairwise-complete indicator
    # pairs, or a merge miss) must stay NA rather than comparing False and being
    # counted as a reversal (2026-08-21 review, finding 10).
    flags = pd.array(((lat + 1e-3) >= obs).to_numpy(), dtype="boolean")
    flags[obs.isna().to_numpy()] = pd.NA
    merged["latent_ge_observed"] = flags
    return merged
