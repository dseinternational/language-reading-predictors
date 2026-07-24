# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Descriptive and comparison summaries for the longitudinal correlated-factor (LCF) model.

The LCF family's triangulation, items-scale translation and concurrent-comparison
computations (#394 pillar 6): mostly-pure functions of the fitted context and built
model that return DataFrames, with table persistence left to the family pipeline.
Kept separate from the inference algorithms in :mod:`lcf_inference`, and out of the
pipeline monolith, so the summary calculations are testable without an output
directory, Quarto template or Matplotlib session.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def observed_domain_corr(built) -> pd.DataFrame:
    """Observed same-wave cross-domain indicator correlations for triangulation.

    For each wave and each unique domain pair, compute the mean pairwise
    (pairwise-complete) Pearson correlation between the two domains' standardised
    logit indicators. This is a descriptive comparator, not an attenuation-bound:
    it is not the same estimand as the model's latent factor correlation.
    """
    panel = built.prepared
    domains = built.extras["domains"]
    standardisers = built.extras["standardisers"]
    waves = built.extras["waves"]
    dnames = list(domains)
    # Standardised logit per indicator (pooled, exactly as the factory), (N, T).
    z = {}
    for d, syms in domains.items():
        for s in syms:
            mean, sd = standardisers[s]
            z[s] = (np.asarray(panel.logit[s], dtype=float) - mean) / sd
    rows = []
    for w_i in range(len(waves)):
        for i in range(len(dnames)):
            for j in range(i + 1, len(dnames)):
                vals = []
                for si in domains[dnames[i]]:
                    for sj in domains[dnames[j]]:
                        a = z[si][:, w_i]
                        b = z[sj][:, w_i]
                        m = np.isfinite(a) & np.isfinite(b)
                        if m.sum() >= 3 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                            vals.append(float(np.corrcoef(a[m], b[m])[0, 1]))
                rows.append(
                    {
                        "wave": waves[w_i],
                        "domain_i": dnames[i],
                        "domain_j": dnames[j],
                        "observed_corr": float(np.mean(vals)) if vals else float("nan"),
                        "n_indicator_pairs": len(vals),
                    }
                )
    return pd.DataFrame(rows)


def items_scale(ctx, built) -> pd.DataFrame:
    """Approximate items-scale translation of the headline cross-domain couplings.

    For one representative indicator pair per cross-domain combination (the first
    listed indicator of each domain), the delta-method slope of the target
    indicator's item count per +1 item of the predictor indicator, at the pooled-mean
    operating point, per wave. Derived from the latent correlation and the loadings:
    ``slope_z = lambda_m lambda_k rho / (lambda_k^2 + sigma_k^2)`` on the standardised
    logit scale, scaled to items by the two indicators' logit SDs, denominators, and
    operating-point ``p(1-p)``. **Approximate and descriptive** — a linearisation at
    the mean, comparable to the #312 concurrent items-scale marginals, not a causal
    effect.
    """
    from scipy.special import expit

    post = ctx.trace.posterior
    domains = built.extras["domains"]
    standardisers = built.extras["standardisers"]
    waves = built.extras["waves"]
    hdi = ctx.reporting.ci_prob
    lo_q = (1 - hdi) / 2
    dnames = list(domains)
    ind_names = [str(s) for s in post["indicator"].values]

    def _lam_sig(sym):
        k = ind_names.index(sym)
        lam = post["lambda_load"].isel(indicator=k).stack(sample=("chain", "draw")).values.ravel()
        sig = post["sigma_indicator"].isel(indicator=k).stack(sample=("chain", "draw")).values.ravel()
        return lam, sig

    fc = post["factor_corr"].stack(sample=("chain", "draw"))
    fc = fc.transpose("sample", "wave", "domain", "domain_b")
    corr = np.asarray(fc.values)  # (S, T, D, D)
    dom_idx = {d: i for i, d in enumerate(dnames)}

    rows = []
    for i in range(len(dnames)):
        for j in range(i + 1, len(dnames)):
            di, dj = dnames[i], dnames[j]
            k_sym = domains[dj][0]  # predictor indicator (domain j)
            m_sym = domains[di][0]  # target indicator (domain i)
            lam_k, sig_k = _lam_sig(k_sym)
            lam_m, _ = _lam_sig(m_sym)
            mean_k, sd_k = standardisers[k_sym]
            mean_m, sd_m = standardisers[m_sym]
            N_k = ctx.prepared.n_trials[k_sym]
            N_m = ctx.prepared.n_trials[m_sym]
            p_k = float(expit(mean_k))
            p_m = float(expit(mean_m))
            # ``logit_safe`` uses (y + 0.5)/(N + 1), so the inverse-count
            # derivative is (N + 1) p(1-p), not the binomial N p(1-p).
            info_k = (N_k + 1) * p_k * (1 - p_k)
            info_m = (N_m + 1) * p_m * (1 - p_m)
            for w_i, w in enumerate(waves):
                rho = corr[:, w_i, dom_idx[di], dom_idx[dj]]
                slope_z = lam_m * lam_k * rho / (lam_k**2 + sig_k**2)
                # Δitems_m per +1 item of k at the mean operating point.
                items_slope = slope_z * (sd_m / sd_k) * (info_m / info_k)
                rows.append(
                    {
                        "wave": w,
                        "predictor_indicator": k_sym,
                        "target_indicator": m_sym,
                        "predictor_domain": dj,
                        "target_domain": di,
                        "items_per_item_mean": float(np.mean(items_slope)),
                        "items_per_item_lo": float(np.quantile(items_slope, lo_q)),
                        "items_per_item_hi": float(np.quantile(items_slope, 1 - lo_q)),
                        "prob_pos": float(np.mean(items_slope > 0)),
                    }
                )
    return pd.DataFrame(rows)


_LCF_CA_COMPARISONS = {
    "L": ("R", "E", "TR", "TE"),
    "R": ("L", "B"),
    "E": ("L", "B"),
    "TR": ("L", "B"),
    "TE": ("L", "B"),
}
_LCF_CA_MODEL_IDS = {
    "L": "lrp-rli-ca-002",
    "R": "lrp-rli-ca-005",
    "E": "lrp-rli-ca-006",
    "TR": "lrp-rli-ca-003",
    "TE": "lrp-rli-ca-004",
}


def observed_conditional_slope(
    corr: np.ndarray,
    loadings: np.ndarray,
    residual_sds: np.ndarray,
    *,
    target_domain_idx: int,
    predictor_domain_idx: int,
    target_indicator_idx: int,
    predictor_indicator_idx: int,
) -> np.ndarray:
    """Observed-indicator slope implied by an LCF conditional domain coupling.

    Conditions on every latent domain other than the target and predictor. If
    ``C`` denotes those domains, the relevant factor covariance and predictor
    variance are ``Cov(f_a, f_b | f_C)`` and ``Var(f_b | f_C)``. Mapping through
    the target loading and the noisy predictor indicator gives

    ``lambda_a lambda_b Cov_ab.C / (lambda_b^2 Var_b.C + sigma_b^2)``.

    Using the marginal reliability ``lambda_b / (lambda_b^2 + sigma_b^2)`` here
    would mix a C-conditional factor coefficient with a marginal measurement
    update and overstate or understate the resulting observed-score slope.
    """
    corr = np.asarray(corr, dtype=float)
    n_domains = corr.shape[-1]
    conditioned = [
        idx
        for idx in range(n_domains)
        if idx not in {target_domain_idx, predictor_domain_idx}
    ]
    if conditioned:
        corr_cc = corr[:, :, conditioned, :][:, :, :, conditioned]
        corr_cb = corr[:, :, conditioned, predictor_domain_idx]
        solve_cb = np.linalg.solve(corr_cc, corr_cb[..., None])[..., 0]
        predictor_variance = 1.0 - np.sum(
            corr[:, :, predictor_domain_idx, conditioned] * solve_cb, axis=-1
        )
        conditional_covariance = corr[
            :, :, target_domain_idx, predictor_domain_idx
        ] - np.sum(
            corr[:, :, target_domain_idx, conditioned] * solve_cb, axis=-1
        )
    else:
        predictor_variance = np.ones(corr.shape[:2], dtype=float)
        conditional_covariance = corr[:, :, target_domain_idx, predictor_domain_idx]

    lambda_target = loadings[:, target_indicator_idx, None]
    lambda_predictor = loadings[:, predictor_indicator_idx, None]
    sigma_predictor = residual_sds[:, predictor_indicator_idx, None]
    return (
        lambda_target * lambda_predictor * conditional_covariance
        / (
            lambda_predictor**2 * predictor_variance
            + sigma_predictor**2
        )
    )


def concurrent_comparison(
    ctx,
    built,
    *,
    ca_tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Reproducible directed comparison with matching #312 associations.

    For each cross-domain target/predictor pair shared with CA002--006, translate
    the LCF's latent conditional slope to target items for a one same-wave-SD
    increase in the observed predictor. The translation conditions on the other
    latent domains and is evaluated at the target's observed wave-mean logit. Place
    it beside #312's adjusted average marginal effect for the same direction and
    raw predictor contrast.

    The columns deliberately keep the two estimates separate: #312 conditions on
    five observed tests plus age/group terms and averages a nonlinear marginal over
    children, whereas the LCF conditions on the remaining latent domains and uses a
    mean-operating-point translation. This is a directional triangulation table,
    not a claim that the numbers estimate the same parameter.
    """
    from scipy.special import expit

    post = ctx.trace.posterior
    ci_prob = ctx.reporting.ci_prob
    lo_q = (1.0 - ci_prob) / 2.0
    domains = [str(value) for value in post.coords["domain"].values]
    domain_index = {domain: i for i, domain in enumerate(domains)}
    indicator_names = [str(value) for value in post.coords["indicator"].values]
    indicator_index = {symbol: i for i, symbol in enumerate(indicator_names)}
    domain_of = built.extras["domain_of"]

    corr = (
        post["factor_corr"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "wave", "domain", "domain_b")
        .values
    )
    loadings = (
        post["lambda_load"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "indicator")
        .values
    )
    residual_sds = (
        post["sigma_indicator"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "indicator")
        .values
    )

    if ca_tables is None:
        ca_tables = {}
        models_dir = os.path.dirname(ctx.output_dir)
        config_name = ctx.reporting.config_name
        for target, model_id in _LCF_CA_MODEL_IDS.items():
            path = os.path.join(
                models_dir,
                f"{model_id}-{config_name}",
                "concurrent_marginals.csv",
            )
            if os.path.exists(path):
                ca_tables[target] = pd.read_csv(path)

    rows: list[dict] = []
    panel = built.prepared
    for target, predictors in _LCF_CA_COMPARISONS.items():
        if target not in indicator_index or target not in domain_of:
            continue
        target_domain = domain_of[target]
        target_domain_idx = domain_index[target_domain]
        target_indicator_idx = indicator_index[target]
        target_sd = float(built.extras["standardisers"][target][1])
        target_trials = int(panel.n_trials[target])
        ca_table = ca_tables.get(target)

        for predictor in predictors:
            if predictor not in indicator_index or predictor not in domain_of:
                continue
            predictor_domain = domain_of[predictor]
            if predictor_domain == target_domain:
                continue
            predictor_domain_idx = domain_index[predictor_domain]
            predictor_indicator_idx = indicator_index[predictor]
            predictor_pooled_sd = float(
                built.extras["standardisers"][predictor][1]
            )

            observed_slope = observed_conditional_slope(
                corr,
                loadings,
                residual_sds,
                target_domain_idx=target_domain_idx,
                predictor_domain_idx=predictor_domain_idx,
                target_indicator_idx=target_indicator_idx,
                predictor_indicator_idx=predictor_indicator_idx,
            )

            for wave_idx, wave in enumerate(built.extras["waves"]):
                predictor_wave = np.asarray(panel.logit[predictor][:, wave_idx])
                target_wave = np.asarray(panel.logit[target][:, wave_idx])
                fitted_rows = np.isfinite(target_wave)
                predictor_wave_sd = float(
                    np.nanstd(predictor_wave[fitted_rows], ddof=1)
                )
                target_wave_mean = float(np.nanmean(target_wave[fitted_rows]))
                if not (
                    np.isfinite(predictor_wave_sd)
                    and predictor_wave_sd > 0
                    and np.isfinite(target_wave_mean)
                ):
                    continue
                predictor_delta_z = predictor_wave_sd / predictor_pooled_sd
                target_delta_logit = (
                    observed_slope[:, wave_idx] * predictor_delta_z * target_sd
                )
                # ``logit_safe`` uses the Haldane proportion (y + 0.5)/(N + 1),
                # whose inverse count difference is (N + 1) times the probability
                # difference (the -0.5 constants cancel).
                lcf_items = (target_trials + 1) * (
                    expit(target_wave_mean + target_delta_logit)
                    - expit(target_wave_mean)
                )

                ca_row = None
                if ca_table is not None:
                    matched = ca_table[
                        (ca_table["timepoint"] == int(wave))
                        & (ca_table["adjustment"] == "adjusted")
                        & (ca_table["term"] == predictor)
                        & (ca_table["scale"] == "+1 SD")
                    ]
                    if len(matched) > 1:
                        raise ValueError(
                            f"Expected at most one #312 row for {target} <- "
                            f"{predictor} at wave {wave}; found {len(matched)}"
                        )
                    if len(matched) == 1:
                        ca_row = matched.iloc[0]

                rows.append(
                    {
                        "wave": int(wave),
                        "target_indicator": target,
                        "predictor_indicator": predictor,
                        "target_domain": target_domain,
                        "predictor_domain": predictor_domain,
                        "predictor_contrast": "+1 same-wave SD",
                        "lcf_items_median": float(np.median(lcf_items)),
                        "lcf_items_lo": float(np.quantile(lcf_items, lo_q)),
                        "lcf_items_hi": float(np.quantile(lcf_items, 1.0 - lo_q)),
                        "lcf_prob_pos": float(np.mean(lcf_items > 0)),
                        "ca_items_median": (
                            float(ca_row["items_median"])
                            if ca_row is not None
                            else float("nan")
                        ),
                        "ca_items_lo": (
                            float(ca_row["items_lo"])
                            if ca_row is not None
                            else float("nan")
                        ),
                        "ca_items_hi": (
                            float(ca_row["items_hi"])
                            if ca_row is not None
                            else float("nan")
                        ),
                        "ca_prob_pos": (
                            float(ca_row["prob_pos"])
                            if ca_row is not None
                            else float("nan")
                        ),
                        "ca_available": ca_row is not None,
                        "ca_model_id": _LCF_CA_MODEL_IDS[target],
                    }
                )
    return pd.DataFrame(rows)
