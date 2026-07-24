# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Divergence-influence + eigenvalue-spectrum probe for the mm-001 gate exception.

Reproducible evidence behind the #412 sampling-quality gate exception for
``lrp-rlm-mm-001`` (see the companion notes entry). Given the model's saved
``trace.nc`` it reports:

1. **Eigenvalue spectrum** of the posterior-mean domain-factor correlation matrix
   — the dominant-eigenvalue share (near-unidimensionality) and, across the whole
   posterior, the *smallest* eigenvalue (how hard the correlation matrix sits
   against the positive-definite boundary; the source of the divergences).
2. **Divergence-influence test** — recompute the published deliverables (per-pair
   factor correlations and per-indicator communalities) from ALL posterior draws
   versus with the divergent draws EXCLUDED, and report the largest shift. A shift
   that vanishes at reporting precision is the evidence the divergences do not
   contaminate the stored summary.
3. **Divergence clustering** — divergent draws per chain, to check they are diffuse
   rather than concentrated in one region of the posterior.

Sign-off condition (#412): re-run this against the **reporting-tier** trace before
merge and confirm the smallest eigenvalue stays near zero throughout, the
divergences remain unclustered, and excluding them leaves the correlations and
communalities unmoved at reporting precision. If the residual divergences on the
reparameterised fit *do* cluster, the exception is not justified.

    python notes/assets/202607241200-mm001-gate-exception-probe.py <path/to/trace.nc>

Limitation (stated in the notes): this covers the *crude* failure mode — divergent
draws contaminating the stored summary. It does NOT bound posterior mass the sampler
never reached; that risk is addressed separately by the target_accept=0.999
invariance run and the R-hat 1.006 / ESS mixing of the deliverables.
"""

from __future__ import annotations

import sys

import arviz as az
import numpy as np


def _corr_pairs(factor_corr: np.ndarray, domains: list[str]) -> dict[tuple[str, str], float]:
    """Upper-triangle mean factor correlations from draws of shape (S, D, D)."""
    m = factor_corr.mean(axis=0)
    return {
        (domains[i], domains[j]): float(m[i, j])
        for i in range(len(domains))
        for j in range(i + 1, len(domains))
    }


def probe(trace_path: str) -> dict:
    idata = az.from_netcdf(trace_path)
    post = idata.posterior
    domains = [str(d) for d in post["domain"].values]

    fc = post["factor_corr"].stack(sample=("chain", "draw"))
    fc = np.asarray(fc.transpose("sample", "domain", "domain_b").values)  # (S, D, D)
    comm = np.asarray(
        post["communality"].stack(sample=("chain", "draw")).transpose("sample", "indicator").values
    )  # (S, n_ind)

    # (1) eigenvalue spectrum
    eig_mean = np.linalg.eigvalsh(fc.mean(axis=0))[::-1]  # descending
    dom_share = float(eig_mean[0] / eig_mean.sum())
    per_draw = np.linalg.eigvalsh(fc)  # (S, D), ascending per draw
    min_eig = per_draw[:, 0]
    print(f"[eigenvalues] dominant share (posterior-mean matrix): {dom_share:.3f}")
    print(f"[eigenvalues] full spectrum (mean matrix): {np.round(eig_mean, 3).tolist()}")
    print(
        f"[eigenvalues] smallest eigenvalue over the posterior: "
        f"median {np.median(min_eig):.4f}, 5th pct {np.quantile(min_eig, 0.05):.4f}, "
        f"max {min_eig.max():.4f}"
    )

    # (3) divergences
    div = np.asarray(
        idata.sample_stats["diverging"].stack(sample=("chain", "draw")).values
    ).astype(bool)
    n_div = int(div.sum())
    per_chain = idata.sample_stats["diverging"].sum(dim="draw").values
    print(f"[divergences] total {n_div} of {div.size}; per chain {np.asarray(per_chain).tolist()}")

    # (2) divergence-influence test
    all_pairs = _corr_pairs(fc, domains)
    kept_pairs = _corr_pairs(fc[~div], domains)
    corr_shift = max(abs(all_pairs[k] - kept_pairs[k]) for k in all_pairs)
    comm_shift = float(np.max(np.abs(comm.mean(0) - comm[~div].mean(0))))
    print(
        f"[influence] max |Δ factor correlation| excluding divergences: {corr_shift:.4f}"
    )
    print(f"[influence] max |Δ communality| excluding divergences: {comm_shift:.4f}")

    return {
        "dominant_eigenvalue_share": dom_share,
        "min_eigenvalue_median": float(np.median(min_eig)),
        "n_divergences": n_div,
        "max_corr_shift_excluding_divergences": corr_shift,
        "max_communality_shift_excluding_divergences": comm_shift,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    probe(sys.argv[1])
