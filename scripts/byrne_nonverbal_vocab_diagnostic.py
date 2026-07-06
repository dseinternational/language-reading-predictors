# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Byrne cohort: is non-verbal ability (basmat) associated with receptive vocabulary (bpvs)? (issue #186, Q4 Phase 3).

**Descriptive association only.** The formal analogue of the RLI Q4 read-out — a
bounded-count, covariate-adjusted Bayesian model — is deliberately deferred, because
in the Byrne data:

1. block design's counterpart ``basmat`` is measured only from **wave 3**, so it is
   not a baseline (there is no t1 non-verbal measure to condition on);
2. the ``bpvs`` / ``basmat`` instrument **ceilings are unconfirmed** (only
   ``basread`` = 87 is registered — issue #164, decision 3), so a Beta-Binomial
   denominator cannot be set responsibly;
3. the historical-growth family is a **descriptive group-by-wave** model with no
   covariate-adjusted seam (unlike the ITT ``adjust_for`` path).

So this reports a **scale-free rank association** that needs none of the above: it
answers "does non-verbal ability track receptive vocabulary in the Byrne cohort?",
which is a *marginal* association (block design and vocabulary both load on latent
general ability), not an adjusted/incremental effect. The primary result is the
**wave-3 cross-section** (one row per child, the largest ``basmat`` wave); waves 4–5
are smaller robustness rows.

Usage::

    python scripts/byrne_nonverbal_vocab_diagnostic.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from language_reading_predictors import paths

_BYRNE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "reading-language-memory"
    / "reading_language_memory_data_long.csv"
)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of ``x`` and ``y`` partialling out ``z`` (rank residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    zmat = np.c_[np.ones(len(rz)), rz]

    def _resid(a: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(zmat, a, rcond=None)
        return a - zmat @ beta

    return float(spearmanr(_resid(rx), _resid(ry)).statistic)


def wave_summary(df: pd.DataFrame, wave: int) -> dict[str, object]:
    # Require age as well: the age-adjusted partial below would otherwise propagate
    # NaNs from a missing age (via rankdata/lstsq) while ``n`` still counted the row.
    w = df[
        (df["time"] == wave)
        & df["basmat"].notna()
        & df["bpvs"].notna()
        & df["age"].notna()
    ]
    row: dict[str, object] = {
        "wave": wave,
        "n": len(w),
        "spearman_overall": float(spearmanr(w["basmat"], w["bpvs"]).statistic),
        "spearman_age_partial": partial_spearman(
            w["basmat"].to_numpy(float), w["bpvs"].to_numpy(float), w["age"].to_numpy(float)
        ),
    }
    for g, gg in w.groupby("readgrp"):
        row[f"spearman_grp{int(g)}"] = (
            float(spearmanr(gg["basmat"], gg["bpvs"]).statistic) if len(gg) > 3 else None
        )
    return row


def adjusted_ols_t3(df: pd.DataFrame) -> dict[str, float]:
    """Age + reading-group-adjusted standardized OLS coefficient of basmat on bpvs at t3."""
    import statsmodels.formula.api as smf

    w = df[
        (df["time"] == 3)
        & df["basmat"].notna()
        & df["bpvs"].notna()
        & df["age"].notna()
    ].copy()
    for c in ("basmat", "bpvs", "age"):
        w[f"z_{c}"] = (w[c] - w[c].mean()) / w[c].std()
    fit = smf.ols("z_bpvs ~ z_basmat + z_age + C(readgrp)", data=w).fit()
    lo, hi = fit.conf_int().loc["z_basmat"]
    return {
        "n": int(fit.nobs),
        "basmat_beta_std": float(fit.params["z_basmat"]),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Descriptive basmat->bpvs association in the Byrne cohort (#186 Q4 Phase 3)."
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(_BYRNE)
    waves = pd.DataFrame([wave_summary(df, w) for w in (3, 4, 5)])
    ols = adjusted_ols_t3(df)

    out = args.out or os.path.join(
        paths.output_root(), "comparisons", "byrne_nonverbal_vocab.csv"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    waves.to_csv(out, index=False)

    print("Byrne basmat -> bpvs (marginal rank association; basmat is wave 3+):")
    print(waves.to_string(index=False))
    print(
        "\nWave-3 age + reading-group-adjusted OLS (standardised):"
        f"\n  basmat beta = {ols['basmat_beta_std']:+.3f} "
        f"[95% CI {ols['ci_lo']:+.3f}, {ols['ci_hi']:+.3f}]  (n={ols['n']})"
    )
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
