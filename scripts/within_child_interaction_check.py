# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Between- vs within-child diagnostic for an interaction (confound check).

A frequentist triangulation for the Bayesian interaction models (LRP71/72/73):
an apparent ``mechanism × moderator`` interaction may be a **between-child
ability confound** (children who are higher on one skill tend to be higher on the
other and read better) rather than a within-child effect. This script fits an OLS
of the logit outcome on ``z(mechanism)``, ``z(moderator)`` and their product,
four ways:

1. **pooled (naive)** — classical SE; ignores the 3-4 rows/child, so it
   overstates the interaction (the inflation the exploration warned about).
2. **pooled (cluster-robust by subject)** — honest SE for the pooled estimate.
3. **subject fixed effects on the raw product** — subject dummies absorb the
   between-child *levels*, but by Frisch–Waugh–Lovell the FE estimator still uses
   the within-child variation of the **raw** product ``xL·xM``, which retains
   cross-level terms (between-child levels × within-child changes). So this
   coefficient is **not** a clean within-child interaction. Kept only as a
   contrast against estimator 4. Cluster-robust SE.
4. **within-child (double-demeaned product)** — the genuine within-child
   interaction: the product of the *within-demeaned* components
   ``(xL − x̄L_i)·(xM − x̄M_i)`` alongside subject fixed effects
   (Giesselmann & Schmidt-Catran 2020, DOI 10.1177/0081175020966850). This is the
   estimator the verdict reads. Cluster-robust SE.

If the interaction is large pooled but collapses to ~0 on estimator 4, it is a
between-child confound, not a developmental/skill-combination effect. This is the
check that contextualises LRP71 (phonics×vocab) and LRP72 (blending×letter-sound);
LRP73 (letter-sound×age) is its primary use — age is overwhelmingly a
between-child variable, so the FE-raw-product contamination is largest there.

The outcome / predictors are entered on the same logit scale the Bayesian models
use; OLS here is a rough linear diagnostic, not the inferential model (note: the
within-child SE does not account for the four-wave longitudinal structure beyond
clustering). Symbols are the measure symbols in ``measures.py``; the moderator may
also be ``age`` (a continuous covariate).

Usage::

    python scripts/within_child_interaction_check.py                       # L x age -> W (LRP73)
    python scripts/within_child_interaction_check.py --moderator E         # L x E -> W (LRP71)
    python scripts/within_child_interaction_check.py --moderator B --outcome N  # L x B -> decoding (LRP72)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    logit_safe,
)


def _out_dir() -> str:
    return str(_paths.output_root() / "interaction_screen")


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)


def _interaction_row(fit, term: str = "xL:xM") -> dict:
    return {
        "coef": float(fit.params[term]),
        "se": float(fit.bse[term]),
        "t": float(fit.tvalues[term]),
        "p": float(fit.pvalues[term]),
    }


def run_check(mechanism: str, moderator: str, outcome: str) -> pd.DataFrame:
    # Need the mechanism + outcome (+ moderator if it is a measure) in prepared.
    symbols = {mechanism, outcome}
    mod_is_age = moderator.lower() in {"age", "a"}
    if not mod_is_age:
        symbols.add(moderator)
    prepared = load_and_prepare(phase_mode="all", outcomes=tuple(symbols))

    y_post = prepared.post_counts[outcome]
    l_post = prepared.post_counts[mechanism]
    keep = ~(np.isnan(y_post) | np.isnan(l_post))
    if not mod_is_age:
        keep = keep & ~np.isnan(prepared.post_counts[moderator])

    y = logit_safe(y_post[keep], MEASURES[outcome].n_trials)
    xL = _zscore(logit_safe(l_post[keep], MEASURES[mechanism].n_trials))
    if mod_is_age:
        xM = _zscore(prepared.A_months[keep])
        mod_label = "age"
    else:
        xM = _zscore(logit_safe(prepared.post_counts[moderator][keep], MEASURES[moderator].n_trials))
        mod_label = moderator
    subject = prepared.subject_ids[keep]

    df = pd.DataFrame({"y": y, "xL": xL, "xM": xM, "subject": subject.astype(str)})
    # Within-child (double-)demeaned components and their product — the genuine
    # within-child interaction regressor (estimator 4).
    df["xL_dm"] = df.groupby("subject")["xL"].transform(lambda s: s - s.mean())
    df["xM_dm"] = df.groupby("subject")["xM"].transform(lambda s: s - s.mean())
    df["xLM_dm"] = df["xL_dm"] * df["xM_dm"]
    groups = df["subject"]
    n_obs, n_children = len(df), df["subject"].nunique()

    pooled_naive = smf.ols("y ~ xL * xM", data=df).fit()
    pooled_cluster = smf.ols("y ~ xL * xM", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    within_fe = smf.ols("y ~ xL * xM + C(subject)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    within_demeaned = smf.ols(
        "y ~ xL_dm + xM_dm + xLM_dm + C(subject)", data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": groups})

    rows = []
    for name, fit, term in [
        ("pooled_naive", pooled_naive, "xL:xM"),
        ("pooled_cluster", pooled_cluster, "xL:xM"),
        ("within_child_fe_raw", within_fe, "xL:xM"),
        ("within_child_demeaned", within_demeaned, "xLM_dm"),
    ]:
        r = _interaction_row(fit, term=term)
        r.update({"estimator": name, "n_obs": n_obs, "n_children": n_children})
        rows.append(r)
    out = pd.DataFrame(rows)[
        ["estimator", "coef", "se", "t", "p", "n_obs", "n_children"]
    ]
    out.attrs["label"] = f"{mechanism} x {mod_label} -> {outcome}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", default="L", help="Mechanism symbol (default L).")
    parser.add_argument(
        "--moderator", default="age", help="Moderator: 'age' or a measure symbol (E, B, ...)."
    )
    parser.add_argument("--outcome", default="W", help="Outcome symbol (default W).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root for this run (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR). Default: repo-local output/."
        ),
    )
    args = parser.parse_args()
    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")

    out = run_check(args.mechanism, args.moderator, args.outcome)
    label = out.attrs["label"]
    print(f"\nWithin-child interaction check — {label}\n")
    print(out.to_string(index=False))

    out_dir = _out_dir()
    os.makedirs(out_dir, exist_ok=True)
    slug = f"{args.mechanism}x{args.moderator}_{args.outcome}".lower()
    path = os.path.join(out_dir, f"within_child_{slug}.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")

    naive = out.loc[out.estimator == "pooled_naive", "t"].iloc[0]
    within = out.loc[out.estimator == "within_child_demeaned", "t"].iloc[0]
    print(
        f"\nInteraction |t|: pooled-naive={abs(naive):.2f} -> within-child "
        f"(double-demeaned)={abs(within):.2f}. "
        + (
            "Collapses within-child => between-child confound."
            if abs(within) < 2 <= abs(naive)
            else "Interpret per the table."
        )
    )


if __name__ == "__main__":
    main()
