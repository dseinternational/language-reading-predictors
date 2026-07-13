# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pool the treatment x baseline interaction across the gain-factor outcomes (#228 item 9).

Reads each gain-factor model's ``gamma_int_trt_own`` posterior (mean + sd) from its
fitted ``diagnostics.csv`` and runs a Bayesian random-effects meta-analysis
(:mod:`language_reading_predictors.statistical_models.pooled_moderation`) to give one
posterior for "does the intervention flatten the baseline gradient, on average across
skills?". Writes ``pooled_moderation_summary.csv``, ``pooled_moderation_by_outcome.csv``
and a forest plot.

Second-stage caveat: the outcomes share the same children, so the pooled interval is
somewhat optimistic; the joint-model follow-on removes it (see the companion note).

Usage::

    python scripts/pooled_moderation.py [--suffix reporting] [--out DIR] [--draws N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from language_reading_predictors.statistical_models.environment import (
    STAT_OUTPUT_DIR,
)
from language_reading_predictors.statistical_models.pooled_moderation import (
    run_pooled_moderation,
)

_COEF = "gamma_int_trt_own"


def _read_interactions(models_dir: str, suffix: str):
    """Collect (label, mean, sd) for the trt x own interaction from each gf model."""
    rows = []
    for d in sorted(glob.glob(os.path.join(models_dir, f"lrp-rli-gf-0*-{suffix}"))):
        diag = os.path.join(d, "diagnostics.csv")
        if not os.path.exists(diag):
            continue
        df = pd.read_csv(diag)
        name_col = df.columns[0]
        hit = df[df[name_col].astype(str) == _COEF]
        if not len(hit):
            continue
        label = None
        cfg = os.path.join(d, "config.json")
        if os.path.exists(cfg):
            c = json.load(open(cfg))
            label = c.get("outcome_symbol") or c.get("outcome")
        label = label or os.path.basename(d).replace("lrp-rli-", "").replace(
            f"-{suffix}", ""
        )
        rows.append((label, float(hit["mean"].iloc[0]), float(hit["sd"].iloc[0])))
    return rows


def _forest(pooled: pd.DataFrame, by: pd.DataFrame, path: str) -> None:
    mu = pooled[pooled["term"].str.startswith("pooled")].iloc[0]
    y = range(len(by))
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(by) + 1.5))
    ax.axvspan(mu.lo, mu.hi, color="#d62728", alpha=0.12, label="pooled 95% CrI")
    ax.axvline(mu["median"], color="#d62728", lw=1.5, label="pooled median")
    ax.axvline(0.0, color="#888", lw=0.8, ls=":")
    ax.errorbar(
        by["raw_mean"], [i + 0.15 for i in y], xerr=by["raw_se"], fmt="o",
        color="#999", ms=4, label="per-outcome (raw ± sd)",
    )
    ax.errorbar(
        by["shrunk_median"], [i - 0.15 for i in y],
        xerr=[by["shrunk_median"] - by["shrunk_lo"], by["shrunk_hi"] - by["shrunk_median"]],
        fmt="s", color="#1f77b4", ms=4, label="per-outcome (shrunken)",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(by["outcome"])
    ax.set_xlabel("treatment × own-baseline interaction (logit)")
    ax.set_title("Pooled treatment × baseline moderation (#228 item 9)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default=os.path.join(STAT_OUTPUT_DIR, "models"))
    ap.add_argument("--suffix", default="reporting", help="fit suffix (reporting/dev)")
    ap.add_argument(
        "--out", default=os.path.join(STAT_OUTPUT_DIR, "analyses", "pooled_moderation")
    )
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = _read_interactions(args.models_dir, args.suffix)
    if len(rows) < 2:
        raise SystemExit(
            f"found {len(rows)} gain-factor '{_COEF}' estimates under "
            f"{args.models_dir} (suffix '{args.suffix}'); need >= 2. Fit the "
            "gain-factor family first."
        )
    labels = [r[0] for r in rows]
    effects = [r[1] for r in rows]
    ses = [r[2] for r in rows]

    pooled, by, _ = run_pooled_moderation(
        effects, ses, labels, draws=args.draws, tune=args.draws, seed=args.seed
    )
    os.makedirs(args.out, exist_ok=True)
    pooled.to_csv(os.path.join(args.out, "pooled_moderation_summary.csv"), index=False)
    by.to_csv(os.path.join(args.out, "pooled_moderation_by_outcome.csv"), index=False)
    _forest(pooled, by, os.path.join(args.out, "pooled_moderation_forest"))

    mu = pooled[pooled["term"].str.startswith("pooled")].iloc[0]
    print(f"Pooled {len(labels)} outcomes ({', '.join(labels)}).")
    print(
        f"Pooled trt×baseline interaction: {mu['median']:+.3f} "
        f"(95% CrI {mu.lo:+.3f} to {mu.hi:+.3f}); P(<0) = {mu.prob_lt_0:.3f}."
    )
    print(f"Wrote outputs to {args.out}")


if __name__ == "__main__":
    main()
