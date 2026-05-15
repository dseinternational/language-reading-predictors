# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cross-model comparison report for LRP52-LRP58.

Run after all seven models have been fitted (``python
scripts/fit_statistical_model.py all``). Produces, under
``output/statistical_models/comparison/``:

- ``itt_vs_joint_tau.csv`` — per-outcome tau from LRP52/53/54 univariate
  fits alongside tau_k from LRP55 (consistency check).
- ``tau_forest.png`` — forest plot of the eight LRP55 taus, overlaid with
  the three univariate LRP52/53/54 taus on the shared outcomes (W, R, E).
- ``mechanism_forest.png`` — forest plot of the marginal slope of each
  mechanism GP (LRP56 R->W, LRP57 E->W, LRP58 L->W). Slopes are computed
  from each model's actual posterior ``f_mech`` samples on its own
  ``logit(mech_post)`` grid (not a shared dummy grid).

Requires ``trace.nc`` and ``tau_summary.csv`` under
``output/statistical_models/models/{model_id}-{config}/`` for each model.
"""

from __future__ import annotations

import argparse
import os

import arviz as az
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.environment import (
    STAT_OUTPUT_DIR,
)
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    logit_safe,
)


ITT_IDS: list[tuple[str, str]] = [("lrp52", "W"), ("lrp53", "R"), ("lrp54", "E")]
MECH_IDS: list[tuple[str, str]] = [("lrp56", "R"), ("lrp57", "E"), ("lrp58", "L")]
JOINT_ID = "lrp55"


def _run_dir(model_id: str, config: str) -> str:
    return os.path.join(STAT_OUTPUT_DIR, "models", f"{model_id}-{config}")


# ---------------------------------------------------------------------------
# ITT vs joint
# ---------------------------------------------------------------------------


def build_itt_vs_joint(config: str) -> pd.DataFrame | None:
    rows: list[dict] = []
    for model_id, outcome in ITT_IDS:
        tau_path = os.path.join(_run_dir(model_id, config), "tau_summary.csv")
        if not os.path.exists(tau_path):
            return None
        df = pd.read_csv(tau_path)
        rows.append(
            {
                "outcome": outcome,
                "source": model_id,
                "tau_mean": df["tau_logit_mean"].iloc[0],
                "tau_lo": df["tau_logit_lo"].iloc[0],
                "tau_hi": df["tau_logit_hi"].iloc[0],
            }
        )
    joint_path = os.path.join(_run_dir(JOINT_ID, config), "tau_summary.csv")
    if not os.path.exists(joint_path):
        return None
    joint = pd.read_csv(joint_path)
    for _, row in joint.iterrows():
        rows.append(
            {
                "outcome": row["outcome"],
                "source": JOINT_ID,
                "tau_mean": row["tau_mean"],
                "tau_lo": row["tau_lo"],
                "tau_hi": row["tau_hi"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tau forest plot
# ---------------------------------------------------------------------------


def tau_forest(config: str, out_path: str) -> bool:
    """Forest plot of LRP55's eight taus, overlaid with LRP52/53/54 univariates."""
    joint = pd.read_csv(os.path.join(_run_dir(JOINT_ID, config), "tau_summary.csv"))
    uni: dict[str, tuple[float, float, float]] = {}
    for model_id, outcome in ITT_IDS:
        p = os.path.join(_run_dir(model_id, config), "tau_summary.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        uni[outcome] = (
            float(df["tau_logit_mean"].iloc[0]),
            float(df["tau_logit_lo"].iloc[0]),
            float(df["tau_logit_hi"].iloc[0]),
        )

    outcomes = list(joint["outcome"].values)
    y = np.arange(len(outcomes))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        joint["tau_mean"].values,
        y,
        xerr=[
            joint["tau_mean"].values - joint["tau_lo"].values,
            joint["tau_hi"].values - joint["tau_mean"].values,
        ],
        fmt="o",
        color="#1f77b4",
        label="LRP55 (joint)",
        capsize=3,
    )
    # Univariate overlay, offset vertically for readability.
    uni_y = []
    uni_mean = []
    uni_lo = []
    uni_hi = []
    for k, s in enumerate(outcomes):
        if s in uni:
            m, lo, hi = uni[s]
            uni_y.append(y[k] - 0.25)
            uni_mean.append(m)
            uni_lo.append(m - lo)
            uni_hi.append(hi - m)
    if uni_y:
        ax.errorbar(
            uni_mean,
            uni_y,
            xerr=[uni_lo, uni_hi],
            fmt="s",
            color="#ff7f0e",
            label="LRP52/53/54 (univariate)",
            capsize=3,
        )
    ax.axvline(0.0, color="k", lw=0.75, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(outcomes)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\tau$ (logit scale, coefficient on $G=1$ = control)")
    ax.set_title("Treatment effect by outcome")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


# ---------------------------------------------------------------------------
# Mechanism forest plot — real-grid slopes
# ---------------------------------------------------------------------------


def _mechanism_slope_distribution(
    trace: xr.DataTree, mech_logit: np.ndarray
) -> np.ndarray:
    """Posterior draws of the average slope of ``f_mech`` over ``mech_logit``.

    Uses ``np.gradient`` per draw along the sorted grid and averages across
    the grid, returning a 1-D array of per-draw mean slopes. Input arrays
    must be aligned (one row of ``f_mech`` per ``mech_logit`` entry).
    """
    if "f_mech" not in trace.posterior:
        raise ValueError("Trace has no f_mech variable.")

    f = trace.posterior["f_mech"].stack(sample=("chain", "draw")).values  # (n, S)

    order = np.argsort(mech_logit)
    x = mech_logit[order]
    f_ord = f[order]

    # Drop rows with duplicate x (np.gradient requires monotone x).
    keep = np.concatenate([[True], np.diff(x) > 0])
    x = x[keep]
    f_ord = f_ord[keep]

    # per-draw gradient over the unique grid, then mean.
    grad = np.gradient(f_ord, x, axis=0)  # (n, S)
    slopes = grad.mean(axis=0)  # (S,)
    return slopes


def mechanism_forest(config: str, out_path: str) -> bool:
    labels: list[str] = []
    means: list[float] = []
    los: list[float] = []
    his: list[float] = []

    # Shared prepared (all phases) so we can rebuild the mech_logit vector per model.
    prepared = load_and_prepare(phase_mode="all")

    for model_id, sym in MECH_IDS:
        nc = os.path.join(_run_dir(model_id, config), "trace.nc")
        if not os.path.exists(nc):
            return False
        trace = az.from_netcdf(nc)
        if "f_mech" not in trace.posterior:
            continue

        # Rebuild the logit vector used at fit time. Rows with missing
        # outcome_post or mechanism_post are dropped by the mechanism factory,
        # so filter prepared to rows where both W_post and mech_post are
        # observed (and match the trace's obs_id length).
        mech_post = prepared.post_counts[sym]
        w_post = prepared.post_counts["W"]
        keep = ~(np.isnan(mech_post) | np.isnan(w_post))
        mech_logit = logit_safe(mech_post[keep], MEASURES[sym].n_trials)

        if mech_logit.shape[0] != trace.posterior.sizes["obs_id"]:
            # Fallback: skip rather than silently misalign.
            print(
                f"[warn] {model_id}: mech_logit size ({mech_logit.shape[0]}) "
                f"!= trace obs_id size ({trace.posterior.sizes['obs_id']}); "
                "skipping forest entry."
            )
            continue

        slopes = _mechanism_slope_distribution(trace, mech_logit)
        means.append(float(np.mean(slopes)))
        los.append(float(np.quantile(slopes, 0.025)))
        his.append(float(np.quantile(slopes, 0.975)))
        labels.append(f"{model_id} ({sym}->W)")

    if not labels:
        return False

    y = np.arange(len(labels))
    plt.figure(figsize=(7, 3.5))
    plt.errorbar(
        means,
        y,
        xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
        fmt="o",
        color="#1f77b4",
        capsize=3,
    )
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.axvline(0.0, color="k", lw=0.75, ls="--")
    plt.xlabel("Mean slope of $f^{\\mathrm{mech}}$ (logit scale)")
    plt.title("Mechanism-model average slopes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Also write the underlying numbers.
    pd.DataFrame(
        {"model": labels, "slope_mean": means, "slope_lo": los, "slope_hi": his}
    ).to_csv(out_path.replace(".png", ".csv"), index=False)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dev")
    parser.add_argument("--out", default=os.path.join(STAT_OUTPUT_DIR, "comparison"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    itt_joint = build_itt_vs_joint(args.config)
    if itt_joint is not None:
        path = os.path.join(args.out, "itt_vs_joint_tau.csv")
        itt_joint.to_csv(path, index=False)
        print(f"Wrote {path}")
    else:
        print("Skipping ITT-vs-joint comparison: one or more runs missing.")

    tau_forest_path = os.path.join(args.out, "tau_forest.png")
    if tau_forest(args.config, tau_forest_path):
        print(f"Wrote {tau_forest_path}")
    else:
        print("Skipping tau forest: joint run missing.")

    mech_forest_path = os.path.join(args.out, "mechanism_forest.png")
    if mechanism_forest(args.config, mech_forest_path):
        print(f"Wrote {mech_forest_path}")
    else:
        print("Skipping mechanism forest: one or more mechanism runs missing.")


if __name__ == "__main__":
    main()
