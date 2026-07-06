# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cross-model comparison report for the statistical models.

Run after the models have been fitted (``python
scripts/fit_statistical_model.py all``). Produces, under
``output/statistical_models/comparison/``:

- ``itt_vs_joint_tau.csv`` — per-outcome tau from the LRPITT single-outcome
  fits alongside tau_k from the LRPITT12 joint (consistency check), on the shared
  (non-floored) outcomes W, R, E, L, B.
- ``tau_forest.png`` — forest plot of the LRPITT12 joint taus, overlaid with
  the LRPITT single-outcome taus on those shared outcomes.
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

from language_reading_predictors import paths as _paths
from language_reading_predictors.statistical_models.measures import (
    MEASURES,
    ROPE_DELTA_PROB,
)
from language_reading_predictors.statistical_models.preprocessing import (
    load_and_prepare,
    logit_safe,
)

# Heavily-floored outcomes whose PRIMARY ITT estimand is the binary off-floor
# effect (a risk difference), not the graded logit tau shown in the joint. Any of
# these appearing in a forest/CSV of graded taus is flagged so the artefact does
# not misrepresent it.
FLOORED_SYMBOLS: frozenset[str] = frozenset(ROPE_DELTA_PROB)


# Single-outcome ITT models (LRPITT suite, #119) overlaid on the LRPITT12 joint, on
# the outcomes the joint also carries. The floored outcomes P (lrpitt09) and N
# (lrpitt11) are excluded from the graded overlay: their PRIMARY estimand is the
# binary off-floor effect, read from their own reports rather than compared to the
# joint's graded tau. F/T have no standalone ITT in the suite.
ITT_IDS: list[tuple[str, str]] = [
    ("lrpitt10", "W"),
    ("lrpitt05", "R"),
    ("lrpitt06", "E"),
    ("lrpitt07", "L"),
    ("lrpitt08", "B"),
]
MECH_IDS: list[tuple[str, str]] = [("lrp56", "R"), ("lrp57", "E"), ("lrp58", "L")]
JOINT_ID = "lrpitt12"

# Mechanism models compared by PSIS-LOO: the LRP58 baseline (L -> W) against the
# interaction extensions on the *same* word-reading outcome/rows (a like-for-like
# elpd comparison). LRP70 is intentionally excluded: it was repurposed to a
# ``growth``-kind joint growth-curve model (posterior dims child/wave/outcome, no
# ``obs_id``), so it is a different dataset and a growth-vs-mechanism LOO would be
# the cross-dataset comparison ``_loo_compare`` forbids. The celf mechanism model
# is deferred pending a DAG review (see ``lrp71.py``).
LOO_COMPARE_IDS: list[str] = ["lrp58", "lrp71"]

# Phonics route (LRP72): the interaction model vs its no-interaction baseline,
# same decoding outcome — a clean nested PSIS-LOO test of the L x B interaction.
# NOT comparable to the LOO_COMPARE_IDS set (different outcome: decoding vs W).
PHONICS_LOO_IDS: list[str] = ["lrp72", "lrp72base"]

# Age moderation (LRP73): interaction vs no-interaction baseline, same word-reading
# outcome — a clean nested PSIS-LOO test of the L x age interaction.
AGE_LOO_IDS: list[str] = ["lrp73", "lrp73base"]

# Dose-response (LRP77, #104 Phase 2): the period-varying dose model vs its
# pooled-dose comparator, same word-reading outcome and rows — a nested PSIS-LOO
# test of whether the dose-gain slope varies by period.
DOSE_LOO_IDS: list[str] = ["lrp77", "lrp77base"]

# DiD period-resolved letter-sound dose (LRPDID07, #135): the period-varying dose
# model vs its pooled-dose comparator, same letter-sound outcome and rows — a
# nested PSIS-LOO test of whether the L dose-gain slope varies by period (the
# DAG-clean DiD analogue of the LRP77 word-reading test; never conditions on the
# IS collider attend_cumul).
DID_DOSE_LOO_IDS: list[str] = ["lrpdid07", "lrpdid07base"]


def _run_dir(model_id: str, config: str) -> str:
    return os.path.join(str(_paths.stat_models_dir()), f"{model_id}-{config}")


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
                "config": config,
                "outcome": outcome,
                "source": model_id,
                "floored": outcome in FLOORED_SYMBOLS,
                "tau_median": df["tau_logit_median"].iloc[0],
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
                "config": config,
                "outcome": row["outcome"],
                "source": JOINT_ID,
                "floored": row["outcome"] in FLOORED_SYMBOLS,
                "tau_median": row["tau_median"],
                "tau_lo": row["tau_lo"],
                "tau_hi": row["tau_hi"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tau forest plot
# ---------------------------------------------------------------------------


def tau_forest(config: str, out_path: str) -> bool:
    """Forest plot of the LRPITT12 joint taus, overlaid with the LRPITT single-outcome
    fits on the shared (non-floored) outcomes."""
    joint_path = os.path.join(_run_dir(JOINT_ID, config), "tau_summary.csv")
    if not os.path.exists(joint_path):
        return False  # joint run not fitted — main() reports the skip
    joint = pd.read_csv(joint_path)
    uni: dict[str, tuple[float, float, float]] = {}
    for model_id, outcome in ITT_IDS:
        p = os.path.join(_run_dir(model_id, config), "tau_summary.csv")
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        uni[outcome] = (
            float(df["tau_logit_median"].iloc[0]),
            float(df["tau_logit_lo"].iloc[0]),
            float(df["tau_logit_hi"].iloc[0]),
        )

    outcomes = list(joint["outcome"].values)
    y = np.arange(len(outcomes))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        joint["tau_median"].values,
        y,
        xerr=[
            joint["tau_median"].values - joint["tau_lo"].values,
            joint["tau_hi"].values - joint["tau_median"].values,
        ],
        fmt="o",
        color="#1f77b4",
        label="LRPITT12 (joint)",
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
            label="LRPITT (single-outcome)",
            capsize=3,
        )
    ax.axvline(0.0, color="k", lw=0.75, ls="--")
    ax.set_yticks(y)
    # Flag heavily-floored outcomes (P/N): the graded logit tau shown here is NOT
    # their primary estimand — that is the binary off-floor risk difference, read
    # from their own reports. Marking them keeps the forest from misrepresenting P.
    floored_present = [s for s in outcomes if s in FLOORED_SYMBOLS]
    ax.set_yticklabels(
        [f"{s} †" if s in FLOORED_SYMBOLS else s for s in outcomes]
    )
    if floored_present:
        ax.text(
            0.99,
            -0.14,
            "† floored outcome — graded τ shown; PRIMARY estimand is the "
            "binary off-floor risk difference (see model report)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color="#555555",
        )
    ax.invert_yaxis()
    ax.set_xlabel(r"$\tau$ (logit scale, coefficient on $G=1$ = intervention; positive = benefit)")
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
        #
        # CAVEAT: the factory ALSO drops rows with missing *confounder* post-scores
        # (lrp57 adjusts for R; lrp58 for E and R). This keep-mask does not model
        # that, so if confounder-only missingness ever occurs the reconstructed
        # length will not match the trace and the guard below skips the model. That
        # skip is a *silent drop of the model from the persisted forest/CSV* — the
        # warning is deliberately explicit about that so a reader notices a missing
        # row rather than assuming the model was never fitted.
        mech_post = prepared.post_counts[sym]
        w_post = prepared.post_counts["W"]
        keep = ~(np.isnan(mech_post) | np.isnan(w_post))
        mech_logit = logit_safe(mech_post[keep], MEASURES[sym].n_trials)

        if mech_logit.shape[0] != trace.posterior.sizes["obs_id"]:
            # Skip rather than silently misalign. Most likely cause: the factory
            # dropped rows for missing confounder post-scores that this simplified
            # keep-mask (outcome + mechanism only) does not account for.
            print(
                f"[warn] {model_id}: reconstructed mech_logit size "
                f"({mech_logit.shape[0]}) != trace obs_id size "
                f"({trace.posterior.sizes['obs_id']}) — likely confounder-only "
                "missingness the keep-mask does not model. DROPPING this model "
                "from the persisted mechanism forest AND its CSV (it will be "
                "absent from the artefact, not merely un-plotted)."
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

    # Also write the underlying numbers. Record the config so a dev-config rerun
    # does not silently masquerade as the reporting-config artefact.
    pd.DataFrame(
        {
            "config": config,
            "model": labels,
            "slope_mean": means,
            "slope_lo": los,
            "slope_hi": his,
        }
    ).to_csv(out_path.replace(".png", ".csv"), index=False)
    return True


# ---------------------------------------------------------------------------
# Mechanism LOO comparison (LRP58 baseline vs interaction models)
# ---------------------------------------------------------------------------


def _loo_compare(ids: list[str], config: str, out_path: str) -> bool:
    """Write ``az.compare`` over the fitted models in ``ids`` (LOO).

    Loads every model in ``ids`` that has a trace with a ``log_likelihood``
    group. ``az.compare`` is only a like-for-like elpd-difference when the models
    share the same observations, so this asserts equal ``obs_id`` sizes; if they
    differ it falls back to a per-model ``elpd_loo`` table rather than a
    misleading delta. Returns False if fewer than two models are available.
    """
    traces: dict[str, az.InferenceData] = {}
    for mid in ids:
        nc = os.path.join(_run_dir(mid, config), "trace.nc")
        if not os.path.exists(nc):
            continue
        t = az.from_netcdf(nc)
        # arviz 1.x returns a DataTree whose ``.groups`` is a tuple of paths
        # like "/log_likelihood" (0.x exposed a ``.groups()`` method of bare
        # names) — normalise to leaf names for the membership test.
        group_names = {g.rstrip("/").split("/")[-1] for g in t.groups}
        if "log_likelihood" not in group_names:
            print(f"[warn] {mid}: trace has no log_likelihood group; skipping.")
            continue
        traces[mid] = t

    if len(traces) < 2:
        return False

    # A trace whose posterior has no ``obs_id`` dim is not a single-outcome
    # observation model (e.g. a repurposed ``growth``-kind fit with child/wave/
    # outcome dims) — it cannot participate in a shared-observation LOO, so drop
    # it with a clear warning rather than crashing on the KeyError.
    sizes: dict[str, int] = {}
    for mid, t in list(traces.items()):
        if "obs_id" not in t.posterior.sizes:
            print(
                f"[warn] {mid}: posterior has no 'obs_id' dim "
                f"(dims={tuple(t.posterior.sizes)}); not an observation-level "
                "model — skipping from this LOO comparison."
            )
            del traces[mid]
            continue
        sizes[mid] = t.posterior.sizes["obs_id"]

    if len(traces) < 2:
        return False

    if len(set(sizes.values())) == 1:
        cmp = az.compare(traces)  # ic="loo" by default
        cmp = cmp.copy()
        cmp.insert(0, "config", config)  # record the tier that produced the row
        cmp.to_csv(out_path)
        return True

    # Row sets differ — a shared-observation elpd_diff is not valid.
    print(
        f"[warn] models do not share observations ({sizes}); "
        "writing per-model elpd_loo instead of az.compare deltas."
    )
    rows = []
    for mid, t in traces.items():
        loo = az.loo(t)
        rows.append(
            {
                "config": config,
                "model": mid,
                "n_obs": sizes[mid],
                "elpd_loo": float(loo.elpd),
                "se": float(loo.se),
                "p_loo": float(loo.p),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return True


def mechanism_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of the LRP58 baseline against its interaction extensions."""
    return _loo_compare(LOO_COMPARE_IDS, config, out_path)


def phonics_route_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of LRP72 against its no-interaction baseline (isolates L x B)."""
    return _loo_compare(PHONICS_LOO_IDS, config, out_path)


def age_moderation_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of LRP73 against its no-interaction baseline (isolates L x age)."""
    return _loo_compare(AGE_LOO_IDS, config, out_path)


def dose_response_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of LRP77 against its pooled-dose comparator (does dose vary by period?)."""
    return _loo_compare(DOSE_LOO_IDS, config, out_path)


def did_dose_loo_compare(config: str, out_path: str) -> bool:
    """LOO comparison of LRPDID07 vs its pooled comparator (does the L dose vary by period?)."""
    return _loo_compare(DID_DOSE_LOO_IDS, config, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dev")
    parser.add_argument(
        "--out",
        default=None,
        help="Comparison output dir (default: <output-root>/statistical_models/comparison).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root for this run (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR); the relative layout is unchanged. Default: "
            "repo-local output/."
        ),
    )
    args = parser.parse_args()

    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")
    args.out = args.out or str(_paths.stat_comparison_dir())
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

    loo_compare_path = os.path.join(args.out, "mechanism_loo_compare.csv")
    if mechanism_loo_compare(args.config, loo_compare_path):
        print(f"Wrote {loo_compare_path}")
    else:
        print("Skipping mechanism LOO compare: fewer than two mechanism runs available.")

    phonics_path = os.path.join(args.out, "phonics_route_loo_compare.csv")
    if phonics_route_loo_compare(args.config, phonics_path):
        print(f"Wrote {phonics_path}")
    else:
        print("Skipping phonics-route LOO compare: LRP72 / LRP72base runs missing.")

    age_path = os.path.join(args.out, "age_moderation_loo_compare.csv")
    if age_moderation_loo_compare(args.config, age_path):
        print(f"Wrote {age_path}")
    else:
        print("Skipping age-moderation LOO compare: LRP73 / LRP73base runs missing.")

    dose_path = os.path.join(args.out, "dose_response_loo_compare.csv")
    if dose_response_loo_compare(args.config, dose_path):
        print(f"Wrote {dose_path}")
    else:
        print("Skipping dose-response LOO compare: LRP77 / LRP77base runs missing.")

    did_dose_path = os.path.join(args.out, "did_dose_loo_compare.csv")
    if did_dose_loo_compare(args.config, did_dose_path):
        print(f"Wrote {did_dose_path}")
    else:
        print("Skipping DiD L-dose LOO compare: LRPDID07 / LRPDID07base runs missing.")


if __name__ == "__main__":
    main()
