# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Period-resolved / intervention-aligned gradient-boosting diagnostic (#104, Phase 1).

The gain (change-score) models are near-noise when gains are *pooled* across
all three periods (and
, landed in #102). This script settles a
single pre-committed question: **is that weakness an artefact of pooling
heterogeneous periods, or are gains genuinely near-unpredictable regardless of
resolution?**

For each gain model (registry models whose ``target_var`` ends in ``_gain``) it
takes the model's *committed pruned* predictor set (the 3-8 predictors encoded
in ``models/lrpNN.py`` ``selection_steps`` — **not** the 32-34-predictor full
sets, which n ~ 50 would overfit) and augments it with the study-design terms
needed to read the dose and group contrasts in every stratum:

- ``group``       - the randomised contrast (immediate vs waitlist);
- ``attend``      - the per-period intervention dose (sessions that period);
- ``attend_cumul``- cumulative prior dose, i.e. a dose-stage signal, so a
  dose-stage effect is not misread as a period effect;
- ``age``         - the maturation covariate (groups differ in age by period);
- ``period``      - the period index (skipped if the set already keeps ``time``,
  which is numerically identical).

Each augmented model is then fitted under five stratifications:

1. ``all``          - the all-periods pool (the #102 baseline, for reference);
2. ``period1/2/3``  - one period at a time (n ~ 50 each);
3. ``intervention`` - the intervention-aligned pool (``on_intervention``:
   immediate periods 1-3 + waitlist periods 2-3, n ~ 131).

It reuses the project's own ``EstimatorPipeline`` (``LGBMPipeline``) machinery
and each model's committed hyperparameters — CV, pooled out-of-fold R², and
group-aware out-of-fold permutation importance — with **GroupKFold by
``subject_id``** throughout (a child is never split across folds). A fixed
``k = 10`` is used for every stratum so the comparison is internally consistent
and the n ~ 50 per-period strata still leave several rows per validation fold
(near-leave-one-subject-out would give degenerate 1-row folds). Absolute R²
therefore differs from the #102 near-LOSO numbers; the comparison of interest is
*relative* across strata.

Per stratum it emits: pooled out-of-fold R² (the stable summary) plus per-fold
R² mean/std (so the instability is visible); the ``attend`` (dose) and ``group``
terms' permutation importance + SHAP direction sign; and n.

Pre-committed questions (answered, not an open importance scan):

1. Does resolving by period or intervention status **materially raise** pooled
   R² over the all-periods pool in any single stratum?
2. Does the ``attend`` (dose) contribution **concentrate in period 1** or vary
   by period?
3. Does the ``group`` contribution concentrate in period 1 (its randomised
   contrast)?

Caveat surfaced for interpretation: in period 1 ``attend`` and ``group`` are
near-collinear (the waitlist controls have ``attend == 0``); in period 2
``attend_cumul`` and ``group`` are near-collinear (only the immediate group has
prior dose). Permutation importance splits between collinear terms, so read the
period-1 ``attend``/``group`` split together, not as independent magnitudes.

Outputs (under ``output/replication/period_resolved/``, gitignored):
``results.csv`` (tidy long table) and ``summary.md`` (pivots + question flags),
plus per-(model, stratum) scratch artefacts. The committed decision-gate note is
authored separately under ``notes/``.

Usage
-----

::

    python scripts/period_resolved_gb_diagnostic.py
    python scripts/period_resolved_gb_diagnostic.py --models lrpgbg12 lrpgbg05
    python scripts/period_resolved_gb_diagnostic.py --perm-repeats 30
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

# Headless rendering: the reused pipeline imports matplotlib.pyplot at module
# import time and writes a permutation-importance boxplot per fit.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rich import print  # noqa: E402
from scipy import stats  # noqa: E402

import language_reading_predictors.data_utils as data_utils  # noqa: E402
import language_reading_predictors.models  # noqa: E402,F401  (populate MODELS)
from language_reading_predictors.data_variables import Variables as V  # noqa: E402
from language_reading_predictors.models.base_pipeline import ESTIMATOR_STEP  # noqa: E402
from language_reading_predictors.models.common import RunConfig  # noqa: E402
from language_reading_predictors.models.registry import MODELS  # noqa: E402

# ── configuration ─────────────────────────────────────────────────────────

DIAGNOSTIC_K = 10
"""GroupKFold splits, fixed across strata for comparability and to keep
several rows per validation fold at n ~ 50 (LOSO would give 1-row folds)."""

PERM_REPEATS = 50
"""Permutation-importance repeats per fold (the reporting default)."""

# Study-design covariates ensured present in every augmented set. ``period`` is
# handled separately (skipped when ``time`` is already kept — it is identical).
DESIGN_TERMS = [V.GROUP, V.ATTEND, V.ATTEND_CUMUL, V.AGE]

STRATA = ["all", "period1", "period2", "period3", "intervention"]
PERIOD_STRATA = ["period1", "period2", "period3"]

# Per-stratum n sanity ranges (anchored on the 11 gain targets; see
#  and the Task-A counts).
N_RANGES = {
    "all": (148, 165),
    "period1": (45, 56),
    "period2": (45, 56),
    "period3": (45, 56),
    "intervention": (122, 140),
}

OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "replication"
    / "period_resolved"
)


# ── predictor augmentation + stratum masks ─────────────────────────────────


def augment_predictors(pruned: list[str]) -> list[str]:
    """Committed pruned set + the study-design terms (added only if missing)."""
    out = list(pruned)
    for term in DESIGN_TERMS:
        if term not in out:
            out.append(term)
    if V.PERIOD not in out and V.TIME not in out:
        out.append(V.PERIOD)
    return out


def stratum_mask(df: pd.DataFrame, label: str) -> np.ndarray:
    """Boolean row mask for a stratum, aligned with ``df``."""
    if label == "all":
        mask = pd.Series(True, index=df.index)
    elif label == "intervention":
        mask = df[V.ON_INTERVENTION]
    elif label.startswith("period"):
        mask = df[V.PERIOD] == int(label[-1])
    else:  # pragma: no cover - guarded by STRATA
        raise ValueError(f"unknown stratum {label!r}")
    return mask.astype("bool").to_numpy()


# ── SHAP direction ─────────────────────────────────────────────────────────


def shap_signs(estimator, X: pd.DataFrame, features: list[str]) -> dict[str, str]:
    """Spearman sign of (feature value, SHAP value) for the named features.

    Mirrors ``EstimatorPipeline.shap_direction_diagnostics`` but only for the
    two reported terms. Returns a signed correlation string, ``"const"`` for a
    within-stratum constant feature, or ``"n/a"``.
    """
    import shap

    cols = list(X.columns)
    explainer = shap.TreeExplainer(estimator)
    shap_vals = explainer.shap_values(X)
    out: dict[str, str] = {}
    for feat in features:
        if feat not in cols:
            out[feat] = "n/a"
            continue
        vals = X[feat].to_numpy(dtype="float64")
        finite = ~np.isnan(vals)
        if finite.sum() < 3 or np.unique(vals[finite]).size < 2:
            out[feat] = "const"
            continue
        shap_col = shap_vals[:, cols.index(feat)]
        if np.unique(shap_col[finite]).size < 2:
            # feature present but unused by the model -> no directional effect
            out[feat] = "0"
            continue
        rho = float(stats.spearmanr(vals[finite], shap_col[finite]).statistic)
        if np.isnan(rho) or abs(rho) < 1e-6:
            out[feat] = "0"
        else:
            out[feat] = f"+{rho:.2f}" if rho > 0 else f"{rho:.2f}"
    return out


# ── one (model, stratum) fit ───────────────────────────────────────────────


def fit_stratum(
    model_id: str, predictors: list[str], label: str, perm_repeats: int
) -> dict:
    """Fit one augmented model on one stratum and return a tidy result row."""
    base = MODELS[model_id]
    cfg = replace(base, predictor_vars=list(predictors))

    df, X, y, groups = data_utils.load_and_filter(
        cfg.target_var, cfg.predictor_vars, cfg.outlier_threshold
    )
    mask = stratum_mask(df, label)
    df, X, y, groups = df[mask], X[mask], y[mask], groups[mask]
    df = df.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    n_obs = len(X)
    n_subj = int(groups.nunique())

    lo, hi = N_RANGES[label]
    assert lo <= n_obs <= hi, (
        f"{model_id}/{label}: n={n_obs} outside sanity range [{lo}, {hi}]"
    )

    k = min(DIAGNOSTIC_K, n_subj)
    run = RunConfig(
        name="diagnostic",
        cv_splits=k,
        perm_importance_repeats=perm_repeats,
        skip_shap=True,
        skip_pdp=True,
        skip_correlation=True,
    )

    pipe = cfg.pipeline_cls(cfg, run)
    ctx = pipe.context
    ctx.output_dir = OUT_DIR / model_id / label
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.df, ctx.X, ctx.y, ctx.groups = df, X, y, groups

    pipe.configure_model()
    pipe.cross_validate()
    pipe.permutation_importance_analysis()
    pipe.fit_model()

    perm = ctx.perm_importance_df.set_index("feature")
    estimator = ctx.pipeline.named_steps[ESTIMATOR_STEP]
    signs = shap_signs(estimator, X, [V.ATTEND, V.GROUP])
    fold_r2 = np.asarray(ctx.cv_results["test_r2"], dtype="float64")

    def _imp(feat: str, col: str) -> float:
        return float(perm.loc[feat, col]) if feat in perm.index else float("nan")

    return {
        "model": model_id,
        "target": cfg.target_var,
        "stratum": label,
        "n_obs": n_obs,
        "n_subjects": n_subj,
        "cv_k": k,
        "n_pred": len(predictors),
        "pooled_r2": ctx.pooled_cv_metrics["pooled_r2"],
        "fold_r2_mean": float(np.mean(fold_r2)),
        "fold_r2_std": float(np.std(fold_r2)),
        "attend_imp": _imp(V.ATTEND, "importance_mean"),
        "attend_imp_std": _imp(V.ATTEND, "importance_std"),
        "attend_sign": signs[V.ATTEND],
        "group_imp": _imp(V.GROUP, "importance_mean"),
        "group_imp_std": _imp(V.GROUP, "importance_std"),
        "group_sign": signs[V.GROUP],
    }


# ── reporting helpers ──────────────────────────────────────────────────────


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else format(value, floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def question_flags(results: pd.DataFrame) -> str:
    """Programmatic readout for the three pre-committed questions."""
    lines: list[str] = []

    # Q1 - does any resolved stratum materially beat the all-periods pool?
    lines.append("### Q1 - does resolving raise pooled R² over the all-periods pool?\n")
    q1_rows = []
    for model_id, grp in results.groupby("model", sort=False):
        by = grp.set_index("stratum")["pooled_r2"]
        base = by.get("all", float("nan"))
        resolved = by.reindex(PERIOD_STRATA + ["intervention"])
        best = resolved.max()
        q1_rows.append(
            {
                "model": model_id,
                "r2_all": base,
                "r2_best_resolved": best,
                "delta": best - base,
                "best_stratum": resolved.idxmax() if resolved.notna().any() else "",
            }
        )
    q1 = pd.DataFrame(q1_rows)
    material = q1[q1["delta"] >= 0.10]
    lines.append(md_table(q1))
    lines.append(
        f"\n**{len(material)}/{len(q1)} models** show a resolved stratum beating "
        f"the all-periods pool by >= 0.10 R². Mean delta = {q1['delta'].mean():.3f}; "
        f"max delta = {q1['delta'].max():.3f}"
        + (f" ({material['model'].tolist()})." if len(material) else ".")
    )

    # Q2 / Q3 - does the attend / group contribution concentrate in period 1?
    for term, label in (("attend", "Q2 - attend (dose)"), ("group", "Q3 - group")):
        col = f"{term}_imp"
        lines.append(f"\n### {label} contribution by period\n")
        pivot = results.pivot_table(
            index="model", columns="stratum", values=col, sort=False
        ).reindex(columns=STRATA)
        lines.append(md_table(pivot.reset_index()))
        p1 = pivot["period1"]
        later = pivot[["period2", "period3"]].mean(axis=1)
        concentrated = ((p1 > 0.0) & (p1 >= 2.0 * later.clip(lower=0))).sum()
        lines.append(
            f"\nPeriod-1 {term} importance exceeds 2x the period-2/3 mean in "
            f"**{int(concentrated)}/{len(pivot)} models**. "
            f"Mean period-1 = {p1.mean():.3f}; mean period-2/3 = {later.mean():.3f}."
        )

    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    gain_models = sorted(
        m for m, c in MODELS.items() if c.target_var.endswith("_gain") and c.variant_of is None
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="*",
        default=gain_models,
        help="Model ids to run (default: all gain primaries).",
    )
    parser.add_argument("--perm-repeats", type=int, default=PERM_REPEATS)
    args = parser.parse_args()
    perm_repeats = args.perm_repeats

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[bold]Period-resolved GB diagnostic[/bold] - {len(args.models)} model(s), "
          f"{len(STRATA)} strata, GroupKFold(k={DIAGNOSTIC_K}) by subject\n")

    rows: list[dict] = []
    for model_id in args.models:
        cfg = MODELS[model_id]
        pruned = list(cfg.predictor_vars)
        augmented = augment_predictors(pruned)
        added = [p for p in augmented if p not in pruned]
        print(
            f"[cyan]{model_id}[/cyan] ({cfg.target_var}): pruned {len(pruned)} "
            f"+ design {added} -> {len(augmented)} predictors"
        )
        for label in STRATA:
            row = fit_stratum(model_id, augmented, label, perm_repeats)
            rows.append(row)
            print(
                f"    {label:13s} n={row['n_obs']:>3d} (subj={row['n_subjects']:>2d}, "
                f"k={row['cv_k']:>2d})  pooled_R2={row['pooled_r2']:+.3f}  "
                f"attend_imp={row['attend_imp']:+.3f} ({row['attend_sign']})  "
                f"group_imp={row['group_imp']:+.3f} ({row['group_sign']})"
            )

    results = pd.DataFrame(rows)
    csv_path = OUT_DIR / "results.csv"
    results.to_csv(csv_path, index=False)

    pooled_pivot = results.pivot_table(
        index="model", columns="stratum", values="pooled_r2", sort=False
    ).reindex(columns=STRATA)

    summary = "\n".join(
        [
            "# Period-resolved GB diagnostic - results\n",
            f"Models: {', '.join(args.models)}",
            f"Strata: {', '.join(STRATA)} | GroupKFold(k={DIAGNOSTIC_K}) by subject "
            f"| permutation repeats: {perm_repeats}\n",
            "## Pooled out-of-fold R² by stratum\n",
            md_table(pooled_pivot.reset_index()),
            "\n## Full tidy results\n",
            md_table(results),
            "\n## Pre-committed questions\n",
            question_flags(results),
        ]
    )
    (OUT_DIR / "summary.md").write_text(summary + "\n", encoding="utf-8")

    print("\n[bold]Pooled out-of-fold R² by stratum[/bold]")
    print(pooled_pivot.round(3).to_string())
    print(f"\nWrote {csv_path}")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
