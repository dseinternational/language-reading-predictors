# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Uniform gradient-boosting feature selection (reproducible reconstruction).

Re-derives a reduced, redundancy-free predictor set for a target from the
full default set by the single algorithm documented in
``notes/202606211200-uniform-gb-fs.md`` (the original
``output/replication/uniform_fs.py`` lived under the gitignored ``output/``
tree and was not committed). Used to add the speech / verbal-memory /
language-sample exploratory models (lrp25-42) on the same footing as
lrp01-22.

Algorithm (applied to every target, from the full set)
------------------------------------------------------
Starting from the full default predictor set (``DEFAULT_GAIN`` for gain
targets — which already contains the baseline; ``DEFAULT_LEVEL`` minus the
target for level targets), using deterministic full-set out-of-fold
permutation importance (group-aware, ``neg_root_mean_squared_error``,
seed 47 — exactly as ``base_pipeline.permutation_importance_analysis``) and
the distance-correlation matrix (``stats_utils.distance_corr_matrix``):

1. **Redundancy filter** — rank predictors by full-set OOF permutation
   importance (desc); keep a predictor unless its distance correlation with
   an already-kept, higher-importance predictor is >= ``--dcor`` (default
   0.70). Gain models **force-keep the baseline** (regression-to-the-mean
   anchor) regardless of rank.
2. **Noise-floor cut** — drop remaining predictors with OOF importance
   <= ``--noise-floor`` (default 0.005). The gain baseline is exempt.
3. **Standardised-instrument swap** — prefer the standardised test over its
   bespoke taught sibling (``eowpvt`` <- ``b1exto``, ``rowpvt`` <- ``b1reto``)
   when the swap does not reintroduce a >= ``--dcor`` pair.

The full-set importance ranking uses one fixed, documented LightGBM config
(``_RANKING_PARAMS``) — ranking happens *before* per-model tuning, so the
config is deliberately fixed rather than tuned. The reduced set is then
re-tuned downstream (``scripts/tune_model.py`` on the registered reduced
model, Optuna 150-trial MAE, 10-fold GroupKFold, seed 47).

Writes ``output/feature_selection/<model_id>.json`` (machine-readable: the
``removed`` list to paste into the module's ``SelectionStep``, the kept set,
the importance ranking, and any residual dcor >= threshold pairs — expected
to be empty) and prints a summary.

Usage
-----
    python scripts/uniform_feature_selection.py lrp26 erbnw level
    python scripts/uniform_feature_selection.py lrp25 erbnw_gain gain
    python scripts/uniform_feature_selection.py --batch   # all 18 lrp25-42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

import language_reading_predictors.data_utils as data_utils
from language_reading_predictors.data_variables import Predictors, Variables as V
from language_reading_predictors.stats_utils import distance_corr_matrix

_ROOT_DIR = Path(__file__).resolve().parent.parent
_OUT_DIR = _ROOT_DIR / "output" / "feature_selection"

# Fixed full-set importance-ranking estimator. Ranking precedes per-model
# tuning, so this is deliberately a single reasonable MAE config applied to
# every target (the reduced set is re-tuned later). Distance correlation is
# parameter-free, so only the importance ranking and the noise floor depend
# on this.
_RANKING_PARAMS: dict[str, object] = {
    "objective": "mae",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 10,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
}

# Standardised-instrument swaps: standardised test <- bespoke taught sibling.
_INSTRUMENT_SWAPS: dict[str, str] = {
    V.B1EXTO: V.EOWPVT,  # prefer EOWPVT over taught B1EXTO
    V.B1RETO: V.ROWPVT,  # prefer ROWPVT over taught B1RETO
}

# The 18 exploratory targets (model_id, target_var, kind).
_BATCH: list[tuple[str, str, str]] = [
    ("lrp25", V.ERBNW_GAIN, "gain"),
    ("lrp26", V.ERBNW, "level"),
    ("lrp27", V.ERBWORD_GAIN, "gain"),
    ("lrp28", V.ERBWORD, "level"),
    ("lrp29", V.ERBTO_GAIN, "gain"),
    ("lrp30", V.ERBTO, "level"),
    ("lrp31", V.DEAPPIN_GAIN, "gain"),
    ("lrp32", V.DEAPPIN, "level"),
    ("lrp33", V.DEAPPVO_GAIN, "gain"),
    ("lrp34", V.DEAPPVO, "level"),
    ("lrp35", V.DEAPPAV_GAIN, "gain"),
    ("lrp36", V.DEAPPAV, "level"),
    ("lrp37", V.DEAPP_C, "level"),  # composite has no _gain column
    ("lrp38", V.LSAMMLU, "level"),  # LSAM is t1-t2 only: level models only
    ("lrp39", V.LSAMMAX, "level"),
    ("lrp40", V.LSAMINT, "level"),
    ("lrp41", V.LSAMUN, "level"),
    ("lrp42", V.LSAMTO, "level"),
]


def _full_predictor_set(target_var: str, kind: str) -> tuple[list[str], str | None]:
    """Full default predictor set for *target_var*, mirroring base_model.

    Returns ``(predictors, baseline)`` where *baseline* is the force-kept
    regression-to-the-mean anchor for gain targets (``None`` for levels).
    """
    if kind == "gain":
        base = list(Predictors.DEFAULT_GAIN)
        baseline = target_var.removesuffix("_gain")
        # GainModel prepends the baseline; it is already a DEFAULT_GAIN
        # member, so this is a no-op beyond ordering.
        predictors = [baseline] + [p for p in base if p not in {baseline, target_var}]
        return predictors, baseline
    base = list(Predictors.DEFAULT_LEVEL)
    predictors = [p for p in base if p != target_var]
    return predictors, None


def _oof_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv_splits: int,
    n_repeats: int,
    seed: int,
) -> pd.Series:
    """Group-aware out-of-fold permutation importance (neg-RMSE, mean).

    Replicates ``base_pipeline.permutation_importance_analysis``: for each
    GroupKFold fold the estimator is fit on the training rows and permutation
    importance is computed on the held-out rows; per-fold importance arrays
    (n_repeats columns each) are concatenated and averaged. Scoring is
    ``neg_root_mean_squared_error`` so the units match the headline CV metric.
    """
    from lightgbm import LGBMRegressor

    cv = GroupKFold(n_splits=cv_splits)
    fold_importances = []
    for tr_idx, val_idx in cv.split(X, y, groups):
        est = LGBMRegressor(**_RANKING_PARAMS)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        result = permutation_importance(
            est,
            X.iloc[val_idx],
            y.iloc[val_idx],
            n_repeats=n_repeats,
            random_state=seed,
            scoring="neg_root_mean_squared_error",
        )
        fold_importances.append(result.importances)

    all_importances = np.concatenate(fold_importances, axis=1)
    return pd.Series(all_importances.mean(axis=1), index=list(X.columns))


def _redundancy_filter(
    ranked: list[str],
    dcor: pd.DataFrame,
    threshold: float,
    force_keep: str | None,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Keep the highest-importance representative of each correlated cluster.

    *ranked* is in descending importance order. Returns ``(kept, drops)``
    where *drops* records ``(dropped, kept_representative, dcor)`` triples.
    """
    kept: list[str] = []
    drops: list[tuple[str, str, float]] = []
    for feat in ranked:
        if feat == force_keep:
            kept.append(feat)
            continue
        redundant_with = None
        for k in kept:
            if float(dcor.loc[feat, k]) >= threshold:
                redundant_with = k
                break
        if redundant_with is None:
            kept.append(feat)
        else:
            drops.append((feat, redundant_with, float(dcor.loc[feat, redundant_with])))
    return kept, drops


def _instrument_swaps(
    kept: list[str],
    full: list[str],
    dcor: pd.DataFrame,
    threshold: float,
) -> list[tuple[str, str]]:
    """Apply standardised<-taught swaps that do not reintroduce redundancy."""
    swaps: list[tuple[str, str]] = []
    for taught, standard in _INSTRUMENT_SWAPS.items():
        if taught in kept and standard not in kept and standard in full:
            others = [k for k in kept if k != taught]
            if all(float(dcor.loc[standard, o]) < threshold for o in others):
                idx = kept.index(taught)
                kept[idx] = standard
                swaps.append((taught, standard))
    return swaps


def select(
    model_id: str,
    target_var: str,
    kind: str,
    *,
    cv_splits: int,
    n_repeats: int,
    seed: int,
    dcor_threshold: float,
    noise_floor: float,
) -> dict:
    full, baseline = _full_predictor_set(target_var, kind)
    _df, X, y, groups = data_utils.load_and_filter(target_var, full)
    # Guard: GroupKFold needs at least as many groups as splits.
    n_groups = int(groups.nunique())
    cv_splits = min(cv_splits, n_groups)

    importance = _oof_permutation_importance(
        X, y, groups, cv_splits, n_repeats, seed
    )
    dcor_arr = distance_corr_matrix(X)
    dcor = pd.DataFrame(dcor_arr, index=list(X.columns), columns=list(X.columns))

    ranked = importance.sort_values(ascending=False).index.tolist()
    kept, drops = _redundancy_filter(ranked, dcor, dcor_threshold, baseline)

    # Noise-floor cut (baseline exempt).
    floored = [
        f
        for f in kept
        if f != baseline and float(importance[f]) <= noise_floor
    ]
    kept = [f for f in kept if f not in floored]

    swaps = _instrument_swaps(kept, full, dcor, dcor_threshold)

    # Re-rank kept by importance for a stable, readable order.
    kept_sorted = sorted(
        kept, key=lambda f: float(importance.get(f, 0.0)), reverse=True
    )
    removed = [f for f in full if f not in kept_sorted]

    # Verify 0 residual dcor >= threshold pairs among kept.
    residual_pairs = [
        (a, b, float(dcor.loc[a, b]))
        for i, a in enumerate(kept_sorted)
        for b in kept_sorted[i + 1 :]
        if float(dcor.loc[a, b]) >= dcor_threshold
    ]

    result = {
        "model_id": model_id,
        "target_var": target_var,
        "kind": kind,
        "n_observations": int(len(y)),
        "n_groups": n_groups,
        "cv_splits": cv_splits,
        "n_repeats": n_repeats,
        "seed": seed,
        "dcor_threshold": dcor_threshold,
        "noise_floor": noise_floor,
        "baseline_force_kept": baseline,
        "n_full": len(full),
        "n_kept": len(kept_sorted),
        "kept": kept_sorted,
        "removed": removed,
        "instrument_swaps": [{"from": a, "to": b} for a, b in swaps],
        "noise_floored": floored,
        "redundancy_drops": [
            {"dropped": d, "kept_rep": k, "dcor": c} for d, k, c in drops
        ],
        "residual_dcor_pairs": [
            {"a": a, "b": b, "dcor": c} for a, b, c in residual_pairs
        ],
        "importance": {f: float(importance[f]) for f in ranked},
    }

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / f"{model_id}.json").write_text(json.dumps(result, indent=2))

    print(
        f"[bold green]{model_id}[/bold green] {target_var} ({kind}): "
        f"{len(full)} -> {len(kept_sorted)} predictors "
        f"(n={len(y)}, groups={n_groups})"
    )
    print(f"  kept: {kept_sorted}")
    if swaps:
        print(f"  swaps: {swaps}")
    if residual_pairs:
        print(f"  [yellow]residual dcor >= {dcor_threshold}: {residual_pairs}[/yellow]")
    else:
        print(f"  [dim]0 residual dcor >= {dcor_threshold} pairs[/dim]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", nargs="?", help="e.g. lrp26")
    parser.add_argument("target_var", nargs="?", help="e.g. erbnw")
    parser.add_argument("kind", nargs="?", choices=["level", "gain"])
    parser.add_argument("--batch", action="store_true", help="Run all 18 lrp25-42.")
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--dcor", type=float, default=0.70)
    parser.add_argument("--noise-floor", type=float, default=0.005)
    args = parser.parse_args()

    if args.batch:
        specs = _BATCH
    elif args.model_id and args.target_var and args.kind:
        specs = [(args.model_id, args.target_var, args.kind)]
    else:
        parser.error("provide (model_id target_var kind) or --batch")

    for model_id, target_var, kind in specs:
        select(
            model_id,
            target_var,
            kind,
            cv_splits=args.cv_splits,
            n_repeats=args.n_repeats,
            seed=args.seed,
            dcor_threshold=args.dcor,
            noise_floor=args.noise_floor,
        )


if __name__ == "__main__":
    main()
