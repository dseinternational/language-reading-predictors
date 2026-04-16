# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Optuna hyperparameter tuning for language-reading-predictors models.

Uses ``GroupKFold`` CV grouped by ``subject_id`` to match the fit pipeline,
so selected hyperparameters reflect honest generalisation. Each trial's CV
loop carves an inner ``GroupShuffleSplit`` slice out of the training fold
for early stopping — the outer val fold is never shown to
``early_stopping`` — so ``best_iteration_`` and the fold RMSE are
independent. The mean best iteration across folds is saved as the tuned
``n_estimators``.

Writes results to ``output/tuning/{model_id}/``:

- ``best_params.json`` — winning hyperparameters (ready to paste into the
  per-problem model module, e.g. as a new ``lrpXX_select0N`` variant).
- ``trials.csv`` — every trial's params + CV RMSE for post-hoc review.
- ``study_summary.json`` — n_trials, best_value, direction, seed, etc.

Usage
-----
    python scripts/tune_model.py lrp01                       # 50 trials
    python scripts/tune_model.py lrp01 --n-trials 200
    python scripts/tune_model.py lrp01 --timeout 1800        # cap at 30 min
    python scripts/tune_model.py lrp01 --cv-splits 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from rich import print
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import language_reading_predictors.data_utils as data_utils
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.common import ModelConfig
from language_reading_predictors.models.registry import MODELS

_ROOT_DIR = Path(__file__).resolve().parent.parent
_TUNING_DIR = _ROOT_DIR / "output" / "tuning"


# ── fixed (non-tuned) estimator kwargs ──────────────────────────────────

_LGBM_FIXED: dict[str, Any] = {
    "subsample_freq": 1,
    "n_jobs": 16,
    "verbosity": -1,
}


# ── helpers ─────────────────────────────────────────────────────────────


def _load_frame(cfg: ModelConfig) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load data with the same filter rules as the fit pipeline.

    Leaves NaN in place — both sklearn RandomForest (>= 1.4) and LightGBM
    handle missingness natively, and imputing here would create a
    tune/fit mismatch: the selected hyperparameters would be chosen on
    mean-imputed X but then applied by ``base_pipeline`` to raw-NaN X.
    """
    df = data_utils.load_data()
    df = df[df[cfg.target_var].notna()].copy()
    if cfg.outlier_threshold is not None:
        df = df[df[cfg.target_var] < cfg.outlier_threshold]
    X = df[cfg.predictor_vars].replace({pd.NA: np.nan}).astype("float64")
    y = df[cfg.target_var].astype("float64")
    groups = df[V.SUBJECT_ID]
    return X, y, groups


def _clear_directory(path: Path) -> None:
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except PermissionError:
                pass


# ── search spaces ───────────────────────────────────────────────────────


def _lgbm_search_space(
    trial: optuna.Trial, seed: int, lgbm_objective: str = "regression"
) -> dict[str, Any]:
    params = {
        "objective": lgbm_objective,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 4, 40),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    return {**params, **_LGBM_FIXED, "random_state": seed}


# ── objectives ──────────────────────────────────────────────────────────


_SCORING_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": lambda y_true, preds: float(np.sqrt(np.mean((y_true - preds) ** 2))),
    "mae": lambda y_true, preds: float(np.mean(np.abs(y_true - preds))),
}


# Target transforms: (forward, inverse) functions applied to y before
# fitting / after predicting. Scoring always happens in the original
# target space so tuning metrics remain comparable across transforms.
_TARGET_TRANSFORMS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "none": (lambda y: y, lambda y: y),
    "log1p": (np.log1p, np.expm1),
}


def _lgbm_objective(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv_splits: int,
    seed: int,
    early_stopping_rounds: int,
    max_n_estimators: int,
    early_stopping_fraction: float,
    scoring: str = "rmse",
    lgbm_objective: str = "regression",
    target_transform: str = "none",
) -> Callable[[optuna.Trial], float]:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    score_fn = _SCORING_FUNCS[scoring]
    forward, inverse = _TARGET_TRANSFORMS[target_transform]

    def objective(trial: optuna.Trial) -> float:
        base_params = _lgbm_search_space(trial, seed, lgbm_objective)
        cv = GroupKFold(n_splits=cv_splits)

        fold_scores: list[float] = []
        best_iters: list[int] = []
        for tr_idx, val_idx in cv.split(X, y, groups=groups):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            groups_tr = groups.iloc[tr_idx]

            # Carve an inner early-stopping slice out of the training fold so
            # `best_iteration_` is chosen against held-out subjects that do NOT
            # appear in the outer val fold. This keeps the fold score honest.
            inner = GroupShuffleSplit(
                n_splits=1, test_size=early_stopping_fraction, random_state=seed
            )
            inner_tr_idx, inner_es_idx = next(
                inner.split(X_tr, y_tr, groups=groups_tr)
            )
            X_tr_inner = X_tr.iloc[inner_tr_idx]
            y_tr_inner = y_tr.iloc[inner_tr_idx]
            X_es = X_tr.iloc[inner_es_idx]
            y_es = y_tr.iloc[inner_es_idx]

            # Fit in transformed-y space; invert predictions for scoring so
            # the tuner optimises the original-scale metric.
            y_tr_fit = forward(y_tr_inner.to_numpy())
            y_es_fit = forward(y_es.to_numpy())

            model = LGBMRegressor(**base_params, n_estimators=max_n_estimators)
            model.fit(
                X_tr_inner,
                y_tr_fit,
                eval_set=[(X_es, y_es_fit)],
                callbacks=[
                    early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                    log_evaluation(0),
                ],
            )
            preds = inverse(model.predict(X_val))
            fold_scores.append(score_fn(y_val.to_numpy(), preds))
            best_iters.append(int(model.best_iteration_ or max_n_estimators))

        trial.set_user_attr("mean_best_iteration", int(round(float(np.mean(best_iters)))))
        trial.set_user_attr(f"cv_{scoring}_std", float(np.std(fold_scores)))
        return float(np.mean(fold_scores))

    return objective


# ── study runner ────────────────────────────────────────────────────────


def _make_log_trial(scoring: str):
    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        best = study.best_value
        marker = " [bold green]*[/bold green]" if trial.value == best else ""
        print(
            f"  trial {trial.number:3d}: {scoring}={trial.value:.4f}  "
            f"(best={best:.4f}){marker}"
        )
    return _log_trial


_SUPPORTED_PIPELINES = {"LGBMPipeline", "LGBMLogPipeline"}
_PIPELINE_DEFAULT_TRANSFORM = {
    "LGBMPipeline": "none",
    "LGBMLogPipeline": "log1p",
}


def tune(
    model_id: str,
    n_trials: int,
    timeout: float | None,
    cv_splits: int,
    seed: int,
    early_stopping_rounds: int,
    max_n_estimators: int,
    early_stopping_fraction: float,
    scoring: str = "rmse",
    lgbm_objective: str = "regression",
    target_transform: str | None = None,
) -> None:
    key = model_id.lower()
    if key not in MODELS:
        print(f"[bold red]Unknown model: {model_id}[/bold red]")
        print(f"Available: {', '.join(MODELS.keys())}")
        raise SystemExit(1)

    cfg = MODELS[key]
    pipeline_name = cfg.pipeline_cls.__name__
    if pipeline_name not in _SUPPORTED_PIPELINES:
        print(
            f"[bold red]tune_model.py supports {sorted(_SUPPORTED_PIPELINES)}; "
            f"{key} uses {pipeline_name}[/bold red]"
        )
        raise SystemExit(1)

    if target_transform is None:
        target_transform = _PIPELINE_DEFAULT_TRANSFORM[pipeline_name]
    if target_transform not in _TARGET_TRANSFORMS:
        print(
            f"[bold red]Unknown target_transform {target_transform!r}; "
            f"supported: {sorted(_TARGET_TRANSFORMS)}[/bold red]"
        )
        raise SystemExit(1)

    X, y, groups = _load_frame(cfg)

    print(f"[bold green]Tuning {key} ({pipeline_name})[/bold green]")
    print(f"  Observations: {len(X)}  Predictors: {X.shape[1]}")
    print(f"  CV splits: {cv_splits}  Trials: {n_trials}  Timeout: {timeout}s")
    print(
        f"  Early stopping: {early_stopping_rounds} rounds  "
        f"max_n_estimators: {max_n_estimators}  "
        f"inner ES fraction: {early_stopping_fraction}"
    )
    print(f"  Scoring: {scoring}  LightGBM objective: {lgbm_objective}")
    print(f"  Target transform: {target_transform}")
    objective = _lgbm_objective(
        X,
        y,
        groups,
        cv_splits,
        seed,
        early_stopping_rounds,
        max_n_estimators,
        early_stopping_fraction,
        scoring=scoring,
        lgbm_objective=lgbm_objective,
        target_transform=target_transform,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    print()
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[_make_log_trial(scoring)],
        gc_after_trial=True,
    )

    out_dir = _TUNING_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_directory(out_dir)

    best = study.best_trial

    best_full_params: dict[str, Any] = {
        "objective": lgbm_objective,
        **best.params,
        **_LGBM_FIXED,
        "random_state": seed,
        "n_estimators": best.user_attrs.get("mean_best_iteration", max_n_estimators),
    }

    cv_std_key = f"cv_{scoring}_std"
    best_params_out = {
        "model_id": key,
        "pipeline_cls": pipeline_name,
        "scoring": scoring,
        "target_transform": target_transform,
        f"cv_{scoring}_mean": float(best.value),
        f"cv_{scoring}_std": float(best.user_attrs.get(cv_std_key, float("nan"))),
        "n_trials": len(study.trials),
        "seed": seed,
        "cv_splits": cv_splits,
        "params": best_full_params,
    }
    (out_dir / "best_params.json").write_text(json.dumps(best_params_out, indent=2))

    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state", "duration")
    )
    trials_df.to_csv(out_dir / "trials.csv", index=False)

    summary = {
        "model_id": key,
        "pipeline_cls": pipeline_name,
        "direction": "minimize",
        "scoring": f"cv_{scoring}_mean",
        "target_transform": target_transform,
        "n_trials": len(study.trials),
        "best_value": float(study.best_value),
        "best_trial_number": int(best.number),
        "seed": seed,
        "cv_splits": cv_splits,
        "early_stopping_rounds": early_stopping_rounds,
        "max_n_estimators": max_n_estimators,
        "early_stopping_fraction": early_stopping_fraction,
    }
    (out_dir / "study_summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print(f"[bold green]Best trial: #{best.number}[/bold green]")
    label = scoring.upper()
    print(f"  CV {label} mean: {best.value:.4f}")
    cv_std = best.user_attrs.get(cv_std_key, float("nan"))
    print(f"  CV {label} std:  {cv_std:.4f}")
    if pipeline_name in _SUPPORTED_PIPELINES:
        print(f"  Mean best iteration: {best.user_attrs.get('mean_best_iteration')}")
    print(f"  Params: {json.dumps(best_full_params, indent=2)}")
    print()
    print(f"[bold green]Artifacts saved to: {out_dir}[/bold green]")
    print(
        "\n  [yellow]Paste the params dict into the matching per-problem module "
        "(e.g. a new lrp01_selectNN variant) to reuse them in a fit.[/yellow]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for LRP models."
    )
    parser.add_argument(
        "model", type=str, help="Model id, e.g. lrp01."
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Max seconds for the whole study (None = no limit).",
    )
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--max-n-estimators",
        type=int,
        default=2000,
        help="Ceiling that early stopping caps.",
    )
    parser.add_argument(
        "--early-stopping-fraction",
        type=float,
        default=0.2,
        help=(
            "Early-stopping slice as a fraction of the outer training fold, "
            "carved out by GroupShuffleSplit so early stopping does not peek "
            "at the outer val fold."
        ),
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="rmse",
        choices=list(_SCORING_FUNCS.keys()),
        help="Scoring metric to optimise (default: rmse).",
    )
    parser.add_argument(
        "--lgbm-objective",
        type=str,
        default="regression",
        help=(
            "LightGBM objective function. Use 'regression' (squared error) "
            "for RMSE tuning, 'mae' for MAE tuning. Default: regression."
        ),
    )
    parser.add_argument(
        "--target-transform",
        type=str,
        default=None,
        choices=sorted(_TARGET_TRANSFORMS.keys()),
        help=(
            "Apply a forward transform to y before fitting and inverse to "
            "predictions before scoring. Defaults based on pipeline_cls "
            "(LGBMPipeline → none, LGBMLogPipeline → log1p). Set explicitly "
            "to override."
        ),
    )
    args = parser.parse_args()

    tune(
        model_id=args.model,
        n_trials=args.n_trials,
        timeout=args.timeout,
        cv_splits=args.cv_splits,
        seed=args.seed,
        early_stopping_rounds=args.early_stopping_rounds,
        max_n_estimators=args.max_n_estimators,
        early_stopping_fraction=args.early_stopping_fraction,
        scoring=args.scoring,
        lgbm_objective=args.lgbm_objective,
        target_transform=args.target_transform,
    )


if __name__ == "__main__":
    main()
