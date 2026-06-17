# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Predictability readout for a fitted gain model.

Turns a model's existing cross-validation artifacts into a plain answer to
"how predictable is the outcome, and from what" — *without* fitting a new
model. For each model id it:

1. Reads ``metrics.json`` for the headline out-of-sample skill. The key
   number is ``cv_pooled_r2`` — the pooled out-of-fold
   :math:`1 - \\mathrm{SS_{res}}/\\mathrm{SS_{tot}}` where ``SS_tot`` uses
   each fold's *training mean*. That is exactly the skill over a
   predict-the-mean baseline, i.e. the fraction of individual variation
   explained out of sample.
2. Runs an explicit ``DummyRegressor(strategy="mean")`` through the same
   ``GroupKFold`` splits as a transparency cross-check. By construction the
   dummy's pooled R² is ≈ 0; its pooled RMSE is the denominator the model's
   skill is measured against. This makes the "vs predict-the-mean" baseline
   visible as its own row rather than implied.
3. Reads ``oof_predictions.csv`` and writes ``calibration.png`` —
   CV-predicted gain (x) versus observed gain (y) with the y=x line — so
   over/under-prediction across the range is visible.
4. Reads ``permutation_importance.csv`` and ``shap_direction_diagnostics.csv``
   and prints a top-predictor table that pairs *how much* a predictor
   contributes (permutation importance) with *which direction* it runs and
   *how consistently* (SHAP-feature Spearman sign + monotonicity flag), per
   the project's interpretation rule (see CLAUDE.md, "Interpreting model
   results").

Artifacts are written back into ``output/models/{model_id}/``:
``calibration.png`` and ``predictability_readout.json`` (machine-readable
summary), plus a markdown block printed to stdout for pasting into notes.

Usage
-----

::

    python scripts/predictability_readout.py
    python scripts/predictability_readout.py lrp01_prediction lrp01 --top-n 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupKFold

import language_reading_predictors.data_utils as data_utils
from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.registry import MODELS

_ROOT_DIR = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT_DIR / "output" / "models"

_DEFAULT_MODELS = ["lrp01_prediction", "lrp01"]


def _read_json(path: Path) -> dict:
    if not path.exists():
        msg = (
            f"missing {path.name} at {path} — fit the model first "
            f"(e.g. `python scripts/fit_model.py {path.parent.name} "
            "--config reporting`)"
        )
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def _pooled_metrics(
    y_true: np.ndarray,
    oof_pred: np.ndarray,
    train_means: np.ndarray,
) -> dict[str, float]:
    """Pooled out-of-fold RMSE and R² against the per-fold training mean.

    ``train_means`` holds, for each observation, the mean of the *training*
    target in the fold where that observation was held out — the same
    baseline ``base_pipeline.cross_validate`` uses, so the numbers line up
    with ``metrics.json``.
    """
    resid = y_true - oof_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - train_means) ** 2))
    return {
        "pooled_rmse": float(np.sqrt(np.mean(resid**2))),
        "pooled_r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def _dummy_baseline(model_id: str, cv_splits: int) -> dict[str, float]:
    """Run a predict-the-mean baseline through the model's GroupKFold splits.

    Reloads the model's data exactly as the fit pipeline does (same target,
    predictors, outlier threshold and — once present — row filter), so the
    deterministic ``GroupKFold`` folds match. Returns the baseline's pooled
    RMSE (the denominator of the model's skill) and pooled R² (≈ 0).
    """
    cfg = MODELS[model_id]
    row_filter = getattr(cfg, "row_filter", None)
    if row_filter is not None:
        _, X, y, groups = data_utils.load_and_filter(
            cfg.target_var, cfg.predictor_vars, cfg.outlier_threshold, row_filter
        )
    else:
        _, X, y, groups = data_utils.load_and_filter(
            cfg.target_var, cfg.predictor_vars, cfg.outlier_threshold
        )

    y_true = y.to_numpy(dtype=float)
    oof_pred = np.full_like(y_true, np.nan, dtype=float)
    train_means = np.full_like(y_true, np.nan, dtype=float)

    cv = GroupKFold(n_splits=cv_splits)
    for tr_idx, val_idx in cv.split(X, y, groups):
        dummy = DummyRegressor(strategy="mean").fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_pred[val_idx] = dummy.predict(X.iloc[val_idx])
        train_means[val_idx] = float(np.mean(y_true[tr_idx]))

    pooled = _pooled_metrics(y_true, oof_pred, train_means)
    return {
        "n_observations": int(len(y_true)),
        "target_sd": float(np.std(y_true, ddof=1)),
        "baseline_pooled_rmse": pooled["pooled_rmse"],
        "baseline_pooled_r2": pooled["pooled_r2"],
    }


def _oof_skill(out_dir: Path, predictors: list[str]) -> dict[str, float]:
    """Out-of-fold pooled skill of the *fitted model* on a given predictor set.

    Re-runs the same ``GroupKFold`` machinery as ``base_pipeline`` (same
    splits, target, hyperparameters, seed and outlier threshold read from
    the saved ``config.json``) but on an arbitrary ``predictors`` list. With
    the model's full predictor set it reproduces ``metrics.json``'s
    ``cv_pooled_r2``; with a reduced set (e.g. concurrent features dropped)
    it gives that set's honest out-of-sample skill — without touching the
    registered model. The mean-baseline RMSE (per-fold training mean = a
    ``DummyRegressor(strategy="mean")``) is the denominator of the reduction.
    """
    from lightgbm import LGBMRegressor

    cfg = json.loads((out_dir / "config.json").read_text())
    target = cfg["target_var"]
    params = cfg["model_params"]
    seed = int(cfg["random_seed"])
    cv_splits = int(cfg["cv_splits"])
    outlier = cfg["outlier_threshold"]

    _, X, y, groups = data_utils.load_and_filter(target, predictors, outlier)
    y_true = y.to_numpy(dtype=float)
    oof_pred = np.full_like(y_true, np.nan, dtype=float)
    train_means = np.full_like(y_true, np.nan, dtype=float)

    cv = GroupKFold(n_splits=cv_splits)
    for tr_idx, val_idx in cv.split(X, y, groups):
        est = LGBMRegressor(**{**params, "random_state": seed})
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_pred[val_idx] = est.predict(X.iloc[val_idx])
        train_means[val_idx] = float(np.mean(y_true[tr_idx]))

    pooled = _pooled_metrics(y_true, oof_pred, train_means)
    base_rmse = float(np.sqrt(np.mean((y_true - train_means) ** 2)))
    return {
        "n_predictors": len(predictors),
        "pooled_r2": pooled["pooled_r2"],
        "pooled_rmse": pooled["pooled_rmse"],
        "baseline_pooled_rmse": base_rmse,
        "rmse_reduction_pct": (1.0 - pooled["pooled_rmse"] / base_rmse) * 100.0
        if base_rmse
        else float("nan"),
    }


def _baseline_only_skill(
    out_dir: Path, full_predictors: list[str]
) -> tuple[dict[str, float] | None, list[str]]:
    """Model skill with concurrent/period-related predictors removed.

    Drops ``Variables.PERIOD_RELATED`` (``attend``, ``tascore``, ``tachang`` —
    measured *during* the gain window, so not knowable at baseline). Returns
    ``(skill, dropped)``; ``skill`` is ``None`` when nothing was dropped (the
    full set is already baseline-only).
    """
    period = set(V.PERIOD_RELATED)
    dropped = [p for p in full_predictors if p in period]
    baseline_predictors = [p for p in full_predictors if p not in period]
    if not dropped:
        return None, []
    return _oof_skill(out_dir, baseline_predictors), dropped


def _calibration_plot(model_id: str, out_dir: Path, pooled_r2: float | None) -> Path:
    """Predicted-vs-actual calibration scatter from ``oof_predictions.csv``."""
    oof = pd.read_csv(out_dir / "oof_predictions.csv")
    y_true = oof["y_true"].to_numpy(dtype=float)
    y_pred = oof["y_pred_oof"].to_numpy(dtype=float)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lims = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(lims, lims, color="black", linestyle="--", linewidth=1, label="y = x (perfect)")
    ax.scatter(y_pred, y_true, s=28, alpha=0.7, edgecolor="C0", facecolor="none")

    # Least-squares trend of observed on predicted — a flatter-than-y=x slope
    # is the regression-to-the-mean shrinkage typical of a low-signal target.
    if np.ptp(y_pred) > 0:
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        xs = np.array(lims)
        ax.plot(xs, slope * xs + intercept, color="C3", linewidth=1.5,
                label=f"observed~predicted fit (slope {slope:.2f})")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Cross-validated predicted gain (out-of-fold)")
    ax.set_ylabel("Observed gain")
    title = f"{model_id}: predicted vs observed word-reading gain"
    if pooled_r2 is not None and not np.isnan(pooled_r2):
        title += f"\npooled out-of-fold R² = {pooled_r2:.3f}"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = out_dir / "calibration.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _direction_table(out_dir: Path, top_n: int) -> pd.DataFrame:
    """Top predictors by permutation importance, paired with SHAP direction."""
    perm = pd.read_csv(out_dir / "permutation_importance.csv")
    perm = perm.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    perm["rank"] = perm.index + 1

    diag_path = out_dir / "shap_direction_diagnostics.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)[
            ["feature", "feature_shap_spearman", "shape_flag"]
        ]
        merged = perm.merge(diag, on="feature", how="left")
    else:
        merged = perm.copy()
        merged["feature_shap_spearman"] = np.nan
        merged["shape_flag"] = "—"

    return merged.head(top_n)


_FLAG_DIRECTION = {
    "monotonic_+": "higher → more gain",
    "monotonic_-": "higher → less gain",
    "noisy_+": "higher → more gain (noisy)",
    "noisy_-": "higher → less gain (noisy)",
    "non_monotonic": "mixed / non-monotonic",
}


def _markdown_summary(
    model_id: str,
    metrics: dict,
    baseline: dict,
    full_skill: dict,
    base_skill: dict | None,
    dropped: list[str],
    directions: pd.DataFrame,
) -> str:
    r2 = metrics.get("cv_pooled_r2")
    rmse = metrics.get("cv_pooled_rmse")
    base_rmse = baseline["baseline_pooled_rmse"]
    rmse_reduction = (
        (1.0 - rmse / base_rmse) * 100.0
        if rmse is not None and base_rmse
        else float("nan")
    )

    lines = [
        f"### {model_id} — predictability readout",
        "",
        f"- n = {baseline['n_observations']}, target SD = {baseline['target_sd']:.2f}",
        f"- **Skill vs predict-the-mean (pooled out-of-fold R²): "
        f"{r2:.3f}**" + ("" if r2 is None else f" ({r2 * 100:.0f}% of individual "
                                                "variation explained out of sample)"),
        f"- Pooled OOF RMSE {rmse:.2f} vs mean-baseline RMSE "
        f"{base_rmse:.2f} → {rmse_reduction:.0f}% RMSE reduction",
        f"- DummyRegressor(mean) baseline pooled R² = "
        f"{baseline['baseline_pooled_r2']:.3f} (≈ 0 by construction — confirms wiring)",
        f"- CV RMSE {metrics.get('cv_rmse_mean'):.2f} ± "
        f"{metrics.get('cv_rmse_std'):.2f}; CV MAE "
        f"{metrics.get('cv_mae_mean'):.2f} ± {metrics.get('cv_mae_std'):.2f}",
        "",
        "**Skill by predictor set** (which is the honest 'forecast from baseline' number?)",
        "",
        "| predictor set | n predictors | pooled OOF R² | RMSE reduction vs mean |",
        "|---|---:|---:|---:|",
        f"| full (includes concurrent `{'`, `'.join(dropped) if dropped else '—'}`) "
        f"| {full_skill['n_predictors']} | {full_skill['pooled_r2']:.3f} "
        f"| {full_skill['rmse_reduction_pct']:.0f}% |",
    ]
    if base_skill is not None:
        lines.append(
            f"| **baseline-only (no `{'`, `'.join(dropped)}`)** "
            f"| {base_skill['n_predictors']} | **{base_skill['pooled_r2']:.3f}** "
            f"| {base_skill['rmse_reduction_pct']:.0f}% |"
        )
        lines.append(
            f"\n→ **From baseline information alone, pooled OOF R² = "
            f"{base_skill['pooled_r2']:.3f}** "
            f"({base_skill['pooled_r2'] * 100:.0f}% of variation). The "
            f"{(full_skill['pooled_r2'] - base_skill['pooled_r2']) * 100:.0f} pp "
            f"gap vs the full set is the share of predictability carried by "
            f"concurrent `{'`, `'.join(dropped)}` (dose/timing), not baseline traits."
        )
    lines += [
        "",
        "| rank | predictor | perm. importance | SHAP–feature ρ | direction |",
        "|---:|---|---:|---:|---|",
    ]
    for r in directions.itertuples():
        rho = getattr(r, "feature_shap_spearman", float("nan"))
        rho_s = "—" if pd.isna(rho) else f"{rho:+.2f}"
        flag = getattr(r, "shape_flag", "—")
        direction = _FLAG_DIRECTION.get(flag, flag)
        lines.append(
            f"| {r.rank} | `{r.feature}` | {r.importance_mean:.3f} | "
            f"{rho_s} | {direction} |"
        )
    lines.append("")
    return "\n".join(lines)


def _readout(model_id: str, top_n: int) -> str:
    out_dir = _MODELS_DIR / model_id
    if not out_dir.exists():
        msg = (
            f"no output for {model_id!r} at {out_dir} — fit it first: "
            f"`python scripts/fit_model.py {model_id} --config reporting`"
        )
        raise FileNotFoundError(msg)

    metrics = _read_json(out_dir / "metrics.json")
    cv_splits = int(metrics.get("cv_splits"))

    baseline = _dummy_baseline(model_id, cv_splits)

    full_predictors = json.loads((out_dir / "config.json").read_text())["predictor_vars"]
    full_skill = _oof_skill(out_dir, full_predictors)
    base_skill, dropped = _baseline_only_skill(out_dir, full_predictors)

    # The recomputed full-set skill should match the pipeline's metrics.json —
    # confirms the baseline-only number comes from the same machinery.
    metrics_r2 = metrics.get("cv_pooled_r2")
    if metrics_r2 is not None and abs(full_skill["pooled_r2"] - metrics_r2) > 0.01:
        print(
            f"[yellow]Warning: recomputed full-set R² "
            f"{full_skill['pooled_r2']:.3f} differs from metrics.json "
            f"{metrics_r2:.3f} for {model_id}[/yellow]"
        )

    calib_path = _calibration_plot(model_id, out_dir, metrics.get("cv_pooled_r2"))
    directions = _direction_table(out_dir, top_n)
    md = _markdown_summary(
        model_id, metrics, baseline, full_skill, base_skill, dropped, directions
    )

    summary = {
        "model_id": model_id,
        "n_observations": baseline["n_observations"],
        "target_sd": baseline["target_sd"],
        "cv_pooled_r2": metrics.get("cv_pooled_r2"),
        "cv_pooled_rmse": metrics.get("cv_pooled_rmse"),
        "cv_rmse_mean": metrics.get("cv_rmse_mean"),
        "cv_rmse_std": metrics.get("cv_rmse_std"),
        "cv_mae_mean": metrics.get("cv_mae_mean"),
        "cv_mae_std": metrics.get("cv_mae_std"),
        "baseline_pooled_rmse": baseline["baseline_pooled_rmse"],
        "baseline_pooled_r2": baseline["baseline_pooled_r2"],
        "full_set_skill": full_skill,
        "baseline_only_skill": base_skill,
        "dropped_period_related": dropped,
        "top_predictors": directions[
            ["rank", "feature", "importance_mean", "feature_shap_spearman", "shape_flag"]
        ].to_dict(orient="records"),
    }
    (out_dir / "predictability_readout.json").write_text(json.dumps(summary, indent=2))
    print(f"[green]Wrote[/green] {calib_path.relative_to(_ROOT_DIR)}")
    print(
        f"[green]Wrote[/green] "
        f"{(out_dir / 'predictability_readout.json').relative_to(_ROOT_DIR)}"
    )
    return md


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predictability readout from a fitted gain model's CV artifacts."
    )
    parser.add_argument(
        "models",
        nargs="*",
        default=_DEFAULT_MODELS,
        help=f"Model ids. Default: {_DEFAULT_MODELS}.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=6,
        help="Number of top predictors to show in the direction table. Default: 6.",
    )
    args = parser.parse_args()

    for model_id in args.models:
        if model_id not in MODELS:
            print(f"[bold red]Unknown model: {model_id}[/bold red]")
            raise SystemExit(1)

    blocks = [_readout(model_id, args.top_n) for model_id in args.models]
    print("\n" + "\n\n".join(blocks))


if __name__ == "__main__":
    main()
