# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Measurement-sensitivity audit of the RLI outcome measures.

Motivation
----------

The ITT models (LRPITT suite) show a robust word-reading and letter-sound
effect but no credible vocabulary effect. Before that null is read as "the
vocabulary component does not work", we need to know whether each outcome even
had the *range and reliability* to register a change in the randomised window
(t1 -> t2). A measure that is floored, ceilinged, or barely moves cannot show a
treatment effect regardless of whether one exists.

This audit is purely descriptive (no MCMC). For every bounded-count measure it
reports, per timepoint, how much of the scale is used, the floor/ceiling
fractions, a between-child dispersion proxy, and how many children's scores
actually move between t1 and t2. It then applies a transparent, documented rule
to flag each outcome as detection-"adequate" or detection-"limited".

Two roles
---------

1. Read the vocabulary null honestly: true null vs not measurable.
2. Select which phonics-route measures (letter-sound L, blending B, phonetic
   spelling P) are usable as mediators in the LRP62 reading-route composite.

Outputs (written under ``output/measurement_audit/``)
-----------------------------------------------------

- ``outcome_properties.csv`` -- one row per (measure x timepoint).
- ``detectability_verdict.csv`` -- one row per measure, with the t1/t2 summary,
  the t1->t2 movement stats, and the verdict + reason.

Usage
-----

::

    python scripts/measurement_audit.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rich import print

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models import environment as env
from language_reading_predictors.statistical_models.measures import MEASURES

# Timepoints in the long format (1 = baseline/screening, 2 = post phase-0, ...).
TIMEPOINTS: tuple[int, ...] = (1, 2, 3, 4)

# Candidate phonics-route mediators for the LRP62 composite (need pre + post in
# the randomised window). Reported first in the console summary.
ROUTE_SYMBOLS: tuple[str, ...] = ("L", "B", "P")

# --- Detectability rule (heuristic, documented) ----------------------------
# A measure is flagged "detection-limited" for the t1 -> t2 window if ANY hold:
#   * post-window ceiling crowding leaves little room to rise;
#   * scores at t2 occupy only a tiny band of the scale; or
#   * the measure is heavily floored at baseline AND few children move at all.
CEILING_LIMIT = 0.40  # fraction at the test maximum
MIN_SCALE_USED = 0.25  # observed range as a fraction of the test maximum
FLOOR_LIMIT = 0.40  # fraction at zero
MIN_MOVERS = 0.40  # fraction of children with a non-zero t1 -> t2 change

_OUTPUT_DIR = Path(env.OUTPUT_DIR) / "measurement_audit"


def _timepoint_stats(values: np.ndarray, n_trials: int) -> dict[str, float]:
    """Distribution + scale-use statistics for one measure at one timepoint."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = int(vals.size)
    if n == 0:
        return {
            "n": 0,
            "obs_min": np.nan,
            "obs_max": np.nan,
            "mean": np.nan,
            "sd": np.nan,
            "floor_pct": np.nan,
            "ceiling_pct": np.nan,
            "frac_scale_used": np.nan,
            "sd_rel": np.nan,
        }
    obs_min = float(vals.min())
    obs_max = float(vals.max())
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "obs_min": obs_min,
        "obs_max": obs_max,
        "mean": float(vals.mean()),
        "sd": sd,
        "floor_pct": float((vals <= 0).mean()),
        "ceiling_pct": float((vals >= n_trials).mean()),
        "frac_scale_used": (obs_max - obs_min) / n_trials,
        "sd_rel": sd / n_trials,
    }


def _change_stats(df: pd.DataFrame, column: str, t_pre: int = 1, t_post: int = 2) -> dict[str, float]:
    """Per-child t_pre -> t_post movement for one measure."""
    pre = (
        df.loc[df[V.TIME] == t_pre, [V.SUBJECT_ID, column]]
        .rename(columns={column: "pre"})
    )
    post = (
        df.loc[df[V.TIME] == t_post, [V.SUBJECT_ID, column]]
        .rename(columns={column: "post"})
    )
    merged = pre.merge(post, on=V.SUBJECT_ID, how="inner").dropna(subset=["pre", "post"])
    n_pairs = int(len(merged))
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "frac_nonzero_change": np.nan,
            "frac_improved": np.nan,
            "mean_change": np.nan,
        }
    delta = (merged["post"] - merged["pre"]).to_numpy(dtype=float)
    return {
        "n_pairs": n_pairs,
        "frac_nonzero_change": float((delta != 0).mean()),
        "frac_improved": float((delta > 0).mean()),
        "mean_change": float(delta.mean()),
    }


def _verdict(t1: dict[str, float], t2: dict[str, float], change: dict[str, float]) -> tuple[str, str]:
    """Apply the documented detectability rule; return (verdict, reason)."""
    reasons: list[str] = []
    ceiling_t2 = t2.get("ceiling_pct", np.nan)
    scale_t2 = t2.get("frac_scale_used", np.nan)
    floor_t1 = t1.get("floor_pct", np.nan)
    floor_t2 = t2.get("floor_pct", np.nan)
    movers = change.get("frac_nonzero_change", np.nan)

    if np.isfinite(ceiling_t2) and ceiling_t2 >= CEILING_LIMIT:
        reasons.append(f"ceiling at t2 ({ceiling_t2:.0%} at max)")
    if np.isfinite(scale_t2) and scale_t2 < MIN_SCALE_USED:
        reasons.append(f"narrow range at t2 ({scale_t2:.0%} of scale)")
    if np.isfinite(floor_t2) and floor_t2 >= FLOOR_LIMIT:
        reasons.append(f"floored at t2 ({floor_t2:.0%} at zero post-window)")
    if (
        np.isfinite(floor_t1)
        and np.isfinite(movers)
        and floor_t1 >= FLOOR_LIMIT
        and movers < MIN_MOVERS
    ):
        reasons.append(f"floored at t1 ({floor_t1:.0%} at zero) with few movers ({movers:.0%})")

    if reasons:
        return "detection-limited", "; ".join(reasons)
    return "adequate", "range and movement adequate to detect change"


def build_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-timepoint properties, per-measure verdicts)."""
    property_rows: list[dict[str, object]] = []
    verdict_rows: list[dict[str, object]] = []

    for symbol, measure in MEASURES.items():
        per_tp: dict[int, dict[str, float]] = {}
        for t in TIMEPOINTS:
            stats = _timepoint_stats(
                df.loc[df[V.TIME] == t, measure.column].to_numpy(), measure.n_trials
            )
            per_tp[t] = stats
            property_rows.append(
                {
                    "symbol": symbol,
                    "label": measure.label,
                    "column": measure.column,
                    "n_trials": measure.n_trials,
                    "n_trials_confirmed": measure.n_trials_confirmed,
                    "timepoint": t,
                    **stats,
                }
            )

        change = _change_stats(df, measure.column, t_pre=1, t_post=2)
        verdict, reason = _verdict(per_tp[1], per_tp[2], change)
        verdict_rows.append(
            {
                "symbol": symbol,
                "label": measure.label,
                "n_trials": measure.n_trials,
                "n_trials_confirmed": measure.n_trials_confirmed,
                "floor_t1": per_tp[1]["floor_pct"],
                "ceiling_t1": per_tp[1]["ceiling_pct"],
                "frac_scale_used_t1": per_tp[1]["frac_scale_used"],
                "floor_t2": per_tp[2]["floor_pct"],
                "ceiling_t2": per_tp[2]["ceiling_pct"],
                "frac_scale_used_t2": per_tp[2]["frac_scale_used"],
                "n_pairs_t1t2": change["n_pairs"],
                "frac_nonzero_change": change["frac_nonzero_change"],
                "frac_improved": change["frac_improved"],
                "mean_change": change["mean_change"],
                "verdict": verdict,
                "reason": reason,
            }
        )

    return pd.DataFrame(property_rows), pd.DataFrame(verdict_rows)


def main() -> None:
    csv_path = Path(env.DATA_DIR) / "rli_data_long.csv"
    df = pd.read_csv(csv_path)

    properties, verdicts = build_audit(df)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    properties_path = _OUTPUT_DIR / "outcome_properties.csv"
    verdict_path = _OUTPUT_DIR / "detectability_verdict.csv"
    properties.round(4).to_csv(properties_path, index=False)
    verdicts.round(4).to_csv(verdict_path, index=False)

    print(f"[bold]Measurement-sensitivity audit[/bold] (n={df[V.SUBJECT_ID].nunique()} children)")
    print(f"Wrote {properties_path}")
    print(f"Wrote {verdict_path}\n")

    summary = verdicts.set_index("symbol")[
        ["label", "floor_t1", "floor_t2", "ceiling_t2", "frac_scale_used_t2", "frac_nonzero_change", "verdict"]
    ]
    route = summary.loc[[s for s in ROUTE_SYMBOLS if s in summary.index]]
    print("[bold]Phonics-route candidates (LRP62 composite):[/bold]")
    print(route.to_string())
    print("\n[bold]All measures:[/bold]")
    print(summary.to_string())


if __name__ == "__main__":
    main()
