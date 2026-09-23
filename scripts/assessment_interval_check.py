# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""How long were the RLI assessment intervals, and what do unequal intervals do?

Companion analysis to notes/202609212100-assessment-interval-lengths.md. The data
record age in whole months at each wave and no assessment dates, so every interval
is a difference of two whole-month ages. The script reports:

1. Interval lengths per transition and arm: range, mean, median, SD and the
   whole-month distribution.
2. Immediate-minus-wait-list interval gaps with child-bootstrap intervals, beside
   a standard error from the observed within-arm spread.
3. Pooled gain per month per outcome and transition (total gain / total months).
4. Constant-rate timing scenarios for the randomised t1->t2 contrast of each graded
   outcome: the arm interval gap times a per-month growth rate, using the
   wait-list's untreated rate and the immediate arm's treated rate, beside the
   stored ITT average marginal effect in items where a fit is available.
5. The same scenarios for the onset-aligned windows (immediate t1->t3, wait-list
   t2->t4), beside the stored aligned cohort contrast.

Descriptive only: no model is refitted.

Usage:
    uv run python scripts/assessment_interval_check.py [--output-dir DIR] [--boot 20000] [--seed 20260921]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from language_reading_predictors import paths
from language_reading_predictors.data_utils import load_data
from language_reading_predictors.statistical_models.measures import MEASURES

TRANSITIONS = {"t1->t2": (1, 2), "t2->t3": (2, 3), "t3->t4": (3, 4)}
ARMS = {1: "immediate", 2: "wait-list"}
# Graded outcomes with a single-outcome ITT primary; P and N use the floor rule.
ITT_PRIMARY = {
    "TR": "lrp-rli-itt-001",
    "TE": "lrp-rli-itt-002",
    "UR": "lrp-rli-itt-003",
    "UE": "lrp-rli-itt-004",
    "R": "lrp-rli-itt-005",
    "E": "lrp-rli-itt-006",
    "L": "lrp-rli-itt-007",
    "B": "lrp-rli-itt-008",
    "W": "lrp-rli-itt-010",
    "F": "lrp-rli-itt-025",
    "T": "lrp-rli-itt-026",
}
# Graded aligned outcomes; al-005 (P) is the off-floor Bernoulli branch.
ALIGNED = {
    "W": "lrp-rli-al-001",
    "R": "lrp-rli-al-002",
    "E": "lrp-rli-al-003",
    "L": "lrp-rli-al-004",
    "B": "lrp-rli-al-006",
    "F": "lrp-rli-al-007",
    "T": "lrp-rli-al-008",
}


def wide(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.pivot(index="subject_id", columns="time", values=column).astype(float)


def describe(d: pd.Series) -> dict[str, object]:
    d = d.dropna()
    counts = d.value_counts().sort_index()
    return {
        "n": len(d),
        "min": d.min(),
        "max": d.max(),
        "mean": round(d.mean(), 2),
        "median": d.median(),
        "sd": round(d.std(), 2),
        "months:children": ", ".join(f"{int(k)}:{v}" for k, v in counts.items()),
    }


def arm_gap(d: pd.Series, imm: pd.Series, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    """Immediate minus wait-list mean, with a within-arm child bootstrap."""
    d = d.dropna()
    in_imm = imm[d.index].to_numpy()
    a, b = d[in_imm].to_numpy(), d[~in_imm].to_numpy()
    boot = rng.choice(a, (n_boot, len(a))).mean(1) - rng.choice(b, (n_boot, len(b))).mean(1)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    gap = a.mean() - b.mean()
    return {
        "gap": round(gap, 2),
        "boot_lo89": round(np.quantile(boot, 0.055), 2),
        "boot_hi89": round(np.quantile(boot, 0.945), 2),
        "se": round(se, 3),
        "gap_over_se": round(gap / se, 1),
    }


def pooled_rate(gain: pd.Series, months: pd.Series) -> float:
    ok = gain.notna() & months.notna()
    return float(gain[ok].sum() / months[ok].sum())


def stored_items(models_dir: Path, model_id: str, table: str, column: str, n_trials: int | None) -> float:
    path = models_dir / f"{model_id}-reporting" / table
    if not path.exists():
        return float("nan")
    value = float(pd.read_csv(path)[column].iloc[0])
    return value * n_trials if n_trials is not None else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", help="Output root holding the stored reporting fits.")
    parser.add_argument("--boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    paths.set_output_root(args.output_dir)
    models_dir = paths.stat_models_dir()
    rng = np.random.default_rng(args.seed)
    pd.set_option("display.width", 200)

    df = load_data().sort_values(["subject_id", "time"])
    age = wide(df, "age")
    imm = df.groupby("subject_id")["group"].first().astype(int).eq(1)
    arm = imm.map({True: ARMS[1], False: ARMS[2]})

    print("1. Interval between assessments (months, from whole-month ages)\n")
    rows = []
    for label, (s, e) in TRANSITIONS.items():
        d = age[e] - age[s]
        rows.append({"transition": label, "arm": "both", **describe(d)})
        for a in ARMS.values():
            rows.append({"transition": label, "arm": a, **describe(d[arm[d.index] == a])})
    print(pd.DataFrame(rows).to_string(index=False))
    total = (age[4] - age[1]).dropna()
    print(f"\nt1->t4 span: {total.min():.0f}-{total.max():.0f} months, mean {total.mean():.2f}")

    print("\n2. Immediate minus wait-list interval (months)\n")
    spans = {**TRANSITIONS, "t1->t3": (1, 3), "t1->t4": (1, 4)}
    gaps = [{"interval": k, **arm_gap(age[e] - age[s], imm, rng, args.boot)} for k, (s, e) in spans.items()]
    print(pd.DataFrame(gaps).to_string(index=False))

    print("\n3. Pooled gain per month by outcome and transition\n")
    rate_rows = []
    for sym in ITT_PRIMARY:
        score = wide(df, MEASURES[sym].column)
        row: dict[str, object] = {"outcome": sym}
        for label, (s, e) in TRANSITIONS.items():
            row[label] = round(pooled_rate(score[e] - score[s], age[e] - age[s]), 3)
        rate_rows.append(row)
    print(pd.DataFrame(rate_rows).to_string(index=False))

    print("\n3b. Word reading, t3->t4 against t2->t3 within the same children\n")
    word = wide(df, MEASURES["W"].column)
    g2, g3 = word[3] - word[2], word[4] - word[3]
    m2, m3 = age[3] - age[2], age[4] - age[3]
    paired = g2.notna() & g3.notna() & m2.notna() & m3.notna()
    slow_rows = []
    for label, mask in [("both", paired), *[(a, paired & (arm == a)) for a in ARMS.values()]]:
        a2, a3, b2, b3 = (s[mask].to_numpy() for s in (g2, g3, m2, m3))
        idx = rng.integers(0, len(a2), (args.boot, len(a2)))
        diff = (a3[idx] - a2[idx]).mean(1)
        ratio = (a3[idx].sum(1) / b3[idx].sum(1)) / (a2[idx].sum(1) / b2[idx].sum(1))
        slow_rows.append(
            {
                "arm": label,
                "n": len(a2),
                "gain_t2t3": round(a2.mean(), 2),
                "gain_t3t4": round(a3.mean(), 2),
                "diff": round((a3 - a2).mean(), 2),
                "diff_lo89": round(np.quantile(diff, 0.055), 2),
                "diff_hi89": round(np.quantile(diff, 0.945), 2),
                "rate_t2t3": round(a2.sum() / b2.sum(), 3),
                "rate_t3t4": round(a3.sum() / b3.sum(), 3),
                "rate_ratio": round((a3.sum() / b3.sum()) / (a2.sum() / b2.sum()), 2),
                "ratio_lo89": round(np.quantile(ratio, 0.055), 2),
                "ratio_hi89": round(np.quantile(ratio, 0.945), 2),
                "share_ratio_below_1": round((ratio < 1).mean(), 2),
                "median_child_t2t3": round(np.median(a2 / b2), 3),
                "median_child_t3t4": round(np.median(a3 / b3), 3),
            }
        )
    print(pd.DataFrame(slow_rows).to_string(index=False))

    print("\n4. Constant-rate timing scenarios for the randomised t1->t2 contrast (items)\n")
    print("Assumes whole-interval average rates apply during the extra time; these are not bounds or timing-adjusted effects.")
    itt_rows = []
    for sym, model_id in ITT_PRIMARY.items():
        measure = MEASURES[sym]
        score = wide(df, measure.column)
        gain, months = score[2] - score[1], age[2] - age[1]
        ok = gain.notna() & months.notna()
        gap = months[ok & imm].mean() - months[ok & ~imm].mean()
        r_wl, r_imm = pooled_rate(gain[~imm], months[~imm]), pooled_rate(gain[imm], months[imm])
        lo, hi = sorted((gap * r_wl, gap * r_imm))
        ame = stored_items(models_dir, model_id, "tau_summary.csv", "tau_prob_median", measure.n_trials)
        itt_rows.append(
            {
                "outcome": sym,
                "model": model_id,
                "gap_months": round(gap, 2),
                "rate_waitlist": round(r_wl, 3),
                "rate_immediate": round(r_imm, 3),
                "scenario_min": round(lo, 2),
                "scenario_max": round(hi, 2),
                "stored_ame_items": round(ame, 2),
                "scenario_max_share": round(hi / ame, 2) if np.isfinite(ame) and ame != 0 else np.nan,
            }
        )
    print(pd.DataFrame(itt_rows).to_string(index=False))

    print("\n5. Onset-aligned windows: immediate t1->t3 versus wait-list t2->t4 (items)\n")
    al_rows = []
    for sym, model_id in ALIGNED.items():
        score = wide(df, MEASURES[sym].column)
        gain = pd.concat([(score[3] - score[1])[imm], (score[4] - score[2])[~imm]])
        months = pd.concat([(age[3] - age[1])[imm], (age[4] - age[2])[~imm]])
        ok = gain.notna() & months.notna()
        gap = months[ok & imm].mean() - months[ok & ~imm].mean()
        r_wl, r_imm = pooled_rate(gain[~imm], months[~imm]), pooled_rate(gain[imm], months[imm])
        lo, hi = sorted((gap * r_wl, gap * r_imm))
        contrast = stored_items(models_dir, model_id, "cohort_marginal.csv", "trt_items_median", None)
        al_rows.append(
            {
                "outcome": sym,
                "model": model_id,
                "window_immediate": round(months[ok & imm].mean(), 2),
                "window_waitlist": round(months[ok & ~imm].mean(), 2),
                "gap_months": round(gap, 2),
                "scenario_min": round(lo, 2),
                "scenario_max": round(hi, 2),
                "stored_contrast_items": round(contrast, 2),
            }
        )
    print(pd.DataFrame(al_rows).to_string(index=False))


if __name__ == "__main__":
    main()
