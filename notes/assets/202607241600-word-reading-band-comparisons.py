# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What does the word-reading LEVEL go with? Band comparisons across every measure.

Supports ``notes/202607241600-findings-word-reading-bands.md``. Deliberately
**model-free and descriptive** — the companion Bayesian work is the ``concurrent``
family (``lrp-rli-ca-*``) and the letter-sound probe
(``202607241000-ls-wr-association-probe.py``). Nothing here is causal, and nothing
here is even an *adjusted* association: these are raw band contrasts, so latent
general ability (``GA``) and age sit inside every number.

Three passes:

P1  Split the cohort at the within-wave MEDIAN word-reading score (``ewrswr``),
    separately at each of t1-t4, and contrast the two halves on every measure.
P2  The same with QUARTILES, reporting the full per-quartile profile as well as
    the Q4-vs-Q1 contrast, and testing whether the profile is monotone across
    bands or a threshold.
P3  PROSPECTIVE: split on word reading at wave t, then contrast the two halves on
    each measure's subsequent CHANGE (t -> t+1), both raw and residualised on the
    measure's own level at t. The residual column is the one to read: a raw gain
    contrast conditional on a correlated baseline is regression-to-the-mean plus
    a ceiling constraint, not prediction.

Effect size is **Cliff's delta** with a bootstrap 89% interval (house standard,
``notes/202607172359-credible-interval-standard.md``), not Cohen's d: most of these
measures are bounded item counts with heavy floors, where a mean difference divided
by a pooled SD is not interpretable. Cliff's delta is the probability that a randomly
drawn top-band child outscores a randomly drawn bottom-band child, minus the reverse;
it handles ties explicitly, which matters enormously here (see the tie diagnostics).
``prob_superiority = (delta + 1) / 2`` is reported alongside for readability.

THE FLOOR IS THE HEADLINE CAVEAT. ``ewrswr`` — the splitting variable — has 40% of
the cohort at zero at t1, and the t1 lower-quartile cut is literally 0. So the t1
Q1/Q2 boundary is decided entirely by rank tie-breaking among children with
identical scores, and the t1 median split is nearly as arbitrary. Every band table
carries the tie diagnostics (``n_tied_at_cut``, ``pct_at_floor``) so a band contrast
can never be read without them. The same applies in reverse to the *outcome*
measures: ``nonword`` is 72% floored at t1 and ``spphon`` 78%, so their band
contrasts are compressed toward zero by the floor and their POSITION in the effect-
size ranking reflects where the floor sits, not psychological importance.

Run with the conda env interpreter from the repo root; writes CSVs to ``--out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from language_reading_predictors.data_utils import load_data

CI = 0.89  # house standard
SPLIT_VAR = "ewrswr"  # word reading, EWRSWR, 79 items
N_BOOT = 4000
SEED = 20260724

# Every measure worth contrasting, grouped for readable output. Availability varies
# by wave (lsam* are t1/t2 only; b2* start at t2; attend is t1-t3) — the script
# skips a measure at a wave where either band has fewer than MIN_BAND observations
# rather than reporting a contrast built on three children.
MEASURE_GROUPS: dict[str, list[str]] = {
    "reading": ["ewrswr", "yarcewr", "yarclet", "yarcsi", "nonword", "blending",
                "spphon", "spraw"],
    "vocabulary_language": ["rowpvt", "eowpvt", "trog", "celf", "aptgram", "aptinfo"],
    "taught_words": ["b1retau", "b1extau", "b2retau", "b2extau"],
    "speech_memory": ["erbto", "erbnw", "erbword", "deapp_c", "deappin", "deappvo",
                      "deappfi", "deappav"],
    "language_sample": ["lsammlu", "lsamto", "lsamun", "lsammax", "lsamint"],
    "cognitive_demographic": ["blocks", "age", "hearing_c", "earinf", "sdq", "behav",
                              "numchil", "agespeak"],
    "dose": ["attend", "attend_cumul", "tachang"],
}
ALL_MEASURES = [m for group in MEASURE_GROUPS.values() for m in group]
GROUP_OF = {m: g for g, ms in MEASURE_GROUPS.items() for m in ms}

MIN_BAND = 5  # minimum observations in EACH band before a contrast is reported

# Recorded at every wave but never varying within a child — collected once and
# broadcast. Verified against the data, not assumed: every one of these has exactly
# one distinct value per subject across t1-t4. They are perfectly legitimate in the
# concurrent band profile (P1/P2) but DEGENERATE in the prospective pass (P3), where
# their "change" is identically zero for every child, so P3 skips them. ``sdq`` is
# the surprise in this list: it looks longitudinal in the CSV and is not.
TIME_INVARIANT = frozenset(
    {"blocks", "hearing_c", "earinf", "sdq", "numchil", "agespeak"}
)

# Measures whose scale is bounded and whose ceiling/floor is close enough to the
# cohort range to distort a band contrast. Recorded in the output so the note can
# caveat the right rows rather than all of them.
BOUNDED: dict[str, int] = {
    "ewrswr": 79, "yarclet": 32, "nonword": 6, "blending": 10, "spphon": 92,
    "b1retau": 24, "b1extau": 24, "b2retau": 24, "b2extau": 24,
}


# ---------------------------------------------------------------------------
# effect size
# ---------------------------------------------------------------------------


def cliffs_delta(top: np.ndarray, bottom: np.ndarray) -> float:
    """P(top > bottom) - P(top < bottom). Ties contribute zero to both terms."""
    if len(top) == 0 or len(bottom) == 0:
        return np.nan
    diff = np.sign(top[:, None] - bottom[None, :])
    return float(diff.mean())


def cliffs_delta_ci(top: np.ndarray, bottom: np.ndarray, rng) -> tuple[float, float, float]:
    """Cliff's delta with a percentile bootstrap 89% interval (resampled within band)."""
    point = cliffs_delta(top, bottom)
    if not np.isfinite(point):
        return point, np.nan, np.nan
    boots = np.empty(N_BOOT)
    n_t, n_b = len(top), len(bottom)
    for i in range(N_BOOT):
        boots[i] = cliffs_delta(
            top[rng.integers(0, n_t, n_t)], bottom[rng.integers(0, n_b, n_b)]
        )
    lo, hi = np.quantile(boots, [(1 - CI) / 2, 1 - (1 - CI) / 2])
    return point, float(lo), float(hi)


def _describe(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(values)),
        "median": float(np.median(values)) if len(values) else np.nan,
        "q1": float(np.quantile(values, 0.25)) if len(values) else np.nan,
        "q3": float(np.quantile(values, 0.75)) if len(values) else np.nan,
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "min": float(np.min(values)) if len(values) else np.nan,
        "max": float(np.max(values)) if len(values) else np.nan,
    }


# ---------------------------------------------------------------------------
# banding
# ---------------------------------------------------------------------------


def assign_bands(scores: pd.Series, n_bands: int) -> tuple[pd.Series, dict]:
    """Rank-based band assignment plus the tie diagnostics that make it readable.

    ``pd.qcut`` on the raw scores fails outright when a quantile boundary falls on a
    large tie block (t1 word reading: 40% at zero, lower-quartile cut = 0), so bands
    are cut on ``rank(method="first")`` — every child lands in exactly one band and
    the bands are equal-sized. The cost is that children with IDENTICAL scores are
    split across a boundary in data order, which is arbitrary. ``n_tied_at_cut``
    counts how many children share a score with a band boundary; when it is large the
    band contrast at that boundary is measuring the tie-break, not the construct.
    """
    clean = scores.dropna().astype(float)
    ranks = clean.rank(method="first")
    labels = (
        ["bottom 50%", "top 50%"] if n_bands == 2
        else [f"Q{i + 1}" for i in range(n_bands)]
    )
    bands = pd.qcut(ranks, n_bands, labels=labels)

    # Score value at each internal band boundary, and how many children share it.
    cuts = [float(np.quantile(clean, i / n_bands)) for i in range(1, n_bands)]
    n_tied = int(sum((clean == c).sum() for c in set(cuts)))
    diag = {
        "cuts": cuts,
        "n_tied_at_cut": n_tied,
        "pct_tied_at_cut": 100.0 * n_tied / len(clean) if len(clean) else np.nan,
        "pct_at_floor": 100.0 * float((clean == 0).mean()),
        "n": int(len(clean)),
    }
    return bands.reindex(scores.index), diag


# ---------------------------------------------------------------------------
# P1 / P2 — concurrent band contrasts
# ---------------------------------------------------------------------------


def run_band_contrasts(df: pd.DataFrame, n_bands: int, out: Path, tag: str) -> pd.DataFrame:
    """Per-wave band profile + top-vs-bottom Cliff's delta for every measure."""
    rng = np.random.default_rng(SEED)
    profile_rows: list[dict] = []
    contrast_rows: list[dict] = []
    split_diag: list[dict] = []

    for t in (1, 2, 3, 4):
        wave = df[df.time == t].copy()
        bands, diag = assign_bands(wave[SPLIT_VAR], n_bands)
        wave["band"] = bands
        split_diag.append({"timepoint": t, "n_bands": n_bands, **diag})
        labels = list(bands.cat.categories)

        for measure in ALL_MEASURES:
            if measure not in wave.columns:
                continue
            per_band = {}
            for label in labels:
                vals = (
                    wave.loc[wave.band == label, measure].dropna().astype(float).to_numpy()
                )
                per_band[label] = vals
                if len(vals):
                    profile_rows.append({
                        "timepoint": t, "split": tag, "band": label,
                        "group": GROUP_OF[measure], "measure": measure,
                        **_describe(vals),
                        "pct_at_zero": 100.0 * float((vals == 0).mean()),
                        "pct_at_ceiling": (
                            100.0 * float((vals == BOUNDED[measure]).mean())
                            if measure in BOUNDED else np.nan
                        ),
                    })

            top, bottom = per_band[labels[-1]], per_band[labels[0]]
            if len(top) < MIN_BAND or len(bottom) < MIN_BAND:
                continue
            delta, lo, hi = cliffs_delta_ci(top, bottom, rng)
            pooled = np.concatenate([top, bottom])
            contrast_rows.append({
                "timepoint": t, "split": tag,
                "contrast": f"{labels[-1]} vs {labels[0]}",
                "group": GROUP_OF[measure], "measure": measure,
                "n_top": len(top), "n_bottom": len(bottom),
                "median_top": float(np.median(top)),
                "median_bottom": float(np.median(bottom)),
                "cliffs_delta": delta, "delta_lo": lo, "delta_hi": hi,
                "prob_superiority": (delta + 1) / 2,
                "excludes_zero": bool(lo > 0 or hi < 0),
                # Floor share across BOTH bands: a measure this floored cannot
                # produce a large delta whatever its true importance.
                "pct_at_zero_pooled": 100.0 * float((pooled == 0).mean()),
                "pct_at_ceiling_pooled": (
                    100.0 * float((pooled == BOUNDED[measure]).mean())
                    if measure in BOUNDED else np.nan
                ),
            })
        print(f"  t{t}: {n_bands} bands, cuts={diag['cuts']}, "
              f"{diag['n_tied_at_cut']} children tied at a cut "
              f"({diag['pct_tied_at_cut']:.0f}%)", flush=True)

    profile = pd.DataFrame(profile_rows)
    contrast = pd.DataFrame(contrast_rows)
    profile.to_csv(out / f"{tag}_band_profile.csv", index=False)
    contrast.sort_values(["timepoint", "cliffs_delta"], ascending=[True, False]).to_csv(
        out / f"{tag}_band_contrasts.csv", index=False)
    pd.DataFrame(split_diag).to_csv(out / f"{tag}_split_diagnostics.csv", index=False)
    return contrast


def run_monotonicity(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Is each measure's quartile profile monotone, or does it step at one boundary?

    Reports the three adjacent-quartile Cliff's deltas (Q2-Q1, Q3-Q2, Q4-Q3). A
    monotone construct spreads the separation across all three; a threshold shows one
    large step and two near-zero ones. Read against ``pct_at_zero`` — a floored
    measure shows a spurious "threshold" simply because the lower bands are all at the
    floor and cannot separate from each other.
    """
    rng = np.random.default_rng(SEED + 1)
    rows: list[dict] = []
    for t in (1, 2, 3, 4):
        wave = df[df.time == t].copy()
        bands, _ = assign_bands(wave[SPLIT_VAR], 4)
        wave["band"] = bands
        for measure in ALL_MEASURES:
            if measure not in wave.columns:
                continue
            vals = {
                q: wave.loc[wave.band == q, measure].dropna().astype(float).to_numpy()
                for q in ["Q1", "Q2", "Q3", "Q4"]
            }
            if any(len(v) < MIN_BAND for v in vals.values()):
                continue
            steps = {}
            for lo_q, hi_q in (("Q1", "Q2"), ("Q2", "Q3"), ("Q3", "Q4")):
                d, lo, hi = cliffs_delta_ci(vals[hi_q], vals[lo_q], rng)
                steps[f"delta_{hi_q}_{lo_q}"] = d
                steps[f"delta_{hi_q}_{lo_q}_lo"] = lo
                steps[f"delta_{hi_q}_{lo_q}_hi"] = hi
            magnitudes = np.abs([steps["delta_Q2_Q1"], steps["delta_Q3_Q2"],
                                 steps["delta_Q4_Q3"]])
            total = magnitudes.sum()
            rows.append({
                "timepoint": t, "group": GROUP_OF[measure], "measure": measure,
                **steps,
                # Share of the total separation carried by the single largest step.
                # ~0.33 = evenly spread (monotone); ~1.0 = one threshold.
                "largest_step_share": float(magnitudes.max() / total) if total > 0 else np.nan,
                "largest_step_at": ["Q2_Q1", "Q3_Q2", "Q4_Q3"][int(np.argmax(magnitudes))],
                "pct_at_zero": 100.0 * float(
                    (np.concatenate(list(vals.values())) == 0).mean()),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out / "quartile_monotonicity.csv", index=False)
    return df_out


# ---------------------------------------------------------------------------
# P3 — prospective: does the word-reading band predict the SUBSEQUENT change?
# ---------------------------------------------------------------------------


def _censoring(frame: pd.DataFrame, top: pd.DataFrame, bottom: pd.DataFrame,
               measure: str) -> dict[str, float | str]:
    """Where each band is sitting on a bounded scale, and which way that censors it.

    Decisive for reading a NULL prospective contrast. A near-zero ``delta_residual``
    means "the bands moved alike" only if both bands were free to move. If the top
    band starts near the ceiling it cannot gain whatever the truth is, and the
    contrast is **censored, not null** — the letter-sound case in this cohort (top
    half at a median 28 of 32 by t3, a third at >= 30, a ninth already at 32). A
    linear residualisation on the starting level does not repair this: a hard bound
    is not a linear trend. The mirror case is a bottom band pinned at the floor
    (nonword, phonetic spelling), which censors the contrast the other way.
    """
    ceiling = BOUNDED.get(measure)
    if ceiling is None:
        return {"headroom_top": np.nan, "headroom_bottom": np.nan,
                "pct_at_ceiling_top": np.nan, "pct_at_floor_bottom": np.nan,
                "censoring": "unbounded"}
    med_top = float(top["level"].median())
    med_bottom = float(bottom["level"].median())
    pct_ceil_top = 100.0 * float((top["level"] == ceiling).mean())
    pct_floor_bottom = 100.0 * float((bottom["level"] == 0).mean())
    # Headroom as a share of the scale, so 6-item and 92-item measures compare.
    top_squeezed = (ceiling - med_top) / ceiling < 0.25 or pct_ceil_top >= 10
    bottom_pinned = pct_floor_bottom >= 50
    label = (
        "both" if top_squeezed and bottom_pinned
        else "top-censored" if top_squeezed
        else "bottom-censored" if bottom_pinned
        else "clear"
    )
    return {
        "headroom_top": ceiling - med_top,
        "headroom_bottom": ceiling - med_bottom,
        "pct_at_ceiling_top": pct_ceil_top,
        "pct_at_floor_bottom": pct_floor_bottom,
        "censoring": label,
    }


def run_prospective(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Split on word reading at t; contrast each measure's t -> t+1 change.

    Two columns per measure, and the difference between them is the entire point:

    ``delta_raw``       Cliff's delta on the raw change. Contaminated: the bands
                        differ on the measure's own level at t (everything correlates
                        with word reading), so a high band starts higher and has less
                        headroom on a bounded scale. Regression to the mean pushes
                        this negative for any positively-correlated measure.
    ``delta_residual``  The same contrast on the change RESIDUALISED on the measure's
                        own level at t (OLS, within-wave). This is the honest
                        descriptive read of "does word-reading standing predict
                        subsequent movement beyond where the child already was".

    Even ``delta_residual`` is not causal and not adjusted for anything else — age,
    latent ability and the intervention arm all remain inside it. A linear residual
    also cannot fully de-bias a bounded score near its floor or ceiling.
    """
    rng = np.random.default_rng(SEED + 2)
    rows: list[dict] = []
    for t in (1, 2, 3):
        wave = df[df.time == t].copy()
        nxt = df[df.time == t + 1].set_index("subject_id")
        bands, _ = assign_bands(wave[SPLIT_VAR], 2)
        wave["band"] = bands
        wave = wave.set_index("subject_id")

        for measure in ALL_MEASURES:
            if measure in TIME_INVARIANT:
                continue
            if measure not in wave.columns or measure not in nxt.columns:
                continue
            frame = pd.DataFrame({
                "band": wave["band"],
                "level": wave[measure].astype(float),
                "later": nxt[measure].reindex(wave.index).astype(float),
            }).dropna()
            if len(frame) < 2 * MIN_BAND:
                continue
            frame["change"] = frame["later"] - frame["level"]
            # Residualise the change on the measure's own starting level.
            x = frame["level"].to_numpy()
            y = frame["change"].to_numpy()
            if np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                frame["change_resid"] = y - (slope * x + intercept)
            else:
                frame["change_resid"] = y - y.mean()
                slope = np.nan

            top = frame[frame.band == "top 50%"]
            bottom = frame[frame.band == "bottom 50%"]
            if len(top) < MIN_BAND or len(bottom) < MIN_BAND:
                continue
            d_raw, raw_lo, raw_hi = cliffs_delta_ci(
                top["change"].to_numpy(), bottom["change"].to_numpy(), rng)
            d_res, res_lo, res_hi = cliffs_delta_ci(
                top["change_resid"].to_numpy(), bottom["change_resid"].to_numpy(), rng)
            rows.append({
                "from_time": t, "to_time": t + 1,
                "group": GROUP_OF[measure], "measure": measure,
                "n_top": len(top), "n_bottom": len(bottom),
                "median_change_top": float(top["change"].median()),
                "median_change_bottom": float(bottom["change"].median()),
                "median_level_top": float(top["level"].median()),
                "median_level_bottom": float(bottom["level"].median()),
                "delta_raw": d_raw, "delta_raw_lo": raw_lo, "delta_raw_hi": raw_hi,
                "delta_residual": d_res, "delta_residual_lo": res_lo,
                "delta_residual_hi": res_hi,
                "residual_excludes_zero": bool(res_lo > 0 or res_hi < 0),
                # Negative slope = regression to the mean on this measure: children
                # starting higher change less. Quantifies the contamination in delta_raw.
                "rtm_slope": float(slope),
                **_censoring(frame, top, bottom, measure),
            })
        print(f"  t{t} -> t{t + 1}: {sum(r['from_time'] == t for r in rows)} measures",
              flush=True)

    out_df = pd.DataFrame(rows)
    out_df.sort_values(["from_time", "delta_residual"], ascending=[True, False]).to_csv(
        out / "prospective_gain_contrasts.csv", index=False)
    return out_df


# ---------------------------------------------------------------------------
# band membership stability
# ---------------------------------------------------------------------------


def run_stability(df: pd.DataFrame, out: Path) -> None:
    """How much does band membership churn across waves?

    A band contrast is only interesting if the bands mean something durable. Reports
    per-child band membership at each wave for both splits, plus the wave-to-wave
    agreement and the rank correlation of the underlying continuous score. If
    membership churns while the continuous score is stable, that is the tie-breaking
    artefact rather than real movement — read the two together.
    """
    for n_bands, tag in ((2, "half"), (4, "quartile")):
        memb = {}
        for t in (1, 2, 3, 4):
            wave = df[df.time == t]
            bands, _ = assign_bands(wave[SPLIT_VAR], n_bands)
            memb[f"t{t}"] = pd.Series(bands.to_numpy(), index=wave.subject_id.to_numpy())
        table = pd.DataFrame(memb)
        table.index.name = "subject_id"
        table.to_csv(out / f"{tag}_band_membership.csv")

        agree = []
        for a in range(1, 5):
            for b in range(a + 1, 5):
                both = table[[f"t{a}", f"t{b}"]].dropna()
                agree.append({
                    "wave_a": a, "wave_b": b, "n": len(both),
                    "pct_same_band": 100.0 * float(
                        (both[f"t{a}"] == both[f"t{b}"]).mean()),
                })
        pd.DataFrame(agree).to_csv(out / f"{tag}_band_agreement.csv", index=False)

    wide = df.pivot_table(index="subject_id", columns="time", values=SPLIT_VAR)
    wide.columns = [f"t{c}" for c in wide.columns]
    wide.corr(method="spearman").to_csv(out / "word_reading_rank_stability.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("output/notes/202607241600-wr-bands"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print("P1 — median split")
    run_band_contrasts(df, 2, args.out, "half")
    print("P2 — quartile split")
    run_band_contrasts(df, 4, args.out, "quartile")
    run_monotonicity(df, args.out)
    print("P3 — prospective change")
    run_prospective(df, args.out)
    run_stability(df, args.out)
    print(f"Wrote CSVs to {args.out}")


if __name__ == "__main__":
    main()
