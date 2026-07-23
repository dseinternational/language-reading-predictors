# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Descriptive exploratory pass for the Byrne reading-language-memory cohort (#409 item A).

Mirrors the RLI descriptive work for the observational Byrne, MacDonald & Buckley
(2002) cohort (``study_id="rlm"``). Three purely descriptive views, each written as
one figure per file (PNG + SVG + CSV) under ``output/exploratory/rlm/``:

1. **Per-wave correlation matrices** over the battery + age, and the **between-child
   vs within-child decomposition** of the pairwise correlations. The two answer
   different questions -- "do children who read well tend to score well elsewhere"
   (between) vs "is a good year for reading a good year for the other skill"
   (within) -- and are reported separately, as in the RLI strand.
2. **RTM-corrected baseline -> gain partials**: for every predictor-outcome pair and
   group, the association of the predictor's wave-1 level with the outcome's w1->w3
   gain, **conditioning on the outcome's own wave-1 level**. Raw baseline -> gain
   correlations in this design are regression-to-the-mean-confounded by
   construction; the partial is the honest descriptive analogue (the correction that
   flipped the taught-vocabulary reading in the RLI strand, #405).
3. **Within-group age check**: a crude age -> word-reading-gain diagnostic
   (conditioning only on baseline word reading) -- is the pooled association also
   present inside each ``readgrp``, or partly cohort composition? This is *not* a
   reproduction of ``lrp-rlm-adj-001``, which uses a different analytic sample and a
   full covariate adjustment set; it is a descriptive prompt, not that estimate.

Nothing here is causal. ``readgrp`` is an observational cohort factor; every
association is an adjusted/observed descriptive correlate with a residual-confounding
caveat. Correlations use Pearson's r throughout for coherence with the between/within
variance decomposition and the linear RTM residualisation; Spearman gives the same
qualitative pattern and is a cheap robustness check.

Run::

    python scripts/exploratory/rlm_associations.py

Writes to ``output/exploratory/rlm/`` (gitignored); commit the script, not the
figures.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from language_reading_predictors import figure_io, paths
from language_reading_predictors.statistical_models.datasets import (
    RLM_GROUP_LABELS,
    RLM_MEASURES,
)

# Seven core reading/language/memory measures (all with a wave-1 baseline) plus the
# two ability proxies. ``basmat`` is wave-3+ only; ``bassim`` is the sole wave-1
# ability proxy (see #409 D4). Age is carried separately.
CORE_MEASURES = ["basread", "basspel", "woco", "bpvs", "trog", "basdig", "basnum"]
ABILITY_MEASURES = ["bassim", "basmat"]
ALL_MEASURES = CORE_MEASURES + ABILITY_MEASURES

# Measures with a usable wave-1 baseline. ``basmat`` is wave-3+ only, so it has no
# wave-1 level and cannot serve as a baseline predictor (it would be an all-blank row
# in every baseline -> gain matrix); ``bassim`` is the sole wave-1 ability proxy (see
# #409 D4). ``basmat`` still appears in the per-wave and between/within matrices,
# where it is observed at the later waves.
BASELINE_MEASURES = CORE_MEASURES + ["bassim"]

LABELS = {sym: RLM_MEASURES[sym].label for sym in ALL_MEASURES}
LABELS["age"] = "age (months)"

# Baseline wave and the outcome-gain wave for the RTM-corrected partials and the age
# check. w1->w3 matches the fitted ``lrp-rlm-adj-001`` window and keeps every group
# in-sample (the panel runs to w5, but w4/w5 are progressively DS-only -- the later
# waves are #409 item D2, deferred).
BASELINE_WAVE = 1
GAIN_WAVE = 3

MIN_PAIRWISE = 8  # skip a correlation cell with fewer than this many complete pairs

_OUT_DIR = os.path.join(str(paths.output_root()), "exploratory", "rlm")


def load_rlm() -> pd.DataFrame:
    """Load the Byrne reading-language-memory long panel."""
    path = (
        paths.DATA_DIR
        / "reading-language-memory"
        / "reading_language_memory_data_long.csv"
    )
    return pd.read_csv(path)


def _pairwise_corr(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pearson correlation matrix over ``cols`` using pairwise-complete rows, with
    cells backed by fewer than ``MIN_PAIRWISE`` complete pairs blanked to NaN."""
    corr = frame[cols].corr(method="pearson", min_periods=MIN_PAIRWISE)
    return corr.where(_pairwise_n(frame, cols).to_numpy() >= MIN_PAIRWISE)


def _pairwise_n(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Number of pairwise-complete rows behind each cell of the ``cols`` matrix.

    Pairwise-complete cells can each rest on a different n, so these counts are
    written alongside the per-wave matrices rather than a single headline n."""
    counts = frame[cols].notna().astype(float)
    return (counts.T @ counts).astype(int)


def _heatmap(matrix: pd.DataFrame, title: str, name: str) -> None:
    """Save a labelled correlation heatmap (diverging, centred at zero) + its CSV."""
    labelled = matrix.rename(index=LABELS, columns=LABELS)
    n_rows, n_cols = labelled.shape
    fig, ax = plt.subplots(figsize=(1.5 + 0.75 * n_cols, 0.9 + 0.6 * n_rows))
    data = labelled.to_numpy(dtype=float)
    im = ax.imshow(data, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(labelled.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labelled.index, fontsize=8)
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(v) > 0.55 else "black",
                )
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson's r")
    figure_io.save_styled_figure(_OUT_DIR, name, fig=fig)
    # Write the CSV separately with the measure/predictor row labels preserved
    # (save_styled_figure's data= path writes index-free, which drops them).
    figure_io.save_plot_data(_OUT_DIR, name, matrix.round(4), index=True)


def per_wave_matrices(df: pd.DataFrame) -> None:
    """One cross-sectional correlation matrix per wave over the observed battery + age.

    The long file carries a (child, wave) row even where a child has no observation
    at that wave, so the reported n is the number of children with at least one
    observed measure at the wave (the analytic sample), not the placeholder row
    count; the header also names the groups actually represented, because the later
    waves are progressively Down-syndrome-only. Individual cells rest on
    pairwise-complete pairs whose n is smaller and varies by pair -- those counts are
    written to the companion ``per_wave_pair_n_w*`` CSVs."""
    for wave in sorted(df["time"].dropna().unique()):
        sub = df[df["time"] == wave]
        observed = sub[sub[ALL_MEASURES].notna().any(axis=1)]
        cols = [m for m in ALL_MEASURES if observed[m].notna().sum() >= MIN_PAIRWISE]
        cols = cols + ["age"]
        matrix = _pairwise_corr(observed, cols)
        n_children = int(observed["subject_id"].nunique())
        groups = ", ".join(
            RLM_GROUP_LABELS[g]
            for g in sorted(observed["readgrp"].dropna().unique())
            if g in RLM_GROUP_LABELS
        )
        _heatmap(
            matrix,
            f"Wave {int(wave)} level correlations ({groups}; n = {n_children}, pairwise)",
            f"per_wave_corr_w{int(wave)}",
        )
        figure_io.save_plot_data(
            _OUT_DIR, f"per_wave_pair_n_w{int(wave)}", _pairwise_n(observed, cols), index=True
        )


def between_within(df: pd.DataFrame) -> None:
    """Between-child (child means) vs within-child (departures from the child mean)
    correlation matrices over the battery + age.

    Restricted to the prespecified common w1-w3 window: pooling every wave would
    compute child means over different developmental windows (the panel runs to w5,
    but w4/w5 are progressively Down-syndrome-only), so a child's mean level and its
    within-child departures would not be comparable across children. Even within
    w1-w3 children contribute unequal numbers of observed waves, so the within-child
    correlation still weights the better-observed children more heavily -- read these
    as a descriptive decomposition, not a balanced variance partition."""
    df = df[df["time"].between(BASELINE_WAVE, GAIN_WAVE)]
    cols = ALL_MEASURES + ["age"]
    grouped = df.groupby("subject_id")
    child_means = grouped[cols].transform("mean")
    between = df[["subject_id"]].join(child_means).groupby("subject_id").first()
    within = df[cols] - child_means

    between_corr = _pairwise_corr(between, cols)
    within_corr = _pairwise_corr(within, cols)
    _heatmap(
        between_corr,
        "Between-child correlations (child mean levels)",
        "between_child_corr",
    )
    _heatmap(
        within_corr,
        "Within-child correlations (departures from the child mean)",
        "within_child_corr",
    )

    # A focused between-vs-within comparison for the reading-anchored pairs, the RLI
    # strand's headline contrast (reading coupled ~0.82 between vs ~0.48 within).
    rows = []
    for other in [c for c in cols if c != "basread"]:
        rows.append(
            {
                "pair": f"basread-{other}",
                "between_r": between_corr.loc["basread", other],
                "within_r": within_corr.loc["basread", other],
            }
        )
    comp = pd.DataFrame(rows).dropna(subset=["between_r", "within_r"])
    comp = comp.sort_values("between_r", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.5 + 0.5 * len(comp)))
    y = np.arange(len(comp))
    ax.scatter(comp["between_r"], y, label="between-child", color="#2166ac", zorder=3)
    ax.scatter(comp["within_r"], y, label="within-child", color="#b2182b", zorder=3)
    for yi, (_, r) in zip(y, comp.iterrows(), strict=True):
        ax.plot([r["within_r"], r["between_r"]], [yi, yi], color="#999999", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(p.split("-")[1], p) for p in comp["pair"]], fontsize=8)
    ax.axvline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Pearson's r with word reading (basread)")
    ax.set_title("Reading coupling: between-child vs within-child")
    ax.legend(loc="lower right", fontsize=8)
    figure_io.save_styled_figure(
        _OUT_DIR, "reading_between_vs_within", fig=fig, data=comp.round(4)
    )


def _partial_corr(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, int]:
    """Partial Pearson correlation of ``x`` and ``y`` given ``z`` (residualise each
    on ``[1, z]`` by least squares, then correlate). Returns (r, n)."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if int(mask.sum()) < MIN_PAIRWISE:
        return float("nan"), int(mask.sum())
    xz, yz, zz = x[mask], y[mask], z[mask]
    design = np.column_stack([np.ones_like(zz), zz])
    rx = xz - design @ np.linalg.lstsq(design, xz, rcond=None)[0]
    ry = yz - design @ np.linalg.lstsq(design, yz, rcond=None)[0]
    if np.std(rx) < 1e-9 or np.std(ry) < 1e-9:
        return float("nan"), int(mask.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(mask.sum())


def _child_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per child: each measure + age at the baseline wave and its w1->w3 gain."""
    base = df[df["time"] == BASELINE_WAVE].set_index("subject_id")
    later = df[df["time"] == GAIN_WAVE].set_index("subject_id")
    out = pd.DataFrame(index=base.index)
    out["readgrp"] = base["readgrp"]
    out["age"] = base["age"]
    for m in ALL_MEASURES:
        out[f"{m}_base"] = base[m]
        out[f"{m}_gain"] = later[m] - base[m]
    return out


def rtm_partials(df: pd.DataFrame) -> None:
    """RTM-corrected partial-correlation matrices (predictor baseline -> outcome gain,
    conditioning on the outcome's own baseline) alongside the raw baseline -> gain
    correlations, per group and pooled."""
    wide = _child_wide(df)
    predictors = BASELINE_MEASURES + ["age"]  # basmat has no wave-1 baseline
    outcomes = CORE_MEASURES  # gains in the core battery
    groups = [(None, "pooled")] + [(g, RLM_GROUP_LABELS[g]) for g in sorted(RLM_GROUP_LABELS)]

    for gid, gname in groups:
        frame = wide if gid is None else wide[wide["readgrp"] == gid]
        partial = pd.DataFrame(index=predictors, columns=outcomes, dtype=float)
        raw = pd.DataFrame(index=predictors, columns=outcomes, dtype=float)
        for out_m in outcomes:
            y = frame[f"{out_m}_gain"].to_numpy(dtype=float)
            y_base = frame[f"{out_m}_base"].to_numpy(dtype=float)
            for pred in predictors:
                pred_col = "age" if pred == "age" else f"{pred}_base"
                x = frame[pred_col].to_numpy(dtype=float)
                partial.loc[pred, out_m], _ = _partial_corr(x, y, y_base)
                mask = np.isfinite(x) & np.isfinite(y)
                raw.loc[pred, out_m] = (
                    float(np.corrcoef(x[mask], y[mask])[0, 1])
                    if int(mask.sum()) >= MIN_PAIRWISE
                    else float("nan")
                )
        slug = "pooled" if gid is None else f"group{gid}"
        n_children = int(frame.shape[0])
        _heatmap(
            partial,
            f"RTM-corrected baseline -> w1->w3 gain partials, {gname} (n = {n_children})",
            f"rtm_partial_{slug}",
        )
        _heatmap(
            raw,
            f"Raw baseline -> w1->w3 gain correlations, {gname} (n = {n_children})",
            f"raw_baseline_gain_{slug}",
        )


def age_within_group(df: pd.DataFrame) -> None:
    """Crude age -> word-reading-gain diagnostic, pooled vs within each group.

    Reports the raw and RTM-corrected age->gain correlation, where "corrected"
    conditions only on baseline word reading. This is a deliberately crude,
    child-level (between-child within each group) descriptive check -- not a
    reproduction of ``lrp-rlm-adj-001``, whose analytic sample and adjustment set
    (bpvs, trog, basdig, bassim, basnum and group nuisance terms) both differ."""
    wide = _child_wide(df)
    rows = []
    groups = [(None, "pooled")] + [(g, RLM_GROUP_LABELS[g]) for g in sorted(RLM_GROUP_LABELS)]
    for gid, gname in groups:
        frame = wide if gid is None else wide[wide["readgrp"] == gid]
        age = frame["age"].to_numpy(dtype=float)
        gain = frame["basread_gain"].to_numpy(dtype=float)
        base = frame["basread_base"].to_numpy(dtype=float)
        partial_r, n = _partial_corr(age, gain, base)
        mask = np.isfinite(age) & np.isfinite(gain)
        raw_r = (
            float(np.corrcoef(age[mask], gain[mask])[0, 1])
            if int(mask.sum()) >= MIN_PAIRWISE
            else float("nan")
        )
        rows.append(
            {"group": gname, "n": n, "raw_r": raw_r, "rtm_partial_r": partial_r}
        )
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    y = np.arange(len(table))
    ax.scatter(table["raw_r"], y, label="raw", color="#999999", zorder=3)
    ax.scatter(
        table["rtm_partial_r"], y, label="RTM-corrected", color="#b2182b", zorder=3
    )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.group} (n={r.n})" for r in table.itertuples()], fontsize=9)
    ax.axvline(0.0, color="black", lw=0.8, ls=":")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("correlation of baseline age with w1->w3 word-reading gain")
    ax.set_title("Age -> reading-gain signal: pooled vs within group")
    ax.legend(loc="lower right", fontsize=8)
    figure_io.save_styled_figure(
        _OUT_DIR, "age_within_group_check", fig=fig, data=table.round(4)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.makedirs(_OUT_DIR, exist_ok=True)
    df = load_rlm()
    per_wave_matrices(df)
    between_within(df)
    rtm_partials(df)
    age_within_group(df)
    print(f"Wrote RLM exploratory figures to {_OUT_DIR}")


if __name__ == "__main__":
    main()
