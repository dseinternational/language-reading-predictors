# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistent floor-sitters on phonics/nonword: the descriptive data cut.

The cohesive descriptive backing for the floor-sitter note
(``notes/…-persistent-floor-sitters-nonword-spelling.md``, issue #230 §5). It
characterises the children still at floor on nonword reading (``nonword``) and
phonetic spelling (``spphon``) at the final wave: how the floor rate falls over
the four waves, how much of it is a sustained group versus boundary flicker, and
how strongly it tracks incomplete letter-sound prerequisites rather than an
inability to decode.

**Purely descriptive — no models, no causal language.** By the final wave both
arms have been treated, so floor status is prognostic, not evidence of failure.

Unlike the one-figure ``plotNN_*`` scripts in this directory, this is a single
generator that emits the *linked set* of tables and figures the note needs (the
floor-sitter story is one argument across several panels, not an archivable
one-off scatter). It is still standalone: it loads ``data/rli_data_long.csv``
directly, hardcodes its own columns and definitions, and resolves output through
``language_reading_predictors.paths``. Do not refactor its logic into a shared
helper.

All tables are printed to stdout (so the note's inline numbers can be
re-verified) and written as CSVs; figures are written as PNGs. Output goes to
``output/descriptive/`` (gitignored) — commit the script, not the images.

Run::

    python scripts/descriptive/floor_sitters.py
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from language_reading_predictors import paths as _paths

# --- Floored measures (both floor at a score of zero) ---
FLOORED = [("nonword", "Nonword reading"), ("spphon", "Phonetic spelling")]
# Concurrent letter-sound knowledge (YARC-LSK, out of 32) — the prerequisite axis.
L_COL = "yarclet"
# Letter-sound bands used in the prerequisite cross-tab.
L_BANDS = [("L<16", 0, 15), ("16-25", 16, 25), ("26-32", 26, 32)]
# The production-puzzle screen: near-complete letter sounds yet zero nonword.
PUZZLE_L_MIN = 26
TIMEPOINTS = [1, 2, 3, 4]
DPI = 140

# matplotlib palette shared with the other descriptive scripts.
_BLUE, _RED, _GREY = "#4C72B0", "#C44E52", "#8C8C8C"

_DATA_PATH = _paths.DATA_DIR / "rli_data_long.csv"


def _out_dir():
    d = _paths.output_root() / "descriptive"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _band(letter_sounds: float) -> str:
    for name, lo, hi in L_BANDS:
        if lo <= letter_sounds <= hi:
            return name
    return "NA"


def _floor_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Proportion at the floor (score == 0), by wave, of non-missing scores."""
    rows = []
    for col, label in FLOORED:
        for t in TIMEPOINTS:
            s = df.loc[df.time == t, col]
            nn = int(s.notna().sum())
            z = int((s == 0).sum())
            rows.append(
                {
                    "measure": label,
                    "time": t,
                    "at_floor": z,
                    "n": nn,
                    "floor_rate": (z / nn) if nn else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _completer_breakdown(df: pd.DataFrame, col: str) -> dict:
    """Sustained/flicker/persistent split among four-wave completers on ``col``."""
    wide = df.pivot_table(index="subject_id", columns="time", values=col)
    completers = wide.dropna(how="any")
    tmax = max(TIMEPOINTS)
    never_off = (completers == 0).all(axis=1)
    ever_off = (completers > 0).any(axis=1)
    at_floor_t4 = completers[tmax] == 0
    flicker = ever_off & at_floor_t4  # came off, fell back to the floor by t4
    off_at_t4 = completers[tmax] > 0
    return {
        "completers": int(len(completers)),
        "never_off_floor": int(never_off.sum()),
        "flicker_back_to_floor": int(flicker.sum()),
        "off_floor_at_t4": int(off_at_t4.sum()),
    }


def _band_crosstab(t4: pd.DataFrame) -> pd.DataFrame:
    """t4 nonword-floor rate by concurrent letter-sound band."""
    sub = t4.dropna(subset=["nonword", L_COL]).copy()
    sub["band"] = sub[L_COL].apply(_band)
    tab = (
        sub.assign(at_floor=(sub["nonword"] == 0))
        .groupby("band")["at_floor"]
        .agg(at_floor="sum", n="count")
    )
    order = [name for name, _, _ in L_BANDS]
    return tab.reindex(order)


def _print_and_save(df: pd.DataFrame, name: str, out_dir) -> None:
    print(f"\n=== {name} ===")
    print(df.to_string(index=False) if isinstance(df, pd.DataFrame) else df)
    if isinstance(df, pd.DataFrame):
        df.to_csv(out_dir / f"floor_sitters_{name}.csv", index=False)


def _fig_floor_trajectories(rates: pd.DataFrame, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (col, label), colour in zip(FLOORED, (_RED, _BLUE), strict=True):
        sub = rates[rates.measure == label].sort_values("time")
        ax.plot(
            sub["time"], sub["floor_rate"] * 100, marker="o",
            linewidth=2.5, color=colour, label=label,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                f"{r.at_floor:.0f}/{r.n:.0f}", (r.time, r.floor_rate * 100),
                textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8,
            )
    ax.set_xticks(TIMEPOINTS)
    ax.set_xlabel("timepoint")
    ax.set_ylabel("percent at floor (score = 0)")
    ax.set_ylim(0, 100)
    ax.set_title("Floor rate over the four waves")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "floor_sitters_fig1_floor_trajectories.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"wrote {p}")


def _fig_band_crosstab(tab: pd.DataFrame, out_dir) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rate = (tab["at_floor"] / tab["n"] * 100).to_numpy()
    ax.bar(range(len(tab)), rate, color=_BLUE)
    for i, (_, r) in enumerate(tab.iterrows()):
        ax.annotate(
            f"{r.at_floor:.0f}/{r.n:.0f}", (i, r.at_floor / r.n * 100),
            textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9,
        )
    ax.set_xticks(range(len(tab)))
    ax.set_xticklabels(tab.index)
    ax.set_xlabel("concurrent letter-sound knowledge (YARC-LSK, /32)")
    ax.set_ylabel("percent at nonword floor")
    ax.set_ylim(0, 100)
    ax.set_title("t4 nonword floor by letter-sound band")
    fig.tight_layout()
    p = out_dir / "floor_sitters_fig2_lettersound_band.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"wrote {p}")


def _fig_child_trajectories(df: pd.DataFrame, out_dir) -> None:
    """Per-child nonword trajectories, coloured by completer category."""
    wide = df.pivot_table(index="subject_id", columns="time", values="nonword")
    completers = wide.dropna(how="any")
    tmax = max(TIMEPOINTS)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sid, row in completers.iterrows():
        if (row == 0).all():
            colour, z = _RED, 3  # never off floor
        elif row[tmax] == 0:
            colour, z = "#DD8452", 2  # flicker: off then back
        else:
            colour, z = _GREY, 1  # off floor at t4
        ax.plot(
            TIMEPOINTS, row.reindex(TIMEPOINTS),
            color=colour, alpha=0.6, linewidth=1, zorder=z,
        )
    handles = [
        plt.Line2D([], [], color=_RED, label="never off floor"),
        plt.Line2D([], [], color="#DD8452", label="off then back to floor (flicker)"),
        plt.Line2D([], [], color=_GREY, label="off floor at t4"),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xticks(TIMEPOINTS)
    ax.set_xlabel("timepoint")
    ax.set_ylabel("nonword reading (score, /6)")
    ax.set_title("Per-child nonword trajectories (four-wave completers)")
    fig.tight_layout()
    p = out_dir / "floor_sitters_fig3_child_trajectories.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print(f"wrote {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Override the output root for this run (highest precedence, above "
            "DSE_LRP_OUTPUT_DIR). Default: repo-local output/."
        ),
    )
    args = parser.parse_args()
    _paths.set_output_root(args.output_dir)
    print(f"Output root: {_paths.describe_output_root()}")

    df = pd.read_csv(_DATA_PATH)
    tmax = max(TIMEPOINTS)
    out_dir = _out_dir()

    # 1. Floor-rate trajectories.
    rates = _floor_rates(df)
    _print_and_save(rates, "floor_rates_by_wave", out_dir)

    # 2. t4 overlap and arm balance.
    t4 = df[df.time == tmax].copy()
    # Overlap and arm balance use NON-MISSING scores only, so a missing t4 score is not
    # silently counted as "not at floor" and inflating the denominator (consistent with
    # the per-wave floor-rate denominators above; #293 review).
    both_obs = t4.dropna(subset=["nonword", "spphon"])
    both = int(((both_obs.nonword == 0) & (both_obs.spphon == 0)).sum())
    print(
        f"\n=== t4 overlap ===\nat floor on BOTH nonword and spphon: {both} "
        f"(of {len(both_obs)} children with both scores observed)"
    )
    arm_rows = []
    for col, label in FLOORED:
        obs = t4.dropna(subset=[col])
        agg = (
            obs.assign(at_floor=(obs[col] == 0))
            .groupby("group")["at_floor"]
            .agg(at_floor="sum", n="count")
        )
        for grp, r in agg.iterrows():
            arm_rows.append(
                {"measure": label, "group": int(grp),
                 "at_floor": int(r.at_floor), "n": int(r.n)}
            )
    arm = pd.DataFrame(arm_rows)
    print("\n=== t4 floor by arm (group 1 = immediate, group 2 = waitlist; observed only) ===")
    print(arm.to_string(index=False))

    # 3. Sustained / flicker / persistent breakdown (both measures).
    breakdown = pd.DataFrame(
        {label: _completer_breakdown(df, col) for col, label in FLOORED}
    ).T.reset_index(names="measure")
    _print_and_save(breakdown, "completer_breakdown", out_dir)

    # 4. Letter-sound prerequisite cross-tab (nonword at t4).
    band = _band_crosstab(t4).reset_index()
    _print_and_save(band, "nonword_floor_by_lettersound_band", out_dir)

    # 5. Floored-children profile (t4 nonword floor).
    fl = t4.dropna(subset=["nonword"])
    fl = fl[fl.nonword == 0]
    profile = pd.DataFrame(
        [
            {
                "n_floored": int(len(fl)),
                "median_lettersounds_/32": float(fl[L_COL].median()),
                "median_blending_/10": float(fl.blending.median()),
                "reaching_blending_ge6": int((fl.blending >= 6).sum()),
                "median_wordreading_/79": float(fl.ewrswr.median()),
                "max_wordreading_/79": float(fl.ewrswr.max()),
            }
        ]
    )
    _print_and_save(profile, "floored_child_profile", out_dir)

    # 6. Production-puzzle screen: near-complete letter sounds yet zero nonword.
    puzzle = t4[(t4[L_COL] >= PUZZLE_L_MIN) & (t4.nonword == 0)][
        ["subject_id", L_COL, "blending", "ewrswr", "nonword", "spphon"]
    ]
    _print_and_save(puzzle.reset_index(drop=True), "production_puzzle", out_dir)

    # 7. Reconcile completer counts against the full-sample *_none indicators.
    recon = pd.DataFrame(
        [
            {
                "indicator": f"{col}_none == 1 (full-sample never-off flag)",
                "n_subjects": int(df.groupby("subject_id")[f"{col}_none"].max().eq(1).sum()),
            }
            for col in ("nonword", "spphon")
        ]
    )
    _print_and_save(recon, "none_indicator_reconciliation", out_dir)

    # Figures.
    _fig_floor_trajectories(rates, out_dir)
    _fig_band_crosstab(_band_crosstab(t4), out_dir)
    _fig_child_trajectories(df, out_dir)


if __name__ == "__main__":
    main()
