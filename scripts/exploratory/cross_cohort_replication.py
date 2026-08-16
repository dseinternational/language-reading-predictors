# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Matched RLI/Byrne cross-cohort replication for issue #409.

The script deliberately uses the same simple estimator in each cohort rather than
placing unlike fitted-model coefficients on one axis. It estimates three
associations:

1. baseline age -> later word reading, adjusting for baseline word reading,
   baseline verbal memory and study-group indicators;
2. baseline verbal memory -> later word reading from that same model; and
3. the stable between-child correlation of word reading with receptive vocabulary,
   after removing study-group-by-wave means over the common first three waves.

Continuous regression variables are standardised within study and the bounded
scores use a Haldane-Anscombe corrected logit. Uncertainty is an 89% equal-tailed
interval from a non-parametric bootstrap of children, stratified by study group.
These are exploratory associations, not causal effects. The cohorts use different
instruments and follow-up spans, so directions may be compared but magnitudes must
not be pooled. The Byrne source-lineage discrepancy (96 raw-export participants
versus 97 in the prepared extract) also keeps every cross-cohort output explicitly
non-publication-ready.

Run::

    python scripts/exploratory/cross_cohort_replication.py

Outputs are written beneath ``output/exploratory/cross_cohort/`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from language_reading_predictors import figure_io, paths
from language_reading_predictors.data_variables import Categories
from language_reading_predictors.statistical_models.datasets import (
    RLM_DATASET,
    RLM_MEASURES,
)
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.preprocessing import logit_safe

INTERVAL_MASS = 0.89
DEFAULT_BOOTSTRAPS = 2_000
DEFAULT_SEED = 20260816


@dataclass(frozen=True)
class CohortSpec:
    """Columns, waves and input-status metadata for one matched cohort analysis."""

    study_id: str
    study_label: str
    path: Path
    subject_col: str
    wave_col: str
    group_col: str
    group_labels: dict[int, str]
    baseline_wave: int
    followup_wave: int
    stable_waves: tuple[int, ...]
    reading_col: str
    reading_ceiling: int
    reading_label: str
    vocabulary_col: str
    vocabulary_ceiling: int
    vocabulary_label: str
    memory_col: str
    memory_label: str
    measures_confirmed: bool
    source_provenance_confirmed: bool
    source_note: str


RLI = CohortSpec(
    study_id="rli",
    study_label="RLI intervention study",
    path=paths.DATA_DIR / "rli_data_long.csv",
    subject_col="subject_id",
    wave_col="time",
    group_col="group",
    group_labels=dict(Categories.GROUP),
    baseline_wave=1,
    followup_wave=4,
    stable_waves=(1, 2, 3),
    reading_col=MEASURES["W"].column,
    reading_ceiling=MEASURES["W"].n_trials,
    reading_label=MEASURES["W"].label,
    vocabulary_col=MEASURES["R"].column,
    vocabulary_ceiling=MEASURES["R"].n_trials,
    vocabulary_label=MEASURES["R"].label,
    memory_col="erbto",
    memory_label="Early Repetition Battery total",
    measures_confirmed=MEASURES["W"].n_trials_confirmed and MEASURES["R"].n_trials_confirmed,
    source_provenance_confirmed=True,
    source_note="Repository intervention-study analysis extract; selected score ceilings are confirmed.",
)

RLM = CohortSpec(
    study_id="rlm",
    study_label="Byrne, MacDonald & Buckley cohort",
    path=RLM_DATASET.path,
    subject_col=RLM_DATASET.subject_col,
    wave_col=RLM_DATASET.wave_col,
    group_col=RLM_DATASET.group_col,
    group_labels=dict(RLM_DATASET.group_labels),
    baseline_wave=1,
    followup_wave=3,
    stable_waves=(1, 2, 3),
    reading_col=RLM_MEASURES["basread"].column,
    reading_ceiling=RLM_MEASURES["basread"].n_trials,
    reading_label=RLM_MEASURES["basread"].label,
    vocabulary_col=RLM_MEASURES["bpvs"].column,
    vocabulary_ceiling=RLM_MEASURES["bpvs"].n_trials,
    vocabulary_label=RLM_MEASURES["bpvs"].label,
    memory_col=RLM_MEASURES["basdig"].column,
    memory_label=RLM_MEASURES["basdig"].label,
    measures_confirmed=all(
        RLM_MEASURES[symbol].n_trials_confirmed
        and RLM_MEASURES[symbol].instrument_identity_confirmed
        for symbol in ("basread", "bpvs", "basdig")
    ),
    source_provenance_confirmed=RLM_DATASET.source_provenance_confirmed,
    source_note=RLM_DATASET.source_provenance_note,
)

COHORTS = (RLI, RLM)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = float(np.std(values, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Cannot standardise a constant or non-finite variable.")
    return (values - float(np.mean(values))) / sd


def _validate_unique_panel(df: pd.DataFrame, spec: CohortSpec) -> None:
    duplicated = df.duplicated([spec.subject_col, spec.wave_col], keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, [spec.subject_col, spec.wave_col]].head().to_dict("records")
        raise ValueError(f"{spec.study_id}: duplicate child-wave rows: {examples}")


def prepare_followup_rows(df: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    """Return one complete-case baseline/follow-up row per child."""
    _validate_unique_panel(df, spec)
    baseline_cols = [
        spec.subject_col,
        spec.group_col,
        "age",
        spec.reading_col,
        spec.memory_col,
    ]
    followup_cols = [spec.subject_col, spec.group_col, spec.reading_col]
    baseline = df.loc[df[spec.wave_col] == spec.baseline_wave, baseline_cols].rename(
        columns={
            spec.group_col: "group",
            "age": "age_raw",
            spec.reading_col: "reading_baseline_raw",
            spec.memory_col: "memory_raw",
        }
    )
    followup = df.loc[df[spec.wave_col] == spec.followup_wave, followup_cols].rename(
        columns={
            spec.group_col: "group_followup",
            spec.reading_col: "reading_followup_raw",
        }
    )
    rows = baseline.merge(followup, on=spec.subject_col, how="inner", validate="one_to_one")
    rows = rows.dropna(
        subset=[
            "group",
            "group_followup",
            "age_raw",
            "reading_baseline_raw",
            "reading_followup_raw",
            "memory_raw",
        ]
    ).copy()
    inconsistent = rows["group"] != rows["group_followup"]
    if inconsistent.any():
        raise ValueError(f"{spec.study_id}: study group changes between baseline and follow-up.")
    rows["reading_baseline_logit"] = logit_safe(rows["reading_baseline_raw"], spec.reading_ceiling)
    rows["reading_followup_logit"] = logit_safe(rows["reading_followup_raw"], spec.reading_ceiling)
    rows = rows.rename(columns={spec.subject_col: "subject_id"})
    return rows[
        [
            "subject_id",
            "group",
            "age_raw",
            "memory_raw",
            "reading_baseline_logit",
            "reading_followup_logit",
        ]
    ].reset_index(drop=True)


def estimate_followup_associations(rows: pd.DataFrame) -> dict[str, float]:
    """Fit the common standardised baseline-adjusted follow-up OLS model."""
    groups = rows["group"].to_numpy()
    levels = np.unique(groups)
    group_terms = [(groups == level).astype(float) for level in levels[1:]]
    columns = [
        np.ones(len(rows)),
        _zscore(rows["reading_baseline_logit"].to_numpy()),
        _zscore(rows["age_raw"].to_numpy()),
        _zscore(rows["memory_raw"].to_numpy()),
        *group_terms,
    ]
    design = np.column_stack(columns)
    if len(rows) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The baseline-adjusted follow-up design matrix is rank deficient.")
    outcome = _zscore(rows["reading_followup_logit"].to_numpy())
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    return {
        "age": float(coefficients[2]),
        "verbal_memory": float(coefficients[3]),
    }


def prepare_stable_children(df: pd.DataFrame, spec: CohortSpec) -> pd.DataFrame:
    """Return one balanced child row containing reading/vocabulary at each wave."""
    _validate_unique_panel(df, spec)
    keep = df[df[spec.wave_col].isin(spec.stable_waves)][
        [
            spec.subject_col,
            spec.wave_col,
            spec.group_col,
            spec.reading_col,
            spec.vocabulary_col,
        ]
    ].copy()
    keep["reading_logit"] = logit_safe(keep[spec.reading_col], spec.reading_ceiling)
    keep["vocabulary_logit"] = logit_safe(keep[spec.vocabulary_col], spec.vocabulary_ceiling)
    if keep.groupby(spec.subject_col)[spec.group_col].nunique(dropna=True).gt(1).any():
        raise ValueError(f"{spec.study_id}: study group changes within the stable-wave window.")

    reading = keep.pivot(index=spec.subject_col, columns=spec.wave_col, values="reading_logit")
    vocabulary = keep.pivot(index=spec.subject_col, columns=spec.wave_col, values="vocabulary_logit")
    reading = reading.reindex(columns=spec.stable_waves)
    vocabulary = vocabulary.reindex(columns=spec.stable_waves)
    groups = keep.groupby(spec.subject_col)[spec.group_col].first()
    complete = reading.notna().all(axis=1) & vocabulary.notna().all(axis=1) & groups.notna()

    children = pd.DataFrame({"subject_id": reading.index[complete], "group": groups.loc[complete].to_numpy()})
    for wave in spec.stable_waves:
        children[f"reading_w{wave}"] = reading.loc[complete, wave].to_numpy()
        children[f"vocabulary_w{wave}"] = vocabulary.loc[complete, wave].to_numpy()
    return children.reset_index(drop=True)


def estimate_stable_correlation(children: pd.DataFrame, waves: tuple[int, ...]) -> dict[str, float]:
    """Correlate child means after removing each group-by-wave mean."""
    groups = children["group"].to_numpy()
    reading = children[[f"reading_w{wave}" for wave in waves]].to_numpy(dtype=float)
    vocabulary = children[[f"vocabulary_w{wave}" for wave in waves]].to_numpy(dtype=float)
    reading_residual = np.empty_like(reading)
    vocabulary_residual = np.empty_like(vocabulary)
    for group in np.unique(groups):
        group_rows = groups == group
        reading_residual[group_rows] = reading[group_rows] - reading[group_rows].mean(axis=0)
        vocabulary_residual[group_rows] = vocabulary[group_rows] - vocabulary[group_rows].mean(axis=0)
    reading_stable = reading_residual.mean(axis=1)
    vocabulary_stable = vocabulary_residual.mean(axis=1)
    correlation = np.corrcoef(reading_stable, vocabulary_stable)[0, 1]
    if not np.isfinite(correlation):
        raise ValueError("The stable-level correlation is not finite.")
    return {"receptive_vocabulary": float(correlation)}


def bootstrap_intervals(
    children: pd.DataFrame,
    estimator: Callable[[pd.DataFrame], dict[str, float]],
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float, int]]:
    """Stratified child bootstrap with equal-tailed 89% percentile intervals."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    groups = children["group"].to_numpy()
    strata = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    draws: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in strata])
        try:
            estimates = estimator(children.iloc[sampled].reset_index(drop=True))
        except ValueError:
            continue
        for key, value in estimates.items():
            if np.isfinite(value):
                draws.setdefault(key, []).append(float(value))

    alpha = (1.0 - INTERVAL_MASS) / 2.0
    intervals: dict[str, tuple[float, float, int]] = {}
    for key, values in draws.items():
        valid = np.asarray(values, dtype=float)
        if len(valid) < np.ceil(0.8 * n_bootstrap):
            raise RuntimeError(f"Only {len(valid)}/{n_bootstrap} valid bootstrap draws for {key}.")
        lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
        intervals[key] = (float(lower), float(upper), len(valid))
    return intervals


def analyse_cohort(
    spec: CohortSpec,
    *,
    n_bootstrap: int,
    followup_rng: np.random.Generator,
    stable_rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Run all three matched estimands for one cohort."""
    df = pd.read_csv(spec.path)
    followup = prepare_followup_rows(df, spec)
    stable = prepare_stable_children(df, spec)
    followup_point = estimate_followup_associations(followup)
    followup_intervals = bootstrap_intervals(
        followup,
        estimate_followup_associations,
        n_bootstrap=n_bootstrap,
        rng=followup_rng,
    )
    def stable_estimator(frame: pd.DataFrame) -> dict[str, float]:
        return estimate_stable_correlation(frame, spec.stable_waves)

    stable_point = stable_estimator(stable)
    stable_intervals = bootstrap_intervals(
        stable,
        stable_estimator,
        n_bootstrap=n_bootstrap,
        rng=stable_rng,
    )

    study_input_ready = spec.measures_confirmed and spec.source_provenance_confirmed
    shared = {
        "study_id": spec.study_id,
        "study": spec.study_label,
        "source_path": str(spec.path.resolve()),
        "source_sha256": _sha256(spec.path),
        "source_provenance_confirmed": spec.source_provenance_confirmed,
        "measure_inputs_confirmed": spec.measures_confirmed,
        "study_input_ready": study_input_ready,
        "comparison_publication_ready": all(
            cohort.measures_confirmed and cohort.source_provenance_confirmed for cohort in COHORTS
        ),
        "input_note": spec.source_note,
        "interval": "89% equal-tailed child-bootstrap percentile interval",
        "n_bootstrap_requested": n_bootstrap,
        "group_adjustment": "study-group fixed indicators; bootstrap stratified by study group",
    }
    rows: list[dict[str, object]] = []
    followup_metadata = {
        **shared,
        "analysis": "baseline-adjusted later word reading",
        "scale": "within-study standardised OLS coefficient",
        "n_children": len(followup),
        "waves": f"{spec.baseline_wave}->{spec.followup_wave}",
        "reading_measure": spec.reading_label,
        "adjustment": f"baseline {spec.reading_label}; age; {spec.memory_label}; study group",
    }
    for key, label in (("age", "baseline age"), ("verbal_memory", "baseline verbal memory")):
        lower, upper, valid = followup_intervals[key]
        rows.append(
            {
                **followup_metadata,
                "estimand": key,
                "estimand_label": label,
                "comparison_measure": "age in months" if key == "age" else spec.memory_label,
                "estimate": followup_point[key],
                "lower_89": lower,
                "upper_89": upper,
                "n_bootstrap_valid": valid,
            }
        )

    lower, upper, valid = stable_intervals["receptive_vocabulary"]
    rows.append(
        {
            **shared,
            "analysis": "stable child-level reading-vocabulary association",
            "estimand": "receptive_vocabulary",
            "estimand_label": "stable receptive-vocabulary correlation",
            "scale": "Pearson correlation of group-by-wave-demeaned child means",
            "n_children": len(stable),
            "waves": ",".join(map(str, spec.stable_waves)),
            "reading_measure": spec.reading_label,
            "comparison_measure": spec.vocabulary_label,
            "adjustment": "study-group-by-wave cell means removed from both measures",
            "estimate": stable_point["receptive_vocabulary"],
            "lower_89": lower,
            "upper_89": upper,
            "n_bootstrap_valid": valid,
        }
    )
    return rows


def run_analysis(*, n_bootstrap: int = DEFAULT_BOOTSTRAPS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return the tidy matched-comparison table for both cohorts."""
    rngs = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2 * len(COHORTS))]
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(COHORTS):
        rows.extend(
            analyse_cohort(
                spec,
                n_bootstrap=n_bootstrap,
                followup_rng=rngs[2 * index],
                stable_rng=rngs[2 * index + 1],
            )
        )
    return pd.DataFrame(rows)


def _forest_plot(results: pd.DataFrame, *, estimand: str, title: str, xlabel: str, output_dir: Path, name: str) -> None:
    subset = results[results["estimand"] == estimand].copy()
    short_labels = {
        "rli": "RLI intervention study",
        "rlm": "Byrne et al. (2002) cohort",
    }
    subset["study_plot_label"] = subset.apply(
        lambda row: f"{short_labels[row['study_id']]} (n = {int(row['n_children'])})"
        + (" *" if not row["study_input_ready"] else ""),
        axis=1,
    )
    y = np.arange(len(subset))[::-1]
    estimates = subset["estimate"].to_numpy(dtype=float)
    lower = subset["lower_89"].to_numpy(dtype=float)
    upper = subset["upper_89"].to_numpy(dtype=float)
    colours = ["#2166ac", "#b2182b"][: len(subset)]
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for estimate, lower_limit, upper_limit, ypos, colour in zip(
        estimates, lower, upper, y, colours, strict=True
    ):
        ax.errorbar(
            estimate,
            ypos,
            xerr=np.array([[estimate - lower_limit], [upper_limit - estimate]]),
            fmt="o",
            color=colour,
            ecolor=colour,
            markersize=7,
            elinewidth=2.0,
            capsize=4,
            zorder=3,
        )
    ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels(subset["study_plot_label"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    for estimate, ypos in zip(estimates, y, strict=True):
        vertical_offset = -20 if ypos == max(y) else 10
        ax.annotate(
            f"{estimate:+.2f}",
            (estimate, ypos),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.subplots_adjust(left=0.32, right=0.98, top=0.80, bottom=0.28)
    fig.text(
        0.01,
        0.025,
        "89% child-bootstrap intervals. * Byrne source lineage unresolved; comparison is not publication-ready.",
        fontsize=8,
    )
    figure_io.save_styled_figure(output_dir.as_posix(), name, fig=fig, data=subset)


def write_outputs(results: pd.DataFrame, output_dir: Path) -> None:
    """Write the tidy table and three deliberately separate forest plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "matched_associations.csv", index=False)
    _forest_plot(
        results,
        estimand="age",
        title="Age and baseline-adjusted later word reading",
        xlabel="Standardised coefficient (within study)",
        output_dir=output_dir,
        name="age_reading_replication",
    )
    _forest_plot(
        results,
        estimand="verbal_memory",
        title="Verbal memory and baseline-adjusted later word reading",
        xlabel="Standardised coefficient (within study)",
        output_dir=output_dir,
        name="memory_reading_replication",
    )
    _forest_plot(
        results,
        estimand="receptive_vocabulary",
        title="Stable child-level receptive-vocabulary and word-reading association",
        xlabel="Pearson correlation",
        output_dir=output_dir,
        name="vocabulary_reading_stable_correlation",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.output_root() / "exploratory" / "cross_cohort",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_analysis(n_bootstrap=args.n_bootstrap, seed=args.seed)
    write_outputs(results, args.output_dir)
    print(results[["study_id", "estimand", "estimate", "lower_89", "upper_89", "n_children"]].to_string(index=False))
    print(f"Wrote matched cross-cohort artefacts to {args.output_dir}")
    if not bool(results["comparison_publication_ready"].all()):
        print(f"NOT PUBLICATION-READY: {RLM.source_note}")


if __name__ == "__main__":
    main()
