# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Summary builders for the historical growth models (RLMHG, #165).

Pure functions that turn a fitted historical-growth trace + its
:class:`preprocessing.LongitudinalPanel` into the report CSVs:

- :func:`observed_baseline` - the observed group-by-wave means/SDs/counts (on
  the complete-case core window this is the audit baseline, equal to the
  paper's Table 2 complete-case reproduction for the modelled measure;
  extension waves (#338) report their own attrition-selected per-cell ``n``);
- :func:`cell_summary` - posterior fitted means vs the observed baseline, one
  row per supported (group, wave) cell;
- :func:`growth_summary` - within-group interval growth on common-subject
  cells and pairwise total-growth contrasts over the common window.

These are ports of the summary logic in the (now-removed) standalone
``scripts/fit_historical_growth_model.py``, adapted to the package trace /
panel objects so the historical model runs through the shared pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.preprocessing import (
    LongitudinalPanel,
)


def _summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "q_lo": float(np.quantile(values, 0.055)),
        "q50": float(np.quantile(values, 0.5)),
        "q_hi": float(np.quantile(values, 0.945)),
        "p_gt_0": float(np.mean(values > 0)),
    }


def observed_baseline(
    panel: LongitudinalPanel, measure: str, measure_label: str
) -> pd.DataFrame:
    """Observed group-by-wave means/SDs/counts for ``measure``.

    One row per group with ``time_{w}_mean`` / ``time_{w}_sd`` / ``time_{w}_n``
    columns over the panel's full window. On the complete-case **core** waves
    this matches the Table 2 complete-case audit baseline (``n_complete``
    subjects per group); extension waves (#338) are an attrition-selected
    follow-up tail, so their per-wave ``n`` shrinks and unsupported cells (e.g.
    non-Down-syndrome groups at the Byrne wave 5) are left as NaN with ``n=0``.
    """
    df = panel.long
    label_col = panel.group_label_col
    wave_c = panel.dataset.wave_col
    rows: list[dict[str, Any]] = []
    for code, label in zip(panel.group_codes, panel.group_labels, strict=True):
        part = df[df[label_col] == label]
        core = part[part[wave_c].isin(panel.waves)]
        n_complete = int(core[panel.dataset.subject_col].nunique())
        row: dict[str, Any] = {
            "measure": measure,
            "measure_label": measure_label,
            "readgrp": code,
            "readgrp_label": label,
            "n_complete": n_complete,
        }
        for wave in panel.all_waves:
            values = part.loc[part[wave_c] == wave, measure].dropna()
            row[f"time_{wave}_n"] = int(len(values))
            row[f"time_{wave}_mean"] = (
                float(values.mean()) if len(values) else float("nan")
            )
            row[f"time_{wave}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_obs_index(
    panel: LongitudinalPanel, label: str, wave: int, subjects: set | None = None
) -> np.ndarray:
    """Row indices of one (group, wave) cell, optionally restricted to ``subjects``."""
    df = panel.long
    mask = (df[panel.group_label_col] == label) & (
        df[panel.dataset.wave_col] == wave
    )
    if subjects is not None:
        mask &= df[panel.dataset.subject_col].isin(subjects)
    return df.index[mask].to_numpy()


def _cell_values(
    posterior,
    panel: LongitudinalPanel,
    label: str,
    wave: int,
    subjects: set | None = None,
    fitted_var: str = "fitted_mean_items_obs",
) -> np.ndarray:
    """Posterior draws of the fitted mean count for one (group, wave) cell.

    ``subjects`` restricts the cell to a subject subset - used to compute
    interval growth on the children observed at **both** endpoint waves, so the
    per-child offsets cancel exactly and attrition at an extension wave (#338)
    cannot masquerade as growth.
    """
    fitted = posterior[fitted_var].stack(sample=("chain", "draw"))
    obs_idx = _cell_obs_index(panel, label, wave, subjects)
    if len(obs_idx) == 0:
        raise ValueError(f"No observations for group={label!r}, wave={wave!r}.")
    return fitted.isel(obs=obs_idx).mean(dim="obs").values


def _cell_subjects(panel: LongitudinalPanel, label: str, wave: int) -> set:
    """Subjects observed in one (group, wave) cell."""
    df = panel.long
    mask = (df[panel.group_label_col] == label) & (
        df[panel.dataset.wave_col] == wave
    )
    return set(df.loc[mask, panel.dataset.subject_col])


def _group_waves(panel: LongitudinalPanel, measure: str) -> dict[str, list[int]]:
    """Supported waves per group label, from the panel's observed cells."""
    code_to_label = dict(zip(panel.group_codes, panel.group_labels, strict=True))
    out: dict[str, list[int]] = {label: [] for label in panel.group_labels}
    for code, wave in panel.cells(measure):
        out[code_to_label[code]].append(wave)
    return {label: sorted(waves) for label, waves in out.items()}


def _common_waves(panel: LongitudinalPanel, measure: str) -> list[int]:
    """Waves supported in every group (the between-group window)."""
    per_group = _group_waves(panel, measure)
    common = set(panel.all_waves)
    for waves in per_group.values():
        common &= set(waves)
    return sorted(common)


def cell_summary(
    trace,
    panel: LongitudinalPanel,
    measure: str,
    measure_label: str,
    baseline: pd.DataFrame,
    mean_var: str = "mean_items",
    fitted_var: str = "fitted_mean_items_obs",
) -> pd.DataFrame:
    """Posterior fitted group-by-wave means vs the observed baseline.

    One row per **supported** (group, wave) cell. ``window`` marks whether the
    cell is on the complete-case core window (the Table 2 audit cells) or the
    attrition-selected extension tail (#338); ``n_obs`` is the cell's own count.
    """
    posterior = trace.posterior
    population = posterior[mean_var].stack(sample=("chain", "draw"))
    cells = panel.cells(measure)
    cell_pos = {cell: i for i, cell in enumerate(cells)}
    code_to_label = dict(zip(panel.group_codes, panel.group_labels, strict=True))
    rows: list[dict[str, Any]] = []
    for code, wave in cells:
        label = code_to_label[code]
        audit_row = baseline[baseline["readgrp_label"] == label]
        if len(audit_row) != 1:
            raise ValueError(f"Expected one baseline row for group {label!r}.")
        audit_row = audit_row.iloc[0]
        fitted = _summarize(
            _cell_values(posterior, panel, label, wave, fitted_var=fitted_var)
        )
        pop = _summarize(population.isel(cell=cell_pos[(code, wave)]).values)
        observed_mean = float(audit_row[f"time_{wave}_mean"])
        rows.append(
            {
                "measure": measure,
                "measure_label": measure_label,
                "readgrp_label": label,
                "time": int(wave),
                "window": "core" if wave in panel.waves else "extension",
                "n_complete": int(audit_row["n_complete"]),
                "n_obs": int(audit_row[f"time_{wave}_n"]),
                "observed_complete_case_mean": observed_mean,
                "observed_complete_case_sd": float(audit_row[f"time_{wave}_sd"]),
                "posterior_mean_minus_observed_mean": fitted["mean"] - observed_mean,
                "posterior_population_mean": pop["mean"],
                "posterior_population_q_lo": pop["q_lo"],
                "posterior_population_q_hi": pop["q_hi"],
                **{f"posterior_{k}": v for k, v in fitted.items()},
            }
        )
    return pd.DataFrame(rows)


def growth_summary(
    trace,
    panel: LongitudinalPanel,
    measure: str,
    fitted_var: str = "fitted_mean_items_obs",
) -> pd.DataFrame:
    """Within-group interval growth + pairwise total-growth contrasts (in items).

    Every interval is computed on the children observed at **both** endpoint
    waves (``n_subjects``), so extension-wave attrition (#338) cannot leak into
    the growth read-out; on the complete-case core window this restriction is a
    no-op. Within-group intervals cover each group's own supported window (the
    Down-syndrome group's extension tail included); the pairwise total-growth
    contrasts are taken over the **common window** - the first-to-last wave
    every group supports - so they compare like horizons.
    """
    posterior = trace.posterior
    rows: list[dict[str, Any]] = []
    per_group = _group_waves(panel, measure)
    common = _common_waves(panel, measure)

    def _interval(label: str, start: int, end: int) -> dict[str, Any]:
        subjects = _cell_subjects(panel, label, start) & _cell_subjects(
            panel, label, end
        )
        start_vals = _cell_values(
            posterior, panel, label, start, subjects, fitted_var=fitted_var
        )
        end_vals = _cell_values(
            posterior, panel, label, end, subjects, fitted_var=fitted_var
        )
        window = (
            "core"
            if (start in panel.waves and end in panel.waves)
            else "extension"
        )
        return {
            "readgrp_label": label,
            "window": window,
            "n_subjects": len(subjects),
            **_summarize(end_vals - start_vals),
        }

    # Within-group growth across consecutive supported intervals and
    # first-to-last of the group's own window.
    for label in panel.group_labels:
        waves = per_group[label]
        intervals = [(waves[i], waves[i + 1]) for i in range(len(waves) - 1)]
        if len(waves) > 2:
            intervals.append((waves[0], waves[-1]))
        for start, end in intervals:
            rows.append(
                {
                    "quantity": f"growth_{start}_{end}_items",
                    "label": f"Wave {start} to wave {end}",
                    **_interval(label, start, end),
                }
            )
    # Pairwise total growth contrasts between groups, over the common window.
    if len(common) >= 2:
        start, end = common[0], common[-1]
        total = {}
        for label in panel.group_labels:
            subjects = _cell_subjects(panel, label, start) & _cell_subjects(
                panel, label, end
            )
            total[label] = _cell_values(
                posterior, panel, label, end, subjects, fitted_var=fitted_var
            ) - _cell_values(
                posterior, panel, label, start, subjects, fitted_var=fitted_var
            )
        labels = panel.group_labels
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                rows.append(
                    {
                        "quantity": f"total_growth_{b}_minus_{a}",
                        "label": (
                            f"Total growth (waves {start}-{end}): {b} minus {a}"
                        ),
                        "readgrp_label": "",
                        "window": (
                            "core"
                            if (start in panel.waves and end in panel.waves)
                            else "extension"
                        ),
                        "n_subjects": pd.NA,
                        **_summarize(total[b] - total[a]),
                    }
                )
    return pd.DataFrame(rows)
