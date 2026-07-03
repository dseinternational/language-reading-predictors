# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Summary builders for the historical growth models (RLMHG, #165).

Pure functions that turn a fitted historical-growth trace + its
:class:`preprocessing.LongitudinalPanel` into the report CSVs:

- :func:`observed_baseline` - the complete-case observed group-by-wave means/SDs
  (the audit baseline; equal to the paper's Table 2 complete-case reproduction
  for the modelled measure, computed straight from the complete-case panel);
- :func:`cell_summary` - posterior fitted group-by-wave means vs the observed
  baseline;
- :func:`growth_summary` - within-group interval growth and pairwise
  total-growth contrasts.

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
        "q2_5": float(np.quantile(values, 0.025)),
        "q50": float(np.quantile(values, 0.5)),
        "q97_5": float(np.quantile(values, 0.975)),
        "p_gt_0": float(np.mean(values > 0)),
    }


def observed_baseline(
    panel: LongitudinalPanel, measure: str, measure_label: str
) -> pd.DataFrame:
    """Observed complete-case group-by-wave means/SDs for ``measure``.

    One row per group with ``time_{w}_mean`` / ``time_{w}_sd`` columns, matching
    the Table 2 complete-case audit baseline. Computed directly from the panel's
    complete-case frame (identical to a per-measure complete-case selection).
    """
    df = panel.long
    label_col = panel.group_label_col
    wave_c = panel.dataset.wave_col
    rows: list[dict[str, Any]] = []
    for code, label in zip(panel.group_codes, panel.group_labels, strict=True):
        part = df[df[label_col] == label]
        n_complete = int(part[panel.dataset.subject_col].nunique())
        row: dict[str, Any] = {
            "measure": measure,
            "measure_label": measure_label,
            "readgrp": code,
            "readgrp_label": label,
            "n_complete": n_complete,
        }
        for wave in panel.waves:
            values = part.loc[part[wave_c] == wave, measure]
            row[f"time_{wave}_mean"] = float(values.mean())
            row[f"time_{wave}_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_values(posterior, panel: LongitudinalPanel, label: str, wave: int) -> np.ndarray:
    """Posterior draws of the fitted mean count for one (group, wave) cell."""
    fitted = posterior["fitted_mean_items_obs"].stack(sample=("chain", "draw"))
    df = panel.long
    obs_idx = df.index[
        (df[panel.group_label_col] == label) & (df[panel.dataset.wave_col] == wave)
    ].to_numpy()
    if len(obs_idx) == 0:
        raise ValueError(f"No observations for group={label!r}, wave={wave!r}.")
    return fitted.isel(obs=obs_idx).mean(dim="obs").values


def cell_summary(
    trace,
    panel: LongitudinalPanel,
    measure: str,
    measure_label: str,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """Posterior fitted group-by-wave means vs the observed baseline."""
    posterior = trace.posterior
    population = posterior["mean_items"].stack(sample=("chain", "draw"))
    rows: list[dict[str, Any]] = []
    for label in panel.group_labels:
        audit_row = baseline[baseline["readgrp_label"] == label]
        if len(audit_row) != 1:
            raise ValueError(f"Expected one baseline row for group {label!r}.")
        audit_row = audit_row.iloc[0]
        for wave in panel.waves:
            fitted = _summarize(_cell_values(posterior, panel, label, wave))
            pop = _summarize(population.sel(group=label, wave=wave).values)
            observed_mean = float(audit_row[f"time_{wave}_mean"])
            rows.append(
                {
                    "measure": measure,
                    "measure_label": measure_label,
                    "readgrp_label": label,
                    "time": int(wave),
                    "n_complete": int(audit_row["n_complete"]),
                    "observed_complete_case_mean": observed_mean,
                    "observed_complete_case_sd": float(audit_row[f"time_{wave}_sd"]),
                    "posterior_mean_minus_observed_mean": fitted["mean"] - observed_mean,
                    "posterior_population_mean": pop["mean"],
                    "posterior_population_q2_5": pop["q2_5"],
                    "posterior_population_q97_5": pop["q97_5"],
                    **{f"posterior_{k}": v for k, v in fitted.items()},
                }
            )
    return pd.DataFrame(rows)


def growth_summary(trace, panel: LongitudinalPanel, measure: str) -> pd.DataFrame:
    """Within-group interval growth + pairwise total-growth contrasts (in items)."""
    posterior = trace.posterior
    waves = list(panel.waves)
    rows: list[dict[str, Any]] = []
    # Within-group growth across consecutive intervals and first-to-last.
    intervals = [(waves[i], waves[i + 1]) for i in range(len(waves) - 1)]
    if len(waves) >= 2:
        intervals.append((waves[0], waves[-1]))
    for start, end in intervals:
        for label in panel.group_labels:
            start_vals = _cell_values(posterior, panel, label, start)
            end_vals = _cell_values(posterior, panel, label, end)
            rows.append(
                {
                    "quantity": f"growth_{start}_{end}_items",
                    "label": f"Wave {start} to wave {end}",
                    "readgrp_label": label,
                    **_summarize(end_vals - start_vals),
                }
            )
    # Pairwise total (first-to-last) growth contrasts between groups.
    if len(waves) >= 2:
        total = {
            label: _cell_values(posterior, panel, label, waves[-1])
            - _cell_values(posterior, panel, label, waves[0])
            for label in panel.group_labels
        }
        labels = panel.group_labels
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                rows.append(
                    {
                        "quantity": f"total_growth_{b}_minus_{a}",
                        "label": f"Total growth: {b} minus {a}",
                        "readgrp_label": "",
                        **_summarize(total[b] - total[a]),
                    }
                )
    return pd.DataFrame(rows)
