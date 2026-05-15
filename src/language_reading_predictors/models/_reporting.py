# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rich-based formatting helpers for model pipeline console output.

This module preserves the public surface that ``base_pipeline``,
``statistical_models.pipeline``, and the ``scripts/*`` entry points have
historically imported from here, but the primitives (banner, section
header, key/value and dataframe tables, value formatter) now live in
:mod:`dse_research_utils.console`. The functions below are thin shims that
delegate to the shared implementation; ML-specific composers
(``cv_fold_metrics_table``, ``pooled_oof_table``, ``in_sample_metrics_table``,
``model_header_panel``, ``stat_model_header_panel``) stay here because they
encode domain-specific content even though they build on shared tables
underneath.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from rich.panel import Panel
from rich.table import Table

from dse_research_utils.console.console import print_panel, print_table
from dse_research_utils.console.sections import section_header as _section_header
from dse_research_utils.console.tables import (
    dataframe_table as _dataframe_table,
)
from dse_research_utils.console.tables import (
    metrics_table as _metrics_table,
)
from dse_research_utils.console.tables import (
    params_table as _params_table,
)


def section_header(title: str) -> None:
    """Print a green rule-based section header.

    Historically this rendered as a dashed banner on three lines; the shared
    implementation uses a single rich ``Rule`` with the same title and
    styling, which is visually tighter but carries the same information.
    """
    _section_header(title)


def metrics_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    columns: Sequence[str] | None = None,
    precision: int = 4,
) -> Table:
    """Build a key/value-style metrics table.

    Thin wrapper over
    :func:`dse_research_utils.console.tables.metrics_table` with identical
    semantics.
    """
    return _metrics_table(rows, title=title, columns=columns, precision=precision)


def ranked_dataframe_table(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    columns: Sequence[str] | None = None,
    precision: int = 4,
    max_rows: int | None = None,
    rank_column: bool = True,
) -> Table:
    """Render a ranked DataFrame as a table.

    Wraps :func:`dse_research_utils.console.tables.dataframe_table` with the
    ``truncation="head"`` mode (LRP historically truncated to the first
    ``max_rows`` rather than splitting head/tail).
    """
    return _dataframe_table(
        df,
        title=title,
        columns=columns,
        max_rows=max_rows,
        truncation="head",
        rank_column="#" if rank_column else None,
        show_index=False,
        precision=precision,
    )


def cv_fold_metrics_table(scores_df: pd.DataFrame, *, precision: int = 4) -> Table:
    """Aggregate CV-fold scores into a single metrics table."""
    rows: list[Mapping[str, Any]] = []
    for col in ("mae", "rmse", "r2", "medae"):
        if col not in scores_df.columns:
            continue
        rows.append(
            {
                "metric": col.upper() if col != "r2" else "R²",
                "mean": float(scores_df[col].mean()),
                "std": float(scores_df[col].std()),
                "min": float(scores_df[col].min()),
                "max": float(scores_df[col].max()),
            }
        )
    return metrics_table(
        rows,
        title="Cross-validation (per fold)",
        columns=["metric", "mean", "std", "min", "max"],
        precision=precision,
    )


def pooled_oof_table(pooled: Mapping[str, float], *, precision: int = 4) -> Table:
    """Render pooled out-of-fold metrics as a two-column table."""
    rows: list[Mapping[str, Any]] = []
    for key, label in (
        ("pooled_mae", "MAE"),
        ("pooled_rmse", "RMSE"),
        ("pooled_r2", "R²"),
        ("pooled_medae", "MedAE"),
    ):
        value = pooled.get(key)
        if value is None:
            continue
        rows.append({"metric": label, "value": float(value)})
    return metrics_table(
        rows,
        title="Pooled out-of-fold (R² vs. per-fold train mean)",
        columns=["metric", "value"],
        precision=precision,
    )


def in_sample_metrics_table(
    mae: float, rmse: float, r2: float, medae: float, *, precision: int = 4
) -> Table:
    """Render in-sample metrics as a two-column table."""
    rows = [
        {"metric": "MAE", "value": mae},
        {"metric": "RMSE", "value": rmse},
        {"metric": "R²", "value": r2},
        {"metric": "MedAE", "value": medae},
    ]
    return metrics_table(
        rows,
        title="In-sample",
        columns=["metric", "value"],
        precision=precision,
    )


def model_header_panel(
    *,
    model_id: str,
    description: str,
    pipeline_cls: str,
    run_config: str,
    target: str,
    n_predictors: int,
    variant_of: str | None = None,
) -> Panel:
    """Build the banner panel printed at the start of each fit."""
    lines = [
        f"[bold]{model_id.upper()}[/bold]: {description}",
        "",
        f"[dim]Pipeline:[/dim]   {pipeline_cls}",
        f"[dim]Run config:[/dim] {run_config}",
        f"[dim]Target:[/dim]     {target}",
        f"[dim]Predictors:[/dim] {n_predictors}",
    ]
    if variant_of:
        lines.append(f"[dim]Variant of:[/dim] {variant_of}")
    return Panel("\n".join(lines), border_style="green", padding=(1, 2))


def run_summary_panel(*, output_dir: Path | str, status: str = "Done") -> Panel:
    """Build the banner panel printed at the end of each fit."""
    return Panel(
        f"[bold green]{status}[/bold green]\n\n[dim]Artifacts:[/dim] {output_dir}",
        border_style="green",
        padding=(1, 2),
    )


def stat_model_header_panel(
    *,
    model_id: str,
    title: str,
    kind: str,
    config_name: str,
    outcome_symbol: str | None = None,
    mechanism_symbol: str | None = None,
    adjustment: Sequence[str] | None = None,
    n_obs: int | None = None,
    n_children: int | None = None,
    n_phases: int | None = None,
) -> Panel:
    """Build the banner panel printed at the start of a statistical-model fit."""
    lines = [
        f"[bold]{model_id.upper()}[/bold]: {title}",
        "",
        f"[dim]Kind:[/dim]       {kind}",
        f"[dim]Run config:[/dim] {config_name}",
    ]
    if outcome_symbol:
        lines.append(f"[dim]Outcome:[/dim]    {outcome_symbol}")
    if mechanism_symbol:
        lines.append(f"[dim]Mechanism:[/dim]  {mechanism_symbol}")
    if adjustment:
        lines.append(f"[dim]Adjustment:[/dim] {', '.join(adjustment)}")
    if n_obs is not None:
        detail = f"{n_obs:,}"
        if n_children is not None and n_phases is not None:
            detail += f"  ([dim]{n_children} children × {n_phases} phases[/dim])"
        lines.append(f"[dim]Observations:[/dim] {detail}")
    return Panel("\n".join(lines), border_style="green", padding=(1, 2))


def params_table(
    params: Mapping[str, Any],
    *,
    title: str | None = None,
    precision: int = 6,
) -> Table:
    """Render a parameter dict as a two-column (``param``, ``value``) table."""
    return _params_table(params, title=title, precision=precision)


__all__ = [
    "cv_fold_metrics_table",
    "in_sample_metrics_table",
    "metrics_table",
    "model_header_panel",
    "params_table",
    "pooled_oof_table",
    "print_panel",
    "print_table",
    "ranked_dataframe_table",
    "run_summary_panel",
    "section_header",
    "stat_model_header_panel",
]
