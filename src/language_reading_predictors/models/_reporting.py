# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rich-based formatting helpers for model pipeline console output.

Centralises the patterns used across ``base_pipeline``, ``fit_model.py`` and
``tune_model.py`` so every script reports results consistently:

* ``metrics_table`` — one-row-per-metric summary with optional std column.
* ``ranked_dataframe_table`` — ranked per-feature tables (permutation
  importance, construct importance, SHAP direction, stability selection).
* ``model_header_panel`` — the banner printed at the top of a fit.
* ``run_summary_panel`` — "Done. Artifacts saved to ..." banner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_console = Console()


def print_table(table: Table) -> None:
    """Print a rich Table via a shared Console."""
    _console.print(table)


def print_panel(panel: Panel) -> None:
    """Print a rich Panel via a shared Console."""
    _console.print(panel)


def section_header(title: str) -> None:
    """Print a green dashed-banner section header.

    Matches the style used across both estimator (``base_pipeline``) and
    statistical (``statistical_models.pipeline``) pipelines so every
    script shows the same section dividers.
    """
    _console.print()
    _console.print("[green]" + "-" * 60 + "[/green]")
    _console.print(f"[bold green]{title}[/bold green]")
    _console.print("[green]" + "-" * 60 + "[/green]")


def metrics_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    columns: Sequence[str] | None = None,
    precision: int = 4,
) -> Table:
    """Build a key/value-style metrics table.

    Parameters
    ----------
    rows
        Ordered sequence of mappings. Each row is rendered in column order;
        the first column is typically the metric name (left-aligned), the
        rest are numeric (right-aligned).
    title
        Optional table title, shown above the header row.
    columns
        Ordered column names. Inferred from the first row when omitted.
    precision
        Decimal places for float values. Integers render as-is; strings pass
        through untouched.
    """
    if not rows:
        return Table(title=title)
    if columns is None:
        columns = list(rows[0].keys())

    table = Table(title=title, title_style="bold", show_lines=False)
    for i, col in enumerate(columns):
        justify = "left" if i == 0 else "right"
        table.add_column(col, justify=justify, no_wrap=(i == 0))

    for row in rows:
        rendered = [_fmt(row.get(col), precision) for col in columns]
        table.add_row(*rendered)
    return table


def ranked_dataframe_table(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    columns: Sequence[str] | None = None,
    precision: int = 4,
    max_rows: int | None = None,
    rank_column: bool = True,
) -> Table:
    """Render a ranked DataFrame (permutation importance, SHAP, ...) as a table.

    Parameters
    ----------
    df
        DataFrame already sorted in the order it should appear.
    title
        Optional table title.
    columns
        Columns to render. Defaults to every column in ``df``.
    precision
        Decimal places for numeric columns.
    max_rows
        If set, truncate to the first ``max_rows`` rows and append an
        ellipsis row indicating how many were hidden.
    rank_column
        When True, prepend a ``#`` column with the 1-based row index.
    """
    cols = list(columns) if columns is not None else list(df.columns)
    table = Table(title=title, title_style="bold", show_lines=False)

    if rank_column:
        table.add_column("#", justify="right", style="dim", no_wrap=True)
    for i, col in enumerate(cols):
        justify = "left" if i == 0 else "right"
        table.add_column(col, justify=justify, no_wrap=(i == 0))

    display_df = df if max_rows is None else df.head(max_rows)
    for idx, (_, row) in enumerate(display_df.iterrows(), start=1):
        rendered = [_fmt(row[col], precision) for col in cols]
        if rank_column:
            rendered = [str(idx), *rendered]
        table.add_row(*rendered)

    if max_rows is not None and len(df) > max_rows:
        filler = ["..."] * (len(cols) + (1 if rank_column else 0))
        table.add_row(*filler)
        hidden = len(df) - max_rows
        table.caption = f"(+{hidden} more row{'s' if hidden != 1 else ''})"

    return table


def cv_fold_metrics_table(scores_df: pd.DataFrame, *, precision: int = 4) -> Table:
    """Aggregate CV-fold scores into a single metrics table.

    ``scores_df`` is the per-fold DataFrame saved as ``cv_scores.csv``:
    one row per fold with columns ``mae``, ``rmse``, ``r2``, ``medae``.
    """
    rows = []
    for col in ("mae", "rmse", "r2", "medae"):
        if col not in scores_df.columns:
            continue
        rows.append(
            {
                "metric": col.upper() if col != "r2" else "R\u00b2",
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
    rows = []
    for key, label in (
        ("pooled_mae", "MAE"),
        ("pooled_rmse", "RMSE"),
        ("pooled_r2", "R\u00b2"),
        ("pooled_medae", "MedAE"),
    ):
        value = pooled.get(key)
        if value is None:
            continue
        rows.append({"metric": label, "value": float(value)})
    return metrics_table(
        rows,
        title="Pooled out-of-fold (vs. global mean)",
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
        {"metric": "R\u00b2", "value": r2},
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
    return Panel(
        "\n".join(lines),
        border_style="green",
        padding=(1, 2),
    )


def run_summary_panel(
    *, output_dir: Path | str, status: str = "Done"
) -> Panel:
    """Build the banner panel printed at the end of each fit."""
    return Panel(
        f"[bold green]{status}[/bold green]\n\n"
        f"[dim]Artifacts:[/dim] {output_dir}",
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
    """Build the banner panel printed at the start of a statistical-model fit.

    Covers ITT, joint and mechanism model shapes — only the fields that are
    set render into the panel body.
    """
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
            detail += f"  ([dim]{n_children} children \u00d7 {n_phases} phases[/dim])"
        lines.append(f"[dim]Observations:[/dim] {detail}")
    return Panel(
        "\n".join(lines),
        border_style="green",
        padding=(1, 2),
    )


def params_table(
    params: Mapping[str, Any],
    *,
    title: str | None = None,
    precision: int = 6,
) -> Table:
    """Render a parameter dict as a two-column table (``param`` / ``value``)."""
    rows = [{"param": k, "value": params[k]} for k in params]
    return metrics_table(
        rows,
        title=title,
        columns=["param", "value"],
        precision=precision,
    )


def _fmt(value: Any, precision: int) -> str:
    """Format a single cell value for a rich Table."""
    if value is None:
        return "\u2014"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN check
            return "\u2014"
        # Use fixed precision for large magnitudes, scientific for tiny values.
        abs_v = abs(value)
        if abs_v != 0 and (abs_v < 1e-3 or abs_v >= 1e6):
            return f"{value:.{precision}e}"
        return f"{value:.{precision}f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


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
