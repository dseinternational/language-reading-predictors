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
    n_waves: int | None = None,
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
        if n_children is not None and n_waves is not None:
            detail += f"  ([dim]{n_children} children × {n_waves} waves[/dim])"
        elif n_children is not None and n_phases is not None:
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


def gb_ranking_markdown(
    ranking: pd.DataFrame,
    *,
    target_label: str | None = None,
    top_n: int = 6,
) -> str:
    """Narrate the cluster-first GB predictor ranking as Markdown (issue #208).

    Reads a ``predictor_ranking.csv`` frame (columns ``member``, ``cluster_id``,
    ``perm_imp_mean``, ``mean_abs_shap``, ``sign``, ``same_skill_of_outcome``, …)
    and states the findings a reader would otherwise reconstruct from the tables:
    the leading predictors and the direction of their SHAP association, any
    permutation-vs-SHAP disagreement on the lead, concurrent-restatement (leakage)
    flags, and collinear clusters. Returns Markdown for an ``#| output: asis``
    chunk. Kept CSV-derived so it never drifts from the fitted model on re-fit.
    """
    what = f" of {target_label}" if target_label else ""
    if ranking is None or getattr(ranking, "empty", True) or "member" not in ranking.columns:
        return f"_Findings summary{what} is produced at the reporting/test tiers._"

    df = ranking.copy()
    sign_word = {
        "+": "higher values predict a larger",
        "-": "higher values predict a smaller",
        "0": "no clear directional link to the",
    }

    has_perm = "perm_imp_mean" in df.columns
    perm = df.sort_values("perm_imp_mean", ascending=False) if has_perm else df
    nonzero = perm[perm["perm_imp_mean"] > 0] if has_perm else perm
    lead = (nonzero if not nonzero.empty else perm).head(top_n)

    lead_names = ", ".join(f"`{m}`" for m in lead["member"])
    lines: list[str] = [
        f"The model leans most on {lead_names}{what}. Ranked cluster-first by "
        "out-of-fold permutation importance, with the SHAP direction of each "
        "association:",
        "",
    ]
    for _, r in lead.iterrows():
        direction = sign_word.get(str(r.get("sign", "")).strip(), "an unclear link to the")
        shap = r.get("mean_abs_shap")
        shap_txt = f", mean|SHAP| {shap:.2f}" if pd.notna(shap) else ""
        flag = (
            " — **concurrent restatement of the outcome (treat as leakage)**"
            if bool(r.get("same_skill_of_outcome", False))
            else ""
        )
        lines.append(f"- `{r['member']}` — {direction} outcome{shap_txt}{flag}")

    if has_perm and "mean_abs_shap" in df.columns and not perm.empty:
        perm_top = perm.iloc[0]["member"]
        shap_top = df.sort_values("mean_abs_shap", ascending=False).iloc[0]["member"]
        if perm_top != shap_top:
            lines += [
                "",
                f"Permutation importance and SHAP disagree on the lead predictor "
                f"(permutation: `{perm_top}`; mean|SHAP|: `{shap_top}`) — read both, "
                "as they answer different questions (drop in held-out accuracy vs "
                "average contribution to predictions).",
            ]

    if "same_skill_of_outcome" in df.columns:
        leaks = df[df["same_skill_of_outcome"].astype(bool)]["member"].tolist()
        if leaks:
            lines += [
                "",
                "**Leakage caution:** "
                + ", ".join(f"`{m}`" for m in leaks)
                + " concurrently restate the outcome skill and should be discounted.",
            ]

    if "cluster_id" in df.columns:
        multi = [
            ms
            for ms in df.groupby("cluster_id", sort=False)["member"].apply(list)
            if len(ms) > 1
        ]
        if multi:
            grouped = "; ".join(
                "{" + ", ".join(f"`{m}`" for m in ms) + "}" for ms in multi
            )
            lines += [
                "",
                "Collinear predictors were clustered by distance-correlation and "
                f"ranked as a group (importance is shared within a cluster): {grouped}.",
            ]

    return "\n".join(lines)


__all__ = [
    "cv_fold_metrics_table",
    "gb_ranking_markdown",
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
