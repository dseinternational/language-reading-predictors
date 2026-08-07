# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Console banners, report-template publication and the model-graph render.

The presentation surface every family fit shares: the start-of-fit and
end-of-fit panels, the LOO summary row, the copy of ``index.qmd`` plus the
shared Quarto partials into the fit's output directory, and the Graphviz render
of the built model. Split out of ``pipeline.py`` for #394 so orchestration
modules depend on presentation rather than containing it.
"""

from __future__ import annotations

import os
import shutil

from rich import print as rprint

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_panel,
    print_table,
    run_summary_panel,
    stat_model_header_panel,
)
from language_reading_predictors.statistical_models.artifacts import guard_optional
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.environment import DOCS_DIR


def print_header(ctx: StatisticalFitContext) -> None:
    """Print the start-of-fit banner panel."""
    spec = ctx.spec
    prepared = ctx.prepared
    rprint()
    print_panel(
        stat_model_header_panel(
            model_id=spec.model_id,
            title=spec.title,
            kind=spec.kind,
            config_name=ctx.reporting.config_name,
            outcome_symbol=spec.outcome_symbol,
            mechanism_symbol=spec.mechanism_symbol,
            adjustment=spec.adjustment or None,
            n_obs=prepared.n_obs if prepared else None,
            n_children=prepared.n_children if prepared else None,
            n_phases=prepared.n_phases if prepared else None,
            n_waves=getattr(prepared, "n_waves", None) if prepared else None,
        )
    )


def print_footer(ctx: StatisticalFitContext) -> None:
    """Print the end-of-fit banner panel."""
    rprint()
    print_panel(run_summary_panel(output_dir=ctx.output_dir))


def print_loo_row(ctx: StatisticalFitContext) -> None:
    """Render the LOO ELPD / p / se summary as a small table.

    arviz 1.x ``ELPDData`` exposes ``elpd`` / ``se`` / ``p`` (the 0.x
    ``elpd_loo`` / ``p_loo`` / ``looic`` attributes were removed).
    """
    if ctx.loo is None:
        return
    rows = [
        {"metric": "elpd_loo", "value": float(ctx.loo.elpd)},
        {"metric": "se", "value": float(ctx.loo.se)},
        {"metric": "p_loo", "value": float(ctx.loo.p)},
    ]
    print_table(
        metrics_table(
            rows,
            title="LOO-PSIS",
            columns=["metric", "value"],
        )
    )


def copy_report_template(context: StatisticalFitContext) -> None:
    src = os.path.join(DOCS_DIR, "models", context.spec.model_id, "index.qmd")
    dst = os.path.join(context.output_dir, "index.qmd")
    if os.path.exists(src):
        shutil.copy(src, dst)
        if os.environ.get("LRP_OFFLINE_QUARTO") == "1":
            _strip_quarto_code_links(dst)
        rprint(f"  Report template copied to {dst}")
    else:
        rprint(f"  [yellow]No report template found at {src}[/yellow]")

    # Copy the shared Quarto partials alongside the report so ``{{< include
    # _partials/... >}}`` resolves at render time in the output dir (issue #125
    # step 0a). Quarto resolves includes relative to the rendered file.
    partials_src = os.path.join(DOCS_DIR, "models", "_partials")
    partials_dst = os.path.join(context.output_dir, "_partials")
    if os.path.isdir(partials_src):
        shutil.copytree(partials_src, partials_dst, dirs_exist_ok=True)


def _strip_quarto_code_links(path: str) -> None:
    """Remove copied ``code-links: repo`` metadata for offline Quarto renders.

    Quarto resolves ``code-links: repo`` by probing the GitHub remote, including
    ``git ls-remote origin gh-pages``. In restricted reporting environments that
    optional link can make an otherwise valid report render fail after all cells
    execute. The source templates are left intact; only the copied output QMD is
    made renderable when ``LRP_OFFLINE_QUARTO=1`` is set.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    text = text.replace("    code-links:\n      - repo\n", "")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def render_model_graph(context: StatisticalFitContext) -> None:
    with guard_optional(
        context, "Graphviz render",
        filename="model_graph.png", kind="figure", verb="failed",
    ):
        g = _graphviz(context.model)
        g.render(
            filename=os.path.join(context.output_dir, "model_graph"),
            format="png",
            cleanup=True,
        )


def _graphviz(model):
    import pymc as pm

    g = pm.model_to_graphviz(model)
    g.graph_attr["fontname"] = "Helvetica"
    # Raster PNG output (not SVG): the DAG's many nodes/edges make a large SVG
    # slow to browse, so render to PNG and bump DPI to keep the lightbox legible.
    g.graph_attr["dpi"] = "150"
    g.node_attr["fontname"] = "Helvetica"
    g.edge_attr["fontname"] = "Helvetica"
    return g
