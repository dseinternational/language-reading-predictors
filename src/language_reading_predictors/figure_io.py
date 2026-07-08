# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared figure-saving helpers for both model systems (GB and Bayesian).

These centralise the report figure-artifact policy (issue #208) so every plotting
call site behaves identically:

* always write ``<name>.png`` — the artifact the report templates reference
  (raster keeps the model-output pages quick to browse);
* also write ``<name>.svg`` unless it would exceed ~2 MB, in which case the vector
  sibling is dropped (very large SVGs are the ones that make the viewer slow —
  exactly what #208 wants to avoid);
* optionally write ``<name>.csv`` of the data behind the plot.

Consistent house style (fonts, colours, grid, DPI) comes from ``dse_research_utils``
(``set_matplotlib_default_style``), applied once at the fit entry points; these
helpers only standardise *saving* — PNG + SVG sibling + optional data CSV — and
closing the figure. Both matplotlib figures and ``arviz_plots`` ``PlotCollection``
objects route through here so a single change propagates to every model.

Kept dependency-light (matplotlib + pandas + the shared style constants only) so
neither the GB nor the Bayesian package pulls in the other's heavy imports.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from rich import print as rprint

from dse_research_utils.plot.styles import DPI_FILE

# Issue #208: still emit SVGs, but skip very large ones (they are what make the
# report viewer slow). ~2 MB is comfortably above a typical vector figure and well
# below the multi-megabyte beeswarm/interaction grids we want to keep raster.
SVG_MAX_BYTES = 2 * 1024 * 1024


def _stem(name: str) -> str:
    """Return ``name`` without a trailing ``.png``/``.svg`` so callers may pass
    either the historical ``"trace_plot.png"`` or a bare ``"trace_plot"`` stem."""
    for ext in (".png", ".svg"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _write_svg_sibling(save, base: str, svg_max_bytes: int = SVG_MAX_BYTES) -> None:
    """Write ``base + '.svg'`` via ``save(path)`` then drop it if it is too large.

    ``save`` is a one-arg callable (``fig.savefig`` or ``pc.savefig``) so this works
    for both matplotlib figures and ``arviz_plots`` collections. Guarded so an
    SVG-backend hiccup never costs us the (already-written) PNG.
    """
    svg = base + ".svg"
    try:
        save(svg)
        if os.path.getsize(svg) > svg_max_bytes:
            os.remove(svg)
    except Exception as exc:  # pragma: no cover - defensive
        rprint(f"[yellow]SVG sibling for {os.path.basename(base)} skipped: {exc}[/yellow]")
        if os.path.exists(svg):
            try:
                os.remove(svg)
            except OSError:
                pass


def save_plot_data(output_dir: str, name: str, data: Any, *, index: bool = False) -> str:
    """Write the data behind a plot as ``<name>.csv`` (issue #208)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{_stem(name)}.csv")
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    df.to_csv(path, index=index)
    return path


def save_styled_figure(
    output_dir: str,
    name: str,
    *,
    fig: Any | None = None,
    dpi: float = DPI_FILE,
    bbox_inches: str = "tight",
    close: bool = True,
    svg: bool = True,
    data: Any | None = None,
) -> str:
    """Save a matplotlib figure as PNG (+ SVG sibling, + optional data CSV).

    ``name`` may be a stem or carry a ``.png`` extension. Returns the PNG path.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig = fig or plt.gcf()
    base = os.path.join(output_dir, _stem(name))
    png = base + ".png"
    fig.savefig(png, dpi=dpi, bbox_inches=bbox_inches)
    if svg:
        _write_svg_sibling(
            lambda p: fig.savefig(p, format="svg", bbox_inches=bbox_inches), base
        )
    if data is not None:
        save_plot_data(output_dir, name, data)
    if close:
        plt.close(fig)
    return png


def _pc_figure(pc: Any):
    """Best-effort matplotlib ``Figure`` behind an ``arviz_plots`` collection."""
    try:
        return pc.viz["figure"].item()
    except Exception:  # pragma: no cover - defensive
        try:
            return plt.gcf()
        except Exception:
            return None


def save_plotcollection(
    pc: Any,
    output_dir: str,
    name: str,
    *,
    suptitle: str | None = None,
    dpi: float = DPI_FILE,
    svg: bool = True,
    data: Any | None = None,
) -> None:
    """Save an ``arviz_plots`` ``PlotCollection`` as PNG (+ SVG sibling).

    Adds a figure-level ``suptitle`` (ArviZ plots render untitled) and emits the
    SVG through ``pc.savefig`` so the collection lays out correctly.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, _stem(name))
    if suptitle:
        fig = _pc_figure(pc)
        if fig is not None:
            try:
                fig.suptitle(suptitle)
            except Exception:  # pragma: no cover - defensive
                pass
    pc.savefig(base + ".png", dpi=dpi)
    if svg:
        _write_svg_sibling(lambda p: pc.savefig(p), base)
    if data is not None:
        save_plot_data(output_dir, name, data)
    plt.close("all")


__all__ = [
    "DPI_FILE",
    "SVG_MAX_BYTES",
    "save_plot_data",
    "save_plotcollection",
    "save_styled_figure",
]
