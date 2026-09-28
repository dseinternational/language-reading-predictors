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
(``set_matplotlib_default_style``), applied once at the fit entry points through
:func:`use_house_style`; the other helpers only standardise *saving* — PNG + SVG
sibling + optional data CSV — and closing the figure. Both matplotlib figures and
``arviz_plots`` ``PlotCollection`` objects route through here so a single change
propagates to every model. Standalone scripts that lay out their own figures take
only the house fonts, through :func:`use_house_fonts`.

The save mechanics moved into ``dse_research_utils.plot.io`` in v0.12.0 (they were
one of five parallel implementations across the research repositories); this module
now applies this repository's policy to them — the SVG size cap is on by default
here, whereas the shared helpers leave it opt-in. :data:`SVG_MAX_BYTES` is read at
call time so it stays configurable (and monkeypatchable in tests).
"""

from __future__ import annotations

from typing import Any

import dse_research_utils.plot.io as plot_io
import matplotlib as mpl
from dse_research_utils.plot.styles import (
    DEFAULT_STYLE_DICT,
    DPI_FILE,
    default_font_families,
    set_matplotlib_default_style,
)

# Issue #208: still emit SVGs, but skip very large ones (they are what make the
# report viewer slow). ~2 MB is comfortably above a typical vector figure and well
# below the multi-megabyte beeswarm/interaction grids we want to keep raster.
SVG_MAX_BYTES = plot_io.SVG_MAX_BYTES

# The house font settings: the shared style's sans-serif list and mathtext fonts
# (Noto Sans Math for equations). Read from the shared style so the two cannot
# drift. Font size stays out because it changes layout, which is what the
# fonts-only route exists to leave alone. ``font.family`` stays out because it
# depends on which fonts are installed (#695), so :func:`use_house_fonts` reads it
# when it runs.
HOUSE_FONT_RCPARAMS: dict[str, Any] = {
    key: value for key, value in DEFAULT_STYLE_DICT.items() if key == "font.sans-serif" or key.startswith("mathtext.")
}


def use_house_fonts() -> None:
    """Set the house fonts without the rest of the house style.

    For standalone scripts that size and lay out their own figures (with
    ``tight_layout``, which the full style's constrained layout would fight).
    Model fits and scripts that redraw fit figures use :func:`use_house_style`.

    ``font.family`` comes from the shared style's fallback list, so symbols that
    Noto Sans lacks (→ ≈ ≤ ✓) fall back to Noto Sans Math or DejaVu Sans rather
    than drawing as empty boxes (#693). The list names only installed fonts, so a
    machine without the Noto fonts does not log a lookup failure per text element.
    """
    mpl.rcParams.update(HOUSE_FONT_RCPARAMS)
    mpl.rcParams["font.family"] = default_font_families()


def use_house_style() -> None:
    """Apply the shared house style, including its font fallback list."""
    set_matplotlib_default_style()


def save_plot_data(output_dir: str, name: str, data: Any, *, index: bool = False) -> str:
    """Write the data behind a plot as ``<name>.csv`` (issue #208)."""
    return plot_io.save_plot_data(output_dir, name, data, index=index)


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
    return plot_io.save_styled_figure(
        output_dir,
        name,
        fig=fig,
        dpi=dpi,
        bbox_inches=bbox_inches,
        close=close,
        svg=svg,
        svg_max_bytes=SVG_MAX_BYTES,
        data=data,
    )


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
    plot_io.save_plotcollection(
        pc,
        output_dir,
        name,
        suptitle=suptitle,
        dpi=dpi,
        svg=svg,
        svg_max_bytes=SVG_MAX_BYTES,
        data=data,
    )


__all__ = [
    "DPI_FILE",
    "HOUSE_FONT_RCPARAMS",
    "SVG_MAX_BYTES",
    "save_plot_data",
    "save_plotcollection",
    "save_styled_figure",
    "use_house_fonts",
    "use_house_style",
]
