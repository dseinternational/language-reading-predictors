# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the shared figure-artifact helpers (issue #208)."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from language_reading_predictors import figure_io  # noqa: E402
from language_reading_predictors.figure_io import (  # noqa: E402
    save_plot_data,
    save_plotcollection,
    save_styled_figure,
)


def _tiny_fig():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig


def test_save_styled_figure_writes_png_and_svg_and_closes(tmp_path):
    fig = _tiny_fig()
    png = save_styled_figure(str(tmp_path), "demo", fig=fig)
    assert (tmp_path / "demo.png").exists()
    assert (tmp_path / "demo.svg").exists()  # #208: SVG sibling
    assert png.endswith("demo.png")
    # close=True by default -> no lingering figures
    assert plt.get_fignums() == []


def test_save_styled_figure_accepts_png_extension_in_name(tmp_path):
    save_styled_figure(str(tmp_path), "with_ext.png", fig=_tiny_fig())
    assert (tmp_path / "with_ext.png").exists()
    assert (tmp_path / "with_ext.svg").exists()
    assert not (tmp_path / "with_ext.png.png").exists()


def test_save_styled_figure_svg_size_guard(tmp_path, monkeypatch):
    # Force the SVG over the cap -> it is written then dropped, PNG kept.
    monkeypatch.setattr(figure_io, "SVG_MAX_BYTES", 1)
    save_styled_figure(str(tmp_path), "big", fig=_tiny_fig())
    assert (tmp_path / "big.png").exists()
    assert not (tmp_path / "big.svg").exists()


def test_save_styled_figure_svg_opt_out(tmp_path):
    save_styled_figure(str(tmp_path), "nosvg", fig=_tiny_fig(), svg=False)
    assert (tmp_path / "nosvg.png").exists()
    assert not (tmp_path / "nosvg.svg").exists()


def test_save_styled_figure_writes_data_csv(tmp_path):
    df = pd.DataFrame({"x": [0, 1], "y": [1, 2]})
    save_styled_figure(str(tmp_path), "withdata", fig=_tiny_fig(), data=df)
    out = pd.read_csv(tmp_path / "withdata.csv")
    assert list(out.columns) == ["x", "y"]
    assert len(out) == 2


def test_save_plot_data_stem_and_no_index(tmp_path):
    save_plot_data(str(tmp_path), "curve.png", {"a": [1, 2, 3]})
    assert (tmp_path / "curve.csv").exists()
    assert (tmp_path / "curve.csv").read_text().splitlines()[0] == "a"


class _Item:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakePlotCollection:
    """Minimal stand-in for an arviz_plots PlotCollection."""

    def __init__(self, fig):
        self._fig = fig
        self.viz = {"figure": _Item(fig)}

    def savefig(self, path, **kwargs):
        self._fig.savefig(path)


def test_save_plotcollection_png_svg_and_suptitle(tmp_path):
    fig = _tiny_fig()
    pc = _FakePlotCollection(fig)
    save_plotcollection(pc, str(tmp_path), "trace_plot.png", suptitle="My title")
    assert (tmp_path / "trace_plot.png").exists()
    assert (tmp_path / "trace_plot.svg").exists()
    assert fig._suptitle is not None and fig._suptitle.get_text() == "My title"


def test_init_plotting_applies_house_style():
    """Regression guard: the Bayesian fit path must apply the DSE style."""
    from language_reading_predictors.statistical_models.environment import (
        init_plotting,
    )

    plt.rcParams["savefig.dpi"] = 72  # perturb
    init_plotting()
    # set_matplotlib_default_style pins the file DPI at 300.
    assert plt.rcParams["savefig.dpi"] == pytest.approx(300)


def test_house_style_sets_noto_fonts():
    """Issue #693: figures use Noto Sans text and Noto Sans Math math."""
    from language_reading_predictors.statistical_models.environment import (
        init_plotting,
    )

    with matplotlib.rc_context():
        init_plotting()
        assert plt.rcParams["font.family"] == ["Noto Sans", "Noto Sans Math", "DejaVu Sans"]
        assert plt.rcParams["font.sans-serif"][0] == "Noto Sans"
        assert plt.rcParams["mathtext.fontset"] == "custom"
        assert plt.rcParams["mathtext.rm"] == "Noto Sans Math"
        assert plt.rcParams["savefig.dpi"] == pytest.approx(300)  # and the rest of the shared style


def test_house_style_draws_symbols_noto_sans_lacks():
    """Noto Sans has no arrows or relations; the family list falls back per glyph.

    Without the Noto fonts (as in CI) DejaVu Sans draws all of it, so this holds
    either way; with them, a generic "sans-serif" family would draw empty boxes.
    """
    with matplotlib.rc_context():
        figure_io.use_house_style()
        fig, ax = plt.subplots()
        ax.set_title("average effect ≈ +1.4 items → ≤ ≥ ↔ ≠ ✓")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.canvas.draw()
        plt.close(fig)
    assert not [w for w in caught if "missing from font" in str(w.message)]


def test_use_house_fonts_sets_fonts_and_leaves_layout():
    with matplotlib.rc_context(
        {
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 9,
            "figure.constrained_layout.use": False,
        }
    ):
        figure_io.use_house_fonts()
        assert plt.rcParams["font.family"] == ["Noto Sans", "Noto Sans Math", "DejaVu Sans"]
        assert plt.rcParams["font.sans-serif"][0] == "Noto Sans"
        assert plt.rcParams["mathtext.fontset"] == "custom"
        assert plt.rcParams["mathtext.rm"] == "Noto Sans Math"
        assert plt.rcParams["mathtext.it"] == "Noto Sans:italic"
        assert plt.rcParams["font.size"] == pytest.approx(9)
        assert plt.rcParams["figure.constrained_layout.use"] is False


def test_save_plotcollection_leaves_unrelated_figure_open(tmp_path):
    unrelated = _tiny_fig()
    owned = _tiny_fig()
    try:
        save_plotcollection(_FakePlotCollection(owned), str(tmp_path), "owned")
        assert not plt.fignum_exists(owned.number)
        assert plt.fignum_exists(unrelated.number)
    finally:
        plt.close(unrelated)
        plt.close(owned)
