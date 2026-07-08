# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the shared figure-artifact helpers (issue #208)."""

from __future__ import annotations

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
