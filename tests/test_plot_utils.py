# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`plot_utils` output-root resolution.

Regression guard for the historical bug where the plot output dir resolved to
``src/output`` instead of the repo-root ``output/`` that every other module
writes to. Since #180 the root is resolved via ``paths.output_root()`` **at call
time** (env / ``--output-dir`` aware) rather than a module constant, so this
checks the default root that ``save_figure`` / ``display_image`` pass through.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from language_reading_predictors import paths


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv(paths.OUTPUT_ROOT_ENV_VAR, raising=False)
    paths.set_output_root(None)
    yield
    paths.set_output_root(None)


def test_plot_helpers_default_to_repo_root_output(monkeypatch):
    import language_reading_predictors.plot_utils as pu

    seen: dict[str, str] = {}
    monkeypatch.setattr(
        pu, "_shared_save_figure", lambda fn, root, **kw: seen.update(root=root)
    )
    pu.save_figure("fig.png")

    root = Path(seen["root"])
    assert root.name == "output"
    # The repo root is identified by pyproject.toml living next to output/, not a
    # hard-coded absolute path (robust to the checkout location).
    assert (root.parent / "pyproject.toml").exists()
    # Regression guard: the output root must not be nested under src/.
    assert "src" not in root.parts


# --- SHAP scatter overplotting -------------------------------------------------
#
# Regression guard: whole-number scores and a coarse tree ensemble put many
# observations on exactly the same point, and SHAP's opaque, unjittered defaults
# drew the 157 rows of ``lrp-rli-gbg-012`` as 57 visible dots.


def _stacked_explanation():
    import numpy as np
    import shap

    # 40 rows on 4 whole-number scores and 2 SHAP levels: 8 distinct positions.
    scores = np.repeat([0.0, 1.0, 2.0, 3.0], 10)
    values = np.where(scores < 2, -0.5, 0.5) + np.tile([0.0, 0.1], 20)
    return shap.Explanation(
        values=values.reshape(-1, 1),
        base_values=np.zeros(len(scores)),
        data=scores.reshape(-1, 1),
        feature_names=["score"],
    )


def _offsets(**kwargs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import language_reading_predictors.plot_utils as pu

    explanation = _stacked_explanation()
    fig, ax = plt.subplots()
    try:
        pu.draw_shap_scatter(explanation[:, "score"], ax, **kwargs)
        points = max(ax.collections, key=lambda c: len(c.get_offsets()))
        return points.get_offsets().copy(), points.get_alpha(), explanation
    finally:
        plt.close(fig)


def test_shap_scatter_separates_stacked_observations():
    import numpy as np

    offsets, alpha, explanation = _offsets()
    scores = explanation.data[:, 0]

    assert len(offsets) == len(scores)
    assert alpha == pytest.approx(0.5)
    # Every observation is drawn at its own position ...
    assert len(np.unique(offsets.round(6), axis=0)) == len(scores)
    # ... within +/-0.3 of its score, so it cannot be read as a neighbouring value ...
    order = np.argsort(offsets[:, 0], kind="stable")
    assert np.all(np.abs(offsets[order, 0] - np.sort(scores)) <= 0.3 + 1e-9)
    # ... and the SHAP values themselves are never jittered.
    assert sorted(offsets[:, 1].round(9)) == sorted(explanation.values[:, 0].round(9))


def test_shap_scatter_defaults_can_be_switched_off():
    import numpy as np

    offsets, alpha, _ = _offsets(alpha=1.0, x_jitter=0.0)
    assert alpha == pytest.approx(1.0)
    assert len(np.unique(offsets.round(6), axis=0)) == 8


def test_shap_scatter_jitter_is_reproducible_and_leaves_the_global_generator_alone():
    import numpy as np

    np.random.seed(123)
    expected = np.random.random_sample(3)

    np.random.seed(123)
    first, _, _ = _offsets()
    second, _, _ = _offsets()
    after = np.random.random_sample(3)

    assert np.array_equal(first, second)
    assert np.array_equal(after, expected)


# --- Categorical scatter colours -----------------------------------------------
#
# The design language allows six categorical series, and the shared
# ``categorical_palette`` raises beyond that, so the default must fall back for
# a figure with more categories rather than fail.


def _scatter_colours(n_categories: int) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import to_hex

    import language_reading_predictors.plot_utils as pu

    n_rows = 2 * n_categories
    df = pd.DataFrame({"x": range(n_rows), "y": range(n_rows), "g": [i % n_categories for i in range(n_rows)]})
    try:
        pu.scatter_plot(df, "x", "y", color="g", categorical=True)
        return [to_hex(c.get_facecolor()[0]) for c in plt.gca().collections]
    finally:
        plt.close("all")


def test_categorical_scatter_takes_the_chart_colours():
    from dse_research_utils.plot.styles import CHART_COLOURS

    assert _scatter_colours(2) == list(CHART_COLOURS[:2])


def test_categorical_scatter_draws_more_than_six_categories():
    colours = _scatter_colours(8)
    assert len(colours) == 8
    assert len(set(colours)) == 8
