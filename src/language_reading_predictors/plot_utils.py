# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import matplotlib.axes as mpaxes
import matplotlib.pyplot as plt
import seaborn as sns
import dse_research_utils.plot.styles as plot_styles

from language_reading_predictors import paths as _paths

# Generic plotting helpers now live in dse_research_utils.plot; re-exported here
# for backwards compatibility (the GP / graph helpers and histogram grid are
# behind the shared package's optional extras).
from dse_research_utils.plot.gaussian_process import (  # noqa: F401  (re-export)
    gaussian_process_prior,
    plot_2d_function,
    plot_errorbar,
    plot_gaussian_process,
    plot_gaussian_process_prior,
    plot_kernel_function,
    plot_line,
    plot_scatter,
    plot_x_errorbar,
)
from dse_research_utils.plot.graphs import (  # noqa: F401  (re-export)
    draw_causal_graph,
    plot_graph,
)
from dse_research_utils.plot.grids import plot_histograms  # noqa: F401  (re-export)
from dse_research_utils.plot.heatmap import plot_heatmap  # noqa: F401  (re-export)
from dse_research_utils.plot.io import display_image as _shared_display_image
from dse_research_utils.plot.io import save_figure as _shared_save_figure

from typing import Literal
from pathlib import Path
from scipy.cluster import hierarchy
from language_reading_predictors.data_variables import Variables as vars


def violin_plot(df: pd.DataFrame, x, y):
    plt.figure(figsize=plot_styles.FIGSIZE_LG)
    sns.violinplot(x=df[x], y=df[y])
    plt.xlabel(vars.get_variable_name(x))
    plt.ylabel(vars.get_variable_name(y))


def scatter_plot(df: pd.DataFrame, x, y, color=None, palette=None, categorical=None):
    """Scatter plot with automatic continuous/categorical colour handling.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : column names for the axes
    color : optional column name to colour points by
    palette : optional palette specification.
        - For continuous colour: a matplotlib colormap name (default 'viridis').
        - For categorical colour: a qualitative colormap name (default 'tab10'),
          a list/tuple of colours, or a dict mapping category values to colours
          (e.g. {1: '#1f77b4', 2: '#d62728'}).
    categorical : optional bool to override auto-detection. Use True to force
        discrete colours on a numeric column (e.g. 1/2 coded sex), or False
        to force a gradient on a categorical-dtype column.
    """
    plt.figure(figsize=plot_styles.FIGSIZE_LG)

    if color is None:
        plt.scatter(df[x], df[y], alpha=0.5)

    else:
        # Decide continuous vs categorical
        if categorical is None:
            is_continuous = (
                pd.api.types.is_numeric_dtype(df[color])
                and not pd.api.types.is_bool_dtype(df[color])
                and not isinstance(df[color].dtype, pd.CategoricalDtype)
            )
        else:
            is_continuous = not categorical

        if is_continuous:
            # Continuous → gradient colormap
            cmap_name = palette if isinstance(palette, str) else "viridis"
            sc = plt.scatter(df[x], df[y], c=df[color], alpha=0.5, cmap=cmap_name)
            plt.colorbar(sc, label=vars.get_variable_name(color))

        else:
            # Categorical → fixed discrete colours
            if isinstance(df[color].dtype, pd.CategoricalDtype):
                categories = list(df[color].cat.categories)
            else:
                categories = sorted(df[color].dropna().unique())

            if isinstance(palette, dict):
                color_map = palette
            elif isinstance(palette, (list, tuple)):
                color_map = {
                    cat: palette[i % len(palette)] for i, cat in enumerate(categories)
                }
            else:
                cmap = plt.get_cmap(palette or "tab10")
                n = len(categories)
                if cmap.N >= 256:  # continuous colormap used as qualitative
                    color_map = {
                        cat: cmap(i / max(n - 1, 1)) for i, cat in enumerate(categories)
                    }
                else:  # qualitative ListedColormap
                    color_map = {
                        cat: cmap(i % cmap.N) for i, cat in enumerate(categories)
                    }

            for cat in categories:
                mask = df[color] == cat
                plt.scatter(
                    df.loc[mask, x],
                    df.loc[mask, y],
                    c=[color_map[cat]],
                    alpha=0.5,
                    label=str(cat),
                )
            plt.legend(title=vars.get_variable_name(color))

    plt.xlabel(vars.get_variable_name(x))
    plt.ylabel(vars.get_variable_name(y))


def hierarchy_dendrogram(
    Z,
    axis: mpaxes.Axes,
    labels=None,
    orientation: Literal["top", "right", "bottom", "left"] = "right",
):
    dendro = hierarchy.dendrogram(Z, labels=labels, orientation=orientation, ax=axis)
    return dendro


def _plot_violinplot(data: pd.DataFrame, variable: str, time_points: list[int]):
    n = len(time_points)
    datasets = [data[data[vars.TIME] == t][variable].dropna() for t in time_points]

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4), sharey=True)

    for ax, d, t in zip(axes, datasets, time_points):
        parts = ax.violinplot(
            d,
            showmeans=True,
            showmedians=True,
        )
        parts["cmeans"].set_color("red")
        parts["cmedians"].set_color("green")
        parts["cmedians"].set_linewidth(2)
        ax.set_title(f"Time {t}")
        ax.set_xticks([])

    axes[0].set_ylabel(f"{variable} score")
    plt.suptitle(f"Violin plots of {variable} by time point")
    return fig, axes


def plot_violinplot_t1_to_t3(data: pd.DataFrame, variable: str):
    return _plot_violinplot(data, variable, [1, 2, 3])


def plot_violinplot_t1_to_t4(data: pd.DataFrame, variable: str):
    return _plot_violinplot(data, variable, [1, 2, 3, 4])


# plot_heatmap now lives in dse_research_utils.plot.heatmap (imported above) and is
# re-exported here for backwards compatibility.


def save_shap_scatter_plots(
    explanation,
    predictors: list[str],
    output_dir: Path,
    *,
    color=None,
    color_name: str | None = None,
    filename_prefix: str = "shap_scatter",
    filename_suffix: str | None = None,
    dpi: int = 300,
) -> list[Path]:
    """Save one ``shap.plots.scatter`` per predictor as PNG and SVG.

    The saved filenames follow the pattern::

        {filename_prefix}_{feature}[_{filename_suffix}].{png,svg}

    so repeated calls with different ``color`` values (and different
    suffixes) can coexist in the same directory without clobbering.

    Parameters
    ----------
    explanation : shap.Explanation
        Full SHAP explanation (n_samples × n_features).
    predictors : list[str]
        Feature names to plot — one scatter per name. Must match feature
        names embedded in ``explanation``.
    output_dir : Path
        Directory where PNG and SVG files are written.
    color : shap.Explanation | array-like | None, optional
        How to colour each point.

        * ``None`` (default): SHAP's auto-colouring picks the strongest
          interaction feature.
        * ``shap.Explanation`` (e.g. ``explanation[:, "other_feature"]``):
          colour by another feature already in the explanation. This is the
          classical "SHAP dependence plot" form and surfaces interaction
          effects between two predictors.
        * array-like: colour by an external variable (one value per sample).
          Use ``color_name`` to label the colour bar.
    color_name : str, optional
        Colour-bar label. Required when ``color`` is an array-like;
        ignored when ``color`` is an Explanation (shap reads the name from
        the Explanation itself).
    filename_prefix : str
        Leading component of the output filename.
    filename_suffix : str, optional
        Trailing component (e.g. ``"by_ewrswr"``) appended before the file
        extension. Useful when generating multiple sets with different
        ``color`` values.
    dpi : int
        Resolution of the PNG (SVG is vector and ignores this).

    Returns
    -------
    list of Path
        PNG paths written, in the same order as ``predictors``.
    """
    import shap

    resolved_color = None
    if color is not None:
        if isinstance(color, shap.Explanation):
            resolved_color = color
        else:
            if color_name is None:
                msg = "color_name must be provided when color is an array"
                raise ValueError(msg)
            arr = np.asarray(color, dtype=float)
            resolved_color = shap.Explanation(
                values=arr,
                base_values=None,
                data=arr,
                feature_names=color_name,
            )

    output_dir = Path(output_dir)
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    written: list[Path] = []
    for feature in predictors:
        fig = plt.figure()
        ax = fig.gca()
        if resolved_color is not None:
            shap.plots.scatter(
                explanation[:, feature],
                color=resolved_color,
                ax=ax,
                show=False,
            )
        else:
            shap.plots.scatter(explanation[:, feature], ax=ax, show=False)
        base = output_dir / f"{filename_prefix}_{feature}{suffix}"
        png_path = Path(f"{base}.png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(f"{base}.svg", bbox_inches="tight")
        plt.close(fig)
        written.append(png_path)
    return written


def save_figure(
    filename: str,
    format: str = "png",
    dpi: int = 300,
    bbox_inches: str = "tight",
    output_dir: str | None = None,
):
    """Save the current figure under the configured output root.

    ``output_dir`` defaults to :func:`paths.output_root` resolved **at call time**,
    so a ``DSE_LRP_OUTPUT_DIR`` env var or a CLI ``--output-dir`` (``set_output_root``)
    is honoured (#180); pass an explicit directory to override.
    """
    root = str(output_dir) if output_dir is not None else str(_paths.output_root())
    return _shared_save_figure(
        filename, root, format=format, dpi=dpi, bbox_inches=bbox_inches
    )


def display_image(filename: str, width: int = 600, output_dir: str | None = None):
    """Display an image from the configured output root in a notebook.

    ``output_dir`` defaults to :func:`paths.output_root` resolved at call time (#180).
    """
    root = str(output_dir) if output_dir is not None else str(_paths.output_root())
    return _shared_display_image(filename, root, width=width)
