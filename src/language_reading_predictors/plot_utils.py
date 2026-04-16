# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
import numpy as np
import pandas as pd
import matplotlib.axes as mpaxes
import matplotlib.figure as mpfig
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import graphviz as gr
import networkx as nx
from typing import Any, Iterable, Literal
from pathlib import Path
from scipy.cluster import hierarchy
from IPython.display import Image, display

from language_reading_predictors.data_variables import Variables as vars


HERE = Path(".")
OUTPUT_DIR = HERE / "output"


def hierarchy_dendrogram(
    Z,
    axis: mpaxes.Axes,
    labels=None,
    orientation: Literal["top", "right", "bottom", "left"] = "right",
):
    dendro = hierarchy.dendrogram(Z, labels=labels, orientation=orientation, ax=axis)
    return dendro


def plot_violinplot_t1_to_t3(data: pd.DataFrame, variable: str):
    data_t1 = data[data[vars.TIME] == 1][variable].dropna()
    data_t2 = data[data[vars.TIME] == 2][variable].dropna()
    data_t3 = data[data[vars.TIME] == 3][variable].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)

    for ax, d, t in zip(axes, [data_t1, data_t2, data_t3], [1, 2, 3, 4]):
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


def plot_violinplot_t1_to_t4(data: pd.DataFrame, variable: str):
    data_t1 = data[data[vars.TIME] == 1][variable].dropna()
    data_t2 = data[data[vars.TIME] == 2][variable].dropna()
    data_t3 = data[data[vars.TIME] == 3][variable].dropna()
    data_t4 = data[data[vars.TIME] == 4][variable].dropna()

    fig, axes = plt.subplots(1, 4, figsize=(12, 4), sharey=True)

    for ax, d, t in zip(axes, [data_t1, data_t2, data_t3, data_t4], [1, 2, 3, 4]):
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


def plot_histograms(
    data: pd.DataFrame,
    sharex: bool = False,
    sharey: bool = False,
    max_cols: int = 3,
    max_bins: int = 10,
    col_width: float = 4,
    col_height: float = 4,
    name_lookup: dict[str, str] | None = None,
) -> tuple[mpfig.Figure, Any]:
    """
    Plot histograms of all numeric columns in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing numeric columns.
    max_cols : int, optional
        Maximum number of columns in the subplot grid, by default 3.
    max_bins : int, optional
        Maximum number of bins for the histograms, by default 10.
    col_width : float, optional
        Width of each subplot column in inches, by default 4.
    col_height : float, optional
        Height of each subplot row in inches, by default 4.

    Returns
    -------
    fig : plt.Figure
        The figure object containing the subplots.
    axes : Any
        The array of the subplots.
    """
    df = data.select_dtypes(include=["number"])
    n_cols = df.shape[1]
    cols = min(max_cols, n_cols)
    rows = math.ceil(n_cols / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(col_width * cols, col_height * rows),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    fig.tight_layout(pad=4.0, w_pad=2.0, h_pad=4.0)

    axes_flat = axes.flatten()

    for ax, v in zip(axes_flat, data.columns):
        series = df[v].dropna().to_numpy(dtype=np.float64)
        bins = min(max_bins, max(2, int(series.max() - series.min())))

        sns.histplot(data=series, bins=bins, kde=True, ax=ax)

        if name_lookup is not None:
            name = name_lookup.get(v)
            name = name if name is not None else v
        else:
            name = v

        ax.set_title(f"{name} distribution")
        ax.set_xlabel(name)
        ax.set_ylabel("Count")

    for ax in axes_flat[n_cols:]:
        ax.set_visible(False)

    return fig, axes


def plot_kernel_function(
    kernel_function, max_distance=1, resolution=100, label=None, ax=None, **line_kwargs
):
    """Helper to plot a kernel function"""
    X = np.linspace(0, max_distance, resolution)[:, None]
    covariance = kernel_function(X, X)
    distances = np.linspace(0, max_distance, resolution)
    if ax is not None:
        plt.sca(ax)
    plot_line(distances, covariance[0, :], label=label, **line_kwargs)
    plt.xlim([0, max_distance])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("|X1-X2|")
    plt.ylabel("covariance")
    if label is not None:
        plt.legend()


# Helper functions
def plot_gaussian_process(
    X,
    samples: Iterable | None = None,
    mean=None,
    cov=None,
    X_obs=None,
    Y_obs=None,
    uncertainty_prob=0.89,
):
    X = X.ravel()

    # Plot GP samples
    for ii, sample in enumerate(samples):
        label = "GP samples" if not ii else None
        plot_line(X, sample, color=f"C{ii}", linewidth=1, label=label)

    # Add GP mean, if provided
    if mean is not None:
        mean = mean.ravel()
        plot_line(X, mean, color="k", label="GP mean")

        # Add uncertainty around mean; requires covariance matrix
        if cov is not None:
            z = stats.norm.ppf(1 - (1 - uncertainty_prob) / 2)
            uncertainty = z * np.sqrt(np.diag(cov))
            plt.fill_between(
                X,
                mean + uncertainty,
                mean - uncertainty,
                alpha=0.1,
                color="gray",
                zorder=1,
                label="GP uncertainty",
            )

    # Add any training data points, if provided
    if X_obs is not None:
        plot_scatter(X_obs, Y_obs, color="k", label="observations", zorder=100, alpha=1)

    plt.xlim([X.min(), X.max()])
    plt.ylim([-5, 5])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()


def plot_gaussian_process_prior(
    kernel_function, n_samples=3, figsize=(10, 5), resolution=100
):
    X = np.linspace(-5, 5, resolution)[:, None]

    prior = gaussian_process_prior(X, kernel_function)
    samples = prior.rvs(n_samples)

    _, axs = plt.subplots(1, 2, figsize=figsize)
    plt.sca(axs[0])
    plot_gaussian_process(X, samples=samples)

    plt.sca(axs[1])
    plot_kernel_function(kernel_function, color="k")
    plt.title("kernel function")
    return axs


def gaussian_process_prior(X_pred, kernel_function):
    """Initializes a Gaussian Process prior distribution for provided Kernel function"""
    mean = np.zeros(X_pred.shape).ravel()
    cov = kernel_function(X_pred, X_pred)
    return stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)


# from https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
def draw_causal_graph(
    edge_list, node_props=None, edge_props=None, graph_direction="UD"
):
    """Utility to draw a causal (directed) graph"""
    g = gr.Digraph(graph_attr={"rankdir": graph_direction})

    edge_props = {} if edge_props is None else edge_props
    for e in edge_list:
        props = edge_props[e] if e in edge_props else {}
        g.edge(e[0], e[1], **props)

    if node_props is not None:
        for name, props in node_props.items():
            g.node(name=name, **props)
    return g


# from https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
def plot_scatter(xs, ys, **scatter_kwargs):
    """Draw scatter plot with consistent style (e.g. unfilled points)"""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none"}

    for k, v in defaults.items():
        val = scatter_kwargs.get(k, v)
        scatter_kwargs[k] = val

    plt.scatter(xs, ys, **scatter_kwargs)


# from https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
def plot_line(xs, ys, **plot_kwargs):
    """Plot line with consistent style (e.g. bordered lines)"""
    linewidth = plot_kwargs.get("linewidth", 3)
    plot_kwargs["linewidth"] = linewidth

    # Copy settings for background
    background_plot_kwargs = {k: v for k, v in plot_kwargs.items()}
    background_plot_kwargs["linewidth"] = linewidth + 2
    background_plot_kwargs["color"] = "white"
    del background_plot_kwargs["label"]  # no legend label for background

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)


# from https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
def plot_errorbar(
    xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3
):
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    """Draw thick error bars with consistent style"""
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            yerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


# from https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
def plot_x_errorbar(
    xs, ys, error_lower, error_upper, colors="C0", error_width=12, alpha=0.3
):
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    """Draw thick error bars with consistent style"""
    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            xerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


# from https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
def plot_graph(graph, **graph_kwargs):
    """Draw a network graph.

    graph: Union[networkx.DiGraph, np.ndarray]
        if ndarray, assume `graph` is an adjacency matrix defining
        a directed graph.

    """
    # convert to networkx.DiGraph, if needed
    G = (
        nx.from_numpy_array(graph, create_using=nx.DiGraph)
        if isinstance(graph, np.ndarray)
        else graph
    )

    # Set default styling
    np.random.seed(123)  # for consistent spring-layout
    if "layout" in graph_kwargs:
        graph_kwargs["pos"] = graph_kwargs["layout"](G)

    default_graph_kwargs = {
        "node_color": "C0",
        "node_size": 500,
        "arrowsize": 30,
        "width": 3,
        "alpha": 0.7,
        "connectionstyle": "arc3,rad=0.1",
        "pos": nx.kamada_kawai_layout(G),
    }
    for k, v in default_graph_kwargs.items():
        if k not in graph_kwargs:
            graph_kwargs[k] = v

    nx.draw(G, **graph_kwargs)
    # return the node layout for consistent graphing
    return graph_kwargs["pos"]


# from https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
def plot_2d_function(xrange, yrange, func, ax=None, **countour_kwargs):
    """Evaluate the function `func` over the values of xrange and yrange and
    plot the resulting value contour over that range.

    Parameters
    ----------
    xrange : np.ndarray
        The horizontal values to evaluate/plot
    yrange : p.ndarray
        The horizontal values to evaluate/plot
    func : Callable
        function of two arguments, xs and ys. Should return a single value at
        each point.
    ax : matplotlib.Axis, optional
        An optional axis to plot the function, by default None

    Returns
    -------
    contour : matplotlib.contour.QuadContourSet
    """
    resolution = len(xrange)
    xs, ys = np.meshgrid(xrange, yrange)
    xs = xs.ravel()
    ys = ys.ravel()

    value = func(xs, ys)

    if ax is not None:
        plt.sca(ax)

    return plt.contour(
        xs.reshape(resolution, resolution),
        ys.reshape(resolution, resolution),
        value.reshape(resolution, resolution),
        **countour_kwargs,
    )


OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def plot_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    *,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    ax: mpaxes.Axes | None = None,
    tick_fontsize: float = 12,
    title_fontsize: float = 12,
    colorbar_fraction: float = 0.03,
    colorbar_pad: float = 0.025,
) -> tuple[mpfig.Figure, mpaxes.Axes]:
    """Plot a square heatmap using ``imshow``.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix of values to display.
    labels : list[str]
        Tick labels for both axes (length must match matrix dimensions).
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size. Defaults to ``(max(8, 0.35*n), max(8, 0.35*n))``.
    cmap : str
        Matplotlib colour map name.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If ``None`` a new figure is created.
    tick_fontsize : float
        Font size for tick labels.
    title_fontsize : float
        Font size for the title.
    colorbar_fraction : float
        Fraction of axes given to the colorbar.
    colorbar_pad : float
        Padding between axes and colorbar.

    Returns
    -------
    fig, ax
    """
    n = len(labels)
    idx = np.arange(n)

    if ax is None:
        sz = figsize or (max(8, 0.35 * n), max(8, 0.35 * n))
        fig, ax = plt.subplots(figsize=sz)
    else:
        fig = ax.get_figure()

    with plt.rc_context(
        {
            "ytick.labelsize": tick_fontsize,
            "xtick.labelsize": tick_fontsize,
            "axes.titlesize": title_fontsize,
        }
    ):
        im = ax.imshow(matrix, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(idx)
        ax.set_yticks(idx)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=colorbar_fraction, pad=colorbar_pad)

    return fig, ax


# from https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
def save_figure(
    filename: str,
    format: str = "png",
    dpi: int = 300,
    bbox_inches: str = "tight",
):
    """Save a figure to the `./images` directory"""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    figure_path = OUTPUT_DIR / filename
    print(f"saving figure to {figure_path}")
    plt.savefig(figure_path, format=format, dpi=dpi, bbox_inches=bbox_inches)


# from https://github.com/pymc-devs/pymc-examples/blob/main/examples/statistical_rethinking_lectures/utils.py
def display_image(filename: str, width: int = 600):
    """
    Display an image saved to the `./output` directory
    """
    return display(Image(filename=f"output/{filename}", width=width))
