import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from bwlogger import StyleAdapter

logger = StyleAdapter(logging.getLogger(__name__))


def _format_percentage(x, _):
    return f"{x * 100:.1f}%"


def plot_results(
    df_returns_ln_acc: pd.DataFrame,
    s_return_ln_acc_strategy: pd.Series,
    s_alpha: pd.Series,
    s_beta: pd.Series,
    output_path: Path,
) -> None:
    x_range = (df_returns_ln_acc.index.min(), df_returns_ln_acc.index.max())

    # plot return
    fig, axes = plt.subplots(figsize=(18, 12), nrows=2)

    axes[0].set_title("BRL x Brazil CDS")
    df_returns_ln_acc.plot(ax=axes[0], color=["blue", "orange"])
    s_return_ln_acc_strategy.plot(ax=axes[0], label="Trading Strategy", color="green")
    axes[0].legend()

    # Add gridlines to the first plot
    axes[0].grid(True, color="gray", linestyle="--")

    # Add a horizontal line at y=0 to the first plot
    axes[0].axhline(y=0, color="black")

    # Move the y-axis label and ticks to the right for the first plot
    axes[0].yaxis.set_label_position("right")
    axes[0].yaxis.tick_right()
    axes[0].set_ylabel("Cumulative Log Returns")

    # Apply the percentage formatter to the first plot
    axes[0].yaxis.set_major_formatter(FuncFormatter(_format_percentage))

    # Set x-axis limits to the first and last data points
    axes[0].set_xlim(x_range)

    # beta and alpha plot
    s_beta.plot(ax=axes[1], color="blue", label="Beta")
    ax2 = axes[1].twinx()
    s_alpha.plot(ax=ax2, color="orange", label="Alpha")

    # Add gridlines to the second plot
    axes[1].grid(True, color="gray", linestyle="--")

    # Set y-axis labels
    axes[1].set_ylabel("Beta")
    ax2.set_ylabel("Alpha")

    # Set x-axis limits to the first and last data points
    axes[1].set_xlim(x_range)

    y_min, y_max = axes[1].get_ylim()
    y_range = y_max - y_min
    y_mid = (y_max + y_min) / 2
    axes[1].set_ylim(y_mid - y_range / 2, y_mid + y_range / 2 * 1.2)

    y_min, y_max = ax2.get_ylim()
    y_range = y_max - y_min
    y_mid = (y_max + y_min) / 2
    ax2.set_ylim(y_mid - y_range / 2, y_mid + y_range / 2 * 1.2)

    # Combine legends from both axes
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left", ncol=2)

    fig.tight_layout(
        rect=(0, 0, 1, 0.96)
    )  # Adjust the layout to make space for the title

    logger.info(f"Saving plot to {output_path}")
    fig.savefig(output_path)
    plt.close()


def scatter_plot(
    x_values: pd.Series,
    y_values: pd.Series,
    title_font_size: Optional[float] = None,
    font_family: Optional[float] = "Calibri",
    font_size: Optional[float] = 10,
    figure_size: Optional[tuple[float, tuple]] = (6.5, 4.5),
    marker_size: Optional[float] = 15,
    marker_transparency: Optional[float] = 0.3,
    marker_color: Optional[str] = "black",
    fit_type: Optional[Literal["linear", " quadratic"]] = "linear",
    line_color: Optional[str] = "gray",
    line_width: Optional[float] = 1,
    line_style: Optional[str] = "-.",
    yaxis_title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    plot_title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> None:
    DEFAULT_TITLE_SCATTER_PLOT = "Scatter Plot of %s vs. %s"

    x_values, y_values = x_values.copy(), y_values.copy()

    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figure_size)
    ax.scatter(
        x_values, y_values, s=marker_size, c=marker_color, alpha=marker_transparency
    )

    ax.axhline(y=0, color="gray")
    ax.axvline(x=0, color="gray")
    ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5)

    if fit_type:
        x_lim = ax.get_xlim()
        x_aux = np.linspace(*x_lim, 200)
        if fit_type == "linear":
            X = sm.add_constant(x_values)
            model = sm.OLS(y_values, X).fit()
        elif fit_type == "quadratic":
            X = np.column_stack((x_values, x_values**2))
            X = sm.add_constant(X)
            model = sm.OLS(y_values, X).fit()
            x_aux = np.column_stack((x_aux, x_aux**2))
        else:
            raise ValueError(
                "if passed, fit_type must be either 'linear' or 'quadratic'"
            )
        x_aux = sm.add_constant(x_aux)
        predictions = model.predict(x_aux)
        ax.plot(
            x_aux[:, 1],
            predictions,
            color=line_color,
            linewidth=line_width,
            linestyle=line_style,
        )
        ax.set_xlim(x_lim)

    ax.yaxis.set_major_formatter(FuncFormatter(_format_percentage))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_percentage))
    yaxis_title = yaxis_title or y_values.name
    xaxis_title = xaxis_title or x_values.name
    plot_title = plot_title or DEFAULT_TITLE_SCATTER_PLOT % (
        yaxis_title,
        xaxis_title,
    )

    ax.set_ylabel(yaxis_title, fontsize=font_size)
    ax.set_xlabel(xaxis_title, fontsize=font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size)
    ax.set_title(
        plot_title,
        fontsize=title_font_size or 1.2 * font_size,
    )
    if output_path:
        fig.savefig(output_path)
    plt.show()
    return


def stacked_area_plot(
    df_weights: pd.Series,
    title_font_size: Optional[float] = None,
    font_family: Optional[float] = "Calibri",
    font_size: Optional[float] = 10,
    figure_size: Optional[tuple[float, tuple]] = (8.5, 4.5),
    color_list: Optional[list[tuple[float, float, float]]] = None,
    xaxis_title: Optional[str] = None,
    plot_title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> None:
    DEFAULT_TITLE_SCATTER_PLOT = "% of Total Allocation"

    df = df_weights.copy()
    df = df[df.mean().sort_values(ascending=False).index].dropna(how="all")
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figure_size)

    color_list = color_list or sns.color_palette("tab20")

    df.plot(kind="area", stacked=True, ax=ax, alpha=0.7, color=color_list)

    ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5, zorder=2.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_percentage))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim(0, 1)

    if xaxis_title:
        ax.set_xlabel(xaxis_title, fontsize=font_size)

    plot_title = plot_title or DEFAULT_TITLE_SCATTER_PLOT

    ax.tick_params(axis="both", which="major", labelsize=font_size)

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.52, 1),
        ncol=5,
        fontsize=8,
        frameon=False,
    )

    fig.canvas.draw()
    legend_box = legend.get_window_extent()
    legend_coords = legend_box.transformed(ax.transAxes.inverted())

    fig.text(
        0.5,
        legend_coords.y0 - 0.03,
        plot_title,
        ha="center",
        va="bottom",
        fontsize=title_font_size or 1.2 * font_size,
        fontfamily=font_family,
    )

    plt.subplots_adjust(top=0.85)
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", format="svg")

    plt.show()
    return


def _add_gradient(
    ax,
    s_data,
    color,
    data_min,
    data_max,
    inverted: bool = False,
):
    percentil_default_max = 0.2
    percentil_default_min = 0
    percentil_min = percentil_default_min if not inverted else percentil_default_max
    percentil_max = 1 - percentil_default_max if not inverted else 1
    grad1 = ax.imshow(
        np.linspace(0, 1, 256).reshape(-1, 1),
        cmap=sns.light_palette(color, reverse=inverted, as_cmap=True),
        vmax=_area_percentil_to_y_norm(s_data, percentil_max),
        vmin=_area_percentil_to_y_norm(s_data, percentil_min),
        norm="symlog",
        aspect="auto",
        extent=[
            s_data.index.min(),
            s_data.index.max(),
            data_min,
            data_max,
        ],
        origin="lower",
        alpha=0.7,
        zorder=1.5,
    )
    poly_pos = ax.fill_between(s_data.index, s_data, 0)
    grad1.set_clip_path(poly_pos.get_paths()[0], transform=ax.transData)
    poly_pos.remove()


def line_plot(
    s_data: pd.Series,
    title_font_size: Optional[float] = None,
    font_family: Optional[float] = "Calibri",
    font_size: Optional[float] = 10,
    figure_size: Optional[tuple[float, tuple]] = (8.5, 4.5),
    color: Optional[tuple[float, float, float]] = "black",
    xaxis_title: Optional[str] = None,
    y_max: Optional[float] = None,
    y_min: Optional[float] = None,
    inverted: bool = False,
    plot_title: Optional[str] = None,
    add_gradient: Optional[bool] = True,
    output_path: Optional[Path] = None,
) -> None:
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figure_size)

    s_data.plot(ax=ax, color=color)

    ax.grid(True, which="both", color="gray", linestyle=":", linewidth=0.5, zorder=2.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_percentage))
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    if xaxis_title:
        ax.set_xlabel(xaxis_title, fontsize=font_size)

    if s_data.name:
        plot_title = plot_title or s_data.name
    if plot_title:
        ax.set_title(
            plot_title,
            fontsize=title_font_size or 1.2 * font_size,
        )

    ax.tick_params(axis="both", which="major", labelsize=font_size)
    y_min = y_min if y_min is not None else ax.get_ylim()[0]
    y_max = y_max if y_max is not None else ax.get_ylim()[1]
    ax.set_ylim(y_min, y_max)

    if add_gradient:
        data_min = 0 if y_min == 0 else s_data.min()
        data_max = 0 if y_max == 0 else s_data.max()
        _add_gradient(ax, s_data, color, data_min, data_max, inverted=inverted)

    ax.set_ylim(y_min, y_max)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight", format="svg")

    plt.show()
    return


def _area_percentil_to_y_norm(s_data: pd.Series, percentil: float) -> float:
    acc_pct = (s_data.sort_values() / s_data.sum()).cumsum()
    percentil_index = (acc_pct - percentil).abs().idxmin()
    percentil_value = s_data[percentil_index]
    return (percentil_value - s_data.min()) / (s_data.max() - s_data.min())
