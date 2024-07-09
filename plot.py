import logging
from pathlib import Path
from typing import Optional

import pandas as pd
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
    output_path: Optional[Path] = None,
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

    output_path = output_path or SCRIPT_DIR.joinpath("plot.svg")
    logger.info(f"Saving plot to {output_path}")
    fig.savefig(output_path)
    plt.close()
