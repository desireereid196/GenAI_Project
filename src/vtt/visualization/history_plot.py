"""
history_plot.py

This module provides visualization utilities for plotting training metrics
from a Keras model training process.

Specifically, it contains a function that takes a `History` object returned
by `model.fit()` and plots the training and validation metrics (such as loss,
accuracy, etc.) over epochs. These plots are useful for diagnosing issues like
overfitting, underfitting, and convergence behavior.
"""

from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tensorflow.keras.callbacks import History


def plot_training_history(
    history: History,
    metrics: Sequence[str] = ("loss",),
    figsize: Tuple[int, int] = (8, 4),
    grid: bool = False,
) -> None:
    """
    Plot training and validation metrics from a Keras History object.

    This function iterates over the specified metrics and generates line plots
    comparing training and validation values for each epoch. It is useful for
    visualizing model performance over time.

    Args:
        history (History): The `History` object returned by `model.fit()`, which
                           contains training and validation metrics for each epoch.
        metrics (Sequence[str]): A list or tuple of metric names to plot
                                 (e.g., ("loss", "accuracy")). Defaults to ("loss",).
        figsize (Tuple[int, int]): Size of the figure in inches. Defaults to (8, 4).
        grid (bool): Whether to display a grid on the plots. Defaults to False.

    Raises:
        ValueError: If the input does not appear to be a valid Keras `History` object.
    """
    # Check that the input is a valid Keras History object
    if not hasattr(history, "history"):
        raise ValueError("Invalid history object passed to plot_training_history().")

    hist = history.history

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Iterate through each metric to plot
    for metric in metrics:
        if metric not in hist:
            print(f"Metric '{metric}' not found in history. Skipping.")
            continue

        # Plot training metric
        ax.plot(hist[metric], label=f"Train {metric.title()}")

        # Plot validation metric if available
        val_metric = f"val_{metric}"
        if val_metric in hist:
            ax.plot(hist[val_metric], label=f"Val {metric.title()}")

    # Axis labels and title
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(grid)
    # Ensure x-ticks are on whole integers only
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Display the plot
    fig.tight_layout()
    plt.show()
