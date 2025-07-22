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
import logging


# Configure module-specific logger
logger = logging.getLogger(__name__)


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

    # Determine the number of epochs
    # Safely get the length of any non-empty metric list to determine num_epochs
    num_epochs = 0
    if hist:
        # Get the first metric list that exists to determine the number of epochs
        for metric_list in hist.values():
            if metric_list:  # Ensure the list itself is not empty
                num_epochs = len(metric_list)
                break

    if num_epochs == 0:
        print("No training data found in history object. Plotting empty.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Training History (No Data)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        fig.tight_layout()
        plt.show()
        return

    epochs = range(num_epochs)

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Iterate through each metric to plot
    for metric in metrics:
        if metric not in hist:
            print(f"Metric '{metric}' not found in history. Skipping.")
            continue

        # Plot training metric
        ax.plot(
            epochs,
            hist[metric],
            "o-",
            markersize=6,
            label=f"Train {metric.title()}",
        )

        # Plot validation metric if available
        val_metric = f"val_{metric}"
        if val_metric in hist:
            ax.plot(
                epochs,
                hist[val_metric],
                "o-",
                markersize=6,
                label=f"Val {metric.title()}",
            )

    # Axis labels and title
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(grid)

    # Ensure x-ticks are on whole integers only
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Adjust x-axis limits for single epoch to clearly show the point
    if num_epochs == 1:
        # Set x-limits to slightly before and after the epoch index 0
        ax.set_xlim(-0.5, 0.5)
        # Ensure only 0 is shown as a tick
        ax.set_xticks([0])
        # Also adjust y-axis limits to give the point some vertical space
        min_val = float("inf")
        max_val = float("-inf")
        for metric in metrics:
            if metric in hist and hist[metric]:  # Check if metric exists and has data
                min_val = min(min_val, hist[metric][0])
                max_val = max(max_val, hist[metric][0])
            val_metric = f"val_{metric}"
            if (
                val_metric in hist and hist[val_metric]
            ):  # Check if val_metric exists and has data
                min_val = min(min_val, hist[val_metric][0])
                max_val = max(max_val, hist[val_metric][0])

        if min_val != float("inf") and max_val != float("-inf"):
            # Add a small padding, adjust if min_val and max_val are the same
            padding = (max_val - min_val) * 0.2 if (max_val - min_val) > 0 else 0.1
            ax.set_ylim(min_val - padding, max_val + padding)

    # Display the plot
    fig.tight_layout()
    plt.show()
