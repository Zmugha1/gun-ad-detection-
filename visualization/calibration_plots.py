"""
Calibration curves (reliability diagrams), probability histograms.
"""
import numpy as np
from typing import Optional, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from evaluation.calibration_metrics import reliability_diagram_data, ece
except ImportError:
    from ..evaluation.calibration_metrics import reliability_diagram_data, ece


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    ax: Optional[plt.Axes] = None,
    label: str = "Model",
) -> plt.Figure:
    """Plot predicted probability vs actual frequency with diagonal reference."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    centers, accs, counts = reliability_diagram_data(y_true, y_prob, n_bins)
    ece_val = ece(y_true, y_prob, n_bins)
    ax.bar(centers, accs, width=0.08, alpha=0.7, label=f"{label} (ECE={ece_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig


def plot_probability_histogram(
    y_prob: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Predicted probability distribution",
    label: str = "Scores",
    bins: int = 20,
) -> plt.Figure:
    """Histogram of predicted probabilities."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    ax.hist(y_prob, bins=bins, range=(0, 1), alpha=0.7, label=label, edgecolor="black")
    ax.set_xlabel("Predicted P(weapons)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    return fig
