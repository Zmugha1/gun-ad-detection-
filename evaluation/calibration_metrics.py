"""
Calibration metrics: ECE, MCE, Brier score, reliability diagram data.
References: Guo et al. (2017), Platt (1999).
"""
import numpy as np
from typing import Tuple, List


def _bin_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Return list of (bin_probs, bin_labels, bin_size) per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        result.append((y_prob[mask], y_true[mask], mask.sum()))
    return result


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted average of |acc(bin) - conf(bin)|."""
    bin_data = _bin_predictions(y_true, y_prob, n_bins)
    n = len(y_true)
    if n == 0:
        return 0.0
    ece_val = 0.0
    for probs, labels, count in bin_data:
        conf = probs.mean()
        acc = labels.mean()
        ece_val += (count / n) * abs(acc - conf)
    return float(ece_val)


def mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Maximum Calibration Error: max over bins of |acc - conf|."""
    bin_data = _bin_predictions(y_true, y_prob, n_bins)
    if not bin_data:
        return 0.0
    mce_val = 0.0
    for probs, labels, _ in bin_data:
        conf = probs.mean()
        acc = labels.mean()
        mce_val = max(mce_val, abs(acc - conf))
    return float(mce_val)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error of probability predictions."""
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (bin_centers, bin_accuracies, bin_counts) for reliability diagram.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    centers = []
    accs = []
    counts = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(y_true[mask].mean())
        counts.append(mask.sum())
    return np.array(centers), np.array(accs), np.array(counts)
