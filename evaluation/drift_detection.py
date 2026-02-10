"""
Concept drift: KL divergence between training and production score distributions.
PSI (Population Stability Index) optional. Alert when KL > threshold.
"""
import numpy as np
from typing import Tuple


def _histogram_pdf(values: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Binned PDF; returns (bin_edges_mid, probs) with small epsilon to avoid log(0)."""
    hist, bin_edges = np.histogram(values, bins=bins, range=(0, 1), density=False)
    probs = hist / (hist.sum() + 1e-12) + 1e-10
    mid = (bin_edges[:-1] + bin_edges[1:]) / 2
    return mid, probs


def kl_divergence_bins(
    p_train: np.ndarray,
    p_prod: np.ndarray,
    bins: int = 20,
) -> float:
    """
    KL(prod || train) on binned score distributions in [0,1].
    """
    train_hist, _ = np.histogram(p_train, bins=bins, range=(0, 1), density=False)
    prod_hist, _ = np.histogram(p_prod, bins=bins, range=(0, 1), density=False)
    train_p = train_hist / (train_hist.sum() + 1e-12) + 1e-10
    prod_p = prod_hist / (prod_hist.sum() + 1e-12) + 1e-10
    return float(np.sum(prod_p * (np.log(prod_p) - np.log(train_p))))


def psi_score(p_train: np.ndarray, p_prod: np.ndarray, bins: int = 20) -> float:
    """Population Stability Index between train and prod score distributions."""
    train_hist, _ = np.histogram(p_train, bins=bins, range=(0, 1), density=False)
    prod_hist, _ = np.histogram(p_prod, bins=bins, range=(0, 1), density=False)
    train_p = train_hist / (train_hist.sum() + 1e-12) + 1e-10
    prod_p = prod_hist / (prod_hist.sum() + 1e-12) + 1e-10
    return float(np.sum((prod_p - train_p) * (np.log(prod_p) - np.log(train_p))))


def detect_drift(
    p_train: np.ndarray,
    p_prod: np.ndarray,
    kl_threshold: float = 0.1,
) -> Tuple[bool, float]:
    """Returns (is_drift, kl_value)."""
    kl = kl_divergence_bins(p_train, p_prod)
    return (kl > kl_threshold, kl)
