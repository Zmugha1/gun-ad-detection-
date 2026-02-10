"""
Cost matrices for weapons detection: FN/FP costs by scenario.
Theory: Cost-Sensitive Learning (Elkan, 2001); stricter for content targeting minors.
"""
import numpy as np
from typing import Literal

AudienceType = Literal["minors", "adults", "general"]


def get_cost_matrix(audience: AudienceType = "general") -> np.ndarray:
    """
    Returns 2x2 cost matrix [predicted][actual]:
    rows: predicted (0=allow, 1=ban), cols: actual (0=benign, 1=weapons).
    Cost[i,j] = cost when we predict i and true label is j.
    """
    fn_cost = 50_000 if audience == "minors" else 10_000
    fp_cost = 100
    # [[TN_cost, FN_cost], [FP_cost, TP_cost]]
    return np.array([
        [0, fn_cost],   # predict allow: TN=0, FN=fn_cost
        [fp_cost, 0],   # predict ban: FP=fp_cost, TP=0
    ])


def get_cost_for_prediction(
    y_true: int,
    y_pred: int,
    audience: AudienceType = "general",
) -> float:
    """Cost of a single prediction given true label and audience."""
    cm = get_cost_matrix(audience)
    return float(cm[y_pred, y_true])


def total_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    audience: AudienceType = "general",
) -> float:
    """Total cost over a set of predictions."""
    cm = get_cost_matrix(audience)
    total = 0.0
    for a, p in zip(y_true, y_pred):
        total += cm[p, a]
    return total
