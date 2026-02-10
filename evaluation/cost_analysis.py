"""
Business impact: total cost (FN/FP weighted), human review queue size.
"""
import numpy as np
from typing import Literal

try:
    from data.cost_matrices import get_cost_matrix, total_cost as _total_cost
    AudienceType = Literal["minors", "adults", "general"]
except ImportError:
    from ..data.cost_matrices import get_cost_matrix, total_cost as _total_cost
    AudienceType = Literal["minors", "adults", "general"]


def compute_total_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    audience: AudienceType = "general",
) -> float:
    """Total cost given predictions and audience."""
    return _total_cost(y_true, y_pred, audience)


def human_review_queue_size(
    scores: np.ndarray,
    low: float = 0.40,
    high: float = 0.90,
) -> int:
    """Count of samples in [low, high) (human review tier)."""
    return int(((scores >= low) & (scores < high)).sum())
