"""
Approach 2: Calibrated classification.
Same BERT base + Platt scaling (sigmoid) or temperature scaling. Threshold 0.7.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Optional, Union

from .baseline_model import BaselineWeaponsClassifier

try:
    import config
    THRESHOLD = getattr(config, "THRESHOLD_CALIBRATED", 0.7)
except ImportError:
    THRESHOLD = 0.7


class CalibratedWeaponsClassifier:
    """
    BERT + post-hoc Platt scaling on validation scores.
    Decision: ban if calibrated P(weapons) > 0.7.
    """

    def __init__(
        self,
        base_classifier: Optional[BaselineWeaponsClassifier] = None,
        threshold: float = None,
    ):
        self.base = base_classifier or BaselineWeaponsClassifier()
        self.threshold = threshold if threshold is not None else THRESHOLD
        self.platt: Optional[LogisticRegression] = None  # fits logit(score) -> calibrated prob

    def fit_calibration(self, X_raw: List, y_true: np.ndarray):
        """
        Fit Platt scaling: X_raw = list of texts, y_true = binary labels.
        Uses base model to get scores, then fits sigmoid( A * score + B ).
        Falls back to no calibration (raw scores) if fit fails or only one class.
        """
        self.platt = None
        try:
            scores = self.base.predict_proba(X_raw)
        except Exception:
            return self
        if scores.ndim > 1:
            scores = scores.ravel()
        scores = np.asarray(scores, dtype=np.float64, copy=True).ravel()
        # Only keep finite scores in [0,1]; drop invalid rows
        valid = np.isfinite(scores) & (scores >= 0) & (scores <= 1)
        y_true = np.asarray(y_true).ravel()
        if len(y_true) != len(scores):
            return self
        scores = scores[valid]
        y_true = y_true[valid]
        if len(scores) < 10 or len(np.unique(y_true)) < 2:
            return self
        # Ensure binary 0/1 and contiguous for sklearn
        y_true = (y_true != 0).astype(np.int32)
        X = np.ascontiguousarray(scores.reshape(-1, 1).astype(np.float64))
        try:
            self.platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            self.platt.fit(X, y_true)
        except Exception:
            self.platt = None
        return self

    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        """Calibrated P(weapons). If calibration not fitted, returns base scores."""
        raw = self.base.predict_proba(text)
        if raw.ndim > 1:
            raw = raw.ravel()
        if self.platt is None:
            return raw
        X = np.ascontiguousarray(np.asarray(raw, dtype=np.float64).reshape(-1, 1))
        return self.platt.predict_proba(X)[:, 1]

    def predict(self, text: Union[str, List[str]]) -> np.ndarray:
        proba = self.predict_proba(text)
        return (proba > self.threshold).astype(np.int64)

    def decision_tier(self, score: float, audience: str = "general") -> str:
        return "ban" if score > self.threshold else "allow"
