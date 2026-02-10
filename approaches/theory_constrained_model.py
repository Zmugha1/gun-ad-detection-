"""
Approach 3: Theory-constrained cost-sensitive system.
Cost-weighted loss, Isotonic calibration, 3-tier decision + age-dependent thresholds.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import List, Optional, Union

from .baseline_model import BaselineWeaponsClassifier

try:
    import config
    THRESHOLD_AUTO_BAN = getattr(config, "THRESHOLD_AUTO_BAN", 0.90)
    THRESHOLD_HUMAN_LOW = getattr(config, "THRESHOLD_HUMAN_REVIEW_LOW", 0.40)
    THRESHOLD_MINORS = getattr(config, "THRESHOLD_MINORS", 0.30)
    THRESHOLD_ADULTS = getattr(config, "THRESHOLD_ADULTS", 0.50)
    THRESHOLD_GENERAL = getattr(config, "THRESHOLD_GENERAL", 0.50)
except ImportError:
    THRESHOLD_AUTO_BAN = 0.90
    THRESHOLD_HUMAN_LOW = 0.40
    THRESHOLD_MINORS = 0.30
    THRESHOLD_ADULTS = 0.50
    THRESHOLD_GENERAL = 0.50


def _audience_threshold(audience: str) -> float:
    if audience == "minors":
        return THRESHOLD_MINORS
    if audience == "adults":
        return THRESHOLD_ADULTS
    return THRESHOLD_GENERAL


class TheoryConstrainedWeaponsClassifier:
    """
    Cost-sensitive setup (conceptually: weighted loss at train time; here we use
    same base BERT and calibrate with Isotonic). 3-tier decision + age-dependent thresholds.
    """

    def __init__(self, base_classifier: Optional[BaselineWeaponsClassifier] = None):
        self.base = base_classifier or BaselineWeaponsClassifier()
        self.isotonic: Optional[IsotonicRegression] = None

    def fit_calibration(self, X_raw: List, y_true: np.ndarray):
        """Fit Isotonic regression: raw score -> calibrated P(weapons). Fall back to no calibration on any error."""
        self.isotonic = None
        try:
            scores = self.base.predict_proba(X_raw)
            if hasattr(scores, "ndim") and scores.ndim > 1:
                scores = scores.ravel()
            scores = np.asarray(scores, dtype=np.float64).ravel()
            y_true = np.asarray(y_true).ravel()
            if len(scores) != len(X_raw) or len(y_true) != len(X_raw):
                return self
            valid = np.isfinite(scores) & (scores >= 0) & (scores <= 1)
            scores = scores[valid]
            y_true = y_true[valid]
            if len(scores) < 5:
                return self
            self.isotonic = IsotonicRegression(out_of_bounds="clip")
            self.isotonic.fit(scores, y_true)
        except Exception:
            self.isotonic = None
        return self

    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        raw = self.base.predict_proba(text)
        if raw.ndim > 1:
            raw = raw.ravel()
        if self.isotonic is None:
            return raw
        return self.isotonic.predict(raw)

    def predict(self, text: Union[str, List[str]], audience: str = "general") -> np.ndarray:
        proba = self.predict_proba(text)
        th = _audience_threshold(audience)
        return (proba > th).astype(np.int64)

    def decision_tier(self, score: float, audience: str = "general") -> str:
        """
        3-tier: >0.90 auto-ban; 0.40–0.90 human review; <0.40 allow.
        Age-dependent: minors use 0.30 as lower bound for human review (stricter).
        """
        th_ban = THRESHOLD_AUTO_BAN
        th_review_low = THRESHOLD_MINORS if audience == "minors" else THRESHOLD_HUMAN_LOW
        if score >= th_ban:
            return "auto_ban"
        if score >= th_review_low:
            return "human_review"
        return "allow"
