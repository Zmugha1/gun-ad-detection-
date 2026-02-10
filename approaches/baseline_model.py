"""
Approach 1: Baseline binary classification.
BERT fine-tuned, standard 0.5 threshold. No calibration, no cost sensitivity.
"""
import numpy as np
from typing import List, Optional, Union
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    torch = None  # type: ignore
    HAS_TRANSFORMERS = False

# Default config
try:
    import config
    BERT_NAME = getattr(config, "BERT_MODEL_NAME", "bert-base-uncased")
    DEVICE = getattr(config, "DEVICE", "cpu")
    THRESHOLD = getattr(config, "THRESHOLD_BASELINE", 0.5)
except ImportError:
    BERT_NAME = "bert-base-uncased"
    DEVICE = "cpu"
    THRESHOLD = 0.5


class BaselineWeaponsClassifier:
    """
    Standard BERT binary classifier. Decision: ban if P(weapons) > 0.5.
    """

    def __init__(
        self,
        model_name: str = None,
        threshold: float = None,
        device: str = None,
    ):
        self.model_name = model_name or BERT_NAME
        self.threshold = threshold if threshold is not None else THRESHOLD
        self.device = device or DEVICE
        self.model = None
        self.tokenizer = None
        self._fallback = None  # keyword-based fallback if no torch/transformers

    def _ensure_model(self):
        if self.model is not None:
            return
        if not HAS_TRANSFORMERS or not torch.cuda.is_available() and self.device == "cuda":
            self.device = "cpu"
        if HAS_TRANSFORMERS:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=2
            )
            self.model.to(self.device)
            self.model.eval()
        else:
            self._fallback = _KeywordFallback()

    def _text_to_input(self, text: Union[str, List[str]]) -> str:
        if isinstance(text, list):
            return " ".join(str(t) for t in text)
        return str(text)

    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        """Return P(weapons) for each sample. Shape (n,) or (1,) for single."""
        self._ensure_model()
        if self._fallback is not None:
            return self._fallback.predict_proba(text)
        single = isinstance(text, str) or (isinstance(text, list) and len(text) > 0 and isinstance(text[0], str))
        if single:
            text = [self._text_to_input(text)]
        else:
            text = [self._text_to_input(t) for t in text]
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        return probs

    def predict(self, text: Union[str, List[str]]) -> np.ndarray:
        """Binary prediction: 1 if P(weapons) > threshold else 0."""
        proba = self.predict_proba(text)
        return (proba > self.threshold).astype(np.int64)

    def decision_tier(self, score: float, audience: str = "general") -> str:
        """Always uses single threshold (no age-dependent tiers)."""
        return "ban" if score > self.threshold else "allow"


class _KeywordFallback:
    """Simple keyword-based score when transformers not available."""
    WEAPONS_WORDS = {"tactical", "sale", "serial", "cash", "firearm", "rifle", "ammo", "private", "no background", "magazine", "receiver", "concealed", "carry", "reloading"}

    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        text = (text or "").lower()
        hits = sum(1 for w in self.WEAPONS_WORDS if w in text)
        # Saturate around 0.2–0.9
        score = min(0.95, 0.2 + 0.7 * min(1.0, hits / 5))
        return np.array([score])
