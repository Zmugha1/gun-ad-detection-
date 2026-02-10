"""
Configuration for Weapons Detection Content Moderation System.
Thresholds, costs, model paths, and theory-constrained decision architecture.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure dirs exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Model
BERT_MODEL_NAME = "bert-base-uncased"
DEVICE = "cpu"  # Use "cuda" if GPU available

# Decision thresholds (3-tier: Theory-Constrained)
THRESHOLD_AUTO_BAN = 0.90   # Score > 0.90: Auto-ban
THRESHOLD_HUMAN_REVIEW_LOW = 0.40   # Score 0.40-0.90: Human review
THRESHOLD_ALLOW = 0.40      # Score < 0.40: Allow with monitoring

# Age-dependent thresholds (strict liability for minors)
THRESHOLD_MINORS = 0.30   # Content targeting <18: stricter recall
THRESHOLD_ADULTS = 0.50   # Content targeting 18+: balanced
THRESHOLD_GENERAL = 0.50  # General audience

# Baseline/Calibrated default threshold
THRESHOLD_BASELINE = 0.5
THRESHOLD_CALIBRATED = 0.7

# Cost matrix (business impact)
COST_FALSE_NEGATIVE = 10_000   # $ per missed weapons ad (legal/safety)
COST_FALSE_POSITIVE = 100     # $ per wrongful ban
COST_FN_MINORS = 50_000       # $ per FN when targeting minors (regulatory/child safety)

# Cost-sensitive loss: FN weight vs FP (Elkan-style)
FN_WEIGHT = 10.0   # Weight for positive class in weighted cross-entropy

# Calibration
N_BINS_CALIBRATION = 10
DRIFT_KL_THRESHOLD = 0.1   # KL divergence > this triggers retraining warning

# Data
SYNTHETIC_ADS_PATH = DATA_DIR / "synthetic_ads.csv"
SYNTHETIC_N_SAMPLES = 5000
SYNTHETIC_POSITIVE_RATE = 0.03  # 3% weapons (rare event)
RANDOM_STATE = 42

# Human review
HUMAN_REVIEW_COST_PER_HOUR = 25.0  # Default $/hour
