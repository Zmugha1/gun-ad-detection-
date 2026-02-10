"""
Synthetic ad dataset for weapons detection.
Realistic class imbalance (1-5% positive), text + metadata; no external APIs.
"""
import random
from pathlib import Path
import pandas as pd
import numpy as np

# Project config
try:
    import config
    DATA_DIR = config.DATA_DIR
    SYNTHETIC_ADS_PATH = config.SYNTHETIC_ADS_PATH
    SYNTHETIC_N_SAMPLES = config.SYNTHETIC_N_SAMPLES
    SYNTHETIC_POSITIVE_RATE = config.SYNTHETIC_POSITIVE_RATE
    RANDOM_STATE = config.RANDOM_STATE
except ImportError:
    DATA_DIR = Path(__file__).resolve().parent
    SYNTHETIC_ADS_PATH = DATA_DIR / "synthetic_ads.csv"
    SYNTHETIC_N_SAMPLES = 5000
    SYNTHETIC_POSITIVE_RATE = 0.03
    RANDOM_STATE = 42

# Realistic keywords and phrases
WEAPONS_TITLE_PHRASES = [
    "tactical gear", "private sale", "no serial", "cash only", "face to face",
    "collector piece", "vintage firearm", "hunting rifle", "self defense",
    "concealed carry", "ammo bulk", "reloading supplies", "AR parts",
    "upper receiver", "lower receiver", "magazine 30rd", "no background",
]
BENIGN_TITLE_PHRASES = [
    "vintage camera", "furniture sale", "kids toys", "books", "bicycle",
    "garden tools", "kitchen set", "laptop", "phone", "clothing",
    "sports equipment", "camping tent", "fishing rod", "art supplies",
]

WEAPONS_DESC_PHRASES = [
    "no paperwork", "discreet shipping", "OBO", "trade considered",
    "never fired", "like new", "FFL not required", "private party",
    "cash preferred", "meet local", "no questions",
]
BENIGN_DESC_PHRASES = [
    "pickup only", "negotiable", "good condition", "smoke free home",
    "moving sale", "best offer", "local pickup", "shipping available",
]

AGE_TARGETING = ["minors", "adults", "general"]


def _random_text(phrases: list, n_words: int = 5) -> str:
    return " ".join(random.choices(phrases, k=min(n_words, len(phrases))))


def _one_ad(is_weapons: bool, rng: np.random.Generator) -> dict:
    if is_weapons:
        title = _random_text(WEAPONS_TITLE_PHRASES, rng.integers(2, 5))
        desc = _random_text(WEAPONS_DESC_PHRASES, rng.integers(3, 8))
        keywords = " ".join(rng.choice(WEAPONS_TITLE_PHRASES, size=rng.integers(2, 5), replace=True))
    else:
        title = _random_text(BENIGN_TITLE_PHRASES, rng.integers(2, 5))
        desc = _random_text(BENIGN_DESC_PHRASES, rng.integers(3, 8))
        keywords = " ".join(rng.choice(BENIGN_TITLE_PHRASES, size=rng.integers(2, 5), replace=True))

    # Simulated metadata (in production: CLIP / image model)
    embedding_dim = 64
    image_meta = rng.standard_normal(embedding_dim).tolist()
    account_age_days = int(rng.integers(1, 1000))
    history_score = float(rng.uniform(0.0, 1.0))
    age_target = rng.choice(AGE_TARGETING)

    return {
        "ad_id": None,  # filled in batch
        "title": title,
        "description": desc,
        "keywords": keywords,
        "image_embedding": image_meta,
        "account_age_days": account_age_days,
        "history_score": history_score,
        "age_targeting": age_target,
        "label": 1 if is_weapons else 0,
    }


def generate_synthetic_ads(
    n: int = None,
    positive_rate: float = None,
    output_path: Path = None,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate synthetic ad dataset. Saves CSV and returns DataFrame.
    Columns: ad_id, title, description, keywords, account_age_days, history_score,
             age_targeting, label. (image_embedding stored as string repr for CSV.)
    """
    n = n or SYNTHETIC_N_SAMPLES
    positive_rate = positive_rate or SYNTHETIC_POSITIVE_RATE
    output_path = output_path or SYNTHETIC_ADS_PATH
    seed = seed if seed is not None else RANDOM_STATE
    rng = np.random.default_rng(seed)

    n_pos = int(round(n * positive_rate))
    n_neg = n - n_pos
    labels = [1] * n_pos + [0] * n_neg
    rng.shuffle(labels)

    rows = []
    for i, is_weapons in enumerate(labels):
        row = _one_ad(bool(is_weapons), rng)
        row["ad_id"] = i
        rows.append(row)

    df = pd.DataFrame(rows)
    # Store image embedding as string for CSV (optional: drop or use pickle for full vector)
    if "image_embedding" in df.columns:
        df["image_embedding_str"] = df["image_embedding"].apply(lambda x: str(x)[:200])
        df = df.drop(columns=["image_embedding"])
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_synthetic_ads()
    print("Generated", len(df), "ads. Positive rate:", df["label"].mean())
    print("Sample:", df.head(2))
