"""
features/engineering.py — Session-level tabular features.

What this file does:
  Reads sequences.parquet and derives ~15 simple number features
  per session — things the Logistic Regression and XGBoost
  baselines will train on in Week 2.

  These features deliberately have NO sequence information.
  That is the point — they represent the best a non-sequence
  model can do. When the Transformer beats them, the win is earned.

Features produced per session:
  seq_len        — total events (including gap tokens)
  n_views        — number of product views
  n_carts        — number of add-to-cart events
  n_purchases    — number of purchase events
  cart_rate      — n_carts / n_views (intent signal)
  view_depth     — n_views / seq_len (viewing density)
  has_cart       — 1 if user ever added to cart, else 0
  n_gap_short    — number of hesitation gaps (2-10 min)
  n_gap_long     — number of abandonment gaps (10+ min)
  gap_ratio      — total gaps / seq_len
  last_event     — token ID of the final action in the session
  last_is_cart   — 1 if final action was add-to-cart
  last_is_view   — 1 if final action was a view
  view_cart_ratio — n_views per cart action (lower = more decisive)
  source_otto    — 1 if session from OTTO, 0 if Retail Rocket
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sessionscout.config import cfg

logger = logging.getLogger(__name__)


def build_session_features(sequences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive tabular features from the sequences DataFrame.

    Args:
        sequences_df: the output of build_sequence_dataset()
                      must have columns: session_id, source, sequence,
                      seq_len, label, n_views, n_carts, n_purchases

    Returns:
        DataFrame with one row per session and ~15 feature columns.
    """
    df = sequences_df.copy()
    feat = pd.DataFrame()

    # Identity columns
    feat["session_id"] = df["session_id"]
    feat["label"]      = df["label"]
    feat["source"]     = df["source"]

    # Raw counts — come directly from the sequences DataFrame
    feat["seq_len"]     = df["seq_len"]
    feat["n_views"]     = df["n_views"]
    feat["n_carts"]     = df["n_carts"]
    feat["n_purchases"] = df["n_purchases"]

    # Ratio features
    # cart_rate: how often does viewing lead to adding to cart?
    # A high cart_rate = decisive shopper, more likely to buy
    feat["cart_rate"] = df["n_carts"] / (df["n_views"] + 1e-8)

    # view_depth: what fraction of session events are views?
    feat["view_depth"] = df["n_views"] / (df["seq_len"] + 1e-8)

    # has_cart: binary flag — did the user ever add to cart?
    # This single feature is one of the strongest predictors
    feat["has_cart"] = (df["n_carts"] > 0).astype(int)

    # Gap counts — count specific token IDs inside the sequence
    def count_token(seq, token_id):
        return sum(1 for t in seq if t == token_id)

    feat["n_gap_short"] = df["sequence"].apply(
        lambda s: count_token(s, cfg.vocab.gap_short)
    )
    feat["n_gap_long"] = df["sequence"].apply(
        lambda s: count_token(s, cfg.vocab.gap_long)
    )

    # gap_ratio: what fraction of the session was inactivity?
    # High gap_ratio with a cart = hesitating shopper (our target)
    feat["gap_ratio"] = (
        (feat["n_gap_short"] + feat["n_gap_long"]) / (df["seq_len"] + 1e-8)
    )

    # Last event features — what did the user do right before leaving?
    def last_real_token(seq):
        """Return the last non-PAD token in the sequence."""
        for tok in reversed(seq):
            if tok != cfg.vocab.pad:
                return tok
        return cfg.vocab.pad

    feat["last_event"]   = df["sequence"].apply(last_real_token)
    feat["last_is_cart"] = (feat["last_event"] == cfg.vocab.add_cart).astype(int)
    feat["last_is_view"] = (feat["last_event"] == cfg.vocab.view).astype(int)

    # view_cart_ratio: how many views per cart action?
    # A decisive buyer adds to cart quickly (low ratio)
    # A browser views many times but rarely carts (high ratio)
    feat["view_cart_ratio"] = df["n_views"] / (df["n_carts"] + 1e-8)
    feat["view_cart_ratio"] = feat["view_cart_ratio"].clip(upper=100)

    # Source encoding
    feat["source_otto"] = (df["source"] == "otto").astype(int)

    feat = feat.fillna(0)

    logger.info(
        f"Feature matrix: {feat.shape} | "
        f"conversion rate: {feat['label'].mean():.3f} | "
        f"{feat.shape[1] - 3} features"
    )
    return feat


def build_feature_matrix(save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load sequences.parquet → compute features → save features.parquet

    Requires sequences.parquet to already exist.
    Run sequences.py first if it does not.
    """
    seq_path = cfg.paths.sequences_parquet
    if not seq_path.exists():
        raise FileNotFoundError(
            f"sequences.parquet not found at {seq_path}\n"
            "Run first: python -m sessionscout.features.sequences"
        )

    logger.info(f"Loading sequences from {seq_path}...")
    df = pd.read_parquet(seq_path)
    logger.info(f"  {len(df):,} sessions loaded")

    features = build_session_features(df)

    out = save_path or cfg.paths.features_parquet
    cfg.paths.data_processed.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)
    logger.info(f"Features saved → {out}")

    return features


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        stream=sys.stdout,
    )
    df = build_feature_matrix()
    print(f"\nFeature matrix: {df.shape}")
    print(df.describe().round(3))
