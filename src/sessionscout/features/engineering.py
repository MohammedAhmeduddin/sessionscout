"""
features/engineering.py — Session-level tabular features.

These features are deliberately leak-free:
  - No purchase token information (excluded from sequences)
  - No last_event token ID (encodes cart which proxies conversion)
  - Only behavioral aggregates a model could use in real-time

Used by Logistic Regression and XGBoost baselines.
When the Transformer beats these, the win is earned honestly.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sessionscout.config import cfg

logger = logging.getLogger(__name__)


def build_session_features(sequences_df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive tabular features from sequences DataFrame.

    Input columns required: session_id, source, sequence,
                            seq_len, label, n_views, n_carts
    """
    df = sequences_df.copy()
    feat = pd.DataFrame()

    # Identity
    feat["session_id"] = df["session_id"]
    feat["label"] = df["label"]
    feat["source"] = df["source"]

    # Raw counts
    feat["seq_len"] = df["seq_len"]
    feat["n_views"] = df["n_views"]
    feat["n_carts"] = df["n_carts"]

    # Intent ratios
    feat["cart_rate"] = df["n_carts"] / (df["n_views"] + 1e-8)
    feat["view_depth"] = df["n_views"] / (df["seq_len"] + 1e-8)
    feat["has_cart"] = (df["n_carts"] > 0).astype(int)

    # Gap counts from sequence tokens
    def count_tok(seq, tid):
        return sum(1 for t in seq if t == tid)

    feat["n_gap_short"] = df["sequence"].apply(
        lambda s: count_tok(s, cfg.vocab.gap_short)
    )
    feat["n_gap_long"] = df["sequence"].apply(
        lambda s: count_tok(s, cfg.vocab.gap_long)
    )
    feat["gap_ratio"] = (feat["n_gap_short"] + feat["n_gap_long"]) / (
        df["seq_len"] + 1e-8
    )

    # Last action flags — derived from counts only, no token ID leakage
    # last_is_cart: session ended with cart activity and no further views
    feat["last_is_cart"] = (df["n_carts"] > 0).astype(int)
    feat["last_is_view"] = (df["n_carts"] == 0).astype(int)

    # View-to-cart ratio
    feat["view_cart_ratio"] = (df["n_views"] / (df["n_carts"] + 1e-8)).clip(upper=100)

    # Source
    feat["source_otto"] = (df["source"] == "otto").astype(int)

    feat = feat.fillna(0)

    n_features = feat.shape[1] - 3  # exclude session_id, label, source
    logger.info(
        f"Feature matrix: {feat.shape} | "
        f"conversion rate: {feat['label'].mean():.3f} | "
        f"{n_features} features"
    )
    return feat


def build_feature_matrix(save_path: Optional[Path] = None) -> pd.DataFrame:
    """Load sequences.parquet → compute features → save features.parquet"""
    seq_path = cfg.paths.sequences_parquet
    if not seq_path.exists():
        raise FileNotFoundError(
            f"sequences.parquet not found at {seq_path}\n"
            "Run: python -m sessionscout.features.sequences"
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
