"""
config.py — Single source of truth for all SessionScout settings.

Every path, number, and threshold lives here.
Every other file imports: from sessionscout.config import cfg
Never hardcode a value anywhere else.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Project root — the sessionscout/ folder
ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class PathConfig:
    root: Path = ROOT_DIR
    data_raw: Path          = ROOT_DIR / "data" / "raw"
    retailrocket_dir: Path  = ROOT_DIR / "data" / "raw" / "retailrocket"
    otto_dir: Path          = ROOT_DIR / "data" / "raw" / "otto"
    rr_events: Path  = ROOT_DIR / "data" / "raw" / "retailrocket" / "events.csv"
    otto_train: Path = ROOT_DIR / "data" / "raw" / "otto" / "train.jsonl"
    data_processed: Path    = ROOT_DIR / "data" / "processed"
    sequences_parquet: Path = ROOT_DIR / "data" / "processed" / "sequences.parquet"
    features_parquet: Path  = ROOT_DIR / "data" / "processed" / "features.parquet"
    vocab_json: Path        = ROOT_DIR / "data" / "processed" / "vocab.json"
    models_dir: Path        = ROOT_DIR / "models"
    mlflow_dir: Path        = ROOT_DIR / "mlruns"


@dataclass
class VocabConfig:
    """
    5 tokens — every single one is defensible in an interview.

    PAD      = empty slot (left-padding for right-aligned sequences)
    VIEW     = user viewed a product page
    ADD_CART = user added item to cart
    PURCHASE = user completed a purchase (our positive label)
    GAP_SHORT= inactivity 2-10 min (hesitation signal)
    GAP_LONG = inactivity 10+ min (likely abandonment)
    """
    pad: int       = 0
    view: int      = 1
    add_cart: int  = 2
    purchase: int  = 3
    gap_short: int = 4
    gap_long: int  = 5
    size: int      = 6


@dataclass
class SequenceConfig:
    max_len: int       = 64    # max events per session
    min_len: int       = 3     # drop sessions shorter than this
    gap_short_min: int = 120   # 2 minutes in seconds
    gap_long_min: int  = 600   # 10 minutes in seconds
    val_size: float    = 0.10
    test_size: float   = 0.10
    random_seed: int   = 42


@dataclass
class ModelConfig:
    vocab_size: int         = 6
    embed_dim: int          = 64
    num_heads: int          = 4
    num_encoder_layers: int = 2
    ff_dim: int             = 128
    dropout: float          = 0.1
    max_seq_len: int        = 64
    lstm_hidden: int        = 64
    lstm_layers: int        = 2
    lstm_dropout: float     = 0.1
    output_dim: int         = 1


@dataclass
class TrainingConfig:
    batch_size: int    = 512
    learning_rate: float = 1e-3
    weight_decay: float  = 1e-4
    epochs: int        = 20
    early_stopping_patience: int = 4
    grad_clip: float   = 1.0
    random_seed: int   = 42
    # ~5% conversion rate means 95% negative, 5% positive
    # pos_weight = (1 - 0.05) / 0.05 = 19
    # This tells the loss function to penalise missing a purchase 19x more
    pos_weight: float  = 19.0


@dataclass
class BusinessConfig:
    # Used in sensitivity_analysis.py (drawback D3 fix)
    aov_range: list   = field(default_factory=lambda: [45, 65, 85, 120])
    uplift_range: list = field(default_factory=lambda: [0.05, 0.10, 0.15])
    intervention_cost: float = 2.50
    top_k_sessions: int      = 500


@dataclass
class MLflowConfig:
    tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        str(ROOT_DIR / "mlruns")
    )
    experiment_name: str       = "sessionscout-conversion"
    registered_model_name: str = "sessionscout-transformer"


@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    model_path: str = os.getenv(
        "MODEL_PATH",
        str(ROOT_DIR / "models" / "production")
    )
    redis_url: str         = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl_seconds: int = 300


@dataclass
class Config:
    paths:    PathConfig    = field(default_factory=PathConfig)
    vocab:    VocabConfig   = field(default_factory=VocabConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    model:    ModelConfig   = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)
    mlflow:   MLflowConfig  = field(default_factory=MLflowConfig)
    api:      APIConfig     = field(default_factory=APIConfig)


cfg = Config()

# Token lookup tables — used by sequences.py and interpretability layer
VOCAB = {
    cfg.vocab.pad:       "PAD",
    cfg.vocab.view:      "VIEW",
    cfg.vocab.add_cart:  "ADD_CART",
    cfg.vocab.purchase:  "PURCHASE",
    cfg.vocab.gap_short: "GAP_SHORT",
    cfg.vocab.gap_long:  "GAP_LONG",
}
EVENT_TO_TOKEN = {v: k for k, v in VOCAB.items()}
