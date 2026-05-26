"""
tests/test_features.py — Coverage for features/sequences.py + features/engineering.py

Uses real tmp files so no mocking of pandas needed.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sessionscout.config import cfg


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_rr_csv(path):
    """Write a minimal Retail Rocket events.csv to path."""
    df = pd.DataFrame({
        "timestamp":     [1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000],
        "visitorid":     [1, 1, 1, 2, 2, 2],
        "event":         ["view", "view", "addtocart", "view", "view", "transaction"],
        "itemid":        [101] * 6,
    })
    df.to_csv(path, index=False)


def make_otto_jsonl(path, n_sessions=3):
    """Write minimal OTTO train.jsonl to path."""
    sessions = []
    for i in range(n_sessions):
        sessions.append({
            "session": i,
            "events": [
                {"aid": 1, "ts": float(1_000_000 + i * 1000),     "type": "clicks"},
                {"aid": 1, "ts": float(1_000_000 + i * 1000 + 60), "type": "clicks"},
                {"aid": 1, "ts": float(1_000_000 + i * 1000 + 120),"type": "carts"},
                {"aid": 1, "ts": float(1_000_000 + i * 1000 + 180),"type": "orders"},
            ],
        })
    with open(path, "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")


def make_seq_df(n=20):
    """Create synthetic sequences DataFrame."""
    seqs = [[0] * 60 + [1, 1, 2, 1] for _ in range(n)]
    return pd.DataFrame({
        "session_id": [f"s{i}" for i in range(n)],
        "source":     ["retailrocket"] * (n // 2) + ["otto"] * (n - n // 2),
        "sequence":   seqs,
        "seq_len":    [4] * n,
        "label":      [1 if i % 5 == 0 else 0 for i in range(n)],
        "n_views":    [3] * n,
        "n_carts":    [1] * n,
    })


# ── load_retailrocket ─────────────────────────────────────────────────────────

class TestLoadRetailRocket:
    def test_success(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import load_retailrocket
        csv_path = tmp_path / "events.csv"
        make_rr_csv(csv_path)
        monkeypatch.setattr(cfg.paths, "rr_events", csv_path)

        df = load_retailrocket()
        assert len(df) == 6
        assert "purchase" in df["event_type"].values
        assert all(df["session_id"].str.startswith("rr_"))
        assert "timestamp_sec" in df.columns
        assert "source" in df.columns

    def test_file_not_found(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import load_retailrocket
        monkeypatch.setattr(cfg.paths, "rr_events", tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError):
            load_retailrocket()

    def test_event_normalisation(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import load_retailrocket
        csv_path = tmp_path / "events.csv"
        make_rr_csv(csv_path)
        monkeypatch.setattr(cfg.paths, "rr_events", csv_path)

        df = load_retailrocket()
        assert "transaction" not in df["event_type"].values  # renamed to purchase
        assert "purchase" in df["event_type"].values


# ── load_otto ─────────────────────────────────────────────────────────────────

class TestLoadOtto:
    def test_success(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import load_otto
        jsonl_path = tmp_path / "train.jsonl"
        make_otto_jsonl(jsonl_path, n_sessions=3)
        monkeypatch.setattr(cfg.paths, "otto_train", jsonl_path)

        df = load_otto()
        assert len(df) == 12       # 4 events × 3 sessions
        assert "purchase" in df["event_type"].values
        assert "view"     in df["event_type"].values
        assert all(df["session_id"].str.startswith("otto_"))

    def test_max_sessions_limits_output(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import load_otto
        jsonl_path = tmp_path / "train.jsonl"
        make_otto_jsonl(jsonl_path, n_sessions=5)
        monkeypatch.setattr(cfg.paths, "otto_train", jsonl_path)

        df = load_otto(max_sessions=2)
        assert df["session_id"].nunique() == 2

    def test_file_not_found(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import load_otto
        monkeypatch.setattr(cfg.paths, "otto_train", tmp_path / "missing.jsonl")
        with pytest.raises(FileNotFoundError):
            load_otto()

    def test_event_type_mapping(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import load_otto
        jsonl_path = tmp_path / "train.jsonl"
        make_otto_jsonl(jsonl_path)
        monkeypatch.setattr(cfg.paths, "otto_train", jsonl_path)

        df = load_otto()
        # OTTO "orders" should become "purchase"
        assert "orders" not in df["event_type"].values
        # OTTO "clicks" should become "view"
        assert "clicks" not in df["event_type"].values


# ── build_sequence_dataset ────────────────────────────────────────────────────

class TestBuildSequenceDataset:
    def test_with_retailrocket_only(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import build_sequence_dataset

        csv_path = tmp_path / "events.csv"
        make_rr_csv(csv_path)

        processed = tmp_path / "processed"
        processed.mkdir()

        monkeypatch.setattr(cfg.paths, "rr_events",          csv_path)
        monkeypatch.setattr(cfg.paths, "otto_train",          tmp_path / "missing.jsonl")
        monkeypatch.setattr(cfg.paths, "data_processed",      processed)
        monkeypatch.setattr(cfg.paths, "sequences_parquet",   processed / "sequences.parquet")
        monkeypatch.setattr(cfg.paths, "vocab_json",          processed / "vocab.json")

        df = build_sequence_dataset()
        assert len(df) > 0
        assert "sequence" in df.columns
        assert "label"    in df.columns
        assert (processed / "vocab.json").exists()
        assert (processed / "sequences.parquet").exists()

    def test_no_data_raises(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import build_sequence_dataset
        monkeypatch.setattr(cfg.paths, "rr_events",  tmp_path / "a.csv")
        monkeypatch.setattr(cfg.paths, "otto_train", tmp_path / "b.jsonl")
        with pytest.raises(RuntimeError):
            build_sequence_dataset()

    def test_custom_save_path(self, tmp_path, monkeypatch):
        from sessionscout.features.sequences import build_sequence_dataset

        csv_path = tmp_path / "events.csv"
        make_rr_csv(csv_path)
        out = tmp_path / "custom.parquet"

        processed = tmp_path / "processed"
        processed.mkdir()

        monkeypatch.setattr(cfg.paths, "rr_events",     csv_path)
        monkeypatch.setattr(cfg.paths, "otto_train",    tmp_path / "missing.jsonl")
        monkeypatch.setattr(cfg.paths, "data_processed", processed)
        monkeypatch.setattr(cfg.paths, "vocab_json",    processed / "vocab.json")

        df = build_sequence_dataset(save_path=out)
        assert out.exists()


# ── build_session_features ────────────────────────────────────────────────────

class TestBuildSessionFeatures:
    def test_output_columns(self):
        from sessionscout.features.engineering import build_session_features
        df     = make_seq_df(20)
        result = build_session_features(df)

        for col in ["cart_rate", "view_depth", "has_cart",
                    "n_gap_short", "n_gap_long", "gap_ratio",
                    "last_is_cart", "last_is_view",
                    "view_cart_ratio", "source_otto"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_row_count(self):
        from sessionscout.features.engineering import build_session_features
        df     = make_seq_df(30)
        result = build_session_features(df)
        assert len(result) == 30

    def test_value_ranges(self):
        from sessionscout.features.engineering import build_session_features
        df     = make_seq_df(20)
        result = build_session_features(df)

        assert (result["has_cart"].isin([0, 1])).all()
        assert (result["cart_rate"] >= 0).all()
        assert (result["view_cart_ratio"] <= 100).all()
        assert (result["source_otto"].isin([0, 1])).all()

    def test_no_nulls(self):
        from sessionscout.features.engineering import build_session_features
        result = build_session_features(make_seq_df(15))
        assert result.isnull().sum().sum() == 0


# ── build_feature_matrix ──────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_success(self, tmp_path, monkeypatch):
        from sessionscout.features.engineering import build_feature_matrix

        seq_path  = tmp_path / "sequences.parquet"
        feat_path = tmp_path / "features.parquet"
        make_seq_df(50).to_parquet(seq_path, index=False)

        monkeypatch.setattr(cfg.paths, "sequences_parquet", seq_path)
        monkeypatch.setattr(cfg.paths, "features_parquet",  feat_path)
        monkeypatch.setattr(cfg.paths, "data_processed",    tmp_path)

        result = build_feature_matrix()
        assert len(result) == 50
        assert feat_path.exists()

    def test_missing_sequences_raises(self, tmp_path, monkeypatch):
        from sessionscout.features.engineering import build_feature_matrix
        monkeypatch.setattr(cfg.paths, "sequences_parquet", tmp_path / "none.parquet")
        with pytest.raises(FileNotFoundError):
            build_feature_matrix()

    def test_custom_save_path(self, tmp_path, monkeypatch):
        from sessionscout.features.engineering import build_feature_matrix

        seq_path  = tmp_path / "sequences.parquet"
        out       = tmp_path / "custom_feat.parquet"
        make_seq_df(20).to_parquet(seq_path, index=False)

        monkeypatch.setattr(cfg.paths, "sequences_parquet", seq_path)
        monkeypatch.setattr(cfg.paths, "data_processed",    tmp_path)

        build_feature_matrix(save_path=out)
        assert out.exists()
