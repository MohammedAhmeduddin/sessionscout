"""
tests/test_explainability.py — Coverage for shap_deep.py + attention_viz.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sessionscout.config import cfg

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_feature_df(n=100):
    cols = [
        "seq_len",
        "n_views",
        "n_carts",
        "cart_rate",
        "view_depth",
        "has_cart",
        "n_gap_short",
        "n_gap_long",
        "gap_ratio",
        "last_is_cart",
        "last_is_view",
        "view_cart_ratio",
        "source_otto",
    ]
    X = np.random.rand(n, len(cols)).astype(np.float32)
    df = pd.DataFrame(X, columns=cols)
    df["label"] = (np.random.rand(n) > 0.9).astype(float)
    df["session_id"] = [f"s{i}" for i in range(n)]
    df["source"] = "test"
    return df


def make_seq_df(n=30):
    seqs = [[0] * 58 + [1, 2, 4, 1, 5, 1] for _ in range(n)]
    return pd.DataFrame(
        {
            "session_id": [f"s{i}" for i in range(n)],
            "source": ["test"] * n,
            "sequence": seqs,
            "seq_len": [6] * n,
            "label": [1 if i % 5 == 0 else 0 for i in range(n)],
            "n_views": [3] * n,
            "n_carts": [1] * n,
        }
    )


def make_mock_xgb_model(n_features=13):
    """Mock XGBoost model to avoid MPS segfault on Apple Silicon."""

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.column_stack(
        [np.random.rand(50), np.random.rand(50)]
    )
    mock_model.feature_importances_ = np.random.rand(n_features)
    return mock_model


# ── shap_deep.py ──────────────────────────────────────────────────────────────


class TestShapDeep:
    def test_print_top_features_returns_ranked(self):
        from sessionscout.explainability.shap_deep import print_top_features

        shap_values = np.random.randn(50, 5)
        feature_names = ["a", "b", "c", "d", "e"]
        ranked = print_top_features(shap_values, feature_names, top_k=3)
        assert len(ranked) == len(feature_names)
        vals = [v for _, v in ranked]
        assert vals == sorted(vals, reverse=True)

    def test_plot_shap_summary_creates_file(self, tmp_path):
        from sessionscout.explainability.shap_deep import plot_shap_summary

        shap_values = np.random.randn(30, 5)
        X_sample = np.random.rand(30, 5).astype(np.float32)
        feature_names = ["a", "b", "c", "d", "e"]
        save_path = tmp_path / "shap_summary.png"

        result = plot_shap_summary(shap_values, X_sample, feature_names, save_path)
        assert result == save_path
        assert save_path.exists()

    def test_run_shap_missing_features(self, tmp_path, monkeypatch):
        from sessionscout.explainability.shap_deep import run_shap_analysis

        monkeypatch.setattr(cfg.paths, "features_parquet", tmp_path / "none.parquet")
        with pytest.raises(FileNotFoundError):
            run_shap_analysis()

    def test_compute_shap_values_shape(self, tmp_path, monkeypatch):
        """Test SHAP computation with a fully mocked XGBoost + SHAP."""
        from sessionscout.explainability.shap_deep import compute_shap_values

        # Mock the TreeExplainer to avoid real XGBoost init
        n_samples, n_features = 30, 5
        feature_names = ["a", "b", "c", "d", "e"]
        X_test = np.random.rand(n_samples, n_features).astype(np.float32)
        fake_shap = np.random.randn(n_samples, n_features)

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = fake_shap

        mock_shap_module = MagicMock()
        mock_shap_module.TreeExplainer.return_value = mock_explainer

        mock_model = MagicMock()

        with patch.dict("sys.modules", {"shap": mock_shap_module}):

            shap_values, explainer, X_sample = compute_shap_values(
                mock_model, X_test, feature_names, max_samples=30
            )

        assert shap_values.shape == (n_samples, n_features)
        assert len(X_sample) <= 30

    def test_run_shap_analysis_mocked(self, tmp_path, monkeypatch):
        from sessionscout.explainability.shap_deep import run_shap_analysis

        feat_path = tmp_path / "features.parquet"
        make_feature_df(200).to_parquet(feat_path, index=False)
        monkeypatch.setattr(cfg.paths, "features_parquet", feat_path)
        monkeypatch.setattr(cfg.paths, "models_dir", tmp_path)

        # Mock XGBoost entirely
        n_features = 13
        fake_shap_vals = np.random.randn(40, n_features)

        mock_xgb_model = MagicMock()
        mock_xgb_clf = MagicMock(return_value=mock_xgb_model)
        mock_xgb = MagicMock()
        mock_xgb.XGBClassifier = mock_xgb_clf

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = fake_shap_vals
        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer

        with patch.dict("sys.modules", {"xgboost": mock_xgb, "shap": mock_shap}):
            shap_values, feature_cols, ranked = run_shap_analysis()

        assert len(ranked) > 0


# ── attention_viz.py ──────────────────────────────────────────────────────────


class TestAttentionViz:
    def test_tokens_to_names(self):
        from sessionscout.explainability.attention_viz import tokens_to_names

        seq = [0, 1, 2, 3, 4, 5]
        names = tokens_to_names(seq)
        assert names == ["PAD", "VIEW", "ADD_CART", "PURCHASE", "GAP_SHORT", "GAP_LONG"]

    def test_tokens_to_names_unknown(self):
        from sessionscout.explainability.attention_viz import tokens_to_names

        names = tokens_to_names([99])
        assert names[0].startswith("tok_")

    def test_plot_attention_heatmap_creates_file(self, tmp_path):
        from sessionscout.explainability.attention_viz import plot_attention_heatmap

        # 4 heads, 10 positions
        attn = torch.rand(4, 10, 10)
        seq = [0] * 4 + [1, 2, 4, 1, 5, 1]
        out = tmp_path / "attn.png"

        result = plot_attention_heatmap(attn, seq, "test_session", 0, out)
        assert result == out
        assert out.exists()

    def test_plot_attention_heatmap_all_pad(self, tmp_path):
        from sessionscout.explainability.attention_viz import plot_attention_heatmap

        attn = torch.rand(4, 5, 5)
        seq = [0] * 5  # all PAD
        result = plot_attention_heatmap(attn, seq, "all_pad", 0, tmp_path / "out.png")
        assert result is None  # nothing to plot

    def test_analyse_session(self, tmp_path):
        from sessionscout.explainability.attention_viz import analyse_session
        from sessionscout.model.transformer import SessionTransformer

        model = SessionTransformer()
        sequence = [0] * 58 + [1, 1, 2, 4, 1, 1]

        attn = analyse_session(model, sequence, "test_001", save_dir=tmp_path)
        assert attn.shape[0] == cfg.model.num_heads
        # Should save 4 heatmap files
        png_files = list(tmp_path.glob("attention_head*.png"))
        assert len(png_files) == 4

    def test_load_best_transformer(self, tmp_path, monkeypatch):
        from sessionscout.explainability.attention_viz import load_best_transformer
        from sessionscout.model.transformer import SessionTransformer

        model_path = tmp_path / "transformer_best.pt"
        model = SessionTransformer()
        torch.save(model.state_dict(), model_path)
        monkeypatch.setattr(cfg.paths, "models_dir", tmp_path)

        loaded = load_best_transformer(model_path)
        assert loaded is not None

    def test_load_best_transformer_missing(self, tmp_path):
        from sessionscout.explainability.attention_viz import load_best_transformer

        with pytest.raises(FileNotFoundError):
            load_best_transformer(tmp_path / "missing.pt")

    def test_run_attention_analysis(self, tmp_path, monkeypatch):
        from sessionscout.explainability.attention_viz import run_attention_analysis
        from sessionscout.model.transformer import SessionTransformer

        model_path = tmp_path / "transformer_best.pt"
        model = SessionTransformer()
        torch.save(model.state_dict(), model_path)

        seq_path = tmp_path / "sequences.parquet"
        make_seq_df(20).to_parquet(seq_path, index=False)

        monkeypatch.setattr(cfg.paths, "models_dir", tmp_path)
        monkeypatch.setattr(cfg.paths, "sequences_parquet", seq_path)

        run_attention_analysis()
        png_files = list(tmp_path.glob("attention_head*.png"))
        assert len(png_files) >= 1
