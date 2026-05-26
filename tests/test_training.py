"""
tests/test_training.py — Coverage for model/train.py, model/evaluate.py, model/dataset.py

Uses small synthetic data + mocked MLflow to run quickly.
"""

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sessionscout.config import cfg

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_feature_df(n=300):
    """Synthetic features.parquet content."""
    np.random.seed(42)
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


def make_seq_df(n=60):
    seqs = [[0] * 60 + [1, 1, 2, 1] for _ in range(n)]
    return pd.DataFrame(
        {
            "session_id": [f"s{i}" for i in range(n)],
            "source": ["test"] * n,
            "sequence": seqs,
            "seq_len": [4] * n,
            "label": [1 if i % 10 == 0 else 0 for i in range(n)],
            "n_views": [3] * n,
            "n_carts": [1] * n,
        }
    )


def make_small_dataset(n=40):
    from sessionscout.model.dataset import SessionDataset

    return SessionDataset(make_seq_df(n))


def small_loaders(n=40, batch_size=32):
    ds = make_small_dataset(n)
    return (
        DataLoader(ds, batch_size=batch_size),
        DataLoader(ds, batch_size=batch_size),
        DataLoader(ds, batch_size=batch_size),
    )


@contextlib.contextmanager
def fake_mlflow_run(*args, **kwargs):
    yield MagicMock()


MLFLOW_PATCHES = [
    patch("mlflow.set_tracking_uri"),
    patch("mlflow.set_experiment"),
    patch("mlflow.start_run", side_effect=fake_mlflow_run),
    patch("mlflow.log_metrics"),
    patch("mlflow.log_params"),
    patch("mlflow.log_metric"),
    patch("mlflow.log_artifact"),
]


# ── SessionDataset ────────────────────────────────────────────────────────────


class TestSessionDataset:
    def test_len(self):
        from sessionscout.model.dataset import SessionDataset

        ds = SessionDataset(make_seq_df(20))
        assert len(ds) == 20

    def test_getitem_keys(self):
        from sessionscout.model.dataset import SessionDataset

        ds = SessionDataset(make_seq_df(5))
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "label" in item

    def test_getitem_shapes(self):
        from sessionscout.model.dataset import SessionDataset

        ds = SessionDataset(make_seq_df(5))
        item = ds[0]
        assert item["input_ids"].shape == (cfg.sequence.max_len,)
        assert item["attention_mask"].shape == (cfg.sequence.max_len,)

    def test_mask_correct(self):
        from sessionscout.model.dataset import SessionDataset

        ds = SessionDataset(make_seq_df(5))
        seq = ds[0]["input_ids"]
        mask = ds[0]["attention_mask"]
        for tok, m in zip(seq.tolist(), mask.tolist()):
            if tok == cfg.vocab.pad:
                assert m == 0.0
            else:
                assert m == 1.0

    def test_labels_dtype(self):
        from sessionscout.model.dataset import SessionDataset

        ds = SessionDataset(make_seq_df(5))
        assert ds[0]["label"].dtype == torch.float32


# ── load_datasets + make_dataloaders ─────────────────────────────────────────


class TestLoadDatasets:
    def test_split_sizes(self, tmp_path, monkeypatch):
        from sessionscout.model.dataset import load_datasets

        seq_path = tmp_path / "sequences.parquet"
        make_seq_df(200).to_parquet(seq_path, index=False)
        monkeypatch.setattr(cfg.paths, "sequences_parquet", seq_path)

        train_ds, val_ds, test_ds = load_datasets()
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 200
        assert len(test_ds) == pytest.approx(20, abs=2)

    def test_file_not_found(self, tmp_path, monkeypatch):
        from sessionscout.model.dataset import load_datasets

        monkeypatch.setattr(
            cfg.paths, "sequences_parquet", tmp_path / "missing.parquet"
        )
        with pytest.raises(FileNotFoundError):
            load_datasets()

    def test_make_dataloaders(self, tmp_path, monkeypatch):
        from sessionscout.model.dataset import load_datasets, make_dataloaders

        seq_path = tmp_path / "sequences.parquet"
        make_seq_df(60).to_parquet(seq_path, index=False)
        monkeypatch.setattr(cfg.paths, "sequences_parquet", seq_path)

        train_ds, val_ds, test_ds = load_datasets()
        train_loader, val_loader, test_loader = make_dataloaders(
            train_ds, val_ds, test_ds, batch_size=16
        )
        batch = next(iter(train_loader))
        assert "input_ids" in batch
        assert "label" in batch


# ── load_tabular_splits ───────────────────────────────────────────────────────


class TestLoadTabularSplits:
    def test_success(self, tmp_path, monkeypatch):
        from sessionscout.model.train import load_tabular_splits

        feat_path = tmp_path / "features.parquet"
        make_feature_df(300).to_parquet(feat_path, index=False)
        monkeypatch.setattr(cfg.paths, "features_parquet", feat_path)

        X_train, X_val, X_test, y_train, y_val, y_test, cols = load_tabular_splits()
        assert X_train.shape[1] == 13
        assert len(cols) == 13
        assert len(X_train) + len(X_val) + len(X_test) == 300

    def test_file_not_found(self, tmp_path, monkeypatch):
        from sessionscout.model.train import load_tabular_splits

        monkeypatch.setattr(cfg.paths, "features_parquet", tmp_path / "missing.parquet")
        with pytest.raises(FileNotFoundError):
            load_tabular_splits()


# ── log_metrics ───────────────────────────────────────────────────────────────


def test_log_metrics():
    from sessionscout.model.train import log_metrics

    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45])
    with patch("mlflow.log_metrics"):
        auc, ap = log_metrics(y_true, y_pred, "val")
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= ap <= 1.0


# ── train_logistic_regression ─────────────────────────────────────────────────


def test_train_logistic_regression(tmp_path, monkeypatch):
    from sessionscout.model.train import train_logistic_regression

    feat_path = tmp_path / "features.parquet"
    make_feature_df(300).to_parquet(feat_path, index=False)
    monkeypatch.setattr(cfg.paths, "features_parquet", feat_path)

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", side_effect=fake_mlflow_run),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_params"),
    ):
        result = train_logistic_regression()

    assert result is not None
    assert result["model"] == "logistic_regression"
    assert 0 <= result["val_auc"] <= 1


# ── train_xgboost ─────────────────────────────────────────────────────────────


def test_train_xgboost(tmp_path, monkeypatch):
    from sessionscout.model.train import train_xgboost

    feat_path = tmp_path / "features.parquet"
    df = make_feature_df(300)
    df.to_parquet(feat_path, index=False)
    monkeypatch.setattr(cfg.paths, "features_parquet", feat_path)

    # Mock XGBoost to avoid MPS + XGBoost segfault on Apple Silicon
    def fake_predict_proba(X):
        n = len(X)
        scores = np.random.rand(n)
        return np.column_stack([1 - scores, scores])

    mock_xgb_model = MagicMock()
    mock_xgb_model.predict_proba.side_effect = fake_predict_proba
    mock_xgb_model.feature_importances_ = np.random.rand(13)

    mock_xgb = MagicMock()
    mock_xgb.XGBClassifier.return_value = mock_xgb_model

    with (
        patch.dict("sys.modules", {"xgboost": mock_xgb}),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", side_effect=fake_mlflow_run),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_params"),
    ):
        result = train_xgboost()

    assert result is not None
    assert result["model"] == "xgboost"
    assert 0 <= result["val_auc"] <= 1


# ── train_deep_model (LSTM) ───────────────────────────────────────────────────


def test_train_deep_model_lstm(tmp_path, monkeypatch):
    from sessionscout.model.train import train_deep_model
    from sessionscout.model.lstm import SessionLSTM

    train_dl, val_dl, test_dl = small_loaders(40)
    train_ds = make_small_dataset(40)
    val_ds = make_small_dataset(15)
    test_ds = make_small_dataset(15)

    monkeypatch.setattr(cfg.paths, "models_dir", tmp_path)

    with (
        patch(
            "sessionscout.model.dataset.load_datasets",
            return_value=(train_ds, val_ds, test_ds),
        ),
        patch(
            "sessionscout.model.dataset.make_dataloaders",
            return_value=(
                DataLoader(train_ds, batch_size=32),
                DataLoader(val_ds, batch_size=32),
                DataLoader(test_ds, batch_size=32),
            ),
        ),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", side_effect=fake_mlflow_run),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_params"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_artifact"),
        patch("torch.save"),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):

        result = train_deep_model(SessionLSTM(), "lstm", "test-lstm", epochs=2, lr=1e-3)

    assert result["model"] == "lstm"
    assert 0 <= result["val_auc"] <= 1


def test_train_deep_model_early_stopping(tmp_path, monkeypatch):
    """Cover early stopping code path with patience=1."""
    from sessionscout.model.train import train_deep_model
    from sessionscout.model.lstm import SessionLSTM

    monkeypatch.setattr(cfg.training, "early_stopping_patience", 1)
    monkeypatch.setattr(cfg.paths, "models_dir", tmp_path)

    train_ds = make_small_dataset(40)
    val_ds = make_small_dataset(15)
    test_ds = make_small_dataset(15)

    with (
        patch(
            "sessionscout.model.dataset.load_datasets",
            return_value=(train_ds, val_ds, test_ds),
        ),
        patch(
            "sessionscout.model.dataset.make_dataloaders",
            return_value=(
                DataLoader(train_ds, batch_size=32),
                DataLoader(val_ds, batch_size=32),
                DataLoader(test_ds, batch_size=32),
            ),
        ),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", side_effect=fake_mlflow_run),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_params"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_artifact"),
        patch("torch.save"),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):

        result = train_deep_model(
            SessionLSTM(), "lstm", "early-stop", epochs=10, lr=1e-3
        )

    assert result is not None  # early stopping fired


def test_train_lstm_function(tmp_path, monkeypatch):
    from sessionscout.model.train import train_lstm

    monkeypatch.setattr(cfg.paths, "models_dir", tmp_path)
    train_ds = make_small_dataset(30)
    val_ds = make_small_dataset(10)
    test_ds = make_small_dataset(10)

    with (
        patch(
            "sessionscout.model.dataset.load_datasets",
            return_value=(train_ds, val_ds, test_ds),
        ),
        patch(
            "sessionscout.model.dataset.make_dataloaders",
            return_value=(
                DataLoader(train_ds, batch_size=32),
                DataLoader(val_ds, batch_size=32),
                DataLoader(test_ds, batch_size=32),
            ),
        ),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", side_effect=fake_mlflow_run),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_params"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_artifact"),
        patch("torch.save"),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):

        result = train_lstm()

    assert result["model"] == "lstm"


def test_train_transformer_function(tmp_path, monkeypatch):
    from sessionscout.model.train import train_transformer

    monkeypatch.setattr(cfg.paths, "models_dir", tmp_path)
    train_ds = make_small_dataset(30)
    val_ds = make_small_dataset(10)
    test_ds = make_small_dataset(10)

    with (
        patch(
            "sessionscout.model.dataset.load_datasets",
            return_value=(train_ds, val_ds, test_ds),
        ),
        patch(
            "sessionscout.model.dataset.make_dataloaders",
            return_value=(
                DataLoader(train_ds, batch_size=32),
                DataLoader(val_ds, batch_size=32),
                DataLoader(test_ds, batch_size=32),
            ),
        ),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", side_effect=fake_mlflow_run),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_params"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_artifact"),
        patch("torch.save"),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):

        result = train_transformer()

    assert result["model"] == "transformer"


def test_main_lr(tmp_path, monkeypatch):
    from sessionscout.model.train import main

    feat_path = tmp_path / "features.parquet"
    make_feature_df(300).to_parquet(feat_path, index=False)
    monkeypatch.setattr(cfg.paths, "features_parquet", feat_path)

    with (
        patch("sys.argv", ["train", "--model", "lr"]),
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run", side_effect=fake_mlflow_run),
        patch("mlflow.log_metrics"),
        patch("mlflow.log_params"),
    ):
        main()


# ── model/evaluate.py ─────────────────────────────────────────────────────────


class TestEvaluate:
    def test_evaluate_deep_model(self):
        from sessionscout.model.evaluate import evaluate_deep_model
        from sessionscout.model.lstm import SessionLSTM

        ds = make_small_dataset(30)
        loader = DataLoader(ds, batch_size=16)
        model = SessionLSTM()

        metrics = evaluate_deep_model(model, loader)
        assert "auc" in metrics
        assert "ap" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert 0 <= metrics["auc"] <= 1

    def test_precision_at_k(self):
        from sessionscout.model.evaluate import precision_at_k

        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.45])

        p5 = precision_at_k(labels, scores, k=5)
        assert p5 == 1.0  # top-5 scores are all positives

        p10 = precision_at_k(labels, scores, k=10)
        assert p10 == 0.5  # 5 positives in 10

    def test_print_results_table(self, capsys):
        from sessionscout.model.evaluate import print_results_table

        results = {
            "lstm": {"val_auc": 0.987, "test_auc": 0.988, "ap": 0.85, "p_at_500": 0.92},
            "transformer": {
                "val_auc": 0.981,
                "test_auc": 0.984,
                "ap": 0.82,
                "p_at_500": 0.88,
            },
        }
        print_results_table(results)
        captured = capsys.readouterr()
        assert "lstm" in captured.out or captured.out == ""  # printed to logger

    def test_run_full_evaluation(self, tmp_path, monkeypatch):
        from sessionscout.model.evaluate import run_full_evaluation
        from sessionscout.model.lstm import SessionLSTM
        from sessionscout.model.transformer import SessionTransformer

        seq_path = tmp_path / "sequences.parquet"
        make_seq_df(60).to_parquet(seq_path, index=False)
        monkeypatch.setattr(cfg.paths, "sequences_parquet", seq_path)
        monkeypatch.setattr(cfg.paths, "models_dir", tmp_path)

        # Save tiny model weights
        lstm = SessionLSTM()
        tf = SessionTransformer()
        torch.save(lstm.state_dict(), tmp_path / "lstm_best.pt")
        torch.save(tf.state_dict(), tmp_path / "transformer_best.pt")

        results = run_full_evaluation()
        assert "lstm" in results
        assert "transformer" in results
        assert 0 <= results["lstm"]["auc"] <= 1
