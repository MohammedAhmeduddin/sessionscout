"""
model/train.py — Training all 4 models with MLflow tracking.

Usage:
  python -m sessionscout.model.train --model lr
  python -m sessionscout.model.train --model xgb
  python -m sessionscout.model.train --model lstm
  python -m sessionscout.model.train --model transformer
  python -m sessionscout.model.train --model all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sessionscout.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Shared utilities ──────────────────────────────────────────────────────────

def load_tabular_splits():
    path = cfg.paths.features_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {path}\n"
            "Run: python -m sessionscout.features.engineering"
        )
    df = pd.read_parquet(path)
    feature_cols = [
        c for c in df.columns
        if c not in ["session_id", "label", "source"]
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y,
        test_size=cfg.sequence.test_size,
        stratify=y,
        random_state=cfg.sequence.random_seed,
    )
    adjusted_val = cfg.sequence.val_size / (1 - cfg.sequence.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=adjusted_val,
        stratify=y_tv,
        random_state=cfg.sequence.random_seed,
    )
    logger.info(
        f"Tabular splits: train={len(X_train):,} "
        f"val={len(X_val):,} test={len(X_test):,}"
    )
    logger.info(
        f"Conversion rates — train: {y_train.mean():.3f} "
        f"val: {y_val.mean():.3f} test: {y_test.mean():.3f}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def log_metrics(y_true, y_pred_proba, prefix):
    auc = roc_auc_score(y_true, y_pred_proba)
    ap  = average_precision_score(y_true, y_pred_proba)
    mlflow.log_metrics({
        f"{prefix}_auc": round(auc, 4),
        f"{prefix}_ap":  round(ap, 4),
    })
    return auc, ap


# ── Model 1: Logistic Regression ─────────────────────────────────────────────

def train_logistic_regression():
    """Brain 1 — establishes the AUC floor."""
    logger.info("=" * 55)
    logger.info("Training Brain 1: Logistic Regression")
    logger.info("=" * 55)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
        load_tabular_splits()
    )

    with mlflow.start_run(run_name="logistic-regression"):
        t0 = time.time()
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=cfg.sequence.random_seed,
            C=1.0,
        )
        model.fit(X_train_s, y_train)
        elapsed = time.time() - t0

        val_proba  = model.predict_proba(X_val_s)[:, 1]
        test_proba = model.predict_proba(X_test_s)[:, 1]

        val_auc,  val_ap  = log_metrics(y_val,  val_proba,  "val")
        test_auc, test_ap = log_metrics(y_test, test_proba, "test")

        mlflow.log_params({
            "model": "logistic_regression", "C": 1.0,
            "class_weight": "balanced",
            "n_features": len(feature_cols),
            "training_secs": round(elapsed, 1),
        })

        coef_importance = sorted(
            zip(feature_cols, abs(model.coef_[0])),
            key=lambda x: x[1], reverse=True
        )[:5]

        logger.info(f"\nLogistic Regression results:")
        logger.info(f"  Val  AUC: {val_auc:.4f}  |  AP: {val_ap:.4f}")
        logger.info(f"  Test AUC: {test_auc:.4f}  |  AP: {test_ap:.4f}")
        logger.info(f"  Training time: {elapsed:.1f}s")
        logger.info(f"\n  Top 5 features:")
        for feat, coef in coef_importance:
            logger.info(f"    {feat:<25} {coef:.4f}")

        return {"model": "logistic_regression",
                "val_auc": val_auc, "test_auc": test_auc}


# ── Model 2: XGBoost ──────────────────────────────────────────────────────────

def train_xgboost():
    """Brain 2 — main tabular baseline."""
    logger.info("=" * 55)
    logger.info("Training Brain 2: XGBoost")
    logger.info("=" * 55)

    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed.")
        return

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
        load_tabular_splits()
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.1f}")

    params = {
        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight, "eval_metric": "auc",
        "random_state": cfg.sequence.random_seed, "n_jobs": -1,
    }

    with mlflow.start_run(run_name="xgboost"):
        t0 = time.time()
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        elapsed = time.time() - t0

        val_proba  = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        val_auc,  val_ap  = log_metrics(y_val,  val_proba,  "val")
        test_auc, test_ap = log_metrics(y_test, test_proba, "test")
        mlflow.log_params({**params, "training_secs": round(elapsed, 1)})

        importance = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:5]

        logger.info(f"\nXGBoost results:")
        logger.info(f"  Val  AUC: {val_auc:.4f}  |  AP: {val_ap:.4f}")
        logger.info(f"  Test AUC: {test_auc:.4f}  |  AP: {test_ap:.4f}")
        logger.info(f"  Training time: {elapsed:.1f}s")
        logger.info(f"\n  Top 5 features:")
        for feat, imp in importance:
            logger.info(f"    {feat:<25} {imp:.4f}")

        return {"model": "xgboost",
                "val_auc": val_auc, "test_auc": test_auc}


# ── Shared deep learning training loop ───────────────────────────────────────

def train_deep_model(model, model_name, run_name, epochs=None, lr=None):
    """Generic training loop for LSTM and Transformer."""
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from sessionscout.model.dataset import load_datasets, make_dataloaders

    epochs = epochs or cfg.training.epochs
    lr     = lr     or cfg.training.learning_rate

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    logger.info(f"Device: {device}")

    train_ds, val_ds, test_ds = load_datasets()
    train_loader, val_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds
    )

    model = model.to(device)
    pos_weight = torch.tensor([cfg.training.pos_weight]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser  = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.training.weight_decay,
    )

    best_val_auc   = 0.0
    patience_count = 0
    best_state     = None

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": model_name, "epochs": epochs, "lr": lr,
            "batch_size": cfg.training.batch_size,
            "pos_weight": cfg.training.pos_weight,
            "device": str(device),
            "n_params": sum(p.numel() for p in model.parameters()),
        })

        for epoch in range(1, epochs + 1):

            # Training
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y    = batch["label"].to(device)

                logits = model(ids, mask)
                loss   = criterion(logits, y)

                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_clip
                )
                optimiser.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    ids  = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    logits = model(ids, mask)
                    all_logits.extend(
                        torch.sigmoid(logits).cpu().tolist()
                    )
                    all_labels.extend(batch["label"].tolist())

            val_auc = roc_auc_score(all_labels, all_logits)
            mlflow.log_metrics(
                {"train_loss": round(avg_loss, 4),
                 "val_auc":    round(val_auc, 4)},
                step=epoch
            )
            logger.info(
                f"  Epoch {epoch:>2}/{epochs} | "
                f"loss: {avg_loss:.4f} | val AUC: {val_auc:.4f}"
            )

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc   = val_auc
                patience_count = 0
                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                patience_count += 1
                if patience_count >= cfg.training.early_stopping_patience:
                    logger.info(
                        f"  Early stopping at epoch {epoch}"
                    )
                    break

        # Test evaluation
        model.load_state_dict(best_state)
        model.eval()
        test_logits, test_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(ids, mask)
                test_logits.extend(
                    torch.sigmoid(logits).cpu().tolist()
                )
                test_labels.extend(batch["label"].tolist())

        test_auc = roc_auc_score(test_labels, test_logits)
        mlflow.log_metric("test_auc", round(test_auc, 4))

        cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = cfg.paths.models_dir / f"{model_name}_best.pt"
        torch.save(best_state, model_path)
        mlflow.log_artifact(str(model_path))

        logger.info(f"\n{model_name} results:")
        logger.info(f"  Best val AUC: {best_val_auc:.4f}")
        logger.info(f"  Test AUC:     {test_auc:.4f}")
        logger.info(f"  Saved →       {model_path}")

        return {"model": model_name,
                "val_auc": best_val_auc, "test_auc": test_auc}


# ── Model 3: LSTM ─────────────────────────────────────────────────────────────

def train_lstm():
    """Brain 3 — Bidirectional LSTM on event sequences."""
    logger.info("=" * 55)
    logger.info("Training Brain 3: Bidirectional LSTM")
    logger.info("=" * 55)
    from sessionscout.model.lstm import SessionLSTM
    model = SessionLSTM()
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return train_deep_model(model, "lstm", "lstm")


# ── Model 4: Transformer ──────────────────────────────────────────────────────

def train_transformer():
    """Brain 4 — Transformer encoder on event sequences."""
    logger.info("=" * 55)
    logger.info("Training Brain 4: Transformer")
    logger.info("=" * 55)
    from sessionscout.model.transformer import SessionTransformer
    model = SessionTransformer()
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return train_deep_model(model, "transformer", "transformer")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SessionScout models")
    parser.add_argument(
        "--model",
        choices=["lr", "xgb", "lstm", "transformer", "all"],
        default="lr",
    )
    args = parser.parse_args()
    results = []

    if args.model in ("lr", "all"):
        results.append(train_logistic_regression())
    if args.model in ("xgb", "all"):
        results.append(train_xgboost())
    if args.model in ("lstm", "all"):
        results.append(train_lstm())
    if args.model in ("transformer", "all"):
        results.append(train_transformer())

    if results:
        logger.info("\n" + "=" * 55)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 55)
        for r in results:
            if r:
                logger.info(
                    f"  {r['model']:<25} "
                    f"val AUC: {r['val_auc']:.4f}  "
                    f"test AUC: {r['test_auc']:.4f}"
                )
        logger.info("\nView all runs: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
