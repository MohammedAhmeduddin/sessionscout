"""
model/train.py — Training all 4 models with MLflow tracking.

Run order:
  1. Logistic Regression  (tabular features, ~30 seconds)
  2. XGBoost              (tabular features, ~2 minutes)
  3. LSTM                 (sequences, ~10 minutes)
  4. Transformer          (sequences, ~20 minutes)

Each model logs to MLflow so you can compare them all in one view.

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
    """
    Load features.parquet and split into train/val/test.
    Returns X_train, X_val, X_test, y_train, y_val, y_test as numpy arrays.
    """
    path = cfg.paths.features_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {path}\n"
            "Run: python -m sessionscout.features.engineering"
        )

    df = pd.read_parquet(path)

    # Feature columns — exclude identity and label columns
    feature_cols = [
        c for c in df.columns
        if c not in ["session_id", "label", "source"]
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    # Train / val / test split — stratified
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
        f"Conversion rates — "
        f"train: {y_train.mean():.3f} "
        f"val: {y_val.mean():.3f} "
        f"test: {y_test.mean():.3f}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def log_metrics(y_true, y_pred_proba, prefix):
    """Compute and return AUC + AP metrics."""
    auc = roc_auc_score(y_true, y_pred_proba)
    ap  = average_precision_score(y_true, y_pred_proba)
    mlflow.log_metrics({
        f"{prefix}_auc": round(auc, 4),
        f"{prefix}_ap":  round(ap, 4),
    })
    return auc, ap


# ── Model 1: Logistic Regression ─────────────────────────────────────────────

def train_logistic_regression():
    """
    Brain 1 — Logistic Regression on tabular features.

    This is the AUC floor. Any more complex model must beat this
    to justify its added complexity. Takes ~30 seconds.

    What it can do:
      - Learn that has_cart=1 is a strong positive signal
      - Learn that high gap_ratio is a negative signal
      - Combine all 15 features linearly

    What it cannot do:
      - Understand the ORDER of events
      - Know that VIEW after GAP_LONG is different from VIEW before
    """
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

        # Scale features — LR is sensitive to feature scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        # class_weight='balanced' handles the 9% conversion rate
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=cfg.sequence.random_seed,
            C=1.0,
        )
        model.fit(X_train_s, y_train)

        elapsed = time.time() - t0

        # Evaluate
        val_proba  = model.predict_proba(X_val_s)[:, 1]
        test_proba = model.predict_proba(X_test_s)[:, 1]

        val_auc,  val_ap  = log_metrics(y_val,  val_proba,  "val")
        test_auc, test_ap = log_metrics(y_test, test_proba, "test")

        # Log params
        mlflow.log_params({
            "model":         "logistic_regression",
            "C":             1.0,
            "class_weight":  "balanced",
            "n_features":    len(feature_cols),
            "train_size":    len(X_train),
            "training_secs": round(elapsed, 1),
        })

        # Top 5 most important features by coefficient magnitude
        coef_importance = sorted(
            zip(feature_cols, abs(model.coef_[0])),
            key=lambda x: x[1], reverse=True
        )[:5]

        logger.info(f"\nLogistic Regression results:")
        logger.info(f"  Val  AUC: {val_auc:.4f}  |  AP: {val_ap:.4f}")
        logger.info(f"  Test AUC: {test_auc:.4f}  |  AP: {test_ap:.4f}")
        logger.info(f"  Training time: {elapsed:.1f}s")
        logger.info(f"\n  Top 5 features by coefficient:")
        for feat, coef in coef_importance:
            logger.info(f"    {feat:<25} {coef:.4f}")

        return {
            "model": "logistic_regression",
            "val_auc": val_auc,
            "test_auc": test_auc,
        }


# ── Model 2: XGBoost ──────────────────────────────────────────────────────────

def train_xgboost():
    """
    Brain 2 — XGBoost on tabular features.

    This is the main baseline. XGBoost is what most companies
    actually use for problems like this. The deep learning models
    only earn their place by clearly beating this.

    What it adds over LR:
      - Captures non-linear relationships
      - Automatic feature interactions
      - Built-in handling of missing values

    What it still cannot do:
      - Understand event sequences and temporal order
    """
    logger.info("=" * 55)
    logger.info("Training Brain 2: XGBoost")
    logger.info("=" * 55)

    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        return

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = (
        load_tabular_splits()
    )

    # Compute scale_pos_weight to handle class imbalance
    # = number of negatives / number of positives
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.1f}")

    params = {
        "n_estimators":      500,
        "learning_rate":     0.05,
        "max_depth":         6,
        "min_child_weight":  5,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "scale_pos_weight":  scale_pos_weight,
        "eval_metric":       "auc",
        "random_state":      cfg.sequence.random_seed,
        "n_jobs":            -1,
    }

    with mlflow.start_run(run_name="xgboost"):
        t0 = time.time()

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        elapsed = time.time() - t0

        val_proba  = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        val_auc,  val_ap  = log_metrics(y_val,  val_proba,  "val")
        test_auc, test_ap = log_metrics(y_test, test_proba, "test")

        mlflow.log_params({**params, "training_secs": round(elapsed, 1)})

        # Feature importance
        importance = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )[:5]

        logger.info(f"\nXGBoost results:")
        logger.info(f"  Val  AUC: {val_auc:.4f}  |  AP: {val_ap:.4f}")
        logger.info(f"  Test AUC: {test_auc:.4f}  |  AP: {test_ap:.4f}")
        logger.info(f"  Training time: {elapsed:.1f}s")
        logger.info(f"\n  Top 5 features by importance:")
        for feat, imp in importance:
            logger.info(f"    {feat:<25} {imp:.4f}")

        return {
            "model":    "xgboost",
            "val_auc":  val_auc,
            "test_auc": test_auc,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SessionScout models")
    parser.add_argument(
        "--model",
        choices=["lr", "xgb", "lstm", "transformer", "all"],
        default="lr",
        help="Which model to train"
    )
    args = parser.parse_args()

    results = []

    if args.model in ("lr", "all"):
        results.append(train_logistic_regression())

    if args.model in ("xgb", "all"):
        results.append(train_xgboost())

    if args.model in ("lstm", "transformer", "all"):
        logger.info(
            "\nLSTM and Transformer require model/lstm.py "
            "and model/transformer.py — coming next."
        )

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
