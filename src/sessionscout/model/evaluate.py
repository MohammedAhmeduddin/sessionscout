"""
model/evaluate.py — Evaluation metrics for all 4 models.

Generates the comparison table that goes in the README.

Usage:
  python -m sessionscout.model.evaluate
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)

from sessionscout.config import cfg

logger = logging.getLogger(__name__)


def evaluate_deep_model(model, loader, device="cpu") -> dict:
    """
    Run inference on a DataLoader and return evaluation metrics.

    Returns dict with: auc, ap, precision_at_50pct, recall_at_50pct
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            all_logits.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    probs  = np.array(all_logits)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    return {
        "auc":       round(roc_auc_score(labels, probs), 4),
        "ap":        round(average_precision_score(labels, probs), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, zero_division=0), 4),
    }


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of top-K predicted sessions that actually converted."""
    top_k = np.argsort(scores)[::-1][:k]
    return labels[top_k].mean()


def print_results_table(results: dict):
    """Print a formatted comparison table of all models."""
    logger.info("\n" + "=" * 65)
    logger.info("MODEL COMPARISON TABLE")
    logger.info("=" * 65)
    logger.info(
        f"  {'Model':<25} {'Val AUC':>8} {'Test AUC':>9} "
        f"{'AP':>7} {'P@500':>7}"
    )
    logger.info(f"  {'-'*25} {'-'*8} {'-'*9} {'-'*7} {'-'*7}")

    for model_name, metrics in results.items():
        logger.info(
            f"  {model_name:<25} "
            f"{metrics.get('val_auc', 0):>8.4f} "
            f"{metrics.get('test_auc', 0):>9.4f} "
            f"{metrics.get('ap', 0):>7.4f} "
            f"{metrics.get('p_at_500', 0):>7.4f}"
        )

    logger.info("=" * 65)
    logger.info("\nNote: AUC values are from the dev dataset (50K OTTO sessions).")
    logger.info("Full dataset results will differ.")


def run_full_evaluation():
    """Load all trained models and evaluate on the test set."""
    from sessionscout.model.dataset import load_datasets, make_dataloaders
    from sessionscout.model.lstm import SessionLSTM
    from sessionscout.model.transformer import SessionTransformer

    logger.info("Loading test dataset...")
    _, _, test_ds = load_datasets()
    _, _, test_loader = make_dataloaders(test_ds, test_ds, test_ds)

    results = {}

    # LSTM
    lstm_path = cfg.paths.models_dir / "lstm_best.pt"
    if lstm_path.exists():
        logger.info("Evaluating LSTM...")
        model = SessionLSTM()
        model.load_state_dict(
            torch.load(lstm_path, map_location="cpu", weights_only=True)
        )
        metrics = evaluate_deep_model(model, test_loader)

        # Precision@500
        all_scores, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch["input_ids"], batch["attention_mask"])
                all_scores.extend(torch.sigmoid(logits).tolist())
                all_labels.extend(batch["label"].tolist())
        metrics["p_at_500"] = round(
            precision_at_k(np.array(all_labels), np.array(all_scores), 500), 4
        )
        results["lstm"] = metrics
        logger.info(f"  LSTM: {metrics}")

    # Transformer
    tf_path = cfg.paths.models_dir / "transformer_best.pt"
    if tf_path.exists():
        logger.info("Evaluating Transformer...")
        model = SessionTransformer()
        model.load_state_dict(
            torch.load(tf_path, map_location="cpu", weights_only=True)
        )
        metrics = evaluate_deep_model(model, test_loader)
        results["transformer"] = metrics
        logger.info(f"  Transformer: {metrics}")

    print_results_table(results)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        stream=sys.stdout,
    )
    run_full_evaluation()
