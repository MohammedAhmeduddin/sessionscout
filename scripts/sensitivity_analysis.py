"""
scripts/sensitivity_analysis.py — Business impact table.

The honest version of business impact.

Instead of claiming one fabricated dollar figure, this script
produces a table crossing average order value (AOV) with
intervention uplift under clearly documented assumptions.

This is what real analysts at real companies produce.
It answers: "Under what assumptions does this project pay for itself?"

Assumptions documented explicitly:
  - AOV range:         $45 to $120 (typical e-commerce range)
  - Uplift range:      5% to 15% (intervention conversion lift)
  - Intervention cost: $2.50 per triggered action (discount/notification)
  - Top-K sessions:    500 flagged per day (precision@500 from LSTM)
  - Operating days:    365 per year
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sessionscout.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def compute_precision_at_k(y_true, y_scores, k):
    """What fraction of the top-K predicted sessions actually converted?"""
    top_k_idx  = np.argsort(y_scores)[::-1][:k]
    top_k_true = y_true[top_k_idx]
    return top_k_true.mean()


def run_sensitivity_analysis(
    precision_at_k: float = None,
    k: int = None,
    save_path: Path = None,
):
    """
    Generate sensitivity table: AOV × uplift → daily recoverable revenue.

    Args:
        precision_at_k: fraction of top-K sessions that are true positives.
                        If None, loads LSTM predictions from saved model.
        k:              number of sessions to flag per day
        save_path:      where to save the CSV

    Returns:
        DataFrame with the sensitivity table
    """
    k = k or cfg.business.top_k_sessions

    # If precision not provided, compute from saved model
    if precision_at_k is None:
        precision_at_k = _compute_lstm_precision_at_k(k)

    logger.info("=" * 60)
    logger.info("Business Impact Sensitivity Analysis")
    logger.info("=" * 60)
    logger.info(f"\nAssumptions:")
    logger.info(f"  Top-K sessions flagged per day: {k:,}")
    logger.info(f"  Model precision@{k}:              {precision_at_k:.3f}")
    logger.info(
        f"  True positives per day:          "
        f"{precision_at_k * k:.0f} sessions"
    )
    logger.info(f"  Intervention cost per session:   ${cfg.business.intervention_cost:.2f}")
    logger.info(f"\nSensitivity table (daily recoverable revenue $):")

    rows = []
    for aov in cfg.business.aov_range:
        row = {"AOV ($)": aov}
        for uplift in cfg.business.uplift_range:
            # True positive sessions that the intervention tips over
            recoverable_sessions = precision_at_k * k * uplift
            # Revenue from those sessions minus intervention costs
            gross_revenue = recoverable_sessions * aov
            intervention_cost = k * cfg.business.intervention_cost
            net_revenue = gross_revenue - intervention_cost
            row[f"Uplift {int(uplift*100)}%"] = f"${net_revenue:,.0f}"
        rows.append(row)

    table = pd.DataFrame(rows)
    table = table.set_index("AOV ($)")

    # Print table
    logger.info(f"\n{table.to_string()}")

    # Annual numbers for the best case
    best_aov    = cfg.business.aov_range[-1]
    best_uplift = cfg.business.uplift_range[-1]
    daily_best  = (
        precision_at_k * k * best_uplift * best_aov
        - k * cfg.business.intervention_cost
    )
    annual_best = daily_best * 365

    logger.info(f"\nBest-case annual impact:")
    logger.info(
        f"  AOV=${best_aov}, Uplift={int(best_uplift*100)}%, "
        f"Precision@{k}={precision_at_k:.3f}"
    )
    logger.info(f"  Daily:  ${daily_best:,.0f}")
    logger.info(f"  Annual: ${annual_best:,.0f}")

    logger.info(f"\n⚠ These are estimates under documented assumptions.")
    logger.info(f"  Real impact requires A/B testing with the deployed model.")

    # Save
    if save_path is None:
        save_path = cfg.paths.models_dir / "sensitivity_analysis.csv"
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(save_path)
    logger.info(f"\nTable saved → {save_path}")

    return table


def _compute_lstm_precision_at_k(k: int) -> float:
    """
    Load LSTM model and compute precision@K on the test set.
    Falls back to a conservative estimate if model not found.
    """
    try:
        import torch
        from sessionscout.model.lstm import SessionLSTM
        from sessionscout.model.dataset import load_datasets, make_dataloaders

        model_path = cfg.paths.models_dir / "lstm_best.pt"
        if not model_path.exists():
            logger.warning(
                "lstm_best.pt not found. "
                "Using conservative precision estimate of 0.35."
            )
            return 0.35

        model = SessionLSTM()
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        _, _, test_ds = load_datasets()
        _, _, test_loader = make_dataloaders(
            test_ds, test_ds, test_ds, batch_size=512
        )

        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch["input_ids"], batch["attention_mask"])
                all_scores.extend(torch.sigmoid(logits).tolist())
                all_labels.extend(batch["label"].tolist())

        scores = np.array(all_scores)
        labels = np.array(all_labels)
        p_at_k = compute_precision_at_k(labels, scores, k)
        logger.info(f"LSTM Precision@{k}: {p_at_k:.4f}")
        return p_at_k

    except Exception as e:
        logger.warning(f"Could not compute precision@K: {e}. Using 0.35.")
        return 0.35


if __name__ == "__main__":
    run_sensitivity_analysis()
