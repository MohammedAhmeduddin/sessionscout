"""
explainability/shap_deep.py — SHAP feature importance for XGBoost.

What this does:
  SHAP (SHapley Additive exPlanations) explains why the XGBoost model
  made a specific prediction. For each session it answers:
  "Which features pushed the conversion probability up or down,
  and by how much?"

Why this matters for DS roles:
  Most companies require model explainability — regulators, product
  teams, and executives all want to know WHY the model predicted what
  it did. "The model said 73% because this session had 3 cart events
  and a long gap" is a sentence that drives business decisions.

Output:
  - SHAP summary plot saved to models/shap_summary.png
  - SHAP values for individual sessions (for attention_viz notebook)
  - Top features printed to terminal
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sessionscout.config import cfg

logger = logging.getLogger(__name__)


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list,
    max_samples: int = 2000,
):
    """
    Compute SHAP values for an XGBoost model.

    Args:
        model:         trained XGBClassifier
        X:             feature matrix (N, n_features)
        feature_names: list of feature column names
        max_samples:   limit samples for speed (SHAP is O(N))

    Returns:
        shap_values: np.ndarray shape (N, n_features)
        explainer:   the SHAP TreeExplainer object
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Run: pip install shap")

    # Use a random subset for speed if dataset is large
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    logger.info(f"Computing SHAP values for {len(X_sample):,} samples...")

    # TreeExplainer is exact for tree-based models (not approximate)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    logger.info("SHAP values computed.")
    return shap_values, explainer, X_sample


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    save_path: Path = None,
):
    """
    Generate and save a SHAP summary (beeswarm) plot.

    Each dot is one session. Position on x-axis = SHAP value
    (how much that feature pushed the prediction up or down).
    Color = feature value (red=high, blue=low).
    """
    import shap

    save_path = save_path or cfg.paths.models_dir / "shap_summary.png"
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP Feature Importance — XGBoost Session Scoring", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"SHAP summary plot saved → {save_path}")
    return save_path


def print_top_features(shap_values: np.ndarray, feature_names: list, top_k: int = 10):
    """Print mean absolute SHAP value per feature — global importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)

    logger.info(f"\nTop {top_k} features by mean |SHAP value|:")
    logger.info(f"  {'Feature':<25} {'Mean |SHAP|':>12}")
    logger.info(f"  {'-'*25} {'-'*12}")
    for feat, val in ranked[:top_k]:
        bar = "█" * int(val * 100)
        logger.info(f"  {feat:<25} {val:>12.4f}  {bar}")

    return ranked


def run_shap_analysis():
    """
    Full SHAP analysis pipeline:
      1. Load features and retrain XGBoost
      2. Compute SHAP values
      3. Save summary plot
      4. Print top features
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    logger.info("=" * 55)
    logger.info("SHAP Analysis — XGBoost Feature Importance")
    logger.info("=" * 55)

    # Load features
    path = cfg.paths.features_parquet
    if not path.exists():
        raise FileNotFoundError(
            f"features.parquet not found at {path}\n"
            "Run: python -m sessionscout.features.engineering"
        )

    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c not in ["session_id", "label", "source"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=cfg.sequence.random_seed,
    )

    # Train XGBoost
    logger.info("Training XGBoost for SHAP analysis...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.sequence.random_seed,
        n_jobs=-1,
        eval_metric="auc",
    )
    model.fit(X_train, y_train, verbose=False)

    # SHAP values on test set
    shap_values, explainer, X_sample = compute_shap_values(model, X_test, feature_cols)

    # Summary plot
    plot_path = plot_shap_summary(shap_values, X_sample, feature_cols)

    # Print rankings
    ranked = print_top_features(shap_values, feature_cols)

    logger.info("\nKey insight:")
    top_feat = ranked[0][0]
    logger.info(f"  '{top_feat}' has the highest SHAP impact on conversion prediction.")
    logger.info(f"  Plot saved → {plot_path}")

    return shap_values, feature_cols, ranked


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        stream=sys.stdout,
    )
    run_shap_analysis()
