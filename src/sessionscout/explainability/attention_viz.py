"""
explainability/attention_viz.py — Attention weight visualization.

What this does:
  Extracts the attention weights from the trained Transformer and
  plots them as a heatmap for a specific session. This shows WHICH
  events the model attended to when making its prediction.

Why this matters:
  This is the interview "wow" moment. You can point to the heatmap
  and say: "The model learned to attend strongly to ADD_CART events
  that are followed by GAP_LONG — that hesitation pattern is the
  strongest signal." That is a finding a business can act on.

Output:
  - Attention heatmap saved to models/attention_heatmap.png
  - Token importance scores printed to terminal
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sessionscout.config import cfg, VOCAB

logger = logging.getLogger(__name__)


def load_best_transformer(model_path: Optional[Path] = None):
    """Load the best saved Transformer weights."""
    from sessionscout.model.transformer import SessionTransformer

    path = model_path or cfg.paths.models_dir / "transformer_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"transformer_best.pt not found at {path}\n"
            "Run: python -m sessionscout.model.train --model transformer"
        )

    model = SessionTransformer()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Transformer loaded from {path}")
    return model


def tokens_to_names(sequence: list) -> list:
    """Convert a list of token IDs to human-readable event names."""
    return [VOCAB.get(t, f"tok_{t}") for t in sequence]


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    sequence: list,
    session_id: str = "example",
    head_idx: int = 0,
    save_path: Optional[Path] = None,
):
    """
    Plot attention weights as a heatmap.

    attention_weights: (num_heads, seq_len, seq_len)
    sequence:          list of token IDs (length seq_len)
    head_idx:          which attention head to visualise (0-3)

    Rows = query positions (what is attending)
    Cols = key positions (what is being attended to)
    Darker = stronger attention
    """
    save_path = save_path or cfg.paths.models_dir / "attention_heatmap.png"
    cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)

    # Get weights for the selected head
    weights = attention_weights[head_idx].numpy()  # (seq_len, seq_len)

    # Only show non-PAD positions — strip leading zeros
    non_pad_mask = [t != cfg.vocab.pad for t in sequence]
    non_pad_idx = [i for i, m in enumerate(non_pad_mask) if m]

    if len(non_pad_idx) == 0:
        logger.warning("All tokens are PAD — nothing to visualise.")
        return None

    # Slice to non-PAD region
    w_trimmed = weights[np.ix_(non_pad_idx, non_pad_idx)]
    token_names = [tokens_to_names(sequence)[i] for i in non_pad_idx]

    # Shorten long names for display
    short_names = {
        "VIEW": "VIEW",
        "ADD_CART": "CART",
        "PURCHASE": "BUY",
        "GAP_SHORT": "G_S",
        "GAP_LONG": "G_L",
        "PAD": "PAD",
    }
    labels = [short_names.get(n, n) for n in token_names]

    # Plot
    n = len(labels)
    fig_size = max(6, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    im = ax.imshow(w_trimmed, cmap="Blues", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention weight")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Keys (what is being attended to)")
    ax.set_ylabel("Queries (what is attending)")
    ax.set_title(
        f"Transformer Attention Weights\n"
        f"Session: {session_id} | Head {head_idx + 1}/4",
        pad=10,
    )

    # Annotate cells with values > 0.1 for readability
    for i in range(n):
        for j in range(n):
            val = w_trimmed[i, j]
            if val > 0.15:
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if val > 0.5 else "black",
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Attention heatmap saved → {save_path}")
    return save_path


def analyse_session(
    model,
    sequence: list,
    session_id: str = "session",
    save_dir: Optional[Path] = None,
):
    """
    Full attention analysis for one session.

    Extracts attention weights, plots heatmap for all 4 heads,
    and identifies the top attended token pairs.

    Args:
        model:      loaded SessionTransformer
        sequence:   list of 64 token IDs
        session_id: label for the plot title

    Returns:
        attention_weights: (num_heads, seq_len, seq_len)
    """
    save_dir = save_dir or cfg.paths.models_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    ids = torch.tensor(sequence, dtype=torch.long)
    mask = (ids != cfg.vocab.pad).float()

    with torch.no_grad():
        attn_weights = model.get_attention_weights(ids, mask)

    # Plot all 4 heads
    for head_idx in range(attn_weights.shape[0]):
        plot_attention_heatmap(
            attn_weights,
            sequence,
            session_id=session_id,
            head_idx=head_idx,
            save_path=save_dir / f"attention_head{head_idx+1}.png",
        )

    # Print which event pairs have highest attention
    non_pad = [(i, t) for i, t in enumerate(sequence) if t != cfg.vocab.pad]
    mean_attn = attn_weights.mean(dim=0).numpy()

    logger.info("\nTop 5 attended event pairs (averaged across heads):")
    pairs = []
    for qi, qt in non_pad:
        for ki, kt in non_pad:
            if qi != ki:
                pairs.append(
                    (
                        mean_attn[qi, ki],
                        VOCAB.get(qt, str(qt)),
                        VOCAB.get(kt, str(kt)),
                    )
                )
    pairs.sort(reverse=True)

    for score, q_name, k_name in pairs[:5]:
        logger.info(f"  {q_name:<12} attends to {k_name:<12} (weight: {score:.3f})")

    return attn_weights


def run_attention_analysis():
    """
    Run attention analysis on interesting sessions from the test set.

    Finds one converting session and one non-converting session
    and plots attention heatmaps for both.
    """
    import pandas as pd

    logger.info("=" * 55)
    logger.info("Attention Weight Analysis — Transformer")
    logger.info("=" * 55)

    model = load_best_transformer()

    # Load sequences
    seq_df = pd.read_parquet(cfg.paths.sequences_parquet)

    # Find an interesting converting session
    # (one with carts and gaps — the hesitation pattern)
    converted = seq_df[
        (seq_df["label"] == 1)
        & (seq_df["n_carts"] > 0)
        & (seq_df["seq_len"] >= 5)
        & (seq_df["seq_len"] <= 20)
    ]

    if len(converted) == 0:
        logger.warning("No suitable converting session found.")
        return

    # Pick the first one
    row = converted.iloc[0]
    session_id = row["session_id"]
    sequence = row["sequence"]

    logger.info(f"\nAnalysing converting session: {session_id}")
    logger.info(f"  Sequence length: {row['seq_len']} events")
    logger.info(f"  Events: {tokens_to_names(sequence)}")

    analyse_session(model, sequence, session_id=session_id)

    # Also analyse a non-converting session for comparison
    not_converted = seq_df[
        (seq_df["label"] == 0)
        & (seq_df["n_carts"] > 0)
        & (seq_df["seq_len"] >= 5)
        & (seq_df["seq_len"] <= 20)
    ]

    if len(not_converted) > 0:
        row2 = not_converted.iloc[0]
        logger.info(f"\nAnalysing non-converting session: {row2['session_id']}")
        logger.info(f"  Sequence length: {row2['seq_len']} events")
        analyse_session(
            model,
            row2["sequence"],
            session_id=row2["session_id"],
            save_dir=cfg.paths.models_dir / "non_converting",
        )

    logger.info("\nAttention analysis complete.")
    logger.info(f"Heatmaps saved in: {cfg.paths.models_dir}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        stream=sys.stdout,
    )
    run_attention_analysis()
