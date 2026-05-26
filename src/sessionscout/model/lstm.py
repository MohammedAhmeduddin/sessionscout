"""
model/lstm.py — Bidirectional LSTM for session sequence modeling.

Brain 3 in the 4-model ladder.

Why LSTM after XGBoost?
  XGBoost sees counts and ratios — it knows a session had 3 views
  and 1 cart, but not whether the cart came before or after the views.
  LSTM reads the sequence left to right (and right to left, because
  it is bidirectional) and maintains a hidden state that carries
  memory of what happened earlier in the session.

What LSTM adds:
  - Temporal ordering — view THEN cart is different from cart THEN view
  - Hidden state memory — earlier events influence later predictions
  - Recurrent processing — each event is processed in context

What LSTM still cannot do well:
  - Long-range dependencies — hidden state degrades over long sequences
  - Parallel attention — cannot directly compare event at position 2
    with event at position 47 without passing through all steps between

That limitation is what the Transformer fixes.
"""

import torch
import torch.nn as nn
from sessionscout.config import cfg


class SessionLSTM(nn.Module):
    """
    Bidirectional LSTM for binary session conversion prediction.

    Architecture:
      1. Embedding layer: token ID → dense vector (vocab_size → embed_dim)
      2. Bidirectional LSTM: processes sequence in both directions
      3. Mean pooling over non-PAD positions: collapses sequence → vector
      4. Output head: vector → single logit (sigmoid applied in loss)

    Args:
        vocab_size:  number of distinct token types (6: PAD,VIEW,CART,BUY,GAP_S,GAP_L)
        embed_dim:   embedding dimension (default 64)
        hidden_dim:  LSTM hidden state size (default 64)
        num_layers:  number of stacked LSTM layers (default 2)
        dropout:     dropout between LSTM layers (default 0.1)
    """

    def __init__(
        self,
        vocab_size: int = None,
        embed_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        dropout: float = None,
    ):
        super().__init__()

        # Use config defaults if not overridden
        vocab_size = vocab_size or cfg.model.vocab_size
        embed_dim = embed_dim or cfg.model.embed_dim
        hidden_dim = hidden_dim or cfg.model.lstm_hidden
        num_layers = num_layers or cfg.model.lstm_layers
        dropout = dropout if dropout is not None else cfg.model.lstm_dropout

        # Layer 1: Embedding
        # Maps each token ID to a learnable dense vector
        # padding_idx=0 means PAD tokens always produce zero vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=cfg.vocab.pad)

        # Layer 2: Bidirectional LSTM
        # bidirectional=True doubles the output size (hidden_dim * 2)
        # batch_first=True means input shape is (batch, seq_len, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer 3: Output head
        # hidden_dim * 2 because bidirectional doubles the hidden size
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            # No sigmoid here — BCEWithLogitsLoss applies it internally
        )

        self.hidden_dim = hidden_dim

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len)
        attention_mask: torch.Tensor,  # (batch, seq_len) — 1=real, 0=PAD
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids:      token IDs, shape (B, 64)
            attention_mask: 1.0 for real tokens, 0.0 for PAD, shape (B, 64)

        Returns:
            logits: shape (B,) — raw scores before sigmoid
        """
        # (B, 64) → (B, 64, embed_dim)
        x = self.embedding(input_ids)

        # LSTM output: (B, 64, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # Mean pool over non-PAD positions only
        # Expand mask to match hidden dimension: (B, 64, 1)
        mask = attention_mask.unsqueeze(-1)

        # Zero out PAD positions, sum real positions, divide by count
        summed = (lstm_out * mask).sum(dim=1)  # (B, hidden_dim*2)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = summed / counts  # (B, hidden_dim*2)

        # (B, hidden_dim*2) → (B, 1) → (B,)
        logits = self.head(pooled).squeeze(-1)
        return logits
