"""
model/transformer.py — Custom Transformer encoder for session scoring.

Brain 4 in the 4-model ladder.

Why Transformer after LSTM?
  LSTM reads events one by one and passes a hidden state forward.
  By the time it reaches event 60, the hidden state has been updated
  59 times and early events have faded. It cannot directly compare
  event 3 with event 58.

  The Transformer uses attention: every event looks at every other
  event simultaneously and decides how much to weight each one.
  The pattern "ADD_CART at position 10, then GAP_LONG at position 11,
  then VIEW at position 12" is captured by attention between those
  three positions regardless of how far apart they are in the sequence.

Architecture:
  1. Token embedding: token ID → 64-dim vector
  2. Positional encoding: adds position information (sinusoidal)
  3. Transformer encoder: 2 layers, 4 attention heads
  4. Masked mean pooling: average over non-PAD positions only
  5. Output head: 64-dim → 1 logit
"""

import math
import torch
import torch.nn as nn
from sessionscout.config import cfg


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds a unique position signal to each token embedding so the
    model knows whether an ADD_CART happened early or late in the
    session. Without this, position 5 and position 50 look identical.

    Uses fixed sin/cos patterns — not learned parameters.
    This generalises better to sequence lengths not seen in training.
    """

    def __init__(self, embed_dim: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the positional encoding matrix once
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Frequency denominator: 10000^(2i/d)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims

        # Shape: (1, max_len, embed_dim) — broadcast over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, embed_dim)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SessionTransformer(nn.Module):
    """
    Transformer encoder for binary session conversion prediction.

    Architecture:
      Embedding → Positional Encoding → Transformer Encoder → Mean Pool → Head

    Args:
        vocab_size:   number of token types (default: cfg.model.vocab_size)
        embed_dim:    token embedding size (default: 64)
        num_heads:    attention heads (default: 4)
        num_layers:   encoder layers (default: 2)
        ff_dim:       feedforward dimension inside encoder (default: 128)
        dropout:      dropout rate (default: 0.1)
        max_seq_len:  maximum sequence length (default: 64)
    """

    def __init__(
        self,
        vocab_size: int = None,
        embed_dim: int = None,
        num_heads: int = None,
        num_layers: int = None,
        ff_dim: int = None,
        dropout: float = None,
        max_seq_len: int = None,
    ):
        super().__init__()

        vocab_size = vocab_size or cfg.model.vocab_size
        embed_dim = embed_dim or cfg.model.embed_dim
        num_heads = num_heads or cfg.model.num_heads
        num_layers = num_layers or cfg.model.num_encoder_layers
        ff_dim = ff_dim or cfg.model.ff_dim
        dropout = dropout if dropout is not None else cfg.model.dropout
        max_seq_len = max_seq_len or cfg.model.max_seq_len

        # Layer 1: Token embedding
        # padding_idx=0 ensures PAD tokens produce zero vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=cfg.vocab.pad)

        # Layer 2: Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Layer 3: Transformer encoder
        # Each encoder layer has:
        #   - Multi-head self-attention (4 heads, each sees 64/4=16 dims)
        #   - Feed-forward network (64 → 128 → 64)
        #   - Layer normalisation + residual connections
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # input shape: (batch, seq, embed)
            norm_first=True,  # pre-norm: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer 4: Output head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            # No sigmoid — BCEWithLogitsLoss applies it internally
        )

        self.embed_dim = embed_dim
        self._init_weights()

    def _init_weights(self):
        """
        Initialise embedding and linear weights.
        Xavier uniform is standard for Transformer layers.
        """
        nn.init.xavier_uniform_(self.embedding.weight)
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, seq_len)
        attention_mask: torch.Tensor,  # (B, seq_len) — 1=real, 0=PAD
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids:      token IDs shape (B, 64)
            attention_mask: 1.0 for real tokens, 0.0 for PAD shape (B, 64)

        Returns:
            logits: shape (B,)
        """
        # (B, 64) → (B, 64, embed_dim)
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        # PyTorch TransformerEncoder uses src_key_padding_mask where
        # True = IGNORE this position (opposite of our attention_mask)
        pad_mask = attention_mask == 0  # True where PAD

        # (B, 64, embed_dim) — each token attends to all non-PAD tokens
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Masked mean pool — average only over real (non-PAD) positions
        mask = attention_mask.unsqueeze(-1)  # (B, 64, 1)
        summed = (x * mask).sum(dim=1)  # (B, embed_dim)
        counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = summed / counts  # (B, embed_dim)

        # (B, embed_dim) → (B, 1) → (B,)
        logits = self.head(pooled).squeeze(-1)
        return logits

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Extract attention weights from the first encoder layer.
        Used by explainability/attention_viz.py to visualise
        which events the model attended to for a given session.

        Returns:
            attention_weights: (num_heads, seq_len, seq_len)
        """
        x = self.embedding(input_ids.unsqueeze(0))
        x = self.pos_encoding(x)
        pad_mask = attention_mask.unsqueeze(0) == 0

        # Access first layer's attention directly
        layer = self.encoder.layers[0]
        attn_out, attn_weights = layer.self_attn(
            x,
            x,
            x,
            key_padding_mask=pad_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        return attn_weights.squeeze(0).detach()
