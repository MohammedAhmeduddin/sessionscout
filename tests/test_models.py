"""
tests/test_models.py — Architecture tests for LSTM and Transformer.

Runs without any trained weights or datasets.
Verifies output shapes, dtypes, and basic properties.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sessionscout.config import cfg
from sessionscout.model.lstm import SessionLSTM
from sessionscout.model.transformer import SessionTransformer

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_batch(batch_size=4, seq_len=None, n_pad=50):
    """Create a synthetic batch with some PAD tokens."""
    seq_len = seq_len or cfg.sequence.max_len
    ids = torch.randint(1, cfg.vocab.size, (batch_size, seq_len))
    ids[:, :n_pad] = cfg.vocab.pad
    mask = (ids != cfg.vocab.pad).float()
    return ids, mask


# ── LSTM tests ────────────────────────────────────────────────────────────────


class TestLSTM:
    def test_output_shape(self):
        model = SessionLSTM()
        ids, mask = make_batch(4)
        out = model(ids, mask)
        assert out.shape == (4,)

    def test_output_dtype(self):
        model = SessionLSTM()
        ids, mask = make_batch(2)
        out = model(ids, mask)
        assert out.dtype == torch.float32

    def test_probabilities_in_range(self):
        model = SessionLSTM()
        ids, mask = make_batch(8)
        logits = model(ids, mask)
        probs = torch.sigmoid(logits)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_batch_size_one(self):
        model = SessionLSTM()
        ids, mask = make_batch(1)
        out = model(ids, mask)
        assert out.shape == (1,)

    def test_all_pad_sequence(self):
        """Model should not crash on a mostly-PAD sequence."""
        model = SessionLSTM()
        ids = torch.zeros(2, cfg.sequence.max_len, dtype=torch.long)
        ids[:, -1] = cfg.vocab.view  # one real token
        mask = (ids != cfg.vocab.pad).float()
        out = model(ids, mask)
        assert out.shape == (2,)

    def test_parameter_count(self):
        model = SessionLSTM()
        n = sum(p.numel() for p in model.parameters())
        assert n > 10_000, f"Too few parameters: {n}"
        assert n < 10_000_000, f"Too many parameters: {n}"

    def test_gradients_flow(self):
        model = SessionLSTM()
        ids, mask = make_batch(4)
        labels = torch.zeros(4)
        logits = model(ids, mask)
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ── Transformer tests ─────────────────────────────────────────────────────────


class TestTransformer:
    def test_output_shape(self):
        model = SessionTransformer()
        ids, mask = make_batch(4)
        out = model(ids, mask)
        assert out.shape == (4,)

    def test_output_dtype(self):
        model = SessionTransformer()
        ids, mask = make_batch(2)
        out = model(ids, mask)
        assert out.dtype == torch.float32

    def test_probabilities_in_range(self):
        model = SessionTransformer()
        ids, mask = make_batch(8)
        logits = model(ids, mask)
        probs = torch.sigmoid(logits)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_batch_size_one(self):
        model = SessionTransformer()
        ids, mask = make_batch(1)
        out = model(ids, mask)
        assert out.shape == (1,)

    def test_attention_weights_shape(self):
        model = SessionTransformer()
        ids = torch.randint(1, cfg.vocab.size, (cfg.sequence.max_len,))
        mask = (ids != cfg.vocab.pad).float()
        attn = model.get_attention_weights(ids, mask)
        assert attn.shape == (
            cfg.model.num_heads,
            cfg.sequence.max_len,
            cfg.sequence.max_len,
        )

    def test_parameter_count_smaller_than_lstm(self):
        lstm = SessionLSTM()
        tf = SessionTransformer()
        n_lstm = sum(p.numel() for p in lstm.parameters())
        n_tf = sum(p.numel() for p in tf.parameters())
        # Transformer should be smaller (attention is parameter-efficient)
        assert n_tf < n_lstm, f"Expected Transformer ({n_tf}) < LSTM ({n_lstm})"

    def test_gradients_flow(self):
        model = SessionTransformer()
        ids, mask = make_batch(4)
        labels = torch.zeros(4)
        logits = model(ids, mask)
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_positional_encoding_changes_output(self):
        """Verify positional encoding actually affects the output."""
        model = SessionTransformer()
        model.eval()
        ids = torch.randint(1, cfg.vocab.size, (1, cfg.sequence.max_len))
        mask = (ids != cfg.vocab.pad).float()
        with torch.no_grad():
            out1 = model(ids, mask)
            # Same tokens, different positions — shift sequence by one
            ids2 = ids.clone()
            ids2[:, 1:] = ids[:, :-1]
            ids2[:, 0] = cfg.vocab.pad
            mask2 = (ids2 != cfg.vocab.pad).float()
            out2 = model(ids2, mask2)
        assert out1.item() != out2.item()
