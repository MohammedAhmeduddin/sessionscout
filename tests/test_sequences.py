"""
tests/test_sequences.py

Runs entirely on synthetic data — no Kaggle download needed.
These tests run in CI on every push to GitHub.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sessionscout.config import cfg, VOCAB, EVENT_TO_TOKEN
from sessionscout.features.sequences import (
    inject_gap_tokens,
    build_session_sequence,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_session(events, gap_secs=30):
    """Build a synthetic session DataFrame from a list of event type strings."""
    rows = []
    ts = 1_000_000.0
    for ev in events:
        rows.append({
            "session_id":    "test",
            "timestamp_sec": ts,
            "event_type":    ev,
            "source":        "test",
        })
        ts += gap_secs
    return pd.DataFrame(rows)


# ── vocabulary ────────────────────────────────────────────────────────────────

def test_pad_is_zero():
    assert cfg.vocab.pad == 0

def test_view_is_one():
    assert cfg.vocab.view == 1

def test_add_cart_is_two():
    assert cfg.vocab.add_cart == 2

def test_purchase_is_three():
    assert cfg.vocab.purchase == 3

def test_gap_short_is_four():
    assert cfg.vocab.gap_short == 4

def test_gap_long_is_five():
    assert cfg.vocab.gap_long == 5

def test_vocab_size_is_six():
    assert cfg.vocab.size == 6

def test_vocab_dict_complete():
    assert set(VOCAB.keys()) == {0, 1, 2, 3, 4, 5}

def test_event_to_token_reverse():
    for tok_id, name in VOCAB.items():
        assert EVENT_TO_TOKEN[name] == tok_id

def test_no_duplicate_token_ids():
    ids = list(VOCAB.keys())
    assert len(ids) == len(set(ids))


# ── gap injection ─────────────────────────────────────────────────────────────

def test_no_gap_below_threshold():
    # 60s < 120s minimum → no gap token
    events = [(1_000_000.0, "view"), (1_000_060.0, "view")]
    result = inject_gap_tokens(events)
    assert len(result) == 2

def test_short_gap_injected():
    # 300s is between 120s and 600s → GAP_SHORT
    events = [(1_000_000.0, "view"), (1_000_300.0, "addtocart")]
    names = [e for _, e in inject_gap_tokens(events)]
    assert "GAP_SHORT" in names

def test_long_gap_injected():
    # 1200s > 600s → GAP_LONG
    events = [(1_000_000.0, "view"), (1_001_200.0, "view")]
    names = [e for _, e in inject_gap_tokens(events)]
    assert "GAP_LONG" in names

def test_gap_token_between_events_not_appended():
    events = [(1_000_000.0, "view"), (1_000_300.0, "addtocart")]
    result = inject_gap_tokens(events)
    assert result[0][1] == "view"
    assert result[-1][1] == "addtocart"

def test_single_event_unchanged():
    events = [(1_000_000.0, "view")]
    assert inject_gap_tokens(events) == events

def test_original_events_preserved_after_gap_injection():
    events = [(1_000_000.0, "view"), (1_000_300.0, "purchase")]
    names = [e for _, e in inject_gap_tokens(events)]
    assert "view" in names
    assert "purchase" in names


# ── session sequence builder ──────────────────────────────────────────────────

def test_output_length_always_max_len():
    r = build_session_sequence(make_session(["view"] * 20 + ["purchase"]))
    assert r is not None
    assert len(r["sequence"]) == cfg.sequence.max_len

def test_purchase_session_label_one():
    r = build_session_sequence(make_session(["view", "addtocart", "purchase"]))
    assert r is not None
    assert r["label"] == 1

def test_browse_session_label_zero():
    r = build_session_sequence(make_session(["view", "view", "addtocart"]))
    assert r is not None
    assert r["label"] == 0

def test_too_short_returns_none():
    r = build_session_sequence(make_session(["view"]))
    assert r is None

def test_two_events_below_min_len_returns_none():
    r = build_session_sequence(make_session(["view", "view"]))
    assert r is None

def test_left_padding_with_zeros():
    r = build_session_sequence(make_session(["view", "view", "addtocart"]))
    assert r is not None
    n_pad = cfg.sequence.max_len - r["seq_len"]
    if n_pad > 0:
        assert r["sequence"][0] == cfg.vocab.pad
        assert r["sequence"][n_pad] != cfg.vocab.pad

def test_all_tokens_in_valid_range():
    r = build_session_sequence(make_session(["view", "addtocart", "purchase"] * 5))
    assert r is not None
    assert all(0 <= t < cfg.vocab.size for t in r["sequence"])

def test_long_session_truncated_to_max_len():
    r = build_session_sequence(make_session(["view"] * 200))
    assert r is not None
    assert len(r["sequence"]) == cfg.sequence.max_len
    assert r["seq_len"] == cfg.sequence.max_len

def test_right_aligned_last_token_is_most_recent():
    r = build_session_sequence(make_session(["view", "view", "purchase"]))
    assert r is not None
    last_real = next(t for t in reversed(r["sequence"]) if t != cfg.vocab.pad)
    assert last_real == cfg.vocab.purchase

def test_n_views_counted_correctly():
    r = build_session_sequence(make_session(["view", "view", "view", "addtocart"]))
    assert r is not None
    assert r["n_views"] == 3

def test_n_carts_counted_correctly():
    r = build_session_sequence(make_session(["view", "addtocart", "addtocart"]))
    assert r is not None
    assert r["n_carts"] == 2

def test_purchase_token_present_in_converted_session():
    r = build_session_sequence(make_session(["view", "addtocart", "purchase"]))
    assert r is not None
    assert cfg.vocab.purchase in r["sequence"]

def test_seq_len_matches_non_pad_count():
    r = build_session_sequence(make_session(["view", "view", "addtocart"]))
    assert r is not None
    non_pad = sum(1 for t in r["sequence"] if t != cfg.vocab.pad)
    assert non_pad == r["seq_len"]


# ── config ────────────────────────────────────────────────────────────────────

def test_max_len_is_64():
    assert cfg.sequence.max_len == 64

def test_min_len_less_than_max_len():
    assert cfg.sequence.min_len < cfg.sequence.max_len

def test_gap_short_less_than_gap_long():
    assert cfg.sequence.gap_short_min < cfg.sequence.gap_long_min

def test_model_vocab_size_matches_vocab_config():
    assert cfg.model.vocab_size == cfg.vocab.size

def test_pos_weight_positive():
    assert cfg.training.pos_weight > 0
