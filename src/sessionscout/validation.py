"""
validation.py — Keep scoring requests inside what the model was trained on.

Training sequences (features/sequences.py) are cut at the first purchase and
contain gap tokens only BETWEEN two real events. So a training sequence never:
  - contains PURCHASE (3) or PAD (0) as an event
  - starts or ends with a gap token, or has two gaps in a row
  - has fewer than cfg.sequence.min_len (3) events

Scoring anything else is out-of-distribution: the model still returns a
number, but it means nothing. The API rejects such input with a clear reason
instead. Unknown token IDs are rejected too, because they would index past
the embedding table and crash the model.
"""

from typing import List, Optional

from sessionscout.config import cfg

# Longest sequence accepted per request. Longer sessions are fine: only the
# last cfg.sequence.max_len (64) events are scored, exactly as in training.
MAX_INPUT_EVENTS = 1000

EVENT_TOKENS = {cfg.vocab.view, cfg.vocab.add_cart}
GAP_TOKENS = {cfg.vocab.gap_short, cfg.vocab.gap_long}
ALLOWED_TOKENS = EVENT_TOKENS | GAP_TOKENS


def validate_sequence(sequence: List[int]) -> Optional[str]:
    """Return None if the sequence can be scored, otherwise the reason it can't."""
    for position, token in enumerate(sequence):
        if token == cfg.vocab.purchase:
            return (
                f"PURCHASE (3) at position {position}: score sessions before the "
                "purchase; the model never sees PURCHASE tokens."
            )
        if token == cfg.vocab.pad:
            return f"PAD (0) at position {position}: send events only, without padding."
        if token not in ALLOWED_TOKENS:
            return (
                f"Unknown token {token} at position {position}. Allowed: VIEW=1, "
                "ADD_CART=2, GAP_SHORT=4, GAP_LONG=5."
            )

    if len(sequence) < cfg.sequence.min_len:
        return (
            f"Need at least {cfg.sequence.min_len} events; got {len(sequence)}. "
            "The model was trained on sessions of 3+ events."
        )
    if sequence[0] in GAP_TOKENS:
        return "A session can't start with a gap token; gaps sit between two events."
    if sequence[-1] in GAP_TOKENS:
        return (
            "A session can't end with a gap token; send the gap together with the "
            "next VIEW or ADD_CART."
        )
    for position in range(1, len(sequence)):
        if sequence[position] in GAP_TOKENS and sequence[position - 1] in GAP_TOKENS:
            return f"Two gap tokens in a row at positions {position - 1}-{position}."
    return None
