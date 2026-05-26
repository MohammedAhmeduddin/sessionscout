"""
features/sequences.py — Raw events → padded token sequences.

What this file does in plain English:
  1. Loads raw Retail Rocket events.csv and OTTO train.jsonl
  2. Normalises both into the same format (session_id, timestamp, event_type)
  3. For each session, injects gap tokens where the user was inactive
  4. Converts each event name to its token ID number
  5. Right-aligns the sequence and left-pads to length 64
  6. Saves to data/processed/sequences.parquet

The 5 token types:
  VIEW(1)      — user viewed a product
  ADD_CART(2)  — user added to cart
  PURCHASE(3)  — user bought (this is our label source)
  GAP_SHORT(4) — user was inactive 2-10 min (hesitation)
  GAP_LONG(5)  — user was inactive 10+ min (likely leaving)
  PAD(0)       — empty slot (left padding)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from sessionscout.config import cfg, VOCAB, EVENT_TO_TOKEN

logger = logging.getLogger(__name__)


# ── Step 1: Load raw data ─────────────────────────────────────────────────────

def load_retailrocket() -> pd.DataFrame:
    """
    Load Retail Rocket events.csv and normalise column names.

    Raw columns: timestamp (ms), visitorid, event, itemid
    We treat visitorid as session_id (short dataset, no session splits).
    Event values: 'view', 'addtocart', 'transaction' → we rename 'transaction' to 'purchase'.
    """
    path = cfg.paths.rr_events
    if not path.exists():
        raise FileNotFoundError(
            f"\nRetail Rocket file not found: {path}"
            "\nFix: kaggle datasets download -d retailrocket/ecommerce-dataset"
            " -p data/raw/retailrocket --unzip"
        )

    logger.info("Loading Retail Rocket...")
    df = pd.read_csv(path)

    df = df.rename(columns={
        "visitorid": "session_id",
        "event":     "event_type",
    })

    # Convert milliseconds to seconds
    df["timestamp_sec"] = df["timestamp"] / 1000.0
    df["source"]        = "retailrocket"
    df["session_id"]    = "rr_" + df["session_id"].astype(str)

    # Normalise event names
    df["event_type"] = (
        df["event_type"]
        .str.lower()
        .str.strip()
        .replace({"transaction": "purchase"})
    )

    df = df[["session_id", "timestamp_sec", "event_type", "source"]]

    logger.info(
        f"  Retail Rocket: {len(df):,} events | "
        f"{df['session_id'].nunique():,} sessions | "
        f"types: {df['event_type'].value_counts().to_dict()}"
    )
    return df


def load_otto(max_sessions: Optional[int] = None) -> pd.DataFrame:
    """
    Load OTTO train.jsonl and normalise to the same format.

    OTTO format — one JSON object per line:
      {"session": 12345,
       "events": [{"aid": 111, "ts": 1661723400, "type": "clicks"}, ...]}

    OTTO event types map to ours:
      clicks → view
      carts  → addtocart
      orders → purchase

    max_sessions: pass an integer during development to load only
    the first N sessions. Use None for the full 220M event dataset.
    """
    path = cfg.paths.otto_train
    if not path.exists():
        raise FileNotFoundError(
            f"\nOTTO file not found: {path}"
            "\nFix: kaggle competitions download"
            " -c otto-recommender-system -p data/raw/otto --unzip"
        )

    logger.info(
        f"Loading OTTO"
        f"{f' (first {max_sessions:,} sessions)' if max_sessions else ' (full dataset)'}..."
    )

    type_map = {
        "clicks": "view",
        "carts":  "addtocart",
        "orders": "purchase",
    }

    records = []
    n_sessions = 0

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = f"otto_{obj['session']}"
            for ev in obj["events"]:
                records.append({
                    "session_id":    sid,
                    "timestamp_sec": float(ev["ts"]),
                    "event_type":    type_map.get(ev["type"], ev["type"]),
                    "source":        "otto",
                })
            n_sessions += 1
            if max_sessions and n_sessions >= max_sessions:
                break

    df = pd.DataFrame(records)
    logger.info(
        f"  OTTO: {len(df):,} events | "
        f"{df['session_id'].nunique():,} sessions | "
        f"types: {df['event_type'].value_counts().to_dict()}"
    )
    return df


# ── Step 2: Inject gap tokens ─────────────────────────────────────────────────

def inject_gap_tokens(
    events: List[Tuple[float, str]]
) -> List[Tuple[float, str]]:
    """
    Insert GAP_SHORT or GAP_LONG tokens between events based on time gap.

    Why: a 5-minute silence after adding to cart is a real behavioral
    signal — the user is hesitating. Without gap tokens, the model
    only sees the events and misses the timing information entirely.

    Args:
        events: list of (timestamp_seconds, event_type) sorted ascending

    Returns:
        Same list with gap tokens inserted where inactivity occurred.

    Example:
        Input:  [(100, 'view'), (400, 'addtocart')]   ← 300s gap
        Output: [(100, 'view'), (101, 'GAP_SHORT'), (400, 'addtocart')]
    """
    if len(events) < 2:
        return events

    result = []
    for i, (ts, ev) in enumerate(events):
        if i > 0:
            gap_seconds = ts - events[i - 1][0]
            if cfg.sequence.gap_short_min <= gap_seconds < cfg.sequence.gap_long_min:
                # 2-10 minute gap → hesitation
                result.append((events[i - 1][0] + 1.0, "GAP_SHORT"))
            elif gap_seconds >= cfg.sequence.gap_long_min:
                # 10+ minute gap → likely abandonment
                result.append((events[i - 1][0] + 1.0, "GAP_LONG"))
        result.append((ts, ev))

    return result


# ── Step 3: Build one session sequence ───────────────────────────────────────

def build_session_sequence(session_events: pd.DataFrame) -> Optional[Dict]:
    """
    Convert one session's raw events into a padded integer sequence.

    Steps:
      1. Sort events by timestamp
      2. Drop sessions shorter than min_len
      3. Inject gap tokens between events
      4. Map event names → token IDs
      5. Keep only the LAST max_len tokens (recent events matter most)
      6. Left-pad with zeros to reach exactly max_len length

    Label = 1 if the session contains any purchase event, else 0.

    Returns None if the session is too short.
    Returns a dict with: sequence, seq_len, label, n_views, n_carts, n_purchases
    """
    session_events = session_events.sort_values("timestamp_sec")

    # Build list of (timestamp, event_type) tuples
    raw_events: List[Tuple[float, str]] = [
        (row["timestamp_sec"], row["event_type"])
        for _, row in session_events.iterrows()
    ]

    # Drop sessions that are too short to contain a useful pattern
    if len(raw_events) < cfg.sequence.min_len:
        return None

    # Inject gap tokens between events with inactivity
    events_with_gaps = inject_gap_tokens(raw_events)

    # Map event type strings → integer token IDs
    token_map = {
        "view":      cfg.vocab.view,
        "addtocart": cfg.vocab.add_cart,
        "purchase":  cfg.vocab.purchase,
        "GAP_SHORT": cfg.vocab.gap_short,
        "GAP_LONG":  cfg.vocab.gap_long,
    }
    tokens = [
        token_map.get(ev, cfg.vocab.view)
        for _, ev in events_with_gaps
    ]

    # Compute label and counts from ORIGINAL events (before gap injection)
    original_types = [ev for _, ev in raw_events]
    label       = int("purchase" in original_types)
    n_views     = original_types.count("view")
    n_carts     = original_types.count("addtocart")
    n_purchases = original_types.count("purchase")

    # Keep only the LAST max_len tokens
    # Reason: the most recent behavior is the strongest predictor of purchase
    max_len = cfg.sequence.max_len
    if len(tokens) > max_len:
        tokens = tokens[-max_len:]

    actual_len = len(tokens)

    # Left-pad with PAD tokens (0) so every sequence is exactly max_len long
    # Example: actual_len=6, max_len=64 → 58 zeros + 6 real tokens
    padded = [cfg.vocab.pad] * (max_len - actual_len) + tokens

    return {
        "sequence":    padded,       # List[int], length always = 64
        "seq_len":     actual_len,   # how many real (non-PAD) tokens
        "label":       label,        # 1 = purchased, 0 = did not
        "n_views":     n_views,
        "n_carts":     n_carts,
        "n_purchases": n_purchases,
    }


# ── Step 4: Full pipeline ─────────────────────────────────────────────────────

def build_sequence_dataset(
    max_otto_sessions: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Full pipeline: raw files → sequences.parquet

    Loads Retail Rocket + OTTO (or whichever is available),
    tokenizes every session, saves the result.

    Args:
        max_otto_sessions: integer to limit OTTO during development.
                           Example: 50000 loads only the first 50K sessions.
                           None = full dataset (takes ~45 min).
        save_path: override output path. Defaults to cfg.paths.sequences_parquet.

    Returns:
        DataFrame with columns:
            session_id, source, sequence, seq_len, label,
            n_views, n_carts, n_purchases
    """
    logger.info("=" * 55)
    logger.info("Building session sequence dataset")
    logger.info("=" * 55)

    # Load whichever datasets are available
    frames = []
    for loader, name in [
        (load_retailrocket,                         "Retail Rocket"),
        (lambda: load_otto(max_otto_sessions), "OTTO"),
    ]:
        try:
            frames.append(loader())
        except FileNotFoundError as e:
            logger.warning(str(e))

    if not frames:
        raise RuntimeError(
            "No datasets found.\n"
            "Run: make download-retailrocket  OR  make download-otto"
        )

    # Combine into one events DataFrame
    events_df = pd.concat(frames, ignore_index=True)
    logger.info(
        f"\nCombined: {len(events_df):,} events | "
        f"{events_df['session_id'].nunique():,} total sessions"
    )

    # Tokenize every session
    logger.info("\nTokenizing sessions...")
    records  = []
    skipped  = 0
    groups   = list(events_df.groupby("session_id"))
    total    = len(groups)

    for i, (session_id, group) in enumerate(groups):
        # Progress update every 100K sessions
        if i > 0 and i % 100_000 == 0:
            logger.info(f"  {i:,} / {total:,} sessions processed...")

        result = build_session_sequence(group)

        if result is None:
            skipped += 1
            continue

        records.append({
            "session_id": session_id,
            "source":     group["source"].iloc[0],
            **result,
        })

    df = pd.DataFrame(records)
    conversion_rate = df["label"].mean()

    logger.info(
        f"\n✓ Complete:"
        f"\n  Sessions built:  {len(df):,}"
        f"\n  Skipped (short): {skipped:,}"
        f"\n  Purchased:       {df['label'].sum():,}"
        f"\n  Conversion rate: {conversion_rate:.3f}"
        f"\n  Median seq len:  {df['seq_len'].median():.0f}"
    )

    # Save vocabulary lookup file
    cfg.paths.data_processed.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(cfg.paths.vocab_json, "w") as f:
        _json.dump({
            "id_to_event":  VOCAB,
            "event_to_id":  EVENT_TO_TOKEN,
        }, f, indent=2)
    logger.info(f"Vocab saved → {cfg.paths.vocab_json}")

    # Save sequences
    out = save_path or cfg.paths.sequences_parquet
    df.to_parquet(out, index=False)
    logger.info(f"Sequences saved → {out}")

    return df


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Build SessionScout sequence dataset"
    )
    parser.add_argument(
        "--max-otto-sessions", type=int, default=None,
        help="Limit OTTO sessions for dev. E.g. --max-otto-sessions 50000"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output parquet path"
    )
    args = parser.parse_args()

    df = build_sequence_dataset(
        max_otto_sessions=args.max_otto_sessions,
        save_path=Path(args.output) if args.output else None,
    )

    print(f"\nShape: {df.shape}")
    print(df[["seq_len", "label", "n_views", "n_carts"]].describe().round(2))
