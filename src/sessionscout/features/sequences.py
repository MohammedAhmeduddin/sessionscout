"""
features/sequences.py — Raw events → padded token sequences.

Key design decision:
  PURCHASE tokens are EXCLUDED from sequences.
  The label (did this session convert?) comes from purchase events,
  but the sequence only contains browsing behavior — views, carts,
  and gaps. This matches real inference: we score a session BEFORE
  the purchase decision is made, not after.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from sessionscout.config import cfg, VOCAB, EVENT_TO_TOKEN

logger = logging.getLogger(__name__)


def load_retailrocket() -> pd.DataFrame:
    path = cfg.paths.rr_events
    if not path.exists():
        raise FileNotFoundError(
            f"\nRetail Rocket not found: {path}"
            "\nRun: kaggle datasets download "
            "-d retailrocket/ecommerce-dataset -p data/raw/retailrocket --unzip"
        )
    logger.info("Loading Retail Rocket...")
    df = pd.read_csv(path)
    df = df.rename(columns={"visitorid": "session_id", "event": "event_type"})
    df["timestamp_sec"] = df["timestamp"] / 1000.0
    df["source"] = "retailrocket"
    df["session_id"] = "rr_" + df["session_id"].astype(str)
    df["event_type"] = (
        df["event_type"].str.lower().str.strip().replace({"transaction": "purchase"})
    )
    df = df[["session_id", "timestamp_sec", "event_type", "source"]]
    logger.info(
        f"  RR: {len(df):,} events | "
        f"{df['session_id'].nunique():,} sessions | "
        f"types: {df['event_type'].value_counts().to_dict()}"
    )
    return df


def load_otto(max_sessions: Optional[int] = None) -> pd.DataFrame:
    path = cfg.paths.otto_train
    if not path.exists():
        raise FileNotFoundError(
            f"\nOTTO not found: {path}"
            "\nRun: kaggle competitions download "
            "-c otto-recommender-system -p data/raw/otto --unzip"
        )
    logger.info(
        f"Loading OTTO"
        f"{f' (first {max_sessions:,} sessions)' if max_sessions else ' (full dataset)'}..."
    )
    type_map = {"clicks": "view", "carts": "addtocart", "orders": "purchase"}
    records, n = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = f"otto_{obj['session']}"
            for ev in obj["events"]:
                records.append(
                    {
                        "session_id": sid,
                        "timestamp_sec": float(ev["ts"]),
                        "event_type": type_map.get(ev["type"], ev["type"]),
                        "source": "otto",
                    }
                )
            n += 1
            if max_sessions and n >= max_sessions:
                break
    df = pd.DataFrame(records)
    logger.info(
        f"  OTTO: {len(df):,} events | "
        f"{df['session_id'].nunique():,} sessions | "
        f"types: {df['event_type'].value_counts().to_dict()}"
    )
    return df


def inject_gap_tokens(events: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """
    Insert GAP_SHORT or GAP_LONG tokens between events with inactivity.

    A 5-minute pause after adding to cart is a real hesitation signal.
    Without gap tokens the model only sees events and misses timing.

    GAP_SHORT = 2-10 min inactivity
    GAP_LONG  = 10+ min inactivity
    """
    if len(events) < 2:
        return events
    result = []
    for i, (ts, ev) in enumerate(events):
        if i > 0:
            gap = ts - events[i - 1][0]
            if cfg.sequence.gap_short_min <= gap < cfg.sequence.gap_long_min:
                result.append((events[i - 1][0] + 1.0, "GAP_SHORT"))
            elif gap >= cfg.sequence.gap_long_min:
                result.append((events[i - 1][0] + 1.0, "GAP_LONG"))
        result.append((ts, ev))
    return result


def build_session_sequence(session_events: pd.DataFrame) -> Optional[Dict]:
    """
    Convert one session into a padded integer sequence.

    PURCHASE tokens are stripped from the sequence — the label is
    derived from whether a purchase occurred, but the model only
    sees the browsing behavior that preceded it.

    Returns None if session is too short after stripping purchases.
    """
    session_events = session_events.sort_values("timestamp_sec")

    raw_events: List[Tuple[float, str]] = [
        (row["timestamp_sec"], row["event_type"])
        for _, row in session_events.iterrows()
    ]

    if len(raw_events) < cfg.sequence.min_len:
        return None

    # Label and counts from original events (before any filtering)
    original_types = [ev for _, ev in raw_events]
    label = int("purchase" in original_types)
    n_views = original_types.count("view")
    n_carts = original_types.count("addtocart")

    # Inject gap tokens
    events_with_gaps = inject_gap_tokens(raw_events)

    # Convert to token IDs — PURCHASE intentionally excluded
    # At inference time the model scores sessions before purchase occurs
    token_map = {
        "view": cfg.vocab.view,
        "addtocart": cfg.vocab.add_cart,
        "GAP_SHORT": cfg.vocab.gap_short,
        "GAP_LONG": cfg.vocab.gap_long,
    }
    tokens = [token_map[ev] for _, ev in events_with_gaps if ev in token_map]

    # After removing purchase tokens, re-check min_len
    if len(tokens) < cfg.sequence.min_len:
        return None

    # Keep last max_len tokens (most recent = most predictive)
    max_len = cfg.sequence.max_len
    if len(tokens) > max_len:
        tokens = tokens[-max_len:]

    actual_len = len(tokens)

    # Left-pad with zeros
    padded = [cfg.vocab.pad] * (max_len - actual_len) + tokens

    return {
        "sequence": padded,
        "seq_len": actual_len,
        "label": label,
        "n_views": n_views,
        "n_carts": n_carts,
    }


def build_sequence_dataset(
    max_otto_sessions: Optional[int] = None,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Full pipeline: raw files → sequences.parquet"""
    logger.info("=" * 55)
    logger.info("Building session sequence dataset")
    logger.info("=" * 55)

    frames = []
    for loader, name in [
        (load_retailrocket, "Retail Rocket"),
        (lambda: load_otto(max_otto_sessions), "OTTO"),
    ]:
        try:
            frames.append(loader())
        except FileNotFoundError as e:
            logger.warning(str(e))

    if not frames:
        raise RuntimeError("No datasets found. Run: make download-retailrocket")

    events_df = pd.concat(frames, ignore_index=True)
    logger.info(
        f"\nCombined: {len(events_df):,} events | "
        f"{events_df['session_id'].nunique():,} sessions"
    )

    logger.info("\nTokenizing sessions...")
    records, skipped = [], 0
    groups = list(events_df.groupby("session_id"))
    total = len(groups)

    for i, (session_id, group) in enumerate(groups):
        if i > 0 and i % 100_000 == 0:
            logger.info(f"  {i:,} / {total:,}")
        result = build_session_sequence(group)
        if result is None:
            skipped += 1
            continue
        records.append(
            {
                "session_id": session_id,
                "source": group["source"].iloc[0],
                **result,
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        f"\n✓ Done:"
        f"\n  Sessions:        {len(df):,}"
        f"\n  Skipped:         {skipped:,}"
        f"\n  Purchased:       {df['label'].sum():,}"
        f"\n  Conversion rate: {df['label'].mean():.3f}"
        f"\n  Median seq len:  {df['seq_len'].median():.0f}"
    )

    cfg.paths.data_processed.mkdir(parents=True, exist_ok=True)
    import json as _json

    with open(cfg.paths.vocab_json, "w") as f:
        _json.dump({"id_to_event": VOCAB, "event_to_id": EVENT_TO_TOKEN}, f, indent=2)

    out = save_path or cfg.paths.sequences_parquet
    df.to_parquet(out, index=False)
    logger.info(f"Saved → {out}")
    return df


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-otto-sessions", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    df = build_sequence_dataset(
        max_otto_sessions=args.max_otto_sessions,
        save_path=Path(args.output) if args.output else None,
    )
    print(f"\nShape: {df.shape}")
    print(df[["seq_len", "label", "n_views", "n_carts"]].describe().round(2))
