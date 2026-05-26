"""
scripts/simulate_session.py — Real-time session replay demo.

What this does:
  Replays a real session from the test set event by event,
  calling /predict after each new event, and printing the
  conversion probability updating live in the terminal.

  This makes the demo feel like a real production system
  scoring an active user session as it unfolds.

Why this matters:
  A static notebook showing AUC numbers does not feel real.
  This script shows the model working in real time — probability
  climbing as the user adds to cart, dipping during gaps,
  spiking when they return to look again.

Usage:
  # Make sure the API is running first:
  uvicorn sessionscout.api.main:app --port 8000

  # Then in another terminal:
  python scripts/simulate_session.py
  python scripts/simulate_session.py --session-id otto_10014
  python scripts/simulate_session.py --converting    # show a buyer
  python scripts/simulate_session.py --abandoning    # show a browser

Record for README:
  pip install asciinema
  asciinema rec demo.cast
  python scripts/simulate_session.py --converting
  # Ctrl+D to stop recording
  asciinema upload demo.cast
"""

import argparse
import sys
import time
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sessionscout.config import cfg

API_URL = "http://localhost:8000/api/v1/predict"

# Token display names for terminal output
TOKEN_DISPLAY = {
    cfg.vocab.pad: "PAD",
    cfg.vocab.view: "👁  VIEW     ",
    cfg.vocab.add_cart: "🛒 ADD_CART ",
    cfg.vocab.purchase: "💰 PURCHASE ",
    cfg.vocab.gap_short: "⏸  GAP_SHORT",
    cfg.vocab.gap_long: "⏳ GAP_LONG ",
}

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"


def prob_to_bar(prob: float, width: int = 30) -> str:
    """Convert probability to a colored ASCII bar."""
    filled = int(prob * width)
    empty = width - filled

    if prob >= 0.65:
        color = GREEN
    elif prob >= 0.35:
        color = YELLOW
    else:
        color = RED

    bar = color + "█" * filled + RESET + "░" * empty
    return bar


def prob_to_label(prob: float) -> str:
    """Convert probability to a business action label."""
    if prob >= 0.75:
        return f"{GREEN}{BOLD}HIGH — send discount now{RESET}"
    elif prob >= 0.50:
        return f"{YELLOW}{BOLD}MEDIUM — show reminder{RESET}"
    elif prob >= 0.30:
        return f"{YELLOW}LOW-MED — watch session{RESET}"
    else:
        return f"{RED}LOW — do nothing{RESET}"


def call_predict(session_id: str, sequence: list) -> dict:
    """Call the /predict endpoint and return the response."""
    try:
        resp = requests.post(
            API_URL,
            json={"session_id": session_id, "sequence": sequence},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(f"\n{RED}Cannot connect to API at {API_URL}{RESET}")
        print("Start the API first:")
        print(f"  {CYAN}uvicorn sessionscout.api.main:app --port 8000{RESET}\n")
        sys.exit(1)


def load_session(
    session_id: str = None, converting: bool = False, abandoning: bool = False
) -> tuple:
    """
    Load a session from sequences.parquet.

    Returns (session_id, sequence_tokens, label)
    where sequence_tokens excludes PAD tokens (real events only).
    """
    df = pd.read_parquet(cfg.paths.sequences_parquet)

    if session_id:
        row = df[df["session_id"] == session_id]
        if len(row) == 0:
            print(f"{RED}Session '{session_id}' not found.{RESET}")
            print(
                f"Available sessions (sample): " f"{df['session_id'].head(5).tolist()}"
            )
            sys.exit(1)
        row = row.iloc[0]
    elif converting:
        # Find an interesting converting session
        candidates = df[
            (df["label"] == 1) & (df["n_carts"] > 0) & (df["seq_len"].between(5, 15))
        ]
        row = (
            candidates.iloc[0] if len(candidates) > 0 else df[df["label"] == 1].iloc[0]
        )
    elif abandoning:
        # Find an interesting non-converting session with a cart
        candidates = df[
            (df["label"] == 0) & (df["n_carts"] > 0) & (df["seq_len"].between(5, 15))
        ]
        row = (
            candidates.iloc[0] if len(candidates) > 0 else df[df["label"] == 0].iloc[0]
        )
    else:
        # Default: pick a converting session
        candidates = df[(df["label"] == 1) & (df["seq_len"].between(5, 15))]
        row = candidates.iloc[0]

    # Extract non-PAD tokens only (real events)
    full_seq = row["sequence"]
    real_tokens = [t for t in full_seq if t != cfg.vocab.pad]

    return str(row["session_id"]), [int(t) for t in real_tokens], int(row["label"])


def simulate(session_id: str, tokens: list, label: int, delay: float = 0.8):
    """
    Replay a session event by event with live probability updates.
    """
    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  SessionScout — Live Session Scoring{RESET}")
    print(f"{'═' * 60}")
    print(f"  Session:  {CYAN}{session_id}{RESET}")
    print(f"  Events:   {len(tokens)} total")
    actual = "PURCHASED ✓" if label == 1 else "ABANDONED ✗"
    color = GREEN if label == 1 else RED
    print(f"  Outcome:  {color}{actual}{RESET}  (hidden from model)")
    print(f"{'═' * 60}\n")

    time.sleep(0.5)

    current_sequence = []

    for i, token in enumerate(tokens):
        current_sequence.append(token)
        event_name = TOKEN_DISPLAY.get(token, f"tok_{token}")

        # Call the API
        result = call_predict(session_id, current_sequence)
        prob = result["conversion_probability"]

        # Clear line and print update
        bar = prob_to_bar(prob)

        print(
            f"  Event {i+1:>2}/{len(tokens)}  "
            f"{BLUE}{event_name}{RESET}  "
            f"│  {bar}  "
            f"{BOLD}{prob:.1%}{RESET}"
        )

        # Show action recommendation at decision points
        if token == cfg.vocab.add_cart:
            print(f"           {YELLOW}↑ Cart detected — probability spike{RESET}")
        elif token in (cfg.vocab.gap_short, cfg.vocab.gap_long):
            gap = "2-10 min" if token == cfg.vocab.gap_short else "10+ min"
            print(f"           {YELLOW}⏸ Inactivity ({gap}) — hesitation signal{RESET}")
        elif (
            token == cfg.vocab.view
            and i > 0
            and tokens[i - 1] in (cfg.vocab.gap_short, cfg.vocab.gap_long)
        ):
            print(f"           {GREEN}↩ Returned after gap — strong buy signal{RESET}")

        time.sleep(delay)

    # Final verdict
    print(f"\n  {'─' * 56}")
    final_prob = call_predict(session_id, current_sequence)["conversion_probability"]
    print(f"\n  {BOLD}Final score:  {final_prob:.1%}{RESET}")
    print(f"  Recommendation: {prob_to_label(final_prob)}")
    print(f"  Actual outcome: {color}{actual}{RESET}")
    print(f"\n{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Simulate a live session scoring demo")
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument(
        "--converting",
        action="store_true",
        help="Show a session that ended in purchase",
    )
    parser.add_argument(
        "--abandoning", action="store_true", help="Show a session that abandoned"
    )
    parser.add_argument(
        "--delay", type=float, default=0.8, help="Seconds between events (default 0.8)"
    )
    args = parser.parse_args()

    session_id, tokens, label = load_session(
        session_id=args.session_id,
        converting=args.converting,
        abandoning=args.abandoning,
    )

    simulate(session_id, tokens, label, delay=args.delay)


if __name__ == "__main__":
    main()
