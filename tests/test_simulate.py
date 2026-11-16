"""
tests/test_simulate.py — The live-session simulator must respect API rules.

Regression: after the API started rejecting out-of-distribution sequences
(fewer than 3 events, ending on a gap), the simulator still called the API
after every event and crashed on the first 422.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import simulate_session  # noqa: E402
from sessionscout.validation import validate_sequence  # noqa: E402


def test_simulator_only_sends_sequences_the_api_accepts():
    sent = []

    def fake_call(session_id, sequence):
        sent.append(list(sequence))
        return {"conversion_probability": 0.5}

    with (
        patch.object(simulate_session, "call_predict", side_effect=fake_call),
        patch.object(simulate_session.time, "sleep", MagicMock()),
    ):
        simulate_session.simulate("s", [1, 1, 2, 5, 1, 2, 4, 1], label=1, delay=0)

    assert sent, "the simulator should score at least once"
    assert all(validate_sequence(seq) is None for seq in sent)
    assert [1, 1, 2, 5] not in sent  # never scored on a trailing pause
