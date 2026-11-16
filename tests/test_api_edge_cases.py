"""
tests/test_api_edge_cases.py — Regression tests for API edge cases.

Each test maps to a confirmed bug found by probing the API (2026-11):
  1. Unknown / negative token IDs crashed the model (500).
  2. The cache key used only sequence length, so a different sequence of the
     same length returned a stale score.
  3. A Redis failure mid-request crashed every call (500).
  4. One bad session failed a whole batch (500).
  5-6. Out-of-distribution input (PURCHASE/PAD, <3 events, gaps first,
       last or doubled) was silently scored.
  7. Sessions longer than 64 events were rejected instead of truncated.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sessionscout.config import cfg  # noqa: E402
from sessionscout.validation import validate_sequence  # noqa: E402

VIEW, CART, GAP_S, GAP_L = 1, 2, 4, 5


def _sequence_sensitive_model():
    """Mock model whose score depends on the tokens, so caching bugs show up."""
    model = MagicMock()
    model.side_effect = lambda ids, mask: ids.float().sum(dim=1) / 100.0 - 0.5
    return model


class FakeRedis(dict):
    def get(self, key):
        return dict.get(self, key)

    def setex(self, key, ttl, value):
        self[key] = value


class BrokenRedis:
    def get(self, key):
        raise ConnectionError("redis is down")

    def setex(self, key, ttl, value):
        raise ConnectionError("redis is down")


def _client(redis):
    from fastapi.testclient import TestClient
    from sessionscout.api.main import app

    patches = (
        patch(
            "sessionscout.api.main.load_model",
            return_value=_sequence_sensitive_model(),
        ),
        patch("sessionscout.api.main.get_redis_client", return_value=redis),
    )
    return patches, TestClient(app)


@pytest.fixture()
def client_with_cache():
    patches, tc = _client(FakeRedis())
    with patches[0], patches[1], tc as c:
        yield c


@pytest.fixture()
def client_broken_redis():
    patches, tc = _client(BrokenRedis())
    with patches[0], patches[1], tc as c:
        yield c


def _predict(client, sequence, session_id="s1"):
    return client.post(
        "/api/v1/predict", json={"session_id": session_id, "sequence": sequence}
    )


# ── 1. Unknown token IDs → 422, never a crash ────────────────────────────────


@pytest.mark.parametrize("bad", [99, -1, 6])
def test_unknown_token_is_rejected_not_crashed(client_with_cache, bad):
    resp = _predict(client_with_cache, [VIEW, CART, bad])
    assert resp.status_code == 422
    assert "Unknown token" in resp.text


# ── 2. Cache keys on the full sequence ───────────────────────────────────────


def test_same_session_same_length_different_events_not_served_from_cache(
    client_with_cache,
):
    first = _predict(client_with_cache, [VIEW, VIEW, VIEW], session_id="X").json()
    second = _predict(client_with_cache, [CART, CART, CART], session_id="X").json()
    assert second["cached"] is False
    assert second["conversion_probability"] != first["conversion_probability"]


def test_identical_request_is_served_from_cache(client_with_cache):
    _predict(client_with_cache, [VIEW, CART, VIEW], session_id="Y")
    again = _predict(client_with_cache, [VIEW, CART, VIEW], session_id="Y").json()
    assert again["cached"] is True


# ── 3. Redis failure degrades to no-cache ────────────────────────────────────


def test_redis_failure_still_returns_a_score(client_broken_redis):
    resp = _predict(client_broken_redis, [VIEW, CART, VIEW])
    assert resp.status_code == 200
    assert resp.json()["cached"] is False


# ── 4. One bad session doesn't fail the batch ────────────────────────────────


def test_batch_scores_valid_sessions_and_reports_invalid_ones(client_with_cache):
    resp = client_with_cache.post(
        "/api/v1/batch",
        json={
            "sessions": [
                {"session_id": "good", "sequence": [VIEW, CART, VIEW]},
                {"session_id": "bad", "sequence": [VIEW, CART, 99]},
                {"session_id": "short", "sequence": [VIEW]},
            ]
        },
    )
    assert resp.status_code == 200
    good, bad, short = resp.json()["results"]
    assert good["conversion_probability"] is not None and good["error"] is None
    assert bad["conversion_probability"] is None and "Unknown token" in bad["error"]
    assert short["conversion_probability"] is None and "at least 3" in short["error"]


def test_batch_with_only_invalid_sessions_returns_200(client_with_cache):
    resp = client_with_cache.post(
        "/api/v1/batch",
        json={"sessions": [{"session_id": "a", "sequence": [GAP_L, VIEW, CART]}]},
    )
    assert resp.status_code == 200
    assert resp.json()["results"][0]["error"]


# ── 5-6. Out-of-distribution sequences → 422 with a reason ───────────────────


@pytest.mark.parametrize(
    "sequence, reason",
    [
        ([VIEW, CART, 3], "PURCHASE"),
        ([VIEW, 0, CART], "PAD"),
        ([VIEW], "at least 3"),
        ([VIEW, CART], "at least 3"),
        ([GAP_S, VIEW, CART], "start with a gap"),
        ([VIEW, CART, GAP_L], "end with a gap"),
        ([VIEW, GAP_S, GAP_L, CART], "Two gap tokens"),
    ],
)
def test_out_of_distribution_sequence_rejected(client_with_cache, sequence, reason):
    resp = _predict(client_with_cache, sequence)
    assert resp.status_code == 422
    assert reason in resp.text


@pytest.mark.parametrize(
    "sequence",
    [
        [VIEW, VIEW, VIEW],
        [VIEW, GAP_L, CART],
        [VIEW, CART, GAP_S, VIEW, GAP_L, VIEW],
    ],
)
def test_valid_sequences_pass_validation(sequence):
    assert validate_sequence(sequence) is None


# ── 7. Long sessions are truncated, not rejected ─────────────────────────────


def test_long_session_scored_on_last_64_events(client_with_cache):
    long_seq = [VIEW] * 200 + [CART] * cfg.sequence.max_len
    tail_only = [CART] * cfg.sequence.max_len
    long_resp = _predict(client_with_cache, long_seq, session_id="L1")
    tail_resp = _predict(client_with_cache, tail_only, session_id="L2")
    assert long_resp.status_code == 200
    assert (
        long_resp.json()["conversion_probability"]
        == tail_resp.json()["conversion_probability"]
    )
