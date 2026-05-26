"""
tests/test_api.py — FastAPI endpoint tests.

Uses TestClient so no real server needed.
Tests request validation, response shapes, and edge cases.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """
    Create a FastAPI test client with a mock model.
    Uses unittest.mock so no trained weights needed.
    """
    from unittest.mock import MagicMock, patch
    import torch
    from fastapi.testclient import TestClient
    from sessionscout.api.main import app

    mock_model = MagicMock()
    # return_value must be a single-element tensor per call
    mock_model.side_effect = lambda ids, mask: torch.zeros(ids.shape[0])

    with (
        patch("sessionscout.api.main.load_model", return_value=mock_model),
        patch("sessionscout.api.main.get_redis_client", return_value=None),
    ):
        with TestClient(app) as c:
            yield c


# ── Health endpoint ────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model" in data
        assert "vocab_size" in data
        assert "max_seq_len" in data

    def test_health_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_vocab_size_correct(self, client):
        from sessionscout.config import cfg

        data = client.get("/health").json()
        assert data["vocab_size"] == cfg.vocab.size


# ── Predict endpoint ──────────────────────────────────────────────────────────


class TestPredict:
    def test_predict_returns_200(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "session_id": "test_001",
                "sequence": [1, 1, 2, 4, 1],
            },
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "session_id": "test_002",
                "sequence": [1, 2, 1],
            },
        )
        data = resp.json()
        assert "session_id" in data
        assert "conversion_probability" in data
        assert "top_signals" in data
        assert "cached" in data
        assert "latency_ms" in data

    def test_predict_session_id_echoed(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "session_id": "my_session_xyz",
                "sequence": [1, 1, 1],
            },
        )
        assert resp.json()["session_id"] == "my_session_xyz"

    def test_predict_probability_is_float(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "session_id": "test_003",
                "sequence": [1, 2, 4, 1],
            },
        )
        prob = resp.json()["conversion_probability"]
        assert isinstance(prob, float)

    def test_predict_rejects_empty_sequence(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "session_id": "bad_session",
                "sequence": [],
            },
        )
        assert resp.status_code == 422  # Validation error

    def test_predict_rejects_sequence_too_long(self, client):
        from sessionscout.config import cfg

        resp = client.post(
            "/api/v1/predict",
            json={
                "session_id": "long_session",
                "sequence": [1] * (cfg.sequence.max_len + 1),
            },
        )
        assert resp.status_code == 422

    def test_predict_rejects_missing_session_id(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "sequence": [1, 2, 1],
            },
        )
        assert resp.status_code == 422

    def test_predict_top_signals_is_list(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "session_id": "test_004",
                "sequence": [1, 1, 2],
            },
        )
        assert isinstance(resp.json()["top_signals"], list)

    def test_predict_latency_is_positive(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "session_id": "test_005",
                "sequence": [1, 2, 5, 1],
            },
        )
        assert resp.json()["latency_ms"] > 0

    def test_predict_not_cached_first_call(self, client):
        resp = client.post(
            "/api/v1/predict",
            json={
                "session_id": "unique_session_no_cache",
                "sequence": [1, 1, 2],
            },
        )
        # Without Redis, cached should always be False
        assert resp.json()["cached"] is False


# ── Batch endpoint ────────────────────────────────────────────────────────────


class TestBatch:
    def test_batch_returns_200(self, client):
        resp = client.post(
            "/api/v1/batch",
            json={
                "sessions": [
                    {"session_id": "a", "sequence": [1, 1, 2]},
                    {"session_id": "b", "sequence": [1, 1, 1]},
                ]
            },
        )
        assert resp.status_code == 200

    def test_batch_response_schema(self, client):
        resp = client.post(
            "/api/v1/batch",
            json={"sessions": [{"session_id": "x", "sequence": [1, 2]}]},
        )
        data = resp.json()
        assert "results" in data
        assert "total" in data
        assert "latency_ms" in data

    def test_batch_total_matches_input(self, client):
        sessions = [{"session_id": f"s{i}", "sequence": [1, 2, 1]} for i in range(5)]
        resp = client.post("/api/v1/batch", json={"sessions": sessions})
        assert resp.json()["total"] == 5

    def test_batch_results_order_preserved(self, client):
        sessions = [
            {"session_id": "first", "sequence": [1, 1, 2]},
            {"session_id": "second", "sequence": [1, 1, 1]},
            {"session_id": "third", "sequence": [2, 1, 1]},
        ]
        resp = client.post("/api/v1/batch", json={"sessions": sessions})
        results = resp.json()["results"]
        assert results[0]["session_id"] == "first"
        assert results[1]["session_id"] == "second"
        assert results[2]["session_id"] == "third"

    def test_batch_each_result_has_probability(self, client):
        resp = client.post(
            "/api/v1/batch",
            json={"sessions": [{"session_id": "t", "sequence": [1, 2]}]},
        )
        result = resp.json()["results"][0]
        assert "session_id" in result
        assert "conversion_probability" in result

    def test_batch_rejects_empty_sessions(self, client):
        resp = client.post("/api/v1/batch", json={"sessions": []})
        assert resp.status_code == 422
