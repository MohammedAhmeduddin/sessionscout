"""
api/routes/predict.py — Real-time single session scoring.

POST /api/v1/predict
  Input:  session_id + sequence of up to 64 token IDs
  Output: conversion probability + top signals

Redis caching:
  Cache key = "{session_id}:{seq_len}"
  Why seq_len in the key? Because the same session grows as the
  user browses. session_123:5 and session_123:8 are different
  states of the same session — both get cached separately.
  TTL = 5 minutes (cfg.api.cache_ttl_seconds)
"""

import json
import logging
import time
from typing import List

import torch
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from sessionscout.config import cfg, VOCAB

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    sequence: List[int] = Field(
        ...,
        description=(
            "List of token IDs representing the session's event sequence. "
            "Length must be between 1 and 64. "
            "Tokens: PAD=0, VIEW=1, ADD_CART=2, PURCHASE=3, "
            "GAP_SHORT=4, GAP_LONG=5"
        ),
        min_length=1,
        max_length=cfg.sequence.max_len,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "user_4421_session_1",
                "sequence": [1, 1, 2, 4, 1, 1],
            }
        }
    }


class PredictResponse(BaseModel):
    session_id: str
    conversion_probability: float = Field(
        ...,
        description="Predicted probability of purchase (0.0 to 1.0)",
    )
    top_signals: List[str] = Field(
        ...,
        description="The most informative events in this session",
    )
    cached: bool = Field(
        ...,
        description="True if this result was served from Redis cache",
    )
    latency_ms: float = Field(
        ...,
        description="Inference time in milliseconds",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def pad_sequence(sequence: List[int]) -> List[int]:
    """Left-pad a sequence to max_len with PAD tokens."""
    max_len = cfg.sequence.max_len
    if len(sequence) >= max_len:
        return sequence[-max_len:]
    return [cfg.vocab.pad] * (max_len - len(sequence)) + sequence


def get_top_signals(sequence: List[int], top_k: int = 3) -> List[str]:
    """
    Identify the most informative non-PAD events in the sequence.
    Returns human-readable event names.
    """
    # Count non-PAD token types
    counts = {}
    for tok in sequence:
        if tok != cfg.vocab.pad:
            name = VOCAB.get(tok, f"token_{tok}")
            counts[name] = counts.get(name, 0) + 1

    # Sort by count, return top_k
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [f"{name}×{count}" for name, count in ranked[:top_k]]


def run_inference(model, sequence: List[int]) -> float:
    """Run model inference and return conversion probability."""
    padded = pad_sequence(sequence)
    ids    = torch.tensor([padded], dtype=torch.long)
    mask   = (ids != cfg.vocab.pad).float()

    with torch.no_grad():
        logit = model(ids, mask)
        prob  = torch.sigmoid(logit).item()

    return round(prob, 4)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    """
    Score a single browsing session.

    The sequence should contain events up to the current moment
    in the session. Call this endpoint again as new events occur
    to get updated probability scores.

    The model was trained to score sessions BEFORE any purchase
    event — do not include PURCHASE tokens (3) in the input.
    """
    t_start = time.perf_counter()

    model = request.app.state.model
    redis = request.app.state.redis

    cache_key = f"predict:{req.session_id}:{len(req.sequence)}"

    # ── Check Redis cache ──────────────────────────────────────────
    if redis is not None:
        cached_result = redis.get(cache_key)
        if cached_result is not None:
            data = json.loads(cached_result)
            data["cached"]     = True
            data["latency_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
            return PredictResponse(**data)

    # ── Run inference ──────────────────────────────────────────────
    prob        = run_inference(model, req.sequence)
    top_signals = get_top_signals(req.sequence)
    latency_ms  = round((time.perf_counter() - t_start) * 1000, 2)

    result = {
        "session_id":             req.session_id,
        "conversion_probability": prob,
        "top_signals":            top_signals,
        "cached":                 False,
        "latency_ms":             latency_ms,
    }

    # ── Store in Redis ─────────────────────────────────────────────
    if redis is not None:
        redis.setex(
            cache_key,
            cfg.api.cache_ttl_seconds,
            json.dumps({k: v for k, v in result.items()
                        if k != "latency_ms"}),
        )

    logger.info(
        f"predict | session={req.session_id} | "
        f"prob={prob:.4f} | {latency_ms:.1f}ms"
    )

    return PredictResponse(**result)
