"""
api/routes/batch.py — Batch session scoring.

POST /api/v1/batch
  Input:  list of session_id + sequence pairs (up to 1000)
  Output: list of conversion probabilities

Why batch matters:
  Nightly jobs score all active sessions from the previous day
  so the marketing team can prioritise outreach in the morning.
  Scoring one at a time would take minutes. Batching runs them
  all through the model in parallel — much faster.

  We process in mini-batches of 256 to avoid running out of
  memory if someone sends 10,000 sessions at once.
"""

import logging
import time
from typing import List

import torch
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from sessionscout.config import cfg
from sessionscout.api.routes.predict import pad_sequence

logger = logging.getLogger(__name__)
router = APIRouter()


class BatchItem(BaseModel):
    session_id: str
    sequence: List[int] = Field(min_length=1, max_length=cfg.sequence.max_len)


class BatchRequest(BaseModel):
    sessions: List[BatchItem] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of sessions to score. Maximum 1000 per request.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "sessions": [
                    {"session_id": "user_001", "sequence": [1, 1, 2, 4, 1]},
                    {"session_id": "user_002", "sequence": [1, 2, 1]},
                ]
            }
        }
    }


class BatchResultItem(BaseModel):
    session_id: str
    conversion_probability: float


class BatchResponse(BaseModel):
    results: List[BatchResultItem]
    total:   int
    latency_ms: float


@router.post("/batch", response_model=BatchResponse)
async def batch_predict(req: BatchRequest, request: Request):
    """
    Score multiple sessions in one call.

    Processes sessions in mini-batches of 256 for memory efficiency.
    Returns results in the same order as the input.
    """
    t_start = time.perf_counter()
    model   = request.app.state.model

    # Pad all sequences to max_len
    padded_seqs = [pad_sequence(item.sequence) for item in req.sessions]
    session_ids = [item.session_id for item in req.sessions]

    all_probs = []
    mini_batch_size = 256

    for i in range(0, len(padded_seqs), mini_batch_size):
        batch_seqs = padded_seqs[i : i + mini_batch_size]
        ids  = torch.tensor(batch_seqs, dtype=torch.long)
        mask = (ids != cfg.vocab.pad).float()

        with torch.no_grad():
            logits = model(ids, mask)
            probs  = torch.sigmoid(logits).tolist()

        all_probs.extend(probs)

    results = [
        BatchResultItem(
            session_id=sid,
            conversion_probability=round(prob, 4),
        )
        for sid, prob in zip(session_ids, all_probs)
    ]

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    logger.info(
        f"batch | n={len(results)} sessions | {latency_ms:.1f}ms"
    )

    return BatchResponse(
        results=results,
        total=len(results),
        latency_ms=latency_ms,
    )
