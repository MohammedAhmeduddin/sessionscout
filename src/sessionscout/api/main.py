"""
api/main.py — FastAPI application.

What this does:
  Serves the best trained model (LSTM) as a REST API.
  The model loads ONCE at startup and stays in memory.
  Every request reuses the same loaded model — not reloading
  on each call, which would make it 10-100x slower.

Two endpoints:
  POST /api/v1/predict  — score one session in real time
  POST /api/v1/batch    — score multiple sessions at once

Redis caching:
  If the same session is scored twice, the second call returns
  the cached result instantly without running inference.
  Cache key = session_id + sequence length.
  This is a real production pattern — active sessions hit the
  API on every page load, not just once.
"""

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sessionscout.api.routes.predict import router as predict_router
from sessionscout.api.routes.batch import router as batch_router
from sessionscout.config import cfg

logger = logging.getLogger(__name__)


def load_model():
    """Load the best LSTM model from disk into memory."""
    from sessionscout.model.lstm import SessionLSTM

    model_path = cfg.paths.models_dir / "lstm_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"lstm_best.pt not found at {model_path}\n"
            "Run: python -m sessionscout.model.train --model lstm"
        )

    model = SessionLSTM()
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Model loaded from {model_path}")
    return model


def get_redis_client():
    """Connect to Redis for prediction caching. Returns None if unavailable."""
    try:
        import redis
        client = redis.from_url(cfg.api.redis_url, decode_responses=True)
        client.ping()
        logger.info(f"Redis connected at {cfg.api.redis_url}")
        return client
    except Exception as e:
        logger.warning(
            f"Redis not available ({e}). "
            "Running without cache — predictions will not be cached."
        )
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context: runs on startup and shutdown.

    Startup:  load model + connect Redis → store on app.state
    Shutdown: nothing to clean up (model is in-process)
    """
    logger.info("SessionScout API starting up...")
    app.state.model = load_model()
    app.state.redis = get_redis_client()
    logger.info("API ready.")
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="SessionScout",
    description=(
        "Real-time e-commerce session conversion scoring. "
        "POST a sequence of browsing events, get back a "
        "purchase probability score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1")
app.include_router(batch_router,   prefix="/api/v1")


@app.get("/health")
async def health():
    """Health check — confirms model is loaded."""
    return {
        "status":      "healthy",
        "model":       "lstm",
        "redis":       app.state.redis is not None,
        "vocab_size":  cfg.vocab.size,
        "max_seq_len": cfg.sequence.max_len,
    }
