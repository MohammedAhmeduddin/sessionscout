<div align="center">

# SessionScout

**Real-time purchase-intent scoring for e-commerce sessions**

Scores a live browsing session in **27 ms** so a store can intervene only with shoppers who are likely to buy but hesitating.

[![CI](https://github.com/MohammedAhmeduddin/sessionscout/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammedAhmeduddin/sessionscout/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM%20%7C%20Transformer-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Redis-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/tests-125%20passed-22C55E)
![Coverage](https://img.shields.io/badge/coverage-93%25-22C55E)
![Test AUC](https://img.shields.io/badge/LSTM%20test%20AUC-0.958-7C3AED)

[**Live demo**](https://huggingface.co/spaces/AhmeduddinMohammed/sessionscout) · [**Portfolio**](https://ahmed-portfolio-blue.vercel.app) · [**LinkedIn**](https://www.linkedin.com/in/mohammed-ahmeduddin/)

</div>

---

## Overview

Most e-commerce sessions end without a purchase: only **7.6%** of the 241,773 sessions in this project convert. Retargeting tools usually send the same discount to everyone who leaves, which wastes money on two groups:

- shoppers who were going to buy anyway, and
- shoppers who were never going to buy.

The group worth targeting is the **hesitating buyer**: someone who added to cart, paused, and came back. SessionScout reads the *sequence* of a session (views, add-to-carts and pauses) and scores purchase intent after every action, so an intervention fires only where it can change the outcome.

## Results

| Metric | Value |
|---|---|
| Best model | **Bidirectional LSTM: 0.9563 val AUC, 0.9582 test AUC** |
| Lift over best tabular model | **+0.019 AUC** over XGBoost |
| Inference latency | **27 ms** per session (CPU), **6 ms** for a 2-session batch |
| Training data | **241,773 sessions** (Retail Rocket + OTTO) |
| Quality | **125 tests, 93% coverage**, CI on Python 3.10 and 3.11 |

### Model comparison

| Model | Val AUC | Test AUC | Notes |
|---|---|---|---|
| Logistic Regression | 0.9357 | 0.9393 | Tabular baseline |
| XGBoost | 0.9371 | 0.9386 | 500 trees, depth 6 |
| **LSTM** | **0.9563** | **0.9582** | Bidirectional, 2 layers, 170K parameters |
| Transformer | 0.9555 | 0.9571 | 2 encoder layers, 4 heads, 69K parameters |

**Findings**

- Sequence models beat tabular models by about 0.02 AUC: the order and timing of events carry signal that counts alone miss.
- The LSTM edges the Transformer only slightly. With a median session length of 6 events, long-range attention has little to add; the Transformer matches the LSTM with 2.5× fewer parameters.

---

## Engineering note: a label leak, found and fixed

While testing the live demo, sessions ending in a pause scored close to 1.0. Tracing it back showed that gap tokens were injected **before** the PURCHASE event was removed, so a pause right before buying survived as a trailing gap token. That made "ends with a pause" a perfect predictor of a purchase.

| | Sessions ending in a gap token | Share that were buyers |
|---|---|---|
| Before the fix | 5,640 | **100%** |
| After the fix | 0 | — |

**Fix.** Each session is now cut at the first purchase, and gap tokens are injected only after that cut. This also removes any browsing that happened after the purchase.

**Impact.** All four models were retrained. The leak had inflated validation AUC by 0.02–0.04 (LSTM 0.9868 → 0.9563). The LSTM remained the best model, so its advantage is real. Four regression tests now guard against the leak.

---

## How it works

```
Raw events (Retail Rocket + OTTO)
        │
        ▼
features/sequences.py      Cut each session at the first purchase, tokenize events,
                           insert GAP_SHORT (2–10 min) and GAP_LONG (10+ min) tokens
features/engineering.py    13 tabular features for the baseline models
        │
        ▼
model/train.py             Train LR, XGBoost, LSTM and Transformer; log every run to MLflow
        │
        ▼
models/lstm_best.pt        Best model weights
        │
        ▼
api/main.py                FastAPI app; model loads once at startup
api/routes/predict.py      POST /api/v1/predict   27 ms, Redis cache (5-min TTL)
api/routes/batch.py        POST /api/v1/batch     6 ms for 2 sessions
        │
        ▼
Docker + GitHub Actions    Containerized service with CI on every push
```

**Token vocabulary:** `PAD=0  VIEW=1  ADD_CART=2  PURCHASE=3  GAP_SHORT=4  GAP_LONG=5`

**Leakage controls**

- The label comes from whether a purchase happened; the model only sees events **before** the first purchase.
- No sequence ends in a gap token (verified on the full dataset).
- Purchase-derived features (`n_purchases`, `last_event`) are excluded, and view and cart counts use pre-purchase events only.

A scaled-out design (Kafka, Flink, feature store) is described in [`docs/production_architecture.md`](docs/production_architecture.md).

---

## Interpretability

**SHAP on XGBoost: top drivers by mean |SHAP|**

| Rank | Feature | Mean \|SHAP\| | Meaning |
|---|---|---|---|
| 1 | `n_carts` | **2.34** | Number of add-to-cart events |
| 2 | `source_otto` | 0.54 | Which store the session came from |
| 3 | `cart_rate` | 0.27 | Share of views that led to a cart action |
| 4 | `n_gap_long` | 0.18 | Pauses of 10+ minutes |
| 5 | `n_gap_short` | 0.18 | Pauses of 2–10 minutes |

**Transformer attention**

- In converting sessions, **GAP_LONG attends to ADD_CART (0.33)**: the model links a long pause back to the cart action, which is the hesitation pattern.
- VIEW events consistently attend to ADD_CART: 0.32 in converting sessions and 0.55 in non-converting ones.

SHAP shows *which* signals matter. Attention shows *how* events relate to each other in time.

---

## Live demo

[Open the demo on Hugging Face](https://huggingface.co/spaces/AhmeduddinMohammed/sessionscout) and build a session one event at a time.

| Step | Event | Score | What it means |
|---|---|---|---|
| 1 | VIEW | — | Scoring starts at 3 events |
| 2 | VIEW | — | |
| 3 | ADD_CART | **0.904** | High intent |
| 4 | GAP_LONG | — | Pause recorded; the score updates on the next action |
| 5 | VIEW (returns) | **0.589** | Intent dropped after the pause: show a reminder |

A browsing-only session (five VIEW events) scores **0.027**.

Scores rank sessions by purchase intent. The model is trained with class weighting, so they are not calibrated probabilities.

---

## API

**Score one session**

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user_001", "sequence": [1, 1, 2, 5, 1]}'
```

```json
{
  "session_id": "user_001",
  "conversion_probability": 0.589,
  "top_signals": ["VIEW×3", "ADD_CART×1", "GAP_LONG×1"],
  "cached": false,
  "latency_ms": 27.06
}
```

**Score a batch**

```bash
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"sessions": [
        {"session_id": "user_A", "sequence": [1, 1, 2, 5, 1]},
        {"session_id": "user_B", "sequence": [1, 1, 1, 1, 1]}
      ]}'
```

```json
{
  "results": [
    {"session_id": "user_A", "conversion_probability": 0.589},
    {"session_id": "user_B", "conversion_probability": 0.027}
  ],
  "total": 2,
  "latency_ms": 6.07
}
```

Interactive API docs are served at `http://localhost:8000/docs` when the API is running.

---

## Getting started

**Requirements:** Python 3.10+, a Kaggle account (for the datasets), and Docker (optional).

```bash
# 1. Clone and install
git clone https://github.com/MohammedAhmeduddin/sessionscout.git
cd sessionscout
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Download data (needs Kaggle credentials)
make download-all

# 3. Build sequences and features (dev set, about 5 minutes)
make pipeline-dev

# 4. Train all four models
make train

# 5. Serve the API at http://localhost:8000/docs
make api

# 6. Replay a live session
make simulate
```

**Run with Docker**

```bash
docker-compose up --build
# API:   http://localhost:8000
# Redis: localhost:6379
```

---

## Project structure

```
sessionscout/
├── src/sessionscout/
│   ├── config.py                    # Single source of truth for all settings
│   ├── features/
│   │   ├── sequences.py             # Events → token sequences (leak-safe)
│   │   └── engineering.py           # 13 tabular features
│   ├── model/
│   │   ├── dataset.py               # PyTorch Dataset + attention masks
│   │   ├── lstm.py                  # Bidirectional LSTM (170K params)
│   │   ├── transformer.py           # Transformer encoder (69K params)
│   │   ├── train.py                 # Training loop + MLflow logging
│   │   └── evaluate.py              # Metrics + comparison table
│   ├── explainability/
│   │   ├── shap_deep.py             # SHAP feature importance (XGBoost)
│   │   └── attention_viz.py         # Transformer attention heatmaps
│   └── api/
│       ├── main.py                  # FastAPI app; model loads once at startup
│       └── routes/
│           ├── predict.py           # POST /api/v1/predict (Redis cache)
│           └── batch.py             # POST /api/v1/batch
├── tests/                           # 125 tests, 93% coverage
│   ├── test_sequences.py            # 38 tests: tokenization, gaps, leak regression
│   ├── test_models.py               # 15 tests: LSTM + Transformer architecture
│   ├── test_api.py                  # 20 tests: endpoints, validation
│   ├── test_features.py             # 17 tests: feature pipeline
│   ├── test_training.py             # 22 tests: training loop, early stopping
│   └── test_explainability.py       # 13 tests: SHAP + attention
├── scripts/
│   ├── run_pipeline.py              # Pipeline orchestrator
│   ├── simulate_session.py          # Live session replay
│   └── sensitivity_analysis.py      # Business impact model
├── docs/
│   └── production_architecture.md   # Kafka + Flink + feature store design
├── Dockerfile
├── docker-compose.yml               # API + Redis
├── Makefile                         # Pipeline, training and serving commands
└── .github/workflows/ci.yml         # Lint, tests (3.10, 3.11), validation
```

---

## Testing and CI

```bash
pytest -q                       # full suite: 125 tests
pytest --cov=src/sessionscout   # coverage report
```

GitHub Actions runs on every push and pull request to `main`:

| Job | What it checks |
|---|---|
| Lint | `ruff` and `black` (versions pinned) |
| Tests (3.10, 3.11) | Data-pipeline tests, including the leak regression tests |
| Validate | Config loads, features import, LSTM and Transformer forward passes |

---

## Business impact (illustrative)

Daily value of targeting the top 500 sessions, assuming precision of 0.35 and a $2.50 cost per intervention:

| Average order value | 5% uplift | 10% uplift | 15% uplift |
|---|---|---|---|
| $45 | −$125/day | $1,000/day | $2,125/day |
| $65 | $375/day | $2,000/day | $3,625/day |
| $85 | $875/day | $3,000/day | $5,125/day |
| $120 | $1,750/day | $4,750/day | $7,750/day |

The negative cell matters: with low order values and low uplift, interventions cost more than they recover. These figures are assumptions (see `scripts/sensitivity_analysis.py`); real impact requires an A/B test.

---

## Data

| Dataset | Events | Sessions | Source |
|---|---|---|---|
| Retail Rocket | 2.7M | 1.4M | [Kaggle](https://kaggle.com/retailrocket/ecommerce-dataset) |
| OTTO | 220M | 12M | [Kaggle](https://kaggle.com/competitions/otto-recommender-system) |

The dev pipeline uses all Retail Rocket sessions plus 50,000 OTTO sessions, which yields 241,773 sessions after filtering.

---

## Limitations and next steps

| Area | Current state | Next step |
|---|---|---|
| Calibration | Scores rank sessions but are not calibrated probabilities | Fit isotonic calibration on the validation set |
| Data scope | Two public datasets; Retail Rocket is from 2015 | Retrain on the target retailer's data |
| Features | Event sequence + 13 tabular features; no price or category | Add item embeddings as an auxiliary input |
| Dataset size | Dev set with 50K OTTO sessions | Run the full 12M-session pipeline |
| Evaluation | Precision@500 measured on a small test set | Evaluate on a larger held-out set |

---

## Tech stack

| Layer | Tools |
|---|---|
| Data | Pandas, PyArrow, NumPy |
| Modeling | PyTorch, scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Interpretability | SHAP, attention analysis, Matplotlib |
| Serving | FastAPI, Uvicorn, Redis |
| Infrastructure | Docker, docker-compose, GitHub Actions |
| Testing | pytest, pytest-cov |

---

## Author

**Ahmeduddin Mohammed** — ML Engineer

[Portfolio](https://ahmed-portfolio-blue.vercel.app) · [LinkedIn](https://www.linkedin.com/in/mohammed-ahmeduddin/) · [GitHub](https://github.com/MohammedAhmeduddin)
