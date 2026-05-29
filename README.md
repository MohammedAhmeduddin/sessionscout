<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Tests-121%20passed-22C55E?style=flat-square"/>
<img src="https://img.shields.io/badge/Coverage-93%25-22C55E?style=flat-square"/>
<img src="https://img.shields.io/badge/LSTM%20AUC-0.9868-7C3AED?style=flat-square"/>
<img src="https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white"/>

<br/><br/>

# SessionScout 🔍

### Real-time E-commerce Session Conversion Scoring

*Predicts whether an active browsing session will convert to a purchase — before the user leaves — so e-commerce sites can intervene only where it matters. Scores a session in **27ms**.*

<br/>

[![CI](https://github.com/MohammedAhmeduddin/sessionscout/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammedAhmeduddin/sessionscout/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/AhmeduddinMohammed/sessionscout)
[![API Docs](https://img.shields.io/badge/API-Swagger%20UI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](http://localhost:8000/docs)

<br/>

</div>

---

## Live Demo

| Component | URL |
|---|---|
| 🎛️ **Interactive Demo** | [huggingface.co/spaces/AhmeduddinMohammed/sessionscout](https://huggingface.co/spaces/AhmeduddinMohammed/sessionscout) |
| ⚡ **API** | `http://localhost:8000` (run locally) |
| 📖 **API Docs** | `http://localhost:8000/docs` |
| 💻 **GitHub** | [github.com/MohammedAhmeduddin/sessionscout](https://github.com/MohammedAhmeduddin/sessionscout) |

**Try this sequence in the demo — a hesitating buyer:**

| Step | Event | Probability | Action |
|---|---|---|---|
| 1 | 👁 VIEW | ~0.04 | 🔴 Do nothing |
| 2 | 👁 VIEW | ~0.04 | 🔴 Do nothing |
| 3 | 🛒 ADD_CART | ~0.23 | 🟠 Monitor |
| 4 | ⏳ GAP_LONG | ~0.65 | 🟡 Show reminder |
| 5 | 👁 VIEW (returns) | **~0.71** | 🟢 **Send discount now** |

---

## The Problem

72% of e-commerce sessions abandon. Most retargeting tools fire the same discount at every abandoning user. But there are three types of leaving user:

- **Was always going to buy** — got a phone call, coming back. Sending a discount wastes money.
- **Genuinely hesitating** — added to cart, paused 5 minutes, came back to look again. A small nudge tips them over. **This is the target.**
- **Never going to buy** — price comparing, just browsing. No intervention works.

A rule-based system cannot tell them apart. SessionScout reads the **sequence of behavior** and scores the probability in real time.

---

## Key Results

| Metric | Value |
|---|---|
| Best model | **LSTM — 0.9868 val AUC, 0.9883 test AUC** |
| API latency | **27ms** per session (CPU) |
| Batch latency | **6ms** for 2 sessions |
| Training data | **245,503 sessions** from Retail Rocket + OTTO |
| Sequence vocabulary | **6 tokens** — PAD, VIEW, ADD_CART, PURCHASE, GAP_SHORT, GAP_LONG |
| Top SHAP feature | **n_carts (2.23)** — cart count dominates |
| Test coverage | **93%** across 121 tests |
| CI/CD | GitHub Actions — green on Python 3.10 and 3.11 |

---

## Model Ladder

| Model | Val AUC | Test AUC | Notes |
|---|---|---|---|
| Logistic Regression | 0.9575 | 0.9573 | Tabular features baseline — AUC floor |
| XGBoost | 0.9748 | 0.9750 | Non-linear tabular — 500 trees, depth 6 |
| **LSTM (winner)** | **0.9868** | **0.9883** | Bidirectional, 170K params, MPS-accelerated |
| Transformer | 0.9814 | 0.9841 | 4-head encoder, 69K params |

**Honest finding:** The LSTM beat the Transformer. Sessions have a median length of 7 events — too short for long-range attention to provide an advantage over sequential memory. The Transformer is parameter-efficient (69K vs 170K) but needs longer sequences to exploit attention. This is documented, not hidden.

---

## Architecture

```
Raw events (Retail Rocket + OTTO datasets)
         │
         ▼
features/sequences.py          — tokenize events, inject gap tokens
features/engineering.py        — 13 tabular features (leak-free)
         │
         ▼
model/train.py                 — 4-model ladder with MLflow tracking
         │
         ├── logistic_regression  (val AUC 0.9575)
         ├── xgboost              (val AUC 0.9748)
         ├── lstm                 (val AUC 0.9868) ← winner
         └── transformer          (val AUC 0.9814)
         │
         ▼
models/lstm_best.pt            — best weights saved to disk
         │
         ▼
api/main.py (FastAPI)          — model loads ONCE at startup
api/routes/predict.py          — POST /predict · 27ms · Redis cache
api/routes/batch.py            — POST /batch · 6ms for 2 sessions
         │
         ▼
Docker + GitHub Actions CI     — containerized, tested, deployed
```

See [`docs/production_architecture.md`](docs/production_architecture.md) for the Kafka + Flink + Feature Store at-scale version.

---

## Interpretability

### SHAP — XGBoost Feature Importance

| Rank | Feature | Mean \|SHAP\| | Meaning |
|---|---|---|---|
| 1 | `n_carts` | **2.23** | Cart count — strongest single signal |
| 2 | `gap_ratio` | 0.92 | Fraction of session spent inactive |
| 3 | `cart_rate` | 0.78 | Views that resulted in cart actions |
| 4 | `n_gap_short` | 0.60 | Short pauses (2–10 min) |
| 5 | `seq_len` | 0.31 | Total session length |

### Attention — Transformer Head Analysis

The Transformer learned the hesitation pattern without being told:

- **VIEW → ADD_CART** attention weight **0.56** — every VIEW event attends strongly to the cart action
- **ADD_CART → GAP_LONG** attention weight **0.32** — the model reads the gap after carting as a key signal

XGBoost knew *that* cart events matter. The Transformer knows *how the cart event relates to surrounding events in time*. These are complementary explanations of the same underlying signal.

---

## API

```bash
# Single session score
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user_001", "sequence": [1, 1, 2, 5, 1, 1]}'
```

```json
{
  "session_id": "user_001",
  "conversion_probability": 0.697,
  "top_signals": ["VIEW×4", "ADD_CART×1", "GAP_LONG×1"],
  "cached": false,
  "latency_ms": 27.06
}
```

```bash
# Batch scoring — nightly job pattern
curl -X POST http://localhost:8000/api/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "sessions": [
      {"session_id": "user_A", "sequence": [1, 1, 2, 5, 1]},
      {"session_id": "user_B", "sequence": [1, 1, 1, 1, 1]}
    ]
  }'
```

```json
{
  "results": [
    {"session_id": "user_A", "conversion_probability": 0.7065},
    {"session_id": "user_B", "conversion_probability": 0.0376}
  ],
  "total": 2,
  "latency_ms": 6.07
}
```

**Token vocabulary:** `PAD=0  VIEW=1  ADD_CART=2  PURCHASE=3  GAP_SHORT=4  GAP_LONG=5`

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/MohammedAhmeduddin/sessionscout.git
cd sessionscout

# 2. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Download data (requires Kaggle credentials)
make download-all

# 4. Run dev pipeline (~5 min, 50K OTTO sessions)
make pipeline-dev

# 5. Train all 4 models
make train

# 6. Start API
make api
# → http://localhost:8000/docs

# 7. Run live session simulator
make simulate
```

### Docker

```bash
docker-compose up --build
# API:   http://localhost:8000
# Redis: localhost:6379
```

---

## Project Structure

```
sessionscout/
├── src/sessionscout/
│   ├── config.py                    # Single source of truth — cfg object
│   ├── features/
│   │   ├── sequences.py             # Raw events → padded token sequences
│   │   └── engineering.py          # 13 tabular features, leak-free
│   ├── model/
│   │   ├── dataset.py              # PyTorch Dataset + attention masks
│   │   ├── lstm.py                 # Bidirectional LSTM (winner — 170K params)
│   │   ├── transformer.py          # Transformer encoder (69K params)
│   │   ├── train.py                # Training loop + MLflow logging
│   │   └── evaluate.py             # Metrics + comparison table
│   ├── explainability/
│   │   ├── shap_deep.py            # SHAP feature importance for XGBoost
│   │   └── attention_viz.py        # Transformer attention heatmaps
│   └── api/
│       ├── main.py                 # FastAPI app — model loads once at startup
│       └── routes/
│           ├── predict.py          # POST /predict — 27ms, Redis cache
│           └── batch.py            # POST /batch — 6ms for 2 sessions
├── tests/                          # 121 tests, 93% coverage
│   ├── test_sequences.py           # 34 tests — tokenization, gap injection
│   ├── test_models.py              # 15 tests — LSTM + Transformer architecture
│   ├── test_api.py                 # 20 tests — endpoints, validation, mock model
│   ├── test_features.py            # 17 tests — feature engineering pipeline
│   ├── test_training.py            # 22 tests — training loop, MLflow mocked
│   └── test_explainability.py      # 13 tests — SHAP + attention viz
├── scripts/
│   ├── simulate_session.py         # Live session replay demo
│   ├── sensitivity_analysis.py     # Business impact table
│   └── run_pipeline.py             # Full pipeline orchestrator
├── docs/
│   └── production_architecture.md  # Kafka + Flink + Feature Store design
├── Dockerfile
├── docker-compose.yml              # API + Redis
└── .github/workflows/ci.yml        # Lint + Tests (3.10, 3.11) + Validate
```

---

## Business Impact

Conservative estimate (AOV=$65, 10% uplift, Precision@500=0.35, $2.50/intervention):

| AOV | Uplift 5% | Uplift 10% | Uplift 15% |
|---|---|---|---|
| $45 | -$125/day | $1,000/day | $2,125/day |
| $65 | $375/day | $2,000/day | $3,625/day |
| $85 | $875/day | $3,000/day | $5,125/day |
| $120 | $1,750/day | $4,750/day | $7,750/day |

At $85 AOV and 10% uplift: **$3,000/day → $1.1M/year**.

Note the bottom-left cell: at low AOV ($45) and low uplift (5%), the system costs more than it recovers. This is the honest finding that tells you the minimum viable deployment conditions.

All assumptions documented in `scripts/sensitivity_analysis.py`. **Real impact requires A/B testing.**

---

## Data

| Dataset | Events | Sessions | License |
|---|---|---|---|
| Retail Rocket | 2.7M | 1.4M | [Kaggle — retailrocket/ecommerce-dataset](https://kaggle.com/retailrocket/ecommerce-dataset) |
| OTTO | 220M | 12M | [Kaggle — otto-recommender-system](https://kaggle.com/competitions/otto-recommender-system) |

Both free on Kaggle. The dev pipeline uses 50K OTTO sessions (~5 min). Full pipeline uses all 12M (~45 min).

**Anti-leakage design:**
- PURCHASE tokens stripped from sequences — label set from purchase presence, but model only sees browsing behavior
- `n_purchases` and `last_event` removed from tabular features
- Gap tokens injected between events (120s = GAP_SHORT, 600s = GAP_LONG)

---

## Testing Strategy

```
tests/
├── test_sequences.py    34 tests  — tokenization, gap injection, left-padding, anti-leakage
├── test_models.py       15 tests  — output shapes, gradient flow, parameter counts
├── test_api.py          20 tests  — endpoints, request validation, mock model fixture
├── test_features.py     17 tests  — feature pipeline, no-null guarantee, value ranges
├── test_training.py     22 tests  — training loop, early stopping, XGBoost mocked (MPS safety)
└── test_explainability.py 13 tests — SHAP computation, attention heatmap generation
```

XGBoost is mocked in training tests to avoid Apple Silicon MPS + XGBoost segfault — a real production constraint documented honestly.

---

## Tech Stack

| Layer | Tools | Why |
|---|---|---|
| Data | Pandas, PyArrow, NumPy | Parquet pipeline, fast I/O |
| Modeling | PyTorch, Scikit-learn, XGBoost | Deep learning + tabular baselines |
| Tracking | MLflow | All 4 runs tracked, AUC per epoch |
| Interpretability | SHAP, Matplotlib | XGBoost features + Transformer attention |
| Serving | FastAPI, Uvicorn, Redis | 27ms inference, 5-min cache TTL |
| Infrastructure | Docker, GitHub Actions | Containerized, CI on every push |
| Testing | pytest, pytest-cov | 93% coverage gate |

---

## CI/CD

GitHub Actions runs on every push to `main`:

```
Push to main
    │
    ├── Job: Lint
    │   ├── ruff check src/ tests/ scripts/
    │   └── black --check --target-version py311
    │
    ├── Job: Tests (Python 3.10)
    │   ├── pip install -e ".[dev]"
    │   └── pytest tests/test_sequences.py
    │
    ├── Job: Tests (Python 3.11)
    │   └── pytest tests/test_sequences.py --cov
    │
    └── Job: Validate
        ├── Config loads correctly (vocab_size=6, max_len=64)
        ├── Feature imports work
        └── LSTM + Transformer forward pass (random batch)
```

---

## Limitations

| Area | Current state | What changes it |
|---|---|---|
| **Data scope** | Retail Rocket (2015, 1 retailer) | Retrain on target retailer's data |
| **Short sessions** | Median 7 events — LSTM beats Transformer | Transformer wins on longer sessions |
| **No price/category features** | Sequence + 13 tabular only | Add item embeddings as auxiliary input |
| **Dev dataset** | 50K OTTO sessions | Full 12M pipeline pending |
| **Precision@500** | 1.0 on small test set — overstated | Needs larger held-out evaluation set |

---

## Resume Bullets

```
Built 4-model ladder (LR → XGB → LSTM → Transformer) for e-commerce session conversion
prediction on 245K real sessions; LSTM achieved 0.9868 val AUC — 1.2 pts over XGBoost
baseline — with honest documented finding that LSTM outperforms Transformer on short sequences

Deployed real-time scoring API with FastAPI at 27ms latency, Redis caching, and batch
endpoint at 6ms; containerized with Docker + docker-compose including Redis sidecar

Built SHAP interpretability showing n_carts (SHAP=2.23) and gap_ratio (0.92) as top
drivers; Transformer attention analysis revealed VIEW→ADD_CART (0.56) hesitation pattern

121 tests, 93% coverage, GitHub Actions CI on Python 3.10 and 3.11; live demo on
HuggingFace Spaces — huggingface.co/spaces/AhmeduddinMohammed/sessionscout
```

---

## Author

**Ahmeduddin Mohammed**
- GitHub: [@MohammedAhmeduddin](https://github.com/MohammedAhmeduddin)
- LinkedIn: [linkedin.com/in/mohammed-ahmeduddin](https://www.linkedin.com/in/mohammed-ahmeduddin/)
- Portfolio: [SessionScout](https://huggingface.co/spaces/AhmeduddinMohammed/sessionscout) · [CareAgent](https://huggingface.co/spaces/AhmeduddinMohammed/careagent)

---

<div align="center">
<sub>Built with PyTorch · FastAPI · XGBoost · SHAP · MLflow · Docker · GitHub Actions</sub>
</div>
