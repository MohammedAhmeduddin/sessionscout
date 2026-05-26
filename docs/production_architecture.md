# SessionScout — Production Architecture

This document describes what SessionScout looks like deployed at scale.
The current implementation (FastAPI + Redis on a single server) is the
correct architecture for a team validating the model. This document
shows what the system becomes when it works.

---

## Current architecture (single server)

Browser events
→ POST /api/v1/predict
→ FastAPI (uvicorn, 1 worker)
→ SessionLSTM inference (~27ms)
→ Redis cache (TTL 5 min)
→ JSON response

Good for: up to ~500 req/sec on a single GPU server.
Bottleneck: single process, no event streaming.

---

## Production architecture (100K+ sessions/day)

Browser
│ events fire on every page load
▼
Kafka Topic: session-events
│ keyed by session_id
│ retention: 24 hours
▼
Flink Job: session-window-builder
│ rolling 30-min window per session_id
│ injects gap tokens between events
│ writes current sequence to Feature Store
▼
Redis Feature Store
│ key: session:{session_id}
│ value: {sequence: [...], seq_len: N, last_updated: ts}
│ TTL: 30 minutes
▼
Inference Service (FastAPI + GPU)
│ called by marketing platform on trigger events
│ reads sequence from Feature Store
│ runs LSTM inference
│ returns conversion_probability
▼
Marketing Platform
│ applies business rules:
│ prob > 0.75 → trigger discount
│ prob > 0.50 → show reminder banner
│ prob < 0.30 → do nothing
▼
Action (push notification / discount / banner)

---

## Component details

### Kafka

- Topic: `session-events`, partitioned by `session_id`
- Producer: browser SDK sends events on each page interaction
- Consumer group: `session-window-builder` (Flink)
- Retention: 24 hours (replay capability for debugging)

### Flink (session window builder)

- Keyed stream on `session_id`
- Session window: 30 minutes of inactivity closes the window
- Gap token injection: same logic as `features/sequences.py`
- Output: writes `{session_id: seq}` to Redis on every new event

### Redis Feature Store

- Stores the running sequence per active session
- Key pattern: `session:{session_id}`
- TTL: 30 minutes (matches session window)
- On inference: read sequence, run model, return score
- On cache hit: return cached score (same sequence = same result)

### Inference Service

- FastAPI, 2-4 workers behind a load balancer
- GPU server (T4 or A10G): inference drops from 27ms to <5ms
- Model loaded once at startup from MLflow Production registry
- `/predict`: single session, called by marketing platform
- `/batch`: 1000 sessions, called by nightly job

### MLflow Model Registry

- All training runs tracked automatically
- Best model promoted to Production stage
- Inference service loads `models:/sessionscout-transformer/Production`
- Rollback: demote current, promote previous — zero downtime

### Model retraining pipeline

- Weekly: retrain on last 30 days of sessions with confirmed outcomes
- Triggered by: GitHub Actions scheduled workflow
- Validation gate: new model must beat current Production AUC by 0.001
- Deployment: automatic promotion if gate passes, Slack alert if not

---

## Scaling numbers

| Sessions/day | Infrastructure                    | Latency (p99) | Cost estimate |
| ------------ | --------------------------------- | ------------- | ------------- |
| 10K          | Single FastAPI server, no GPU     | 50ms          | ~$50/month    |
| 100K         | 2 FastAPI workers, T4 GPU         | 8ms           | ~$200/month   |
| 1M           | Flink + 4 inference servers, A10G | 5ms           | ~$800/month   |
| 10M          | Full Kafka cluster, auto-scaling  | 3ms           | ~$3,000/month |

---

## What is not in this repo

- Kafka producer SDK (browser-side)
- Flink job (Java/Scala)
- Kubernetes deployment manifests
- Grafana dashboards for latency/throughput monitoring
- A/B testing framework for measuring real intervention lift

These are standard components that any platform team can provide.
The novel part — the model, the tokenization logic, the feature
engineering — is fully implemented here.
