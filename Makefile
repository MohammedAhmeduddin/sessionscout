.PHONY: install install-dev download-retailrocket download-otto \
        pipeline pipeline-dev sequences features \
        train train-lr train-xgb train-lstm train-transformer \
        shap attention sensitivity \
        api mlflow simulate \
        test test-pipeline lint format \
        docker-up docker-down clean

# ── Environment ───────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# ── Data download ─────────────────────────────────────────────────────────────
download-retailrocket:
	kaggle datasets download \
		-d retailrocket/ecommerce-dataset \
		-p data/raw/retailrocket
	cd data/raw/retailrocket && unzip -o "*.zip" && rm -f "*.zip"

download-otto:
	kaggle competitions download \
		-c otto-recommender-system \
		-p data/raw/otto
	cd data/raw/otto && unzip -o "*.zip" && rm -f "*.zip"

download-all: download-retailrocket download-otto

# ── Data pipeline ─────────────────────────────────────────────────────────────
sequences:
	python -m sessionscout.features.sequences

sequences-dev:
	python -m sessionscout.features.sequences --max-otto-sessions 50000

features:
	python -m sessionscout.features.engineering

pipeline: sequences features
	@echo "✓ Full pipeline complete"

pipeline-dev: sequences-dev features
	@echo "✓ Dev pipeline complete (50K OTTO sessions)"

# ── Training ──────────────────────────────────────────────────────────────────
train-lr:
	python -m sessionscout.model.train --model lr

train-xgb:
	python -m sessionscout.model.train --model xgb

train-lstm:
	python -m sessionscout.model.train --model lstm

train-transformer:
	python -m sessionscout.model.train --model transformer

train:
	python -m sessionscout.model.train --model all

# ── Interpretability ──────────────────────────────────────────────────────────
shap:
	python -m sessionscout.explainability.shap_deep

attention:
	python -m sessionscout.explainability.attention_viz

sensitivity:
	python scripts/sensitivity_analysis.py

# ── Services ──────────────────────────────────────────────────────────────────
api:
	uvicorn sessionscout.api.main:app --reload --port 8000

mlflow:
	mlflow ui --port 5000

simulate:
	python scripts/simulate_session.py --converting

simulate-abandon:
	python scripts/simulate_session.py --abandoning

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-pipeline:
	pytest tests/test_sequences.py -v

test-cov:
	pytest tests/ --cov=sessionscout --cov-report=term-missing

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up --build

docker-up-d:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

clean-data:
	rm -f data/processed/*.parquet data/processed/*.json
