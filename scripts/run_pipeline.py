"""
scripts/run_pipeline.py — Orchestrate the full data pipeline.

Usage:
  python scripts/run_pipeline.py                # full run
  python scripts/run_pipeline.py --dev          # 50K OTTO sessions (~5 min)
  python scripts/run_pipeline.py --sequences-only
  python scripts/run_pipeline.py --features-only
  python scripts/run_pipeline.py --validate
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sessionscout.config import cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def validate_raw():
    logger.info("Checking raw data files...")
    files = {
        "Retail Rocket events.csv": cfg.paths.rr_events,
        "OTTO train.jsonl": cfg.paths.otto_train,
    }
    found = 0
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"  ✓ {name:<30} {size_mb:>7.1f} MB")
            found += 1
        else:
            logger.info(f"  ✗ {name:<30} NOT FOUND")

    if found == 0:
        logger.error("No raw data found. Run: make download-all")
        sys.exit(1)


def validate_processed():
    logger.info("Checking processed outputs...")
    for name, path in [
        ("sequences.parquet", cfg.paths.sequences_parquet),
        ("features.parquet", cfg.paths.features_parquet),
    ]:
        if path.exists():
            df = pd.read_parquet(path)
            rate = (
                f"conversion: {df['label'].mean():.3f}" if "label" in df.columns else ""
            )
            logger.info(f"  ✓ {name:<25} {len(df):>8,} rows  {rate}")
        else:
            logger.info(f"  ✗ {name:<25} not found")


def run_sequences(dev: bool = False):
    from sessionscout.features.sequences import build_sequence_dataset

    logger.info("=" * 50)
    logger.info("STEP 1/2 — Building event sequences")
    logger.info("=" * 50)
    t0 = time.time()
    df = build_sequence_dataset(max_otto_sessions=50_000 if dev else None)
    logger.info(f"Done in {(time.time()-t0)/60:.1f} min — {len(df):,} sessions")
    return df


def run_features():
    from sessionscout.features.engineering import build_feature_matrix

    logger.info("=" * 50)
    logger.info("STEP 2/2 — Building feature matrix")
    logger.info("=" * 50)
    t0 = time.time()
    df = build_feature_matrix()
    logger.info(f"Done in {(time.time()-t0)/60:.1f} min — {df.shape}")
    return df


def main():
    parser = argparse.ArgumentParser(description="SessionScout data pipeline")
    parser.add_argument(
        "--dev", action="store_true", help="Dev mode: 50K OTTO sessions (~5 min)"
    )
    parser.add_argument("--sequences-only", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        validate_raw()
        validate_processed()
        return

    validate_raw()
    t_total = time.time()

    if args.sequences_only:
        run_sequences(args.dev)
    elif args.features_only:
        run_features()
    else:
        run_sequences(args.dev)
        run_features()

    logger.info(f"\nTotal: {(time.time()-t_total)/60:.1f} min")
    validate_processed()
    logger.info("\n✓ Pipeline complete")
    logger.info("  Next: make train")


if __name__ == "__main__":
    main()
