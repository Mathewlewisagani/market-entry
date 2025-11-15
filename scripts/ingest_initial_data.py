"""Initial data ingestion script for the market timing platform."""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from etl.pipeline import run_etl_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def ingest_initial_data(symbols: List[str]) -> None:
    """Fetch two years of historical data for the provided symbols."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=730)
    LOGGER.info("Fetching data for %s symbols.", len(symbols))
    LOGGER.info("Date range: %s to %s", start_date.date(), end_date.date())
    results = run_etl_pipeline(
        symbols=symbols,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    LOGGER.info("\n%s", "=" * 50)
    LOGGER.info("DATA INGESTION SUMMARY")
    LOGGER.info("%s", "=" * 50)
    for symbol in symbols:
        result = results.get(symbol, {})
        success = result.get("success", False)
        rows = result.get("rows_inserted", 0)
        status = "✅ SUCCESS" if success else "❌ FAILED"
        LOGGER.info("%s: %s - %s rows", symbol, status, rows)
        if not success and "error" in result:
            LOGGER.info("    Error: %s", result["error"])
    LOGGER.info("%s", "=" * 50)


if __name__ == "__main__":
    DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    ingest_initial_data(DEFAULT_SYMBOLS)
