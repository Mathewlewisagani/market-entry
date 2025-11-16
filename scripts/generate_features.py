"""Generate technical indicators for all symbols with price history."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from config.database import get_session
from features.feature_engineering import generate_features_for_symbol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def generate_all_features(lookback_days: int = 730) -> None:
    """Generate features for every symbol present in stock_prices."""
    with get_session() as session:
        results = session.execute(text("SELECT DISTINCT symbol FROM stock_prices"))
        symbols = [row[0] for row in results]
    LOGGER.info("Generating features for %s symbols.", len(symbols))
    for symbol in symbols:
        try:
            LOGGER.info("Processing %s ...", symbol)
            feature_df = generate_features_for_symbol(symbol, lookback_days=lookback_days)
            LOGGER.info("✅ %s: Generated %s rows", symbol, len(feature_df))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("❌ %s: %s", symbol, exc)
    LOGGER.info("Feature generation complete.")


if __name__ == "__main__":
    generate_all_features()
