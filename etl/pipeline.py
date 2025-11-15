"""CONTEXT: Full ETL orchestration for ingesting equity data.

REQUIREMENTS:
- Extract OHLCV data for multiple symbols (parallel extraction).
- Clean and transform sequentially.
- Load results into ``stock_prices`` with transactional safety.
- Track execution stats and timing with rollback on critical failures.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: pandas, ThreadPoolExecutor
- Design pattern: Pipeline orchestrator
- Error handling: Capture extractor errors and abort gracefully.

INPUT/OUTPUT:
- Input: List of tickers plus time window parameters.
- Output: Dict summary containing timings and row counts.

EXAMPLE USAGE:
```python
from etl.pipeline import run_etl_pipeline

summary = run_etl_pipeline([\"AAPL\", \"MSFT\"], \"2023-01-01\", \"2023-06-01\")
```

TESTING:
- Simulate extractor failure fallback.
- Verify summary stats for sample data.

CODE STYLE:
- Type hints + docstrings.
- 88 char line limit.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from etl.extractors.alpha_vantage import fetch_alpha_vantage_data
from etl.extractors.yahoo_finance import fetch_yahoo_finance_data
from etl.loaders.database_loader import bulk_upsert_stock_prices
from etl.transformers.data_cleaner import clean_ohlcv_data

LOGGER = logging.getLogger(__name__)


def _extract_symbol(
    symbol: str,
    start_date: str | datetime,
    end_date: str | datetime | None,
    interval: str,
) -> Tuple[str, pd.DataFrame]:
    """Extract raw data for a symbol with Yahoo primary and Alpha fallback."""
    try:
        frame = fetch_yahoo_finance_data(symbol, start_date, end_date, interval)
        return symbol, frame
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "Yahoo extractor failed for %s: %s. Falling back to Alpha Vantage.",
            symbol,
            exc,
        )
    fallback = fetch_alpha_vantage_data(symbol)
    return symbol, fallback


def _extract_parallel(
    symbols: Iterable[str],
    start_date: str | datetime,
    end_date: str | datetime | None,
    interval: str,
) -> Dict[str, pd.DataFrame]:
    symbol_list = list(symbols)
    results: Dict[str, pd.DataFrame] = {}
    max_workers = min(len(symbol_list), 8) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_extract_symbol, symbol, start_date, end_date, interval): symbol
            for symbol in symbol_list
        }
        for future in as_completed(future_map):
            symbol = future_map[future]
            try:
                _, frame = future.result()
                results[symbol] = frame
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Extraction failed for %s: %s", symbol, exc)
                raise
    return results


def run_etl_pipeline(
    symbols: List[str],
    start_date: str | datetime,
    end_date: str | datetime | None = None,
    interval: str = "1d",
) -> Dict[str, Dict[str, Any]]:
    """Execute extract -> transform -> load pipeline."""
    pipeline_log: Dict[str, Dict[str, Any]] = {}
    start_time = time.time()
    extracts = _extract_parallel(symbols, start_date, end_date, interval)
    for symbol, raw_df in extracts.items():
        try:
            clean_df, report = clean_ohlcv_data(raw_df)
            clean_df.insert(0, "symbol", symbol)
            loader_stats = bulk_upsert_stock_prices(clean_df)
            pipeline_log[symbol] = {
                **report,
                **loader_stats,
                "rows_inserted": loader_stats.get("inserted", 0),
                "rows_updated": loader_stats.get("updated", 0),
                "success": loader_stats.get("failed", 0) == 0,
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Pipeline failed for %s: %s", symbol, exc)
            pipeline_log[symbol] = {
                "success": False,
                "error": str(exc),
            }
    elapsed = time.time() - start_time
    LOGGER.info("ETL pipeline completed in %.2fs", elapsed)
    pipeline_log["_meta"] = {"elapsed_seconds": round(elapsed, 2)}
    return pipeline_log
