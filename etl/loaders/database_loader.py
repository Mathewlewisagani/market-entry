"""CONTEXT: High-throughput loader for stock price records.

REQUIREMENTS:
- Bulk insert data into ``stock_prices`` with conflict handling.
- Commit in batches of 1000 rows for efficiency.
- Retry on transient database errors and log stats.
- Validate data integrity before loading.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: SQLAlchemy Core
- Design pattern: Loader utility with batching
- Error handling: Retry on SQLAlchemyError / IntegrityError.

INPUT/OUTPUT:
- Input: pandas.DataFrame with canonical OHLCV schema.
- Output: Dict with counts for inserted, updated, failed rows.

EXAMPLE USAGE:
```python
from etl.loaders.database_loader import bulk_upsert_stock_prices

stats = bulk_upsert_stock_prices(df)
```

TESTING:
- Conflict path vs new inserts.
- Retry logic via simulated errors.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from config.database import get_session

LOGGER = logging.getLogger(__name__)

UPSERT_SQL = text(
    """
    INSERT INTO stock_prices
    (symbol, timestamp, open, high, low, close, volume)
    VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
    ON CONFLICT (symbol, timestamp) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume
    RETURNING (xmax = 0) AS inserted_flag
    """
)


def _validate_frame(df: pd.DataFrame) -> None:
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df["symbol"].isna().any():
        raise ValueError("Symbol column contains nulls.")


def _chunk_dataframe(df: pd.DataFrame, batch_size: int) -> Iterable[pd.DataFrame]:
    total_batches = math.ceil(len(df) / batch_size) or 1
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        yield df.iloc[start:end]


def bulk_upsert_stock_prices(
    df: pd.DataFrame,
    batch_size: int = 1000,
    max_retries: int = 3,
) -> Dict[str, int]:
    """Load OHLCV rows into stock_prices."""
    _validate_frame(df)
    inserted = updated = failed = 0
    for batch in _chunk_dataframe(df.reset_index(drop=True), batch_size):
        params = batch.to_dict(orient="records")
        retries = 0
        while retries < max_retries:
            try:
                with get_session() as session:
                    result = session.execute(UPSERT_SQL, params)
                    flags = result.scalars().all()
                inserted_batch = sum(1 for flag in flags if flag)
                inserted += inserted_batch
                updated += len(flags) - inserted_batch
                break
            except SQLAlchemyError as exc:
                retries += 1
                LOGGER.warning(
                    "Batch insert failed (attempt %s/%s): %s",
                    retries,
                    max_retries,
                    exc,
                )
                if retries >= max_retries:
                    failed += len(params)
    LOGGER.info(
        "Loader stats - inserted:%s updated:%s failed:%s",
        inserted,
        updated,
        failed,
    )
    return {"inserted": inserted, "updated": updated, "failed": failed}
