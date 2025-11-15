"""CONTEXT: Market data extraction from Yahoo Finance via yfinance.

REQUIREMENTS:
- Fetch OHLCV data for a symbol and date window.
- Handle API errors with exponential backoff and logging.
- Return pandas DataFrame with canonical columns.
- Validate data completeness (no missing timestamp/prices).

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: yfinance + pandas
- Design pattern: Functional extractor utility
- Error handling: Catch yfinance/shared exceptions and retry

INPUT/OUTPUT:
- Input: symbol, start_date, end_date, interval string (e.g., \"1d\").
- Output: pandas.DataFrame (timestamp, open, high, low, close, volume).

EXAMPLE USAGE:
```python
from etl.extractors.yahoo_finance import fetch_yahoo_finance_data

df = fetch_yahoo_finance_data(\"AAPL\", \"2023-01-01\", \"2023-06-01\", \"1d\")
```

TESTING:
- Missing symbol handling.
- Interval validation.
- Partial data detection.

CODE STYLE:
- Type hints and Google-style docstrings.
- Max line length 88 chars.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure downloader returned the required structure."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns from Yahoo response: {missing_cols}")
    if df.index.hasnans:
        raise ValueError("Timestamp index contains NaN values.")
    return df


def fetch_yahoo_finance_data(
    symbol: str,
    start_date: str | datetime,
    end_date: Optional[str | datetime] = None,
    interval: str = "1d",
    max_retries: int = 5,
) -> pd.DataFrame:
    """Retrieve OHLCV data with retry/backoff."""
    attempt = 0
    end_date = end_date or datetime.utcnow()
    while attempt < max_retries:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                actions=False,
            )
            if data.empty:
                raise ValueError("Empty payload received from Yahoo Finance.")
            data = _validate_dataframe(data)
            frame = (
                data.reset_index()
                .rename(
                    columns={
                        "Date": "timestamp",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )
            )
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            return frame[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as exc:  # noqa: BLE001 - upstream libs raise generic errors
            attempt += 1
            sleep_time = min(2**attempt, 60)
            LOGGER.warning(
                "Yahoo Finance fetch failed (attempt %s/%s): %s. Retrying in %ss.",
                attempt,
                max_retries,
                exc,
                sleep_time,
            )
            time.sleep(sleep_time)
    raise RuntimeError(f"Failed to fetch data for {symbol} after {max_retries} retries.")
