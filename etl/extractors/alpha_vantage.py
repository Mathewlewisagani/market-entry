"""CONTEXT: Alpha Vantage extractor for supplementary equity data.

REQUIREMENTS:
- Fetch daily OHLCV data using alpha_vantage SDK.
- Provide caching to minimize redundant calls.
- Enforce free-tier rate limits (5 calls/minute).
- Return canonical DataFrame matching Yahoo extractor.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: alpha_vantage SDK + pandas
- Design pattern: Functional extractor with memoization
- Error handling: Catch API errors and propagate informative messages

INPUT/OUTPUT:
- Input: symbol, optional outputsize, cache toggle.
- Output: pandas.DataFrame (timestamp, open, high, low, close, volume).

EXAMPLE USAGE:
```python
from etl.extractors.alpha_vantage import fetch_alpha_vantage_data

df = fetch_alpha_vantage_data(\"MSFT\")
```

TESTING:
- Cache hit path vs. miss path.
- Rate limit enforcement across consecutive calls.

CODE STYLE:
- Type hints + docstrings.
- 88 character limit.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, Tuple

import pandas as pd
from alpha_vantage.timeseries import TimeSeries

from config.settings import get_config

LOGGER = logging.getLogger(__name__)

_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}
_REQUEST_LOG: Deque[float] = deque()
MAX_CALLS_PER_MINUTE = 5


def _throttle() -> None:
    """Ensure Alpha Vantage rate limits are respected."""
    now = time.time()
    while _REQUEST_LOG and now - _REQUEST_LOG[0] > 60:
        _REQUEST_LOG.popleft()
    if len(_REQUEST_LOG) >= MAX_CALLS_PER_MINUTE:
        sleep_for = 60 - (now - _REQUEST_LOG[0])
        LOGGER.info("Alpha Vantage rate limit hit. Sleeping %.2fs", sleep_for)
        time.sleep(max(sleep_for, 0))
    _REQUEST_LOG.append(time.time())


def fetch_alpha_vantage_data(
    symbol: str,
    outputsize: str = "compact",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download daily prices with caching."""
    cache_key = (symbol, outputsize)
    if use_cache and cache_key in _CACHE:
        LOGGER.debug("Alpha Vantage cache hit for %s", cache_key)
        return _CACHE[cache_key].copy()

    config = get_config()
    if not config.apis.alpha_vantage_key:
        raise ValueError("Alpha Vantage API key missing.")
    ts = TimeSeries(
        key=config.apis.alpha_vantage_key,
        output_format="pandas",
    )
    _throttle()
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Alpha Vantage API error for %s: %s", symbol, exc)
        raise
    if data.empty:
        raise ValueError(f"Alpha Vantage returned no data for {symbol}")
    frame = (
        data.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )
        .reset_index()
        .rename(columns={"date": "timestamp"})
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    normalized = frame[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ].sort_values("timestamp")
    if use_cache:
        _CACHE[cache_key] = normalized.copy()
    return normalized
