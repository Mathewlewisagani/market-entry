"""CONTEXT: Feature engineering pipeline for ML-ready datasets.

REQUIREMENTS:
- Load historical prices from the database for a symbol/date range.
- Compute technical indicators plus lag, volatility, and targets.
- Persist indicator subset into ``technical_indicators`` table.
- Return a feature matrix ready for downstream modeling.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: pandas + SQLAlchemy
- Design pattern: Functional data preparation utility
- Error handling: Raise ValueError when insufficient data.

INPUT/OUTPUT:
- Input: symbol plus optional start/end date filters.
- Output: pandas.DataFrame containing engineered features.

EXAMPLE USAGE:
```python
from features.feature_engineering import build_feature_matrix

features_df = build_feature_matrix(\"AAPL\", \"2023-01-01\", \"2023-12-31\")
```

TESTING:
- Validate DB read/write operations.
- Confirm forward_return aligns with 5-day horizon.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
from sqlalchemy import text

from config.database import ENGINE, get_session
from features.technical_indicators import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
    calculate_volume_features,
)

LOGGER = logging.getLogger(__name__)


def _fetch_price_history(
    symbol: str,
    start_date: Optional[str | datetime],
    end_date: Optional[str | datetime],
) -> pd.DataFrame:
    query = """
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM stock_prices
        WHERE symbol = :symbol
        {start_clause}
        {end_clause}
        ORDER BY timestamp
    """
    start_clause = "AND timestamp >= :start_date" if start_date else ""
    end_clause = "AND timestamp <= :end_date" if end_date else ""
    formatted_query = query.format(start_clause=start_clause, end_clause=end_clause)
    params: Dict[str, str | datetime] = {"symbol": symbol}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    df = pd.read_sql_query(text(formatted_query), con=ENGINE, params=params)
    if df.empty:
        raise ValueError(f"No price history for {symbol}.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _persist_technical_indicators(df: pd.DataFrame) -> None:
    records = df[
        [
            "symbol",
            "timestamp",
            "rsi",
            "macd",
            "signal_line",
            "sma_20",
            "sma_50",
            "sma_200",
            "bollinger_upper",
            "bollinger_lower",
        ]
    ].dropna()
    if records.empty:
        return
    insert_sql = text(
        """
        INSERT INTO technical_indicators
        (symbol, timestamp, rsi, macd, signal_line, sma_20, sma_50, sma_200,
         bollinger_upper, bollinger_lower)
        VALUES (:symbol, :timestamp, :rsi, :macd, :signal_line, :sma_20, :sma_50,
                :sma_200, :bollinger_upper, :bollinger_lower)
        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            rsi = EXCLUDED.rsi,
            macd = EXCLUDED.macd,
            signal_line = EXCLUDED.signal_line,
            sma_20 = EXCLUDED.sma_20,
            sma_50 = EXCLUDED.sma_50,
            sma_200 = EXCLUDED.sma_200,
            bollinger_upper = EXCLUDED.bollinger_upper,
            bollinger_lower = EXCLUDED.bollinger_lower
        """
    )
    with get_session() as session:
        session.execute(insert_sql, records.to_dict(orient="records"))


def build_feature_matrix(
    symbol: str,
    start_date: Optional[str | datetime] = None,
    end_date: Optional[str | datetime] = None,
) -> pd.DataFrame:
    """Create engineered feature set for a symbol."""
    prices = _fetch_price_history(symbol, start_date, end_date)
    close = prices["close"]
    volume = prices["volume"]
    prices["sma_20"] = calculate_sma(close, 20)
    prices["sma_50"] = calculate_sma(close, 50)
    prices["sma_200"] = calculate_sma(close, 200)
    prices["ema_12"] = calculate_ema(close, 12)
    prices["ema_26"] = calculate_ema(close, 26)
    prices["rsi"] = calculate_rsi(close)
    macd_df = calculate_macd(close)
    prices["macd"] = macd_df["macd"]
    prices["signal_line"] = macd_df["signal_line"]
    bollinger = calculate_bollinger_bands(close)
    prices["bollinger_upper"] = bollinger["bollinger_upper"]
    prices["bollinger_lower"] = bollinger["bollinger_lower"]
    volume_features = calculate_volume_features(volume)
    prices = pd.concat([prices, volume_features], axis=1)
    prices["price_change_1d"] = prices["close"].pct_change(periods=1)
    prices["price_change_5d"] = prices["close"].pct_change(periods=5)
    prices["price_change_20d"] = prices["close"].pct_change(periods=20)
    prices["volatility_20d"] = (
        prices["close"].pct_change().rolling(window=20).std()
    )
    prices["forward_return"] = (
        prices["close"].shift(-5) - prices["close"]
    ) / prices["close"]
    engineered = prices.dropna().reset_index(drop=True)
    _persist_technical_indicators(engineered)
    LOGGER.info(
        "Generated %s feature rows for %s.",
        len(engineered),
        symbol,
    )
    return engineered


def generate_features_for_symbol(
    symbol: str,
    lookback_days: int = 730,
) -> pd.DataFrame:
    """Helper for scripts to build features over rolling window."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days)
    return build_feature_matrix(
        symbol,
        start_date=start_date,
        end_date=end_date,
    )
