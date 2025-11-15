"""CONTEXT: Technical indicator computations for downstream models.

REQUIREMENTS:
- Implement SMA, EMA, RSI, MACD, Bollinger Bands, and volume features.
- Handle NaN values gracefully for insufficient lookback windows.
- Accept pandas Series/DataFrames and maintain index alignment.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: pandas
- Design pattern: Functional utilities
- Error handling: Validate inputs and raise ValueError for invalid windows.

INPUT/OUTPUT:
- Input: Price/volume Series (Close prices, etc.).
- Output: pandas Series/DataFrames containing indicator values.

EXAMPLE USAGE:
```python
from features.technical_indicators import calculate_rsi

rsi = calculate_rsi(close_prices)
```

TESTING:
- Verify indicator lengths match input.
- Confirm RSI bounds (0-100).

CODE STYLE:
- Type hints + docstrings.
- 88 char line limit.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _ensure_series(data: pd.Series | pd.DataFrame, column: str | None = None) -> pd.Series:
    """Normalize Series/DataFrame inputs to Series."""
    if isinstance(data, pd.DataFrame):
        if not column:
            if len(data.columns) != 1:
                raise ValueError("Specify column when passing DataFrame.")
            column = data.columns[0]
        return data[column]
    return data


def calculate_sma(prices: pd.Series | pd.DataFrame, window: int) -> pd.Series:
    """Calculate the Simple Moving Average."""
    if window <= 0:
        raise ValueError("Window must be positive.")
    series = _ensure_series(prices)
    return series.rolling(window=window, min_periods=1).mean()


def calculate_ema(prices: pd.Series | pd.DataFrame, span: int) -> pd.Series:
    """Calculate the Exponential Moving Average."""
    if span <= 0:
        raise ValueError("Span must be positive.")
    series = _ensure_series(prices)
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(prices: pd.Series | pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index."""
    if period <= 0:
        raise ValueError("Period must be positive.")
    series = _ensure_series(prices)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace({0: np.nan})
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0.0)


def calculate_macd(
    prices: pd.Series | pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD line and signal line."""
    fast_ema = calculate_ema(prices, fast)
    slow_ema = calculate_ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "signal_line": signal_line,
            "macd_hist": hist,
        }
    )


def calculate_bollinger_bands(
    prices: pd.Series | pd.DataFrame,
    window: int = 20,
    num_std: int = 2,
) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    sma = calculate_sma(prices, window)
    series = _ensure_series(prices)
    rolling_std = series.rolling(window=window, min_periods=1).std()
    upper = sma + (rolling_std * num_std)
    lower = sma - (rolling_std * num_std)
    return pd.DataFrame(
        {
            "bollinger_mid": sma,
            "bollinger_upper": upper,
            "bollinger_lower": lower,
        }
    )


def calculate_volume_features(volume: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Calculate volume SMA, ratio, and momentum."""
    vol_series = _ensure_series(volume)
    sma = vol_series.rolling(window=20, min_periods=1).mean()
    ratio = vol_series / sma.replace({0: np.nan})
    momentum = vol_series.pct_change(periods=5).fillna(0.0)
    return pd.DataFrame(
        {
            "volume_sma_20": sma,
            "volume_ratio": ratio,
            "volume_momentum": momentum,
        }
    )
