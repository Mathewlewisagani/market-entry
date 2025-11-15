"""CONTEXT: Cleansing pipeline for extracted OHLCV datasets.

REQUIREMENTS:
- Deduplicate timestamps and normalize timezone to UTC.
- Handle missing values (ffill for prices, interpolate for volume).
- Remove weekends and US market holidays.
- Validate OHLC relationships and drop invalid rows.
- Detect outliers via IQR and exclude them.
- Return cleaned DataFrame plus a quality report.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: pandas
- Design pattern: Functional transformer
- Error handling: Raise ValueError on critical validation failures.

INPUT/OUTPUT:
- Input: pandas.DataFrame with timestamp/open/high/low/close/volume.
- Output: Tuple[pd.DataFrame, dict] containing cleaned data and metrics.

EXAMPLE USAGE:
```python
from etl.transformers.data_cleaner import clean_ohlcv_data

cleaned, report = clean_ohlcv_data(raw_df)
```

TESTING:
- Verify weekend removal.
- Confirm outlier logic drops extreme spikes.
- Ensure timezone normalization.

CODE STYLE:
- Type hints + docstrings.
- 88 character line limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class QualityReport:
    """Track data quality metrics."""

    initial_rows: int
    duplicates_removed: int
    weekends_removed: int
    holidays_removed: int
    invalid_prices_removed: int
    outliers_removed: int
    final_rows: int

    def as_dict(self) -> Dict[str, int]:
        """Return a serializable view."""
        return self.__dict__.copy()


def _remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    deduped = df.drop_duplicates(subset=["timestamp"])
    return deduped, before - len(deduped)


def _normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp")


def _remove_weekends(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    filtered = df[df["timestamp"].dt.weekday < 5]
    return filtered, before - len(filtered)


def _remove_holidays(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    cal = USFederalHolidayCalendar()
    start = df["timestamp"].min().date()
    end = df["timestamp"].max().date()
    holidays = cal.holidays(start=start, end=end)
    before = len(df)
    filtered = df[~df["timestamp"].dt.normalize().isin(holidays)]
    return filtered, before - len(filtered)


def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].ffill()
    df["volume"] = df["volume"].interpolate(limit_direction="both")
    return df.dropna(subset=price_cols)


def _validate_prices(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    mask = (
        (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df[["open", "high", "low", "close"]] >= 0).all(axis=1)
    )
    filtered = df[mask]
    return filtered, before - len(filtered)


def _remove_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    q1 = df["close"].quantile(0.25)
    q3 = df["close"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before = len(df)
    filtered = df[(df["close"] >= lower) & (df["close"] <= upper)]
    return filtered, before - len(filtered)


def clean_ohlcv_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Run the full cleaning pipeline."""
    required = {"timestamp", *NUMERIC_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    initial_rows = len(df)
    df = _normalize_timezone(df.copy())
    df, duplicates_removed = _remove_duplicates(df)
    df, weekends_removed = _remove_weekends(df)
    df, holidays_removed = _remove_holidays(df)
    df = _fill_missing_values(df)
    df, invalid_prices_removed = _validate_prices(df)
    df, outliers_removed = _remove_outliers(df)
    report = QualityReport(
        initial_rows=initial_rows,
        duplicates_removed=duplicates_removed,
        weekends_removed=weekends_removed,
        holidays_removed=holidays_removed,
        invalid_prices_removed=invalid_prices_removed,
        outliers_removed=outliers_removed,
        final_rows=len(df),
    )
    return df.reset_index(drop=True), report.as_dict()
