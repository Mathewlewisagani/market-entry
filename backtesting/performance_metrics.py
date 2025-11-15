"""CONTEXT: Portfolio performance metrics for trading strategy evaluation.

REQUIREMENTS:
- Compute Sharpe, Sortino, Calmar ratios, and maximum drawdown.
- Provide win rate and profit factor utilities.
- Accept generic pandas objects for flexibility.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: pandas/numpy
- Design pattern: Functional helpers.
- Error handling: Return 0 when insufficient data.

INPUT/OUTPUT:
- Input: Series/DataFrames of returns, equity curves, or trades.
- Output: Floats summarizing risk/performance ratios.

EXAMPLE USAGE:
```python
from backtesting.performance_metrics import sharpe_ratio

ratio = sharpe_ratio(daily_returns)
```

TESTING:
- Validate behaviour with flat/incomplete data.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _to_series(data: Sequence[float] | pd.Series) -> pd.Series:
    return data if isinstance(data, pd.Series) else pd.Series(data)


def sharpe_ratio(
    returns: Sequence[float] | pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Sharpe ratio using annualized returns."""
    series = _to_series(returns).dropna()
    if series.empty:
        return 0.0
    excess = series - (risk_free_rate / periods_per_year)
    std_dev = series.std()
    if std_dev == 0:
        return 0.0
    return (excess.mean() / std_dev) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: Sequence[float] | pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Sortino ratio using downside volatility."""
    series = _to_series(returns).dropna()
    if series.empty:
        return 0.0
    downside = series[series < 0]
    downside_std = downside.std() or 1e-9
    excess = series.mean() - (risk_free_rate / periods_per_year)
    return (excess / downside_std) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: Sequence[float] | pd.Series) -> float:
    """Largest drawdown percentage."""
    curve = _to_series(equity_curve)
    if curve.empty:
        return 0.0
    cumulative_max = curve.cummax()
    drawdowns = (curve - cumulative_max) / cumulative_max.replace(0, np.nan)
    return float(drawdowns.min())


def calmar_ratio(
    returns: Sequence[float] | pd.Series,
    drawdown: float,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio using annualized return over max drawdown."""
    series = _to_series(returns).dropna()
    if series.empty or drawdown == 0:
        return 0.0
    annual_return = (1 + series.mean()) ** periods_per_year - 1
    return annual_return / abs(drawdown)


def win_rate(trades: pd.DataFrame) -> float:
    """Percent of profitable trades (expects 'pnl' column)."""
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    wins = (trades["pnl"] > 0).sum()
    return wins / len(trades)


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross profit divided by gross loss."""
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = trades.loc[trades["pnl"] < 0, "pnl"].abs().sum()
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)
