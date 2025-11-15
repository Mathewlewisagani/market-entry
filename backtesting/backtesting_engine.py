"""CONTEXT: Backtesting engine to validate trading strategies.

REQUIREMENTS:
- Simulate trading with BUY/SELL/HOLD signals on $10k portfolio.
- Apply 0.1% transaction cost per trade and track cash/positions.
- Produce equity curve, trades log, and performance metrics.
- Compare strategy returns to buy-and-hold benchmark.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: pandas, numpy
- Design pattern: Stateful simulator class.
- Error handling: Validate required columns before running.

INPUT/OUTPUT:
- Input: DataFrame with timestamp, close, and signal columns.
- Output: Dict containing equity_curve, trades_log, metrics.

EXAMPLE USAGE:
```python
from backtesting.backtesting_engine import BacktestingEngine

engine = BacktestingEngine()
results = engine.run(symbol=\"AAPL\", data=predict_df)
```

TESTING:
- Validate trade execution order.
- Confirm benchmark vs. strategy metrics.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from backtesting.performance_metrics import (
    calmar_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of executed trade."""

    timestamp: pd.Timestamp
    action: str
    shares: int
    price: float
    cost: float
    pnl: float


@dataclass
class PortfolioState:
    """Track portfolio balances."""

    cash: float = 10_000.0
    shares: int = 0
    transaction_cost: float = 0.001
    trades: List[Trade] = field(default_factory=list)
    avg_entry_price: float = 0.0

    def total_value(self, price: float) -> float:
        return self.cash + (self.shares * price)


class BacktestingEngine:
    """Execute trades based on predicted signals."""

    def __init__(self, transaction_cost: float = 0.001) -> None:
        self.transaction_cost = transaction_cost

    def _execute_trade(
        self,
    ) -> None:
        """Placeholder to satisfy type checker (logic inline in run)."""

    def run(self, symbol: str, data: pd.DataFrame) -> Dict[str, pd.DataFrame | dict]:
        """Simulate daily trading strategy."""
        required = {"timestamp", "close", "signal"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Backtest data missing {missing}")
        df = data.sort_values("timestamp").reset_index(drop=True)
        state = PortfolioState(transaction_cost=self.transaction_cost)
        equity_curve: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            price = float(row["close"])
            signal = row["signal"]
            timestamp = pd.to_datetime(row["timestamp"], utc=True)
            if signal == "BUY" and state.cash > 0:
                shares_to_buy = int(state.cash / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    fee = cost * self.transaction_cost
                    state.cash -= cost + fee
                    state.shares += shares_to_buy
                    state.avg_entry_price = price
                    state.trades.append(
                        Trade(timestamp, "BUY", shares_to_buy, price, fee, 0.0)
                    )
            elif signal == "SELL" and state.shares > 0:
                proceeds = state.shares * price
                fee = proceeds * self.transaction_cost
                pnl = (price - state.avg_entry_price) * state.shares - fee
                state.cash += proceeds - fee
                state.trades.append(
                    Trade(timestamp, "SELL", state.shares, price, fee, pnl)
                )
                state.shares = 0
                state.avg_entry_price = 0.0
            equity_curve.append(
                {"timestamp": timestamp, "portfolio_value": state.total_value(price)}
            )
        # liquidate at end
        if state.shares > 0:
            final_price = float(df.iloc[-1]["close"])
            proceeds = state.shares * final_price
            fee = proceeds * self.transaction_cost
            pnl = (final_price - state.avg_entry_price) * state.shares - fee
            state.cash += proceeds - fee
            state.trades.append(
                Trade(df.iloc[-1]["timestamp"], "SELL", state.shares, final_price, fee, pnl)
            )
            state.shares = 0
            state.avg_entry_price = 0.0
            equity_curve[-1]["portfolio_value"] = state.total_value(final_price)
        equity_df = pd.DataFrame(equity_curve)
        returns = equity_df["portfolio_value"].pct_change().fillna(0.0)
        drawdown = max_drawdown(equity_df["portfolio_value"])
        trades_df = pd.DataFrame([trade.__dict__ for trade in state.trades])
        metrics = {
            "total_return_pct": (
                (equity_df["portfolio_value"].iloc[-1] / equity_df["portfolio_value"].iloc[0])
                - 1
            )
            * 100,
            "sharpe_ratio": sharpe_ratio(returns),
            "sortino_ratio": sortino_ratio(returns),
            "max_drawdown_pct": drawdown * 100,
            "calmar_ratio": calmar_ratio(returns, drawdown),
            "win_rate_pct": win_rate(trades_df) * 100,
            "profit_factor": profit_factor(trades_df),
            "num_trades": len(trades_df),
            "buy_and_hold_return_pct": (
                (df["close"].iloc[-1] / df["close"].iloc[0]) - 1
            )
            * 100,
        }
        LOGGER.info("Backtest complete for %s: %s", symbol, metrics)
        return {
            "equity_curve": equity_df,
            "trades_log": trades_df,
            "metrics": metrics,
        }
