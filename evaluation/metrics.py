"""CONTEXT: Evaluation utilities for classification and trading performance.

REQUIREMENTS:
- Provide standard classification metrics and confusion matrix.
- Support ROC-AUC computation using probability outputs.
- Calculate profit/risk metrics for a $10K portfolio strategy.
- Expose helper functions for drawdown and win rate.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: numpy, pandas, scikit-learn
- Design pattern: Functional utilities.
- Error handling: Validate shapes and raise ValueError on mismatch.

INPUT/OUTPUT:
- Input: True labels, predictions, probabilities, returns.
- Output: Dict structures summarizing metrics.

EXAMPLE USAGE:
```python
from evaluation.metrics import classification_report

report = classification_report(y_true, y_pred)
```

TESTING:
- Ensure binary and multiclass cases.
- Validate portfolio metrics with synthetic returns.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Dict[str, float]:
    """Compute accuracy/precision/recall/F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


def confusion_matrix_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, list]:
    """Return confusion matrix as nested lists."""
    labels = labels or sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": labels, "matrix": matrix.tolist()}


def roc_auc_multiclass(
    y_true: Sequence[str],
    y_prob: np.ndarray,
    labels: List[str],
) -> float:
    """Compute multi-class ROC-AUC using one-vs-rest strategy."""
    if y_prob.shape[1] != len(labels):
        raise ValueError("Probability array shape mismatch with labels.")
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y_encoded = [label_to_index[label] for label in y_true]
    return roc_auc_score(y_encoded, y_prob, multi_class="ovr")


def _compute_equity_curve(
    signals: Sequence[str],
    returns: Sequence[float],
    initial_capital: float,
) -> pd.Series:
    """Construct an equity curve based on discrete signals."""
    capital = initial_capital
    equity = []
    position = 0
    for signal, ret in zip(signals, returns):
        if signal == "BUY":
            position = 1
        elif signal == "SELL":
            position = -1
        else:
            position = 0
        capital *= 1 + (position * ret)
        equity.append(capital)
    return pd.Series(equity)


def portfolio_metrics(
    signals: Sequence[str],
    returns: Sequence[float],
    initial_capital: float = 10_000.0,
) -> Dict[str, float]:
    """Calculate P&L and derived risk metrics."""
    equity_curve = _compute_equity_curve(signals, returns, initial_capital)
    pnl = equity_curve.iloc[-1] - initial_capital if not equity_curve.empty else 0.0
    total_return = pnl / initial_capital if initial_capital else 0.0
    max_dd = max_drawdown(equity_curve)
    win_rate_value = win_rate(returns, signals)
    return {
        "equity_curve": equity_curve.tolist(),
        "ending_value": float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_capital,
        "pnl": float(pnl),
        "total_return_pct": float(total_return * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "win_rate_pct": float(win_rate_value * 100),
    }


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Largest peak-to-trough decline."""
    curve = pd.Series(equity_curve)
    if curve.empty:
        return 0.0
    rolling_max = curve.cummax()
    drawdown = (curve - rolling_max) / rolling_max
    return float(drawdown.min())


def win_rate(returns: Sequence[float], signals: Sequence[str]) -> float:
    """Percentage of profitable trades."""
    wins = 0
    trades = 0
    for signal, ret in zip(signals, returns):
        if signal in {"BUY", "SELL"}:
            trades += 1
            if signal == "BUY" and ret > 0:
                wins += 1
            elif signal == "SELL" and ret < 0:
                wins += 1
    return wins / trades if trades else 0.0


def profit_factor(returns: Sequence[float], signals: Sequence[str]) -> float:
    """Ratio of gross profits to gross losses."""
    gross_profit = 0.0
    gross_loss = 0.0
    for signal, ret in zip(signals, returns):
        if signal == "BUY" and ret > 0:
            gross_profit += ret
        elif signal == "SELL" and ret < 0:
            gross_profit += abs(ret)
        else:
            gross_loss += abs(ret)
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss
