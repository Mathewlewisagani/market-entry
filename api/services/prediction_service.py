"""CONTEXT: Service layer orchestrating model predictions for the API.

REQUIREMENTS:
- Load cached model artifacts and reuse across requests.
- Fetch recent price data, engineer features, and generate predictions.
- Persist predictions into ``ml_predictions`` table.
- Provide history and performance helper functions.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: Flask, SQLAlchemy, pandas
- Design pattern: Service object encapsulating business logic.
- Error handling: Raise informative exceptions for missing models/data.

INPUT/OUTPUT:
- Input: Symbol identifiers and optional time filters.
- Output: Prediction payloads and analytics dictionaries.

EXAMPLE USAGE:
```python
service = PredictionService()
payload = service.predict_symbol(\"AAPL\")
```

TESTING:
- Ensure DB writes succeed with mocked connections.
- Validate fallback when model artifact missing.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from backtesting.backtesting_engine import BacktestingEngine
from config.database import ENGINE, get_session
from config.settings import get_config
from features.feature_engineering import build_feature_matrix
from models.baseline_models import LogisticRegressionModel
from sklearn.exceptions import NotFittedError

LOGGER = logging.getLogger(__name__)


class PredictionService:
    """Encapsulates prediction workflows."""

    def __init__(self) -> None:
        self.config = get_config()
        self.model = self._load_model()
        self.model_version = self.model.version
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.backtester = BacktestingEngine()

    def _load_model(self) -> LogisticRegressionModel:
        model = LogisticRegressionModel()
        artifact = Path("data/models") / f"{model.model_name}_{model.version}.joblib"
        if artifact.exists():
            model.load(artifact)
        else:
            LOGGER.warning("Model artifact %s missing.", artifact)
        return model

    def _latest_features(self, symbol: str) -> pd.Series:
        features = build_feature_matrix(symbol)
        if features.empty:
            raise ValueError(f"No features generated for {symbol}.")
        latest = features.iloc[-1]
        return latest

    def _feature_vector(self, series: pd.Series) -> pd.DataFrame:
        feature_cols = [
            col
            for col in series.index
            if col
            not in {"symbol", "timestamp", "forward_return", "signal"}
        ]
        return pd.DataFrame([series[feature_cols].to_dict()]), feature_cols

    def predict_symbol(self, symbol: str) -> Dict[str, object]:
        """Generate a single symbol prediction."""
        latest_series = self._latest_features(symbol)
        feature_frame, feature_cols = self._feature_vector(latest_series)
        try:
            proba = self.model.predict_proba(feature_frame)[0]
        except NotFittedError as exc:
            raise ValueError("Model not trained. Train and save prior to predictions.") from exc
        classes = self.model.model.classes_
        max_index = int(np.argmax(proba))
        signal = str(classes[max_index])
        payload = {
            "symbol": symbol,
            "timestamp": latest_series["timestamp"].isoformat(),
            "signal": signal,
            "confidence": float(proba[max_index]),
            "current_price": float(latest_series["close"]),
            "model_version": self.model_version,
        }
        self._store_prediction(payload, feature_cols, feature_frame.iloc[0].to_dict())
        return payload

    def _store_prediction(
        self,
        payload: Dict[str, object],
        feature_names: List[str],
        feature_values: Dict[str, float],
    ) -> None:
        # Use bindparam with JSONB type for proper casting
        insert_sql = text(
            """
            INSERT INTO ml_predictions
            (symbol, timestamp, prediction_date, signal, confidence,
             model_version, features)
            VALUES (:symbol, :timestamp, :prediction_date, :signal, :confidence,
                    :model_version, CAST(:features AS JSONB))
            """
        )
        features_json = json.dumps({"feature_names": feature_names, "values": feature_values})
        db_payload = {
            "symbol": payload["symbol"],
            "timestamp": payload["timestamp"],
            "prediction_date": datetime.utcnow().isoformat(),
            "signal": payload["signal"],
            "confidence": payload["confidence"],
            "model_version": payload["model_version"],
            "features": features_json,
        }
        with get_session() as session:
            session.execute(insert_sql, db_payload)

    def batch_predict(self, symbols: List[str]) -> List[Dict[str, object]]:
        """Predict multiple symbols concurrently."""
        futures = [self.executor.submit(self.predict_symbol, symbol) for symbol in symbols]
        results: List[Dict[str, object]] = []
        for future in futures:
            results.append(future.result())
        return results

    def prediction_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, object]]:
        """Fetch stored predictions."""
        query = """
            SELECT symbol, timestamp, prediction_date, signal, confidence,
                   model_version, features
            FROM ml_predictions
            WHERE symbol = :symbol
            {start_clause}
            {end_clause}
            ORDER BY prediction_date DESC
            LIMIT :limit
        """
        start_clause = "AND prediction_date >= :start_date" if start_date else ""
        end_clause = "AND prediction_date <= :end_date" if end_date else ""
        sql = query.format(start_clause=start_clause, end_clause=end_clause)
        params = {"symbol": symbol, "limit": limit}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        records = pd.read_sql_query(text(sql), ENGINE, params=params)
        return records.to_dict(orient="records")

    def performance(self, symbol: str) -> Dict[str, object]:
        """Run backtest using stored predictions."""
        history = self.prediction_history(symbol, limit=500)
        if not history:
            raise ValueError("No prediction history for performance run.")
        history_df = pd.DataFrame(history)
        price_query = text(
            """
            SELECT timestamp, close
            FROM stock_prices
            WHERE symbol = :symbol AND timestamp BETWEEN :start AND :end
            ORDER BY timestamp
            """
        )
        start = pd.to_datetime(history_df["timestamp"].min())
        end = pd.to_datetime(history_df["timestamp"].max()) + timedelta(days=1)
        prices = pd.read_sql_query(
            price_query,
            ENGINE,
            params={"symbol": symbol, "start": start, "end": end},
        )
        merged = pd.merge_asof(
            prices.sort_values("timestamp"),
            history_df[["timestamp", "signal"]],
            on="timestamp",
            direction="backward",
        ).dropna()
        if merged.empty:
            raise ValueError("Insufficient overlapping price/prediction data.")
        results = self.backtester.run(symbol, merged)
        results["equity_curve"] = results["equity_curve"].to_dict(orient="records")
        results["trades_log"] = results["trades_log"].to_dict(orient="records")
        return results
