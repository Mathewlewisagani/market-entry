"""Train baseline models on engineered features."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_models import (
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from models.target_engineering import prepare_ml_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def train_all_models(symbols: list[str]) -> None:
    """Train and evaluate baseline models for the supplied symbols."""
    splits = prepare_ml_dataset(symbols=symbols)
    LOGGER.info("Training set: %s samples", len(splits.X_train))
    LOGGER.info("Validation set: %s samples", len(splits.X_val))
    LOGGER.info("Test set: %s samples", len(splits.X_test))
    LOGGER.info("Features: %s", len(splits.feature_names))
    models = [
        LogisticRegressionModel(),
        RandomForestModel(),
        XGBoostModel(),
    ]
    comparison: dict[str, dict[str, float]] = {}
    for model in models:
        LOGGER.info("\n%s", "=" * 50)
        LOGGER.info("Training %s ...", model.model_name)
        LOGGER.info("%s", "=" * 50)
        model.train(splits.X_train, splits.y_train, splits.X_val, splits.y_val)
        train_metrics = model.evaluate(splits.X_train, splits.y_train)
        val_metrics = model.evaluate(splits.X_val, splits.y_val)
        test_metrics = model.evaluate(splits.X_test, splits.y_test)
        artifact = Path(f"data/models/{model.model_name}_{model.version}.joblib")
        model.save(artifact)
        LOGGER.info("✅ %s saved to %s", model.model_name, artifact)
        LOGGER.info("Train Accuracy: %.3f", train_metrics["accuracy"])
        LOGGER.info("Val Accuracy: %.3f", val_metrics["accuracy"])
        LOGGER.info("Test Accuracy: %.3f", test_metrics["accuracy"])
        comparison[model.model_name] = test_metrics
    LOGGER.info("\n%s", "=" * 60)
    LOGGER.info("MODEL COMPARISON (Test Set)")
    LOGGER.info("%s", "=" * 60)
    for name, metrics in comparison.items():
        LOGGER.info(
            "%-20s | Acc: %.3f | F1: %.3f",
            name,
            metrics.get("accuracy", 0.0),
            metrics.get("f1", 0.0),
        )
    LOGGER.info("%s", "=" * 60)


if __name__ == "__main__":
    DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    train_all_models(DEFAULT_SYMBOLS)
