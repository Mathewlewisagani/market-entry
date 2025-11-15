"""CONTEXT: Baseline ML models for directional signal prediction.

REQUIREMENTS:
- Define BaseModel with serialization helpers and abstract contract.
- Implement Logistic Regression, Random Forest, and XGBoost classifiers.
- Provide predict/predict_proba along with evaluation logging.
- Persist trained models under ``data/models``.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: scikit-learn, xgboost
- Design pattern: Abstract Base Class with concrete implementations
- Error handling: Validate training data and raise ValueError on misuse.

INPUT/OUTPUT:
- Input: Training/validation datasets from target engineering.
- Output: Predictions, probability scores, serialized artifacts.

EXAMPLE USAGE:
```python
from models.baseline_models import LogisticRegressionModel

model = LogisticRegressionModel()
model.train(X_train, y_train, X_val, y_val)
preds = model.predict(X_test)
```

TESTING:
- Ensure serialization/deserialization cycles.
- Validate evaluate() returns metrics dict.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from evaluation.metrics import classification_report

LOGGER = logging.getLogger(__name__)
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class BaseModel(ABC):
    """Abstract baseline model definition."""

    def __init__(self, model_name: str, version: str = "v1.0") -> None:
        self.model_name = model_name
        self.version = version
        self.trained_at: Optional[datetime] = None
        self.model = None

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate label predictions."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability estimates for each class."""

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict[str, float]:
        """Return classification metrics for dataset."""
        preds = self.predict(X)
        metrics = classification_report(y_true, preds)
        LOGGER.info("%s evaluation metrics: %s", self.model_name, metrics)
        return metrics

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Persist trained model artifact."""
        if self.model is None:
            raise ValueError("Train the model before saving.")
        path = filepath or MODEL_DIR / f"{self.model_name}_{self.version}.joblib"
        joblib.dump(
            {
                "model": self.model,
                "trained_at": self.trained_at,
                "version": self.version,
            },
            path,
        )
        LOGGER.info("Saved model artifact to %s", path)
        return path

    def load(self, filepath: Path) -> None:
        """Load serialized artifact."""
        payload = joblib.load(filepath)
        self.model = payload["model"]
        self.trained_at = payload.get("trained_at")
        self.version = payload.get("version", self.version)
        LOGGER.info("Loaded model %s from %s", self.model_name, filepath)


class LogisticRegressionModel(BaseModel):
    """Regularized logistic regression classifier."""

    def __init__(self, version: str = "v1.0") -> None:
        super().__init__("logistic_regression", version)
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        self.model.fit(X_train, y_train)
        self.trained_at = datetime.utcnow()
        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """Random Forest baseline with feature importance."""

    def __init__(self, version: str = "v1.0") -> None:
        super().__init__("random_forest", version)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> None:
        self.model.fit(X_train, y_train)
        self.trained_at = datetime.utcnow()
        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_


class XGBoostModel(BaseModel):
    """Gradient boosted classifier (XGBoost)."""

    def __init__(self, version: str = "v1.0") -> None:
        super().__init__("xgboost_classifier", version)
        self.model = XGBClassifier(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: int = 10,
    ) -> None:
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
        )
        self.trained_at = datetime.utcnow()
        if X_val is not None and y_val is not None:
            self.evaluate(X_val, y_val)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
