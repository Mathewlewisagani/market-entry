"""CONTEXT: Target generation and time-series splits for modeling.

REQUIREMENTS:
- Create BUY/SELL/HOLD signals based on 5-day forward returns.
- Provide regression targets equals continuous forward return.
- Handle imbalance via optional SMOTE and class weights.
- Produce time-ordered train/val/test splits without leakage.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: pandas, scikit-learn, imbalanced-learn
- Design pattern: Functional helpers returning structured dicts.
- Error handling: Validate dataset length and missing columns.

INPUT/OUTPUT:
- Input: Feature DataFrame with ``forward_return`` column.
- Output: Dict containing splits, targets, feature names, distributions.

EXAMPLE USAGE:
```python
from models.target_engineering import prepare_datasets

splits = prepare_datasets(features_df, use_smote=True)
```

TESTING:
- Confirm split proportions and ordering.
- Validate SMOTE only affects training set.

CODE STYLE:
- Type hints + docstrings.
- 88 char limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

Signal = str


@dataclass
class DatasetSplits:
    """Container for split datasets."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    regression_targets: Dict[str, pd.Series]
    feature_names: List[str]
    class_distribution: Dict[Signal, int]
    class_weights: Optional[Dict[Signal, float]]


def _assign_signals(forward_returns: pd.Series) -> pd.Series:
    buy = forward_returns > 0.02
    sell = forward_returns < -0.02
    return pd.Series(
        np.select([buy, sell], ["BUY", "SELL"], default="HOLD"),
        index=forward_returns.index,
    )


def _split_indices(length: int) -> tuple[int, int]:
    train_end = max(int(length * 0.7), 1)
    val_end = max(train_end + int(length * 0.15), train_end + 1)
    val_end = min(val_end, length - 1)
    return train_end, val_end


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"timestamp", "symbol", "signal", "forward_return"}
    return [col for col in df.columns if col not in excluded]


def prepare_datasets(
    features: pd.DataFrame,
    use_smote: bool = False,
    use_class_weights: bool = False,
    random_state: int = 42,
) -> DatasetSplits:
    """Generate targets and perform time-series split."""
    required = {"forward_return", "timestamp"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Features missing {missing}")
    ordered = features.sort_values("timestamp").reset_index(drop=True)
    ordered["signal"] = _assign_signals(ordered["forward_return"])
    class_distribution = ordered["signal"].value_counts().to_dict()
    feature_names = _select_feature_columns(ordered)
    train_end, val_end = _split_indices(len(ordered))
    X_train = ordered.iloc[:train_end][feature_names]
    X_val = ordered.iloc[train_end:val_end][feature_names]
    X_test = ordered.iloc[val_end:][feature_names]
    y_train = ordered.iloc[:train_end]["signal"]
    y_val = ordered.iloc[train_end:val_end]["signal"]
    y_test = ordered.iloc[val_end:]["signal"]
    regression_targets = {
        "y_train": ordered.iloc[:train_end]["forward_return"],
        "y_val": ordered.iloc[train_end:val_end]["forward_return"],
        "y_test": ordered.iloc[val_end:]["forward_return"],
    }
    class_weights = None
    if use_class_weights:
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train,
        )
        class_weights = dict(zip(classes, weights))
    if use_smote and len(X_train) > 0:
        sampler = SMOTE(random_state=random_state)
        X_train, y_train = sampler.fit_resample(X_train, y_train)
    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        regression_targets=regression_targets,
        feature_names=feature_names,
        class_distribution=class_distribution,
        class_weights=class_weights,
    )
