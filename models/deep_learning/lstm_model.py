"""CONTEXT: LSTM architecture for sequential market signal prediction.

REQUIREMENTS:
- Prepare sliding-window sequences (30 days of features).
- Normalize features with StandardScaler.
- Build two-layer LSTM with dropout and dense head.
- Train with categorical cross-entropy, Adam optimizer, and callbacks.
- Provide predict/save/load helpers returning probability vectors.

TECHNICAL SPECIFICATIONS:
- Programming language: Python 3.9+
- Framework: TensorFlow / Keras
- Design pattern: Class encapsulating preprocessing + model lifecycle.
- Error handling: Validate shapes and raise ValueError on insufficient data.

INPUT/OUTPUT:
- Input: Feature DataFrame with categorical targets (BUY/SELL/HOLD).
- Output: Sequence tensors, trained model predictions, saved .h5 artifacts.

EXAMPLE USAGE:
```python
from models.deep_learning.lstm_model import LSTMSequenceModel

model = LSTMSequenceModel()
X_train, y_train, scaler, encoder = model.prepare_sequences(train_df)
model.build_model(input_shape=X_train.shape[1:], num_classes=3)
model.train(X_train, y_train, X_val, y_val)
```

TESTING:
- Validate sequence creation when dataset shorter than 30 steps.
- Confirm save/load cycle.

CODE STYLE:
- Type hints + docstrings.
- 88 character limit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras


class LSTMSequenceModel:
    """Encapsulates preprocessing + LSTM training loop."""

    def __init__(self, sequence_length: int = 30) -> None:
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model: keras.Model | None = None

    def prepare_sequences(
        self,
        data: pd.DataFrame,
        target_column: str = "signal",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create normalized sequences."""
        if target_column not in data.columns:
            raise ValueError(f"{target_column} column missing.")
        features = data.drop(columns=[target_column])
        targets = data[target_column]
        scaled = self.scaler.fit_transform(features.values)
        encoded_targets = self.label_encoder.fit_transform(targets.values)
        X: list[np.ndarray] = []
        y: list[int] = []
        for idx in range(self.sequence_length, len(features)):
            X.append(scaled[idx - self.sequence_length : idx])
            y.append(encoded_targets[idx])
        if not X:
            raise ValueError("Insufficient rows for sequence creation.")
        X_arr = np.array(X)
        y_arr = keras.utils.to_categorical(y, num_classes=len(self.label_encoder.classes_))
        return X_arr, y_arr

    def build_model(self, input_shape: Tuple[int, int], num_classes: int) -> keras.Model:
        """Construct the LSTM network."""
        inputs = keras.layers.Input(shape=input_shape)
        x = keras.layers.LSTM(64, return_sequences=True)(inputs)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.LSTM(32)(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> keras.callbacks.History:
        """Fit the model with early stopping + checkpoint."""
        if self.model is None:
            raise ValueError("Call build_model before training.")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                filepath="data/models/lstm_best.h5",
                monitor="val_loss",
                save_best_only=True,
            ),
        ]
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return history

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return probability predictions."""
        if self.model is None:
            raise ValueError("Model not built.")
        return self.model.predict(X_test, verbose=0)

    def save_model(self, filepath: Path) -> None:
        """Persist model weights to disk."""
        if self.model is None:
            raise ValueError("Model not available.")
        self.model.save(filepath)

    def load_model(self, filepath: Path) -> None:
        """Load stored model weights."""
        self.model = keras.models.load_model(filepath)
