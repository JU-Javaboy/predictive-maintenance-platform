"""Load the trained C1 anomaly detector and run inference on raw batches.

The saved RandomForest only knows the eight columns in ``FEATURES``. It needs
the frozen parameters from ``imputation_params.json`` and ``constants.py``, plus
the causal ``MachineRepeatIndex`` that carries the 7-day same-machine history.
This module owns both so callers only provide a raw DataFrame.
"""

from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import DEFAULT_IMPUTATION_PARAMS_PATH
from .constants import (
    ANOMALY_TYPES,
    DEFAULT_MODEL_PATH,
    FEATURES,
    REPEAT_WINDOW,
    SENSORS,
)
from .features import load_imputation_params, prepare_frame, validate_imputation_params
from .utils.values_repeated import MachineRepeatIndex

MODEL_PATH_ENV = "C1_ANOMALY_MODEL_PATH"
IMPUTATION_PARAMS_PATH_ENV = "C1_IMPUTATION_PARAMS_PATH"


def _resolve_path(explicit: str | Path | None, env_var: str, default: Path) -> Path:
    """Explicit argument first, then environment override, then repository default."""
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get(env_var, default))


class AnomalyDetector:
    """Stateful wrapper around the trained model and its frozen preprocessing.

    ``predict`` (or ``score``) must be called in chronological order because the
    repeat index only looks at the past. A batch that starts before the previous
    batch raises ``ValueError`` instead of silently corrupting the history.

    The detector is thread-safe: inference requests are serialised while the
    repeat index is updated, then scored. One detector should be reused for the
    whole stream; ``reset_history`` starts a new one.
    """

    def __init__(
        self,
        model: Any,
        imputation_params: dict[str, Any],
        repeat_index: MachineRepeatIndex | None = None,
    ):
        validate_imputation_params(imputation_params)

        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None and list(model_features) != list(FEATURES):
            raise ValueError(
                "The model was trained on different features than FEATURES. "
                f"Model: {list(model_features)}. Code: {FEATURES}."
            )

        labels = [str(label) for label in getattr(model, "classes_", ANOMALY_TYPES)]
        unknown_labels = set(labels) - set(ANOMALY_TYPES)
        if unknown_labels:
            raise ValueError(f"Model produces unknown labels: {sorted(unknown_labels)}")

        self.model = model
        self.imputation_params = imputation_params
        self.labels = labels
        self.repeat_index = repeat_index or MachineRepeatIndex(SENSORS, REPEAT_WINDOW)
        self._last_timestamp: pd.Timestamp | None = None
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        model_path: str | Path | None = None,
        imputation_params_path: str | Path | None = None,
    ) -> AnomalyDetector:
        """Load the pickled model and the shared imputation settings from disk.

        Paths can be passed explicitly or overridden with the
        ``C1_ANOMALY_MODEL_PATH`` and ``C1_IMPUTATION_PARAMS_PATH``
        environment variables. Only load trusted pickle files.
        """
        resolved_model = _resolve_path(model_path, MODEL_PATH_ENV, DEFAULT_MODEL_PATH)
        resolved_params = _resolve_path(
            imputation_params_path, IMPUTATION_PARAMS_PATH_ENV, DEFAULT_IMPUTATION_PARAMS_PATH
        )

        if not resolved_model.is_file():
            raise FileNotFoundError(
                f"Anomaly model not found: {resolved_model}. Train it or set C1_ANOMALY_MODEL_PATH."
            )
        with resolved_model.open("rb") as file:
            model = pickle.load(file)

        imputation_params = load_imputation_params(resolved_params)
        return cls(model, imputation_params)

    def reset_history(self) -> None:
        """Forget the same-machine repetition history (start of a new stream)."""
        with self._lock:
            self.repeat_index = MachineRepeatIndex(SENSORS, REPEAT_WINDOW)
            self._last_timestamp = None

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Update the repeat history and return the full prepared frame."""
        with self._lock:
            timestamps = (
                pd.to_datetime(frame["timestamp"]) if "timestamp" in frame.columns else None
            )
            if timestamps is not None and self._last_timestamp is not None:
                earliest = timestamps.min()
                if pd.notna(earliest) and earliest < self._last_timestamp:
                    raise ValueError(
                        "Batches must be scored in chronological order: this batch starts "
                        f"at {earliest}, before the previous batch ended at "
                        f"{self._last_timestamp}. Call reset_history() to start over."
                    )

            prepared = prepare_frame(frame, self.imputation_params, self.repeat_index)

            if timestamps is not None and len(timestamps) > 0:
                self._last_timestamp = timestamps.max()
            return prepared

    def _predict_prepared(self, prepared: pd.DataFrame) -> pd.DataFrame:
        """Score an already-prepared frame. Does not touch the repeat history."""
        columns = [
            "anomaly_type",
            "anomaly_confidence",
            *[f"probability_{label}" for label in self.labels],
        ]
        if prepared.empty:
            return pd.DataFrame(columns=columns, index=prepared.index)

        probabilities = self.model.predict_proba(prepared[FEATURES])
        labels = np.asarray(self.labels, dtype=object)

        result = pd.DataFrame(
            {
                "anomaly_type": labels[probabilities.argmax(axis=1)],
                "anomaly_confidence": probabilities.max(axis=1),
            },
            index=prepared.index,
        )
        for position, label in enumerate(self.labels):
            result[f"probability_{label}"] = probabilities[:, position]
        return result

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return one prediction per input row.

        Columns: ``anomaly_type`` (predicted label), ``anomaly_confidence``
        (probability of that label) and ``probability_<label>`` for each class.
        The returned index is the input index.
        """
        prepared = self._prepare(frame)
        return self._predict_prepared(prepared)

    def score(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the prepared frame (features + evidence) with its predictions.

        Prediction columns are prefixed so a ground-truth ``anomaly_type``
        column in the input is preserved for evaluation.
        """
        prepared = self._prepare(frame)
        predictions = self._predict_prepared(prepared).rename(
            columns={
                "anomaly_type": "predicted_anomaly_type",
                "anomaly_confidence": "predicted_confidence",
            }
        )
        return prepared.join(predictions)
