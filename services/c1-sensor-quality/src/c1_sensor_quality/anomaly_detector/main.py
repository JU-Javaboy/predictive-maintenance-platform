"""Public entry points for the C1 anomaly detector.

Typical use::

    from c1_sensor_quality.anomaly_detector import detect_anomaly

    predictions = detect_anomaly(raw_batch)

``raw_batch`` must contain ``timestamp``, ``machine_id``, ``operating_mode``
and the five sensor columns. Predictions are returned in the input order.
"""

from __future__ import annotations

from functools import lru_cache

from c1_sensor_quality.anomaly_detector.guardrails import guardrails_prediction
import pandas as pd

from .inference import AnomalyDetector


@lru_cache(maxsize=1)
def get_detector() -> AnomalyDetector:
    """Load the default detector once and reuse its repeat history."""
    return AnomalyDetector.load()


def detect_anomaly(
    df: pd.DataFrame,
    detector: AnomalyDetector | None = None,
) -> pd.DataFrame:
    """Predict the anomaly type of every row in ``df``.

    The first call loads ``models/anomaly_detector_rf.pkl`` and the shared
    imputation settings, then keeps the 7-day same-machine history for later
    calls.
    Pass an explicit ``detector`` when you need another model or a fresh state.
    """
    active_detector = detector if detector is not None else get_detector()
    predictions = active_detector.predict(df)

    return predictions
