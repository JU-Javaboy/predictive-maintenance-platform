"""Deterministic fallback for predictions the model is not confident about.

The RandomForest is the primary detector. These rules only run on rows whose
highest class probability does not clear the caller's confidence threshold.
They read the prepared frame, so they reuse the exact per-mode z-scores and the
exact sensor precision the model saw.

Two rules are implemented:

* ``sensor_frozen``: a reading exposes fewer than ``FROZEN_MAX_DECIMALS``
  decimals, i.e. it lies on a coarser grid than the sensor normally emits.
  The notebook confirms the signal: every frozen vibration is ``5.0`` (one
  decimal) while normal readings expose at least 11.
* ``sensor_spike``: a z-score leaves the band for that sensor. The z-score is
  already relative to the operating mode's median, so one band covers every
  mode.

Rows no rule recognises keep ``anomaly_type`` as ``NaN``: the caller keeps the
model's own prediction. When both rules fire, frozen wins because the
resolution loss is the more specific signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import (
    ANOMALY_TYPES,
    FROZEN_MAX_DECIMALS,
    GUARDRAIL_CONFIDENCE,
    PREDICTION_COLUMNS,
    SENSORS,
    ZSCORE_BANDS,
)
from .features import decimal_places


def _require_columns(frame: pd.DataFrame) -> None:
    """Reject raw frames: the rules need the prepared z-scores and sensors."""
    required = [
        *SENSORS,
        *[f"z_score_{sensor}" for sensor in SENSORS],
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "guardrails_prediction expects a prepared frame; missing column(s): "
            + ", ".join(missing)
        )


def _spike_mask(frame: pd.DataFrame) -> np.ndarray:
    """True where a sensor z-score leaves its band."""
    mask = np.zeros(len(frame), dtype=bool)
    for sensor, (low, high) in ZSCORE_BANDS.items():
        z_scores = frame[f"z_score_{sensor}"].to_numpy(dtype=float)
        mask |= (z_scores < low) | (z_scores > high)
    return mask


def _frozen_mask(frame: pd.DataFrame) -> np.ndarray:
    """True where a sensor lost precision but still reports a plausible value.

    The plausibility check matters: a spike can land on a round value (say
    ``1000.0``), which would otherwise look exactly like a frozen reading.
    A coarse reading whose own z-score is out of band is a spike, not a freeze.
    """
    mask = np.zeros(len(frame), dtype=bool)
    for sensor, (low, high) in ZSCORE_BANDS.items():
        decimals = decimal_places(frame[sensor]).to_numpy()
        # -1 means no reading, so there is no precision to judge.
        coarse = (decimals >= 0) & (decimals < FROZEN_MAX_DECIMALS)
        z_scores = frame[f"z_score_{sensor}"].to_numpy(dtype=float)
        plausible = ~((z_scores < low) | (z_scores > high))
        mask |= coarse & plausible
    return mask


def _assign(results: pd.DataFrame, mask: np.ndarray, anomaly_type: str) -> None:
    """Write one guardrail verdict and its probabilities on the masked rows."""
    if not mask.any():
        return

    probabilities = {label: 0.0 for label in ANOMALY_TYPES}
    probabilities["normal"] = 1.0 - GUARDRAIL_CONFIDENCE
    probabilities[anomaly_type] = GUARDRAIL_CONFIDENCE

    results.loc[mask, "anomaly_type"] = anomaly_type
    results.loc[mask, "anomaly_confidence"] = GUARDRAIL_CONFIDENCE
    for label, probability in probabilities.items():
        results.loc[mask, f"probability_{label}"] = probability


def guardrails_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a rule-based prediction for every row a guardrail recognises.

    ``frame`` must be a prepared frame (raw sensors and the ``z_score_*``
    columns), as returned by ``features.prepare_frame``. The result keeps the
    input index and the same columns as the model prediction. Rows where no rule
    fires have ``anomaly_type`` set to ``NaN``; the caller must keep the model's
    prediction for those.

    A frozen reading also wins over a spike: the precision loss is the more
    specific signal, while the z-score of a coarse reading is still plausible.
    """
    _require_columns(frame)

    results = pd.DataFrame(
        {
            "anomaly_type": pd.Series(index=frame.index, dtype="object"),
            "anomaly_confidence": pd.Series(index=frame.index, dtype="float64"),
            **{
                f"probability_{label}": pd.Series(index=frame.index, dtype="float64")
                for label in ANOMALY_TYPES
            },
        },
        index=frame.index,
        columns=PREDICTION_COLUMNS,
    )

    frozen = _frozen_mask(frame)
    spike = _spike_mask(frame) & ~frozen

    _assign(results, frozen, "sensor_frozen")
    _assign(results, spike, "sensor_spike")
    return results
