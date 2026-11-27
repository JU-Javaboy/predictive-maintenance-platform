"""Tests for the deterministic guardrails used when the model is unsure.

The rules read a prepared frame, so the tests build one with the real scales
from ``constants.py`` and the same per-mode medians shape as the settings file.
"""

from typing import Any

import numpy as np
import pandas as pd
from c1_sensor_quality.anomaly_detector import ANOMALY_TYPES, SENSORS, AnomalyDetector
from c1_sensor_quality.anomaly_detector.constants import (
    FEATURES,
    GUARDRAIL_CONFIDENCE,
    OPERATING_MODES,
)
from c1_sensor_quality.anomaly_detector.features import prepare_frame
from c1_sensor_quality.anomaly_detector.guardrails import guardrails_prediction
from c1_sensor_quality.anomaly_detector.utils.values_repeated import MachineRepeatIndex

NORMAL = {
    "pressure": 96.591234567,
    "rpm": 1510.861234567,
    "temperature": 52.561234567,
    "vibration": 4.541234567,
    "power": 223.511234567,
}


def _imputation_params() -> dict[str, Any]:
    return {"mode_sensor_medians": {mode: dict(NORMAL) for mode in OPERATING_MODES}}


def _row(
    timestamp: str,
    *,
    mode: str = "NORMAL",
    machine: str = "M001",
    **overrides: float,
) -> dict[str, Any]:
    return {
        "timestamp": pd.Timestamp(timestamp),
        "machine_id": machine,
        "operating_mode": mode,
        **NORMAL,
        **overrides,
    }


def _prepared(rows: list[dict[str, Any]], index: list[int] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(rows, index=index)
    return prepare_frame(frame, _imputation_params(), MachineRepeatIndex(SENSORS))


def _probabilities(results: pd.DataFrame) -> pd.DataFrame:
    return results[[f"probability_{label}" for label in ANOMALY_TYPES]]


def test_normal_reading_matches_no_rule() -> None:
    prepared = _prepared([_row("2012-02-01 00:00:00")])

    results = guardrails_prediction(prepared)

    assert results["anomaly_type"].isna().all()
    assert results["anomaly_confidence"].isna().all()


def test_out_of_band_z_score_is_a_spike() -> None:
    prepared = _prepared([_row("2012-02-01 00:00:00", temperature=1000.123456789)])

    results = guardrails_prediction(prepared)

    assert results["anomaly_type"].tolist() == ["sensor_spike"]
    assert results["anomaly_confidence"].tolist() == [GUARDRAIL_CONFIDENCE]
    assert np.allclose(_probabilities(results).sum(axis=1), 1.0)


def test_coarse_reading_is_frozen() -> None:
    prepared = _prepared([_row("2012-02-01 00:00:00", vibration=5.0)])

    results = guardrails_prediction(prepared)

    assert results["anomaly_type"].tolist() == ["sensor_frozen"]
    assert results["probability_sensor_frozen"].tolist() == [GUARDRAIL_CONFIDENCE]


def test_round_spike_is_not_mistaken_for_a_frozen_reading() -> None:
    # 1000.0 exposes one decimal, but its z-score is far outside the band.
    prepared = _prepared([_row("2012-02-01 00:00:00", temperature=1000.0)])

    results = guardrails_prediction(prepared)

    assert results["anomaly_type"].tolist() == ["sensor_spike"]


def test_guardrails_keep_the_input_index_and_ignore_missing_readings() -> None:
    prepared = _prepared(
        [
            _row("2012-02-01 00:00:00"),
            _row("2012-02-01 01:00:00", vibration=float("nan")),
            _row("2012-02-01 02:00:00", vibration=5.0),
        ],
        index=[10, 20, 30],
    )

    results = guardrails_prediction(prepared)

    assert results.index.tolist() == [10, 20, 30]
    assert results["anomaly_type"].isna().tolist() == [True, True, False]
    assert results.loc[30, "anomaly_type"] == "sensor_frozen"


class _UnsureModel:
    """Return a three-way tie so every row takes the low-confidence path."""

    feature_names_in_ = np.asarray(FEATURES)
    classes_ = np.asarray(ANOMALY_TYPES)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile([0.34, 0.33, 0.33], (len(frame), 1))


def test_predict_replaces_only_the_low_confidence_rows_a_rule_recognises() -> None:
    raw = pd.DataFrame(
        [
            _row("2012-02-01 00:00:00"),
            _row("2012-02-01 01:00:00", vibration=5.0),
            _row("2012-02-01 02:00:00", temperature=1000.123456789),
        ],
        index=[10, 20, 30],
    )
    detector = AnomalyDetector(_UnsureModel(), _imputation_params())

    results = detector.predict(raw)

    assert results.index.tolist() == [10, 20, 30]
    # No rule fires on the normal row: the model's own verdict is kept.
    assert results.loc[10, "anomaly_type"] == "normal"
    assert results.loc[10, "anomaly_confidence"] == 0.34
    assert results.loc[20, "anomaly_type"] == "sensor_frozen"
    assert results.loc[30, "anomaly_type"] == "sensor_spike"
