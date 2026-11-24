"""Behavioural tests for the anomaly-detector preprocessing and inference.

The synthetic frames are deliberately small, but they keep the properties the
notebook relies on: normal readings carry many decimals, a frozen reading is
coarse, and the same-machine repeat index must ignore other machines.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from c1_sensor_quality.anomaly_detector import (
    ANOMALY_TYPES,
    SENSORS,
    AnomalyDetector,
)
from c1_sensor_quality.anomaly_detector.constants import (
    DEFAULT_MODEL_PATH,
    FEATURES,
    OPERATING_MODES,
    SHARE_LOOKUP,
)
from c1_sensor_quality.anomaly_detector.features import (
    decimal_places,
    load_imputation_params,
    prepare_frame,
    transform,
)
from c1_sensor_quality.anomaly_detector.utils.values_repeated import MachineRepeatIndex


def _training_frame() -> pd.DataFrame:
    """Four modes, two normal machines each, all sensor values distinct."""
    rows = []
    for mode in OPERATING_MODES:
        for machine in ("M001", "M002"):
            index = len(rows)
            rows.append(
                {
                    "timestamp": pd.Timestamp("2012-01-01") + pd.Timedelta(hours=index),
                    "machine_id": machine,
                    "operating_mode": mode,
                    "pressure": 100.123456789 + index,
                    "rpm": 1500.123456789 + index,
                    "temperature": 50.123456789 + index,
                    "vibration": 2.123456789 + index,
                    "power": 200.123456789 + index,
                    "anomaly_type": "normal",
                }
            )
    return pd.DataFrame(rows)


def _row(
    timestamp: str,
    machine: str,
    vibration: float,
    *,
    mode: str = "NORMAL",
    index: int = 0,
) -> dict[str, Any]:
    """One raw reading; ``index`` keeps the non-target sensors distinct."""
    offset = index * 0.000000001
    return {
        "timestamp": pd.Timestamp(timestamp),
        "machine_id": machine,
        "operating_mode": mode,
        "pressure": 100.123456789 + offset,
        "rpm": 1500.123456789 + offset,
        "temperature": 50.123456789 + offset,
        "vibration": vibration,
        "power": 200.123456789 + offset,
    }


def _prepared(frame: pd.DataFrame, imputation_params: dict[str, Any]) -> pd.DataFrame:
    """Prepare a single batch with a fresh (cold-start) repeat history."""
    return prepare_frame(frame, imputation_params, MachineRepeatIndex(SENSORS))


def _imputation_params() -> dict[str, Any]:
    """The subset of the shared settings the feature builder reads.

    The pressure medians are the midpoints of ``_training_frame`` so every mode
    is centred on zero. Scales live in ``constants.py``; the real medians come
    from ``imputation_params.json``.
    """
    pressure_medians = {
        "NORMAL": 100.623456789,
        "HIGH_LOAD": 102.623456789,
        "IDLE": 104.623456789,
        "OVERLOAD": 106.623456789,
    }
    shared_medians = {
        "rpm": 1500.123456789,
        "temperature": 50.123456789,
        "vibration": 2.123456789,
        "power": 200.123456789,
    }
    return {
        "mode_sensor_medians": {
            mode: {"pressure": pressure, **shared_medians}
            for mode, pressure in pressure_medians.items()
        }
    }


def _write_imputation_params(imputation_params: dict[str, Any], path: Path) -> Path:
    path.write_text(json.dumps(imputation_params), encoding="utf-8")
    return path


@pytest.fixture()
def imputation_params() -> dict[str, Any]:
    return _imputation_params()


class _RuleModel:
    """Tiny stand-in for the RandomForest: frozen when the resolution gap is large."""

    feature_names_in_ = np.asarray(FEATURES)
    classes_ = np.asarray(ANOMALY_TYPES)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        gap = np.clip(frame["resolution_gap_max"].to_numpy(dtype=float), 0.0, 1.0)
        probabilities = np.zeros((len(frame), len(self.classes_)))
        probabilities[:, 0] = 1.0 - gap
        probabilities[:, 1] = gap
        return probabilities


class _WrongFeaturesModel(_RuleModel):
    feature_names_in_ = np.asarray(["not_a_real_feature"])


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def test_decimal_places_counts_the_exposed_resolution() -> None:
    values = pd.Series([5.0, 4.6823187526, np.nan, 1.23e-05, 100.0])

    assert decimal_places(values).tolist() == [1, 10, -1, 7, 1]


def test_prepare_frame_does_not_mutate_the_input(imputation_params: dict[str, Any]) -> None:
    frame = _training_frame()
    before = frame.copy(deep=True)

    _prepared(frame, imputation_params)

    pd.testing.assert_frame_equal(frame, before)


def test_z_scores_are_centred_per_mode_and_react_to_a_spike(
    imputation_params: dict[str, Any],
) -> None:
    prepared = _prepared(_training_frame(), imputation_params)
    by_mode = prepared.groupby("operating_mode")["z_score_pressure"].median()
    assert np.allclose(by_mode.to_numpy(), 0.0, atol=1e-9)

    spike = _row("2012-02-01 00:00:00", "M001", 2.0)
    spike["temperature"] = 95.0
    prepared_spike = _prepared(pd.DataFrame([spike]), imputation_params)
    assert prepared_spike["z_score_temperature"].iloc[0] > 3.0


def test_repeat_index_counts_only_the_same_machine(imputation_params: dict[str, Any]) -> None:
    frame = pd.DataFrame(
        [
            _row("2012-02-01 00:00:00", "M001", 5.0, index=0),
            _row("2012-02-01 01:00:00", "M001", 5.0, index=1),
            _row("2012-02-01 01:00:00", "M002", 5.0, index=2),
            _row("2012-02-01 02:00:00", "M002", 5.0, index=3),
        ]
    )

    prepared = _prepared(frame, imputation_params)

    assert prepared["n_previous_same_machine_vibration"].tolist() == [0, 1, 0, 1]
    assert prepared["repeated_recently_vibration"].tolist() == [0, 1, 0, 1]


def test_unsorted_rows_and_duplicate_index_stay_aligned(
    imputation_params: dict[str, Any],
) -> None:
    frame = pd.DataFrame(
        [
            _row("2012-02-01 01:00:00", "M001", 5.0, index=1),  # later, first in input
            _row("2012-02-01 00:00:00", "M001", 5.0, index=0),  # earlier
        ],
        index=[7, 7],
    )

    prepared = _prepared(frame, imputation_params)

    assert prepared.index.tolist() == [7, 7]
    assert prepared["n_previous_same_machine_vibration"].tolist() == [1, 0]


def test_repeat_index_honours_the_seven_day_window(
    imputation_params: dict[str, Any],
) -> None:
    inside = pd.DataFrame(
        [
            _row("2012-02-01 00:00:00", "M001", 5.0),
            _row("2012-02-07 23:00:00", "M001", 5.0),
        ]
    )
    assert _prepared(inside, imputation_params)["repeated_recently_vibration"].tolist() == [0, 1]

    expired = pd.DataFrame(
        [
            _row("2012-02-01 00:00:00", "M001", 5.0),
            _row("2012-02-08 01:00:00", "M001", 5.0),
        ]
    )
    assert _prepared(expired, imputation_params)["repeated_recently_vibration"].tolist() == [0, 0]


def test_share_columns_report_the_training_fleet() -> None:
    # The frozen lookup says this vibration value was shared by 20 machines.
    fitted = _imputation_params()

    prepared = transform(pd.DataFrame([_row("2012-02-01", "M001", 5.0)]), fitted)

    assert prepared["share_vibration"].iloc[0] == SHARE_LOOKUP["vibration"][5.0]
    assert prepared["n_machines_sharing_value"].iloc[0] == SHARE_LOOKUP["vibration"][5.0]


def test_frozen_reading_gets_a_large_resolution_gap(
    imputation_params: dict[str, Any],
) -> None:
    prepared = _prepared(pd.DataFrame([_row("2012-02-01", "M001", 5.0)]), imputation_params)

    assert prepared["resolution_gap_vibration"].iloc[0] > 0
    assert prepared["resolution_gap_max"].iloc[0] > 0.5


def test_unknown_operating_mode_is_rejected(imputation_params: dict[str, Any]) -> None:
    frame = pd.DataFrame([_row("2012-02-01", "M001", 2.0)])
    frame.loc[0, "operating_mode"] = "TURBO"

    with pytest.raises(ValueError, match="Unknown operating_mode"):
        transform(frame, imputation_params)


def test_missing_input_column_has_a_readable_error(
    imputation_params: dict[str, Any],
) -> None:
    frame = pd.DataFrame([_row("2012-02-01", "M001", 2.0)]).drop(columns=["machine_id"])

    with pytest.raises(ValueError, match="Missing required column"):
        _prepared(frame, imputation_params)


def test_incomplete_imputation_params_are_rejected(
    imputation_params: dict[str, Any], tmp_path: Path
) -> None:
    incomplete = {
        "mode_sensor_medians": {
            mode: dict(medians)
            for mode, medians in imputation_params["mode_sensor_medians"].items()
        }
    }
    del incomplete["mode_sensor_medians"]["NORMAL"]["vibration"]
    path = tmp_path / "imputation_params.json"
    path.write_text(json.dumps(incomplete), encoding="utf-8")

    with pytest.raises(ValueError, match="missing sensor"):
        load_imputation_params(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_predict_returns_labels_and_probabilities(
    imputation_params: dict[str, Any],
) -> None:
    detector = AnomalyDetector(_RuleModel(), imputation_params)
    frame = pd.DataFrame(
        [
            _row("2012-02-01 00:00:00", "M001", 2.123456789, index=0),
            _row("2012-02-01 01:00:00", "M001", 5.0, index=1),
        ],
        index=[10, 20],
    )

    result = detector.predict(frame)

    assert list(result.columns) == [
        "anomaly_type",
        "anomaly_confidence",
        "probability_normal",
        "probability_sensor_frozen",
        "probability_sensor_spike",
    ]
    assert result.index.tolist() == [10, 20]
    assert result["anomaly_type"].tolist() == ["normal", "sensor_frozen"]
    probabilities = result[["probability_normal", "probability_sensor_frozen"]].to_numpy()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_empty_batch_returns_an_empty_prediction_frame(
    imputation_params: dict[str, Any],
) -> None:
    detector = AnomalyDetector(_RuleModel(), imputation_params)
    empty = pd.DataFrame(columns=["timestamp", "machine_id", "operating_mode", *SENSORS])

    result = detector.predict(empty)

    assert result.empty
    assert "probability_normal" in result.columns


def test_score_keeps_ground_truth_and_adds_evidence(
    imputation_params: dict[str, Any],
) -> None:
    detector = AnomalyDetector(_RuleModel(), imputation_params)
    frame = pd.DataFrame([_row("2012-02-01", "M001", 5.0)])
    frame["anomaly_type"] = "normal"

    scored = detector.score(frame)

    assert scored["anomaly_type"].iloc[0] == "normal"
    assert scored["predicted_anomaly_type"].iloc[0] == "sensor_frozen"
    assert "resolution_gap_max" in scored.columns
    assert "n_previous_same_machine_vibration" in scored.columns


def test_batches_must_arrive_in_chronological_order(
    imputation_params: dict[str, Any],
) -> None:
    detector = AnomalyDetector(_RuleModel(), imputation_params)
    detector.predict(pd.DataFrame([_row("2012-03-01", "M001", 2.123456789)]))

    with pytest.raises(ValueError, match="chronological"):
        detector.predict(pd.DataFrame([_row("2012-02-01", "M001", 2.123456789)]))

    detector.reset_history()
    assert len(detector.predict(pd.DataFrame([_row("2012-02-01", "M001", 2.123456789)]))) == 1


def test_model_with_other_features_is_rejected(
    imputation_params: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="different features"):
        AnomalyDetector(_WrongFeaturesModel(), imputation_params)


def test_detector_loads_from_explicit_paths(
    imputation_params: dict[str, Any], tmp_path: Path
) -> None:
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as file:
        pickle.dump(_RuleModel(), file)
    imputation_params_path = _write_imputation_params(
        imputation_params, tmp_path / "imputation_params.json"
    )

    detector = AnomalyDetector.load(
        model_path=model_path, imputation_params_path=imputation_params_path
    )

    assert detector.labels == ANOMALY_TYPES
    frame = pd.DataFrame([_row("2012-02-01", "M001", 5.0)])
    assert detector.predict(frame)["anomaly_type"].iloc[0] == "sensor_frozen"


def test_missing_model_file_has_a_readable_error(
    imputation_params: dict[str, Any], tmp_path: Path
) -> None:
    imputation_params_path = _write_imputation_params(
        imputation_params, tmp_path / "imputation_params.json"
    )

    with pytest.raises(FileNotFoundError, match="Anomaly model not found"):
        AnomalyDetector.load(
            model_path=tmp_path / "missing.pkl",
            imputation_params_path=imputation_params_path,
        )


@pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.is_file(),
    reason="trained model is not present",
)
def test_saved_model_runs_end_to_end() -> None:
    detector = AnomalyDetector.load()
    frame = pd.DataFrame([_row("2012-02-01", "M001", 5.0)])

    result = detector.predict(frame)

    assert result["anomaly_type"].iloc[0] in ANOMALY_TYPES
    assert 0.0 <= result["anomaly_confidence"].iloc[0] <= 1.0
