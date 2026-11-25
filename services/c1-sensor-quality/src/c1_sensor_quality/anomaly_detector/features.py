"""Inference-time feature engineering for the C1 anomaly detector.

The per-mode sensor medians come from ``imputation_params.json``, the same file
the cleaning rules use. The z-score scales, sensor resolutions and fleet-share
lookup are fitted values that the cleaning settings do not contain, so they live
in ``constants.py``. This module only applies frozen values; it never fits or
retrains anything.

1. ``load_imputation_params`` reads the shared settings and checks that every
   operating mode and sensor has a median.
2. ``prepare_frame`` applies those values and adds the causal same-machine
   repetition features. Reuse the same ``MachineRepeatIndex`` across
   chronological batches so the 7-day history carries over.

The saved classifier's contract still contains ``resolution_gap_max``, so that
aggregate is kept exactly as the model was trained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import config
from .constants import (
    MODE_SENSOR_SCALES,
    OPERATING_MODE_CLASSES,
    OPERATING_MODES,
    RESOLUTION_BASELINE,
    SENSORS,
    SHARE_DEFAULT,
    SHARE_LOOKUP,
)
from .utils.values_repeated import MachineRepeatIndex

#: Columns required to build features from a raw batch.
REQUIRED_COLUMNS = ["timestamp", "machine_id", "operating_mode", *SENSORS]


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise a readable error when a raw batch is missing an input column."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def validate_imputation_params(params: dict[str, Any]) -> None:
    """Reject settings that do not describe every mode and sensor."""
    medians = params.get("mode_sensor_medians")
    if not isinstance(medians, dict):
        raise TypeError("imputation_params.json is missing the 'mode_sensor_medians' object.")

    missing_modes = set(OPERATING_MODES) - set(medians)
    if missing_modes:
        raise ValueError(
            f"mode_sensor_medians is missing operating mode(s): {sorted(missing_modes)}"
        )

    unknown_modes = set(medians) - set(MODE_SENSOR_SCALES)
    if unknown_modes:
        raise ValueError(
            f"MODE_SENSOR_SCALES has no entry for operating mode(s): {sorted(unknown_modes)}"
        )

    for mode in OPERATING_MODES:
        missing_sensors = set(SENSORS) - set(medians[mode])
        if missing_sensors:
            raise ValueError(
                f"mode_sensor_medians[{mode!r}] is missing sensor(s): {sorted(missing_sensors)}"
            )


def decimal_places(values: pd.Series) -> pd.Series:
    """Number of decimals in the shortest round-trip representation of a float.

    ``5.0`` -> 1, ``4.6823187526`` -> 10. This is the resolution the sensor
    actually exposes: a value with fewer decimals lies on a coarser grid.
    NaN -> -1 (no reading, so there is no resolution to compare).

    The value has already been parsed as a float upstream, so trailing zeros
    are gone: this measures the resolution of the number, not the wire format.
    """
    text = values.astype(str)
    decimals = text.str.split(".", n=1, regex=False).str[1].str.len().fillna(0)

    # Very large or very small values are rendered as 1.23e-05. The resolution
    # is the exponent plus the digits of the mantissa, not just the exponent.
    scientific = text.str.contains("e|E", regex=True)
    if scientific.any():
        exponent = text.str.extract(r"[eE]([+-]?\d+)$")[0].astype(float)
        mantissa = text.str.extract(r"\.(\d+)")[0].str.len().fillna(0)
        decimals = decimals.where(~scientific, (-exponent + mantissa).clip(lower=0))

    out = decimals.astype("int16")
    out[values.isna()] = -1
    return out


def load_imputation_params(
    path: str | Path = config.DEFAULT_IMPUTATION_PARAMS_PATH,
) -> dict[str, Any]:
    """Load the shared settings and check the anomaly detector can use them."""
    params = config.load_imputation_params(path)
    validate_imputation_params(params)
    return params


def transform(frame: pd.DataFrame, imputation_params: dict[str, Any]) -> pd.DataFrame:
    """Apply the frozen parameters. No statistic is recomputed from ``frame``."""
    validate_imputation_params(imputation_params)
    _require_columns(frame, ["operating_mode", *SENSORS])

    out = frame.copy()

    mode_medians = imputation_params["mode_sensor_medians"]
    unknown = set(out["operating_mode"].astype(str)) - set(mode_medians)
    if unknown:
        raise ValueError(f"Unknown operating_mode(s): {sorted(unknown)}")

    z_cols = [f"z_score_{sensor}" for sensor in SENSORS]
    out[z_cols] = np.nan
    for mode, medians in mode_medians.items():
        mask = out["operating_mode"].astype(str) == mode
        median = pd.Series(medians).loc[SENSORS]
        scale = pd.Series(MODE_SENSOR_SCALES[mode]).loc[SENSORS]
        out.loc[mask, z_cols] = ((out.loc[mask, SENSORS] - median) / scale).to_numpy()

    # Evidence only. A value with fewer decimals than the sensor normally
    # exposes lies on a coarser grid. This assumes that sensor_frozen implies a
    # precision drop: true for the training generator (frozen = literal 5.0),
    # not knowable for future data (a real freeze can replay a full-precision
    # value). The per-sensor gaps are evidence only; their maximum is part of
    # the saved model's feature contract (see ``FEATURES``).
    for sensor in SENSORS:
        decimals = decimal_places(out[sensor])
        baseline = RESOLUTION_BASELINE[sensor]
        gap = (baseline - decimals).clip(lower=0)
        gap[decimals < 0] = 0  # missing reading: no resolution to compare
        out[f"resolution_gap_{sensor}"] = gap.astype("int16")

    # Coarsest sensor, relative to its own normal resolution. The ratio is
    # required: baselines differ per sensor, so a raw max would be dominated by
    # the sensor that normally exposes the most decimals. ``max(baseline, 1)``
    # keeps a fully-integer sensor from dividing by zero; its gap is 0 anyway.
    gap_ratio = [
        out[f"resolution_gap_{sensor}"] / max(RESOLUTION_BASELINE[sensor], 1) for sensor in SENSORS
    ]
    out["resolution_gap_max"] = pd.concat(gap_ratio, axis=1).max(axis=1)

    # Report only: a human can read "this value was seen on 20 machines in the
    # training fleet". This is a fleet property, not a frozen-sensor signal.
    share_cols = []
    for sensor in SENSORS:
        column = f"share_{sensor}"
        lookup = SHARE_LOOKUP.get(sensor, {})
        out[column] = out[sensor].map(lookup).fillna(SHARE_DEFAULT).astype(int)
        share_cols.append(column)
    out["n_machines_sharing_value"] = out[share_cols].max(axis=1)

    return out


def prepare_frame(
    frame: pd.DataFrame,
    imputation_params: dict[str, Any],
    repeat_index: MachineRepeatIndex,
) -> pd.DataFrame:
    """Transform a raw slice and add the causal same-machine repetition features.

    Reuse the same ``repeat_index`` for successive chronological slices so the
    7-day history carries over. A row is ``repeated_recently`` when the same
    machine already emitted the same value in the window.

    ``frame`` is never mutated.
    """
    _require_columns(frame, REQUIRED_COLUMNS)

    prepared_input = frame.copy()
    prepared_input["timestamp"] = pd.to_datetime(prepared_input["timestamp"])

    out = transform(prepared_input, imputation_params)

    prior = repeat_index.transform(prepared_input)

    for sensor in SENSORS:
        out[f"repeated_recently_{sensor}"] = (
            prior[f"n_previous_same_machine_{sensor}"] > 0
        ).astype("int8")
        # Evidence column: how many previous same-machine readings matched.
        out[f"n_previous_same_machine_{sensor}"] = prior[f"n_previous_same_machine_{sensor}"]

    # Any sensor that repeated a value on this machine. The flag is applied per
    # sensor first, so one sensor's history does not depend on another's.
    repeat_cols = [f"repeated_recently_{sensor}" for sensor in SENSORS]
    out["repeated_recently_any"] = out[repeat_cols].max(axis=1).astype("int8")

    out["operating_mode_class"] = out["operating_mode"].map(OPERATING_MODE_CLASSES).astype("int8")
    return out
