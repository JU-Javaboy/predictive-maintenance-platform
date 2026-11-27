"""Constants shared by the C1 anomaly detector.

``FEATURES`` is frozen: it is the exact column order the saved RandomForest was
trained on. The evidence columns computed for the quality report must never be
added to it.

The per-mode medians are not here: they live in ``imputation_params.json``, next
to ``quality.py``, so data cleaning and anomaly detection read the same values.
Everything the detector needs that the cleaning settings do not contain (the
z-score scales fitted on the training slice, the sensor resolutions and the
fleet share lookup) is a constant below.
"""

from pathlib import Path

import pandas as pd


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` until the repository root is found.

    The package may also be installed from a wheel (Docker); in that case there
    is no repository root and the current working directory is the fallback.
    Deployments should set the artifact paths explicitly through the
    ``C1_ANOMALY_MODEL_PATH`` and ``C1_IMPUTATION_PARAMS_PATH`` variables.
    """
    for base in (start, *start.parents):
        if (base / "pyproject.toml").is_file() and (base / "data").is_dir():
            return base
    return Path.cwd()


# Service layout: <repo>/services/c1-sensor-quality/src/c1_sensor_quality/anomaly_detector
REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "anomaly_detector_rf.pkl"

OPERATING_MODES = ["NORMAL", "HIGH_LOAD", "IDLE", "OVERLOAD"]
SENSORS = ["pressure", "rpm", "temperature", "vibration", "power"]

OPERATING_MODE_CLASSES = {mode: index for index, mode in enumerate(OPERATING_MODES)}

#: Labels produced by the trained classifier.
ANOMALY_TYPES = ["normal", "sensor_frozen", "sensor_spike"]

#: Exact model input, in the order expected by the saved classifier.
FEATURES = [
    *[f"z_score_{sensor}" for sensor in SENSORS],
    "repeated_recently_any",
    "operating_mode_class",
    "resolution_gap_max",
]

#: Per-mode scale used with the medians from ``imputation_params.json`` to build
#: the z-scores the classifier was trained on. Fit on the training slice.
MODE_SENSOR_SCALES = {
    "NORMAL": {
        "pressure": 2.396497499590686,
        "rpm": 25.186262879783143,
        "temperature": 11.068097286475194,
        "vibration": 2.413352730897005,
        "power": 15.766554474168501,
    },
    "HIGH_LOAD": {
        "pressure": 2.404953763971218,
        "rpm": 24.891443614401403,
        "temperature": 11.098892456082067,
        "vibration": 2.4163267382199516,
        "power": 15.800736615145281,
    },
    "IDLE": {
        "pressure": 2.3997126655362018,
        "rpm": 24.998563979466475,
        "temperature": 10.98101959960603,
        "vibration": 2.408662675919117,
        "power": 15.724157162087536,
    },
    "OVERLOAD": {
        "pressure": 2.3938877934387155,
        "rpm": 24.771198161615423,
        "temperature": 10.961150237244917,
        "vibration": 2.4167123624928104,
        "power": 15.746947681326466,
    },
}

#: Decimals normally exposed by each sensor. A reading with fewer decimals is
#: on a coarser grid, which is the frozen-sensor signal.
RESOLUTION_BASELINE = {
    "pressure": 12,
    "rpm": 11,
    "temperature": 13,
    "vibration": 14,
    "power": 12,
}

#: Number of training machines that emitted the same value. Report evidence
#: only: a value absent from the lookup is assumed to come from one machine.
SHARE_LOOKUP = {"vibration": {5.0: 20}}
SHARE_DEFAULT = 1

REPEAT_WINDOW = pd.Timedelta(days=7)
