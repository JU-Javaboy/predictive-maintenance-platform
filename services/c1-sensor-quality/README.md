# C1: Sensor Quality Service

```bash
uv run --directory services/c1-sensor-quality uvicorn c1_sensor_quality.main:app --reload
```

Planned endpoints include data validation, quality reports, and model
information.

## Clean the raw dataset

Run these from the project root:

```bash
git lfs pull                                  # get the raw CSV (stored with Git LFS)
uv sync --all-packages                        # install everything
uv run python scripts/clean_raw_data.py       # clean the data
```

- Input: `data/raw/predictive_maintenance_data.csv`
- Output: `data/processed/cleaned_predictive_maintenance_data.csv`
- The output has the 18 original columns, plus `quality_disposition` (valid or repaired)
  and `quality_flags` (what we found or changed in that row).
- The script prints a summary. If the data breaks a rule, it stops with a clear
  message and saves no file.
- Cleaning settings (thresholds, limits, medians) are in
  `src/c1_sensor_quality/imputation_params.json`.

## Anomaly detection

The inference code is in `src/c1_sensor_quality/anomaly_detector/`. It uses:

- `models/anomaly_detector_rf.pkl` - the trained classifier;
- `src/c1_sensor_quality/imputation_params.json` - the shared per-mode medians
  (the same file the cleaning rules use);
- `src/c1_sensor_quality/anomaly_detector/constants.py` - the z-score scales,
  sensor resolutions and fleet-share lookup.

Inference never recomputes a statistic from live data.

Score a raw batch. Rows must contain `timestamp`, `machine_id`,
`operating_mode` and the five sensor columns:

```python
import pandas as pd
from c1_sensor_quality.anomaly_detector import detect_anomaly

batch = pd.read_csv("batch.csv", parse_dates=["timestamp"])
predictions = detect_anomaly(batch)   # anomaly_type + probabilities
# anomaly_type, anomaly_confidence, probability_normal,
# probability_sensor_frozen, probability_sensor_spike
```

`detect_anomaly` reuses one detector, so call it in chronological order: the
7-day same-machine repetition history is causal. Call
`get_detector().reset_history()` to start a new stream.

For the features and reporting evidence alongside the prediction, use
`AnomalyDetector.score`:

```python
from c1_sensor_quality.anomaly_detector import AnomalyDetector

detector = AnomalyDetector.load()
scored = detector.score(batch)
# prepared features/evidence + predicted_anomaly_type, predicted_confidence
```

`AnomalyDetector.load` looks for the model in `models/`. Override its location
with `C1_ANOMALY_MODEL_PATH` and the shared settings file with
`C1_IMPUTATION_PARAMS_PATH`.
