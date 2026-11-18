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