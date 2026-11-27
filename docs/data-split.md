# Dataset split

The dataset is split chronologically by `timestamp`. There is one
implementation: `chronological_split` in
`services/c1-sensor-quality/src/c1_sensor_quality/split.py`. Notebooks and
training code call it instead of defining their own dates.

## Boundaries

Every interval is half-open: the start is included and the end is excluded.

| Partition  | Start (inclusive)   | End (exclusive)     | Rows    | Duration              |
| ---------- | ------------------- | ------------------- | ------- | --------------------- |
| train      | 2012-01-14 00:00:00 | 2013-07-15 00:00:00 | 263,040 | 18 months (548 days)  |
| validation | 2013-07-15 00:00:00 | 2014-01-14 00:00:00 | 87,840  | 6 months (183 days)   |
| test       | 2014-01-14 00:00:00 | 2015-01-13 00:00:00 | 174,720 | 12 months (364 days)  |

The dataset contains 525,600 hourly rows for 20 machines. Its first timestamp
is 2012-01-14 00:00 and its last is 2015-01-12 23:00. The three partitions are
contiguous: together they contain every row exactly once.

The first two years (train + validation) may be used for model training,
feature engineering, model selection and validation. The final year is reserved
exclusively for testing: no test row may be used to fit, tune or select a model
or any preprocessing parameter.

## How the boundaries are computed

The dates are fixed because the dataset is fixed:

- training starts at the first timestamp. Its 18-month anniversary is
  2013-07-14 00:00; the whole 2013-07-14 day stays in training, so the
  exclusive end is 2013-07-15 00:00;
- validation covers the remaining 6 months of the first two years and ends at
  2014-01-14 00:00;
- the test year starts two years after the first timestamp and ends one day
  after the last timestamp (2015-01-12 23:00).

Do not derive the test start from `max(timestamp) - 1 year`: the maximum is
23:00 on the last day, which moves the boundary one day earlier and overlaps
the end of the training/validation period.

## Usage

```python
from c1_sensor_quality.split import chronological_split

train, validation, test = chronological_split(df)
```

The function:

- rejects missing, invalid and timezone-aware timestamps instead of guessing;
- sorts each partition by `timestamp`, then `machine_id` when present;
- does not modify the input frame;
- leaves out rows outside `[train_start, test_end)`.

Rows with `anomaly_type == "missing_value"` are dropped by the anomaly notebook
before fitting the classifier because C1 repairs them with deterministic rules.
That filter is model-specific and happens after the split: the split itself
still covers all 525,600 rows. This is why the fitted preprocessor records
262,510 training rows instead of 263,040.

## Checks

`tests/unit/test_split.py` pins the boundaries, the row counts, the absence of
overlap, and the stability of the split when the input rows are shuffled:

```bash
uv run pytest tests/unit/test_split.py
```
