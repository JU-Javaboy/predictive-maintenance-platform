# Dataset split

Use a chronological split by `timestamp`:

- first two years: training, feature engineering, model selection, and
  validation;
- final year: test data only.

The test period must not be used for training, hyperparameter tuning, or model
selection.
