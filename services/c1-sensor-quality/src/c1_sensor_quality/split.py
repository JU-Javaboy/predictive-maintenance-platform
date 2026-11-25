"""Single chronological train/validation/test split for the C1 dataset.

The dataset must not be shuffled: the model is evaluated on a later period than
the one it was fitted on. ``chronological_split`` is the only implementation of
that rule; notebooks and training scripts call it instead of redefining dates.

The default boundaries reproduce the split used to fit the committed model and
the anomaly preprocessor:

============  ==========================  ==========================
partition     start (inclusive)           end (exclusive)
============  ==========================  ==========================
train         2012-01-14 00:00:00         2013-07-15 00:00:00
validation    2013-07-15 00:00:00         2014-01-14 00:00:00
test          2014-01-14 00:00:00         2015-01-13 00:00:00
============  ==========================  ==========================

The first two partitions cover the first two years of the dataset and may be
used for training, feature engineering, model selection and validation. The
final year is the test set: no row from it may influence any fitted value.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

#: First timestamp of the dataset (2012-01-14 00:00).
TRAIN_START = pd.Timestamp("2012-01-14 00:00:00")
#: End of the 18 training months. The whole 2013-07-14 day stays in training.
TRAIN_END = pd.Timestamp("2013-07-15 00:00:00")
#: End of the 6 validation months.
VALIDATION_END = pd.Timestamp("2014-01-14 00:00:00")
#: End of the final test year (the last timestamp is 2015-01-12 23:00).
TEST_END = pd.Timestamp("2015-01-13 00:00:00")


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """Start and end of each partition; every end is exclusive."""

    train_start: pd.Timestamp = TRAIN_START
    train_end: pd.Timestamp = TRAIN_END
    validation_end: pd.Timestamp = VALIDATION_END
    test_end: pd.Timestamp = TEST_END

    def __post_init__(self) -> None:
        for name in ("train_start", "train_end", "validation_end", "test_end"):
            if not isinstance(getattr(self, name), pd.Timestamp):
                raise TypeError(f"{name} must be a pandas.Timestamp")
        try:
            ordered = self.train_start < self.train_end < self.validation_end < self.test_end
        except TypeError as error:
            raise ValueError("split boundaries must share the same timezone") from error
        if not ordered:
            raise ValueError(
                "split boundaries must be strictly increasing: "
                "train_start < train_end < validation_end < test_end"
            )

    @property
    def validation_start(self) -> pd.Timestamp:
        """First timestamp of the validation partition."""
        return self.train_end

    @property
    def test_start(self) -> pd.Timestamp:
        """First timestamp of the test partition."""
        return self.validation_end


DEFAULT_BOUNDARIES = SplitBoundaries()


def chronological_split(
    frame: pd.DataFrame,
    boundaries: SplitBoundaries = DEFAULT_BOUNDARIES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into train, validation and test by ``timestamp``.

    Rows are assigned with half-open intervals ``[start, end)``: a row that sits
    exactly on a boundary belongs to the partition that starts there, never to
    the previous one. Rows outside ``[train_start, test_end)`` are not returned.

    ``frame`` is left untouched. The returned frames are sorted by ``timestamp``
    (then ``machine_id`` when the column exists) and have a fresh index.

    Raises:
        KeyError: ``frame`` has no ``timestamp`` column.
        TypeError: ``frame`` is not a pandas DataFrame.
        ValueError: timestamps are missing, invalid or timezone-aware.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if "timestamp" not in frame.columns:
        raise KeyError("frame must contain a 'timestamp' column")

    timestamps = frame["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(timestamps.dtype):
        timestamps = pd.to_datetime(timestamps, errors="coerce")
    if timestamps.dt.tz is not None:
        raise ValueError("timezone-aware timestamps are not supported; use naive timestamps")

    missing = int(timestamps.isna().sum())
    if missing:
        raise ValueError(
            f"timestamp contains {missing} missing or invalid value(s); "
            "drop or fix them before splitting"
        )

    ordered = frame.copy()
    ordered["timestamp"] = timestamps
    sort_columns = ["timestamp"]
    if "machine_id" in ordered.columns:
        sort_columns.append("machine_id")
    ordered = ordered.sort_values(sort_columns, kind="stable").reset_index(drop=True)

    train = ordered.loc[
        (ordered["timestamp"] >= boundaries.train_start)
        & (ordered["timestamp"] < boundaries.train_end)
    ].copy()
    validation = ordered.loc[
        (ordered["timestamp"] >= boundaries.validation_start)
        & (ordered["timestamp"] < boundaries.validation_end)
    ].copy()
    test = ordered.loc[
        (ordered["timestamp"] >= boundaries.test_start)
        & (ordered["timestamp"] < boundaries.test_end)
    ].copy()

    return train, validation, test
