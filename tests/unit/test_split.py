"""Tests for the single chronological split.

The split is the anti-leakage boundary between training and evaluation, so
these tests pin the exact dates, the half-open intervals, and the fact that
changing the input row order cannot move a row to another partition.
"""

import pandas as pd
import pytest
from c1_sensor_quality.split import (
    DEFAULT_BOUNDARIES,
    TEST_END,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    SplitBoundaries,
    chronological_split,
)


def _three_year_frame(machines: tuple[str, ...] = ("M001",)) -> pd.DataFrame:
    """Every hourly timestamp of the dataset, for one or more machines."""
    timestamps = pd.date_range("2012-01-14 00:00", "2015-01-12 23:00", freq="h")
    rows = [
        {
            "timestamp": timestamp,
            "machine_id": machine,
            "row_id": f"{machine}-{timestamp}",
        }
        for timestamp in timestamps
        for machine in machines
    ]
    return pd.DataFrame(rows)


def test_default_boundaries_match_the_documented_dates() -> None:
    assert TRAIN_START == pd.Timestamp("2012-01-14 00:00")
    assert TRAIN_END == pd.Timestamp("2013-07-15 00:00")
    assert VALIDATION_END == pd.Timestamp("2014-01-14 00:00")
    assert TEST_END == pd.Timestamp("2015-01-13 00:00")
    assert DEFAULT_BOUNDARIES.validation_start == TRAIN_END
    assert DEFAULT_BOUNDARIES.test_start == VALIDATION_END


def test_default_split_covers_18_6_12_months_without_gap_or_overlap() -> None:
    frame = _three_year_frame()

    train, validation, test = chronological_split(frame)

    assert len(train) == 548 * 24
    assert len(validation) == 183 * 24
    assert len(test) == 364 * 24
    assert len(train) + len(validation) + len(test) == len(frame)

    assert train["timestamp"].min() == TRAIN_START
    assert train["timestamp"].max() == pd.Timestamp("2013-07-14 23:00")
    assert validation["timestamp"].min() == pd.Timestamp("2013-07-15 00:00")
    assert validation["timestamp"].max() == pd.Timestamp("2014-01-13 23:00")
    assert test["timestamp"].min() == pd.Timestamp("2014-01-14 00:00")
    assert test["timestamp"].max() == pd.Timestamp("2015-01-12 23:00")


def test_boundaries_are_half_open() -> None:
    timestamps = pd.to_datetime(
        [
            "2012-01-13 23:00",  # before the dataset
            "2012-01-14 00:00",  # first train row
            "2013-07-14 23:00",  # last train row
            "2013-07-15 00:00",  # first validation row
            "2014-01-13 23:00",  # last validation row
            "2014-01-14 00:00",  # first test row
            "2015-01-12 23:00",  # last test row
            "2015-01-13 00:00",  # after the dataset
        ]
    )
    frame = pd.DataFrame({"timestamp": timestamps, "row_id": range(len(timestamps))})

    train, validation, test = chronological_split(frame)

    assert train["row_id"].tolist() == [1, 2]
    assert validation["row_id"].tolist() == [3, 4]
    assert test["row_id"].tolist() == [5, 6]


def test_partitions_are_disjoint_and_cover_the_input() -> None:
    frame = _three_year_frame(("M001", "M002"))

    train, validation, test = chronological_split(frame)

    train_ids = set(train["row_id"])
    validation_ids = set(validation["row_id"])
    test_ids = set(test["row_id"])
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)
    assert train_ids | validation_ids | test_ids == set(frame["row_id"])


def test_split_is_stable_when_rows_are_shuffled() -> None:
    frame = _three_year_frame()
    shuffled = frame.sample(frac=1.0, random_state=42)

    expected = chronological_split(frame)
    actual = chronological_split(shuffled)

    for expected_part, actual_part in zip(expected, actual, strict=True):
        # Timestamp + machine is unique here, so even the row order is stable.
        pd.testing.assert_frame_equal(actual_part, expected_part)
        assert actual_part["timestamp"].is_monotonic_increasing


def test_input_frame_is_not_modified() -> None:
    frame = _three_year_frame().sample(frac=0.1, random_state=0)
    original = frame.copy(deep=True)

    chronological_split(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_missing_timestamps_are_rejected() -> None:
    frame = pd.DataFrame({"timestamp": [pd.Timestamp("2013-01-01"), pd.NaT]})

    with pytest.raises(ValueError, match="missing or invalid"):
        chronological_split(frame)


def test_unparseable_timestamps_are_rejected() -> None:
    frame = pd.DataFrame({"timestamp": ["2013-01-01 00:00", "not-a-date"]})

    with pytest.raises(ValueError, match="missing or invalid"):
        chronological_split(frame)


def test_timezone_aware_timestamps_are_rejected() -> None:
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2013-01-01"]).tz_localize("UTC")})

    with pytest.raises(ValueError, match="timezone-aware"):
        chronological_split(frame)


def test_missing_timestamp_column_is_rejected() -> None:
    with pytest.raises(KeyError, match="timestamp"):
        chronological_split(pd.DataFrame({"machine_id": ["M001"]}))


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "object"])
def test_empty_frame_returns_three_empty_partitions(dtype: str) -> None:
    frame = pd.DataFrame({"timestamp": pd.Series([], dtype=dtype)})

    parts = chronological_split(frame)

    assert all(part.empty for part in parts)
    assert all(list(part.columns) == ["timestamp"] for part in parts)


def test_boundaries_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        SplitBoundaries(
            train_start=pd.Timestamp("2014-01-01"),
            train_end=pd.Timestamp("2013-01-01"),
            validation_end=pd.Timestamp("2015-01-01"),
            test_end=pd.Timestamp("2016-01-01"),
        )


def test_custom_boundaries_are_applied() -> None:
    frame = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"])}
    )
    boundaries = SplitBoundaries(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-02-01"),
        validation_end=pd.Timestamp("2020-03-01"),
        test_end=pd.Timestamp("2020-04-01"),
    )

    train, validation, test = chronological_split(frame, boundaries)

    assert train["timestamp"].tolist() == [pd.Timestamp("2020-01-15")]
    assert validation["timestamp"].tolist() == [pd.Timestamp("2020-02-15")]
    assert test["timestamp"].tolist() == [pd.Timestamp("2020-03-15")]
