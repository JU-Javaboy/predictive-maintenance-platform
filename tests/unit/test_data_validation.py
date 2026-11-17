"""Tests for the C1 data cleaning rules.

There is one section for each rule we agreed on:
1. Quote unwrapping (outer quotes removed, inner quotes fixed, empty cells kept)
2. Timestamps (real date, right format, not in the future, sorted, spaces removed)
3. Machine IDs (spaces removed, uppercase, M001 format, missing ID set aside)
4. Operating modes (uppercase, unknown modes flagged)
5. Missing sensors in 3 steps (fill under 5%, drop rows at 5-10%, reject over 10%)
6. Duplicates (exact copies skipped, copies that do not match set aside)
7. Physical limits (impossible values set aside, spikes kept with a flag)
8. Labels (empty anomaly_type becomes "normal", probability stays between 0 and 1)
9. Reading raw CSV rows (empty cells become None, bad lines and bad types rejected)
"""

import io
from datetime import datetime, timedelta

import pytest
from c1_sensor_quality.parser import read_csv_rows, unwrap_csv_line
from c1_sensor_quality.quality import process_batch, validate_single_record
from c1_sensor_quality.schemas import BatchRequest, RecordDisposition, SensorRecord
from pydantic import ValidationError

CSV_HEADER = (
    "timestamp,machine_id,operating_mode,temperature,vibration,pressure,rpm,power,"
    "anomaly,anomaly_type,failure_probability_gt,failure_in_7_days,"
    "remaining_useful_life_hours,failure_type,important_features,maintenance_action,"
    "priority,concept_drift_active\n"
)

# The part of a healthy CSV row that comes after the timestamp
HEALTHY_CSV_VALUES = (
    ",M004,NORMAL,48.4,2.9,97.4,1515.4,211.9,"
    "False,,0.37,False,625,cooling_failure,temperature,repair_cooling_system,low,False\n"
)

# A wrapped line, just like the ones in the raw 3-year file
WRAPPED_CSV_LINE = (
    '"2012-01-14 00:00:00,M001,HIGH_LOAD,46.0,1.6,98.8,2188.4,316.1,'
    'False,,0.03,False,963,motor_imbalance,""vibration,rpm"",align_motor,low,False"\n'
)


# ==============================================================================
# Helpers
# ==============================================================================
def make_record(**overrides) -> SensorRecord:
    """Makes a valid row with all 18 columns. Pass any column to change its value."""
    data = {
        "timestamp": "2012-01-14 00:00:00",
        "machine_id": "M001",
        "operating_mode": "NORMAL",
        "temperature": 57.0,
        "vibration": 3.5,
        "pressure": 97.0,
        "rpm": 1650.0,
        "power": 230.0,
        "anomaly": False,
        "anomaly_type": None,
        "failure_probability_gt": 0.05,
        "failure_in_7_days": False,
        "remaining_useful_life_hours": 750.0,
        "failure_type": "cooling_failure",
        "important_features": "temperature",
        "maintenance_action": "repair_cooling_system",
        "priority": "low",
        "concept_drift_active": False,
    }
    data.update(overrides)
    return SensorRecord(**data)


def hourly_timestamp(hour: int) -> str:
    """Gives the timestamp that is a number of hours after 2012-01-14 00:00:00."""
    return (datetime(2012, 1, 14) + timedelta(hours=hour)).strftime("%Y-%m-%d %H:%M:%S")


def batch_with_missing_temperature(total_rows: int, missing_rows: int) -> BatchRequest:
    """Makes a batch of hourly rows. The last `missing_rows` rows have no temperature."""
    records = []
    for hour in range(total_rows):
        if hour >= total_rows - missing_rows:
            records.append(make_record(timestamp=hourly_timestamp(hour), temperature=None))
        else:
            records.append(make_record(timestamp=hourly_timestamp(hour)))
    return BatchRequest(batch_id="test-batch", records=records)


# ==============================================================================
# Rule 1: Quote Unwrapping
# ==============================================================================
def test_wrapped_line_is_unwrapped_and_empty_cells_are_kept():
    """Outer quotes are removed, inner quotes are fixed, and ,, is kept."""
    cleaned = unwrap_csv_line(WRAPPED_CSV_LINE)

    assert not cleaned.startswith('"')
    assert '"vibration,rpm"' in cleaned
    assert "False,,0.03" in cleaned


def test_normal_line_is_left_untouched():
    """A line without outer quotes stays exactly the same."""
    normal_line = "2012-01-14 00:00:00" + HEALTHY_CSV_VALUES
    assert unwrap_csv_line(normal_line) == normal_line


# ==============================================================================
# Rule 2: Timestamps
# ==============================================================================
def test_invalid_timestamp_quarantines_record():
    """Text, month 13 and 30 February are all set aside as invalid timestamps."""
    for bad_timestamp in ["not-a-date", "2012-13-14 12:00:00", "2012-02-30 12:00:00"]:
        result = validate_single_record(make_record(timestamp=bad_timestamp))
        assert result.disposition == RecordDisposition.QUARANTINED
        assert "invalid_timestamp" in result.flags


def test_timestamp_after_now_quarantines_record():
    """A timestamp later than right now is set aside."""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    result = validate_single_record(make_record(timestamp=tomorrow))
    assert result.disposition == RecordDisposition.QUARANTINED
    assert "future_timestamp" in result.flags


def test_timestamp_whitespace_is_trimmed_so_duplicates_are_caught():
    """A timestamp with extra spaces is saved without them, so the duplicate is found."""
    rec1 = make_record(timestamp="2012-01-14 00:00:00")
    rec2 = make_record(timestamp=" 2012-01-14 00:00:00 ")

    processed, summary = process_batch(BatchRequest(batch_id="b", records=[rec1, rec2]))

    assert summary.duplicates_excluded == 1
    assert processed[1].record.timestamp == "2012-01-14 00:00:00"


def test_batch_rows_are_sorted_by_machine_and_time():
    """Rows that come in the wrong order are sorted by machine and time."""
    rec_later = make_record(timestamp="2012-01-14 03:00:00")
    rec_earlier = make_record(timestamp="2012-01-14 00:00:00")

    processed, _ = process_batch(BatchRequest(batch_id="b", records=[rec_later, rec_earlier]))

    assert processed[0].record.timestamp == "2012-01-14 00:00:00"
    assert processed[1].record.timestamp == "2012-01-14 03:00:00"


# ==============================================================================
# Rule 3: Machine IDs
# ==============================================================================
def test_machine_id_is_trimmed_and_uppercased():
    """'  m021  ' becomes 'M021'. A new machine is fine if the format is right."""
    result = validate_single_record(make_record(machine_id="  m021  "))
    assert result.disposition == RecordDisposition.VALID
    assert result.record.machine_id == "M021"


def test_missing_machine_id_quarantines_record():
    """An empty or missing machine_id is set aside."""
    for missing_id in ["", None]:
        result = validate_single_record(make_record(machine_id=missing_id))
        assert result.disposition == RecordDisposition.QUARANTINED
        assert "missing_machine_id" in result.flags


def test_machine_id_with_wrong_format_quarantines_record():
    """A machine_id that does not look like M001 is set aside."""
    for bad_id in ["hello", "M1", "M0001", "001"]:
        result = validate_single_record(make_record(machine_id=bad_id))
        assert result.disposition == RecordDisposition.QUARANTINED
        assert "invalid_machine_id" in result.flags


# ==============================================================================
# Rule 4: Operating Modes
# ==============================================================================
def test_operating_mode_is_trimmed_and_uppercased():
    """'normal ' becomes 'NORMAL'."""
    result = validate_single_record(make_record(operating_mode="normal "))
    assert result.disposition == RecordDisposition.VALID
    assert result.record.operating_mode == "NORMAL"


def test_unknown_operating_mode_is_flagged_and_set_to_normal():
    """An unknown mode gets a flag and becomes NORMAL."""
    result = validate_single_record(make_record(operating_mode="TURBO"))
    assert "unknown_operating_mode" in result.flags
    assert result.record.operating_mode == "NORMAL"


# ==============================================================================
# Rule 5: Missing Sensor Values (3-Tier Policy)
# ==============================================================================
def test_tier1_fills_any_missing_sensor_with_mode_median():
    """Tier 1 (under 5% missing): every missing sensor is filled in and the row is marked repaired."""
    records = [make_record(timestamp=hourly_timestamp(hour)) for hour in range(97)]
    records.append(make_record(timestamp=hourly_timestamp(97), temperature=None))
    records.append(make_record(timestamp=hourly_timestamp(98), vibration=None))
    records.append(make_record(timestamp=hourly_timestamp(99), pressure=None))

    processed, summary = process_batch(BatchRequest(batch_id="b", records=records))

    assert summary.accepted == 100
    assert summary.repaired == 3
    assert processed[97].record.temperature == 52.56  # NORMAL median from the config
    assert "missing_temperature_imputed" in processed[97].flags
    assert "missing_vibration_imputed" in processed[98].flags
    assert "missing_pressure_imputed" in processed[99].flags


def test_tier2_drops_only_rows_with_missing_values():
    """Tier 2 (5% to 10% missing): only the rows with missing values are dropped."""
    _processed, summary = process_batch(batch_with_missing_temperature(100, 8))

    assert summary.accepted == 92
    assert summary.repaired == 0
    assert summary.quarantined == 8
    assert summary.status == "partially_processed"


def test_tier3_rejects_whole_batch():
    """Tier 3 (over 10% missing): the whole batch is rejected."""
    _processed, summary = process_batch(batch_with_missing_temperature(100, 15))

    assert summary.accepted == 0
    assert summary.quarantined == 100
    assert summary.status == "rejected"


def test_small_batch_is_never_rejected_whole():
    """A batch under 100 rows is never rejected. It only drops the missing rows, even above 10%."""
    _processed, summary = process_batch(batch_with_missing_temperature(20, 5))  # 25% missing

    assert summary.accepted == 15
    assert summary.quarantined == 5
    assert summary.status == "partially_processed"


def test_row_with_all_5_sensors_missing_is_quarantined():
    """A row with no sensor values at all is set aside."""
    rec = make_record(temperature=None, vibration=None, pressure=None, rpm=None, power=None)
    result = validate_single_record(rec)
    assert result.disposition == RecordDisposition.QUARANTINED
    assert "all_sensors_missing" in result.flags


# ==============================================================================
# Rule 6: Duplicates
# ==============================================================================
def test_exact_duplicate_is_excluded():
    """An exact second copy is skipped and the first one is kept."""
    processed, summary = process_batch(
        BatchRequest(batch_id="b", records=[make_record(), make_record()])
    )

    assert summary.accepted == 1
    assert summary.duplicates_excluded == 1
    assert processed[0].disposition == RecordDisposition.VALID
    assert processed[1].disposition == RecordDisposition.DUPLICATE_EXCLUDED


def test_conflicting_duplicate_quarantines_both():
    """Two rows for the same machine and time with different values are both set aside."""
    rec1 = make_record(temperature=45.0)
    rec2 = make_record(temperature=105.0)

    processed, summary = process_batch(BatchRequest(batch_id="b", records=[rec1, rec2]))

    assert summary.accepted == 0
    assert summary.quarantined == 2
    assert "duplicate_conflict" in processed[0].flags
    assert "duplicate_conflict" in processed[1].flags


def test_duplicate_with_different_labels_is_a_conflict():
    """Duplicates are compared on every column, not just the sensors."""
    rec1 = make_record(anomaly=False)
    rec2 = make_record(anomaly=True, anomaly_type="sensor_spike")

    _processed, summary = process_batch(BatchRequest(batch_id="b", records=[rec1, rec2]))

    assert summary.duplicates_excluded == 0
    assert summary.quarantined == 2


# ==============================================================================
# Rule 7: Physical Limits
# ==============================================================================
def test_physically_impossible_value_quarantines_record():
    """Negative pressure, rpm and power, and a 15,000 °C temperature, are set aside."""
    impossible_values = [
        {"pressure": -10.0},
        {"rpm": -250.0},
        {"power": -50.0},
        {"temperature": 15000.0},
    ]
    for bad_value in impossible_values:
        result = validate_single_record(make_record(**bad_value))
        assert result.disposition == RecordDisposition.QUARANTINED
        assert "hard_bounds_violated" in result.flags


def test_sensor_spike_is_kept_and_flagged():
    """Spikes from the dataset (1091.8 °C, vibration -0.12) are kept with a flag."""
    for spike in [{"temperature": 1091.8}, {"vibration": -0.12}]:
        result = validate_single_record(make_record(**spike))
        assert result.disposition == RecordDisposition.VALID
        assert "suspect_sensor_spike" in result.flags


def test_spike_flag_is_added_only_once_when_two_sensors_spike():
    """A temperature spike and a vibration spike in one row give only one flag."""
    result = validate_single_record(make_record(temperature=500.0, vibration=-0.1))
    assert result.flags.count("suspect_sensor_spike") == 1


# ==============================================================================
# Rule 8: Label Columns
# ==============================================================================
def test_empty_anomaly_type_becomes_normal_and_real_types_are_kept():
    """None or '' becomes 'normal'. A real anomaly type stays the same."""
    for empty_value in [None, ""]:
        result = validate_single_record(make_record(anomaly_type=empty_value))
        assert result.record.anomaly_type == "normal"

    result = validate_single_record(make_record(anomaly=True, anomaly_type="missing_value"))
    assert result.record.anomaly_type == "missing_value"


def test_failure_probability_is_clipped_to_0_and_1():
    """Values below 0 or above 1 are clipped and flagged. Normal values stay the same."""
    low = validate_single_record(make_record(failure_probability_gt=-0.055))
    assert low.record.failure_probability_gt == 0.0
    assert "failure_probability_clipped" in low.flags

    high = validate_single_record(make_record(failure_probability_gt=1.03))
    assert high.record.failure_probability_gt == 1.0
    assert "failure_probability_clipped" in high.flags

    normal = validate_single_record(make_record(failure_probability_gt=0.37))
    assert normal.record.failure_probability_gt == 0.37
    assert "failure_probability_clipped" not in normal.flags


# ==============================================================================
# Rule 9: Reading Raw CSV Rows
# ==============================================================================
def test_read_csv_rows_unwraps_lines_and_turns_empty_cells_into_none():
    """A wrapped line is read correctly and empty cells (,,) become None."""
    line_number, row = next(read_csv_rows(io.StringIO(CSV_HEADER + WRAPPED_CSV_LINE)))

    assert line_number == 2
    assert row["anomaly_type"] is None
    assert row["important_features"] == "vibration,rpm"


def test_read_csv_rows_treats_nan_and_null_text_as_empty():
    """'nan', 'NaN', 'NULL', 'n/a' and 'None' in a cell become None."""
    raw_csv = CSV_HEADER + (
        "2012-01-14 00:00:00,M004,NORMAL,nan,NaN,NULL,n/a,None,"
        "False,,0.37,False,625,cooling_failure,temperature,repair_cooling_system,low,False\n"
    )
    _line_number, row = next(read_csv_rows(io.StringIO(raw_csv)))
    for sensor in ["temperature", "vibration", "pressure", "rpm", "power"]:
        assert row[sensor] is None


def test_read_csv_rows_handles_excel_byte_order_mark():
    """The hidden character Excel adds at the start of a file is ignored."""
    raw_csv = "\ufeff" + CSV_HEADER + "2012-01-14 00:00:00" + HEALTHY_CSV_VALUES
    _line_number, row = next(read_csv_rows(io.StringIO(raw_csv)))
    assert row["timestamp"] == "2012-01-14 00:00:00"


def test_read_csv_rows_handles_line_where_every_value_is_quoted():
    """A line where each value has its own quotes is read normally."""
    raw_csv = CSV_HEADER + (
        '"2012-01-14 00:00:00","M004","NORMAL","48.4","2.9","97.4","1515.4","211.9",'
        '"False","","0.37","False","625","cooling_failure","temperature",'
        '"repair_cooling_system","low","False"\n'
    )
    _line_number, row = next(read_csv_rows(io.StringIO(raw_csv)))
    assert row["timestamp"] == "2012-01-14 00:00:00"
    assert row["anomaly_type"] is None
    assert row["concept_drift_active"] == "False"


def test_read_csv_rows_rejects_line_with_wrong_column_count():
    """A line with too few or too many values stops with a clear error."""
    raw_csv = CSV_HEADER + "2012-01-14 00:00:00,M001,NORMAL,48.4\n"
    with pytest.raises(ValueError, match="Line 2"):
        list(read_csv_rows(io.StringIO(raw_csv)))


def test_bad_number_is_rejected_by_schema():
    """'abc', 'inf' and NaN in a sensor column fail the schema, so the batch is rejected."""
    for bad_value in ["abc", "inf", float("nan")]:
        with pytest.raises(ValidationError):
            make_record(temperature=bad_value)


def test_raw_csv_rows_flow_through_schema_and_cleaning():
    """The full path: raw CSV text, then the schema, then the batch cleaning."""
    missing_temperature_values = (
        ",M004,NORMAL,,2.9,97.4,1515.4,211.9,"
        "True,missing_value,-0.05,False,625,cooling_failure,temperature,"
        "repair_cooling_system,low,False\n"
    )

    # 30 healthy hours, then 1 hour with a missing temperature (about 3% missing)
    raw_csv = CSV_HEADER
    for hour in range(30):
        raw_csv += hourly_timestamp(hour) + HEALTHY_CSV_VALUES
    raw_csv += hourly_timestamp(30) + missing_temperature_values

    records = [SensorRecord(**row) for _line_number, row in read_csv_rows(io.StringIO(raw_csv))]
    processed, summary = process_batch(BatchRequest(batch_id="csv-flow", records=records))

    assert summary.received == 31
    assert summary.repaired == 1
    assert summary.quarantined == 0

    healthy = processed[0]
    assert healthy.disposition == RecordDisposition.VALID
    assert healthy.record.anomaly_type == "normal"

    repaired = processed[-1]  # The missing temperature row is the latest hour
    assert repaired.disposition == RecordDisposition.REPAIRED
    assert repaired.record.temperature == 52.56
    assert repaired.record.anomaly_type == "missing_value"
    assert repaired.record.failure_probability_gt == 0.0
    assert "missing_temperature_imputed" in repaired.flags
