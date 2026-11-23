"""Cleans and checks sensor data for C1.

These are the cleaning rules we agreed on. Fixing quoted CSV lines is done in parser.py.
1. Timestamps must be real dates in the right format, and not in the future.
2. Machine IDs must look like M001. We remove extra spaces.
3. Operating modes are made uppercase. Unknown modes get a flag.
4. Missing data follows our 3 steps:
   - Less than 5% missing: we fill in the missing values.
   - 5% to 10% missing: we drop only the rows with missing values.
   - More than 10% missing: we reject the whole batch.
     Batches with fewer than 100 rows are never rejected. They only drop the bad rows.
5. Duplicates: we skip exact copies. If two copies do not match, we set both aside.
6. Physical limits: impossible values are set aside. Possible spikes are kept with a flag.
7. Labels: an empty anomaly_type becomes "normal". The failure probability stays between 0 and 1.

All the numbers we may want to change are in imputation_params.json.
"""

import json
import re
from datetime import datetime
from pathlib import Path

from c1_sensor_quality.fallback import apply_conservative_fallback, evaluate_model_behavior
from c1_sensor_quality.monitoring import monitoring
from c1_sensor_quality.schemas import (
    BatchRequest,
    ProcessedRecord,
    QualityReportSummary,
    RecordDisposition,
    SensorRecord,
)

# ==============================================================================
# Load the settings from imputation_params.json
# ==============================================================================
CONFIG_FILE_PATH = Path(__file__).parent / "imputation_params.json"

with open(CONFIG_FILE_PATH, encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)

# Share of rows with a missing sensor. Below 0.05 we fill them in. Above 0.10 we reject the batch. 0.05 and 0.10 in this case means 5% and 10% respectively.
IMPUTE_THRESHOLD_PCT = CONFIG["thresholds"]["impute_missing_pct"]
DROP_ROWS_THRESHOLD_PCT = CONFIG["thresholds"]["drop_rows_missing_pct"]

# In a small batch, one or two missing rows already go over 10%.
# So we never reject a batch that is smaller than this.
MIN_ROWS_FOR_BATCH_REJECT = CONFIG["thresholds"]["min_rows_for_batch_reject"]

# Values outside these [min, max] limits are physically impossible
FATAL_BOUNDS = CONFIG["fatal_bounds"]

# Values past these limits may be sensor glitches. We keep them but add a flag.
TEMPERATURE_SPIKE_ABOVE = CONFIG["spike_flags"]["temperature_above"]
VIBRATION_SPIKE_BELOW = CONFIG["spike_flags"]["vibration_below"]

# Median sensor values for each operating mode, from the first two years of data
MODE_SENSOR_MEDIANS = CONFIG["mode_sensor_medians"]

VALID_OPERATING_MODES = {"NORMAL", "HIGH_LOAD", "IDLE", "OVERLOAD"}
SENSOR_NAMES = ["temperature", "vibration", "pressure", "rpm", "power"]

# Machine IDs look like M001: the letter M and then exactly 3 digits
MACHINE_ID_PATTERN = re.compile(r"M\d{3}")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def quarantine(record: SensorRecord, flags: list[str], reason: str) -> ProcessedRecord:
    """Marks a row as quarantined and keeps the flags and the reason why."""
    return ProcessedRecord(
        record=record,
        disposition=RecordDisposition.QUARANTINED,
        flags=flags,
        quarantine_reason=reason,
    )


def validate_single_record(record: SensorRecord) -> ProcessedRecord:
    """Checks and cleans one row by itself. It does not look at other rows.

    The row comes back as quarantined if we cannot use it, needs_repair if a sensor
    is missing, or valid if all is fine. process_batch() fills in missing values later.
    """
    flags: list[str] = []

    # 1. Machine ID must be there and look like M001
    clean_machine_id = (record.machine_id or "").strip().upper()
    if clean_machine_id == "":
        return quarantine(record, ["missing_machine_id"], "Machine ID is missing or empty.")
    if not MACHINE_ID_PATTERN.fullmatch(clean_machine_id):
        return quarantine(
            record,
            ["invalid_machine_id"],
            f"Machine ID '{record.machine_id}' is not in the format M001.",
        )

    # 2. Timestamp must be a real date and time, and not later than now
    try:
        # strptime only accepts real dates (month 13 or 30 February will fail)
        parsed_timestamp = datetime.strptime((record.timestamp or "").strip(), TIMESTAMP_FORMAT)
    except ValueError:
        return quarantine(
            record,
            ["invalid_timestamp"],
            f"Timestamp '{record.timestamp}' is not in YYYY-MM-DD HH:MM:SS format.",
        )
    if parsed_timestamp > datetime.now():
        return quarantine(
            record, ["future_timestamp"], f"Timestamp '{record.timestamp}' is in the future."
        )

    # 3. Make the operating mode uppercase. An unknown mode gets a flag and becomes NORMAL.
    clean_mode = (record.operating_mode or "").strip().upper()
    if clean_mode not in VALID_OPERATING_MODES:
        flags.append("unknown_operating_mode")
        clean_mode = "NORMAL"

    # 4. Missing sensors. If all 5 are empty, the row has nothing we can use.
    sensor_values = {name: getattr(record, name) for name in SENSOR_NAMES}
    missing_sensors = [name for name, value in sensor_values.items() if value is None]

    if len(missing_sensors) == len(SENSOR_NAMES):
        return quarantine(record, ["all_sensors_missing"], "All 5 sensor values are empty.")

    for sensor in missing_sensors:
        flags.append(f"missing_{sensor}")

    # 5. Physical limits. An impossible value means we cannot use the row.
    bounds_problems = []
    for sensor, (min_limit, max_limit) in FATAL_BOUNDS.items():
        value = sensor_values[sensor]
        if value is not None and (value < min_limit or value > max_limit):
            bounds_problems.append(
                f"{sensor} value {value} is physically impossible "
                f"(allowed range: {min_limit} to {max_limit})."
            )
    if len(bounds_problems) > 0:
        return quarantine(record, flags + ["hard_bounds_violated"], "; ".join(bounds_problems))

    # 6. Possible sensor glitches. We keep the row and add a flag. We do NOT drop it.
    vibration_spike = record.vibration is not None and record.vibration < VIBRATION_SPIKE_BELOW
    temperature_spike = (
        record.temperature is not None and record.temperature > TEMPERATURE_SPIKE_ABOVE
    )
    if vibration_spike or temperature_spike:
        flags.append("suspect_sensor_spike")

    # 7. Labels. A healthy row gets "normal" instead of an empty anomaly_type.
    clean_anomaly_type = record.anomaly_type
    if clean_anomaly_type is None or clean_anomaly_type.strip() == "":
        clean_anomaly_type = "normal"

    # A probability must be between 0 and 1. The data has some small values below 0.
    clean_probability = record.failure_probability_gt
    if clean_probability is not None and (clean_probability < 0.0 or clean_probability > 1.0):
        clean_probability = min(max(clean_probability, 0.0), 1.0)
        flags.append("failure_probability_clipped")

    if len(missing_sensors) > 0:
        disposition = RecordDisposition.NEEDS_REPAIR
    else:
        disposition = RecordDisposition.VALID

    cleaned_record = record.model_copy(
        update={
            "timestamp": parsed_timestamp.strftime(TIMESTAMP_FORMAT),
            "machine_id": clean_machine_id,
            "operating_mode": clean_mode,
            "anomaly_type": clean_anomaly_type,
            "failure_probability_gt": clean_probability,
        }
    )
    behavior = evaluate_model_behavior(cleaned_record)
    fallback_record = apply_conservative_fallback(cleaned_record, behavior)
    return ProcessedRecord(
        record=fallback_record,
        disposition=disposition,
        flags=flags,
        model_status=behavior.status,
        fallback=behavior.fallback,
        model_warnings=behavior.warnings,
    )


def process_batch(batch: BatchRequest) -> tuple[list[ProcessedRecord], QualityReportSummary]:
    """Cleans and checks a whole batch of rows.

    1. Works out how many rows have a missing sensor.
    2. Rejects the whole batch if more than 10% is missing (Tier 3).
       Batches with fewer than 100 rows skip this step.
    3. Checks each row, then sorts the rows by machine ID and time.
    4. Finds duplicates. Exact copies are skipped. Copies that do not match are set aside.
    5. Fills in missing values if less than 5% is missing (Tier 1).
       Otherwise it drops the rows with missing values (Tier 2).
    6. Gives back every row with its status, and a summary.
    """
    total_received = len(batch.records)
    if total_received == 0:
        empty_summary = QualityReportSummary(
            batch_id=batch.batch_id, received=0, accepted=0, quarantined=0, duplicates_excluded=0
        )
        monitoring.record_batch(empty_summary)
        return [], empty_summary

    # Step 1: Share of rows with at least one missing sensor value
    rows_with_missing = sum(
        1
        for record in batch.records
        if any(getattr(record, sensor) is None for sensor in SENSOR_NAMES)
    )
    missing_ratio = rows_with_missing / total_received

    # Step 2: Tier 3. Too much missing data, so we reject the whole batch.
    # Small batches skip this. They go to Tier 2 and only drop the rows with missing values.
    if missing_ratio > DROP_ROWS_THRESHOLD_PCT and total_received >= MIN_ROWS_FOR_BATCH_REJECT:
        reason = (
            f"Too much missing data ({missing_ratio:.1%}). "
            f"Our limit is {DROP_ROWS_THRESHOLD_PCT:.0%}."
        )
        rejected_rows = [
            quarantine(record, ["missing_threshold_exceeded"], reason) for record in batch.records
        ]
        rejected_summary = QualityReportSummary(
            batch_id=batch.batch_id,
            received=total_received,
            accepted=0,
            quarantined=total_received,
            duplicates_excluded=0,
            status="rejected",
            reasons=[reason],
        )
        monitoring.record_batch(rejected_summary)
        return rejected_rows, rejected_summary

    # Step 3: Check each row, then sort by the cleaned machine ID and time.
    # We sort after cleaning, so " m001" and "M001" end up next to each other.
    validated_rows = [validate_single_record(record) for record in batch.records]

    def sort_key(item: ProcessedRecord) -> tuple[str, str]:
        return (item.record.machine_id or "", item.record.timestamp or "")

    validated_rows.sort(key=sort_key)

    # Step 4: Find duplicates (same machine at the same time)
    first_seen: dict[tuple[str, str], ProcessedRecord] = {}
    processed_rows: list[ProcessedRecord] = []
    duplicates_excluded = 0
    reasons: list[str] = []

    for item in validated_rows:
        key = (item.record.machine_id, item.record.timestamp)

        if key not in first_seen:
            first_seen[key] = item
            processed_rows.append(item)
            if item.quarantine_reason:
                reasons.append(item.quarantine_reason)
            continue

        first_item = first_seen[key]

        # Case A: Exact copy (every column is the same). We skip the copy.
        if first_item.record == item.record:
            duplicates_excluded += 1
            processed_rows.append(
                ProcessedRecord(
                    record=item.record,
                    disposition=RecordDisposition.DUPLICATE_EXCLUDED,
                    flags=["duplicate_exact"],
                    quarantine_reason="Exact copy of an earlier row in this batch. Skipped.",
                )
            )
            continue

        # Case B: The copies do not match (some column is different). We set BOTH aside.
        conflict_reason = (
            f"Two different readings for {item.record.machine_id} at {item.record.timestamp}. "
            "We cannot tell which one is right."
        )
        if first_item.disposition != RecordDisposition.QUARANTINED:
            first_item.disposition = RecordDisposition.QUARANTINED
            first_item.flags.append("duplicate_conflict")
            first_item.quarantine_reason = conflict_reason
            reasons.append(conflict_reason)

        item.disposition = RecordDisposition.QUARANTINED
        item.flags.append("duplicate_conflict")
        item.quarantine_reason = conflict_reason
        processed_rows.append(item)
        reasons.append(conflict_reason)

    # Step 5: Rows with a missing sensor are filled in (Tier 1) or dropped (Tier 2)
    repaired = 0
    for item in processed_rows:
        if item.disposition != RecordDisposition.NEEDS_REPAIR:
            continue

        if missing_ratio < IMPUTE_THRESHOLD_PCT:
            # Tier 1: fill each missing sensor with the median for this operating mode
            mode_medians = MODE_SENSOR_MEDIANS[item.record.operating_mode]
            filled_values = {}
            for sensor in SENSOR_NAMES:
                if getattr(item.record, sensor) is None:
                    filled_values[sensor] = mode_medians[sensor]
                    item.flags.append(f"missing_{sensor}_imputed")

            item.record = item.record.model_copy(update=filled_values)
            item.disposition = RecordDisposition.REPAIRED
            repaired += 1
        else:
            # Tier 2: drop only the rows with missing values
            item.disposition = RecordDisposition.QUARANTINED
            item.flags.append("missing_values_dropped_tier2")
            item.quarantine_reason = (
                "Row dropped. It has a missing value and 5% or more of the batch is missing."
            )
            reasons.append(item.quarantine_reason)

    # Step 6: Summary
    accepted = sum(
        1
        for item in processed_rows
        if item.disposition in (RecordDisposition.VALID, RecordDisposition.REPAIRED)
    )
    quarantined = sum(
        1 for item in processed_rows if item.disposition == RecordDisposition.QUARANTINED
    )

    if quarantined == total_received:
        status = "rejected"
    elif quarantined > 0:
        status = "partially_processed"
    else:
        status = "processed"

    summary = QualityReportSummary(
        batch_id=batch.batch_id,
        received=total_received,
        accepted=accepted,
        repaired=repaired,
        quarantined=quarantined,
        duplicates_excluded=duplicates_excluded,
        status=status,
        reasons=reasons,
    )
    monitoring.record_batch(summary)
    return processed_rows, summary
