"""Cleans the 3-year raw dataset with the same rules the C1 API will use.

1. Reads data/raw/predictive_maintenance_data.csv.
2. Checks every row against the SensorRecord schema.
3. Cleans the whole file as one batch with process_batch().
4. Saves the rows we keep to data/processed/cleaned_predictive_maintenance_data.csv,
   with two extra columns: quality_disposition and quality_flags.

If a row does not match the schema, or the batch is rejected, no file is saved.
"""

import csv
import sys
from pathlib import Path

from c1_sensor_quality.parser import read_csv_rows
from c1_sensor_quality.quality import process_batch
from c1_sensor_quality.schemas import (
    BatchRequest,
    ProcessedRecord,
    QualityReportSummary,
    RecordDisposition,
    SensorRecord,
)
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "predictive_maintenance_data.csv"
CLEANED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_predictive_maintenance_data.csv"
)

OUTPUT_COLUMNS = list(SensorRecord.model_fields) + ["quality_disposition", "quality_flags"]
ACCEPTED_DISPOSITIONS = {RecordDisposition.VALID, RecordDisposition.REPAIRED}


def load_batch(csv_path: Path) -> BatchRequest:
    """Reads the raw CSV as one batch. Stops with an error if any row does not match the schema."""
    records: list[SensorRecord] = []
    with open(csv_path, encoding="utf-8") as raw_file:
        for line_number, row in read_csv_rows(raw_file):
            try:
                records.append(SensorRecord(**row))
            except ValidationError as error:
                raise ValueError(f"Line {line_number} does not match the schema.\n{error}") from error
    return BatchRequest(batch_id=csv_path.stem, records=records)


def write_cleaned_csv(processed: list[ProcessedRecord], csv_path: Path) -> None:
    """Saves the valid and repaired rows with their quality columns. Other rows are left out."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as cleaned_file:
        writer = csv.DictWriter(cleaned_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for item in processed:
            if item.disposition not in ACCEPTED_DISPOSITIONS:
                continue
            row = item.record.model_dump()
            row["quality_disposition"] = item.disposition.value
            row["quality_flags"] = ";".join(item.flags) or "none"
            writer.writerow(row)


def print_summary(summary: QualityReportSummary) -> None:
    """Prints the summary from process_batch()."""
    print(f"Status:              {summary.status}")
    print(f"Rows received:       {summary.received:,}")
    print(f"Rows accepted:       {summary.accepted:,}")
    print(f"Rows repaired:       {summary.repaired:,}")
    print(f"Rows quarantined:    {summary.quarantined:,}")
    print(f"Duplicates excluded: {summary.duplicates_excluded:,}")
    for reason in summary.reasons[:10]:
        print(f"  - {reason}")
    if len(summary.reasons) > 10:
        print(f"  ... and {len(summary.reasons) - 10:,} more reasons")


def main() -> int:
    if not RAW_DATA_PATH.exists():
        print(f"Raw file not found: {RAW_DATA_PATH}")
        return 1

    try:
        batch = load_batch(RAW_DATA_PATH)
    except ValueError as error:
        print(f"Batch rejected. {error}")
        return 1

    processed, summary = process_batch(batch)
    print_summary(summary)

    if summary.status == "rejected":
        print("Batch rejected. No file saved.")
        return 1

    write_cleaned_csv(processed, CLEANED_DATA_PATH)
    print(f"Saved cleaned file to: {CLEANED_DATA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
