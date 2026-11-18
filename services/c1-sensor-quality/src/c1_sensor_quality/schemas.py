"""Data shapes for C1.

Describes what one sensor row looks like, what a batch looks like,
and what the summary looks like after we clean a batch.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecordDisposition(str, Enum):
    """The status C1 gives each row after checking it.

    - valid: the row is clean and ready to use.
    - needs_repair: a sensor value is missing and we have not filled it in yet.
    - repaired: we filled in the missing value.
    - quarantined: the row is bad (for example a broken date or a negative speed), so we set it aside.
    - duplicate_excluded: the row is an exact copy of another row, so we skip it.
    """

    VALID = "valid"
    NEEDS_REPAIR = "needs_repair"
    REPAIRED = "repaired"
    QUARANTINED = "quarantined"
    DUPLICATE_EXCLUDED = "duplicate_excluded"


class SensorRecord(BaseModel):
    """One sensor row with all 18 columns from the data file."""

    # allow_inf_nan=False stops "inf" and "nan" from passing as real numbers
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # When and where the reading was taken.
    # These can be empty here. Then quality.py sets aside only that row,
    # instead of the schema rejecting the whole batch.
    timestamp: str | None = Field(
        ...,
        description="Date and time of the reading (YYYY-MM-DD HH:MM:SS).",
        examples=["2012-01-14 00:00:00"],
    )
    machine_id: str | None = Field(
        ...,
        description="Which machine the reading is from (for example M001).",
        examples=["M001"],
    )
    operating_mode: str | None = Field(
        ...,
        description="What the machine is doing: NORMAL, HIGH_LOAD, IDLE, or OVERLOAD.",
        examples=["NORMAL"],
    )

    # Sensor readings (empty if the sensor missed a reading)
    temperature: float | None = Field(default=None, description="Temperature in degrees Celsius.")
    vibration: float | None = Field(default=None, description="Vibration speed in mm/s.")
    pressure: float | None = Field(default=None, description="Pressure in kPa.")
    rpm: float | None = Field(default=None, description="Motor speed in rotations per minute.")
    power: float | None = Field(default=None, description="Power used in kW.")

    # Sensor problem labels that came with the dataset
    anomaly: bool = Field(default=False, description="True if the row has a sensor problem.")
    anomaly_type: str | None = Field(
        default=None,
        description="Type of sensor problem (missing_value, sensor_frozen, sensor_spike). C1 sets it to 'normal' if empty.",
    )

    # Failure information for C2 and C3. C1 only keeps the probability between 0 and 1.
    failure_probability_gt: float | None = Field(
        default=None, description="Chance of failure, from 0 to 1."
    )
    failure_in_7_days: bool = Field(
        default=False, description="True if the machine breaks within 7 days."
    )
    remaining_useful_life_hours: float | None = Field(
        default=None, description="Hours left before the machine breaks."
    )
    failure_type: str | None = Field(default=None, description="Which part is going to break.")
    important_features: str | None = Field(
        default=None, description="Which sensors show the failure coming."
    )
    maintenance_action: str | None = Field(
        default=None, description="Which repair job should be done."
    )
    priority: str | None = Field(
        default=None, description="How urgent the repair is: low, medium, or high."
    )
    concept_drift_active: bool = Field(
        default=False, description="True if the machine has started to behave differently."
    )


class BatchRequest(BaseModel):
    """A group of sensor rows that we clean together (one CSV file or one API upload)."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(..., description="Name or ID of this batch.", examples=["batch-001"])
    records: list[SensorRecord] = Field(..., description="The sensor rows in this batch.")


class ProcessedRecord(BaseModel):
    """A sensor row together with its status and any flags."""

    record: SensorRecord
    disposition: RecordDisposition
    flags: list[str] = Field(default_factory=list)
    quarantine_reason: str | None = None


class QualityReportSummary(BaseModel):
    """Summary of a cleaned batch: how many rows we kept, fixed, or set aside."""

    batch_id: str
    received: int = Field(..., description="Number of rows we got.")
    accepted: int = Field(
        ..., description="Rows we kept in the cleaned data (valid plus repaired)."
    )
    repaired: int = Field(default=0, description="Rows where we filled in a missing sensor value.")
    quarantined: int = Field(..., description="Bad rows we set aside.")
    duplicates_excluded: int = Field(..., description="Exact copies we skipped.")
    status: Literal["processed", "partially_processed", "rejected"] = "processed"
    reasons: list[str] = Field(default_factory=list, description="Why rows were set aside.")
