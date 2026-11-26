"""Conservative model-behavior and concept-drift checks."""

from typing import Literal

from pydantic import BaseModel, Field

from c1_sensor_quality.schemas import SensorRecord


class ModelBehaviorStatus(BaseModel):
    status: Literal["normal", "degraded"] = "normal"
    fallback: Literal["none", "conservative"] = "none"
    warnings: list[str] = Field(default_factory=list)
    concept_drift: bool = False
    out_of_distribution: bool = False
    low_confidence: bool = False


LOW_CONFIDENCE_MIN = 0.45
LOW_CONFIDENCE_MAX = 0.55


def evaluate_model_behavior(record: SensorRecord) -> ModelBehaviorStatus:
    warnings: list[str] = []
    concept_drift = record.concept_drift_active
    out_of_distribution = bool(
        record.anomaly and record.anomaly_type in {"sensor_spike", "sensor_frozen", "missing_value"}
    )
    probability = record.failure_probability_gt
    low_confidence = probability is None or LOW_CONFIDENCE_MIN <= probability <= LOW_CONFIDENCE_MAX

    if concept_drift:
        warnings.append("concept_drift_active")
    if out_of_distribution:
        warnings.append("out_of_distribution")
    if low_confidence:
        warnings.append("low_confidence")

    degraded = concept_drift or out_of_distribution or low_confidence
    return ModelBehaviorStatus(
        status="degraded" if degraded else "normal",
        fallback="conservative" if degraded else "none",
        warnings=warnings,
        concept_drift=concept_drift,
        out_of_distribution=out_of_distribution,
        low_confidence=low_confidence,
    )


def apply_conservative_fallback(
    record: SensorRecord,
    status: ModelBehaviorStatus,
) -> SensorRecord:
    if status.status != "degraded":
        return record
    return record.model_copy(update={"failure_probability_gt": None})
