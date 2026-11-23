"""In-process operational metrics for the C1 sensor-quality service."""

from threading import Lock

from c1_sensor_quality.schemas import QualityReportSummary


class MonitoringState:
    def __init__(self) -> None:
        self._lock = Lock()
        self.batches_total = 0
        self.records_received = 0
        self.records_accepted = 0
        self.records_repaired = 0
        self.records_quarantined = 0
        self.duplicates_excluded = 0
        self.batch_statuses = {"processed": 0, "partially_processed": 0, "rejected": 0}
        self.model_predictions = 0
        self.model_low_confidence = 0
        self.concept_drift_detected = 0

    def record_batch(self, summary: QualityReportSummary) -> None:
        with self._lock:
            self.batches_total += 1
            self.records_received += summary.received
            self.records_accepted += summary.accepted
            self.records_repaired += summary.repaired
            self.records_quarantined += summary.quarantined
            self.duplicates_excluded += summary.duplicates_excluded
            self.batch_statuses[summary.status] += 1

    def record_model_behavior(
        self,
        *,
        predictions: int = 0,
        low_confidence: int = 0,
        concept_drift_detected: int = 0,
    ) -> None:
        with self._lock:
            self.model_predictions += predictions
            self.model_low_confidence += low_confidence
            self.concept_drift_detected += concept_drift_detected

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "batches_total": self.batches_total,
                "records_received": self.records_received,
                "records_accepted": self.records_accepted,
                "records_repaired": self.records_repaired,
                "records_quarantined": self.records_quarantined,
                "duplicates_excluded": self.duplicates_excluded,
                "batch_statuses": dict(self.batch_statuses),
                "model_predictions": self.model_predictions,
                "model_low_confidence": self.model_low_confidence,
                "concept_drift_detected": self.concept_drift_detected,
            }

    def prometheus_text(self) -> str:
        snapshot = self.snapshot()
        statuses = snapshot["batch_statuses"]
        assert isinstance(statuses, dict)
        lines = [
            "# HELP c1_batches_total Number of data-quality batches processed.",
            "# TYPE c1_batches_total counter",
            f"c1_batches_total {snapshot['batches_total']}",
            "# HELP c1_records_received_total Sensor records received.",
            "# TYPE c1_records_received_total counter",
            f"c1_records_received_total {snapshot['records_received']}",
            "# HELP c1_records_accepted_total Sensor records accepted or repaired.",
            "# TYPE c1_records_accepted_total counter",
            f"c1_records_accepted_total {snapshot['records_accepted']}",
            "# HELP c1_records_repaired_total Sensor records repaired.",
            "# TYPE c1_records_repaired_total counter",
            f"c1_records_repaired_total {snapshot['records_repaired']}",
            "# HELP c1_records_quarantined_total Sensor records quarantined.",
            "# TYPE c1_records_quarantined_total counter",
            f"c1_records_quarantined_total {snapshot['records_quarantined']}",
            "# HELP c1_duplicates_excluded_total Duplicate sensor records excluded.",
            "# TYPE c1_duplicates_excluded_total counter",
            f"c1_duplicates_excluded_total {snapshot['duplicates_excluded']}",
            "# HELP c1_batch_status_total Batches grouped by processing status.",
            "# TYPE c1_batch_status_total counter",
        ]
        for status in ("processed", "partially_processed", "rejected"):
            lines.append(f'c1_batch_status_total{{status="{status}"}} {statuses[status]}')
        lines.extend(
            [
                "# HELP c1_model_predictions_total Model predictions observed.",
                "# TYPE c1_model_predictions_total counter",
                f"c1_model_predictions_total {snapshot['model_predictions']}",
                "# HELP c1_model_low_confidence_total Low-confidence model predictions observed.",
                "# TYPE c1_model_low_confidence_total counter",
                f"c1_model_low_confidence_total {snapshot['model_low_confidence']}",
                "# HELP c1_concept_drift_detected_total Concept-drift warnings observed.",
                "# TYPE c1_concept_drift_detected_total counter",
                f"c1_concept_drift_detected_total {snapshot['concept_drift_detected']}",
            ]
        )
        return "\n".join(lines) + "\n"


monitoring = MonitoringState()
