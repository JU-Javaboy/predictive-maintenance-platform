from c1_sensor_quality.main import app as c1_sensor_quality_app
from c1_sensor_quality.monitoring import monitoring
from fastapi.testclient import TestClient

client = TestClient(c1_sensor_quality_app)


def test_all_services_expose_fastapi_apps() -> None:
    assert c1_sensor_quality_app.title == "Sensor Quality Service"


def test_health_and_readiness_endpoints() -> None:
    health = client.get("/health")
    readiness = client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "service": "c1-sensor-quality"}
    assert readiness.status_code == 200
    assert readiness.json()["status"] in {"ready", "degraded"}


def test_metrics_endpoint_reports_c1_counters() -> None:
    monitoring.record_model_behavior(
        predictions=2,
        low_confidence=1,
        concept_drift_detected=1,
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "c1_batches_total" in response.text
    assert "c1_records_received_total" in response.text
    assert "c1_model_predictions_total 2" in response.text
    assert "c1_model_low_confidence_total 1" in response.text


def test_quality_batch_endpoint_ingestion_to_report() -> None:
    response = client.post(
        "/quality/batches",
        json={
            "batch_id": "integration-batch",
            "records": [
                {
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
                    "failure_probability_gt": 0.37,
                    "failure_in_7_days": False,
                    "remaining_useful_life_hours": 750.0,
                    "failure_type": "cooling_failure",
                    "important_features": "temperature",
                    "maintenance_action": "repair_cooling_system",
                    "priority": "low",
                    "concept_drift_active": False,
                }
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["batch_id"] == "integration-batch"
    assert body["summary"]["accepted"] == 1
    assert body["summary"]["status"] == "processed"
    assert body["records"][0]["disposition"] == "valid"
