from failure_prediction.main import app as failure_prediction_app
from maintenance_planning.main import app as maintenance_planning_app
from sensor_quality.main import app as sensor_quality_app


def test_all_services_expose_fastapi_apps() -> None:
    assert failure_prediction_app.title == "Failure Prediction Service"
    assert maintenance_planning_app.title == "Maintenance Planning Service"
    assert sensor_quality_app.title == "Sensor Quality Service"
