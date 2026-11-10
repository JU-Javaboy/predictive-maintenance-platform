from c1_sensor_quality.main import app as c1_sensor_quality_app
from c2_failure_prediction.main import app as c2_failure_prediction_app
from c3_maintenance_planning.main import app as c3_maintenance_planning_app


def test_all_services_expose_fastapi_apps() -> None:
    assert c1_sensor_quality_app.title == "Sensor Quality Service"
    assert c2_failure_prediction_app.title == "Failure Prediction Service"
    assert c3_maintenance_planning_app.title == "Maintenance Planning Service"
