from c1_sensor_quality.main import app as c1_sensor_quality_app


def test_all_services_expose_fastapi_apps() -> None:
    assert c1_sensor_quality_app.title == "Sensor Quality Service"
