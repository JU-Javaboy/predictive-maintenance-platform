# C1: Sensor Quality Service

```bash
uv run --directory services/sensor-quality uvicorn sensor_quality.main:app --reload
```

Planned endpoints include data validation, quality reports, and model
information. Cleaned data will be sent to C2 through a versioned REST
contract.
