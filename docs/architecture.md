# Architecture

The platform currently contains one independently deployable service:

```text
sensor data -> C1 Sensor Quality
```

Services communicate synchronously over versioned REST APIs. Contracts belong
in `shared/predictive_maintenance`, while service-specific models and
implementation remain within each service.
