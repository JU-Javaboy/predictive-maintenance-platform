# Architecture

The platform is split into three independently deployable services:

```text
sensor data -> C1 Sensor Quality -> C2 Failure Prediction -> C3 Maintenance Planning
```

Services communicate synchronously over versioned REST APIs. Contracts belong
in `shared/predictive_maintenance`, while service-specific models and
implementation remain within each service.
