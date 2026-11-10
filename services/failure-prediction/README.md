# C2: Failure Prediction Service

```bash
uv run --directory services/failure-prediction uvicorn failure_prediction.main:app --reload
```

Planned endpoints include failure probability, remaining useful life,
explanations, prediction reports, and model information.
