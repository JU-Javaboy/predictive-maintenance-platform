# C2: Failure Prediction Service

```bash
uv run --directory services/c2-failure-prediction uvicorn c2_failure_prediction.main:app --reload
```

Planned endpoints include failure probability, remaining useful life,
explanations, prediction reports, and model information.
