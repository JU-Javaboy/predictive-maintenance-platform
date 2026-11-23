from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from c1_sensor_quality.monitoring import monitoring
from c1_sensor_quality.quality import process_batch
from c1_sensor_quality.schemas import BatchRequest

app = FastAPI(title="Sensor Quality Service", version="0.1.0")


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "c1-sensor-quality"}


@app.get("/ready", tags=["operations"])
def readiness() -> dict[str, object]:
    snapshot = monitoring.snapshot()
    statuses = snapshot["batch_statuses"]
    assert isinstance(statuses, dict)
    return {
        "status": "degraded" if statuses["rejected"] else "ready",
        "service": "c1-sensor-quality",
        "batches_processed": snapshot["batches_total"],
    }


@app.get("/model", tags=["operations"])
def model_information() -> dict[str, str]:
    return {"model_version": "untrained", "status": "not_configured"}


@app.post("/quality/batches", tags=["quality"])
def process_quality_batch(batch: BatchRequest) -> dict[str, object]:
    processed, summary = process_batch(batch)
    return {"summary": summary, "records": processed}


@app.get("/metrics", tags=["operations"], response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        monitoring.prometheus_text(),
        media_type="text/plain",
        headers={"Cache-Control": "no-store"},
    )
