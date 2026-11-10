from fastapi import FastAPI

app = FastAPI(title="Maintenance Planning Service", version="0.1.0")


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "c3-maintenance-planning"}


@app.get("/model", tags=["operations"])
def model_information() -> dict[str, str]:
    return {"model_version": "untrained", "status": "not_configured"}
