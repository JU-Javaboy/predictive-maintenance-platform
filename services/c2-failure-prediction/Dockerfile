FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv pip install --system .

EXPOSE 8000
CMD ["uvicorn", "c2_failure_prediction.main:app", "--host", "0.0.0.0", "--port", "8000"]
