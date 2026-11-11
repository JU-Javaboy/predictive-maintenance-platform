# Predictive Maintenance Platform

An AI-enabled predictive maintenance platform centered on an independently
deployable REST service:

- **C1** sensor quality and data validation

## Development

Install the workspace environments and development tools:

```bash
uv sync --all-packages
```

Run the test suite and lint checks:

```bash
uv run pytest
uv run ruff check .
```

The C1 service has its own `pyproject.toml` and can be run independently with
the command documented in its service directory.

## Data split

The dataset must be split chronologically: the first two years are available
for training and validation, while the final year is reserved exclusively for
testing. Raw, processed, and model artifacts are intentionally kept outside
the service packages.
