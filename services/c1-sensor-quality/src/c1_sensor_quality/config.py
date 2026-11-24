"""Load the C1 settings shared by data cleaning and anomaly detection.

``imputation_params.json`` is the single source of truth for the per-mode
medians (and for the cleaning thresholds). Both ``quality.py`` and the anomaly
detector read it through this module so they cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_IMPUTATION_PARAMS_PATH = Path(__file__).parent / "imputation_params.json"


def load_imputation_params(
    path: str | Path = DEFAULT_IMPUTATION_PARAMS_PATH,
) -> dict[str, Any]:
    """Read and parse the settings file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Imputation settings not found: {source}. "
            "The anomaly detector and the cleaning rules share this file."
        )
    with source.open(encoding="utf-8") as file:
        return json.load(file)
