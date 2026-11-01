"""Shared helpers for reproducible RIO synthetic datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SEED = 20261031
GENERATOR_VERSION = "1.0.0"


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return the one random generator used by a generation run."""
    return np.random.default_rng(seed)


def choose(rng: np.random.Generator, values: list[Any]) -> Any:
    return values[int(rng.integers(0, len(values)))]


def write_metadata(
    path: Path,
    *,
    dataset_name: str,
    row_count: int,
    seed: int,
    output_file: str,
    columns: list[str],
) -> None:
    metadata = {
        "dataset_name": dataset_name,
        "synthetic": True,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "row_count": row_count,
        "output_file": output_file,
        "columns": columns,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "intended_use": "Pipeline development only; not evidence of real-world performance.",
        "future_real_data_policy": "Keep real-world datasets in a separate directory and metadata record.",
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def json_list(value: list[str]) -> str:
    return json.dumps(value, ensure_ascii=True)
