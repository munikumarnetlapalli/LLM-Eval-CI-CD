"""Dataset loader — reads and validates golden datasets from disk."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from .schema import DatasetManifest, EvaluationCase, RAGEvaluationCase


class DatasetLoader:
    """Load versioned golden datasets from the filesystem."""

    def load(self, path: Union[str, Path]) -> DatasetManifest:
        """Load a golden.json dataset file and return a validated DatasetManifest."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        # Coerce cases to the correct type based on presence of RAG fields
        raw_cases = raw.get("cases", [])
        typed_cases: list[EvaluationCase | RAGEvaluationCase] = []
        for c in raw_cases:
            if "required_sources" in c or "context_snippet" in c:
                typed_cases.append(RAGEvaluationCase(**c))
            else:
                typed_cases.append(EvaluationCase(**c))
        raw["cases"] = typed_cases

        return DatasetManifest(**raw)

    def load_version(self, datasets_dir: Union[str, Path], version: str) -> DatasetManifest:
        """Load a specific version of the dataset (e.g., v1)."""
        datasets_dir = Path(datasets_dir)
        path = datasets_dir / version / "golden.json"
        return self.load(path)
