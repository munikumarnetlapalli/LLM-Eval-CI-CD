"""Dataset versioning — tracks and compares dataset versions."""
from __future__ import annotations

from pathlib import Path


class DatasetVersionManager:
    """Manages versioned golden datasets on the filesystem."""

    def __init__(self, datasets_dir: str | Path):
        self.datasets_dir = Path(datasets_dir)

    def list_versions(self) -> list[str]:
        """Return sorted list of available dataset versions."""
        if not self.datasets_dir.exists():
            return []
        versions = [
            d.name
            for d in self.datasets_dir.iterdir()
            if d.is_dir() and (d / "golden.json").exists()
        ]
        return sorted(versions)

    def latest_version(self) -> str | None:
        """Return the latest version string, or None if no datasets exist."""
        versions = self.list_versions()
        return versions[-1] if versions else None

    def version_exists(self, version: str) -> bool:
        return (self.datasets_dir / version / "golden.json").exists()

    def path_for_version(self, version: str) -> Path:
        return self.datasets_dir / version / "golden.json"
