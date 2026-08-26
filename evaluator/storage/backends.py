"""Storage backend abstraction — local filesystem and Azure Blob Storage."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class StorageBackend(ABC):
    """Abstract storage backend for evaluation results and artifacts."""

    @abstractmethod
    def save(self, key: str, data: dict[str, Any]) -> str:
        """Save data under the given key. Returns the storage path/URL."""
        ...

    @abstractmethod
    def load(self, key: str) -> dict[str, Any]:
        """Load data for the given key."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in storage."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys, optionally filtered by prefix."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a key from storage."""
        ...


class LocalStorageBackend(StorageBackend):
    """Stores evaluation results as JSON files on the local filesystem."""

    def __init__(self, base_dir: str | Path = "eval-results"):
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Sanitize key to be a valid filename
        safe_key = key.replace("/", "__").replace("\\", "__")
        if not safe_key.endswith(".json"):
            safe_key += ".json"
        return self._base / safe_key

    def save(self, key: str, data: dict[str, Any]) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return str(path)

    def load(self, key: str) -> dict[str, Any]:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"Storage key not found: {key}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        keys = []
        for f in self._base.glob("*.json"):
            key = f.stem  # without .json
            key = key.replace("__", "/")
            if key.startswith(prefix):
                keys.append(key)
        return sorted(keys)

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.exists():
            path.unlink()


class AzureBlobStorageBackend(StorageBackend):
    """Stores evaluation results in Azure Blob Storage.

    Falls back gracefully if Azure credentials are not configured.
    Switch to this backend via STORAGE_BACKEND=azure environment variable.
    """

    def __init__(
        self,
        account_name: str | None = None,
        container_name: str | None = None,
        connection_string: str | None = None,
    ):
        self._account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self._container_name = container_name or os.getenv("AZURE_STORAGE_CONTAINER", "eval-results")
        self._connection_string = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self._client = None
        self._container_client = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            from azure.storage.blob import BlobServiceClient
            if self._connection_string:
                service = BlobServiceClient.from_connection_string(self._connection_string)
            else:
                from azure.identity import DefaultAzureCredential
                service = BlobServiceClient(
                    account_url=f"https://{self._account_name}.blob.core.windows.net",
                    credential=DefaultAzureCredential(),
                )
            self._container_client = service.get_container_client(self._container_name)
            # Ensure container exists
            try:
                self._container_client.create_container()
            except Exception:
                pass  # Already exists
        except ImportError:
            raise ImportError(
                "azure-storage-blob is required for Azure storage. "
                "Install with: pip install azure-storage-blob azure-identity"
            )

    def save(self, key: str, data: dict[str, Any]) -> str:
        blob_name = f"{key}.json" if not key.endswith(".json") else key
        content = json.dumps(data, indent=2, default=str).encode("utf-8")
        self._container_client.upload_blob(
            name=blob_name, data=content, overwrite=True
        )
        return f"https://{self._account_name}.blob.core.windows.net/{self._container_name}/{blob_name}"

    def load(self, key: str) -> dict[str, Any]:
        blob_name = f"{key}.json" if not key.endswith(".json") else key
        blob_client = self._container_client.get_blob_client(blob_name)
        data = blob_client.download_blob().readall()
        return json.loads(data)

    def exists(self, key: str) -> bool:
        blob_name = f"{key}.json" if not key.endswith(".json") else key
        blob_client = self._container_client.get_blob_client(blob_name)
        return blob_client.exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        blobs = self._container_client.list_blobs(name_starts_with=prefix)
        return [b.name.removesuffix(".json") for b in blobs]

    def delete(self, key: str) -> None:
        blob_name = f"{key}.json" if not key.endswith(".json") else key
        self._container_client.delete_blob(blob_name)


def build_storage_backend(backend_type: str | None = None) -> StorageBackend:
    """Build the appropriate storage backend from the STORAGE_BACKEND env var."""
    backend_type = backend_type or os.getenv("STORAGE_BACKEND", "local")
    if backend_type == "azure":
        return AzureBlobStorageBackend()
    return LocalStorageBackend()
