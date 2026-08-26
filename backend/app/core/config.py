"""Application settings — loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-this-in-production"
    debug: bool = False

    # Database
    database_url: str = "postgresql://evaluser:evalpass@localhost:5432/llmeval"

    # Storage
    storage_backend: str = "local"

    # LLM Provider
    llm_provider: str = "mock"
    openai_api_key: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_openai_api_version: str = "2024-02-01"
    gemini_api_key: Optional[str] = None

    # Azure Storage
    azure_storage_account_name: Optional[str] = None
    azure_storage_container: str = "eval-results"
    azure_storage_connection_string: Optional[str] = None

    # Azure Identity
    azure_client_id: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_client_secret: Optional[str] = None

    # Application Insights
    applicationinsights_connection_string: Optional[str] = None

    # Evaluation
    default_dataset_version: str = "v1"
    default_dataset_path: str = "datasets/v1/golden.json"
    eval_timeout_seconds: int = 60
    eval_max_concurrent: int = 5

    # Logging
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
