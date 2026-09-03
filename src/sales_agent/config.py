"""Configuration: env-driven OpenAI-compatible settings and data paths.

Settings are read entirely from the environment (or a local, git-ignored
`.env`). The OpenAI client is pointed at whatever `OPENAI_BASE_URL` names, so
this code works against api.openai.com or any OpenAI-compatible endpoint
without knowing anything about it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_FILE = (
    _PACKAGE_ROOT / "data" / "Store_Sales_Price_Elasticity_Promotions_Data.parquet"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")

    phoenix_collector_endpoint: str = Field(default="http://localhost:6006")
    phoenix_project_name: str = Field(default="sales-agent")

    data_file_path: Path = Field(default=_DEFAULT_DATA_FILE)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_openai_client() -> OpenAI:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url
    )
