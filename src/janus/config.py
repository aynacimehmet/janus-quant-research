"""Konfigürasyon: sırlar .env'den (pydantic-settings), parametreler config/janus.yaml'dan."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Yalnızca sırlar ve ortam bayrakları. Değerler asla loglanmaz."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    evds_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    mkk_api_key: str = ""
    report_absolute: bool = False
    janus_data_dir: str = "data"

    def secrets_present(self) -> dict[str, bool]:
        return {
            "telegram": bool(self.telegram_bot_token and self.telegram_chat_id),
            "evds": bool(self.evds_api_key),
            "alpaca": bool(self.alpaca_api_key and self.alpaca_secret_key),
            "mkk": bool(self.mkk_api_key),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else ROOT / "config" / "janus.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def store_path(cfg: dict[str, Any]) -> Path:
    rel = cfg.get("store", {}).get("path", "data/janus.duckdb")
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p
