import pathlib
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    host: str = "0.0.0.0"
    port: int = 8080
    ws_path: str = "/ws"
    local_storage: pathlib.Path = pathlib.Path(".") / "data"
    heartbeat_timeout_secs: float = 5.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()