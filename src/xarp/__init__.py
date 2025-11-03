import pathlib
from typing import Set

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = '0.0.0.0'
    port: int = 8080
    ws_route: str = '/ws'
    local_storage: pathlib.PurePath = pathlib.PurePath('.')
    authorized: Set[str] = Field(default_factory=set)


settings = Settings()
