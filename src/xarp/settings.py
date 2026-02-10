import pathlib

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0" #"127.0.0.1"
    port: int = 8080
    ws_path: str = "/ws"
    local_storage: pathlib.PurePath = pathlib.PurePath(".") / "data"
    heartbeat_timeout_secs: float = 5


settings = Settings()
