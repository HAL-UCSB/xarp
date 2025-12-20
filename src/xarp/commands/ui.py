from typing import Literal
from pydantic import Field, JsonValue
from xarp.commands import XRCommand


class WriteCommand(XRCommand):
    cmd: Literal['write'] = Field(default='write', frozen=True)
    text: str
    title: str | None = None


class SayCommand(WriteCommand):
    cmd: Literal['say'] = Field(default='say', frozen=True)


class ReadCommand(WriteCommand):
    cmd: Literal['read'] = Field(default='read', frozen=True)

    def validate_response(self, json_data: JsonValue) -> str:
        return str(json_data)


class PassthroughCommand(XRCommand):
    cmd: Literal['passthrough'] = Field(default='passthrough', frozen=True)
    transparency: float = 0

    def validate_response(self, json_data: JsonValue) -> bool:
        return bool(json_data)
