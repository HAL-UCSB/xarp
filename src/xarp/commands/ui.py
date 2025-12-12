from typing import Literal
from pydantic import Field
from xarp.commands import XRCommand


class WriteCommand(XRCommand):
    cmd: Literal['write'] = Field('write', frozen=True)
    text: str
    title: str | None = None


class SayCommand(WriteCommand):
    cmd: Literal['say'] = Field('say', frozen=True)


class ReadCommand(WriteCommand):
    cmd: Literal['read'] = Field('read', frozen=True)

    def validate_response(self, json_data: dict) -> str:
        return str(json_data)
