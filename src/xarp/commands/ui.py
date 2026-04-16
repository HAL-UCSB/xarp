from typing import Literal

from pydantic import Field

from . import Command


class WriteCommand(Command):
    cmd: Literal["write"] = Field(default="write", frozen=True)
    text: str
    title: str | None = None


class SayCommand(Command):
    cmd: Literal["say"] = Field(default="say", frozen=True)
    text: str


class ReadCommand(Command):
    cmd: Literal["read"] = Field(default="read", frozen=True)

    def validate_response_value(self, value: dict) -> str:
        return str(value)


class PassthroughCommand(Command):
    cmd: Literal["passthrough"] = Field(default="passthrough", frozen=True)
    transparency: float = 0
