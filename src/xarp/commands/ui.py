from typing import Literal
from pydantic import Field
from . import Command, Response


class WriteCommand(Command):
    type: Literal["write"] = Field(default="write", frozen=True)
    text: str
    title: str | None = None


class SayCommand(WriteCommand):
    type: Literal["say"] = Field(default="say", frozen=True)


class ReadCommand(WriteCommand):
    type: Literal["read"] = Field(default="read", frozen=True)

    def validate_response_value(self, value: dict) -> str:
        return str(value)


class PassthroughCommand(Command):
    type: Literal["passthrough"] = Field(default="passthrough", frozen=True)
    transparency: float = 0
