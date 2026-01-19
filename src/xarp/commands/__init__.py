from abc import ABC
from enum import IntEnum
from typing import Any, Literal, Union, Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from xarp.time import utc_ts


class MessageType(IntEnum):
    NOTIFICATION = 0
    SINGLE_RESPONSE = 1
    STREAM_RESPONSE = 2
    BUNDLE = 3


class ResponseMode(IntEnum):
    NONE = 0
    SINGLE = 1
    STREAM = 2


class Notification(BaseModel):
    type: Literal[MessageType.NOTIFICATION] = MessageType.NOTIFICATION
    ts: int = utc_ts()
    error: bool = False
    value: Any | None = None


class SingleResponse(Notification):
    type: Literal[MessageType.SINGLE_RESPONSE] = MessageType.SINGLE_RESPONSE
    xid: int


class StreamResponse(SingleResponse):
    type: Literal[MessageType.STREAM_RESPONSE] = MessageType.STREAM_RESPONSE
    seq: int = -1
    eos: bool


class Command(ABC, BaseModel):
    cmd: Literal[None] = None

    def validate_response_value(self, value: Any) -> Any:
        return value


class Bundle(Command):
    type: Literal[MessageType.BUNDLE] = Field(default=MessageType.BUNDLE, frozen=True)
    xid: int | None = None
    ts: int = Field(default_factory=utc_ts)
    mode: ResponseMode = ResponseMode.SINGLE
    cmds: list[Any] = Field(default_factory=list)
    rt: bool = False

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)

    def validate_response_value(self, value: list[Any]) -> Any:
        return [cmd.validate_response_value(value_i) for cmd, value_i in zip(self.cmds, value)]


IncomingMessage = Annotated[
    Union[SingleResponse, StreamResponse, Notification],
    Field(discriminator="type"),
]

IncomingMessageValidator = TypeAdapter(IncomingMessage)

Response = Annotated[
    Union[SingleResponse, StreamResponse],
    Field(discriminator="type"),
]


class Cancel(Command):
    cmd: Literal["cancel"] = Field(default="cancel", frozen=True)
    target_xid: int
