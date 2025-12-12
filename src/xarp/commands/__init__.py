from enum import IntEnum

from xarp.time import utc_ts
from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict, JsonValue


class ResponseMode(IntEnum):
    NONE = 0,
    SINGLE = 1,
    STREAM = 2


class XRCommand(BaseModel):
    model_config = ConfigDict(
        extra='forbid'
    )
    cmd: Literal[None] = None
    ts: int = Field(default_factory=utc_ts)
    xid: int = None
    delay: int | None = None

    response_mode: ResponseMode = ResponseMode.SINGLE

    def validate_response(self, json_data: dict) -> Any:
        return json_data

    @property
    def expects_response(self) -> bool:
        return self.response_mode != ResponseMode.NONE


class XRResponse(BaseModel):
    ts: int = Field(default_factory=utc_ts)
    xid: int | None = None
    value: JsonValue


class CancelCommand(XRCommand):
    cmd: Literal['cancel'] = Field('cancel', frozen=True)
    target_xid: int

    response_mode: ResponseMode = ResponseMode.NONE
