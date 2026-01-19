from typing import Literal

from pydantic import Field

from . import Command, Response
from xarp.data_models import DeviceInfo


class InfoCommand(Command):
    cmd: Literal["info"] = Field(default="info", frozen=True)

    def validate_response_value(self, value: Response) -> DeviceInfo:
        return DeviceInfo.model_validate(value)
