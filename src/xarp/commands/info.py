"""Command for querying XR device and runtime information."""

from typing import Literal

from pydantic import Field

from . import Command, Response
from xarp.data_models import DeviceInfo


class InfoCommand(Command):
    """Request the connected XR client's capabilities and configuration.

    Attributes:
        cmd: Wire discriminator, always ``"info"``.
    """

    cmd: Literal["info"] = Field(default="info", frozen=True)

    def validate_response_value(self, value: Response) -> DeviceInfo:
        """Validate the response payload as :class:`xarp.data_models.DeviceInfo`.

        Args:
            value: Deserialized device-information payload.

        Returns:
            Validated device information.

        Raises:
            pydantic.ValidationError: If the payload does not match
                :class:`xarp.data_models.DeviceInfo`.
        """
        return DeviceInfo.model_validate(value)
