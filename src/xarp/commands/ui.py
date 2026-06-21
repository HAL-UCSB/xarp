"""Commands for text, speech, input, and passthrough user interfaces."""

from typing import Literal

from pydantic import Field

from . import Command


class WriteCommand(Command):
    """Display a temporary text panel on the XR client.

    Attributes:
        cmd: Wire discriminator, always ``"write"``.
        text: Message shown to the user.
        title: Optional panel title.
        hide_after_seconds: Time in seconds before the client hides the panel.
    """

    cmd: Literal["write"] = Field(default="write", frozen=True)
    text: str
    title: str | None = None
    hide_after_seconds: int = 5


class SayCommand(Command):
    """Request speech synthesis and wait for playback completion.

    Attributes:
        cmd: Wire discriminator, always ``"say"``.
        text: Text spoken by the XR client.
    """

    cmd: Literal["say"] = Field(default="say", frozen=True)
    text: str


class ReadCommand(Command):
    """Prompt the user for textual input.

    Attributes:
        cmd: Wire discriminator, always ``"read"``.
    """

    cmd: Literal["read"] = Field(default="read", frozen=True)

    def validate_response_value(self, value: dict) -> str:
        """Convert the returned input payload to a string.

        Args:
            value: Deserialized value returned by the XR client.

        Returns:
            String representation of the returned value.
        """
        return str(value)


class PassthroughCommand(Command):
    """Set how much of the physical environment is visible.

    Attributes:
        cmd: Wire discriminator, always ``"passthrough"``.
        transparency: Passthrough amount, conventionally from ``0.0`` (fully
            virtual) to ``1.0`` (fully passthrough). This model does not clamp or
            validate the range.
    """

    cmd: Literal["passthrough"] = Field(default="passthrough", frozen=True)
    transparency: float = 0
