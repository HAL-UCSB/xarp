"""Wire protocol models and base command types used by XARP transports.

Commands are normally bundled and executed by :class:`xarp.remote.RemoteXRClient`.
Incoming MessagePack payloads are parsed through
:data:`IncomingMessageValidator` before response values are converted by their
corresponding command objects.
"""

from abc import ABC
from enum import IntEnum
from typing import Any, Literal, Union, Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from xarp.time import utc_ts


class MessageType(IntEnum):
    """Discriminator values for messages received from or sent to an XR client.

    Attributes:
        NOTIFICATION: Unsolicited notification or heartbeat.
        SINGLE_RESPONSE: One response to a bundle transaction.
        STREAM_RESPONSE: One frame in a streaming transaction.
        BUNDLE: Outgoing collection of commands.
    """

    NOTIFICATION = 0
    SINGLE_RESPONSE = 1
    STREAM_RESPONSE = 2
    BUNDLE = 3


class ResponseMode(IntEnum):
    """Response policy requested for an outgoing :class:`Bundle`.

    Attributes:
        NONE: Send without allocating a transaction ID or awaiting a response.
        SINGLE: Await one response containing a value for each command.
        STREAM: Yield responses until end-of-stream or cancellation.
    """

    NONE = 0
    SINGLE = 1
    STREAM = 2


class Notification(BaseModel):
    """Incoming notification or heartbeat not tied to a transaction.

    Attributes:
        type: Message discriminator, always :attr:`MessageType.NOTIFICATION`.
        ts: Sender timestamp in UTC milliseconds.
        error: Whether the notification represents an error.
        value: Optional notification payload.
    """

    type: Literal[MessageType.NOTIFICATION] = MessageType.NOTIFICATION
    ts: int = utc_ts()
    error: bool = False
    value: Any | None = None


class SingleResponse(Notification):
    """Terminal response to a bundle using :attr:`ResponseMode.SINGLE`.

    Attributes:
        type: Message discriminator, always
            :attr:`MessageType.SINGLE_RESPONSE`.
        xid: Transaction identifier of the originating bundle.
    """

    type: Literal[MessageType.SINGLE_RESPONSE] = MessageType.SINGLE_RESPONSE
    xid: int


class StreamResponse(SingleResponse):
    """One response frame from a bundle using :attr:`ResponseMode.STREAM`.

    Attributes:
        type: Message discriminator, always
            :attr:`MessageType.STREAM_RESPONSE`.
        seq: Stream sequence number supplied by the XR client.
        eos: Whether this frame terminates the stream. End-of-stream frames are
            not yielded as application data.
    """

    type: Literal[MessageType.STREAM_RESPONSE] = MessageType.STREAM_RESPONSE
    seq: int = -1
    eos: bool


class Command(ABC, BaseModel):
    """Base model for a command sent to an XR client.

    Attributes:
        cmd: Wire command discriminator. Concrete commands override this with a
            frozen string literal.
    """

    cmd: Literal[None] = None

    def validate_response_value(self, value: Any) -> Any:
        """Convert a raw response value into the command's public result.

        The base implementation returns the value unchanged. Concrete commands
        override this hook when their response needs validation or decoding.

        Args:
            value: Deserialized response payload supplied by the XR client.

        Returns:
            Converted response value.
        """
        return value


class Bundle(Command):
    """Collection of commands executed as one remote transaction.

    Attributes:
        type: Message discriminator, always :attr:`MessageType.BUNDLE`.
        xid: Transaction identifier assigned by the transport. It is omitted in
            :attr:`ResponseMode.NONE`.
        ts: Bundle creation timestamp in UTC milliseconds.
        mode: Requested response policy.
        cmds: Commands executed in list order.
        rt: For streaming bundles, retain only the latest normal frame when the
            consumer falls behind. Error and end-of-stream frames are never
            dropped.
    """

    type: Literal[MessageType.BUNDLE] = Field(default=MessageType.BUNDLE, frozen=True)
    xid: int | None = None
    ts: int = Field(default_factory=utc_ts)
    mode: ResponseMode = ResponseMode.SINGLE
    cmds: list[Any] = Field(default_factory=list)
    rt: bool = False

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """Serialize the bundle while always omitting fields whose value is ``None``.

        Args:
            *args: Positional arguments forwarded to Pydantic's ``model_dump``.
            **kwargs: Keyword arguments forwarded to Pydantic's ``model_dump``.

        Returns:
            Dictionary ready for transport serialization.
        """
        kwargs["exclude_none"] = True
        return super().model_dump(*args, **kwargs)

    def validate_response_value(self, value: list[Any]) -> Any:
        """Convert bundle response values with their corresponding commands.

        Values and commands are paired in list order. Pairing stops at the
        shorter input, following :func:`zip` semantics.

        Args:
            value: Raw response values from the XR client.

        Returns:
            List of converted values in command order.
        """
        return [cmd.validate_response_value(value_i) for cmd, value_i in zip(self.cmds, value)]


#: Discriminated union of all accepted incoming protocol messages.
IncomingMessage = Annotated[
    Union[SingleResponse, StreamResponse, Notification],
    Field(discriminator="type"),
]

#: Pydantic adapter used to parse deserialized incoming message payloads.
IncomingMessageValidator = TypeAdapter(IncomingMessage)

#: Discriminated union of incoming messages associated with transactions.
Response = Annotated[
    Union[SingleResponse, StreamResponse],
    Field(discriminator="type"),
]


class Cancel(Command):
    """Stop a streaming transaction before its end-of-stream frame arrives.

    Cancellation is sent in a fire-and-forget bundle when a stream generator is
    closed early.

    Attributes:
        cmd: Wire discriminator, always ``"cancel"``.
        target_xid: Transaction identifier of the stream to stop.
    """

    cmd: Literal["cancel"] = Field(default="cancel", frozen=True)
    target_xid: int
