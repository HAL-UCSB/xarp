import pathlib
import uuid
import numpy as np
from datetime import datetime, timezone
from typing import ClassVar, List, Dict, Tuple, Any

from pydantic import BaseModel, Field

from xarp.spatial import Transform, FloatArrayLike


def utc_ts():
    return int(datetime.now(timezone.utc).timestamp())


class ChatMessageContent(BaseModel):
    ts: int = Field(default_factory=utc_ts)
    text: List[str] = Field(default_factory=list)
    files: List[pathlib.PurePath] = Field(default_factory=list)


class ChatMessage(BaseModel):
    user: ClassVar[str] = 'user'
    assistant: ClassVar[str] = 'assistant'
    system: ClassVar[str] = 'system'

    ts: int = Field(default_factory=utc_ts)
    role: str
    content: ChatMessageContent

    @classmethod
    def from_user(cls, *text, files=None):
        if files is None:
            files = []
        return cls(
            role=cls.user,
            content=ChatMessageContent(
                text=list(filter(lambda s: s is not None, text)),
                files=files))

    @classmethod
    def from_assistant(cls, *text, files=None):
        if files is None:
            files = []
        return cls(
            role=cls.assistant,
            content=ChatMessageContent(
                text=list(filter(lambda s: s is not None, text)),
                files=files))

    @classmethod
    def from_system(cls, *text, files=None):
        if files is None:
            files = []
        return cls(
            role=cls.system,
            content=ChatMessageContent(
                text=list(filter(lambda s: s is not None, text)),
                files=files))


class Session(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: int = Field(default_factory=utc_ts)
    chat: List[ChatMessage] = Field(default_factory=list)


class User(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sessions: List[Session] = Field(default_factory=list)


class XRCommand(BaseModel):
    ts: int = Field(default_factory=utc_ts)
    cmd: str
    args: Tuple[Any, ...] = Field(default_factory=tuple)
    kwargs: Dict[Any, Any] = Field(default_factory=dict)


class Hands(BaseModel):
    left: Tuple[Transform, ...] = Field(default_factory=tuple)
    right: Tuple[Transform, ...] = Field(default_factory=tuple)