import uuid
from typing import ClassVar, Any, Iterable, Union

from pydantic import BaseModel, Field

from xarp.data_models.binaries import ImageResource
from xarp.data_models.data import Hands, CameraIntrinsics, DeviceInfo
from xarp.data_models.spatial import Vector3, Quaternion, Pose, Transform
from xarp.time import utc_ts

ChatUIDataTypes = Union[
    DeviceInfo,
    CameraIntrinsics,
    Hands,
    Transform,
    Vector3,
    Quaternion,
    Pose,
    ImageResource
]


class ChatMessage(BaseModel):
    user: ClassVar[str] = 'user'
    assistant: ClassVar[str] = 'assistant'
    system: ClassVar[str] = 'system'

    ts: int = Field(default_factory=utc_ts)
    role: str
    content: list[ChatUIDataTypes]

    @classmethod
    def from_user(cls, content: Any) -> 'ChatMessage':
        return cls(
            role=cls.user,
            content=list(content) if isinstance(content, Iterable) else [content])

    @classmethod
    def from_assistant(cls, content: Any) -> 'ChatMessage':
        return cls(
            role=cls.assistant,
            content=list(content) if isinstance(content, Iterable) else [content])

    @classmethod
    def from_system(cls, content: Any) -> 'ChatMessage':
        return cls(
            role=cls.system,
            content=list(content) if isinstance(content, Iterable) else [content])


class Session(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: int = Field(default_factory=utc_ts)
    chat: list[ChatMessage] = Field(default_factory=list)


class User(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sessions: list[Session] = Field(default_factory=list)
