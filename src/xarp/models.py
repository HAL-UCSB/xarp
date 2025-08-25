import base64
import pathlib
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, Any, Annotated
from typing import Optional, ClassVar, List
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, BeforeValidator
from scipy.spatial.transform import Rotation


def utc_timestamp():
    return int(datetime.now(timezone.utc).timestamp())


class ChatFile(BaseModel):
    mime_type: str
    path: pathlib.PurePath = None
    raw: Optional[BytesIO] = Field(exclude=True, default=None)
    base64_encoded: Optional[str] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    def save(self, path: pathlib.PurePath = None):
        if path:
            self.path = path
        with self.path.open('wb') as file:
            data = self.raw.getbuffer() if self.raw else base64.decode(self.base64_encoded)
            file.write(data)


class ChatContent(BaseModel):
    ts: int = Field(default_factory=utc_timestamp)
    text: str
    files: list[ChatFile | pathlib.PurePath] = Field(default_factory=list)

    def save_files(self, dir_path: pathlib.PurePath):
        for i, file in enumerate(self.files):
            if file.path is None:
                file.path = dir_path / i
            file.save(dir_path)


class ChatMessage(BaseModel):
    user: ClassVar[str] = 'user'
    assistant: ClassVar[str] = 'assistant'
    system: ClassVar[str] = 'system'
    ts: int = Field(default_factory=utc_timestamp)
    role: str
    content: ChatContent

    @classmethod
    def from_user(cls, text='', files=[]):
        return cls(
            role=cls.user,
            content=ChatContent(
                text=text,
                files=files))

    @classmethod
    def from_assistant(cls, text='', files=[]):
        return cls(
            role=cls.assistant,
            content=ChatContent(
                text=text,
                files=files))

    @classmethod
    def from_system(cls, text='', files=[]):
        return cls(
            role=cls.system,
            content=ChatContent(
                text=text,
                files=files))




class Chat(BaseModel):
    messages: List[ChatMessage] = Field(default_factory=list)


class XRCommand(BaseModel):
    ts: int = Field(default_factory=utc_timestamp)
    cmd: str
    args: List[object] = Field(default_factory=list)
    kwargs: Dict[object, object] = Field(default_factory=dict)


class Device(BaseModel):
    device_id: UUID = Field(default_factory=uuid.uuid4)
    capabilities: List[str] = Field(default_factory=list)


class Session(BaseModel):
    ts: int = Field(default_factory=utc_timestamp)
    chat: Chat = Field(default_factory=Chat)
    devices: List[Device] = Field(default_factory=list)


class User(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sessions: List[Session] = list()


def validate_numpy_array_length(expected_length):
    def _inner(v: Any) -> np.ndarray:
        array = np.array(v, dtype=float)
        if len(array) == expected_length:
            return array
        raise ValueError(f'length must be {expected_length}')

    return _inner


class Pose(BaseModel):
    position: Annotated[
        np.ndarray,
        Field(default_factory=lambda: np.zeros(3)),
        BeforeValidator(validate_numpy_array_length(3))
    ]
    rotation: Annotated[
        np.ndarray,
        Field(default_factory=lambda: np.array([0, 0, 0, 1])),
        BeforeValidator(validate_numpy_array_length(4))
    ]

    @computed_field
    def euler_angles(self) -> np.ndarray:
        return Rotation.from_quat(self.rotation).as_euler('xyz', degrees=True)

    @euler_angles.setter
    def euler_angles(self, v: np.ndarray) -> None:
        self.rotation = Rotation.from_euler('xyz', v, degrees=True).as_quat()

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }


class Transform(Pose):
    scale: Annotated[
        np.ndarray,
        Field(default_factory=lambda: np.ones(3)),
        BeforeValidator(validate_numpy_array_length(3))
    ]
