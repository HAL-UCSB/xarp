import uuid
from typing import List

from pydantic import BaseModel, Field

from xarp.time import utc_ts
from xarp.data_models.chat import ChatMessage


class Session(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: int = Field(default_factory=utc_ts)
    chat: List[ChatMessage] = Field(default_factory=list)


class User(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sessions: List[Session] = Field(default_factory=list)
