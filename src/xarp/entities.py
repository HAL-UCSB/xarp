import uuid
from abc import abstractmethod, ABC
from typing import Optional, Generator

from pydantic import BaseModel, Field

from xarp.time import utc_ts
from xarp.chat import ChatMessage


class Session(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts: int = Field(default_factory=utc_ts)
    chat: list[ChatMessage] = Field(default_factory=list)


class User(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sessions: list[Session] = Field(default_factory=list)


class SessionRepository(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, user_id: str, ts: int) -> Optional[Session]:
        pass

    @abstractmethod
    def all(self, user_id: str = None) -> Generator[Session]:
        pass

    @abstractmethod
    def save(self, session: Session) -> None:
        pass


class UserRepository(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, user_id: str) -> Optional[User]:
        pass

    @abstractmethod
    def all(self) -> Generator[User]:
        pass

    @abstractmethod
    def save(self, user: User) -> None:
        pass
