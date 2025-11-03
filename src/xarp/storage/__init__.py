from abc import ABC, abstractmethod
from typing import Generator, Optional

from xarp.data_models import Session, User


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
