import pathlib
from typing import Generator, Optional

from .entities import Session, User, SessionRepository, UserRepository
from .resources import BinaryResource

CHAT_FILE = "chat.json"
FILES_DIR = "files"


def _is_session_dir(path: pathlib.Path) -> bool:
    return path.is_dir() and (path / CHAT_FILE).exists()


def _load_session(path: pathlib.Path) -> Session:
    chat_path = path / CHAT_FILE
    with chat_path.open() as f:
        chat_json = f.read()
        return Session.model_validate_json(chat_json)


def _load_user(path: pathlib.Path) -> User | None:
    if not path.is_dir():
        return None
    sessions = [_load_session(session_path) for session_path in path.iterdir() if _is_session_dir(session_path)]
    return User(user_id=path.name, sessions=sessions)


class FileSessionRepository(SessionRepository):

    def __init__(self, local_storage: pathlib.Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_storage = pathlib.Path(local_storage)

    def get(self, user_id: str, ts: int) -> Optional[Session]:
        user_session_path = self._local_storage / user_id / str(ts)
        return _load_session(user_session_path)

    def all(self, user_id: str = None) -> Generator[Session]:
        if user_id is None:
            for user_path in self._local_storage.iterdir():
                yield from self.all(user_path.name)
        else:
            path = self._local_storage / str(user_id)
            for session_path in path.iterdir():
                if _is_session_dir(session_path):
                    yield _load_session(session_path)

    def save(self, session: Session) -> None:
        session_path = self._local_storage / str(session.user_id) / str(session.ts)
        session_path.mkdir(parents=True, exist_ok=True)
        chat_path = session_path / CHAT_FILE
        files_path = session_path / FILES_DIR
        files_path.mkdir(parents=True, exist_ok=True)

        i = 0
        for chat_message in session.chat:
            for content in chat_message.content:
                bin_path = files_path / f"{i}.png"
                if isinstance(content, BinaryResource):
                    content.to_file(bin_path)
                    i += 1

        session_json = session.model_dump_json()
        chat_path.write_text(session_json, encoding="utf-8")


class FileUserRepository(UserRepository):

    def __init__(self, local_storage: pathlib.Path = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_storage = pathlib.Path(local_storage)

    def get(self, user_id: str) -> Optional[User]:
        pass

    def all(self) -> Generator[User]:
        for user_path in self._local_storage.iterdir():
            if user := _load_user(user_path):
                yield user

    def save(self, user: User) -> None:
        pass
