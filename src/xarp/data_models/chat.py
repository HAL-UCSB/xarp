from typing import ClassVar, Any

from pydantic import BaseModel, Field

from xarp.time import utc_ts


class ChatMessage(BaseModel):
    user: ClassVar[str] = 'user'
    assistant: ClassVar[str] = 'assistant'
    system: ClassVar[str] = 'system'

    ts: int = Field(default_factory=utc_ts)
    role: str
    mimetype: str = 'text/plain'
    content: Any

    @classmethod
    def from_user(cls,
                  content: Any,
                  mimetype: str) -> 'ChatMessage':
        return cls(
            role=cls.user,
            mimetype=mimetype,
            content=content)

    @classmethod
    def from_assistant(cls,
                       content: Any,
                       mimetype: str = 'text/plain') -> 'ChatMessage':
        return cls(
            role=cls.assistant,
            mimetype=mimetype,
            content=content)

    @classmethod
    def from_system(cls,
                    content: Any,
                    mimetype: str = 'text/plain') -> 'ChatMessage':
        return cls(
            role=cls.system,
            mimetype=mimetype,
            content=content)