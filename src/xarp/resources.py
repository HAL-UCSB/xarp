import base64
import mimetypes
from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, ClassVar

from PIL import Image
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_serializer, Field


class MIMEType(str, Enum):
    PLAIN = mimetypes.types_map['.txt']
    PNG = mimetypes.types_map['.png']
    JPEG = mimetypes.types_map['.jpg']
    MP3 = mimetypes.types_map['.mp3']
    WAV = mimetypes.types_map['.wav']
    OGG = 'audio/ogg'
    MP4 = mimetypes.types_map['.mp4']
    GLB = 'model/gltf-binary'


class BinaryResource(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mime_type: Literal[None] = None
    mode: Literal['memory', 'file'] = 'memory'
    path: Path | None = None
    data: str | None = None

    _obj: Any | None = PrivateAttr(default=None)

    @abstractmethod
    def _encode_obj(self, obj: Any) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def _decode_obj(self, data: bytes) -> Any:
        raise NotImplementedError

    def model_post_init(self, __context: Any) -> None:
        if self.mode == 'memory' and self._obj is None and self.data is not None:
            raw = base64.b64decode(self.data)
            self._obj = self._decode_obj(raw)
            self.data = None

    @classmethod
    def from_obj(cls, obj: Any) -> 'BinaryResource':
        inst = cls(mode='memory', path=None, data=None)
        inst._obj = obj
        return inst

    @classmethod
    def from_path(cls, path: str | Path, *, load: bool = False) -> 'BinaryResource':
        path = Path(path)
        if load:
            with path.open('rb') as f:
                raw = f.read()
            inst = cls(mode='memory', path=path)
            inst._obj = inst._decode_obj(raw)
        else:
            inst = cls(mode='file', path=path)
        return inst

    def to_file(self, path: str | Path | None = None, *, overwrite: bool = True) -> 'BinaryResource':
        obj = self._require_obj()

        if path is None:
            if self.path is None:
                raise ValueError('No path provided and no existing .path set.')
            path = self.path

        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f'File exists: {path}')

        raw = self._encode_obj(obj)
        with path.open('wb') as f:
            f.write(raw)

        self.path = path
        self._obj = None
        self.mode = 'file'
        return self

    def to_memory(self) -> 'BinaryResource':
        if self._obj is None:
            if self.path is None:
                raise RuntimeError('Cannot load into memory; no path set.')
            with self.path.open('rb') as f:
                raw = f.read()
            self._obj = self._decode_obj(raw)
        self.mode = 'memory'
        self.path = None
        return self

    def _require_obj(self) -> Any:
        if self._obj is None:
            if self.mode == 'file':
                self.to_memory()
            else:
                raise RuntimeError('No in-memory object available.')
        return self._obj

    def as_obj(self) -> Any:
        return self._require_obj()

    @model_serializer(mode='plain')
    def serialize(self):
        if self.mode == 'memory':
            raw = self._encode_obj(self._require_obj())
            # b64 = base64.b64encode(raw).decode('ascii')
            return {
                'mime_type': self.mime_type,
                'data': raw,  # 'data': b64,
            }

        if self.mode == 'file':
            if self.path is None:
                raise RuntimeError('mode="file" but no path is set.')
            return {
                'mime_type': self.mime_type,
                'path': str(self.path),
            }

        raise RuntimeError(f'Unknown mode: {self.mode!r}')


class ImageResource(BinaryResource):
    mime_type: Literal[MIMEType.PNG.value] = Field(default=MIMEType.PNG.value, frozen=True)
    format: ClassVar[str] = 'PNG'
    _obj: Image.Image | None = PrivateAttr(default=None)

    def _encode_obj(self, obj: Image.Image) -> bytes:
        buf = BytesIO()
        obj.save(buf, format=ImageResource.format)  # obj.convert('RGBA').save(buf, format=ImageResource.format)
        return buf.getvalue()

    def _decode_obj(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data))

    @classmethod
    def from_image(cls, image: Image.Image) -> 'ImageResource':
        return cls.from_obj(image)


class TextResource(BinaryResource):
    mime_type: Literal[MIMEType.PLAIN.value] = Field(default=MIMEType.PLAIN.value, frozen=True)
    _obj: str | None = PrivateAttr(default=None)

    def _encode_obj(self, obj: str) -> bytes:
        return str.encode(obj, 'utf-8')

    def _decode_obj(self, data: bytes) -> str:
        return data.decode('utf-8')


class GLBResource(BinaryResource):
    mime_type: Literal[MIMEType.GLB.value] = Field(default=MIMEType.GLB.value, frozen=True)

    _obj: bytes | None = PrivateAttr(default=None)

    def _encode_obj(self, obj: bytes) -> bytes:
        return obj

    def _decode_obj(self, data: bytes) -> bytes:
        return data
