import base64
import mimetypes
from enum import Enum
from io import BytesIO
from typing import Literal, ClassVar, TypeVar, Generic, Self

import trimesh
from PIL import Image
from pydantic import BaseModel, model_validator
from pydantic import ConfigDict, PrivateAttr, model_serializer, Field
from trimesh import Trimesh

from xarp.spatial import Transform, Pose

T = TypeVar("T")


class MIMEType(str, Enum):
    TXT = mimetypes.types_map[".txt"]
    PNG = mimetypes.types_map[".png"]
    JPEG = mimetypes.types_map[".jpg"]
    MP3 = mimetypes.types_map[".mp3"]
    WAV = mimetypes.types_map[".wav"]
    OGG = "audio/ogg"
    MP4 = mimetypes.types_map[".mp4"]
    GLB = "model/gltf-binary"
    XARP_DEFAULT = "application/vnd.xarp.default"

    @staticmethod
    def from_extension(ext: str) -> "MIMEType":
        ext = ext if ext.startswith(".") else "." + ext
        fallback = {
            ".ogg": MIMEType.OGG.value,
            ".glb": MIMEType.GLB.value,
        }
        mime = mimetypes.types_map.get(ext) or fallback.get(ext)
        return MIMEType(mime)


class Asset(BaseModel, Generic[T]):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        strict=True,
        use_enum_values=True,
        validate_assignment=True
    )

    asset_key: str | None = None
    mime_type: MIMEType = None
    raw: bytes | None = None

    _obj: T | None = PrivateAttr(default=None)

    @property
    def obj(self) -> T:
        if self._obj is None:
            if self.raw is None:
                raise RuntimeError("Either raw or obj must be provided")
            self._obj = self._raw_to_obj(self.raw)
        return self._obj

    def _obj_to_raw(self, obj: T) -> bytes:
        return obj

    def _raw_to_obj(self, raw: bytes) -> T:
        return raw

    @model_serializer(mode="plain")
    def serialize(self):
        raw_value = None
        if self.raw is not None:
            raw_value = self.raw
        elif self._obj is not None:
            raw_value = self._obj_to_raw(self._obj)

        return dict(
            asset_key=self.asset_key,
            mime_type=self.mime_type,
            raw=raw_value
        )

    @classmethod
    def from_obj(cls, obj: T, mime_type: MIMEType = None, asset_key: str = None) -> Self:
        if mime_type is None:
            self = cls(asset_key=asset_key)
        else:
            self = cls(asset_key=asset_key, mime_type=mime_type)
        self._obj = obj
        return self


class DefaultAssets:
    SPHERE = Asset(mime_type=MIMEType.XARP_DEFAULT, raw=b"Sphere")
    CUBE = Asset(mime_type=MIMEType.XARP_DEFAULT, raw=b"Cube")
    CAPSULE = Asset(mime_type=MIMEType.XARP_DEFAULT, raw=b"Capsule")
    CYLINDER = Asset(mime_type=MIMEType.XARP_DEFAULT, raw=b"Cylinder")
    PLANE = Asset(mime_type=MIMEType.XARP_DEFAULT, raw=b"Plane")
    QUAD = Asset(mime_type=MIMEType.XARP_DEFAULT, raw=b"Quad")


class ImageAsset(Asset[Image.Image]):
    mime_type: Literal[MIMEType.PNG, MIMEType.JPEG] = Field(default=MIMEType.PNG, frozen=True)

    def to_base64(self) -> str:
        raw = self._obj_to_raw()
        return base64.b64encode(raw).decode()

    def _obj_to_raw(self, obj: Image.Image) -> bytes:
        buf = BytesIO()
        data_format = self.mime_type.split("/")[-1]
        obj.save(buf, format=data_format)
        return buf.getvalue()

    def _raw_to_obj(self, raw: bytes) -> Image.Image:
        buffer = BytesIO(raw)
        return Image.open(buffer)


class TextAsset(Asset[str]):
    mime_type: Literal[MIMEType.TXT] = Field(default=MIMEType.TXT, frozen=True)

    _encoding: ClassVar[Literal["utf-8"]] = "utf-8"

    def _obj_to_raw(self, obj: str) -> bytes:
        return str.encode(obj, self._encoding)

    def _raw_to_obj(self, raw: bytes) -> str:
        return raw.decode(self._encoding)

    @classmethod
    def from_obj(cls, obj: T, mime_type: MIMEType = MIMEType.TXT, asset_key: str = None) -> Self:
        return super().from_obj(obj, mime_type, asset_key)


class GLBAsset(Asset[Trimesh]):
    mime_type: Literal[MIMEType.GLB] = Field(default=MIMEType.GLB, frozen=True)

    _file_type: ClassVar[Literal["glb"]] = "glb"

    def _obj_to_raw(self, obj: Trimesh) -> bytes:
        return obj.export(file_type='glb')

    def _raw_to_obj(self, raw: bytes) -> Trimesh:
        buffer = BytesIO(raw)
        return trimesh.load(buffer, file_type=self._file_type)


class Element(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

    key: str = ""
    active: bool = True
    transform: Transform = Transform()
    eye: Pose | None = None
    distance: float | None = None
    color: tuple[float, float, float, float] | None = None
    asset: Asset | None = None

    @model_validator(mode="after")
    def validate_asset_key_data(self):
        if self.eye is None and self.transform is None:
            self.transform = Transform()
        return self
