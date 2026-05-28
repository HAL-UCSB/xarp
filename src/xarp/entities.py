import base64
import mimetypes
from io import BytesIO
from typing import Any, Literal, ClassVar, TypeVar, Generic, Self

import trimesh
from PIL import Image
from pydantic import BaseModel, SkipValidation
from pydantic import ConfigDict, model_serializer, Field
from trimesh import Trimesh

from xarp.spatial import Transform

T = TypeVar("T")


class MIMEType:
    TXT = "text/plain"
    PNG = "image/png"
    JPEG = "image/jpeg"
    MP3 = "audio/mpeg"
    WAV = "audio/wav"
    OGG = "audio/ogg"
    MP4 = "video/mp4"
    GLB = "model/gltf-binary"
    XARP_DEFAULT = "application/vnd.xarp.default"

    @staticmethod
    def from_extension(ext: str) -> str:
        ext = ext if ext.startswith(".") else "." + ext
        fallback = {
            ".ogg": MIMEType.OGG,
            ".glb": MIMEType.GLB,
        }
        mime = mimetypes.types_map.get(ext) or fallback.get(ext)
        if mime is None:
            raise ValueError(f"No MIME type found for extension: {ext}")
        return mime


class Asset(BaseModel, Generic[T]):
    """
    Binary data container that can be attached to an Element for display in the virtual space.

    `raw` is always the source of truth. `obj` is a convenience deserialisation
    into an ergonomic Python type; subclasses define the conversion via
    `_obj_to_raw` and `_raw_to_obj`. Use `from_obj` to construct from a Python
    object — it encodes to `raw` immediately so serialisation is always safe.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )

    asset_key: str | None = None
    mime_type: str | None = None
    raw: bytes | None = None

    @property
    def obj(self) -> T:
        """Deserialize `raw` to the ergonomic Python type. Not cached."""
        if self.raw is None:
            raise RuntimeError("Asset has no raw data to deserialize")
        return self._raw_to_obj(self.raw)

    def _obj_to_raw(self, obj: T) -> bytes:
        return obj

    def _raw_to_obj(self, raw: bytes) -> T:
        return raw

    @model_serializer(mode="plain")
    def serialize(self):
        return dict(
            asset_key=self.asset_key,
            mime_type=self.mime_type,
            raw=self.raw,
        )

    @classmethod
    def from_obj(cls, obj: T, mime_type: str | None = None, asset_key: str | None = None) -> Self:
        """Construct from a Python object, encoding to raw immediately."""
        # Instantiate minimally to get access to _obj_to_raw on the correct subclass,
        # then reconstruct with raw populated so serialisation is always safe.
        if mime_type is None:
            asset = cls(asset_key=asset_key)
        else:
            asset = cls(asset_key=asset_key, mime_type=mime_type)
        asset.raw = asset._obj_to_raw(obj)
        return asset


class DefaultAssets:
    @staticmethod
    def sphere() -> Asset:
        return Asset(asset_key="Sphere", mime_type=MIMEType.XARP_DEFAULT, raw=b"Sphere")

    @staticmethod
    def cube() -> Asset:
        return Asset(asset_key="Cube", mime_type=MIMEType.XARP_DEFAULT, raw=b"Cube")

    @staticmethod
    def capsule() -> Asset:
        return Asset(asset_key="Capsule", mime_type=MIMEType.XARP_DEFAULT, raw=b"Capsule")

    @staticmethod
    def cylinder() -> Asset:
        return Asset(asset_key="Cylinder", mime_type=MIMEType.XARP_DEFAULT, raw=b"Cylinder")

    @staticmethod
    def plane() -> Asset:
        return Asset(asset_key="Plane", mime_type=MIMEType.XARP_DEFAULT, raw=b"Plane")

    @staticmethod
    def quad() -> Asset:
        return Asset(asset_key="Quad", mime_type=MIMEType.XARP_DEFAULT, raw=b"Quad")

    @staticmethod
    def axes() -> Asset:
        return Asset(asset_key="Axes", mime_type=MIMEType.XARP_DEFAULT, raw=b"Axis")


class ImageAsset(Asset[Image.Image]):
    mime_type: Literal["image/png", "image/jpeg"] = Field(default=MIMEType.PNG, frozen=True)

    def to_base64(self) -> str:
        return base64.b64encode(self.raw).decode()

    def _obj_to_raw(self, obj: Image.Image) -> bytes:
        buf = BytesIO()
        obj.save(buf, format=self.mime_type.split("/")[-1])
        return buf.getvalue()

    def _raw_to_obj(self, raw: bytes) -> Image.Image:
        return Image.open(BytesIO(raw))


class TextAsset(Asset[str]):
    mime_type: Literal["text/plain"] = Field(default=MIMEType.TXT, frozen=True)

    _encoding: ClassVar[Literal["utf-8"]] = "utf-8"

    def _obj_to_raw(self, obj: str) -> bytes:
        return obj.encode(self._encoding)

    def _raw_to_obj(self, raw: bytes) -> str:
        return raw.decode(self._encoding)

    @classmethod
    def from_obj(cls, obj: str, mime_type: str = MIMEType.TXT, asset_key: str | None = None) -> Self:
        return super().from_obj(obj, mime_type, asset_key)


class GLBAsset(Asset[Trimesh]):
    mime_type: Literal["model/gltf-binary"] = Field(default=MIMEType.GLB, frozen=True)

    _file_type: ClassVar[Literal["glb"]] = "glb"

    def _obj_to_raw(self, obj: Trimesh) -> bytes:
        return obj.export(file_type="glb")

    def _raw_to_obj(self, raw: bytes) -> Trimesh:
        result = trimesh.load(BytesIO(raw), file_type=self._file_type)
        if isinstance(result, trimesh.Scene):
            result = result.dump(concatenate=True)
        return result


class Element(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    key: str = ""
    active: bool = True
    transform: Transform = Field(default_factory=Transform)
    color: tuple[float, float, float, float] | None = None  # RGBA, components in [0, 1]
    asset: SkipValidation[Asset[Any]] | None = None
    parent: str | None = None

    play: bool | None = None
    time: float | None = None

    # @model_validator(mode="after")
    # def validate_asset_not_empty(self) -> "Element":
    #     if self.asset is not None and self.asset.raw is None:
    #         raise ValueError(
    #             "An attached asset must have raw data. "
    #             "Use Asset.from_obj() to construct from a Python object."
    #         )
    #     return self
