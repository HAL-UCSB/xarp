import base64
import pathlib
from typing import Tuple, Optional

import PIL.Image as PIL_Image
from pydantic import BaseModel, Field, field_serializer, field_validator

from xarp.data_models.spatial import Transform


class Hands(BaseModel):
    left: Tuple[Transform, ...] = Field(default_factory=tuple)
    right: Tuple[Transform, ...] = Field(default_factory=tuple)


class Image(BaseModel):
    pixels: Optional[bytes] = None
    width: int
    height: int
    pil_img_mode: str = 'RGBA'
    path: Optional[pathlib.PurePath] = None

    @field_validator("pixels", mode="before")
    def decode_if_base64(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return base64.b64decode(v)
            except Exception:
                raise ValueError("Invalid base64 encoding")
        return v

    @field_serializer('pixels')
    def encode_to_base64(self, v, _info):
        if v is None:
            return None
        return base64.b64encode(v)

    def load_pixels(self, scale: float = None) -> 'Image':
        img = PIL_Image.open(self.path).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)
        self.width, self.height = img.size
        if scale is not None:
            img.thumbnail((self.width * scale, self.height * scale))
        self.pixels = img.tobytes()
        return self

    def dump_pixels(self, path: pathlib.PurePath) -> 'Image':
        pil_img = self.as_pil_image()
        with open(path, 'wb') as f:
            pil_img.save(path)
        self.pixels = None
        self.path = path
        return self

    def as_pil_image(self) -> PIL_Image.Image:
        if self.path:
            return PIL_Image.open(self.path)
        return PIL_Image.frombytes(
            self.pil_img_mode,
            (self.width, self.height),
            self.pixels).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)


class SenseResult(BaseModel):
    eye: Optional[Transform] = None
    head: Optional[Transform] = None
    image: Optional[Image] = None
    depth: Optional[Image] = None
    hands: Optional[Hands] = None
