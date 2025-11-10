import base64
import pathlib
from typing import Tuple, Optional

import PIL.Image as PIL_Image
from pydantic import BaseModel, Field

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

    def dump_to_image_file(self, path: pathlib.PurePath) -> None:
        pil_img = self.as_pil_image()
        with open(path, 'wb') as f:
            pil_img.save(path)
        self.pixels = None
        self.path = path

    def as_pil_image(self) -> PIL_Image.Image:
        if self.path:
            with open(self.path, 'rb') as f:
                return PIL_Image.open(f)

        size = (self.width, self.height)
        pixels = base64.b64decode(self.pixels)
        pil_img = PIL_Image.frombytes(
            self.pil_img_mode,
            size,
            pixels)
        return pil_img.transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)
