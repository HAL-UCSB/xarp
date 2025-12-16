import base64
from typing import ClassVar, Literal

from PIL import Image as PIL_Image
from pydantic import Field, JsonValue

from xarp.commands import XRCommand
from xarp.data_models.binaries import ImageResource
from xarp.data_models.data import Hands
from xarp.data_models.spatial import Pose


class ImageCommand(XRCommand):
    cmd: Literal['image'] = Field(default='image', frozen=True)
    pil_img_mode: ClassVar[str] = 'RGBA'

    @classmethod
    def validate_response(cls, json_data: JsonValue) -> ImageResource:
        pixels = base64.b64decode(json_data['pixels'])
        size = json_data['width'], json_data['height']
        pil_image = PIL_Image.frombytes(cls.pil_img_mode, size, pixels).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)
        return ImageResource.from_image(pil_image)


class DepthCommand(ImageCommand):
    cmd: Literal['depth'] = Field(default='depth', frozen=True)
    pil_img_mode: ClassVar[str] = 'I;16'


class EyeCommand(XRCommand):
    cmd: Literal['eye'] = Field(default='eye', frozen=True)

    @classmethod
    def validate_response(cls, json_data: JsonValue) -> Pose:
        return Pose.model_validate(json_data)


class HeadCommand(XRCommand):
    cmd: Literal['head'] = Field(default='head', frozen=True)

    @classmethod
    def validate_response(cls, json_data: JsonValue) -> Pose:
        return Pose.model_validate(json_data)


class HandsCommand(XRCommand):
    cmd: Literal['hands'] = Field(default='hands', frozen=True)

    def validate_response(self, json_data: JsonValue) -> Hands:
        return Hands.model_validate(json_data)
