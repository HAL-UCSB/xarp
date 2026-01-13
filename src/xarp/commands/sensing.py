import base64
from typing import ClassVar, Literal

from PIL import Image as PIL_Image
from pydantic import Field

from . import Command, Response
from xarp.data_models import Hands
from xarp.resources import ImageResource
from xarp.spatial import Pose


class ImageCommand(Command):
    type: Literal["image"] = Field(default="image", frozen=True)
    pil_img_mode: ClassVar[str] = "RGBA"

    @classmethod
    def validate_response_value(cls, value: Response) -> ImageResource:
        pixels = value["pixels"]
        size = value["width"], value["height"]
        pil_image = PIL_Image.frombytes(cls.pil_img_mode, size, pixels).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)
        return ImageResource.from_image(pil_image)


class VirtualImageCommand(ImageCommand):
    type: Literal["virtual_image"] = Field(default="virtual_image", frozen=True)


class DepthCommand(ImageCommand):
    type: Literal["depth"] = Field(default="depth", frozen=True)
    pil_img_mode: ClassVar[str] = "I;16"


class EyeCommand(Command):
    type: Literal["eye"] = Field(default="eye", frozen=True)

    @classmethod
    def validate_response_value(cls, value: Response) -> Pose:
        return Pose.model_validate(value)


class HeadCommand(EyeCommand):
    type: Literal["head"] = Field(default="head", frozen=True)


class HandsCommand(Command):
    type: Literal["hands"] = Field(default="hands", frozen=True)

    def validate_response_value(self, value: Response) -> Hands:
        return Hands.model_validate(value)
