from typing import ClassVar, Literal

from PIL import Image as PIL_Image
from pydantic import Field

from xarp.data_models import Hands
from xarp.entities import ImageAsset
from xarp.spatial import Pose
from . import Command, Response


class ImageCommand(Command):
    cmd: Literal["image"] = Field(default="image", frozen=True)
    pil_img_mode: ClassVar[str] = "RGBA"

    @classmethod
    def validate_response_value(cls, value: Response) -> ImageAsset:
        pixels = value["pixels"]
        size = value["width"], value["height"]
        pil_image = PIL_Image.frombytes(cls.pil_img_mode, size, pixels).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)
        return ImageAsset.from_obj(pil_image)


class VirtualImageCommand(ImageCommand):
    cmd: Literal["virtual_image"] = Field(default="virtual_image", frozen=True)


class DepthCommand(ImageCommand):
    cmd: Literal["depth"] = Field(default="depth", frozen=True)
    pil_img_mode: ClassVar[str] = "I;16"


class EyeCommand(Command):
    cmd: Literal["eye"] = Field(default="eye", frozen=True)

    @classmethod
    def validate_response_value(cls, value: Response) -> Pose:
        return Pose.model_validate(value)


class HeadCommand(EyeCommand):
    cmd: Literal["head"] = Field(default="head", frozen=True)


class HandsCommand(Command):
    cmd: Literal["hands"] = Field(default="hands", frozen=True)

    def validate_response_value(self, value: Response) -> Hands:
        return Hands.model_validate(value)
