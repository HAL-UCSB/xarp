import base64
from typing import ClassVar, Literal

from PIL import Image as PIL_Image
from pydantic import Field

from xarp.commands import XRCommand, ResponseMode
from xarp.data_models.binaries import ImageResource
from xarp.data_models.data import Hands, SenseResult
from xarp.data_models.spatial import Transform, Pose


class ImageCommand(XRCommand):
    cmd: Literal['image'] = Field('image', frozen=True)
    pil_img_mode: ClassVar[str] = 'RGBA'

    @classmethod
    def validate_response(cls, json_data: dict) -> ImageResource:
        pixels = base64.b64decode(json_data['pixels'])
        size = json_data['width'], json_data['height']
        pil_image = PIL_Image.frombytes(cls.pil_img_mode, size, pixels).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)
        return ImageResource.from_image(pil_image)


class DepthCommand(ImageCommand):
    cmd: Literal['depth'] = Field('depth', frozen=True)
    pil_img_mode: ClassVar[str] = 'I;16'


class EyeCommand(XRCommand):
    cmd: Literal['eye'] = Field('eye', frozen=True)

    @classmethod
    def validate_response(cls, json_data: dict) -> Pose:
        return Pose.model_validate(json_data)


class HeadCommand(XRCommand):
    cmd: Literal['head'] = Field('head', frozen=True)

    @classmethod
    def validate_response(cls, json_data: dict) -> Transform:
        return Transform.model_validate(json_data)


class HandsCommand(XRCommand):
    cmd: Literal['hands'] = Field('hands', frozen=True)

    def validate_response(self, json_data: dict) -> Hands:
        return Hands.model_validate(json_data)


class SenseCommand(XRCommand):
    cmd: Literal['sense'] = Field('sense', frozen=True)
    response_mode: ResponseMode = ResponseMode.STREAM

    _validation_map: ClassVar[dict] = dict(
        eye=EyeCommand,
        head=HeadCommand,
        image=ImageCommand,
        depth=DepthCommand,
        hands=HandsCommand)

    def validate_response(self, json_data: dict) -> SenseResult:
        json_data = {k: SenseCommand._validation_map[k].validate_response(v) for k, v in json_data.items()}
        return SenseResult.model_validate(json_data)
