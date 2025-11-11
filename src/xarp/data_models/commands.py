import base64
from typing import Any, List, Dict, ClassVar, Callable, Annotated, Union

from pydantic import BaseModel, Field, TypeAdapter

from xarp.data_models.app import Hands, Image
from xarp.time import utc_ts
from xarp.data_models.spatial import Transform, FloatArrayLike


class XRCommand(BaseModel):
    ts: int = Field(default_factory=utc_ts)
    cmd: str
    args: List = Field(default_factory=list)
    kwargs: Dict[Any, Any] = Field(default_factory=dict)

    def result(self, json_string: str) -> Any:
        pass


class ClearCommand(XRCommand):
    cmd: str = 'clear'
    _cmd: ClassVar[str] = 'clear'


class WriteCommand(XRCommand):
    cmd: str = 'write'
    _cmd: ClassVar[str] = 'write'

    def __init__(self,
                 *text,
                 title=None,
                 key=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.args = list(text)
        if title:
            self.kwargs['title'] = title or ''
        if key:
            self.kwargs['key'] = key


class ReadCommand(WriteCommand):
    cmd: str = 'read'
    _cmd: ClassVar[str] = 'read'

    def result(self, json_string: str) -> str:
        return str(json_string)


class ImageCommand(XRCommand):
    cmd: str = 'image'
    pil_img_mode: ClassVar[str] = 'RGBA'
    _cmd: ClassVar[str] = 'image'

    def result(self, json_string: str) -> Image:
        img = Image.model_validate_json(json_string)
        img.pil_img_mode = ImageCommand.pil_img_mode
        return img


class DepthCommand(ImageCommand):
    cmd: str = 'depth'
    pil_img_mode: ClassVar[str] = 'I;16'
    _cmd: ClassVar[str] = 'depth'

    def result(self, json_string: str) -> Image:
        img = Image.model_validate_json(json_string)
        img.pil_img_mode = DepthCommand.pil_img_mode
        return img


class EyeCommand(XRCommand):
    cmd: str = 'eye'
    _cmd: ClassVar[str] = 'eye'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def result(self, json_string: str) -> Transform:
        return Transform.model_validate_json(json_string)


class DisplayCommand(XRCommand):
    cmd: str = 'display'
    _cmd: ClassVar[str] = 'display'

    def __init__(self,
                 image: Image = None,
                 depth: float = None,
                 opacity: float = 1.0,
                 eye=None,
                 visible=True,
                 key=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.kwargs.update({
            k: v for k, v in dict(
                image=image,
                depth=depth,
                opacity=opacity,
                eye=eye,
                visible=visible,
                key=key
            ).items() if v is not None})


class HandsCommand(XRCommand):
    cmd: str = 'hands'
    _cmd: ClassVar[str] = 'hands'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def result(self, json_string: str) -> Hands:
        return Hands.model_validate_json(json_string)


class SphereCommand(XRCommand):
    cmd: str = 'sphere'
    _cmd: ClassVar[str] = 'sphere'

    def __init__(self,
                 position: FloatArrayLike,
                 scale: float = .1,
                 color: FloatArrayLike = (1, 1, 1, 1),
                 key=None):
        super().__init__(args=position)
        if scale:
            self.kwargs['scale'] = scale
        if color:
            self.kwargs['color'] = color
        if key:
            self.kwargs['key'] = key


Bundlable = TypeAdapter(List[Union[
    Transform,
    Hands,
    Image
]])


class XRCommandBundle(XRCommand):
    cmd: str = 'bundle'
    _cmd: ClassVar[str] = 'bundle'

    bundle_map: ClassVar[Dict[str, Callable]] = {
        EyeCommand._cmd: EyeCommand,
        HandsCommand._cmd: HandsCommand,
        ImageCommand._cmd: ImageCommand,
        DepthCommand._cmd: DepthCommand
    }

    def result(self, json_string: str) -> Bundlable:
        return Bundlable.validate_json(json_string)
