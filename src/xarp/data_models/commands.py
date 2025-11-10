from typing import Any, List, Dict, ClassVar, Callable, Annotated

from pydantic import BaseModel, Field

from xarp.data_models.app import Hands, Image
from xarp.time import utc_ts
from xarp.data_models.spatial import Transform, FloatArrayLike


class XRCommand(BaseModel):
    ts: int = Field(default_factory=utc_ts)
    cmd: str
    args: List = Field(default_factory=list)
    kwargs: Dict[Any, Any] = Field(default_factory=dict)

    def result(self, data) -> Any:
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

    def result(self, data: str) -> str:
        return str(data)


class ImageCommand(XRCommand):
    cmd: str = 'image'
    pil_img_mode: ClassVar[str] = 'RGBA'
    _cmd: ClassVar[str] = 'image'

    def result(self, image_dict: dict) -> Image:
        return Image(**image_dict, pil_img_mode=ImageCommand.pil_img_mode)


class DepthCommand(ImageCommand):
    cmd: str = 'depth'
    pil_img_mode: ClassVar[str] = 'I;16'
    _cmd: ClassVar[str] = 'depth'

    def result(self, image_dict: dict) -> Image:
        return Image(**image_dict, pil_img_mode=DepthCommand.pil_img_mode)


class EyeCommand(XRCommand):
    cmd: str = 'eye'
    _cmd: ClassVar[str] = 'eye'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def result(self, data: dict) -> Transform:
        return Transform.model_validate(data)


class DisplayCommand(XRCommand):
    cmd: str = 'display'
    _cmd: ClassVar[str] = 'display'

    def __init__(self,
                 content: bytes,
                 width: int,
                 height: int,
                 depth: float,
                 opacity: float = 1.0,
                 eye: Transform = None,
                 key: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.args = [content, width, height, depth]
        if opacity:
            self.kwargs['opacity'] = opacity
        if eye:
            self.kwargs['eye'] = eye
        if key:
            self.kwargs['key'] = key


class HandsCommand(XRCommand):
    cmd: str = 'hands'
    _cmd: ClassVar[str] = 'hands'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def result(self, data: dict) -> Hands:
        return Hands.model_validate(data)


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


class XRCommandBundle(XRCommand):
    cmd: str = 'bundle'
    _cmd: ClassVar[str] = 'bundle'

    bundle_map: ClassVar[Dict[str, Callable]] = {
        EyeCommand._cmd: EyeCommand,
        HandsCommand._cmd: HandsCommand,
        ImageCommand._cmd: ImageCommand,
        DepthCommand._cmd: DepthCommand
    }

    def result(self, results: List) -> List:
        return [command.result(result) for command, result in zip(self.args, results)]
