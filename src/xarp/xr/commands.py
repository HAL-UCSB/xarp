import base64
from typing import Any, List, Dict, ClassVar

from PIL import Image
from pydantic import BaseModel, Field

from xarp.data_models import Hands, utc_ts
from xarp.spatial import Transform, FloatArrayLike


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
                 key=None):
        super().__init__()
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
    pil_img_mode: str = 'RGBA'
    _cmd: ClassVar[str] = 'image'

    def result(self, image_dict: dict) -> Image:
        pixels = base64.b64decode(image_dict['pixels'])
        size = image_dict['width'], image_dict['height']
        img = Image.frombytes(self.pil_img_mode, size, pixels)
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


class DepthCommand(ImageCommand):
    cmd: str = 'depth'
    pil_img_mode: str = 'I;16'
    _cmd: ClassVar[str] = 'depth'


class EyeCommand(XRCommand):
    cmd: str = 'eye'
    _cmd: ClassVar[str] = 'eye'

    def __init__(self):
        super().__init__()

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
                 key: str = None):
        super().__init__()
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

    def __init__(self):
        super().__init__()

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

    def image(self) -> None:
        self.args.append(ImageCommand())

    def depth(self) -> None:
        self.args.append(DepthCommand())

    def eye(self) -> None:
        self.args.append(EyeCommand())

    def hands(self) -> None:
        self.args.append(HandsCommand())

    def result(self, results: List) -> List:
        return [command.result(result) for command, result in zip(self.args, results)]


bundle_map = {
    EyeCommand._cmd: EyeCommand,
    HandsCommand._cmd: HandsCommand,
    ImageCommand._cmd: ImageCommand,
    DepthCommand._cmd: DepthCommand
}
