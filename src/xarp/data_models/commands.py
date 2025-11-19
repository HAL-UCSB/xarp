from typing import Any, List, Dict, ClassVar, Optional

from pydantic import BaseModel, Field, JsonValue

from xarp.data_models.responses import Hands, Image, SenseResult
from xarp.data_models.spatial import Transform, FloatArrayLike
from xarp.time import utc_ts


class XRCommand(BaseModel):
    ts: int = Field(default_factory=utc_ts)
    xid: Optional[int] = None
    cmd: str
    args: List = Field(default_factory=list)
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def validate_result(cls, json_data: Dict) -> Any:
        pass

    @classmethod
    def expects_response(cls) -> bool:
        super_validate_result = XRCommand.__dict__['validate_result']
        cls_validate_result = cls.__dict__.get('validate_result')
        return cls_validate_result is not None and cls_validate_result is not super_validate_result


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


class SayCommand(WriteCommand):
    cmd: str = 'say'
    _cmd: ClassVar[str] = 'say'

    def __init__(self,
                 *text,
                 title=None,
                 key=None,
                 **kwargs):
        super().__init__(*text, title=title, key=key, **kwargs)

    @classmethod
    def validate_result(cls, json_data: Dict) -> Any:
        # wait until said
        pass


class ReadCommand(WriteCommand):
    cmd: str = 'read'
    _cmd: ClassVar[str] = 'read'

    @classmethod
    def validate_result(cls, json_data: Dict) -> str:
        return str(json_data)


class ImageCommand(XRCommand):
    cmd: str = 'image'
    pil_img_mode: ClassVar[str] = 'RGBA'
    _cmd: ClassVar[str] = 'image'

    @classmethod
    def validate_result(cls, json_data: Dict) -> Image:
        img = Image.model_validate(json_data)
        img.pil_img_mode = ImageCommand.pil_img_mode
        return img


class DepthCommand(ImageCommand):
    cmd: str = 'depth'
    pil_img_mode: ClassVar[str] = 'I;16'
    _cmd: ClassVar[str] = 'depth'

    @classmethod
    def validate_result(cls, json_data: Dict) -> Image:
        img = Image.model_validate(json_data)
        img.pil_img_mode = DepthCommand.pil_img_mode
        return img


class EyeCommand(XRCommand):
    cmd: str = 'eye'
    _cmd: ClassVar[str] = 'eye'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate_result(cls, json_data: Dict) -> Transform:
        return Transform.model_validate(json_data)


class HeadCommand(XRCommand):
    cmd: str = 'head'
    _cmd: ClassVar[str] = 'head'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate_result(cls, json_data: Dict) -> Transform:
        return Transform.model_validate(json_data)


class HandsCommand(XRCommand):
    cmd: str = 'hands'
    _cmd: ClassVar[str] = 'hands'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def validate_result(cls, json_data: Dict) -> Hands:
        return Hands.model_validate(json_data)


class SenseCommand(XRCommand):
    cmd: str = 'sense'
    _cmd: ClassVar[str] = 'sense'
    _validation_map: ClassVar[Dict] = dict(
        eye=EyeCommand,
        head=HeadCommand,
        image=ImageCommand,
        depth=DepthCommand,
        hands=HandsCommand)

    def __init__(self, **kwargs):
        super().__init__(kwargs=kwargs)

    @classmethod
    def validate_result(cls, json_data: Dict) -> SenseResult:
        json_data = {k: cls._validation_map[k].validate_result(v) for k, v in json_data.items()}
        return SenseResult.model_validate(json_data)


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


class ClearCommand(XRCommand):
    cmd: str = 'clear'
    _cmd: ClassVar[str] = 'clear'

class SaveCommand(XRCommand):
    cmd: str = 'save'
    _cmd: ClassVar[str] = 'save'

    def __init__(self, *args):
        super().__init__(args=args)


class LoadCommand(XRCommand):
    cmd: str = 'load'
    _cmd: ClassVar[str] = 'load'

    def __init__(self, *args):
        super().__init__(args=args)


class BundleCommand(XRCommand):
    cmd: str = 'bundle'
    _cmd: ClassVar[str] = 'bundle'

    def __init__(self, **kwargs):
        super().__init__(kwargs=kwargs)

class XRResult(BaseModel):
    ts: int = Field(default_factory=utc_ts)
    xid: Optional[int] = None
    value: JsonValue
