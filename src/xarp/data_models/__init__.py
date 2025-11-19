from xarp.data_models.spatial import Transform
from xarp.data_models.responses import Hands, Image
from xarp.data_models.chat import ChatMessage
from xarp.data_models.commands import ClearCommand, WriteCommand, ReadCommand, ImageCommand, HandsCommand, \
    SphereCommand, DepthCommand, EyeCommand, DisplayCommand, XRCommand

__model_classes__ = [
    Hands,
    Image,
    ChatMessage,
    XRCommand,
    ClearCommand,
    WriteCommand,
    ReadCommand,
    ImageCommand,
    DepthCommand,
    EyeCommand,
    DisplayCommand,
    HandsCommand,
    SphereCommand,
    Transform
]

mimetype_to_model_cls = {f'application/xarp/{cls.__name__.lower()}': cls for cls in __model_classes__}

model_cls_to_mimetype = {v: k for k, v in mimetype_to_model_cls.items()}
