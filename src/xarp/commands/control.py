from typing import Literal, Any, Annotated
from pydantic import Field, JsonValue

from xarp import ImageCommand, DepthCommand, EyeCommand
from xarp.commands.sensing import HeadCommand, HandsCommand
from xarp.data_models.data import DeviceInfo
from xarp.commands import XRCommand, CancelCommand
from xarp.commands.assets import AssetCommand, Element, DestroyAssetCommand, DestroyElementCommand
from xarp.commands.ui import WriteCommand, SayCommand, ReadCommand


class InfoCommand(XRCommand):
    cmd: Literal['info'] = Field(default='info', frozen=True)

    def validate_response(self, json_data: JsonValue) -> DeviceInfo:
        return DeviceInfo.model_validate(json_data)


AllowedBundleCommands = Annotated[
    AssetCommand | Element | DestroyAssetCommand | DestroyElementCommand |
    WriteCommand | SayCommand | ReadCommand |
    CancelCommand |
    ImageCommand | DepthCommand | HeadCommand | EyeCommand | HandsCommand,
    Field(discriminator='cmd')
]


class BundleCommand(XRCommand):
    cmd: Literal['bundle'] = Field(default='bundle', frozen=True)
    subcommands: list[AllowedBundleCommands] = Field(default_factory=list)

    def validate_response(self, json_data: list[JsonValue]) -> list[Any]:
        return [sub.validate_response(item) for sub, item in zip(self.subcommands, json_data)]
