from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from xarp.commands import XRCommand
from xarp.data_models.binaries import BinaryResource
from xarp.data_models.data import MIMEType
from xarp.data_models.encoding import LazyBase64Bytes
from xarp.data_models.spatial import Transform, Pose


class DefaultAssets(str, Enum):
    SPHERE = 'Sphere'
    CUBE = 'Cube'


class AssetCommand(XRCommand):
    cmd: Literal['asset'] = Field('asset', frozen=True)
    key: str
    data: LazyBase64Bytes
    mime_type: MIMEType


class DestroyAssetCommand(XRCommand):
    cmd: Literal['destroy_asset'] = Field('destroy_asset', frozen=True)
    key: str | None = None
    all: bool = False

    @model_validator(mode='after')
    def validate_key_all_logic(self):
        if self.all:
            if self.key is not None:
                raise ValueError('When "all" is True, "asset_key" must not be provided.')
            return self
        if not self.key:
            raise ValueError('When "all" is False, "asset_key" must be a non-empty string.')
        return self


class ElementCommand(XRCommand):
    cmd: Literal['element'] = Field('element', frozen=True)
    key: str

    active: bool | None = True
    transform: Transform | None = None
    eye: Pose | None = None
    distance: float | None = None
    color: tuple[float, float, float, float] | None = None

    asset_key: str | None = None
    binary: BinaryResource | None = None

    @model_validator(mode='after')
    def validate_key_all_logic(self):
        if self.asset_key is not None and self.binary is not None:
            raise ValueError('When "asset_key" is provided, "data" will be ignored by the client.')
        return self


class DestroyElementCommand(XRCommand):
    cmd: Literal['destroy_element'] = Field('destroy_element', frozen=True)
    key: str | None = None
    all: bool = False

    @model_validator(mode='after')
    def validate_key_all_logic(self):
        if self.all:
            if self.key is not None:
                raise ValueError('When "all" is True, "key" must not be provided.')
            return self
        if not self.key:
            raise ValueError('When "all" is False, "key" must be a non-empty string.')
        return self
