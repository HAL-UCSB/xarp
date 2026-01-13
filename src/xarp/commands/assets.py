from enum import Enum
from typing import Literal, Any

from pydantic import Field, model_validator

from . import Command,Response
from xarp.resources import BinaryResource
from xarp.spatial import Transform, Pose


class DefaultAssets(str, Enum):
    SPHERE = "Sphere"
    CUBE = "Cube"


class AssetCommand(Command):
    type: Literal["asset"] = Field(default="asset", frozen=True)
    asset_key: str
    data: BinaryResource


class ListAssetsCommand(Command):
    type: Literal["list_assets"] = Field(default="list_assets", frozen=True)

    def validate_response_value(self, value: Any) -> list[str]:
        return [str(i) for i in value]


class DestroyAssetCommand(Command):
    type: Literal["destroy_asset"] = Field(default="destroy_asset", frozen=True)
    asset_key: list[str] | str | None = None
    all_assets: bool = False

    @model_validator(mode="after")
    def validate_asset_key_all(self):
        if self.all_assets:
            if self.asset_key is not None:
                raise ValueError("When \"all_assets\" is True, \"asset_key\" must not be provided.")
            return self
        if not self.asset_key:
            raise ValueError("When \"all_assets\" is False, \"asset_key\" must be a non-empty string or list.")
        return self


class Element(Command):
    type: Literal["element"] = Field(default="element", frozen=True)
    key: str

    active: bool | None = True
    transform: Transform  = None
    eye: Pose | None = None
    distance: float | None = None
    color: tuple[float, float, float, float] | None = None

    asset_key: str | None = None
    binary: BinaryResource | None = None

    @model_validator(mode="after")
    def validate_asset_key_data(self):
        if self.asset_key is not None and self.binary is not None:
            raise ValueError("When \"asset_key\" is provided, \"data\" will be ignored by the client.")
        if self.eye is None and self.transform is None:
            self.transform = Transform()
        return self


class DestroyElementCommand(Command):
    type: Literal["destroy_element"] = Field(default="destroy_element", frozen=True)
    key: str | None = None
    all_elements: bool = False

    @model_validator(mode="after")
    def validate_key_all(self):
        if self.all_elements:
            if self.key is not None:
                raise ValueError("When \"all_elements\" is True, \"key\" must not be provided.")
            return self
        if not self.key:
            raise ValueError("When \"all_elements\" is False, \"key\" must be a non-empty string.")
        return self
