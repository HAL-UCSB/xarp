from typing import Literal, ClassVar

from pydantic import Field
from pydantic import model_validator

from xarp.commands import Command
from xarp.entities import Element, Asset


class CreateOrUpdateAssetsCommand(Command):
    cmd: Literal["save"] = Field(default="save", frozen=True)
    assets: list[Asset]
    _required_attrs: ClassVar = frozenset(("asset_key", "mime_type", "raw"))

    @model_validator(mode="after")
    def _validate_assets(self):
        assets = self.assets if isinstance(self.assets, list) else [self.assets]
        for asset in assets:
            for attr in CreateOrUpdateAssetsCommand._required_attrs:
                if not getattr(asset, attr, None):
                    raise ValueError(f'"{self.cmd}" requires all assets to have a non-empty {attr}.')
        return self


class CreateOrUpdateElementCommand(Command):
    cmd: Literal["update"] = Field(default="update", frozen=True)
    elements: list[Element]
    _required_attrs: ClassVar = frozenset(("key",))

    @model_validator(mode="after")
    def _validate_elements(self):
        elements = self.elements if isinstance(self.elements, list) else [self.elements]
        for element in elements:
            for attr in CreateOrUpdateElementCommand._required_attrs:
                if not getattr(element, attr, None):
                    raise ValueError(f'"{self.cmd}" requires all elements to have a non-empty {attr}.')
        return self


class ListAssetsCommand(Command):
    cmd: Literal["list_assets"] = Field(default="list_assets", frozen=True)

    def validate_response_value(self, value: list) -> list[str]:
        return [str(i) for i in value]


class DestroyAssetCommand(Command):
    cmd: Literal["destroy_asset"] = Field(default="destroy_asset", frozen=True)
    asset_key: list[str] | str | None = None
    all_assets: bool = False

    @model_validator(mode="after")
    def validate_asset_key_all(self):
        if self.all_assets:
            if self.asset_key is not None:
                raise ValueError('When "all_assets" is True, "asset_key" must not be provided.')
            return self
        if not self.asset_key:
            raise ValueError('When "all_assets" is False, "asset_key" must be a non-empty string or list.')
        if isinstance(self.asset_key, list) and any(not k for k in self.asset_key):
            raise ValueError('"asset_key" list must not contain empty strings.')
        return self


class ListElementsCommand(Command):
    cmd: Literal["list_elements"] = Field(default="list_elements", frozen=True)

    def validate_response_value(self, value: list) -> list[str]:
        return [str(i) for i in value]


class DestroyElementCommand(Command):
    cmd: Literal["destroy_element"] = Field(default="destroy_element", frozen=True)
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
