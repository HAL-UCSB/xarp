"""Commands for storing assets and managing virtual scene elements."""

from typing import Literal, ClassVar

from pydantic import Field
from pydantic import model_validator
from xarp.commands import Command
from xarp.entities import Element, Asset


class CreateOrUpdateAssetsCommand(Command):
    """Store one or more encoded assets on the XR client.

    Every asset must expose a non-empty asset key, MIME type, and decoded object
    payload. Invalid assets cause Pydantic model validation to fail.

    Attributes:
        cmd: Wire discriminator, always ``"save"``.
        assets: Assets to create or replace.
        alt_path: Optional client-specific alternative storage path.
    """

    cmd: Literal["save"] = Field(default="save", frozen=True)
    assets: list[Asset]
    alt_path: str | None = None
    _required_attrs: ClassVar = frozenset(("asset_key", "mime_type", "obj"))

    @model_validator(mode="after")
    def _validate_assets(self):
        assets = self.assets if isinstance(self.assets, list) else [self.assets]
        for asset in assets:
            for attr in CreateOrUpdateAssetsCommand._required_attrs:
                if not getattr(asset, attr, None):
                    raise ValueError(f'"{self.cmd}" requires all assets to have a non-empty {attr}.')
        return self


class CreateOrUpdateElementCommand(Command):
    """Create or replace virtual elements by key.

    Each element must have a non-empty key. Invalid elements cause Pydantic
    model validation to fail.

    Attributes:
        cmd: Wire discriminator, always ``"update"``.
        elements: Desired element states to upsert.
    """

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
    """Request the keys of all assets stored on the XR client.

    Attributes:
        cmd: Wire discriminator, always ``"list_assets"``.
    """

    cmd: Literal["list_assets"] = Field(default="list_assets", frozen=True)

    def validate_response_value(self, value: list) -> list[str]:
        """Convert every returned asset key to a string.

        Args:
            value: Sequence of asset keys returned by the XR client.

        Returns:
            Asset keys converted with :class:`str`.
        """
        return [str(i) for i in value]


class DestroyAssetCommand(Command):
    """Delete selected assets or all stored assets.

    Exactly one deletion target must be selected: provide a non-empty ``keys``
    list or set ``all_assets=True``. Empty keys and conflicting targets cause
    Pydantic model validation to fail.

    Attributes:
        cmd: Wire discriminator, always ``"destroy_asset"``.
        keys: Asset keys to delete, or ``None`` when deleting all assets.
        all_assets: Whether to delete every stored asset.
    """

    cmd: Literal["destroy_asset"] = Field(default="destroy_asset", frozen=True)
    keys: list[str] | None = None
    all_assets: bool = False

    @model_validator(mode="after")
    def validate_asset_key_all(self):
        """Validate that exactly one non-empty asset deletion target is set.

        Returns:
            The validated command instance.

        Raises:
            ValueError: If the target selection is empty, conflicting, or
                contains an empty key.
        """
        if self.all_assets:
            if self.keys is not None:
                raise ValueError('When "all_assets" is True, "asset_key" must not be provided.')
            return self
        if not self.keys:
            raise ValueError('When "all_assets" is False, "asset_key" must be a non-empty string or list of strings.')
        if isinstance(self.keys, list) and any(not k for k in self.keys):
            raise ValueError('"asset_key" list must not contain empty strings.')
        return self


class ListElementsCommand(Command):
    """Request the keys of all active and inactive scene elements.

    Attributes:
        cmd: Wire discriminator, always ``"list_elements"``.
    """

    cmd: Literal["list_elements"] = Field(default="list_elements", frozen=True)

    def validate_response_value(self, value: list) -> list[str]:
        """Convert every returned element key to a string.

        Args:
            value: Sequence of element keys returned by the XR client.

        Returns:
            Element keys converted with :class:`str`.
        """
        return [str(i) for i in value]


class DestroyElementCommand(Command):
    """Destroy selected scene elements or all scene elements.

    Exactly one deletion target must be selected: provide a non-empty ``keys``
    list or set ``all_elements=True``. Empty keys and conflicting targets cause
    Pydantic model validation to fail.

    Attributes:
        cmd: Wire discriminator, always ``"destroy_element"``.
        keys: Element keys to destroy, or ``None`` when destroying all elements.
        all_elements: Whether to destroy every scene element.
    """

    cmd: Literal["destroy_element"] = Field(default="destroy_element", frozen=True)
    keys: list[str] | None = None
    all_elements: bool = False

    @model_validator(mode="after")
    def validate_key_all(self):
        """Validate that exactly one non-empty element deletion target is set.

        Returns:
            The validated command instance.

        Raises:
            ValueError: If the target selection is empty, conflicting, or
                contains an empty key.
        """
        if self.all_elements:
            if self.keys is not None:
                raise ValueError("When \"all_elements\" is True, \"keys\" must not be provided.")
            return self
        if not self.keys:
            raise ValueError("When \"all_elements\" is False, \"keys\" must be a non-empty string or list of strings.")
        if isinstance(self.keys, list) and any(not k for k in self.keys):
            raise ValueError('"keys" list must not contain empty strings.')
        return self
