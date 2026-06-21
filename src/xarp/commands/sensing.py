"""Commands for capturing images and tracked spatial poses from an XR client."""

import io
from typing import ClassVar, Literal

import PIL.Image
from PIL import Image as PIL_Image
from pydantic import Field

from xarp.data_models import Hands
from xarp.entities import ImageAsset
from xarp.spatial import Pose
from . import Command, Response


class ImageCommand(Command):
    """Capture one encoded RGB image of the physical environment.

    Attributes:
        cmd: Wire discriminator, always ``"image"``.
    """

    cmd: Literal["image"] = Field(default="image", frozen=True)

    @classmethod
    def validate_response_value(cls, value: Response) -> ImageAsset:
        """Decode image metadata and cache a Pillow representation.

        Args:
            value: Mapping containing encoded ``raw`` bytes and a ``mime_type``.

        Returns:
            Image asset preserving the encoded bytes and cached Pillow image.

        Raises:
            KeyError: If a required response field is missing.
            PIL.UnidentifiedImageError: If the encoded image cannot be opened.
        """
        raw = value["raw"]
        pil_image = PIL_Image.open(io.BytesIO(raw))
        asset = ImageAsset(
            mime_type=value["mime_type"],
            asset_key=None,
            raw=raw)
        asset._obj = pil_image
        return asset


class VirtualImageCommand(ImageCommand):
    """Capture one encoded RGBA render of the virtual environment.

    Response decoding is inherited from :class:`ImageCommand`.

    Attributes:
        cmd: Wire discriminator, always ``"virtual_image"``.
    """

    cmd: Literal["virtual_image"] = Field(default="virtual_image", frozen=True)


class DepthCommand(ImageCommand):
    """Capture one unsigned 16-bit depth frame of the physical environment.

    Attributes:
        cmd: Wire discriminator, always ``"depth"``.
        pil_img_mode: Pillow mode used to interpret the raw depth buffer.
    """

    cmd: Literal["depth"] = Field(default="depth", frozen=True)
    pil_img_mode: ClassVar[str] = "I;16"

    def validate_response_value(cls, value: Response) -> ImageAsset:
        """Decode a raw depth buffer using its reported dimensions.

        Args:
            value: Mapping containing ``raw`` bytes, ``width``, ``height``, and
                ``mime_type``.

        Returns:
            Image asset preserving the raw buffer and caching a Pillow ``I;16``
            image with the reported dimensions.

        Raises:
            KeyError: If a required response field is missing.
            ValueError: If the byte count does not match the reported dimensions.
        """
        raw = value["raw"]
        size = value["width"], value["height"]
        pil_image = PIL_Image.frombytes(cls.pil_img_mode, size, raw)
        asset = ImageAsset(
            mime_type=value["mime_type"],
            asset_key=None,
            raw=raw)
        asset._obj = pil_image
        return asset


class EyeCommand(Command):
    """Request the main camera pose, typically representing the user's eyes.

    Attributes:
        cmd: Wire discriminator, always ``"eye"``.
    """

    cmd: Literal["eye"] = Field(default="eye", frozen=True)

    @classmethod
    def validate_response_value(cls, value: Response) -> Pose:
        """Validate the response payload as a spatial pose.

        Args:
            value: Deserialized pose payload.

        Returns:
            Validated camera pose.

        Raises:
            pydantic.ValidationError: If the payload is not a valid pose.
        """
        return Pose.model_validate(value)


class HeadCommand(EyeCommand):
    """Request the tracked XR-device or head pose.

    Response validation is inherited from :class:`EyeCommand`.

    Attributes:
        cmd: Wire discriminator, always ``"head"``.
    """

    cmd: Literal["head"] = Field(default="head", frozen=True)


class HandsCommand(Command):
    """Request joint poses for the currently tracked left and right hands.

    Attributes:
        cmd: Wire discriminator, always ``"hands"``.
    """

    cmd: Literal["hands"] = Field(default="hands", frozen=True)

    def validate_response_value(self, value: Response) -> Hands:
        """Validate the response payload as tracked hands.

        Untracked hands are represented by empty pose tuples.

        Args:
            value: Deserialized hand-tracking payload.

        Returns:
            Validated left and right hand poses.

        Raises:
            pydantic.ValidationError: If the payload is not a valid hands model.
        """
        return Hands.model_validate(value)
