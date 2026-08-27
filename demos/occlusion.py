"""Occlusion demo — decide, per element, whether the physical world hides it.

Places one of each element type in a row in front of you and flips
``Element.occlusion`` on all of them when you pinch. Pass a hand between yourself
and the row to see the difference: with occlusion on, your hand hides whatever is
behind it; with it off, the elements draw on top of passthrough regardless of
what is actually in front of them.

The status label above the row is pinned to ``occlusion=False`` so it stays
readable either way — occlusion is per element, not a global mode.
"""

from typing import Any

import requests
from PIL import Image, ImageDraw

from xarp import colors
from xarp.data_models import Hands
from xarp.entities import DefaultAssets, Element, GLBAsset, ImageAsset, TextAsset
from xarp.express import SyncXR
from xarp.gestures import pinch
from xarp.server import make_qrcode_image, run
from xarp.spatial import Transform, Vector3

GLB_URL = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/refs/heads/main/2.0/Duck/glTF-Binary/Duck.glb"

ROW_HEIGHT = 1.2
ROW_DISTANCE = 0.6


def swatch(size: int = 256) -> Image.Image:
    """A checkerboard, so the occluded edge is easy to read against a hand."""
    image = Image.new("RGB", (size, size), (30, 34, 38))
    draw = ImageDraw.Draw(image)
    step = size // 8
    for row in range(8):
        for column in range(8):
            if (row + column) % 2:
                continue
            x, y = column * step, row * step
            draw.rectangle([x, y, x + step, y + step], fill=(235, 180, 60))
    return image


def at(x: float, scale: Vector3 | None = None) -> Transform:
    """Slot an element into the row at horizontal offset ``x``."""
    return Transform(
        position=Vector3(x, ROW_HEIGHT, ROW_DISTANCE),
        scale=Vector3.one() if scale is None else scale,
    )


def app(xr: SyncXR, kwargs: dict[str, Any]) -> None:
    xr.destroy_element(all_elements=True)
    xr.destroy_asset(all_assets=True)

    response = requests.get(GLB_URL, headers={"User-Agent": "python"}, timeout=10)
    response.raise_for_status()

    # Stored once by key. Every toggle below re-sends the elements, and dropping
    # raw afterwards keeps those updates from shipping the bytes again — the
    # client skips the reload when the asset key it already holds is unchanged.
    duck_asset = GLBAsset(asset_key="duck", raw=response.content)
    photo_asset = ImageAsset.from_obj(swatch(), asset_key="swatch")
    xr.save(duck_asset)
    xr.save(photo_asset)
    duck_asset.raw = None
    photo_asset.raw = None

    occlusion = True

    # One of each element type. Video is left out only to keep the demo quick to
    # start; it renders through the same material as the primitive below it.
    row = [
        Element(
            key="duck",
            asset=duck_asset,
            transform=at(-0.36, Vector3.one() * 0.1),
            occlusion=occlusion,
        ),
        Element(
            key="ball",
            asset=DefaultAssets.sphere(),
            color=colors.CYAN,
            transform=at(-0.12, Vector3.one() * 0.12),
            occlusion=occlusion,
        ),
        Element(
            key="photo",
            asset=photo_asset,
            transform=at(0.12),
            occlusion=occlusion,
        ),
        Element(
            key="caption",
            asset=TextAsset.from_obj("occluded"),
            transform=at(0.36),
            occlusion=occlusion,
        ),
    ]
    caption = row[3]

    status = Element(
        key="status",
        asset=TextAsset.from_obj("occlusion ON — pinch to toggle"),
        transform=Transform(position=Vector3(0.0, ROW_HEIGHT + 0.3, ROW_DISTANCE)),
        occlusion=False,  # always readable, whatever the row is doing
    )

    xr.update(row + [status])

    was_pinching = False
    stream = xr.sense(hands=True)
    for frame in stream:
        hands: Hands = frame["hands"]
        hand = hands.right or hands.left
        if not hand:
            was_pinching = False
            continue

        pinching = bool(pinch(hand))
        if pinching and not was_pinching:
            occlusion = not occlusion
            for element in row:
                element.occlusion = occlusion
            caption.asset = TextAsset.from_obj("occluded" if occlusion else "always on top")
            status.asset = TextAsset.from_obj(
                f"occlusion {'ON' if occlusion else 'OFF'} — pinch to toggle"
            )
            xr.update(row + [status])
        was_pinching = pinching

    stream.close()


if __name__ == "__main__":
    make_qrcode_image()
    run(app)
