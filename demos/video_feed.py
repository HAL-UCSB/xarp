from typing import Any

import PIL
import numpy as np

from xarp.entities import Element, ImageAsset
from xarp.express import SyncXR, AsyncXR
from xarp.gestures import INDEX_TIP
from xarp.server import run, show_qrcode_link

panel = Element(
    key="panel",
    distance=.49
)


def depth_panel(xr: SyncXR, kwargs):
    stream = xr.sense(depth=True, eye=True)
    for frame in stream:
        panel.asset = frame['depth']
        panel.eye = frame['eye']
        xr.update(panel)
    stream.close()


async def async_depth_panel(axr: AsyncXR, kwargs) -> None:
    stream = axr.sense(depth=True, eye=True)
    async for frame in stream:
        panel.asset = frame['depth']
        panel.eye = frame['eye']
        await axr.update(panel)
    await stream.aclose()


def rgb_panel(xr: SyncXR, kwargs) -> None:
    stream = xr.sense(image=True, eye=True)
    for frame in stream:
        panel.asset = frame['image']
        panel.eye = frame['eye']
        xr.update(panel)
    stream.close()


async def async_rgb_panel(axr: AsyncXR, kwargs: dict[str, Any]) -> None:
    stream = axr.sense(image=True, eye=True)
    async for frame in stream:
        panel.asset = frame['image']
        panel.eye = frame['eye']
        await axr.update(panel)
    await stream.aclose()


async def cursor_panel(axr: AsyncXR, kwargs: dict[str, Any]) -> None:
    info = await axr.info()
    stream = axr.sense(image=True, eye=True, hands=True)

    async for frame in stream:
        panel.asset = frame['image']
        panel.eye = frame['eye']
        hands = frame["hands"]
        if hands.right:
            img = panel.asset.obj
            cursor = hands.right[INDEX_TIP]
            pixel = info.camera_intrinsics.world_point_to_panel_pixel(
                cursor.position,
                panel.eye,
                img.width,
                img.height,
                panel.distance)
            draw = PIL.ImageDraw.Draw(img)
            cx, cy = pixel.tolist()
            r = 10
            bbox = (cx - r, cy - r, cx + r, cy + r)
            draw.ellipse(bbox, fill="red", width=2)

        await axr.update(panel)
    await stream.aclose()


async def depth_filter_panel(axr: AsyncXR, kwargs):
    stream = axr.sense(image=True, depth=True, eye=True)
    async for frame in stream:
        rgb_asset =frame["image"]
        rgb = rgb_asset.obj
        size = rgb.width, rgb.height

        depth = frame["depth"].obj
        depth_values = np.asarray(depth, dtype=np.int16)
        depth_threshold = depth_values.mean()
        mask = (depth_values <= depth_threshold).astype(np.uint8) * 255
        stretched = PIL.Image.fromarray(mask, mode="L")
        stretched = stretched.resize(size, resample=PIL.Image.BILINEAR)

        rgb.putalpha(stretched)
        panel.asset = rgb_asset
        panel.eye = frame['eye']
        await axr.update(panel)
    await stream.aclose()


if __name__ == '__main__':
    show_qrcode_link()
    run(depth_panel)
