from typing import Any

from xarp.entities import Element
from xarp.express import SyncXR, AsyncXR
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


if __name__ == '__main__':
    show_qrcode_link()
    run(async_rgb_panel)
