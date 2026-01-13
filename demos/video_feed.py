from typing import Any

from xarp.commands.assets import Element
from xarp.express import SyncXR, AsyncXR
from xarp.server import run, show_qrcode_link
from xarp.settings import settings
from xarp.spatial import Pose

panel = Element(
    key="panel",
    eye=Pose(),
    distance=.49
)


def sync_app(xr: SyncXR, kwargs: dict[str, Any]) -> None:
    xr.say("Video Feed")
    senses = xr.sense(image=True, eye=True)
    for frame in senses:
        panel.binary = frame['image']
        panel.eye = frame['eye']
        xr.update(panel)
    senses.close()


async def async_app(axr: AsyncXR, kwargs: dict[str, Any]) -> None:
    await axr.say("Video Feed")
    senses = axr.sense(image=True, eye=True)
    async for frame in senses:
        panel.binary = frame['image']
        panel.eye = frame['eye']
        await axr.update(panel)
    senses.aclose()


async def async_app_depth(axr: AsyncXR, kwargs: dict[str, Any]) -> None:
    await axr.say("Video Feed")
    senses = axr.sense(depth=True, eye=True)
    async for frame in senses:
        panel.binary = frame['depth']
        panel.eye = frame['eye']
        await axr.update(panel)
    senses.aclose()

if __name__ == '__main__':
    show_qrcode_link(path=settings.ws_path)
    run(async_app_depth)
