from typing import Any

from xarp.data_models import Hands
from xarp.entities import Element, DefaultAssets
from xarp.express import AsyncXR, SyncXR
from xarp.gestures import pinch, INDEX_TIP
from xarp.server import run, show_qrcode_link
from xarp.spatial import Transform, Vector3

gray = (.5, .5, .5, 1)
red = (1, 0, 0, 1)
green = (0, 1, 0, 1)
blue = (0, 0, 1, 1)

brush = Element(
    key="cursor",
    color=red,
    asset=DefaultAssets.SPHERE,
    transform=Transform(
        scale=Vector3.one() * .02
    )
)

paint = Element(
    key="",
    color=green,
    asset=DefaultAssets.SPHERE,
    transform=brush.transform
)


def sync_app(xr: SyncXR, kwargs: dict[str, Any]) -> None:
    xr.say("Brush XR")

    i = 0
    senses = xr.sense(hands=True)
    for frame in senses:
        hands: Hands = frame['hands']

        # Clear
        if hands.left and pinch(hands.left):
            i = 0
            xr.destroy_element(all_elements=True)
            continue

        brush.active = bool(hands.right)
        # No brush
        if not brush.active:
            xr.update(brush)
            continue

        brush.transform.position = hands.right[INDEX_TIP].position
        xr.update(brush)

        # Not painting
        if not pinch(hands.right):
            continue

        paint.key = f"paint_{i}"
        xr.update(paint)
        i += 1

    senses.close()


async def async_app(axr: AsyncXR, kwargs: dict[str, Any]) -> None:
    await axr.say("Brush XR")

    i = 0
    senses = axr.sense(hands=True)
    async for frame in senses:
        hands: Hands = frame['hands']

        brush.active = bool(hands.right)
        # No brush
        if not brush.active:
            await axr.update(brush)
            continue

        brush.transform.position = hands.right[INDEX_TIP].position
        await axr.update(brush)

        # Not painting
        if not pinch(hands.right):
            continue

        paint.key = f"paint_{i}"
        await axr.update(paint)
        i += 1

        # Clear
        if hands.left and pinch(hands.left):
            i = 0
            await axr.destroy_element(all_elements=True)
            continue

    await senses.aclose()


if __name__ == '__main__':
    show_qrcode_link()
    run(sync_app)
