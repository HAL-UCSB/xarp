from xarp.data_models import Hands
from xarp.entities import Element, DefaultAssets
from xarp.express import SyncXR
from xarp.gestures import pinch, THUMB_TIP
from xarp.server import run, make_qrcode_image
from xarp.spatial import Transform, Vector3

gray = (.5, .5, .5, 1)
red = (1, 0, 0, .5)
green = (0, 1, 0, .5)
blue = (0, 0, 1, .5)

cursor_scale = Transform(scale=Vector3.one() * .02)

brush = Element(
    key="brush",
    asset=DefaultAssets.sphere(),
    transform=cursor_scale,
    color=gray)

paint = Element(
    key="paint",
    asset=DefaultAssets.sphere(),
    transform=cursor_scale,
    color=green)


def app(xr: SyncXR, *args, **kwargs) -> None:
    xr.say("Brush XR")
    i = 0

    stream = xr.sense(hands=True)
    for frame in stream:
        hands: Hands = frame['hands']

        if hands.right:
            brush.transform.position = hands.right[THUMB_TIP].position
            if pinch(hands.right):
                paint.transform.position = brush.transform.position
                paint.key = f"paint_{i}"
                i += 1
                xr.update(paint)
            xr.update(brush)

        if hands.left and pinch(hands.left):
            i = 0
            xr.destroy_element(all_elements=True)

    stream.close()


if __name__ == '__main__':
    make_qrcode_image()
    run(app)
