"""Draw with Quest controllers using a client with the "controllers" executor.

Python API (following xr.hands() / xr.sense(hands=True)):
    controllers = xr.controllers()              # one sample
    stream = xr.sense(controllers=True)         # repeated samples
    controllers = frame["controllers"]

Model: Controllers(left, right).
Each side always exists, even when unavailable, with:
    tracked: bool
    grip_pose: Pose | None       # valid when tracked
    trigger: float              # normalized 0..1
    grip: float                 # normalized squeeze, 0..1
    thumbstick: tuple[float, float]  # each axis -1..1
    buttons.primary: bool       # X on left, A on right
    buttons.secondary: bool     # Y on left, B on right

Buttons report current held state. The demo detects a clear press locally
when the left primary button changes from released to pressed. A disconnected
left controller is treated as released; its button does not require a valid pose.
With rt=True, intermediate samples may be dropped: brief button presses or
tracking interruptions between delivered samples can be missed. Reliable event
delivery is not provided by this sampling API.

The Quest client converts grip poses into the world space used by scene elements.
Release the right trigger once at startup and after tracking loss to arm drawing.
Press the left X button to clear all scene elements.
"""

from xarp import colors
from xarp.data_models import Controllers
from xarp.entities import DefaultAssets, Element
from xarp.express import SyncXR
from xarp.server import make_qrcode_image, run
from xarp.spatial import Transform, Vector3

# Start drawing at 0.8 and stop at 0.2 to avoid toggling from trigger jitter.
TRIGGER_ON = 0.8
TRIGGER_OFF = 0.2


def app(xr: SyncXR, *args, **kwargs) -> None:
    brush = Element(
        key="brush", asset=DefaultAssets.sphere(), color=colors.GRAY,
        transform=Transform(scale=Vector3.one() * 0.02), active=False,
    )
    paint = Element(
        key="paint", asset=DefaultAssets.sphere(), color=colors.GREEN,
        transform=Transform(scale=Vector3.one() * 0.02),
    )
    i = 0
    clear_was_pressed = False
    armed = False
    drawing = False
    cursor_visible = False

    # Requires a Quest client with a registered "controllers" executor.
    stream = xr.sense(controllers=True)
    try:
        for frame in stream:
            controllers: Controllers = frame["controllers"]
            left, right = controllers.left, controllers.right

            clear_pressed = left.buttons.primary
            clear_now = clear_pressed and not clear_was_pressed
            clear_was_pressed = clear_pressed
            if clear_now:
                # Like the original hand demo, clear all scene elements.
                xr.destroy_element(all_elements=True)
                i = 0
                cursor_visible = False

            valid = right.tracked and right.grip_pose is not None
            if not valid:
                armed = drawing = False
                if cursor_visible:
                    brush.active = False
                    xr.update(brush)
                    cursor_visible = False
                continue

            brush.active = True
            brush.transform.position = right.grip_pose.position
            xr.update(brush)
            cursor_visible = True

            if right.trigger <= TRIGGER_OFF:
                armed = True
                drawing = False
            elif armed and right.trigger >= TRIGGER_ON:
                drawing = True

            # Clear skips painting this frame. If the trigger remains held,
            # drawing resumes next frame, even while the clear button is held.
            if drawing and not clear_now:
                paint.key = f"paint_{i}"
                paint.transform.position = right.grip_pose.position
                xr.update(paint)
                i += 1
    finally:
        stream.close()


if __name__ == "__main__":
    make_qrcode_image()
    run(app)
