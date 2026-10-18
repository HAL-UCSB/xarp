"""API DESIGN DRAFT: controller support is NOT implemented in XARP yet.

Proposed public API (following xr.hands() / xr.sense(hands=True)):
    controllers = xr.controllers()              # one sample
    stream = xr.sense(controllers=True)         # repeated samples
    controllers = frame["controllers"]

Proposed model: Controllers(left, right).
Each side always exists, even when unavailable, with:
    connected: bool
    tracked: bool
    grip_pose: Pose | None       # valid when connected and tracked
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
delivery is a separate design question, not implemented by this draft.

Poses must use the same space as scene elements. Exact conversion and units
must be verified in the client before implementation. Trigger thresholds below
are example values to confirm on hardware. This file is syntax-checkable but
cannot run against the current xr.sense() signature.
"""

from xarp import colors
from xarp.entities import DefaultAssets, Element
from xarp.express import SyncXR
from xarp.server import make_qrcode_image, run
from xarp.spatial import Transform, Vector3

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

    # Proposed API: NOT supported by the current XARP implementation.
    stream = xr.sense(controllers=True)
    try:
        for frame in stream:
            controllers = frame["controllers"]
            left, right = controllers.left, controllers.right

            clear_pressed = left.connected and left.buttons.primary
            clear_now = clear_pressed and not clear_was_pressed
            clear_was_pressed = clear_pressed
            if clear_now:
                # Like the original hand demo, clear all scene elements.
                xr.destroy_element(all_elements=True)
                i = 0
                cursor_visible = False

            valid = right.connected and right.tracked and right.grip_pose is not None
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

            # Clear wins over drawing in the current processed frame.
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
