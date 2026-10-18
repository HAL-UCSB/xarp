"""API DESIGN DRAFT: controller support is NOT implemented in XARP yet.

Proposed public API (following xr.hands() / xr.sense(hands=True)):
    controllers = xr.controllers()              # one sample
    stream = xr.sense(controllers=True)         # repeated samples
    controllers = frame["controllers"]

Proposed model: Controllers(session_id, left, right).
Each side always exists, even when unavailable, with:
    connected: bool
    tracked: bool
    grip_pose: Pose | None       # valid when connected and tracked
    trigger: float              # normalized 0..1
    grip: float                 # normalized squeeze, 0..1
    thumbstick: tuple[float, float]  # each axis -1..1
    buttons.primary: bool       # X on left, A on right
    buttons.secondary: bool     # Y on left, B on right
    buttons.primary_press_count: int
    tracking_generation: int

Counter semantics PROPOSED FOR REVIEW:
- primary_press_count increments on each press; holding does not increment it.
- Counters accumulate within a session, surviving dropped stream samples.
- tracking_generation increments on each loss of valid connection/tracking,
  even if the loss and recovery occur between delivered samples.
- session_id changes on a new connection; counters are baselined on first sample.
- Device-side input sampling must capture these transitions independently of
  Python consumption. Counters cannot recover transitions never captured.

These fields are a proposal, not an agreed or implemented protocol. They allow
this example to notice clear presses and tracking interruptions across dropped
frames without claiming to provide a general ordered event history.

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
    session_id = None
    tracking_generation = None
    clear_count = 0
    armed = False
    drawing = False
    cursor_visible = False

    # Proposed API: NOT supported by the current XARP implementation.
    stream = xr.sense(controllers=True)
    try:
        for frame in stream:
            controllers = frame["controllers"]
            left, right = controllers.left, controllers.right

            if controllers.session_id != session_id:
                session_id = controllers.session_id
                tracking_generation = right.tracking_generation
                clear_count = left.buttons.primary_press_count
                armed = drawing = False

            # Detect an interruption even if no invalid sample reached Python.
            if right.tracking_generation != tracking_generation:
                tracking_generation = right.tracking_generation
                armed = drawing = False

            clear_now = left.buttons.primary_press_count > clear_count
            clear_count = left.buttons.primary_press_count
            if clear_now:
                # Like the original hand demo, clear all scene elements.
                # Multiple pending clears collapse to one for this demo.
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
