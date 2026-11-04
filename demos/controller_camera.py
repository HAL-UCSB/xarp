"""Frame a live camera crop between two Quest controllers.

Release both index-finger triggers first, then hold both to show the crop.
Controller positions define opposite corners; the window follows the camera's
orientation. Release either trigger to leave the last image in place. After
tracking loss, release both triggers before starting again.
Requires a Quest client with camera access and the controllers executor.
"""

from camera import crop_image_rectangle, ray_plane_intersection

from xarp.entities import Element
from xarp.express import SyncXR
from xarp.server import make_qrcode_image, run


def app(xr: SyncXR, *args, **kwargs) -> None:
    panel = Element(key="image_panel")
    armed = False
    framing = False
    stream = xr.sense(image=True, eye=True, controllers=True)
    try:
        for frame in stream:
            controllers = frame["controllers"]
            left, right = controllers.left, controllers.right
            if not all(c.connected and c.tracked and c.grip_pose is not None
                       for c in (left, right)):
                armed = framing = False
                continue

            if left.trigger <= 0.2 and right.trigger <= 0.2:
                armed = True
            if left.trigger <= 0.2 or right.trigger <= 0.2:
                framing = False
            elif armed and min(left.trigger, right.trigger) >= 0.8:
                framing = True
            if not framing:
                continue

            eye = frame["eye"]
            left_position = left.grip_pose.position
            right_position = right.grip_pose.position
            try:
                corner_a = ray_plane_intersection(eye, left_position - eye.position)
                corner_b = ray_plane_intersection(eye, right_position - eye.position)
                if abs(corner_a.x - corner_b.x) < 1e-6 or abs(corner_a.y - corner_b.y) < 1e-6:
                    continue
                image = crop_image_rectangle(frame["image"], corner_a, corner_b)
            except ValueError:
                # Corners behind the camera or outside its image cannot form a crop.
                continue
            panel.asset = image
            panel.transform.position = (left_position + right_position) * 0.5
            panel.transform.rotation = eye.rotation
            xr.update(panel)
    finally:
        stream.close()


if __name__ == "__main__":
    make_qrcode_image()
    run(app)
