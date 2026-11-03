"""Move, rotate, and scale a duck by squeezing both Quest controller grips.

Release both grips at startup and after tracking loss, then squeeze both to grab.
Move both controllers to translate, turn their connecting line to rotate, and
spread them apart to scale. Release either grip to leave the duck in place.
Requires a Quest client with the controllers executor.
"""

import requests

from xarp.entities import Element, GLBAsset
from xarp.express import SyncXR
from xarp.server import make_qrcode_image, run
from xarp.spatial import Transform, Vector3


def manipulate(xr: SyncXR, duck: Element) -> None:
    armed = False
    initial = None
    stream = xr.sense(controllers=True)
    try:
        for frame in stream:
            controllers = frame["controllers"]
            left, right = controllers.left, controllers.right
            if not all(c.connected and c.tracked and c.grip_pose is not None
                       for c in (left, right)):
                armed = False
                initial = None
                continue

            if left.grip <= 0.2 and right.grip <= 0.2:
                armed = True
            if left.grip <= 0.2 or right.grip <= 0.2:
                initial = None
                continue
            if not armed or (initial is None and min(left.grip, right.grip) < 0.8):
                continue

            vector = left.grip_pose.position - right.grip_pose.position
            # Avoid unstable scale and rotation when the controllers coincide.
            if vector.norm() < 0.01:
                initial = None
                continue
            midpoint = right.grip_pose.position + vector * 0.5
            if initial is None:
                initial = duck.transform.model_copy()
                initial_vector = vector
                initial_midpoint = midpoint

            rotation = initial_vector.aligning_quaternion(vector)
            scale = vector.norm() / initial_vector.norm()
            duck.transform.position = midpoint + rotation.rotate_vector(
                (initial.position - initial_midpoint) * scale)
            duck.transform.rotation = rotation * initial.rotation
            duck.transform.scale = initial.scale * scale
            xr.update(duck)
    finally:
        stream.close()


def app(xr: SyncXR, *args, **kwargs) -> None:
    response = requests.get(
        "https://github.com/KhronosGroup/glTF-Sample-Models/raw/refs/heads/main/2.0/Duck/glTF-Binary/Duck.glb",
        headers={"User-Agent": "python"}, timeout=10,
    )
    response.raise_for_status()
    asset = GLBAsset(asset_key="duck_asset", raw=response.content)
    xr.save(asset)
    asset.raw = None
    duck = Element(
        key="duck", asset=asset,
        transform=Transform(position=xr.head().ray_point(0.25), scale=Vector3.one() * 0.1),
    )
    xr.update(duck)
    manipulate(xr, duck)


if __name__ == "__main__":
    make_qrcode_image()
    run(app)
