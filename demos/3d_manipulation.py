import numpy as np
import requests

from xarp.entities import Element, GLBAsset
from xarp.express import SyncXR, AsyncGeneratorIterator
from xarp.gestures import pinch, PALM, fist
from xarp.server import run, make_qrcode_image
from xarp.spatial import Quaternion, Transform, Vector3, matrix_to_trs

glb_url = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/refs/heads/main/2.0/Duck/glTF-Binary/Duck.glb"
response = requests.get(
    glb_url,
    headers={"User-Agent": "python"},
    timeout=10
)
response.raise_for_status()
glb_bytes = response.content


def double_hand_element_manipulation(
        xr: SyncXR,
        element: Element,
        stream: AsyncGeneratorIterator,
        gesture=pinch,
        anchor_joint=PALM,
        freeze_position: bool = False,
        freeze_rotation: bool = False,
        freeze_scale: bool = False,
        freezer_rotation: bool | None = None) -> None:
    if freezer_rotation is not None:
        freeze_rotation = freezer_rotation

    initial_transform = None

    for frame in stream:
        hands = frame["hands"]

        manipulating = True
        for hand in hands:
            if not hand or not gesture(hand):
                manipulating = False
                break

        if not manipulating:
            return

        anchor_hand, supporting_hand = hands.right, hands.left

        if not initial_transform:
            initial_transform = element.transform.model_copy()
            initial_vector = supporting_hand[anchor_joint].position - anchor_hand[anchor_joint].position
            initial_position = anchor_hand[anchor_joint].position + initial_vector * .5

        # Current hand state
        manipulation_vector = supporting_hand[anchor_joint].position - anchor_hand[anchor_joint].position
        manipulation_position = anchor_hand[anchor_joint].position + manipulation_vector * .5

        delta_position = manipulation_position - initial_position
        delta_rotation = initial_vector.aligning_quaternion(manipulation_vector)
        delta_scale = manipulation_vector.norm() / initial_vector.norm()

        if freeze_position:
            delta_position = Vector3.zero()
        if freeze_rotation:
            delta_rotation = Quaternion.identity()
        if freeze_scale:
            delta_scale = 1.0

        # Build pivot transformation matrix:
        # 1. T_pivot_inv: bring initial_position to origin
        # 2. S · R:        apply scale and rotation around origin
        # 3. T_pivot:      move back to initial_position
        # 4. T_delta:      apply positional drift of the midpoint

        pivot = initial_position.to_numpy()

        T_pivot_inv = np.eye(4, dtype=np.float32)
        T_pivot_inv[:3, 3] = -pivot

        SR = np.eye(4, dtype=np.float32)
        SR[:3, :3] = delta_rotation.to_matrix() * delta_scale  # uniform scale

        T_pivot = np.eye(4, dtype=np.float32)
        T_pivot[:3, 3] = pivot

        T_delta = np.eye(4, dtype=np.float32)
        T_delta[:3, 3] = delta_position.to_numpy()

        delta_M = T_delta @ T_pivot @ SR @ T_pivot_inv

        # Apply delta to initial transform matrix
        final_M = delta_M @ initial_transform.to_matrix()

        # Decompose back into components
        position, rotation, scale = matrix_to_trs(final_M)

        element.transform.position = initial_transform.position if freeze_position else position
        element.transform.rotation = initial_transform.rotation if freeze_rotation else rotation
        element.transform.scale = initial_transform.scale if freeze_scale else scale

        xr.update(element)


def app(xr: SyncXR, *args, **kwargs) -> None:
    xr.destroy_asset(all_assets=True)
    duck_asset = GLBAsset(asset_key="duck_asset", raw=glb_bytes)
    xr.save(duck_asset)
    duck_asset.raw = None  # avoid resending the bytes

    duck = Element(
        key="duck",
        asset=duck_asset,
        transform=Transform(
            position=xr.head().ray_point(.25),
            scale=Vector3.one() * .1,
        )
    )
    xr.update(duck)

    stream = xr.sense(hands=True)
    while True:
        double_hand_element_manipulation(xr, duck, stream, gesture=fist, freeze_scale=True)


if __name__ == '__main__':
    make_qrcode_image()
    run(app)
