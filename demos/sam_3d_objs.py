from typing import Any

from xarp.commands.assets import Element, DefaultAssets
from xarp.data_models import Hands
from xarp.express import AsyncXR, SyncXR
from xarp.gestures import pinch, INDEX_TIP
from xarp.resources import BinaryResource, GLBResource
from xarp.server import run, show_qrcode_link
from xarp.spatial import Transform, Vector3




def sync_app(xr: SyncXR, kwargs: dict[str, Any]) -> None:
    import requests

    url = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/refs/heads/main/2.0/Duck/glTF-Binary/Duck.glb"

    response = requests.get(url)
    response.raise_for_status()  # fail if download failed

    glb_data: bytes = response.content

    duck = Element(
        key="duck",
        data=GLBResource.from_obj(glb_data)
    )




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



if __name__ == '__main__':
    show_qrcode_link()
    run(sync_app)
