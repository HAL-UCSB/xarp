from typing import Any

import requests

from xarp.entities import Element, MIMEType, Asset
from xarp.express import SyncXR
from xarp.server import run, make_qrcode_image
from xarp.spatial import Transform, Vector3

ogg_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg"
response = requests.get(
    ogg_url,
    headers={"User-Agent": "python"},
    timeout=10
)
response.raise_for_status()
audio_bytes = response.content

# # duck (GLB) asset
# glb_url = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/refs/heads/main/2.0/Duck/glTF-Binary/Duck.glb"
# response = requests.get(
#     glb_url,
#     headers={"User-Agent": "python"},
#     timeout=10
# )
# response.raise_for_status()
# glb_bytes = response.content

# CC-BY nature clip (Chinese lanterns, HD 1080p) via Internet Archive
video_url = "https://archive.org/download/ChineseLanternCCBYNatureClip/Chinese%20lantern%20CC-BY%20NatureClip.mp4"
response = requests.get(
    video_url,
    headers={"User-Agent": "python"},
    timeout=10
)
response.raise_for_status()
video_bytes = response.content


def app(xr: SyncXR, kwargs: dict[str, Any]) -> None:
    xr.destroy_asset(all_assets=True)

    # # duck asset
    # duck_asset = GLBAsset(asset_key="duck_asset", raw=glb_bytes)
    # xr.save(duck_asset)
    # duck_asset.raw = None  # avoid resending the bytes
    #
    # duck = Element(
    #     key="duck",
    #     asset=duck_asset,
    #     transform=Transform(
    #         position=Vector3.up() * 0.05,
    #         scale=Vector3(1, -1, 1) * .1,
    #     )
    # )
    # xr.update(duck)

    # # label asset
    # label = Element(
    #     key="label",
    #     asset=TextAsset.from_obj("hello"),
    # )
    # xr.update(label)

    example_audio = Asset(asset_key="example_audio", raw=audio_bytes, mime_type=MIMEType.OGG)
    xr.save(example_audio)
    audio = Element(
        key="audio",
        asset=example_audio,
    )
    xr.update(audio)

    video_asset = Asset(asset_key="video_asset", raw=video_bytes, mime_type=MIMEType.MP4)
    xr.save(video_asset)
    video_asset.raw = None

    video = Element(
        key="video",
        asset=video_asset,
        play=True,
        transform=Transform(
            position=Vector3.forward() + Vector3.up() * 1.25,
        )
    )
    xr.update(video)

    stream = xr.sense(hands=True)
    for frame in stream:
        pass  # hand track