from typing import Any

import requests

from xarp.entities import Element, MIMEType, Asset
from xarp.express import SyncXR
from xarp.gestures import pinch
from xarp.server import run, make_qrcode_image
from xarp.spatial import Vector3, Transform

video_url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"
response = requests.get(
    video_url,
    headers={"User-Agent": "python"},
    timeout=10
)
response.raise_for_status()
video_bytes = response.content


def app(xr: SyncXR, kwargs: dict[str, Any]) -> None:
    xr.destroy_asset(all_assets=True)
    video_asset = Asset(asset_key="video_asset", raw=video_bytes, mime_type=MIMEType.MP4)
    xr.save(video_asset)
    video_asset.raw = None

    video = Element(
        key="video",
        asset=video_asset,
        play=True,
        transform=Transform(
            position=Vector3.forward(),
        )
    )
    xr.update(video)
    video.asset = None

    stream = xr.sense(hands=True)
    video.time = 0
    video.play = True
    latch = False
    for frame in stream:
        hands = frame["hands"]
        hand = hands.right or hands.left
        if pinch(hand):
            latch = True
        elif latch:
            latch = False
            video.time = 3.0 - video.time
            xr.update(video)


if __name__ == '__main__':
    make_qrcode_image()
    run(app)
