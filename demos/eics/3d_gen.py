import io
from time import sleep

import PIL
import httpx
import numpy as np
import requests

from xarp.entities import Element, GLBAsset, ImageAsset
from xarp.express import SyncXR
from xarp.gestures import pinch, PALM
from xarp.server import show_qrcode_link, run
from xarp.spatial import Transform, Vector3

START_SAM3D_OBJS_URL = "http://128.111.28.83:8000/start"
STATUS_SAM3D_OBJS_URL = "http://128.111.28.83:8000/status"
DOWNLOAD_SAM3D_OBJS_URL = "http://128.111.28.83:8000/download"


def create_binary_vignette(img: ImageAsset, pixel_radius) -> PIL.Image.Image:
    img_obj = img.obj
    w, h = img_obj.size

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    r2 = float(pixel_radius) * float(pixel_radius)

    ys = (np.arange(h, dtype=np.float32) - cy)[:, None]  # shape (h, 1)
    xs = (np.arange(w, dtype=np.float32) - cx)[None, :]  # shape (1, w)
    dist2 = ys * ys + xs * xs

    mask = (dist2 <= r2).astype(np.uint8) * 255

    return PIL.Image.fromarray(mask, mode="L")


def create_image_to_3d_job(img: ImageAsset, mask: PIL.Image.Image):
    img = img.obj
    masked = img.copy()
    masked.putalpha(mask)
    masked.show()

    fmt = "png"
    img_buf = io.BytesIO()
    mask_buf = io.BytesIO()
    img.save(img_buf, format=fmt)
    mask.save(mask_buf, format=fmt)
    img_buf.seek(0)
    mask_buf.seek(0)

    files = {
        "upload_image": (f"image.{fmt}", img_buf, f"image/{fmt}"),
        "upload_mask": (f"mask.{fmt}", mask_buf, f"image/{fmt}"),
    }

    with httpx.Client() as client:
        resp = client.post(START_SAM3D_OBJS_URL, files=files)
        resp.raise_for_status()
        return resp.json()["job_id"]


def is_in_progress(job_id: str) -> bool:
    with httpx.Client() as client:
        resp = client.get(f"{STATUS_SAM3D_OBJS_URL}/{job_id}")
        resp.raise_for_status()
        status_payload = resp.json()
        status = status_payload.get("status")

        if status == "failed":
            raise Exception("SAM3D-Objects job failure")
        if status == "done":
            return False
        return True


def download_glb(job_id: str) -> bytes:
    while is_in_progress(job_id):
        sleep(1)
    with httpx.Client() as client:
        resp = client.get(f"{DOWNLOAD_SAM3D_OBJS_URL}/{job_id}")
        resp.raise_for_status()
        return resp.content


def app(xarp: SyncXR, params):
    job_id = None

    stream = xarp.sense(hands=True)
    for frame in stream:
        hands = frame["hands"]
        if hands.right and pinch(hands.right):
            img = xarp.image()
            mask = create_binary_vignette(img, 200)
            job_id = create_image_to_3d_job(img, mask)
            break

    mesh_asset = GLBAsset(
        asset_key="mesh_asset",
        raw=download_glb(job_id)
    )
    xarp.save(mesh_asset)
    mesh_asset.raw = None

    mesh = Element(
        key="mesh",
        asset=mesh_asset,
        distance=1
    )
    xarp.update(mesh)
    mesh.distance = None

    # stream = xarp.sense(hands=True)
    for frame in stream:
        hands = frame["hands"]
        if hand := hands.right or hands.left:
            palm = hand[PALM]
            mesh.transform.position = palm.position
            mesh.transform.rotation = palm.rotation
            mesh.transform.scale = Vector3.one() * .259
            xarp.update(mesh)
    stream.close()


def debug_app(xarp: SyncXR, params):
    glb_url = "https://github.com/KhronosGroup/glTF-Sample-Models/raw/refs/heads/main/2.0/Duck/glTF-Binary/Duck.glb"
    response = requests.get(
        glb_url,
        headers={"User-Agent": "python"},
        timeout=10
    )
    response.raise_for_status()
    glb_bytes = response.content

    duck_asset = GLBAsset(asset_key="duck_asset", raw=glb_bytes)
    xarp.save(duck_asset)
    duck_asset.raw = None

    duck = Element(
        key="duck",
        asset=duck_asset,
        transform=Transform(
            scale=Vector3.from_xyz(1, -1, 1) * 0.05
        )
    )
    xarp.update(duck)


if __name__ == '__main__':
    show_qrcode_link()
    run(app)
