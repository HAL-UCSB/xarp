import asyncio
import io
import os
from typing import List, Tuple

import httpx
import msgpack
import websockets
from PIL import Image as PIL_Image

from xarp import run_xr_app, AsyncXR, Image, Transform
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem

from xarp.time import utc_ts


# -----------------------------
# Helpers
# -----------------------------

async def fetch_fal_ai_token(allowed_apps: List[str] | None = None, expiration: int = 5000) -> str:
    '''
    Fetch a short-lived fal.ai token and return the token string.
    '''
    allowed_apps = allowed_apps or ['krea-wan-14b']
    api_key = os.getenv('fal_api_key')
    if not api_key:
        raise RuntimeError('Missing env var FAL_API_KEY')

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            'https://rest.alpha.fal.ai/tokens/',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Key {api_key}',
            },
            json={
                'allowed_apps': allowed_apps,
                'token_expiration': expiration,
            },
        )
        resp.raise_for_status()
        return resp.json()


def image_to_jpeg_bytes(pil_image: PIL_Image.Image, width: int, height: int) -> bytes:
    '''
    Convert a PIL image to exact (width, height) JPEG bytes.
    '''
    pil_resized = pil_image.convert('RGB').resize((width, height))
    buf = io.BytesIO()
    pil_resized.save(buf, format='JPEG')
    return buf.getvalue()


async def xr_image_eye(xr: AsyncXR, width: int, height: int) -> Tuple[bytes, Transform]:
    '''
    Capture the current XR frame and eye data, returning encoded JPEG bytes and eye.
    '''
    img, eye = await xr.bundle(xr.image, xr.eye)
    pil = img.to_pil_image()
    return image_to_jpeg_bytes(pil, width, height), eye


def msgpack_payload(payload: dict) -> bytes:
    '''
    Msgpack-encode a payload with binary-safe settings.
    '''
    return msgpack.packb(payload, use_bin_type=True)


def decode_image_from_bytes(b: bytes) -> PIL_Image.Image:
    '''
    Open an image from raw bytes and flip vertically.
    '''
    return PIL_Image.open(io.BytesIO(b)).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM).convert('RGBA')


async def fal_step(ws, payload: dict) -> PIL_Image.Image:
    '''
    Send a msgpacked payload to fal ws and receive a raw image back.
    '''
    packed = msgpack_payload(payload)
    await ws.send(packed)
    msg = await ws.recv()
    # The service responds with raw image bytes (not msgpack) in this flow.
    return decode_image_from_bytes(msg)


async def display_rgba(xr: AsyncXR, pil_img: PIL_Image.Image, eye: Transform) -> None:
    await xr.display(
        image=Image(
            pixels=pil_img.tobytes(),
            width=pil_img.width,
            height=pil_img.height),
        eye=eye,
        visible=True,
    )


# -----------------------------
# App
# -----------------------------

async def my_app(xr: AsyncXR):
    width, height = 832, 480
    num_blocks = 50
    prompt = 'I am in a spaceship escaping aliens.'
    strength = 0.45

    token = await fetch_fal_ai_token()
    fal_ws_url = f'wss://fal.run/fal-ai/krea-wan-14b/ws?fal_jwt_token={token}'

    async with websockets.connect(fal_ws_url, max_size=None) as fal_ws:
        # Wait for ready signal (text frame)
        ready = await fal_ws.recv()
        assert ready == '{"status":"ready"}'

        # ---- Initial frame ----
        img, eye = await xr.bundle(xr.image, xr.eye)
        pil = img.to_pil_image()
        pixels = image_to_jpeg_bytes(pil, width, height)
        initial_payload = dict(
            pixels=pixels,
            prompt=prompt,
            num_blocks=num_blocks,
            strength=strength,
            width=width,
            height=height,
            seed=123,
        )
        first_image = await fal_step(fal_ws, initial_payload)
        await display_rgba(xr, first_image, eye)

        # ---- Streaming updates ----
        i = 0
        while True:
            pixels, eye = await xr_image_eye(xr, width, height)
            update_payload = dict(
                pixels=pixels,
                prompt=prompt,
                num_blocks=num_blocks,
                strength=strength,
            )

            print('fal working...', end='')
            frame = await fal_step(fal_ws, update_payload)
            print(' done')

            # Optional local preview (blocks on some platforms)
            # frame.show()

            await display_rgba(xr, frame, eye)
            i += 1
            print('-' * 20, i)


if __name__ == '__main__':
    run_xr_app(my_app, SessionRepositoryLocalFileSystem)
