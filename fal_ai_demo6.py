import io
import os
from typing import List

import httpx
import msgpack
import websockets
from PIL import Image as PIL_Image

from xarp import run_xr_app, AsyncXR, Image
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem


async def get_fal_ai_token(allowed_apps: List[str] = None, expiration: int = 5000):
    if allowed_apps is None:
        allowed_apps = ['krea-wan-14b']
    async with httpx.AsyncClient() as client:
        fal_tokens_url = 'https://rest.alpha.fal.ai/tokens/'
        response = await client.post(
            fal_tokens_url,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Key {os.getenv("fal_api_key")}'
            },
            json={
                'allowed_apps': allowed_apps,
                'token_experiation': expiration
            }
        )
        response.raise_for_status()
        return response.json()


async def my_app(xr: AsyncXR):
    width = 832
    height = 480
    num_blocks = 50
    #prompt = 'Video in first-person perspective. You are sailing a boat on a sunny day in Hawaii!'
    #prompt = 'Video in first-person perspective. You are playing drums on the stage at a huge live concert. The venue is crowed with fans.'
    prompt = 'Video in first-person perspective. You are a race driver piloting a formula-1 in a track in a grand-prix, driving in high speed.'
    strength = .45

    fal_token = await get_fal_ai_token()
    fal_ws_url = f'wss://fal.run/fal-ai/krea-wan-14b/ws?fal_jwt_token={fal_token}'

    while True:
        async with websockets.connect(fal_ws_url, max_size=None) as fal_ws:
            message = await fal_ws.recv()
            assert message == '{"status":"ready"}'

            img, eye = await xr.bundle(
                xr.image,
                xr.eye)
            pil_image = img.as_pil_image().convert('RGB').resize((width, height))
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            buffer.seek(0)
            pixels = buffer.read()
            payload = dict(
                start_frame=pixels,
                prompt=prompt,
                num_blocks=num_blocks,
                num_denoising_steps=6,
                strength=strength,
                width=width,
                height=height,
                enable_prompt_expansion=False,
                seed=42)
            msgpack_payload = msgpack.packb(payload, use_bin_type=True)
            await fal_ws.send(msgpack_payload)
            msgpack_response = await fal_ws.recv()
            pixels = io.BytesIO(msgpack_response)
            response = PIL_Image.open(pixels).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)
            await xr.display(
                image=Image(
                    pixels=response.convert('RGBA').tobytes(),
                    width=width,
                    height=height),
                opacity=.9,
                eye=eye,
                visible=True)

            img, eye = await xr.bundle(
                xr.image,
                xr.eye)
            pil_image = img.as_pil_image().convert('RGB').resize((width, height))
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            buffer.seek(0)
            pixels = buffer.read()
            payload = dict(
                image=pixels,
                prompt=prompt,
                num_blocks=num_blocks,
            )
            msgpack_payload = msgpack.packb(payload, use_bin_type=True)
            await fal_ws.send(msgpack_payload)
            msgpack_response = await fal_ws.recv()
            pixels = io.BytesIO(msgpack_response)
            response = PIL_Image.open(pixels).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)
            await xr.display(
                image=Image(
                    pixels=response.convert('RGBA').tobytes(),
                    width=width,
                    height=height),
                opacity=.9,
                eye=eye,
                visible=True)


if __name__ == '__main__':
    run_xr_app(my_app, SessionRepositoryLocalFileSystem)
