import asyncio
import io
import os
from typing import List

import httpx
import msgpack
import websockets
from PIL import Image as PIL_Image

from xarp import run_xr_app, AsyncXR, Image
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from asyncio import Queue


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
    num_blocks = 25

    fal_token = await get_fal_ai_token()
    fal_ws_url = f'wss://fal.run/fal-ai/krea-wan-14b/ws?fal_jwt_token={fal_token}'
    async with websockets.connect(fal_ws_url, max_size=None) as fal_ws:
        message = await fal_ws.recv()
        assert message == '{"status":"ready"}'

        image_eye_queue = Queue(maxsize=10)
        frame_eye_queue = Queue(maxsize=10)

        async def xr_coroutine():
            while True:
                img, eye = await xr.bundle(
                    xr.image,
                    xr.eye)

                print('producing image eye...', end='')
                await image_eye_queue.put((img, eye))
                print(f'ok {image_eye_queue.qsize()}')

                print(f'can I have a frame eye? {frame_eye_queue.qsize()}...', end='')
                frame, eye = await frame_eye_queue.get()
                print('frame eye')

                await xr.display(
                    image=Image(
                        pixels=frame.convert('RGBA').tobytes(),
                        width=width,
                        height=height),
                    eye=eye,
                    visible=True)


        async def fal_coroutine():
            while True:
                print(f'can I have a image eye? {image_eye_queue.qsize()}...', end='')
                img, eye = await image_eye_queue.get()
                print('image eye')

                pil_image = img.to_pil_image().convert('RGB')
                pil_image.thumbnail((width, height))
                buffer = io.BytesIO()
                pil_image.save(buffer, format='JPEG')
                pixels = buffer.getvalue()

                payload = dict(
                    start_frame=pixels,
                    prompt='Spaceship',
                    num_blocks=num_blocks,
                    num_denoising_steps=4,
                    strength=.45,
                    width=width,
                    height=height,
                    seed=42)
                msgpack_payload = msgpack.packb(payload)

                print(f'fal working...', end='')
                await fal_ws.send(msgpack_payload)
                print(f' ... ', end='')
                msgpack_response = await fal_ws.recv()
                print('done')

                pixels = io.BytesIO(msgpack_response)
                response = PIL_Image.open(pixels).transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)

                print('producing frame...',end='')
                await frame_eye_queue.put((response, eye))
                print(f'ok {frame_eye_queue.qsize()}')

        await asyncio.gather(
            xr_coroutine(),
            fal_coroutine())


if __name__ == '__main__':
    run_xr_app(my_app, SessionRepositoryLocalFileSystem)
