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
    num_blocks = 1

    fal_token = await get_fal_ai_token()
    fal_ws_url = f'wss://fal.run/fal-ai/krea-wan-14b/ws?fal_jwt_token={fal_token}'
    async with websockets.connect(fal_ws_url, max_size=None) as fal_ws:
        message = await fal_ws.recv()
        assert message == '{"status":"ready"}'

        while True:

            eye, image = await xr.bundle(
                xr.eye,
                xr.image)

            pil_image = image.to_pil_image().convert('RGB')
            pil_image.thumbnail((width, height))

            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            pixels = buffer.getvalue()
            pil_image.show()

            payload = dict(
                start_frame=pixels,
                prompt='Japanese Edo Period',
                num_blocks=num_blocks,
                num_denoising_steps=4,
                strength=.475,
                width=width,
                height=height,
                seed=123123,
            )
            msgpack_payload = msgpack.packb(payload)
            await fal_ws.send(msgpack_payload)

            try:
                await xr.write('Generating...')
                i = 0
                frame_keys = []
                while i < num_blocks * 6:
                    i += 1
                    frame_key = f'frame_{i}'
                    frame_keys.append(frame_key)
                    print(f'waiting {frame_key}')

                    msgpack_response = await fal_ws.recv()
                    response = PIL_Image.open(io.BytesIO(msgpack_response)).transpose(
                        PIL_Image.Transpose.FLIP_TOP_BOTTOM)

                    await xr.display(
                        image=Image(
                            pixels=response.convert('RGBA').tobytes(),
                            width=width,
                            height=height),
                        key=frame_key,
                        visible=False)

            except Exception as e:
                print(e)

            for frame_key in frame_keys:
                await xr.display(
                    eye=eye,
                    key=frame_key,
                    visible=True)
                print('playing', frame_key)


if __name__ == '__main__':
    run_xr_app(my_app, SessionRepositoryLocalFileSystem)
