import io
import os
from typing import List

import httpx
import msgpack
import numpy as np
import websockets
from PIL import Image as PIL_Image

from xarp import XR, settings, Hands
from xarp import run_xr_app, AsyncXR, Image
from xarp.data_models.spatial import centroid
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


RED = 1, 0, 0, 1
GREEN = 0, 1, 0, 1
BLUE = 0, 0, 1, 1


async def preload_expresso_demo(xr: AsyncXR):
    # load session
    sessions = SessionRepositoryLocalFileSystem(settings.local_storage)
    session = sessions.get('foo', 42)

    chat = sorted(session.chat, key=lambda msg: msg.ts)

    eyes = [message.content for message in chat if message.mimetype == 'application/xarp/transform']

    centroid_pairs = []
    for message in chat:
        if message.mimetype != 'application/xarp/hands':
            continue
        hands: Hands = message.content
        pair = (
            centroid(hands.left) if hands.left else None,
            centroid(hands.right) if hands.right else None)
        centroid_pairs.append(pair)

    image_keys = []
    i = 0
    for message in chat:
        if message.mimetype != 'application/xarp/image':
            continue
        image: Image = message.content
        image.load_pixels()
        key = f'frame_{i}'
        i += 1
        # transfer the frame to the client to save time but keep it hidden
        await xr.display(image=image, visible=False, key=key)
        image_keys.append(key)

    return zip(image_keys, centroid_pairs, eyes)


async def playback(xr: AsyncXR, frames_iter):
    for frame in frames_iter:
        image_key, (left_centroid, right_centroid), eye = frame
        error = np.inf
        while error > 0.2:
            hands = await xr.hands()

            if left_centroid is not None:
                if hands.left:
                    i_left_centroid = centroid(hands.left)
                    left_error = np.linalg.norm(left_centroid - i_left_centroid)
                else:
                    left_error = np.inf
            else:
                left_error = 0

            if right_centroid is not None:
                if hands.right:
                    i_right_centroid = centroid(hands.right)
                    right_error = np.linalg.norm(right_centroid - i_right_centroid)
                else:
                    right_error = np.inf
            else:
                right_error = 0

            error = np.max([left_error, right_error])

            # update visual feedback
            await xr.display(
                opacity=np.max([error, .5]),
                visible=True,
                eye=eye,
                key=image_key)

        # frame cleared
        await xr.display(
            visible=False,
            key=image_key)


async def fal(xr: AsyncXR, exp_key: int, fal_ws_url):
    width = 832
    height = 480
    num_blocks = 50

    prompt = [
        'Video in first-person perspective. You are sailing a boat on a sunny day in Hawaii.',
        'Video in first-person perspective. You are playing drums on the stage at a huge live concert. The venue is crowed with fans.',
        'Video in first-person perspective. You are a race driver piloting a formula-1 car.'
    ][exp_key]

    strength = .45

    async with websockets.connect(fal_ws_url, max_size=None) as fal_ws:
        message = await fal_ws.recv()
        assert message == '{"status":"ready"}'

        img, eye = await xr.bundle(
            xr.image,
            xr.eye)
        pil_image = img.to_pil_image().convert('RGB').resize((width, height))
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
        pil_image = img.to_pil_image().convert('RGB').resize((width, height))
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


async def demo_app(xr: AsyncXR):
    fal_token = await get_fal_ai_token()
    fal_ws_url = f'wss://fal.run/fal-ai/krea-wan-14b/ws?fal_jwt_token={fal_token}'

    await xr.write('Loading Demonstration')
    frames = await preload_expresso_demo(xr)
    pass


if __name__ == '__main__':
    run_xr_app(
        demo_app,
        SessionRepositoryLocalFileSystem)
