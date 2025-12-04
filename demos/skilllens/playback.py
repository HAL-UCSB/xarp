import base64
import io
from asyncio import sleep
from functools import partial
from urllib.parse import urlparse

import numpy as np

from xarp import XR, run_xr_app, settings, Hands, AsyncXR
from xarp.data_models.responses import Image
from xarp.data_models.chat import ChatMessage
from xarp.data_models.spatial import centroid
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem

RED = 1, 0, 0, 1
GREEN = 0, 1, 0, 1
BLUE = 0, 0, 1, 1
GREY = .35, .35, .35, .25

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
    image_keys.append(key)

frames = list(zip(image_keys, centroid_pairs, eyes))


async def load_demo(xr: AsyncXR):
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
        await xr.save(key)
        print('loading frame ', key)


def get_stage_caption(_frame_key):
    key = int(_frame_key[len('frame_'):])
    if key < 33:
        return 'Load the grinder with the coffee beans.'
    if key < 80:
        return 'Turn the lever to grind the beans.'
    if key < 131:
        return 'Transfer the coffee powder to the portafilter.'
    if key < 151:
        return 'Attach the portafilter to the expresso machine head.'
    if key < 160:
        return 'Place the mug under the head.'
    if key < 174:
        return 'Press the extraction button.'
    return 'Enjoy your expresso!'


async def playback(xr: AsyncXR):
    previous_instruction = None
    previous_image_frame = None
    pre_data = await xr.sense(eye=True)

    for frame in frames:
        image_key, (left_centroid, right_centroid), eye = frame
        y_offset = pre_data.eye.position[1] - eye.position[1]
        left_centroid[1] += y_offset
        right_centroid[1] += y_offset
        eye.position[1] += y_offset

        error_threshold = 0.15
        error = error_threshold * 1.01

        if previous_image_frame is not None:
            change_frame = {
                previous_image_frame: dict(
                    visible=False
                ),
                image_key: dict(
                    visible=True,
                    eye=eye,
                    opacity=1 - error,  # np.max([error, .5]),
                    depth=.48725 * .4,
                )
            }
            await xr.bundle(**change_frame)
        previous_image_frame = image_key

        current_instruction = get_stage_caption(image_key)
        if previous_instruction != current_instruction:
            previous_instruction = current_instruction
            await xr.say(current_instruction)

        while error > error_threshold:
            data = await xr.sense(hands=True)
            hands = data.hands

            if left_centroid is not None:
                if hands.left:
                    i_left_centroid = centroid(hands.left)
                    left_error = np.linalg.norm(left_centroid - i_left_centroid)
                    await xr.sphere(left_centroid, scale=.025, color=GREY, key='_left')
                else:
                    left_error = np.inf
            else:
                left_error = 0

            if right_centroid is not None:
                if hands.right:
                    i_right_centroid = centroid(hands.right)
                    right_error = np.linalg.norm(right_centroid - i_right_centroid)
                    await xr.sphere(right_centroid, scale=.025, color=GREY, key='_right')
                else:
                    right_error = np.inf
            else:
                right_error = 0

            error = np.max([left_error, right_error]) / 1.2
            if error > 1:
                print(error)

            await xr.display(
                opacity=1 - error,  # np.max([error, .5]),
                visible=True,
                eye=eye,
                depth=.48725 * .4,
                key=image_key)

    await xr.display(key=image_key, visible=False)


async def playback_app(xr: AsyncXR):
    await xr.say('Loading Demonstration')
    # frames = await load_demo(xr)

    # while True:
    #     await sleep(3)

    for ik in image_keys:
        await xr.load(ik)

    await xr.say('Demonstration ready')
    while True:
        await playback(xr)


if __name__ == '__main__':
    run_xr_app(
        playback_app,
        SessionRepositoryLocalFileSystem)
