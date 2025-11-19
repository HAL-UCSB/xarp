import base64
import io
from functools import partial
from urllib.parse import urlparse

import numpy as np

from xarp import XR, run_xr_app, settings, Hands
from xarp.data_models.app import Image
from xarp.data_models.chat import ChatMessage
from xarp.data_models.spatial import centroid
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem

RED = 1, 0, 0, 1
GREEN = 0, 1, 0, 1
BLUE = 0, 0, 1, 1


def load_demo(xr: XR):
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
        xr.display(image=image, visible=False, key=key)
        image_keys.append(key)
        print('loading frame ', key)

    return list(zip(image_keys, centroid_pairs, eyes))

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


def playback(xr: XR, frames):
    last_instruction = None
    for frame in frames:
        image_key, (left_centroid, right_centroid), eye = frame
        error = np.inf
        while error > 0.075:
            hands = xr.hands()

            if left_centroid is not None:
                if hands.left:
                    i_left_centroid = centroid(hands.left)
                    left_error = np.linalg.norm(left_centroid - i_left_centroid)
                    xr.sphere(left_centroid, scale=.025, color=RED, key='_left')
                else:
                    left_error = np.inf
            else:
                left_error = 0

            if right_centroid is not None:
                if hands.right:
                    i_right_centroid = centroid(hands.right)
                    right_error = np.linalg.norm(right_centroid - i_right_centroid)
                    xr.sphere(right_centroid, scale=.025, color=RED, key='_right')
                else:
                    right_error = np.inf
            else:
                right_error = 0

            error = np.max([left_error, right_error])

            xr.display(
                opacity=np.max([error, .85]),
                visible=True,
                eye=eye,
                key=image_key)

            current_instruction = get_stage_caption(image_key)
            if last_instruction != current_instruction:
                last_instruction = current_instruction
                xr.write(current_instruction)

        # frame cleared
        xr.display(
            visible=False,
            key=image_key)


def playback_app(xr: XR):
    xr.write('Loading Demonstration')
    frames = load_demo(xr)
    xr.write('Demonstration ready')
    while True:
        playback(xr, frames)


if __name__ == '__main__':
    run_xr_app(
        playback_app,
        SessionRepositoryLocalFileSystem)
