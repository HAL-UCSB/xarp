import base64
import io
import json
from functools import partial
from time import sleep
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from xarp.data_models import ChatMessage, Hands
from xarp.spatial import Transform, centroid
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.xr import XR, run_app, settings


def playback_app(xr: XR):
    # load session
    sessions = SessionRepositoryLocalFileSystem(settings.local_storage)
    session = sessions.get('foo', 1762397243)

    # cache command calls with the session data
    cmds = []
    gt_hand_centroids = []
    eye, hands, encoded_img = None, None, None

    chat = sorted(session.chat, key=lambda message: message.ts)

    for message in chat:

        if message.role != ChatMessage.user:
            continue

        cmd = message.content.text[0]
        match cmd:
            case 'eye':
                str_data = message.content.text[1]
                model_dict = json.loads(str_data)
                eye = Transform.model_validate(model_dict)
            case 'hands':
                str_data = message.content.text[1]
                model_dict = json.loads(str_data)
                hands = Hands.model_validate(model_dict)
            case 'image':
                img_path = message.content.files[0].as_posix()
                img_path = urlparse(img_path)
                img = Image.open(img_path.path)
                width, height = img.size
                scale = .2
                img.thumbnail((int(width * scale), int(height * scale)))
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                png_bytes = buffer.getvalue()
                encoded_img = base64.b64encode(png_bytes).decode('ascii')

        # prepare a command when we gather eye, hands, and image
        if None not in (eye, encoded_img):
            if hands.left and hands.right:
                partial_cmd = partial(
                    xr.display,
                    encoded_img,
                    width,
                    height,
                    .4,
                    eye=eye)
                cmds.append(partial_cmd)
                centroids = (
                    centroid(hands.left) if hands.left else None,
                    centroid(hands.right) if hands.right else None,
                )
                gt_hand_centroids.append(centroids)
            eye, hands, encoded_img = None, None, None

    while True:
        frames = iter(zip(cmds, gt_hand_centroids))
        i_cmd, (i_left, i_right) = next(frames)

        while True:
            error = np.inf

            while i_right is not None and error > .1:
                hands = xr.hands()
                if hands.right:
                    right = centroid(hands.right)
                    error = np.linalg.norm(right - i_right) / .8
                i_cmd(opacity=np.max((error, .25)))
                xr.sphere(i_right, scale=.025, color=(1, 1, 1, 1), key='_right')

            i_cmd, (i_left, i_right) = next(frames)


# img = Image.open(r'C:\Users\Arthur\PycharmProjects\xarp\data\foo\1761899811\files\2426695927920.png')
# width, height = img.size
# scale = .2
# img.thumbnail((int(width * scale), int(height * scale)))
# while True:
#     img = xr.image()
#     width, height = img.size
#     scale = .2
#     img.thumbnail((int(width * scale), int(height * scale)))
#     buffer = io.BytesIO()
#     img.save(buffer, format='PNG')
#     png_bytes = buffer.getvalue()
#     encoded = base64.b64encode(png_bytes).decode('ascii')
#     eye = xr.eye()
#     xr.display_eye(encoded, width, height, transparency=alpha, eye=eye)
#
# depth = .425
# while True:
#     img = xr.image()
#     width, height = img.size
#     scale = .1
#     img.thumbnail((int(width * scale), int(height * scale)))
#     buffer = io.BytesIO()
#     img.save(buffer, format='PNG')
#     png_bytes = buffer.getvalue()
#     encoded = base64.b64encode(png_bytes).decode("ascii")
#     xr.display(encoded, width, height, depth)


if __name__ == '__main__':
    run_app(
        playback_app,
        SessionRepositoryLocalFileSystem)
