import base64
import io
import json
from functools import partial
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from xarp.data_models import ChatMessage, Hands
from xarp.spatial import Transform
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.xr import XR, run_app, settings


def playback_app(xr: XR):
    # Don't log data
    xr.chat_log = False

    # load session
    sessions = SessionRepositoryLocalFileSystem(settings.local_storage)
    session = sessions.get('foo', 1762008729)

    # cache command calls with the session data
    cmds = []
    gt_hand_centroids = []
    eye, hands, encoded_img = None, None, None

    for message in session.chat:

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
                gt_hand_centroids.append((
                    hands.left_centroid(),
                    hands.right_centroid()
                ))
            case 'image':
                img_path = message.content.files[0].as_posix()
                img_path = urlparse(img_path)
                img = Image.open(img_path.path[1:])
                width, height = img.size
                scale = .5
                img.thumbnail((int(width * scale), int(height * scale)))
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                png_bytes = buffer.getvalue()
                encoded_img = base64.b64encode(png_bytes).decode('ascii')

        # prepare a command when we gather eye, hands, and image
        if None not in (eye, hands, encoded_img):
            partial_cmd = partial(
                xr.display_eye,
                encoded_img,
                width,
                height,
                eye=eye)
            cmds.append(partial_cmd)
            eye, hands, encoded_img = None, None, None

    while True:
        frames = iter(zip(cmds, gt_hand_centroids))
        i_cmd, (i_hand_centroid_left, i_hand_centroid_right) = next(frames)
        while True:
            error = np.inf
            while error > .5:
                hands = xr.hands()
                left_error = np.linalg.norm(hands.left_centroid() - i_hand_centroid_left)
                right_error = np.linalg.norm(hands.right_centroid() - i_hand_centroid_right)
                if left_error > right_error:
                    xr.sphere(i_hand_centroid_left, scale=.1, color=(1,0,0,1))
                else:
                    xr.sphere(i_hand_centroid_right, scale=.1, color=(0, 1, 0, 1))
                error = np.max([left_error, right_error])
                opacity = 1.0 / np.clip(error, 0.0, 1.0)
                i_cmd(opacity=opacity)
                print(error)
            i_cmd, (i_hand_centroid_left, i_hand_centroid_right) = next(frames)


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
