import base64
import io
import json
from functools import partial
from urllib.parse import urlparse

from PIL import Image

from xarp.data_models import ChatMessage, Hands
from xarp.data_models.spatial import Transform, centroid
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.xr import XR, run_xr_app, settings


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

        cmd = message.content.plain[0]
        match cmd:
            case 'eye':
                str_data = message.content.plain[1]
                model_dict = json.loads(str_data)
                eye = Transform.model_validate(model_dict)
            case 'hands':
                str_data = message.content.plain[1]
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
        hands = xr.hands()

        while hands.right:
            rgb, depth = xr.bundle(
                xr.image,
                xr.depth
            )
            pass


if __name__ == '__main__':
    run_xr_app(
        playback_app,
        SessionRepositoryLocalFileSystem)
