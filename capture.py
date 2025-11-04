import base64
import io

from xarp.spatial import centroid
from xarp.xr import XR, run_app
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem


def my_app(xr: XR):
    xr.chat_log = True
    xr.write('Start Recording')
    for i in range(40):
        print(i)
        img = xr.image()
        hands = xr.hands()
        eye = xr.eye()

        # width, height = img.size
        # scale = .1
        # img.thumbnail((int(width * scale), int(height * scale)))
        # buffer = io.BytesIO()
        # img.save(buffer, format='PNG')
        # png_bytes = buffer.getvalue()
        # encoded_img = base64.b64encode(png_bytes).decode('ascii')
        #
        # xr.display_eye(encoded_img, width, height, opacity=.1, eye=eye)
        # xr.sphere(centroid(hands.left), color=(1, 0, 0, 1), key='_left')
        # xr.sphere(centroid(hands.right), color=(0, 1, 0, 1), key='_right')
        # xr.sphere(eye.position, color=(0, 0, 1, 1), key='_eye')


if __name__ == '__main__':
    run_app(
        my_app,
        SessionRepositoryLocalFileSystem)
