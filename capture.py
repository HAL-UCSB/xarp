import base64
import pathlib

from PIL import Image

from xarp import settings
from xarp.data_models import ChatMessage, Hands
from xarp.spatial import Transform
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.xr import XR, run_app, XRCommand


def my_app(xr: XR):
    xr.write('Start Recording')
    for i in range(20):
        print(i)
        xr.bundle(
            xr.image,
            xr.hands,
            xr.eye
        )
    xr.write('Stop Recording')

    files_path = pathlib.Path(settings.local_storage / xr.session.user_id / str(xr.session.ts) / 'files')
    files_path.absolute().mkdir(parents=True)

    for entry in xr.logs:

        if isinstance(entry, XRCommand):
            entry_json = entry.model_dump_json()
            message = ChatMessage.from_system(entry_json)
            xr.session.chat.append(message)
            continue
        else:
            if entry is None:
                continue
            if isinstance(entry, list):
                for entry_item in entry:
                    if 'pixels' in entry_item:
                        img_path = files_path / f'{id(entry_item)}.png'
                        pixels = base64.b64decode(entry_item['pixels'])
                        pil_img = Image.frombytes('RGBA', (entry_item['width'], entry_item['height']), pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                        pil_img.save(str(img_path))

                        message = ChatMessage.from_user('image', files=[img_path])
                        xr.session.chat.append(message)
                        continue
                    if 'position' in entry_item:
                        model = Transform.model_validate(entry_item)
                        model_json = model.model_dump_json()
                        message = ChatMessage.from_user('eye', model_json)
                        xr.session.chat.append(message)
                        continue
                    if 'left' in entry_item or 'right' in entry_item:
                        model = Hands.model_validate(entry_item)
                        model_json = model.model_dump_json()
                        message = ChatMessage.from_user('hands', model_json)
                        xr.session.chat.append(message)
                        continue

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
