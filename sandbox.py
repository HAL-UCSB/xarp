from datetime import datetime
from time import sleep

from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp import XR, AsyncXR, run_xr_app
from PIL import Image as PIL_Image


def sandbox_save(xr:XR):
    xr.say('start')
    keys = []
    for i in range(2):
        key = f'key_{i}'
        keys.append(key)
        data = xr.sense(image=True, eye=True)
        xr.display(
            opacity=.7,
            depth=.48725,
            image=data.image,
            eye=data.eye,
            key=key
        )
        xr.save(key)
        sleep(3)
    while True:
        sleep(3)


def sandbox_load(xr:XR):
    xr.say('start')
    for i in range(2):
        key = f'key_{i}'
        xr.load(key)
    while True:
        sleep(3)

def sandbox(xr: XR):
    xr.say('start')
    keys = []
    for i in range(2):
        key = f'key_{i}'
        keys.append(key)
        data = xr.sense(image=True)
        xr.display(
            opacity=.7,
            depth=.48725,
            image=data.image,
            eye=data.eye,
            key=key
        )
        xr.save(key)
        sleep(3)
    exit()


    while True:
        before = datetime.now().timestamp()
        data = xr.sense(
            eye=True,
            image=True)
        after = datetime.now().timestamp()
        print(1 / (after - before), 'FPS')
        scale = 1
        # scale = .5
        # size = (int(data.image.width* scale), int(data.image.height * scale))
        # resized_img = data.image.as_pil_image().transpose(PIL_Image.Transpose.FLIP_TOP_BOTTOM)
        # resized_img.thumbnail(size)
        # data.image.pixels = resized_img.tobytes()
        # data.image.width, data.image.height = size
        xr.display(
            opacity=.7,
            depth=.48725 * scale,
            image=data.image,
            eye=data.eye)

    while True:
        now = datetime.now().timestamp()
        for i in range(100):
            print(i)
            data = xr.sense(
                eye=True,
                hands=True,
                depth=True,
                head=True,
                image=True)
        fps = 100 / (datetime.now().timestamp() - now)
        xr.say(f'{fps:.2} FPS')
        print(fps)


if __name__ == '__main__':
    run_xr_app(
        sandbox_load,
        SessionRepositoryLocalFileSystem)
