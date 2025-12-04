import base64
import pathlib
from multiprocessing.connection import Client

import numpy as np
from PIL import ImageDraw
from PIL import Image as PIL_Image

from xarp import XR, run_xr_app, Image
from xarp.gestures import double_pinch
from xarp.gestures import pinch, INDEX_TIP
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.time import utc_ts

workspace = r'home/caetano/sam-3d'

sam_3d_obj = Client(('0.0.0.0', 6000), authkey=b'xarp')


def app(xr: XR):
    print('client connected')
    info = xr.info()
    print('info ', info.camera_intrinsics.principal_point)
    image = None
    laso_3d = []

    while True:
        senses = xr.sense(hands=True, eye=True)
        hands = senses.hands
        eye = senses.eye

        if hands.right and pinch(hands.right):
            index_tip_position = hands.right[INDEX_TIP].position
            xr.sphere(
                position=index_tip_position,
                scale=.01,
                key=f'marker_{len(laso_3d)}'
            )
            laso_3d.append(index_tip_position)

        elif hands.left and pinch(hands.left):
            senses = xr.sense(image=True, eye=True)
            image = senses.image.to_pil_image()
            eye = senses.eye

            laso_3d_array = np.vstack(laso_3d)
            laso_2d = info.camera_intrinsics.world_to_pixel(laso_3d_array, eye) + np.array([-10, -100])
            mask = draw_mask(laso_2d, image.size)



            laso_3d.clear()
            xr.clear()
            masked = PIL_Image.alpha_composite(image, mask)
            display_image = Image.from_pil_image(masked)
            xr.display(display_image, eye=eye)

            xr.say('Generating with SAM 3D')

            glb_data = push_to_sam_3d_obj(image, mask)

            xr.say('encoded GLB')
            xr.clear()
            xr.glb(glb_data, list(eye.position))
            xr.say('done!')


def push_to_sam_3d_obj(image, mask):

    input_image_path = pathlib.Path('/' + workspace) / 'input.png'
    input_mask_path = pathlib.Path('/' + workspace) / 'mask.png'

    with input_image_path.open('wb') as f:
        image.save(f, format='PNG')
    with input_mask_path.open('wb') as f:
        mask.save(f, format='PNG')

    print('saved')

    msg = ' '.join([
        input_image_path.absolute().as_posix().replace(workspace, 'workspace'),
        input_mask_path.absolute().as_posix().replace(workspace, 'workspace'),
    ])

    sam_3d_obj.send(msg)
    print('sent path to sam')
    result_path = sam_3d_obj.recv().replace('workspace', workspace)

    print(result_path)

    glb_data = open(result_path, 'rb').read()

    print('read glb')
    return base64.b64encode(glb_data)




def draw_mask(
        points: np.ndarray,
        image_size: tuple[int, int],
        alpha_value: int = 255,
        normalized: bool = False,
        coords_y_up: bool = True) -> PIL_Image:
    width, height = image_size

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be of shape (N, 2)")

    n = pts.shape[0]

    if n == 0:
        img_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        return PIL_Image.fromarray(img_rgba, mode="RGBA")

    if normalized:
        pts = pts.copy()
        pts[:, 0] *= (width - 1)
        pts[:, 1] *= (height - 1)

    pts_int = np.rint(pts).astype(np.int32)

    alpha_value = int(np.clip(alpha_value, 0, 255))

    alpha_img = PIL_Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(alpha_img)

    if n == 1:
        u, v = map(int, pts_int[0])
        if 0 <= u < width and 0 <= v < height:
            alpha_img.putpixel((u, v), alpha_value)
    else:
        pixels = [tuple(uv) for uv in pts_int]
        draw.polygon(pixels, fill=alpha_value)

    alpha_np = np.array(alpha_img, dtype=np.uint8)

    img_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    img_rgba[..., 3] = alpha_np

    return PIL_Image.fromarray(img_rgba)

def app_test_glb(xr: XR):
    glb_data = open(r"C:\Users\Arthur\foo.glb", 'rb').read()
    glb_data = base64.b64encode(glb_data)
    xr.glb(glb_data)


if __name__ == '__main__':
    run_xr_app(
        app,
        SessionRepositoryLocalFileSystem)
