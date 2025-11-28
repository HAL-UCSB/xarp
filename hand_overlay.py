import pyinstrument
import numpy as np

from xarp import settings
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp import XR, run_xr_app


def my_app(xr: XR):
    depth, rgba = xr.bundle(
        xr.depth,
        xr.image)
    print('pass')

    depth = np.array(depth.to_pil_image())
    rgba = np.array(rgba.to_pil_image())

    H, W, _ = rgba.shape
    h, w = depth.shape

    top = (H-h)//2
    left = (W - w) // 2
    bottom = top + h
    right = left + w

    rgba_overlay = rgba[top:bottom, left:right, :]



if __name__ == '__main__':
    run_xr_app(my_app, SessionRepositoryLocalFileSystem)
