from xarp.xr import XR, run_app
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem


def my_app(xr: XR):
    while True:
        print('frame')
        eye = xr.eye()
        left, right = xr.hands()
        img = xr.image()


if __name__ == '__main__':
    run_app(
        my_app,
        SessionRepositoryLocalFileSystem)
