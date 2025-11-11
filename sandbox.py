from time import sleep

from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp import XR, run_xr_app


def sandbox(xr: XR):
    foo = xr.image()
    foo.as_pil_image().show()
    position = [0] * 3
    for i in range(1):
        xr.write(f'hello {i}')
        position[-1] = i
        xr.sphere(position, scale=1.0, color=(1, 0, 0, 1), key=str(i))
        sleep(.25)
    xr.clear()

    b = xr.bundle(
        xr.eye,
            xr.eye
        )

    print(xr)

    # while True:
    #     hands = xr.hands()
    #     left = hands.left_centroid()
    #     right = hands.right_centroid()
    #     xr.sphere(left, color=(1, 0, 0, 1), key='left')
    #     xr.sphere(right, color=(0, 1, 0, 1), key='right')


if __name__ == '__main__':
    run_xr_app(
        sandbox,
        SessionRepositoryLocalFileSystem)
