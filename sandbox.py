from time import sleep

from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.xr import XR, run_app

def sandbox(xr: XR):
    xr.chat_log = False

    left, right = xr.hands()
    print(left)

    # while True:
    #     position =  [0] * 3
    #     for i in range (10):
    #         xr.write(f'hello {i}')
    #         position[-1] = i
    #         xr.sphere(position, scale=1.0, color=(1, 0, 0, 1), key=str(i))
    #         sleep(.25)
    #     xr.clear()


    while True:
        hands = xr.hands()
        left = hands.left_centroid()
        right = hands.right_centroid()
        xr.sphere(left, color=(1, 0, 0, 1), key='left')
        xr.sphere(right, color=(0, 1, 0, 1), key='right')


if __name__ == '__main__':
    run_app(
        sandbox,
        SessionRepositoryLocalFileSystem)
