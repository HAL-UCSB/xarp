from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.xr import XR, run_xr_app


def my_app(xr: XR):
    i = 0
    while True:
        hands = xr.hands()
        if hands is None:
            continue
        if hands.right:
            joint_position = hands.right[0].position
            xr.SPHERE(joint_position, key=str(i))
            i += 1
        if hands.left:
            i = 0
            xr.clear()


if __name__ == '__main__':
    run_xr_app(
        my_app,
        SessionRepositoryLocalFileSystem)
