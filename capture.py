from xarp import settings
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp import XR, run_xr_app
from xarp.time import utc_ts


def my_app(xr: XR):
    xr.log_chat = True
    xr.write('Start Recording')
    i = 0
    now = utc_ts()
    try:
        while True:
            a, b, c = xr.bundle(
                xr.image,
                xr.hands,
                xr.eye
            )
            print(i)
            i += 1
    finally:
        print(i / (utc_ts() - now))
        repo = SessionRepositoryLocalFileSystem(settings.local_storage)
        repo.save(xr.session)

    while True:
        pass


if __name__ == '__main__':
    run_xr_app(my_app, SessionRepositoryLocalFileSystem)
