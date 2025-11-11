from xarp import settings
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp import XR, run_xr_app
from xarp.time import utc_ts


def my_app(xr: XR):
    xr.log_chat = True
    xr.write('Start Recording')
    now = utc_ts()
    n = 10
    for i in range(n):
        print(i)
        a, b, c = xr.bundle(
            xr.image,
            xr.hands,
            xr.eye
        )
    print(n / (utc_ts() - now))
    xr.write('Stop Recording')

    repo = SessionRepositoryLocalFileSystem(settings.local_storage)
    repo.save(xr.session)
    while True:
        pass


if __name__ == '__main__':
    run_xr_app(my_app, SessionRepositoryLocalFileSystem)
