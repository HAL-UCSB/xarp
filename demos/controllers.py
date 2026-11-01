"""Print controller samples from an updated Quest client before testing drawing."""

from xarp.express import SyncXR
from xarp.server import make_qrcode_image, run


def app(xr: SyncXR, *args, **kwargs) -> None:
    print("Single sample:", xr.controllers().model_dump())
    stream = xr.sense(controllers=True)
    try:
        for frame in stream:
            print(frame["controllers"].model_dump())
    finally:
        stream.close()


if __name__ == "__main__":
    make_qrcode_image()
    run(app)
