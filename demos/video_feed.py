from xarp.entities import Element
from xarp.express import SyncXR
from xarp.server import run, make_qrcode_image
from xarp.spatial import Pose

panel = Element(
    key="panel"
)


def depth_video(xr: SyncXR, *args, **kwargs):
    stream = xr.sense(depth=True, head=True)
    for frame in stream:
        panel.asset = frame["depth"]
        head: Pose = frame["head"]
        panel.transform.position = head.ray_point(.49)
        panel.transform.rotation = head.rotation
        xr.update(panel)
    stream.close()

def rgb_video(xr: SyncXR, *args, **kwargs):
    stream = xr.sense(image=True, eye=True)
    for frame in stream:
        panel.asset = frame["image"]
        head: Pose = frame["eye"]
        panel.transform.position = head.ray_point(.49)
        panel.transform.rotation = head.rotation
        xr.update(panel)
    stream.close()


if __name__ == '__main__':
    make_qrcode_image()
    run(rgb_video)
