from xarp.entities import Element, DefaultAssets, ImageAsset
from xarp.express import SyncXR
from xarp.server import show_qrcode_link, run
from xarp.spatial import Pose


def adapt(panel: Element, depth: ImageAsset, eye: Pose, distance=1):
    pass



def app(xarp: SyncXR, params):
    # Rectangle placed 1 meter away from the user

    panel = Element(
        key="panel",
        distance=1,
        asset=DefaultAssets.Quad)

    # Stream depth images and camera positions
    stream = xarp.sense(
        depth=True,
        eye=True)

    # Iterate over the frames in the stream
    for frame in stream:
        adapt(
            panel,
            frame["depth"],
            frame["eye"])
        # Update the panel on the client
        xarp.update(panel)


if __name__ == '__main__':
    show_qrcode_link()
    run()
