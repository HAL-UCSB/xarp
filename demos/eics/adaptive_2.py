from xarp.entities import Element, DefaultAssets, TextAsset
from xarp.express import SyncXR
from xarp.server import show_qrcode_link, run
from xarp.spatial import Vector3, Transform

instructions = """Instructions\nLorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nNullam viverra massa lorem,\nin pellentesque lectus tristique in."""

def app(xarp: SyncXR, params):
    panel = Element(
        key="panel",
        asset=TextAsset.from_obj(instructions),
    )

    stream = xarp.sense(
        hands=True,
        head=True)

    shift_scale = .2

    for frame in stream:
        hands = frame["hands"]
        head = frame["head"]

        shift = head.ray(.4)
        if hands.right:
            shift += Vector3.left() * shift_scale
        elif hands.left:
            shift += Vector3.right() * shift_scale

        panel.transform.position = shift
        panel.transform.rotation = head.rotation
        xarp.update(panel)


if __name__ == '__main__':
    show_qrcode_link()
    run(app)
