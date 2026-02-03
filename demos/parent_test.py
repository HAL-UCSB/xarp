from xarp.entities import Element, DefaultAssets
from xarp.express import SyncXR
from xarp.server import run, show_qrcode_link
from xarp.spatial import Transform, Vector3


def test(xr: SyncXR, params):
    sun = Element(
        key="sun",
        asset=DefaultAssets.SPHERE,
        transform=Transform(
            scale=Vector3.one() * .2
        ),
        color=(1, 1, 0, 1)

    )

    earth = Element(
        key="earth",
        asset=DefaultAssets.SPHERE,
        transform=Transform(
            position=Vector3.forward(),
            scale=Vector3.one() * .1
        ),
        color=(0, 0, 1, 1)
    )

    def add_child(parent: Element, child: Element):
        child.transform.parent = parent.key
        xr.update(
            Element(
                key=child.key,
                transform=child.transform
            )
        )

    xr.update(sun)
    xr.update(earth)
    add_child(sun, earth)


    stream = xr.sense(head=True)
    for frame in stream:
        head = frame["head"]
        sun.transform.position = head.position
        sun.transform.rotation = head.rotation
        xr.update(sun)


if __name__ == '__main__':
    show_qrcode_link()
    run(test)
