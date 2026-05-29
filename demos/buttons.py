from pydantic import PrivateAttr

from xarp import colors
from xarp.data_models import Hands
from xarp.entities import Asset, DefaultAssets, Element
from xarp.express import SyncXR
from xarp.gestures import THUMB_TIP, pinch
from xarp.icons import material_symbol_asset
from xarp.server import make_qrcode_image, run
from xarp.spatial import Quaternion, Transform, Vector3, Vector4


class Button(Element):
    _icon: Element = PrivateAttr()

    @classmethod
    def create(
            cls,
            key: str,
            icon: str = "home",
            *,
            shape: Asset | None = None,
            button_color: Vector4 | tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5),
            icon_color: Vector4 | tuple[float, float, float, float] = colors.WHITE,
            transform: Transform | None = None,
    ) -> "Button":
        if transform is None:
            transform = Transform(
                rotation=Quaternion.from_euler_angles(0, 180, 0),
                scale=Vector3.one() * 0.1,
            )

        button = cls(
            key=key,
            asset=shape or DefaultAssets.cube(),
            color=button_color,
            transform=transform,
        )

        button._icon = Element(
            key=f"icon_{key}",
            asset=material_symbol_asset(icon, 1024, color=icon_color),
            parent=key,
            transform=Transform(position=Vector3.forward() * 0.501),
        )

        return button

    @property
    def icon(self) -> Element:
        return self._icon

    @property
    def elements(self) -> list[Element]:
        return [self, self.icon]


def click_a(xr: SyncXR):
    xr.say('"home"')


def click_b(xr: SyncXR):
    xr.say('"cancel"')


def app(xr: SyncXR, *args, **kwargs):
    xr.destroy_element(all_elements=True)

    button_a = Button.create(
        "button_a",
        icon="home",
        shape=DefaultAssets.cube(),
        button_color=(0.0, 1.0, 0.0, 0.5),
        icon_color=colors.WHITE,
        transform=Transform(
            position=Vector3.forward() * 0.1 + Vector3.left() * .1 + Vector3.up(),
            rotation=Quaternion.from_euler_angles(0, 180, 0),
            scale=Vector3.one() * 0.1,
        ),
    )
    button_b = Button.create(
        "button_b",
        icon="cancel",
        shape=DefaultAssets.sphere(),
        button_color=(1.0, 0.0, 0.0, 0.5),
        icon_color=colors.WHITE,
        transform=Transform(
            position=Vector3.forward() * 0.1 + Vector3.right() * .1 + Vector3.up(),
            rotation=Quaternion.from_euler_angles(0, 180, 0),
            scale=Vector3.one() * 0.1,
        ),
    )
    xr.update(button_a.elements + button_b.elements)

    stream = xr.sense(hands=True)
    for frame in stream:
        hands: Hands = frame["hands"]
        hand = hands.right or hands.left
        if not hand:
            continue

        if pinch(hand):
            cursor: Vector3 = hand[THUMB_TIP].position
            if button_a.transform.position.distance(cursor) < button_a.transform.scale[0] / 2:
                click_a(xr)
                continue
            if button_b.transform.position.distance(cursor) < button_b.transform.scale[0] / 2:
                click_b(xr)
                continue


def app2(xr: SyncXR, *args, **kwargs):
    xr.destroy_element(all_elements=True)

    button = Button.create(
        "foo",
        icon="home",
        shape=DefaultAssets.cube(),
        button_color=(0.0, 0.0, 1.0, 0.5),
        icon_color=colors.BLACK,
        transform=Transform(
            position=Vector3.up() + Vector3.forward() * 0.1,
            rotation=Quaternion.from_euler_angles(0, 180, 0),
            scale=Vector3.one() * 0.1,
        ),
    )
    xr.update(button.elements)

    while True:
        pass


if __name__ == "__main__":
    make_qrcode_image()
    run(app)
