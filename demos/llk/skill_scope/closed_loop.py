from xarp import run_xr, RemoteXRClient
from xarp.commands import ResponseMode
from xarp.commands.assets import Element, DefaultAssets
from xarp.commands.control import BundleCommand
from xarp.commands.sensing import HandsCommand, EyeCommand, ImageCommand, HeadCommand
from xarp.data_models.spatial import Transform, Vector3
from xarp.time import utc_ts


async def closed_loop(xr: RemoteXRClient):
    left_marker = Element(
        key='left_marker',
        asset_key=DefaultAssets.SPHERE,
        color=(1, 0, 0, 1),
    )

    right_marker = Element(
        key='right_marker',
        asset_key=DefaultAssets.SPHERE,
        color=(0, 1, 0, 1),
    )

    image_panel = Element(
        key='image_panel',
        distance=.489
    )

    scene = BundleCommand(
        subcommands=[
            left_marker,
            right_marker,
            image_panel
        ],
        response_mode=ResponseMode.NONE
    )

    tracking_command = BundleCommand(
        subcommands=[
            HandsCommand(),
            EyeCommand(),
            ImageCommand(),
        ],
        response_mode=ResponseMode.STREAM
    )

    tracking = await xr.execute(tracking_command)

    last_ts = None
    try:
        async for frame in tracking:
            hands, eye, image = frame

            now = utc_ts()
            if last_ts is not None and now > last_ts:
                pass
                print(1.0 / (now - last_ts), ' FPS')
            last_ts = now

            left_marker.active = bool(hands.left)
            if left_marker.active:
                left_marker.transform = Transform(
                    position=hands.left[0].position,
                    scale=Vector3.one() * .05)

            right_marker.active = bool(hands.right)
            if right_marker.active:
                right_marker.transform = Transform(
                    position=hands.right[0].position,
                    scale=Vector3.one() * .05)

            image_panel.binary = image
            image_panel.eye = eye

            await xr.execute(scene)
    finally:
        await tracking.aclose()


if __name__ == '__main__':
    run_xr(closed_loop)
