from xarp import run_xr, RemoteXRClient
from xarp.commands import ResponseMode
from xarp.commands.assets import ElementCommand, DefaultAssets
from xarp.commands.control import BundleCommand
from xarp.commands.sensing import HandsCommand, EyeCommand, ImageCommand
from xarp.data_models.spatial import Transform, Vector3
from xarp.time import utc_ts


async def closed_loop(xr: RemoteXRClient):
    left_sphere = 'left_sphere'
    right_sphere = 'right_sphere'
    display = 'display'
    bundle = []
    last_ts = None

    stream_command = BundleCommand(
        subcommands=[
            HandsCommand(),
            EyeCommand(),
            ImageCommand()
        ],
        response_mode=ResponseMode.STREAM,
    )

    stream = await xr.execute(stream_command)

    async for frame in stream:
        bundle.clear()
        hands, eye, image = frame

        now = utc_ts()
        if last_ts is not None and now > last_ts:
            print(1.0 / (now - last_ts), ' FPS')
        last_ts = now

        if hands.left:
            show_left_hand = ElementCommand(
                key=left_sphere,
                asset_key=DefaultAssets.SPHERE,
                active=True,
                transform=Transform(
                    position=hands.left[0].position,
                    scale=Vector3.one() * .05)
            )
            bundle.append(show_left_hand)
        else:
            hide_left_hand = ElementCommand(
                key=left_sphere,
                active=False
            )
            bundle.append(hide_left_hand)

        if hands.right:
            show_right_hand = ElementCommand(
                key=right_sphere,
                asset_key=DefaultAssets.SPHERE,
                active=True,
                transform=Transform(
                    position=hands.right[0].position,
                    scale=Vector3.one() * .05)
            )
            bundle.append(show_right_hand)
        else:
            hide_right_hand = ElementCommand(
                key=right_sphere,
                active=False
            )
            bundle.append(hide_right_hand)

        show_image = ElementCommand(
            key=display,
            binary=image,
            active=True,
            eye=eye,
            distance=.49
        )
        bundle.append(show_image)

        await xr.execute(BundleCommand(subcommands=bundle, response_mode=ResponseMode.NONE))


if __name__ == '__main__':
    run_xr(closed_loop)
