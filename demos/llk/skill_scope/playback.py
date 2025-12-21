from xarp import run_xr, RemoteXRClient, SessionRepositoryLocalFileSystem, \
    settings, ResponseMode
from xarp.commands.assets import Element
from xarp.commands.control import BundleCommand
from xarp.data_models.spatial import Transform, Vector3


async def playback(xr: RemoteXRClient):
    repo = SessionRepositoryLocalFileSystem(settings.local_storage)
    session = list(repo.all('expert'))[-1]

    left_sphere = 'left_sphere'
    right_sphere = 'right_sphere'
    display = 'display'

    bundle = []
    for message in session.chat:
        bundle.clear()
        print(message.ts)
        hands, eye, image = message.content

        if hands.left:
            show_left_hand = Element(
                key=left_sphere,
                asset_key='Sphere',
                active=True,
                transform=Transform(
                    position=hands.left[0].position,
                    scale=Vector3.one() * .05),
                response_mode=ResponseMode.NONE
            )
            bundle.append(show_left_hand)
        else:
            hide_left_hand = Element(
                key=left_sphere,
                active=False,
                response_mode=ResponseMode.NONE
            )
            bundle.append(hide_left_hand)

        if hands.right:
            show_right_hand = Element(
                key=right_sphere,
                asset_key='Sphere',
                active=True,
                transform=Transform(
                    position=hands.right[0].position,
                    scale=Vector3.one() * .05),
                response_mode=ResponseMode.NONE
            )
            bundle.append(show_right_hand)
        else:
            hide_right_hand = Element(
                key=right_sphere,
                active=False,
                response_mode=ResponseMode.NONE
            )
            bundle.append(hide_right_hand)

        image.to_memory()
        show_image = Element(
            key=display,
            binary=image,
            active=True,
            eye=eye,
            distance=.49
        )
        bundle.append(show_image)
        await xr.execute(BundleCommand(subcommands=bundle))


if __name__ == '__main__':
    run_xr(playback)
