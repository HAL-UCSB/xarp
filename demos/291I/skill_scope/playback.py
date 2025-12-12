from xarp import run_xr, RemoteXRClient, HandsCommand, ResponseMode, EyeCommand, SessionRepositoryLocalFileSystem, \
    settings, ChatMessage
from xarp.commands.assets import ElementCommand, DefaultAssets
from xarp.commands.control import BundleCommand
from xarp.data_models.data import MIMEType
from xarp.time import utc_ts
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
        hands, eye, image = message.content

        if hands.left:
            bundle.append(
                ElementCommand(
                    key=left_sphere,
                    asset_key='Sphere',
                    active=True,
                    transform=Transform(
                        position=hands.left[0].position,
                        scale=Vector3.one() * .05)
                ))
        else:
            bundle.append(
                ElementCommand(
                    key=left_sphere,
                    active=False,
                ))

        if hands.right:
            bundle.append(
                ElementCommand(
                    key=right_sphere,
                    asset_key='Sphere',
                    active=True,
                    transform=Transform(
                        position=hands.right[0].position,
                        scale=Vector3.one() * .05)
                ))
        else:
            bundle.append(
                ElementCommand(
                    key=right_sphere,
                    active=False,
                ))

        image.to_memory()
        bundle.append(
            ElementCommand(
                key=display,
                binary=image,
                active=True,
                eye=eye,
                distance=.4
            )
        )

        await xr.execute(BundleCommand(subcommands=bundle))

    # stream = await xr.execute(
    #     BundleCommand(
    #         subcommands=[
    #             HandsCommand(),
    #             EyeCommand()
    #         ],
    #         response_mode=ResponseMode.STREAM,
    #     )
    # )


if __name__ == '__main__':
    run_xr(playback)
