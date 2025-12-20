from xarp import run_xr, RemoteXRClient, HandsCommand, ResponseMode, EyeCommand, ImageCommand, \
    SessionRepositoryLocalFileSystem, settings, ChatMessage, SayCommand, ImageResource
from xarp.commands.assets import Element
from xarp.commands.control import BundleCommand
from xarp.data_models.entities import Session
from xarp.gestures import coarse_grab
from xarp.time import utc_ts


async def capture(xr: RemoteXRClient):
    tracking = await xr.execute(BundleCommand(
        subcommands=[
            HandsCommand(),
            EyeCommand()
        ],
        response_mode=ResponseMode.STREAM))

    image: ImageResource | None = None
    path = []

    async for hands, eye in tracking:
        if hands.left:
            path.clear()
            image: ImageResource = await xr.execute(ImageCommand())
            await xr.execute(Element(
                key='panel',
                binary=image.to_memory(),
                eye=eye,
                distance=.49
            ))
            continue

        if image is not None and hands.right and coarse_grab(hands.right):






if __name__ == '__main__':
    run_xr(capture)
