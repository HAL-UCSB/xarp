from xarp import run_xr, RemoteXRClient, HandsCommand, ResponseMode, EyeCommand, ImageCommand, \
    SessionRepositoryLocalFileSystem, settings, ChatMessage
from xarp.commands.control import BundleCommand
from xarp.data_models.entities import Session
from xarp.time import utc_ts


async def capture(xr: RemoteXRClient):
    repo = SessionRepositoryLocalFileSystem(settings.local_storage)
    session = Session(user_id='expert')

    stream = await xr.execute(
        BundleCommand(
            subcommands=[
                HandsCommand(),
                EyeCommand(),
                ImageCommand()
            ],
            response_mode=ResponseMode.STREAM,
        )
    )

    i = 0
    try:
        start = utc_ts()
        async for frame in stream:
            session.chat.append(ChatMessage.from_user(frame))
            print(i)
            i += 1
        end = utc_ts()
        print(i / (end - start), 'FPS')
    finally:
        repo.save(session)


if __name__ == '__main__':
    run_xr(capture)
