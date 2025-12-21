from typing import Sequence

from xarp import run_xr, RemoteXRClient, SessionRepositoryLocalFileSystem, \
    settings, ResponseMode, HandsCommand, SayCommand, EyeCommand
from xarp.commands.assets import Element, AssetCommand, ListAssetsCommand, DefaultAssets, DestroyAssetCommand
from xarp.commands.control import BundleCommand
from xarp.commands.ui import PassthroughCommand
from xarp.data_models.binaries import TextResource
from xarp.data_models.entities import Session
from xarp.data_models.spatial import Vector3, distance, Pose, angle_quaternion, Transform, point_to_ray_distance
from xarp.gestures import thumbs_up, pinch


async def cache_demonstration(xr: RemoteXRClient, session: Session):
    asset_list = await xr.execute(ListAssetsCommand())
    if str(session.ts) in asset_list:
        return [f'{session.ts}_{i}' for i in range(len(session.chat))]

    # await xr.execute(DestroyAssetCommand(all=True))
    await xr.execute(SayCommand(text=f'Downloading demonstration {session.ts}.'))
    image_keys = []
    for i, message in enumerate(session.chat):
        image = message.content[-1]
        image_key = f'{session.ts}_{i}'

        await xr.execute(
            AssetCommand(
                asset_key=image_key,
                data=image.to_memory()
            ))

        image_keys.append(image_key)
        print(f'{i / len(session.chat):.2%}')

    await xr.execute(
        AssetCommand(
            asset_key=str(session.ts),
            data=TextResource.from_obj('')
        ))

    await xr.execute(SayCommand(text='Download complete!'))
    return image_keys


async def playback(xr: RemoteXRClient):
    await xr.execute(PassthroughCommand(transparency=.1))

    image_panel = Element(
        key='image_panel',
        distance=.489,
        color=(1, 1, 1, 1),
        response_mode=ResponseMode.NONE
    )

    async def update_scene(_demo_frame):
        _image_key, _demo_hands = _demo_frame
        image_panel.asset_key = _image_key
        await xr.execute(image_panel)

    repo = SessionRepositoryLocalFileSystem(settings.local_storage)
    session = repo.get('expert', 1)
    image_keys = await cache_demonstration(xr, session)

    def demo_generator():
        for _chat_message, _image_key in zip(session.chat, image_keys):
            _demo_hands, _ = _chat_message.content
            yield _image_key, _demo_hands

    demo_iterator = iter(demo_generator())

    tracking = await xr.execute(HandsCommand(response_mode=ResponseMode.STREAM, delay=1 / 30))

    try:
        image_panel.eye = await xr.execute(EyeCommand())
        async for hands in tracking:

            if hands.left and pinch(hands.left):
                image_panel.eye = await xr.execute(EyeCommand())
                await xr.execute(image_panel)

            if hands.left and hands.right:
                _image_key, _demo_hands = next(demo_iterator)
                image_panel.asset_key = _image_key
                await xr.execute(image_panel)

            print(bool(hands.left and hands.right))



    except StopIteration:
        await xr.execute(SayCommand(
            text='Demonstration complete',
            response_mode=ResponseMode.NONE
        ))


async def delete_assets(xr: RemoteXRClient):
    await xr.execute(DestroyAssetCommand(all=True))


if __name__ == '__main__':
    # run_xr(delete_assets)
    run_xr(playback)
