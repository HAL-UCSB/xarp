from typing import Sequence

from xarp import run_xr, RemoteXRClient, SessionRepositoryLocalFileSystem, \
    settings, ResponseMode, HandsCommand, SayCommand, EyeCommand
from xarp.commands.assets import Element, AssetCommand, ListAssetsCommand, DefaultAssets, DestroyAssetCommand
from xarp.commands.control import BundleCommand
from xarp.commands.ui import PassthroughCommand
from xarp.data_models.binaries import TextResource
from xarp.data_models.entities import Session
from xarp.data_models.spatial import Vector3, distance, Pose, angle_quaternion, Transform, point_to_ray_distance


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


def calculate_pose_similarity(a: Pose, b: Pose, max_distance=1.5, max_rotation=180) -> float:
    rotation_score = angle_quaternion(a.rotation, b.rotation) / max_rotation
    position_score = distance(a.position, b.position) / max_distance
    return 1 - (rotation_score * .3 + position_score * .7)


def is_hand_visible(hand: Sequence[Pose], eye: Pose, visibility_threshold=.35) -> bool:
    return point_to_ray_distance(hand[0].position, eye) < visibility_threshold


async def reenacted_playback(xr: RemoteXRClient):
    await xr.execute(PassthroughCommand(transparency=.1))

    left_marker = Element(
        key='left_sphere',
        asset_key=DefaultAssets.SPHERE,
        color=(.5, .5, .5, 1),
        response_mode=ResponseMode.NONE
    )

    right_marker = Element(
        key='right_sphere',
        asset_key=DefaultAssets.SPHERE,
        color=(.5, .5, .5, 1),
        response_mode=ResponseMode.NONE
    )

    image_panel = Element(
        key='image_panel',
        distance=.489,
        color=(1, 1, 1, 1),
        response_mode=ResponseMode.NONE
    )

    scene = BundleCommand(
        subcommands=[
            left_marker,
            right_marker,
            image_panel
        ],
        response_mode=ResponseMode.NONE
    )

    async def update_scene(_demo_frame):
        _image_key, _demo_hands, _demo_eye = _demo_frame

        left_marker.active = bool(_demo_hands.left and is_hand_visible(_demo_hands.left, _demo_eye))
        if left_marker.active:
            left_marker.transform = Transform(
                position=_demo_hands.left[0].position,
                rotation=_demo_hands.left[0].rotation,
                scale=Vector3.one() * .1
            )

        right_marker.active = bool(_demo_hands.right and is_hand_visible(_demo_hands.right, _demo_eye))
        if right_marker.active:
            right_marker.transform = Transform(
                position=_demo_hands.right[0].position,
                rotation=_demo_hands.right[0].rotation,
                scale=Vector3.one() * .1
            )

        image_panel.asset_key = _image_key
        image_panel.eye = _demo_eye

        await xr.execute(scene)

    repo = SessionRepositoryLocalFileSystem(settings.local_storage)
    session = repo.get('expert', 1766259602)
    image_keys = await cache_demonstration(xr, session)

    def demo_generator():
        for _image_key, _chat_message in zip(image_keys, session.chat):
            _demo_hands, _demo_eye, _ = _chat_message.content
            yield _image_key, _demo_hands, _demo_eye

    demo_iterator = iter(demo_generator())

    similarity_range = set()
    similarity_threshold = .9
    similarity_score = None

    tracking = await xr.execute(BundleCommand(
        subcommands=[
            HandsCommand(),
            EyeCommand()],
        response_mode=ResponseMode.STREAM,
        delay=1 / 6
    ))

    try:
        demo_hands = None
        demo_eye = None
        async for hands, eye in tracking:

            if similarity_score is None or similarity_score > similarity_threshold:
                demo_frame = next(demo_iterator)
                _, demo_hands, demo_eye = demo_frame
                await update_scene(demo_frame)

            scores = []

            if left_marker.active:
                score = calculate_pose_similarity(hands.left[0], demo_hands.left[0]) if hands.left else 0
                scores.append(score)

            if right_marker.active:
                score = calculate_pose_similarity(hands.right[0], demo_hands.right[0]) if hands.right else 0
                scores.append(score)

            if not left_marker.active and not right_marker.active:
                score = 1 - angle_quaternion(eye.rotation, demo_eye.rotation) / 360.0
                scores.append(score)

            similarity_score = sum(scores) / len(scores)
            transparency = similarity_score ** 2
            similarity_range.add(similarity_score)
            print(similarity_score)

            left_marker.color = (.5, .5, .5, 1 - transparency - .5)
            right_marker.color = (.5, .5, .5, 1 - transparency - .5)
            image_panel.color = (1, 1, 1, transparency * .75)

            await xr.execute(scene)



    except StopIteration:
        await xr.execute(SayCommand(
            text='Demonstration complete',
            response_mode=ResponseMode.NONE
        ))
    finally:
        if similarity_range:
            print('similarity range: ', min(similarity_range), ' ', max(similarity_range))


async def delete_assets(xr: RemoteXRClient):
    await xr.execute(DestroyAssetCommand(all=True))


if __name__ == '__main__':
    # run_xr(delete_assets)
    run_xr(reenacted_playback)
