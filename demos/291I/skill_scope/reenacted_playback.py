from xarp import run_xr, RemoteXRClient, SessionRepositoryLocalFileSystem, \
    settings, ResponseMode, EyeCommand, HandsCommand, SayCommand
from xarp.commands.assets import ElementCommand, AssetCommand, DestroyAssetCommand, ListAssetsCommand, DefaultAssets
from xarp.commands.control import BundleCommand
from xarp.data_models.data import MIMEType
from xarp.data_models.spatial import Vector3, distance, Pose, angle_quaternion, Transform
from xarp.data_models.entities import Session


def pose_similarity(a: Pose, b: Pose, max_distance=2.0) -> float:
    rotation_score = angle_quaternion(a.rotation, b.rotation) / 360.0
    position_score = distance(a.position, b.position) / max_distance
    return (rotation_score + position_score) / 2


async def download_image_assets(xr: RemoteXRClient, session: Session):
    say_command = SayCommand(
        text='Downloading demonstration',
        response_mode=ResponseMode.NONE
    )
    await xr.execute(say_command)

    image_keys = []
    i = 0
    for message in session.chat:
        image = message.content[-1]
        image_key = f'f_{i}'
        i += 1
        image_keys.append(image_key)
        create_image_asset = AssetCommand(
            asset_key=image_key,
            data=image.to_memory()
        )
        await xr.execute(create_image_asset)
    return image_keys


async def reenacted_playback(xr: RemoteXRClient):
    # await xr.execute(DestroyAssetCommand(all=True))
    similarity_threshold = .01

    repo = SessionRepositoryLocalFileSystem(settings.local_storage)
    session = list(repo.all('expert'))[-1]

    image_keys = await xr.execute(ListAssetsCommand())
    if not image_keys:
        image_keys = await download_image_assets(xr, session)

    tracking_command = BundleCommand(
        subcommands=[
            HandsCommand(),
            EyeCommand()
        ],
        response_mode=ResponseMode.STREAM,
    )

    left_hand_sphere = ElementCommand(
        key='left_hand_sphere',
        asset_key=DefaultAssets.SPHERE,
        active=False,
    )

    right_hand_sphere = ElementCommand(
        key='right_hand_sphere',
        asset_key=DefaultAssets.SPHERE,
        active=False,
    )

    image_display = ElementCommand(
        key='image_display',
        asset_key=image_keys[0])

    scene = BundleCommand(
        subcommands=[
            left_hand_sphere,
            right_hand_sphere,
            image_display
        ],
        response_mode=ResponseMode.NONE)

    def demo_frame_generator():
        for _image_key, _chat_message in zip(image_keys, session.chat):
            _demo_hands, _demo_eye, _ = _chat_message.content
            yield _image_key, _demo_hands, _demo_eye

    demo_ite = iter(demo_frame_generator())
    tracking = await xr.execute(tracking_command)

    image_key, demo_hands, demo_eye = next(demo_ite)

    left_hand_sphere.active = True
    left_hand_sphere.transform = Transform(
        position=demo_hands.left[0].position,
        rotation=demo_hands.left[0].rotation,
        scale=Vector3.one() * .1
    )

    right_hand_sphere.active = True
    right_hand_sphere.transform = Transform(
        position=demo_hands.right[0].position,
        rotation=demo_hands.right[0].rotation,
        scale=Vector3.one() * .1
    )

    image_display.active = True
    image_display.asset_key = image_key
    image_display.eye = demo_eye
    image_display.distance = .49

    await xr.execute(scene)

    try:

        async for hands, eye in tracking:

            similarity_score = 0

            if demo_hands.left and hands.left:
                similarity_score = pose_similarity(hands.left[0], demo_hands.left[0])

            if demo_hands.right and hands.right:
                similarity_score += pose_similarity(hands.right[0], demo_hands.right[0])

            if demo_eye and eye:
                similarity_score += pose_similarity(demo_eye, eye)

            similarity_score /= 3
            print(similarity_score)

            if similarity_score < similarity_threshold:
                image_key, demo_hands, demo_eye = next(demo_ite)
                left_hand_sphere.active = True
                left_hand_sphere.transform = Transform(
                    position=demo_hands.left[0].position,
                    rotation=demo_hands.left[0].rotation,
                    scale=Vector3.one() * .1
                )

                right_hand_sphere.active = True
                right_hand_sphere.transform = Transform(
                    position=demo_hands.right[0].position,
                    rotation=demo_hands.right[0].rotation,
                    scale=Vector3.one() * .1
                )

                image_display.active = True
                image_display.asset_key = image_key
                image_display.eye = demo_eye
                image_display.distance = .49

                await xr.execute(scene)



    except StopIteration:
        say_command = SayCommand(
            text='Demonstration complete',
            response_mode=ResponseMode.NONE
        )
        await xr.execute(say_command)


if __name__ == '__main__':
    run_xr(reenacted_playback)
