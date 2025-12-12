from xarp import run_xr, RemoteXRClient, HandsCommand, ResponseMode
from xarp.commands.assets import ElementCommand, DefaultAssets, DestroyElementCommand
from xarp.data_models.spatial import Transform, Vector3
from xarp.gestures import pinch, INDEX_TIP


async def brush(xr: RemoteXRClient):
    i = 0

    stream = await xr.execute(
        HandsCommand(response_mode=ResponseMode.STREAM)
    )

    async  for hands in stream:

        # Draw
        if hands.right and pinch(hands.right):
            i += 1
            point = hands.right[INDEX_TIP]
            await xr.execute(
                ElementCommand(
                    asset_key=DefaultAssets.SPHERE,
                    key=str(i),
                    transform=Transform(
                        position=point.position,
                        scale=Vector3.one() * .05
                    ),
                )
            )

        # Clean
        if hands.left and pinch(hands.left):
            i = 0
            await xr.execute(
                DestroyElementCommand(
                    all=True
                )
            )


if __name__ == '__main__':
    run_xr(brush)

