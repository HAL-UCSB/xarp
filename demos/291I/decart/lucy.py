import asyncio
import ssl
import uuid
from io import BytesIO

import msgpack
import websockets
from websockets import ClientConnection

from xarp import run_xr, RemoteXRClient, ImageCommand, ImageResource, ResponseMode, Vector3, EyeCommand
from xarp.commands.assets import Element
from xarp.commands.control import BundleCommand
from xarp.data_models.spatial import Pose
from xarp.time import utc_ts
from PIL import Image

from decart import DecartClient, models
from decart.realtime import RealtimeClient, RealtimeConnectOptions
from decart.types import ModelState, Prompt


class XRMediaStreamTrack(MediaStreamTrack)

VIDEO_MODEL_SERVER_IP = '152.69.171.158'
PROMPT = """
Chemistry lab. First-person perspective. 
""".strip()


async def connect_to_lucy():
    model = models.realtime("lucy_v2v_720p_rt")

    # Get user's camera stream
    stream = await get_camera_stream(
        audio=True,
        video={
            "frame_rate": model.fps,
            "width": model.width,
            "height": model.height,
        },
    )

    # Create client
    client = DecartClient(api_key="")

    # Connect to realtime API
    realtime_client = await RealtimeClient.connect(
        base_url=client.base_url,
        api_key=client.api_key,
        local_track=stream.video,  # Pass video track
        options=RealtimeConnectOptions(
            model=model,
            on_remote_stream=lambda transformed_stream: (
                # Handle the transformed video in your app
                handle_stream(transformed_stream)
            ),
            initial_state=ModelState(
                prompt=Prompt(text=PROMPT),
            ),
        ),
    )

    # Disconnect when done
    await realtime_client.disconnect()

async def

async def recv_task(ws: ClientConnection, xr: RemoteXRClient):
    try:
        while True:
            received_bytes = await ws.recv()
            msg = msgpack.unpackb(received_bytes)
            buffer = BytesIO(msg['image'])
            pil_image = Image.open(buffer)

            print(msg['request_id'])

            await xr.execute(Element(
                key='krea_rt_panel',
                binary=ImageResource.from_image(pil_image),
                eye=await xr.execute(EyeCommand()),
                distance=.49
            ))
    except Exception as e:
        raise e


async def lucy(xr: RemoteXRClient):
    krea_ws = await connect_to_lucy()
    _recv_task = asyncio.create_task(recv_task(krea_ws, xr))

    buffer = BytesIO()
    stream_command = ImageCommand(
        response_mode=ResponseMode.STREAM
    )

    stream = await xr.execute(stream_command)
    async for image in stream:
        image._obj.convert('RGB').save(buffer, format='JPEG')
        pack = msgpack.packb(dict(
            image=buffer.getvalue(),
            timestamp=utc_ts(),
            strength=DIFFUSION_STRENGTH,
        ))
        await krea_ws.send(pack)


if __name__ == '__main__':
    run_xr(lucy)
