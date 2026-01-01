import asyncio
import ssl
import uuid
from io import BytesIO
from datetime import datetime, timezone

import msgpack
import websockets
from pydantic import BaseModel
from websockets import ClientConnection
from PIL import Image

from xarp import run_xr, RemoteXRClient, ImageCommand, ImageResource, ResponseMode
from xarp.commands.assets import Element
from xarp.time import utc_ts


class GenerateParams(BaseModel):
    prompt: str
    width: int = 832
    height: int = 480

    seed: int | None = None
    resume_latents: bytes | None = None
    strength: float = 1.0
    request_id: str | None = None

    interp_blocks: int = -1
    context_noise: float = 0.0
    keep_first_frame: bool = False
    kv_cache_num_frames: int = 3
    num_blocks: int = 9
    num_denoising_steps: int | None = 5

    block_on_frame: bool = False

    input_video: str | None = None
    start_frame: bytes | str | None = None
    timestep_shift: float = 5.0

    webcam_mode: bool = False
    webcam_fps: int = 10

    class Config:
        arbitrary_types_allowed = True


params = GenerateParams(
    prompt="First-person perspective. Working on a workshop.",
    seed=42,
    strength=0.45,
    keep_first_frame=True,
    num_blocks=32,
    num_denoising_steps=5,
    webcam_mode=True,
)

KREA_RT_SERVER_IP = "209.20.159.132"


async def connect_to_krea_rt() -> ClientConnection:
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    krea_uri = f"wss://{KREA_RT_SERVER_IP}:8000/session/{uuid.uuid4()}"
    ws = await websockets.connect(
        krea_uri,
        ssl=ssl_context,
        max_size=None,          # avoid failing on large frames (tune if you want limits)
        ping_interval=20,
        ping_timeout=20,
    )

    handshake = await ws.recv()
    if isinstance(handshake, bytes):
        handshake = handshake.decode("utf-8", "ignore")
    assert "ready" in handshake.lower(), f"Unexpected handshake: {handshake!r}"

    await ws.send(msgpack.packb(params.model_dump(), use_bin_type=True))
    return ws


async def entrypoint(xr: RemoteXRClient):
    # latest-only mailbox: always keep just the newest decoded frame
    latest_frame_q: asyncio.Queue[Image.Image] = asyncio.Queue(maxsize=1)

    async def recv_task(_ws: ClientConnection):
        while True:
            try:
                msg = await _ws.recv()
                if not isinstance(msg, (bytes, bytearray)):
                    continue

                # If server ever sends msgpack-wrapped frames, handle that here.
                # For raw image bytes, this is a no-op.
                img_bytes = bytes(msg)

                pil_image = Image.open(BytesIO(img_bytes))
                pil_image.load()
            except Exception as e:
                print("recv error:", repr(e))
                continue

            # drop old frame if present, keep newest
            if latest_frame_q.full():
                try:
                    latest_frame_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                latest_frame_q.put_nowait(pil_image)
            except asyncio.QueueFull:
                pass

    async def send_task(_xr: RemoteXRClient, target_fps: float = 6.0):
        image_panel = Element(
            key="image_panel",
            distance=0.35,
            response_mode=ResponseMode.NONE,
        )
        await _xr.execute(image_panel)

        min_period = 1.0 / max(target_fps, 0.1)
        last_sent = 0.0

        while True:
            # block until a frame exists (no polling)
            frame = await latest_frame_q.get()

            # optional rate limit: don't render faster than target_fps
            now = asyncio.get_running_loop().time()
            dt = now - last_sent
            if dt < min_period:
                await asyncio.sleep(min_period - dt)

            image_panel.binary = ImageResource.from_image(frame)
            await _xr.execute(image_panel)
            last_sent = asyncio.get_running_loop().time()

    stream_command = ImageCommand(response_mode=ResponseMode.STREAM)
    stream = await xr.execute(stream_command)

    krea_ws = await connect_to_krea_rt()

    asyncio.create_task(recv_task(krea_ws))
    asyncio.create_task(send_task(xr, target_fps=6))

    # reuse buffer to reduce allocations
    buf = BytesIO()

    last = None
    async for image in stream:
        buf.seek(0)
        buf.truncate(0)

        # JPEG encoding is usually your biggest CPU cost on the outbound path.
        # Lower quality helps a lot.
        image._obj.convert("RGB").save(
            buf,
            format="JPEG",
            # quality=70,
            optimize=False,
            # subsampling=2,
        )
        jpeg_bytes = buf.getvalue()

        pack = msgpack.packb(
            {"image": jpeg_bytes, "timestamp": utc_ts(), "strength": params.strength},
            use_bin_type=True,
        )
        await krea_ws.send(pack)

        now = datetime.now(timezone.utc).timestamp()
        if last is not None:
            dt = now - last
            if dt > 0:
                print((1.0 / dt), "FPS")
        last = now


if __name__ == "__main__":
    run_xr(entrypoint)
