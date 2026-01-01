import asyncio
import json
import threading
from io import BytesIO
from typing import AsyncGenerator

import numpy as np
from PIL import Image
import aiohttp
import av
from PIL import ImageDraw
from decart import DecartClient, models

from xarp import run_xr, RemoteXRClient, ResponseMode, HandsCommand, EyeCommand, ImageCommand, DeviceInfo, \
    ImageResource, SayCommand
from xarp.commands.assets import Element, DestroyAssetCommand
from xarp.commands.control import BundleCommand, InfoCommand
from xarp.data_models.data import CameraIntrinsics
from xarp.data_models.spatial import Pose, Vector3, Quaternion
from xarp.gestures import pinch_middle, fist, MIDDLE_METACARPAL, MIDDLE_PROXIMAL


async def mp4_bytes_to_pil_frames(mp4_bytes: bytes) -> AsyncGenerator[Image.Image, None]:
    q = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def decode():
        try:
            c = av.open(BytesIO(mp4_bytes))
            s = c.streams.video[0]
            for f in c.decode(s):
                loop.call_soon_threadsafe(q.put_nowait, f.to_image())
            c.close()
            loop.call_soon_threadsafe(q.put_nowait, None)
        except BaseException as e:
            loop.call_soon_threadsafe(q.put_nowait, e)

    threading.Thread(target=decode, daemon=True).start()

    while True:
        x = await q.get()
        if x is None:
            return
        if isinstance(x, BaseException):
            raise x
        yield x


async def lucy_motion(image, trajectory, api_key):
    buf = BytesIO()
    image._obj.save(buf, "PNG")
    buf.seek(0)

    form = aiohttp.FormData()
    form.add_field("trajectory", json.dumps(trajectory))
    form.add_field("data", buf, filename="i.png", content_type="image/png")

    async with aiohttp.ClientSession() as s:
        r = await s.post(
            "https://api.decart.ai/v1/jobs/lucy-motion",
            headers={"x-api-key": api_key},
            data=form,
        )
        job = await r.json()
        jid = job["job_id"]

        while True:
            r = await s.get(
                f"https://api.decart.ai/v1/jobs/{jid}",
                headers={"x-api-key": api_key},
            )
            if (await r.json())["status"] == "completed":
                break
            await asyncio.sleep(1)

        r = await s.get(
            f"https://api.decart.ai/v1/jobs/{jid}/content",
            headers={"x-api-key": api_key},
        )
        mp4 = await r.read()

    with open("lucy_motion.mp4", "wb") as f:
        f.write(mp4)

    async for frame in mp4_bytes_to_pil_frames(mp4):
        yield frame


async def motion_control(image: ImageResource, trajectory):
    image_file = BytesIO()
    image._obj.save(image_file, format='PNG')
    image_file.seek(0)

    async with DecartClient(api_key="dev_RebGKbuoDWDVGjOaBsuSgpHfVqMwjIHTwpRDOZYcWGeXuwUtNQnWWMyjWSxaJMEG") as client:
        token = await client.tokens.create()

        result = await client.queue.submit_and_poll({
            "model": models.video("lucy-motion"),
            "data": image_file,
            "trajectory": trajectory,
            "on_status_change": lambda job: print(f"Status: {job.status}"),
        })

        if result.status == "completed":
            with open("output.mp4", "wb") as f:
                f.write(result.data)

    image_file.close()


def draw_path(img, points: list[tuple[float, float]], inplace=True, color=(255, 0, 0), width=10) -> Image.Image:
    if len(points) < 2:
        return img if inplace else img.copy()

    out = img if inplace else img.copy()
    ImageDraw.Draw(out).line(xy=points, fill=color, width=width, joint="curve")
    return out


def draw_blob_center(img, point, inplace=True, color=(255, 0, 0), radius=10):
    out = img if inplace else img.copy()
    x, y = point

    ImageDraw.Draw(out).ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color
    )
    return out


async def app(xr: RemoteXRClient):
    info: DeviceInfo = await xr.execute(InfoCommand())

    panel = Element(
        key='panel',
        distance=.498,
        color=(1, 1, 1, .85),
        response_mode=ResponseMode.NONE
    )
    await xr.execute(panel)

    tracking = await xr.execute(BundleCommand(
        subcommands=[
            HandsCommand(),
            EyeCommand(),
        ],
        delay=1 / 4,
        response_mode=ResponseMode.STREAM
    ))

    path = []
    picture = None
    width = None
    height = None
    async for hands, eye in tracking:

        # Take a picture
        if hands.left and pinch_middle(hands.left):
            path.clear()
            picture = await xr.execute(ImageCommand())
            width, height = picture._obj.size
            panel.binary = ImageResource.from_image(picture._obj.copy())
            panel.eye = eye
            await xr.execute(panel)
            continue

        if picture is None:
            continue

        if hands.right:
            cursor = info.camera_intrinsics.world_point_to_panel_pixel(
                hands.right[0].position,
                panel.eye,
                width,
                height,
                panel.distance).tolist()

            # pinch to draw the path
            if fist(hands.right):

                path.append(cursor)
                # if len(path) < 2:
                #     path.append(cursor)
                # else:
                #     path[1] = cursor

                draw_path(
                    panel.binary._obj,
                    path,
                    color=(0, 255, 0),
                    inplace=True)
                await xr.execute(panel)
                continue

            # release to end the path and generate
            elif len(path) > 1:
                await xr.execute(SayCommand(text='Generating...', response_mode=ResponseMode.NONE))

                trajectory = [dict(frame=i * 10, x=pixel[0] / width, y=pixel[1] / height) for i, pixel in
                              enumerate(path)]
                path.clear()

                first_frame = True
                async for frame in lucy_motion(picture, trajectory,
                                               'dev_RebGKbuoDWDVGjOaBsuSgpHfVqMwjIHTwpRDOZYcWGeXuwUtNQnWWMyjWSxaJMEG'):
                    if first_frame:
                        first_frame = False
                        await xr.execute(SayCommand(text='Ready!', response_mode=ResponseMode.SINGLE))
                    panel.binary = ImageResource.from_image(frame)
                    await xr.execute(panel)

                picture = ImageResource.from_image(panel.binary._obj.copy())
                width, height = picture._obj.size
                path.clear()
                continue

            # just hovering
            hover_canvas = draw_blob_center(
                picture._obj,
                cursor,
                color=(0, 0, 255),
                inplace=False)
            panel.binary = ImageResource.from_image(hover_canvas)
            await xr.execute(panel)


if __name__ == '__main__':
    run_xr(app)
