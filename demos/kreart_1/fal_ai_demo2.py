import base64

import fal_client

from xarp import run_xr_app, AsyncXR
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from PIL import Image as PIL_Image


async def my_app(xr: AsyncXR):
    width = 832
    height = 480

    img = await xr.image()
    image_uri = fal_client.upload_image(img.to_pil_image().convert('RGB'))
    #image_uri = f'data:image/jpeg;base64,{base64.b64encode(img.pixels).decode()}'

    result = fal_client.subscribe(
        'fal-ai/fast-lcm-diffusion/image-to-image',
        arguments={
            "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
            "image_url": image_uri,
            "prompt": "spaceship",
            "negative_prompt": "cartoon, illustration, animation. face. male, female",
            "image_size": dict(
                width=width,
                height=height),
            "num_inference_steps": 6,
            "guidance_scale": 1.5,
            "strength": 0.95,
            "sync_mode": True,
            "num_images": 1,
            "enable_safety_checker": False,
            "format": "jpeg"
        },
        with_logs=False
    )
    print(result)
    base64.b64decode(result['images'][0]['url'])


if __name__ == '__main__':
    run_xr_app(my_app, SessionRepositoryLocalFileSystem)
