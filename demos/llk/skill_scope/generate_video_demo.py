import base64
from pathlib import Path

import cv2
import fal_client
import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from xarp import settings
from xarp.data_models.binaries import ImageResource
from xarp.data_models.data import Hands
from xarp.data_models.entities import Session, ChatMessage
from xarp.data_models.spatial import Vector3, Pose, Quaternion
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.time import utc_ts

USER_ID = 'expert'
TS = 1

FIRST_FRAME_PATH = r'C:\Users\Arthur\PycharmProjects\xarp2\data\expert\0\files\10.png'

PROMPT = """
Generate a first-person perspective video showing the following actions: 
1 - Pick up a wooden plank
2- and cut it in half using the bench band saw
""".strip()


def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])


def source_frames(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield Image.fromarray(frame[:, :, ::-1])
        i += 1
    cap.release()


################################################################
import math
from typing import Tuple, Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, RootModel, ConfigDict

# REQUIRED IMPORTS (kept exactly)
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# Required by mediapipe Tasks guide: mp.Image / mp.ImageFormat
import mediapipe as mp


def _pil_to_mp_image(pil_img: Image.Image) -> mp.Image:
    """
    mediapipe==0.10.31: Tasks detect() expects a mediapipe.Image. :contentReference[oaicite:3]{index=3}
    """
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    rgb = np.asarray(pil_img, dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _world_to_unity(points_world: np.ndarray) -> np.ndarray:
    """
    Map world landmarks to Unity camera-local axes (x right, y up, z forward).

    The docs describe world landmarks as real-world 3D coords (meters) :contentReference[oaicite:4]{index=4}
    but axis directions are not spelled out in a way you can trust blindly across pipelines.
    This mapping is the common practical one for Unity visualization:
      unity = ( x,  y, -z )  OR unity=(x, y, z) depending on your observed depth direction.

    We default to flipping Z to make "toward camera" vs "away" align with Unity +z forward
    in many setups. If your hand appears behind/in front incorrectly, flip the sign back.
    """
    return np.stack([points_world[:, 0], points_world[:, 1], -points_world[:, 2]], axis=1)


def detect_hands_from_pil(
    pil_img: Image.Image,
    *,
    model_path: str,
    num_hands: int = 2,
    scale: float = 1.0,
    min_hand_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Hands:
    """
    PIL -> Hands schema.

    NOTE: No `output_hand_world_landmarks` option exists in mediapipe==0.10.31. :contentReference[oaicite:5]{index=5}
    World landmarks are read from result.hand_world_landmarks. :contentReference[oaicite:6]{index=6}
    """
    mp_image = _pil_to_mp_image(pil_img)

    base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    left: Tuple[Pose, ...] = tuple()
    right: Tuple[Pose, ...] = tuple()

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

        # In 0.10.31, HandLandmarkerResult includes world landmarks. :contentReference[oaicite:7]{index=7}
        if not getattr(result, "hand_world_landmarks", None):
            return Hands(left=left, right=right)

        for i, world_lms in enumerate(result.hand_world_landmarks):
            pts_world = np.array([[lm.x, lm.y, lm.z] for lm in world_lms], dtype=np.float64)  # (21,3)
            pts_unity = _world_to_unity(pts_world) * float(scale)

            poses = tuple(
                Pose(
                    position=Vector3.from_xyz(p[0], p[1], p[2]),
                    rotation=Quaternion.identity(),
                )
                for p in pts_unity
            )

            handed_label: Optional[str] = None
            if getattr(result, "handedness", None) and i < len(result.handedness) and result.handedness[i]:
                handed_label = result.handedness[i][0].category_name  # usually "Left"/"Right" :contentReference[oaicite:8]{index=8}

            if handed_label == "Left":
                left = poses
            elif handed_label == "Right":
                right = poses
            else:
                if not left:
                    left = poses
                elif not right:
                    right = poses

    return Hands(left=left, right=right)



##################################################################


result = r"C:\Users\Arthur\Downloads\b7556179813144fda6e79ba83ddc1b11.mp4"
if result is None:
    result = fal_client.subscribe(
        "fal-ai/veo3.1/image-to-video",
        arguments={
            "prompt": PROMPT,
            "aspect_ratio": "auto",
            "duration": "8s",
            "resolution": "720p",
            "generate_audio": False,
            "image_url": f'data:image/png;base64,{base64.b64encode(Image.open(FIRST_FRAME_PATH).tobytes())}'
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )

session = Session(user_id=USER_ID, ts=TS)

for pil_image in source_frames(result):
    frame = [
        detect_hands_from_pil(
            pil_img=pil_image,
            model_path=r'C:\Users\Arthur\PycharmProjects\xarp2\demos\llk\skill_scope\hand_landmarker.task'),
        ImageResource.from_image(pil_image)
    ]
    session.chat.append(ChatMessage.from_user(frame))

repo = SessionRepositoryLocalFileSystem(settings.local_storage)
repo.save(session)
print('data saved')
