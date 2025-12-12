import base64
import mimetypes
import mimetypes
import pathlib
from enum import Enum
from io import BytesIO
from os import PathLike
from typing import Tuple, Optional, ClassVar, IO, Any

import PIL
import PIL.Image as PIL_Image
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, RootModel, field_serializer

from xarp.data_models.encoding import LazyBase64Bytes
from xarp.data_models.spatial import Transform, Pose, Vector3


class MIMEType(str, Enum):
    plain = mimetypes.types_map['.txt']
    png = mimetypes.types_map['.png']
    jpeg = mimetypes.types_map['.jpg']
    mp3 = mimetypes.types_map['.mp3']
    wav = mimetypes.types_map['.wav']
    ogg = 'audio/ogg'
    mp4 = mimetypes.types_map['.mp4']
    glb = 'model/gltf-binary'


class Hands(BaseModel):
    left: Tuple[Pose, ...] = Field(default_factory=tuple)
    right: Tuple[Pose, ...] = Field(default_factory=tuple)

    def __getitem__(self, item):
        if item == 0:
            return self.left
        elif item == 1:
            return self.right
        raise ValueError(f'Invalid hand index {item}')

    def __iter__(self):
        yield self.left
        yield self.right


class Image(BaseModel):
    pixels: LazyBase64Bytes | None = None
    width: int
    height: int
    pil_img_mode: str = 'RGBA'
    path: pathlib.PurePath | None = None

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height

    def load_pixels(self, scale: float = None) -> 'Image':
        img = PIL_Image.open(self.path)
        self.width, self.height = img.size
        if scale is not None:
            img.thumbnail((self.width * scale, self.height * scale))
        self.pixels = LazyBase64Bytes(img.tobytes())
        self.path = None
        return self

    def dump_pixels(self, path: pathlib.PurePath) -> 'Image':
        pil_img = self.to_pil_image()
        with open(path, 'wb') as f:
            pil_img.save(path)
        self.pixels = None
        self.path = path
        return self

    def to_pil_image(self) -> PIL_Image.Image:
        if self.path:
            return PIL_Image.open(self.path)
        temp = BytesIO(self.pixels)
        return PIL_Image.open(temp)

    @classmethod
    def from_pil_image(cls, source: PIL_Image.Image) -> 'Image':
        return Image(
            pixels=LazyBase64Bytes(source.tobytes()),
            pil_img_mode=source.mode,
            height=source.height,
            width=source.width)


class SenseResult(BaseModel):
    eye: Optional[Transform] = None
    head: Optional[Transform] = None
    image: Optional[Image] = None
    depth: Optional[Image] = None
    hands: Optional[Hands] = None


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True)

    focal_length: Tuple[float, float] = None
    principal_point: Tuple[float, float] = None
    sensor_resolution: Tuple[float, float] = None
    lens_offset: Transform

    def to_matrix(self) -> np.ndarray:
        fx, fy = self.focal_length
        cx, cy = self.principal_point

        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        if self.lens_offset is not None:
            du = float(self.lens_offset.position[0])
            dv = float(self.lens_offset.position[1])

            K[0, 2] += du
            K[1, 2] -= dv

        return K

    def world_to_pixel(self,
                       world_point: Vector3,
                       camera_pose: Transform) -> Vector3:
        wp = np.asarray(world_point, dtype=float)
        single_point = (wp.ndim == 1)  # (3,)
        if single_point:
            wp = wp.reshape(1, 3)  # → (1,3)

        # (N,3)
        R_wc = np.asarray(camera_pose.rotation_matrix(), dtype=float).reshape(3, 3)
        C = np.asarray(camera_pose.position, dtype=float).reshape(3)

        R_cw = R_wc.T  # world→camera rotation

        # World → camera
        # (wp - C) → (N,3)
        X_cam = (wp - C) @ R_cw.T  # still (N,3)
        Xc = X_cam[:, 0]
        Yc = X_cam[:, 1]
        Zc = X_cam[:, 2]

        # Perspective divide
        x = Xc / Zc
        y = -Yc / Zc  # flip Y: camera (up) → image (down)

        # Intrinsics
        K = self.to_matrix()  # (3,3)

        # Homogeneous image coords
        ones = np.ones_like(x)
        pts_norm = np.column_stack((x, y, ones))  # (N,3)

        uvw = pts_norm @ K.T  # (N,3)
        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]

        uv = np.column_stack((u, v))  # (N,2)

        return uv[0] if single_point else uv


class DeviceInfo(BaseModel):
    camera_intrinsics: CameraIntrinsics
