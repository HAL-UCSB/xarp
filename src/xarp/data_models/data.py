import mimetypes
from enum import Enum
from typing import Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from xarp.data_models.spatial import Transform, Pose, Vector3, Quaternion


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


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    focal_length: Tuple[float, float]
    principal_point: Tuple[float, float]
    sensor_resolution: Tuple[float, float]
    lens_offset: Pose | None = None  # interpreted as pixel offset in (cx,cy)

    def _fx_fy_cx_cy(self) -> tuple[float, float, float, float]:
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        if self.lens_offset is not None:
            cx += float(self.lens_offset.position.x)
            cy += float(self.lens_offset.position.y)
        return float(fx), float(fy), float(cx), float(cy)

    @staticmethod
    def _world_to_cam(point_world: Vector3, cam_pose: Pose) -> tuple[float, float, float]:
        R_cw = cam_pose.rotation.to_matrix()   # camera->world
        C = cam_pose.position.to_numpy()
        Xw = point_world.to_numpy()
        x, y, z = (R_cw.T @ (Xw - C)).tolist()
        return float(x), float(y), float(z)

    @staticmethod
    def _intr_to_buffer(
        u_intr: float,
        v_intr: float,
        intr_w: float,
        intr_h: float,
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        # aspect-preserving scale then center-crop to the buffer
        intr_aspect = intr_w / intr_h
        img_aspect = img_w / img_h

        if intr_aspect > img_aspect:
            # scale by height, crop width
            s = img_h / intr_h
            crop_x = 0.5 * (intr_w * s - img_w)
            u = u_intr * s - crop_x
            v = v_intr * s
        else:
            # scale by width, crop height
            s = img_w / intr_w
            crop_y = 0.5 * (intr_h * s - img_h)
            u = u_intr * s
            v = v_intr * s - crop_y

        return np.array([u, v], dtype=float)

    def world_to_pixel(self, point_world: Vector3, camera_pose: Pose) -> np.ndarray:
        """
        World point -> pixel in *intrinsics* pixel space (sensor_resolution grid).
        """
        fx, fy, cx, cy = self._fx_fy_cx_cy()
        x, y, z = self._world_to_cam(point_world, camera_pose)
        if z <= 0.0:
            return np.array([np.nan, np.nan], dtype=float)
        return np.array([fx * (x / z) + cx, cy - fy * (y / z)], dtype=float)

    def world_point_to_panel_pixel(
        self,
        point_world: Vector3,
        eye: Pose,
        image_width: int,
        image_height: int,
        distance: float,
    ) -> np.ndarray:
        """
        World point -> pixel on the *displayed* image buffer drawn on a panel whose center is
        at `eye.position + distance * forward(eye.rotation)` and whose normal is eye forward.

        Uses: ray through point intersects plane z=distance (in eye coords), then intrinsics, then
        aspect-preserving scale + center-crop into the delivered buffer size.
        """
        fx, fy, cx, cy = self._fx_fy_cx_cy()
        intr_w, intr_h = map(float, self.sensor_resolution)

        x, y, z = self._world_to_cam(point_world, eye)
        if z <= 0.0:
            return np.array([np.nan, np.nan], dtype=float)

        # intersect the ray with plane z = distance in eye coords
        t = float(distance) / z
        xp = t * x
        yp = t * y

        # project to intrinsics pixel space (distance cancels)
        u_intr = fx * (xp / float(distance)) + cx
        v_intr = cy - fy * (yp / float(distance))

        return self._intr_to_buffer(u_intr, v_intr, intr_w, intr_h, image_width, image_height)



class DeviceInfo(BaseModel):
    camera_intrinsics: CameraIntrinsics
