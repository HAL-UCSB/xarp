from typing import Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from .spatial import Pose, Vector3


class Hands(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )
    left: Tuple[Pose, ...] = Field(default_factory=tuple)
    right: Tuple[Pose, ...] = Field(default_factory=tuple)

    def __iter__(self):
        """Iterate over (left_poses, right_poses) as a pair."""
        yield self.left
        yield self.right

    def __getitem__(self, side: str) -> Tuple[Pose, ...]:
        """hands['left'] or hands['right']."""
        if side == "left":
            return self.left
        elif side == "right":
            return self.right
        raise KeyError(f"Expected 'left' or 'right', got {side!r}")


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    focal_length: Tuple[float, float]
    principal_point: Tuple[float, float]
    sensor_resolution: Tuple[float, float]
    lens_offset: Vector3 | None = None  # pixel offset in (cx, cy); z ignored

    def _fx_fy_cx_cy(self) -> tuple[float, float, float, float]:
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        if self.lens_offset is not None:
            cx += self.lens_offset.x
            cy += self.lens_offset.y
        return float(fx), float(fy), float(cx), float(cy)

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
            u = u_intr * s - 0.5 * (intr_w * s - img_w)
            v = v_intr * s
        else:
            # scale by width, crop height
            s = img_w / intr_w
            u = u_intr * s
            v = v_intr * s - 0.5 * (intr_h * s - img_h)

        return np.array([u, v], dtype=float)

    def world_point_to_panel_pixel(
            self,
            point_world: Vector3,
            eye: Pose,
            image_width: int,
            image_height: int,
    ) -> np.ndarray:
        """
        World point -> pixel on the *displayed* image buffer.

        Projects point_world into eye space using standard pinhole projection,
        then maps to the delivered buffer size via aspect-preserving scale + center-crop.
        """
        fx, fy, cx, cy = self._fx_fy_cx_cy()
        intr_w, intr_h = map(float, self.sensor_resolution)

        local = eye.inverse_transform_point(point_world)
        if local.z <= 0.0:
            return np.array([np.nan, np.nan], dtype=float)

        # Standard pinhole projection. v is flipped because image Y increases downward.
        u_intr = fx * (local.x / local.z) + cx
        v_intr = cy - fy * (local.y / local.z)

        return self._intr_to_buffer(u_intr, v_intr, intr_w, intr_h, image_width, image_height)


class DeviceInfo(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    rgb_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics