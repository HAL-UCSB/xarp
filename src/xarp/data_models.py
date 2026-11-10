from typing import Any, Tuple

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .spatial import Pose, Vector2, Vector3


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


class ControllerButtons(BaseModel):
    """Current held states: primary is X/A, secondary is Y/B (left/right)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary: bool = False
    secondary: bool = False


class Controller(BaseModel):
    """One tracked controller sample with a pose in scene coordinates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pose: Pose
    trigger: float = Field(default=0.0, ge=0.0, le=1.0)
    grip: float = Field(default=0.0, ge=0.0, le=1.0)
    thumbstick: Vector2 = Field(default_factory=Vector2.zero)
    buttons: ControllerButtons = Field(default_factory=ControllerButtons)

    @field_validator("thumbstick")
    @classmethod
    def _validate_thumbstick(cls, value: Vector2) -> Vector2:
        if any(not -1.0 <= axis <= 1.0 for axis in value):
            raise ValueError("Thumbstick axes must be between -1 and 1")
        return value


class Controllers(BaseModel):
    """Left and right controller samples. Untracked sides are None."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left: Controller | None = None
    right: Controller | None = None


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

    @field_validator("lens_offset", mode="before")
    @classmethod
    def _coerce_lens_offset(cls, value: Any) -> Any:
        if isinstance(value, Pose):
            return value.position
        if isinstance(value, dict) and "position" in value:
            return value["position"]
        return value

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
