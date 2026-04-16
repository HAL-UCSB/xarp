from xarp.data_models import CameraIntrinsics, DeviceInfo, Hands
from xarp.spatial import Pose, Quaternion, Vector3


def make_intrinsics(
        fx=500.0, fy=500.0,
        cx=320.0, cy=240.0,
        width=640.0, height=480.0,
        lens_offset=None,
) -> CameraIntrinsics:
    return CameraIntrinsics(
        focal_length=(fx, fy),
        principal_point=(cx, cy),
        sensor_resolution=(width, height),
        lens_offset=lens_offset,
    )


def identity_eye() -> Pose:
    return Pose.identity()
