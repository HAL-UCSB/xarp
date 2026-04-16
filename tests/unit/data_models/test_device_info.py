import unittest

from pydantic import ValidationError
from xarp.data_models import CameraIntrinsics, DeviceInfo, Hands
from xarp.spatial import Pose, Quaternion, Vector3

from tests.unit.data_models import make_intrinsics


# ---------------------------------------------------------------------------
# DeviceInfo
# ---------------------------------------------------------------------------

class TestDeviceInfo(unittest.TestCase):

    def setUp(self):
        self.rgb = make_intrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)
        self.depth = make_intrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0,
                                     width=320.0, height=240.0)

    def test_construction(self):
        info = DeviceInfo(rgb_intrinsics=self.rgb, depth_intrinsics=self.depth)
        self.assertEqual(info.rgb_intrinsics, self.rgb)
        self.assertEqual(info.depth_intrinsics, self.depth)

    def test_rgb_and_depth_are_independent(self):
        info = DeviceInfo(rgb_intrinsics=self.rgb, depth_intrinsics=self.depth)
        self.assertNotEqual(info.rgb_intrinsics.focal_length,
                            info.depth_intrinsics.focal_length)

    def test_frozen_mutation_raises(self):
        info = DeviceInfo(rgb_intrinsics=self.rgb, depth_intrinsics=self.depth)
        with self.assertRaises(ValidationError):
            info.rgb_intrinsics = self.depth

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValidationError):
            DeviceInfo(rgb_intrinsics=self.rgb, depth_intrinsics=self.depth,
                       lidar_intrinsics=self.rgb)
