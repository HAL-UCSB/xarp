import unittest

import numpy as np
from pydantic import ValidationError
from xarp.data_models import CameraIntrinsics, DeviceInfo, Hands
from xarp.spatial import Pose, Quaternion, Vector3

from tests.unit.data_models import make_intrinsics, identity_eye


# ---------------------------------------------------------------------------
# CameraIntrinsics — _fx_fy_cx_cy
# ---------------------------------------------------------------------------

class TestFxFyCxCy(unittest.TestCase):

    def test_no_offset_returns_principal_point_unchanged(self):
        intr = make_intrinsics(fx=600.0, fy=601.0, cx=320.0, cy=240.0)
        fx, fy, cx, cy = intr._fx_fy_cx_cy()
        self.assertAlmostEqual(fx, 600.0)
        self.assertAlmostEqual(fy, 601.0)
        self.assertAlmostEqual(cx, 320.0)
        self.assertAlmostEqual(cy, 240.0)

    def test_lens_offset_adds_to_cx_cy(self):
        intr = make_intrinsics(cx=320.0, cy=240.0,
                               lens_offset=Vector3(10.0, -5.0, 0.0))
        _, _, cx, cy = intr._fx_fy_cx_cy()
        self.assertAlmostEqual(cx, 330.0)
        self.assertAlmostEqual(cy, 235.0)

    def test_lens_offset_z_has_no_effect(self):
        intr_no_z = make_intrinsics(cx=320.0, cy=240.0,
                                    lens_offset=Vector3(10.0, 5.0, 0.0))
        intr_with_z = make_intrinsics(cx=320.0, cy=240.0,
                                      lens_offset=Vector3(10.0, 5.0, 99.0))
        self.assertEqual(intr_no_z._fx_fy_cx_cy(), intr_with_z._fx_fy_cx_cy())

    def test_zero_offset_equivalent_to_no_offset(self):
        intr_none = make_intrinsics(cx=320.0, cy=240.0)
        intr_zero = make_intrinsics(cx=320.0, cy=240.0,
                                    lens_offset=Vector3(0.0, 0.0, 0.0))
        self.assertEqual(intr_none._fx_fy_cx_cy(), intr_zero._fx_fy_cx_cy())


# ---------------------------------------------------------------------------
# CameraIntrinsics — _intr_to_buffer
# ---------------------------------------------------------------------------

class TestIntrToBuffer(unittest.TestCase):

    def _buf(self, u, v, intr_w, intr_h, img_w, img_h):
        return CameraIntrinsics._intr_to_buffer(u, v, intr_w, intr_h, img_w, img_h)

    def test_center_maps_to_buffer_center_same_resolution(self):
        uv = self._buf(320.0, 240.0, 640.0, 480.0, 640, 480)
        np.testing.assert_allclose(uv, [320.0, 240.0])

    def test_center_maps_to_buffer_center_after_width_crop(self):
        # intr wider than buffer → height-scale path
        uv = self._buf(400.0, 300.0, 800.0, 600.0, 640, 480)
        np.testing.assert_allclose(uv, [320.0, 240.0])

    def test_center_maps_to_buffer_center_after_height_crop(self):
        # intr taller than buffer → width-scale path
        uv = self._buf(320.0, 400.0, 640.0, 800.0, 640, 480)
        np.testing.assert_allclose(uv, [320.0, 240.0])

    def test_wide_intrinsic_uses_height_scale_path(self):
        # intr_aspect > img_aspect → scale by height
        uv = self._buf(0.0, 0.0, 800.0, 600.0, 640, 480)
        s = 480.0 / 600.0
        expected_u = 0.0 * s - 0.5 * (800.0 * s - 640)
        expected_v = 0.0 * s
        np.testing.assert_allclose(uv, [expected_u, expected_v])

    def test_tall_intrinsic_uses_width_scale_path(self):
        # intr_aspect < img_aspect → scale by width
        uv = self._buf(0.0, 0.0, 640.0, 800.0, 640, 480)
        s = 640.0 / 640.0
        expected_u = 0.0 * s
        expected_v = 0.0 * s - 0.5 * (800.0 * s - 480)
        np.testing.assert_allclose(uv, [expected_u, expected_v])

    def test_out_of_fov_point_is_outside_buffer_bounds(self):
        uv = self._buf(9999.0, 9999.0, 640.0, 480.0, 640, 480)
        self.assertTrue(uv[0] > 640 or uv[1] > 480)

    def test_no_nan_or_inf_on_extreme_aspect_ratio(self):
        uv = self._buf(50.0, 50.0, 10000.0, 100.0, 640, 480)
        self.assertTrue(np.all(np.isfinite(uv)))


# ---------------------------------------------------------------------------
# CameraIntrinsics — world_point_to_panel_pixel
# ---------------------------------------------------------------------------

class TestWorldPointToPanelPixel(unittest.TestCase):

    def setUp(self):
        self.intr = make_intrinsics(
            fx=500.0, fy=500.0,
            cx=320.0, cy=240.0,
            width=640.0, height=480.0,
        )
        self.eye = identity_eye()
        self.img_w = 640
        self.img_h = 480

    def _project(self, point, eye=None):
        return self.intr.world_point_to_panel_pixel(
            point,
            eye or self.eye,
            self.img_w,
            self.img_h,
        )

    def test_point_on_optical_axis_maps_to_principal_point(self):
        uv = self._project(Vector3(0.0, 0.0, 5.0))
        np.testing.assert_allclose(uv, [320.0, 240.0], atol=1e-9)

    def test_point_behind_camera_returns_nan(self):
        uv = self._project(Vector3(0.0, 0.0, -1.0))
        self.assertTrue(np.all(np.isnan(uv)))

    def test_point_at_camera_origin_returns_nan(self):
        uv = self._project(Vector3(0.0, 0.0, 0.0))
        self.assertTrue(np.all(np.isnan(uv)))

    def test_pinhole_invariant_depth_does_not_change_projection(self):
        uv1 = self._project(Vector3(1.0, 1.0, 5.0))
        uv2 = self._project(Vector3(2.0, 2.0, 10.0))
        np.testing.assert_allclose(uv1, uv2, atol=1e-9)

    def test_point_right_of_axis_maps_right_of_center(self):
        uv = self._project(Vector3(1.0, 0.0, 5.0))
        self.assertGreater(uv[0], 320.0)

    def test_point_left_of_axis_maps_left_of_center(self):
        uv = self._project(Vector3(-1.0, 0.0, 5.0))
        self.assertLess(uv[0], 320.0)

    def test_point_above_axis_maps_above_center(self):
        # Y-up: positive world Y → lower v (higher in image)
        uv = self._project(Vector3(0.0, 1.0, 5.0))
        self.assertLess(uv[1], 240.0)

    def test_point_below_axis_maps_below_center(self):
        uv = self._project(Vector3(0.0, -1.0, 5.0))
        self.assertGreater(uv[1], 240.0)

    def test_identity_eye_treats_world_as_camera_coords(self):
        point = Vector3(0.5, -0.5, 3.0)
        uv = self._project(point)
        fx, fy, cx, cy = self.intr._fx_fy_cx_cy()
        expected_u = fx * (0.5 / 3.0) + cx
        expected_v = cy - fy * (-0.5 / 3.0)
        np.testing.assert_allclose(uv, [expected_u, expected_v], atol=1e-9)

    def test_rotated_eye_90_deg_around_y_shifts_point_off_axis(self):
        eye = Pose(
            position=Vector3.zero(),
            rotation=Quaternion.from_euler_angles(0.0, 90.0, 0.0),
        )
        uv = self._project(Vector3(0.0, 0.0, 5.0), eye=eye)
        self.assertFalse(np.allclose(uv, [320.0, 240.0], atol=1.0))

    def test_result_consistent_with_manual_pipeline(self):
        point = Vector3(0.3, -0.2, 4.0)
        uv = self._project(point)

        local = self.eye.inverse_transform_point(point)
        fx, fy, cx, cy = self.intr._fx_fy_cx_cy()
        intr_w, intr_h = map(float, self.intr.sensor_resolution)
        u_intr = fx * (local.x / local.z) + cx
        v_intr = cy - fy * (local.y / local.z)
        expected = CameraIntrinsics._intr_to_buffer(
            u_intr, v_intr, intr_w, intr_h, self.img_w, self.img_h
        )
        np.testing.assert_allclose(uv, expected, atol=1e-9)

    def test_zero_lens_offset_matches_no_offset(self):
        intr_none = make_intrinsics()
        intr_zero = make_intrinsics(lens_offset=Vector3(0.0, 0.0, 0.0))
        point = Vector3(0.5, 0.5, 3.0)
        uv_none = intr_none.world_point_to_panel_pixel(point, self.eye, self.img_w, self.img_h)
        uv_zero = intr_zero.world_point_to_panel_pixel(point, self.eye, self.img_w, self.img_h)
        np.testing.assert_allclose(uv_none, uv_zero, atol=1e-9)


# ---------------------------------------------------------------------------
# CameraIntrinsics — validation
# ---------------------------------------------------------------------------

class TestCameraIntrinsicsValidation(unittest.TestCase):

    def test_pose_as_lens_offset_raises(self):
        with self.assertRaises(ValidationError):
            make_intrinsics(lens_offset=Pose.identity())

    def test_frozen_mutation_raises(self):
        intr = make_intrinsics()
        with self.assertRaises(ValidationError):
            intr.focal_length = (999.0, 999.0)

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValidationError):
            CameraIntrinsics(
                focal_length=(500.0, 500.0),
                principal_point=(320.0, 240.0),
                sensor_resolution=(640.0, 480.0),
                unknown_field=True,
            )
