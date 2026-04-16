import math
import unittest

import numpy as np

from xarp.spatial import (
    Pose, Quaternion, Vector3,
    convex_hull_2d,
    cosine_similarity,
    matrix_to_trs,
    point_to_ray_distance,
)


# ===========================================================================
# Helpers
# ===========================================================================

def assert_vec3_close(tc: unittest.TestCase, v1: Vector3, v2: Vector3, atol: float = 1e-9) -> None:
    tc.assertAlmostEqual(v1.x, v2.x, delta=atol, msg=f"x: {v1.x} != {v2.x}")
    tc.assertAlmostEqual(v1.y, v2.y, delta=atol, msg=f"y: {v1.y} != {v2.y}")
    tc.assertAlmostEqual(v1.z, v2.z, delta=atol, msg=f"z: {v1.z} != {v2.z}")


def assert_quat_close(tc: unittest.TestCase, q1: Quaternion, q2: Quaternion, atol: float = 1e-9) -> None:
    tc.assertTrue(q1.isclose(q2, atol=atol), f"{q1!r} not close to {q2!r}")


# ===========================================================================
# 1. cosine_similarity
# ===========================================================================

class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity(Vector3.right(), Vector3.right()), 1.0, delta=1e-9)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(cosine_similarity(Vector3.right(), Vector3.left()), -1.0, delta=1e-9)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity(Vector3.right(), Vector3.up()), 0.0, delta=1e-9)

    def test_45_degrees(self):
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(1.0, 1.0, 0.0)
        self.assertAlmostEqual(cosine_similarity(v1, v2), math.cos(math.pi / 4), delta=1e-9)

    def test_scale_invariant(self):
        v = Vector3(1.0, 2.0, 3.0)
        self.assertAlmostEqual(
            cosine_similarity(v, v * 10.0),
            cosine_similarity(v, v),
            delta=1e-9,
        )

    def test_symmetric(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        self.assertAlmostEqual(cosine_similarity(a, b), cosine_similarity(b, a), delta=1e-12)

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            cosine_similarity(Vector3.zero(), Vector3.right())

    def test_result_in_range(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(-4.0, 5.0, -6.0)
        result = cosine_similarity(a, b)
        self.assertGreaterEqual(result, -1.0)
        self.assertLessEqual(result, 1.0)


# ===========================================================================
# 2. matrix_to_trs
# ===========================================================================

class TestMatrixToTrs(unittest.TestCase):

    def _make_trs(self, position, rotation, scale):
        """Build a 4×4 TRS matrix from components."""
        r = rotation.to_matrix()
        m = np.eye(4, dtype=np.float64)
        m[:3, 0] = r[:, 0] * scale.x
        m[:3, 1] = r[:, 1] * scale.y
        m[:3, 2] = r[:, 2] * scale.z
        m[:3, 3] = position.to_numpy()
        return m

    def test_identity_matrix(self):
        pos, rot, scale = matrix_to_trs(np.eye(4))
        assert_vec3_close(self, pos, Vector3.zero())
        assert_quat_close(self, rot, Quaternion.identity())
        assert_vec3_close(self, scale, Vector3.one())

    def test_returns_three_components(self):
        result = matrix_to_trs(np.eye(4))
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], Vector3)
        self.assertIsInstance(result[1], Quaternion)
        self.assertIsInstance(result[2], Vector3)

    def test_translation_extracted(self):
        m = np.eye(4)
        m[:3, 3] = [1.0, 2.0, 3.0]
        pos, _, _ = matrix_to_trs(m)
        assert_vec3_close(self, pos, Vector3(1.0, 2.0, 3.0))

    def test_uniform_scale_extracted(self):
        pos = Vector3.zero()
        rot = Quaternion.identity()
        scale = Vector3(3.0, 3.0, 3.0)
        _, _, recovered = matrix_to_trs(self._make_trs(pos, rot, scale))
        assert_vec3_close(self, recovered, scale)

    def test_non_uniform_scale_extracted(self):
        pos = Vector3.zero()
        rot = Quaternion.identity()
        scale = Vector3(1.0, 2.0, 4.0)
        _, _, recovered = matrix_to_trs(self._make_trs(pos, rot, scale))
        assert_vec3_close(self, recovered, scale)

    def test_rotation_extracted(self):
        rot = Quaternion.from_euler_angles(30, 45, 60)
        m = self._make_trs(Vector3.zero(), rot, Vector3.one())
        _, recovered, _ = matrix_to_trs(m)
        assert_quat_close(self, recovered, rot, atol=1e-9)

    def test_full_trs_round_trip(self):
        pos = Vector3(1.0, -2.0, 3.0)
        rot = Quaternion.from_euler_angles(30, 45, 60)
        scale = Vector3(1.0, 2.0, 3.0)
        m = self._make_trs(pos, rot, scale)
        rpos, rrot, rscale = matrix_to_trs(m)
        assert_vec3_close(self, rpos, pos)
        assert_quat_close(self, rrot, rot, atol=1e-9)
        assert_vec3_close(self, rscale, scale)

    def test_rotation_is_orthonormal_after_decomposition(self):
        pos = Vector3(1.0, 2.0, 3.0)
        rot = Quaternion.from_euler_angles(10, 20, 30)
        scale = Vector3(2.0, 3.0, 4.0)
        _, recovered_rot, _ = matrix_to_trs(self._make_trs(pos, rot, scale))
        self.assertAlmostEqual(recovered_rot.norm(), 1.0, delta=1e-9)

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            matrix_to_trs(np.eye(3))


# ===========================================================================
# 3. convex_hull_2d
# ===========================================================================

class TestConvexHull2d(unittest.TestCase):

    def test_single_point_returns_that_point(self):
        pts = np.array([[1.0, 2.0]])
        hull = convex_hull_2d(pts)
        self.assertEqual(hull.shape, (1, 2))
        np.testing.assert_array_equal(hull, pts)

    def test_two_points_returns_both(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        hull = convex_hull_2d(pts)
        self.assertEqual(hull.shape[1], 2)

    def test_square_has_four_hull_points(self):
        pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ])
        hull = convex_hull_2d(pts)
        self.assertEqual(len(hull), 4)

    def test_interior_point_excluded(self):
        pts = np.array([
            [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0],
            [1.0, 1.0],  # interior — should not appear in hull
        ])
        hull = convex_hull_2d(pts)
        self.assertEqual(len(hull), 4)
        interior = np.array([1.0, 1.0])
        self.assertFalse(any(np.allclose(row, interior) for row in hull))

    def test_collinear_points_excluded(self):
        # Three collinear points — only the two endpoints should be in hull
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        hull = convex_hull_2d(pts)
        # The collinear midpoint (1,0) should not appear
        self.assertFalse(any(np.allclose(row, [1.0, 0.0]) for row in hull))

    def test_output_dtype_is_float64(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        self.assertEqual(convex_hull_2d(pts).dtype, np.float64)

    def test_output_shape_has_two_columns(self):
        pts = np.random.default_rng(0).random((20, 2))
        hull = convex_hull_2d(pts)
        self.assertEqual(hull.shape[1], 2)

    def test_all_hull_vertices_are_from_input(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
        hull = convex_hull_2d(pts)
        for vertex in hull:
            self.assertTrue(any(np.allclose(vertex, p) for p in pts))

    def test_hull_is_convex(self):
        """All cross products of consecutive edges should have the same sign."""
        pts = np.random.default_rng(42).random((30, 2))
        hull = convex_hull_2d(pts)
        n = len(hull)
        if n < 3:
            return
        signs = []
        for i in range(n):
            o = hull[i]
            a = hull[(i + 1) % n]
            b = hull[(i + 2) % n]
            cross = (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            signs.append(cross)
        # All non-zero cross products should share the same sign
        nonzero = [s for s in signs if abs(s) > 1e-12]
        if nonzero:
            self.assertTrue(all(s > 0 for s in nonzero) or all(s < 0 for s in nonzero))

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            convex_hull_2d(np.array([[1.0, 2.0, 3.0]]))


# ===========================================================================
# 4. point_to_ray_distance
# ===========================================================================

class TestPointToRayDistance(unittest.TestCase):

    def test_point_on_ray_origin(self):
        pose = Pose()
        self.assertAlmostEqual(point_to_ray_distance(Vector3.zero(), pose), 0.0, delta=1e-9)

    def test_point_on_ray(self):
        # Point directly along forward (+Z) — distance should be zero
        pose = Pose()
        point = Vector3(0.0, 0.0, 5.0)
        self.assertAlmostEqual(point_to_ray_distance(point, pose), 0.0, delta=1e-9)

    def test_point_perpendicular_to_ray(self):
        # Point 3 units to the right of origin, ray points forward
        pose = Pose()
        point = Vector3(3.0, 0.0, 0.0)
        self.assertAlmostEqual(point_to_ray_distance(point, pose), 3.0, delta=1e-9)

    def test_point_beside_ray_at_depth(self):
        # Point offset sideways from a point along the ray
        pose = Pose()
        point = Vector3(4.0, 0.0, 3.0)  # 4 units right, 3 units forward
        self.assertAlmostEqual(point_to_ray_distance(point, pose), 4.0, delta=1e-9)

    def test_point_behind_ray_origin(self):
        # Point behind origin — closest point is the origin itself
        pose = Pose()
        point = Vector3(0.0, 0.0, -5.0)
        self.assertAlmostEqual(point_to_ray_distance(point, pose), 5.0, delta=1e-9)

    def test_point_behind_and_offset(self):
        # Point behind and to the side — closest point is the origin.
        # z=0 so t<=0 (at boundary); distance = sqrt(3^2 + 4^2) = 5.
        pose = Pose()
        point = Vector3(3.0, 4.0, 0.0)
        self.assertAlmostEqual(point_to_ray_distance(point, pose), 5.0, delta=1e-9)

    def test_translated_pose(self):
        # Shift origin to (0, 0, 10) — same geometry, translated
        pose = Pose(position=Vector3(0.0, 0.0, 10.0))
        point = Vector3(3.0, 0.0, 10.0)  # 3 units right of new origin
        self.assertAlmostEqual(point_to_ray_distance(point, pose), 3.0, delta=1e-9)

    def test_rotated_pose(self):
        # 90° pitch maps forward to +X; point is 3 units up from origin
        pose = Pose(rotation=Quaternion.from_euler_angles(0, 90, 0))
        point = Vector3(0.0, 3.0, 0.0)  # perpendicular to rotated forward
        self.assertAlmostEqual(point_to_ray_distance(point, pose), 3.0, delta=1e-9)

    def test_result_is_non_negative(self):
        pose = Pose(position=Vector3(1.0, 2.0, 3.0),
                    rotation=Quaternion.from_euler_angles(30, 45, 60))
        for x, y, z in [(0, 0, 0), (5, 0, 0), (-1, -1, -1), (3, 4, 5)]:
            result = point_to_ray_distance(Vector3(x, y, z), pose)
            self.assertGreaterEqual(result, 0.0)

    def test_result_is_float(self):
        pose = Pose()
        self.assertIsInstance(point_to_ray_distance(Vector3.zero(), pose), float)


if __name__ == "__main__":
    unittest.main()
