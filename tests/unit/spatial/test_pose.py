import unittest

import numpy as np

from xarp.spatial import Pose, Quaternion, Vector3


# ===========================================================================
# Helpers
# ===========================================================================

def assert_vec3_close(tc: unittest.TestCase, v1: Vector3, v2: Vector3, atol: float = 1e-9) -> None:
    tc.assertAlmostEqual(v1.x, v2.x, delta=atol, msg=f"x: {v1.x} != {v2.x}")
    tc.assertAlmostEqual(v1.y, v2.y, delta=atol, msg=f"y: {v1.y} != {v2.y}")
    tc.assertAlmostEqual(v1.z, v2.z, delta=atol, msg=f"z: {v1.z} != {v2.z}")


def assert_quat_close(tc: unittest.TestCase, q1: Quaternion, q2: Quaternion, atol: float = 1e-9) -> None:
    tc.assertTrue(q1.isclose(q2, atol=atol), f"{q1!r} not close to {q2!r}")


def assert_pose_close(tc: unittest.TestCase, p1: Pose, p2: Pose, atol: float = 1e-9) -> None:
    assert_vec3_close(tc, p1.position, p2.position, atol=atol)
    assert_quat_close(tc, p1.rotation, p2.rotation, atol=atol)


# ===========================================================================
# 1. Construction & validation
# ===========================================================================

class TestConstruction(unittest.TestCase):

    def test_defaults(self):
        p = Pose()
        assert_vec3_close(self, p.position, Vector3.zero())
        assert_quat_close(self, p.rotation, Quaternion.identity())

    def test_identity_classmethod(self):
        assert_pose_close(self, Pose.identity(), Pose())

    def test_explicit_fields(self):
        v = Vector3(1.0, 2.0, 3.0)
        q = Quaternion.from_euler_angles(30, 45, 60)
        p = Pose(position=v, rotation=q)
        assert_vec3_close(self, p.position, v)
        assert_quat_close(self, p.rotation, q)

    def test_extra_fields_forbidden(self):
        with self.assertRaises(Exception):
            Pose(position=Vector3.zero(), rotation=Quaternion.identity(), extra=1)


# ===========================================================================
# 2. Mutation (validate_assignment)
# ===========================================================================

class TestMutation(unittest.TestCase):

    def test_position_assignment(self):
        p = Pose()
        p.position = Vector3(1.0, 2.0, 3.0)
        assert_vec3_close(self, p.position, Vector3(1.0, 2.0, 3.0))

    def test_rotation_assignment(self):
        p = Pose()
        q = Quaternion.from_euler_angles(0, 0, 90)
        p.rotation = q
        assert_quat_close(self, p.rotation, q)

    def test_position_iadd(self):
        p = Pose()
        p.position = p.position + Vector3.one()
        assert_vec3_close(self, p.position, Vector3.one())


# ===========================================================================
# 3. to_matrix / from_matrix
# ===========================================================================

class TestMatrix(unittest.TestCase):

    def test_identity_pose_gives_identity_matrix(self):
        np.testing.assert_allclose(Pose.identity().to_matrix(), np.eye(4), atol=1e-12)

    def test_matrix_dtype_is_float64(self):
        self.assertEqual(Pose.identity().to_matrix().dtype, np.float64)

    def test_matrix_shape(self):
        self.assertEqual(Pose.identity().to_matrix().shape, (4, 4))

    def test_bottom_row_is_0001(self):
        p = Pose(position=Vector3(5.0, -3.0, 2.0),
                 rotation=Quaternion.from_euler_angles(10, 20, 30))
        np.testing.assert_allclose(p.to_matrix()[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_translation_in_last_column(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0))
        np.testing.assert_allclose(p.to_matrix()[:3, 3], [1.0, 2.0, 3.0], atol=1e-12)

    def test_rotation_block_matches_quaternion_matrix(self):
        q = Quaternion.from_euler_angles(0, 0, 90)
        np.testing.assert_allclose(
            Pose(rotation=q).to_matrix()[:3, :3], q.to_matrix(), atol=1e-12
        )

    def test_from_matrix_round_trip(self):
        original = Pose(position=Vector3(1.0, -2.0, 3.0),
                        rotation=Quaternion.from_euler_angles(30, 45, 60))
        assert_pose_close(self, Pose.from_matrix(original.to_matrix()), original, atol=1e-9)

    def test_from_matrix_identity(self):
        assert_pose_close(self, Pose.from_matrix(np.eye(4)), Pose.identity())

    def test_from_matrix_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            Pose.from_matrix(np.eye(3))


# ===========================================================================
# 4. inverse_matrix
# ===========================================================================

class TestInverseMatrix(unittest.TestCase):

    def test_matrix_at_inverse_is_identity(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        np.testing.assert_allclose(p.to_matrix() @ p.inverse_matrix(), np.eye(4), atol=1e-12)

    def test_inverse_at_matrix_is_identity(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        np.testing.assert_allclose(p.inverse_matrix() @ p.to_matrix(), np.eye(4), atol=1e-12)

    def test_identity_pose_inverse_is_identity(self):
        np.testing.assert_allclose(Pose.identity().inverse_matrix(), np.eye(4), atol=1e-12)

    def test_translation_only_inverse(self):
        # Pure translation t=(3,0,0) → inverse translation (-3,0,0)
        p = Pose(position=Vector3(3.0, 0.0, 0.0))
        np.testing.assert_allclose(p.inverse_matrix()[:3, 3], [-3.0, 0.0, 0.0], atol=1e-12)


# ===========================================================================
# 5. inverse()
# ===========================================================================

class TestInverse(unittest.TestCase):

    def test_pose_at_inverse_is_identity(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        assert_pose_close(self, p @ p.inverse(), Pose.identity(), atol=1e-9)

    def test_inverse_at_pose_is_identity(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        assert_pose_close(self, p.inverse() @ p, Pose.identity(), atol=1e-9)

    def test_identity_inverse_is_identity(self):
        assert_pose_close(self, Pose.identity().inverse(), Pose.identity())

    def test_double_inverse_is_original(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(10, 20, 30))
        assert_pose_close(self, p.inverse().inverse(), p, atol=1e-9)

    def test_inverse_matrix_matches_inverse_pose_matrix(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        np.testing.assert_allclose(p.inverse().to_matrix(), p.inverse_matrix(), atol=1e-9)


# ===========================================================================
# 6. Composition (__matmul__)
# ===========================================================================

class TestComposition(unittest.TestCase):

    def test_identity_neutral_left(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        assert_pose_close(self, Pose.identity() @ p, p)

    def test_identity_neutral_right(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        assert_pose_close(self, p @ Pose.identity(), p)

    def test_matches_matrix_product(self):
        p1 = Pose(position=Vector3(1.0, 0.0, 0.0),
                  rotation=Quaternion.from_euler_angles(0, 0, 90))
        p2 = Pose(position=Vector3(0.0, 1.0, 0.0),
                  rotation=Quaternion.from_euler_angles(30, 0, 0))
        np.testing.assert_allclose(
            (p1 @ p2).to_matrix(), p1.to_matrix() @ p2.to_matrix(), atol=1e-9
        )

    def test_associative(self):
        p1 = Pose(position=Vector3(1.0, 0.0, 0.0),
                  rotation=Quaternion.from_euler_angles(10, 0, 0))
        p2 = Pose(position=Vector3(0.0, 2.0, 0.0),
                  rotation=Quaternion.from_euler_angles(0, 20, 0))
        p3 = Pose(position=Vector3(0.0, 0.0, 3.0),
                  rotation=Quaternion.from_euler_angles(0, 0, 30))
        assert_pose_close(self, (p1 @ p2) @ p3, p1 @ (p2 @ p3), atol=1e-9)

    def test_not_commutative(self):
        p1 = Pose(position=Vector3(1.0, 0.0, 0.0),
                  rotation=Quaternion.from_euler_angles(0, 0, 90))
        p2 = Pose(position=Vector3(0.0, 1.0, 0.0),
                  rotation=Quaternion.from_euler_angles(90, 0, 0))
        self.assertFalse((p1 @ p2).position.isclose((p2 @ p1).position))

    def test_unsupported_type_returns_not_implemented(self):
        self.assertIs(Pose.identity().__matmul__(42), NotImplemented)


# ===========================================================================
# 7. Geometry application
# ===========================================================================

class TestTransformPoint(unittest.TestCase):

    def test_identity_leaves_point_unchanged(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert_vec3_close(self, Pose.identity().transform_point(v), v)

    def test_translation_applied(self):
        p = Pose(position=Vector3(1.0, 0.0, 0.0))
        assert_vec3_close(self, p.transform_point(Vector3.zero()), Vector3(1.0, 0.0, 0.0))

    def test_rotation_then_translation(self):
        # 90° yaw rotates +X → +Y, then offset by (0, 0, 5)
        p = Pose(position=Vector3(0.0, 0.0, 5.0),
                 rotation=Quaternion.from_euler_angles(0, 0, 90))
        assert_vec3_close(self, p.transform_point(Vector3.right()),
                          Vector3(0.0, 1.0, 5.0), atol=1e-9)

    def test_matches_matrix_multiply(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        v = Vector3(4.0, 5.0, 6.0)
        via_matrix = p.to_matrix() @ np.array([v.x, v.y, v.z, 1.0])
        assert_vec3_close(self, p.transform_point(v),
                          Vector3(float(via_matrix[0]), float(via_matrix[1]), float(via_matrix[2])),
                          atol=1e-12)


class TestTransformVector(unittest.TestCase):

    def test_translation_does_not_affect_vector(self):
        p = Pose(position=Vector3(100.0, 100.0, 100.0))
        assert_vec3_close(self, p.transform_vector(Vector3.right()), Vector3.right())

    def test_matches_matrix_multiply(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        v = Vector3(1.0, 0.0, 0.0)
        via_matrix = p.to_matrix()[:3, :3] @ v.to_numpy()
        assert_vec3_close(self, p.transform_vector(v),
                          Vector3.from_sequence(via_matrix), atol=1e-12)


class TestInverseTransform(unittest.TestCase):

    def test_point_roundtrip(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        v = Vector3(4.0, 5.0, 6.0)
        assert_vec3_close(self, p.inverse_transform_point(p.transform_point(v)), v, atol=1e-9)

    def test_vector_roundtrip(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        v = Vector3(1.0, 0.0, 0.0)
        assert_vec3_close(self, p.inverse_transform_vector(p.transform_vector(v)), v, atol=1e-9)

    def test_inverse_point_matches_inverse_matrix(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        v = Vector3(7.0, 8.0, 9.0)
        via_matrix = p.inverse_matrix() @ np.array([v.x, v.y, v.z, 1.0])
        assert_vec3_close(self, p.inverse_transform_point(v),
                          Vector3(float(via_matrix[0]), float(via_matrix[1]), float(via_matrix[2])),
                          atol=1e-12)

    def test_inverse_vector_matches_inverse_matrix(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        v = Vector3(1.0, 0.0, 0.0)
        via_matrix = p.inverse_matrix()[:3, :3] @ v.to_numpy()
        assert_vec3_close(self, p.inverse_transform_vector(v),
                          Vector3.from_sequence(via_matrix), atol=1e-12)


# ===========================================================================
# 8. Direction axes
# ===========================================================================

class TestAxes(unittest.TestCase):

    def test_identity_axes_match_world_axes(self):
        p = Pose.identity()
        assert_vec3_close(self, p.forward, Vector3.forward())
        assert_vec3_close(self, p.up, Vector3.up())
        assert_vec3_close(self, p.right, Vector3.right())

    def test_axes_are_unit_vectors(self):
        p = Pose(rotation=Quaternion.from_euler_angles(37, -22, 111))
        self.assertAlmostEqual(p.forward.norm(), 1.0, delta=1e-12)
        self.assertAlmostEqual(p.up.norm(), 1.0, delta=1e-12)
        self.assertAlmostEqual(p.right.norm(), 1.0, delta=1e-12)

    def test_axes_are_mutually_orthogonal(self):
        p = Pose(rotation=Quaternion.from_euler_angles(37, -22, 111))
        self.assertAlmostEqual(p.forward.dot(p.up), 0.0, delta=1e-9)
        self.assertAlmostEqual(p.forward.dot(p.right), 0.0, delta=1e-9)
        self.assertAlmostEqual(p.up.dot(p.right), 0.0, delta=1e-9)

    def test_position_does_not_affect_axes(self):
        q = Quaternion.from_euler_angles(10, 20, 30)
        p1 = Pose(rotation=q)
        p2 = Pose(position=Vector3(100.0, 200.0, 300.0), rotation=q)
        assert_vec3_close(self, p1.forward, p2.forward)
        assert_vec3_close(self, p1.up, p2.up)
        assert_vec3_close(self, p1.right, p2.right)


# ===========================================================================
# 9. ray_point
# ===========================================================================

class TestRayPoint(unittest.TestCase):

    def test_zero_distance_equals_position(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0))
        assert_vec3_close(self, p.ray_point(0.0), p.position)

    def test_unit_distance_along_forward(self):
        p = Pose(position=Vector3(1.0, 0.0, 0.0))
        assert_vec3_close(self, p.ray_point(1.0), Vector3(1.0, 0.0, 1.0))

    def test_negative_distance(self):
        assert_vec3_close(self, Pose().ray_point(-2.0), Vector3(0.0, 0.0, -2.0))

    def test_scales_linearly(self):
        p = Pose()
        self.assertAlmostEqual(p.ray_point(6.0).z, 2.0 * p.ray_point(3.0).z, delta=1e-12)

    def test_with_rotation(self):
        # 90° pitch maps +Z forward → +X
        p = Pose(rotation=Quaternion.from_euler_angles(0, 90, 0))
        assert_vec3_close(self, p.ray_point(1.0), Vector3(1.0, 0.0, 0.0), atol=1e-9)


# ===========================================================================
# 10. Serialization
# ===========================================================================

class TestSerialization(unittest.TestCase):

    def test_model_dump_position_is_array(self):
        p = Pose(position=Vector3(1.0, 2.0, 3.0))
        self.assertEqual(p.model_dump()["position"], [1.0, 2.0, 3.0])

    def test_model_dump_rotation_is_array(self):
        p = Pose(rotation=Quaternion.identity())
        self.assertEqual(p.model_dump()["rotation"], [0.0, 0.0, 0.0, 1.0])

    def test_json_roundtrip(self):
        p = Pose(position=Vector3(1.0, -2.0, 3.0),
                 rotation=Quaternion.from_euler_angles(30, 45, 60))
        assert_pose_close(self, Pose.model_validate_json(p.model_dump_json()), p, atol=1e-9)

    def test_validate_from_dict(self):
        d = {"position": [1.0, 2.0, 3.0], "rotation": [0.0, 0.0, 0.0, 1.0]}
        p = Pose.model_validate(d)
        assert_vec3_close(self, p.position, Vector3(1.0, 2.0, 3.0))
        assert_quat_close(self, p.rotation, Quaternion.identity())


if __name__ == "__main__":
    unittest.main()
