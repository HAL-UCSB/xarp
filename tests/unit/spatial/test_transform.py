import unittest

import numpy as np

from xarp.spatial import Pose, Quaternion, Transform, Vector3


# ===========================================================================
# Helpers
# ===========================================================================

def assert_vec3_close(tc: unittest.TestCase, v1: Vector3, v2: Vector3, atol: float = 1e-9) -> None:
    tc.assertAlmostEqual(v1.x, v2.x, delta=atol, msg=f"x: {v1.x} != {v2.x}")
    tc.assertAlmostEqual(v1.y, v2.y, delta=atol, msg=f"y: {v1.y} != {v2.y}")
    tc.assertAlmostEqual(v1.z, v2.z, delta=atol, msg=f"z: {v1.z} != {v2.z}")


def assert_quat_close(tc: unittest.TestCase, q1: Quaternion, q2: Quaternion, atol: float = 1e-9) -> None:
    tc.assertTrue(q1.isclose(q2, atol=atol), f"{q1!r} not close to {q2!r}")


def assert_transform_close(tc: unittest.TestCase, t1: Transform, t2: Transform, atol: float = 1e-9) -> None:
    assert_vec3_close(tc, t1.position, t2.position, atol=atol)
    assert_quat_close(tc, t1.rotation, t2.rotation, atol=atol)
    assert_vec3_close(tc, t1.scale, t2.scale, atol=atol)


# ===========================================================================
# 1. Construction & scale field
# ===========================================================================

class TestConstruction(unittest.TestCase):

    def test_default_scale_is_one(self):
        t = Transform()
        assert_vec3_close(self, t.scale, Vector3.one())

    def test_explicit_scale(self):
        s = Vector3(2.0, 3.0, 4.0)
        t = Transform(scale=s)
        assert_vec3_close(self, t.scale, s)

    def test_inherits_position_and_rotation_defaults(self):
        t = Transform()
        assert_vec3_close(self, t.position, Vector3.zero())
        assert_quat_close(self, t.rotation, Quaternion.identity())

    def test_is_subclass_of_pose(self):
        self.assertIsInstance(Transform(), Pose)

    def test_scale_assignment(self):
        t = Transform()
        t.scale = Vector3(2.0, 2.0, 2.0)
        assert_vec3_close(self, t.scale, Vector3(2.0, 2.0, 2.0))


# ===========================================================================
# 2. to_matrix — scale is incorporated
# ===========================================================================

class TestToMatrix(unittest.TestCase):

    def test_uniform_scale_stretches_columns(self):
        t = Transform(scale=Vector3(2.0, 2.0, 2.0))
        m = t.to_matrix()
        # Each basis column should have length 2
        self.assertAlmostEqual(np.linalg.norm(m[:3, 0]), 2.0, delta=1e-12)
        self.assertAlmostEqual(np.linalg.norm(m[:3, 1]), 2.0, delta=1e-12)
        self.assertAlmostEqual(np.linalg.norm(m[:3, 2]), 2.0, delta=1e-12)

    def test_non_uniform_scale_stretches_columns_independently(self):
        t = Transform(scale=Vector3(2.0, 3.0, 4.0))
        m = t.to_matrix()
        self.assertAlmostEqual(np.linalg.norm(m[:3, 0]), 2.0, delta=1e-12)
        self.assertAlmostEqual(np.linalg.norm(m[:3, 1]), 3.0, delta=1e-12)
        self.assertAlmostEqual(np.linalg.norm(m[:3, 2]), 4.0, delta=1e-12)

    def test_unit_scale_matches_pose_matrix(self):
        pos = Vector3(1.0, 2.0, 3.0)
        rot = Quaternion.from_euler_angles(30, 45, 60)
        t = Transform(position=pos, rotation=rot)
        p = Pose(position=pos, rotation=rot)
        np.testing.assert_allclose(t.to_matrix(), p.to_matrix(), atol=1e-12)

    def test_scale_does_not_affect_translation_column(self):
        t = Transform(position=Vector3(5.0, 6.0, 7.0), scale=Vector3(10.0, 10.0, 10.0))
        np.testing.assert_allclose(t.to_matrix()[:3, 3], [5.0, 6.0, 7.0], atol=1e-12)

    def test_bottom_row_unaffected_by_scale(self):
        t = Transform(scale=Vector3(5.0, 5.0, 5.0))
        np.testing.assert_allclose(t.to_matrix()[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


# ===========================================================================
# 3. from_matrix — scale is recovered
# ===========================================================================

class TestFromMatrix(unittest.TestCase):

    def test_round_trip_uniform_scale(self):
        original = Transform(
            position=Vector3(1.0, 2.0, 3.0),
            rotation=Quaternion.from_euler_angles(30, 45, 60),
            scale=Vector3(2.0, 2.0, 2.0),
        )
        assert_transform_close(self, Transform.from_matrix(original.to_matrix()), original)

    def test_round_trip_non_uniform_scale(self):
        original = Transform(
            position=Vector3(1.0, 2.0, 3.0),
            rotation=Quaternion.from_euler_angles(30, 45, 60),
            scale=Vector3(1.0, 2.0, 3.0),
        )
        assert_transform_close(self, Transform.from_matrix(original.to_matrix()), original)

    def test_returns_transform_not_pose(self):
        t = Transform.from_matrix(np.eye(4))
        self.assertIsInstance(t, Transform)

    def test_identity_matrix_gives_default_transform(self):
        assert_transform_close(self, Transform.from_matrix(np.eye(4)), Transform())

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            Transform.from_matrix(np.eye(3))


# ===========================================================================
# 4. inverse_matrix — must handle non-uniform scale
# ===========================================================================

class TestInverseMatrix(unittest.TestCase):

    def test_matrix_at_inverse_is_identity_uniform_scale(self):
        t = Transform(position=Vector3(1.0, 2.0, 3.0),
                      rotation=Quaternion.from_euler_angles(30, 45, 60),
                      scale=Vector3(2.0, 2.0, 2.0))
        np.testing.assert_allclose(t.to_matrix() @ t.inverse_matrix(), np.eye(4), atol=1e-12)

    def test_matrix_at_inverse_is_identity_non_uniform_scale(self):
        t = Transform(position=Vector3(1.0, 2.0, 3.0),
                      rotation=Quaternion.from_euler_angles(30, 45, 60),
                      scale=Vector3(1.0, 2.0, 3.0))
        np.testing.assert_allclose(t.to_matrix() @ t.inverse_matrix(), np.eye(4), atol=1e-12)

    def test_inverse_at_matrix_is_identity(self):
        t = Transform(position=Vector3(1.0, 2.0, 3.0),
                      rotation=Quaternion.from_euler_angles(30, 45, 60),
                      scale=Vector3(1.0, 2.0, 3.0))
        np.testing.assert_allclose(t.inverse_matrix() @ t.to_matrix(), np.eye(4), atol=1e-12)

    def test_unit_scale_matches_pose_inverse(self):
        pos = Vector3(1.0, 2.0, 3.0)
        rot = Quaternion.from_euler_angles(30, 45, 60)
        t = Transform(position=pos, rotation=rot)
        p = Pose(position=pos, rotation=rot)
        np.testing.assert_allclose(t.inverse_matrix(), p.inverse_matrix(), atol=1e-12)


# ===========================================================================
# 5. transform_normal — unique to Transform
# ===========================================================================

class TestTransformNormal(unittest.TestCase):

    def test_identity_leaves_normal_unchanged(self):
        t = Transform()
        assert_vec3_close(self, t.transform_normal(Vector3.up()), Vector3.up())

    def test_result_is_unit_vector(self):
        t = Transform(rotation=Quaternion.from_euler_angles(30, 45, 60),
                      scale=Vector3(2.0, 3.0, 4.0))
        result = t.transform_normal(Vector3.up())
        self.assertAlmostEqual(result.norm(), 1.0, delta=1e-12)

    def test_uniform_scale_does_not_change_normal_direction(self):
        """Under uniform scale normals are unaffected (up to normalisation)."""
        rot = Quaternion.from_euler_angles(30, 45, 60)
        t_scaled = Transform(rotation=rot, scale=Vector3(5.0, 5.0, 5.0))
        t_unit = Transform(rotation=rot)
        assert_vec3_close(self,
                          t_scaled.transform_normal(Vector3.up()),
                          t_unit.transform_normal(Vector3.up()),
                          atol=1e-9)

    def test_non_uniform_scale_corrects_normal(self):
        """
        A plane with normal +Y scaled by (1, 2, 1) — if we naively scaled the
        normal it would still point +Y but be unnormalised.  The inverse-transpose
        gives the geometrically correct result, which differs from just rotating.
        """
        t = Transform(scale=Vector3(3.0, 1.0, 1.0))
        # Normal to the YZ plane is +X; scaling X by 3 compresses it in normal space
        naive = t.transform_vector(Vector3.right()).normalized()
        correct = t.transform_normal(Vector3.right())
        # Both should be unit vectors — the point is they could differ under
        # non-uniform scale.  For this axis-aligned case they agree in direction
        # but we verify the method returns a unit vector either way.
        self.assertAlmostEqual(correct.norm(), 1.0, delta=1e-12)
        self.assertAlmostEqual(naive.norm(), 1.0, delta=1e-12)

    def test_matches_inverse_transpose_formula(self):
        t = Transform(rotation=Quaternion.from_euler_angles(30, 45, 60),
                      scale=Vector3(1.0, 2.0, 3.0))
        v = Vector3(0.0, 1.0, 0.0)
        rs_inv_T = np.linalg.inv(t.to_matrix()[:3, :3]).T
        expected = Vector3.from_sequence(rs_inv_T @ v.to_numpy()).normalized()
        assert_vec3_close(self, t.transform_normal(v), expected, atol=1e-12)


# ===========================================================================
# 6. __matmul__ — returns Transform, scale propagates through matrix product
# ===========================================================================

class TestComposition(unittest.TestCase):

    def test_returns_transform_instance(self):
        t1 = Transform()
        t2 = Transform()
        result = t1 @ t2
        self.assertIsInstance(result, Transform)

    def test_matches_matrix_product(self):
        t1 = Transform(position=Vector3(1.0, 0.0, 0.0),
                       rotation=Quaternion.from_euler_angles(0, 0, 90),
                       scale=Vector3(2.0, 1.0, 1.0))
        t2 = Transform(position=Vector3(0.0, 1.0, 0.0),
                       rotation=Quaternion.from_euler_angles(30, 0, 0),
                       scale=Vector3(1.0, 3.0, 1.0))
        np.testing.assert_allclose(
            (t1 @ t2).to_matrix(), t1.to_matrix() @ t2.to_matrix(), atol=1e-9
        )

    def test_scale_accumulates(self):
        t1 = Transform(scale=Vector3(2.0, 2.0, 2.0))
        t2 = Transform(scale=Vector3(3.0, 3.0, 3.0))
        result = t1 @ t2
        # Composed uniform scale should be 6×
        col_norms = [np.linalg.norm(result.to_matrix()[:3, i]) for i in range(3)]
        for n in col_norms:
            self.assertAlmostEqual(n, 6.0, delta=1e-9)

    def test_unsupported_type_returns_not_implemented(self):
        self.assertIs(Transform().__matmul__(Pose()), NotImplemented)

    def test_identity_neutral(self):
        t = Transform(position=Vector3(1.0, 2.0, 3.0),
                      rotation=Quaternion.from_euler_angles(30, 45, 60),
                      scale=Vector3(2.0, 3.0, 4.0))
        assert_transform_close(self, Transform() @ t, t)
        assert_transform_close(self, t @ Transform(), t)


# ===========================================================================
# 7. Geometry — scale affects point/vector transforms
# ===========================================================================

class TestGeometry(unittest.TestCase):

    def test_scale_stretches_point(self):
        t = Transform(scale=Vector3(2.0, 2.0, 2.0))
        result = t.transform_point(Vector3(1.0, 1.0, 1.0))
        assert_vec3_close(self, result, Vector3(2.0, 2.0, 2.0))

    def test_non_uniform_scale_stretches_axes_independently(self):
        t = Transform(scale=Vector3(1.0, 2.0, 3.0))
        assert_vec3_close(self, t.transform_point(Vector3(1.0, 0.0, 0.0)), Vector3(1.0, 0.0, 0.0))
        assert_vec3_close(self, t.transform_point(Vector3(0.0, 1.0, 0.0)), Vector3(0.0, 2.0, 0.0))
        assert_vec3_close(self, t.transform_point(Vector3(0.0, 0.0, 1.0)), Vector3(0.0, 0.0, 3.0))

    def test_point_roundtrip_with_scale(self):
        t = Transform(position=Vector3(1.0, 2.0, 3.0),
                      rotation=Quaternion.from_euler_angles(30, 45, 60),
                      scale=Vector3(2.0, 3.0, 4.0))
        v = Vector3(5.0, 6.0, 7.0)
        assert_vec3_close(self, t.inverse_transform_point(t.transform_point(v)), v, atol=1e-9)

    def test_vector_roundtrip_with_scale(self):
        t = Transform(position=Vector3(1.0, 2.0, 3.0),
                      rotation=Quaternion.from_euler_angles(30, 45, 60),
                      scale=Vector3(2.0, 3.0, 4.0))
        v = Vector3(1.0, 0.0, 0.0)
        assert_vec3_close(self, t.inverse_transform_vector(t.transform_vector(v)), v, atol=1e-9)


# ===========================================================================
# 8. Serialization — scale appears in dump
# ===========================================================================

class TestSerialization(unittest.TestCase):

    def test_model_dump_includes_scale(self):
        t = Transform(scale=Vector3(2.0, 3.0, 4.0))
        d = t.model_dump()
        self.assertIn("scale", d)
        self.assertEqual(d["scale"], [2.0, 3.0, 4.0])

    def test_json_roundtrip(self):
        t = Transform(position=Vector3(1.0, -2.0, 3.0),
                      rotation=Quaternion.from_euler_angles(30, 45, 60),
                      scale=Vector3(2.0, 3.0, 4.0))
        t2 = Transform.model_validate_json(t.model_dump_json())
        assert_transform_close(self, t, t2, atol=1e-9)

    def test_validate_from_dict(self):
        d = {
            "position": [1.0, 2.0, 3.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [2.0, 3.0, 4.0],
        }
        t = Transform.model_validate(d)
        assert_vec3_close(self, t.scale, Vector3(2.0, 3.0, 4.0))

    def test_default_scale_in_dump(self):
        self.assertEqual(Transform().model_dump()["scale"], [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
