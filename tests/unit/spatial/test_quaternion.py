import math
import unittest

import numpy as np

from xarp.spatial import Quaternion, Vector3


# ===========================================================================
# Helpers
# ===========================================================================

def assert_quat_close(tc: unittest.TestCase, q1: Quaternion, q2: Quaternion, atol: float = 1e-9) -> None:
    """Assert two quaternions are equal up to sign (q == -q for rotations)."""
    tc.assertTrue(q1.isclose(q2, atol=atol), f"{q1!r} not close to {q2!r}")


def assert_vec3_close(tc: unittest.TestCase, v1: Vector3, v2: Vector3, atol: float = 1e-9) -> None:
    tc.assertAlmostEqual(v1.x, v2.x, delta=atol, msg=f"x: {v1.x} != {v2.x}")
    tc.assertAlmostEqual(v1.y, v2.y, delta=atol, msg=f"y: {v1.y} != {v2.y}")
    tc.assertAlmostEqual(v1.z, v2.z, delta=atol, msg=f"z: {v1.z} != {v2.z}")


# ===========================================================================
# 1. Construction
# ===========================================================================

class TestConstruction(unittest.TestCase):

    def test_kwargs(self):
        q = Quaternion(x=1.0, y=2.0, z=3.0, w=4.0)
        self.assertEqual((q.x, q.y, q.z, q.w), (1.0, 2.0, 3.0, 4.0))

    def test_four_positional_args(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertEqual((q.x, q.y, q.z, q.w), (1.0, 2.0, 3.0, 4.0))

    def test_single_sequence_arg_list(self):
        q = Quaternion([1.0, 2.0, 3.0, 4.0])
        self.assertEqual((q.x, q.y, q.z, q.w), (1.0, 2.0, 3.0, 4.0))

    def test_single_sequence_arg_tuple(self):
        q = Quaternion((0.0, 0.0, 0.0, 1.0))
        self.assertEqual(q.w, 1.0)

    def test_single_sequence_arg_numpy(self):
        arr = np.array([0.0, 0.0, 0.0, 1.0])
        q = Quaternion(arr)
        self.assertEqual(q.w, 1.0)

    def test_wrong_sequence_length_raises(self):
        with self.assertRaisesRegex(ValueError, "Expected 4 components"):
            Quaternion([1.0, 2.0, 3.0])

    def test_wrong_positional_count_raises(self):
        with self.assertRaisesRegex(ValueError, "Expected 1 or 4 positional arguments"):
            Quaternion(1.0, 2.0, 3.0)

    def test_from_xyzw(self):
        q = Quaternion.from_xyzw(1, 2, 3, 4)
        self.assertEqual((q.x, q.y, q.z, q.w), (1.0, 2.0, 3.0, 4.0))

    def test_from_sequence_list(self):
        q = Quaternion.from_sequence([0.0, 0.0, 0.0, 1.0])
        self.assertEqual(q.w, 1.0)

    def test_from_sequence_wrong_length(self):
        with self.assertRaises(ValueError):
            Quaternion.from_sequence([1.0, 2.0])

    def test_identity(self):
        q = Quaternion.identity()
        self.assertEqual((q.x, q.y, q.z, q.w), (0.0, 0.0, 0.0, 1.0))

    def test_zero(self):
        q = Quaternion.zero()
        self.assertEqual((q.x, q.y, q.z, q.w), (0.0, 0.0, 0.0, 0.0))

    def test_immutable(self):
        q = Quaternion.identity()
        with self.assertRaises(Exception):  # pydantic frozen raises ValidationError or AttributeError
            q.x = 99.0  # type: ignore[misc]

    def test_internal_numpy_array(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        arr = q._arr
        self.assertEqual(arr.shape, (4,))
        self.assertEqual(arr.dtype, np.float64)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0])


# ===========================================================================
# 2. Conversion
# ===========================================================================

class TestConversion(unittest.TestCase):

    def test_to_numpy_returns_copy(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        arr = q.to_numpy()
        arr[0] = 999.0
        self.assertEqual(q.x, 1.0)  # original unchanged

    def test_to_list(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(q.to_list(), [1.0, 2.0, 3.0, 4.0])

    def test_to_tuple(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(q.to_tuple(), (1.0, 2.0, 3.0, 4.0))


# ===========================================================================
# 3. Sequence protocol
# ===========================================================================

class TestSequenceProtocol(unittest.TestCase):

    def test_len(self):
        self.assertEqual(len(Quaternion.identity()), 4)

    def test_iter(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(list(q), [1.0, 2.0, 3.0, 4.0])

    def test_getitem(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(q[0], 1.0)
        self.assertEqual(q[1], 2.0)
        self.assertEqual(q[2], 3.0)
        self.assertEqual(q[3], 4.0)

    def test_unpack(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        x, y, z, w = q
        self.assertEqual((x, y, z, w), (1.0, 2.0, 3.0, 4.0))


# ===========================================================================
# 4. Comparison
# ===========================================================================

class TestComparison(unittest.TestCase):

    def test_isclose_identical(self):
        q = Quaternion.identity()
        self.assertTrue(q.isclose(q))

    def test_isclose_negated(self):
        """q and -q represent the same rotation."""
        q = Quaternion.identity()
        self.assertTrue(q.isclose(-q))

    def test_isclose_within_tolerance(self):
        q1 = Quaternion(0.0, 0.0, 0.0, 1.0)
        q2 = Quaternion(0.0, 0.0, 0.0, 1.0 + 5e-10)
        self.assertTrue(q1.isclose(q2))

    def test_isclose_outside_tolerance(self):
        q1 = Quaternion(0.0, 0.0, 0.0, 1.0)
        q2 = Quaternion(0.1, 0.0, 0.0, 1.0)
        self.assertFalse(q1.isclose(q2))

    def test_repr(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        r = repr(q)
        self.assertIn("Quaternion", r)
        self.assertIn("x=", r)
        self.assertIn("w=", r)


# ===========================================================================
# 5. Core math
# ===========================================================================

class TestCoreMath(unittest.TestCase):

    def test_norm_identity(self):
        self.assertAlmostEqual(Quaternion.identity().norm(), 1.0, delta=1e-12)

    def test_norm_general(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertAlmostEqual(q.norm(), math.sqrt(30.0), delta=1e-12)

    def test_norm_squared(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertAlmostEqual(q.norm_squared(), 30.0, delta=1e-12)

    def test_normalized_is_unit(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0).normalized()
        self.assertAlmostEqual(q.norm(), 1.0, delta=1e-12)

    def test_normalized_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            Quaternion.zero().normalized()

    def test_conjugate(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        c = q.conjugate()
        self.assertEqual((c.x, c.y, c.z, c.w), (-1.0, -2.0, -3.0, 4.0))

    def test_inverse_identity(self):
        assert_quat_close(self, Quaternion.identity().inverse(), Quaternion.identity())

    def test_inverse_unit(self):
        """For unit quaternion, inverse == conjugate."""
        q = Quaternion.from_euler_angles(30, 45, 60)
        assert_quat_close(self, q.inverse(), q.conjugate())

    def test_inverse_general(self):
        """q * q^-1 == identity."""
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        assert_quat_close(self, q * q.inverse(), Quaternion.identity())

    def test_inverse_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            Quaternion.zero().inverse()

    def test_dot_self(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertAlmostEqual(q.dot(q), 30.0, delta=1e-12)

    def test_dot_orthogonal(self):
        q1 = Quaternion(1.0, 0.0, 0.0, 0.0)
        q2 = Quaternion(0.0, 1.0, 0.0, 0.0)
        self.assertAlmostEqual(q1.dot(q2), 0.0, delta=1e-12)

    def test_negation(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        neg = -q
        self.assertEqual((neg.x, neg.y, neg.z, neg.w), (-1.0, -2.0, -3.0, -4.0))


# ===========================================================================
# 6. Hamilton product
# ===========================================================================

class TestHamiltonProduct(unittest.TestCase):

    def test_identity_is_neutral_right(self):
        q = Quaternion.from_euler_angles(30, 45, 60)
        assert_quat_close(self, q * Quaternion.identity(), q)

    def test_identity_is_neutral_left(self):
        q = Quaternion.from_euler_angles(30, 45, 60)
        assert_quat_close(self, Quaternion.identity() * q, q)

    def test_mul_inverse_gives_identity(self):
        q = Quaternion.from_euler_angles(10, 20, 30)
        assert_quat_close(self, q * q.inverse(), Quaternion.identity())

    def test_mul_non_commutative(self):
        q1 = Quaternion.from_euler_angles(90, 0, 0)
        q2 = Quaternion.from_euler_angles(0, 90, 0)
        self.assertFalse((q1 * q2).isclose(q2 * q1))

    def test_mul_unsupported_type(self):
        self.assertIs(Quaternion.identity().__mul__(42), NotImplemented)

    def test_composition_known_result(self):
        """Two 90° yaw rotations should equal one 180° yaw rotation."""
        q90 = Quaternion.from_euler_angles(0, 0, 90)
        q180 = Quaternion.from_euler_angles(0, 0, 180)
        assert_quat_close(self, q90 * q90, q180)

    def test_triple_product_associative(self):
        q1 = Quaternion.from_euler_angles(10, 20, 30)
        q2 = Quaternion.from_euler_angles(5, -10, 45)
        q3 = Quaternion.from_euler_angles(-30, 15, 60)
        assert_quat_close(self, (q1 * q2) * q3, q1 * (q2 * q3))


# ===========================================================================
# 7. Euler angles round-trip
# ===========================================================================

class TestEulerAngles(unittest.TestCase):

    def _round_trip(self, roll, pitch, yaw):
        q = Quaternion.from_euler_angles(roll, pitch, yaw, degrees=True)
        e = q.to_euler_angles(degrees=True)
        q2 = Quaternion.from_euler_angles(e.x, e.y, e.z, degrees=True)
        assert_quat_close(self, q, q2, atol=1e-7)

    def test_euler_round_trip_0_0_0(self):           self._round_trip(0, 0, 0)

    def test_euler_round_trip_90_0_0(self):          self._round_trip(90, 0, 0)

    def test_euler_round_trip_0_45_0(self):          self._round_trip(0, 45, 0)

    def test_euler_round_trip_0_0_neg90(self):       self._round_trip(0, 0, -90)

    def test_euler_round_trip_30_45_60(self):        self._round_trip(30, 45, 60)

    def test_euler_round_trip_neg45_20_neg30(self):  self._round_trip(-45, 20, -30)

    def test_euler_round_trip_180_0_0(self):         self._round_trip(180, 0, 0)

    def test_euler_round_trip_near_gimbal(self):     self._round_trip(0, -89, 0)

    def test_euler_radians(self):
        roll, pitch, yaw = 0.5, -0.3, 1.2
        q = Quaternion.from_euler_angles(roll, pitch, yaw, degrees=False)
        e = q.to_euler_angles(degrees=False)
        q2 = Quaternion.from_euler_angles(e.x, e.y, e.z, degrees=False)
        assert_quat_close(self, q, q2, atol=1e-7)

    def test_identity_euler(self):
        e = Quaternion.identity().to_euler_angles(degrees=True)
        self.assertAlmostEqual(e.x, 0.0, delta=1e-9)
        self.assertAlmostEqual(e.y, 0.0, delta=1e-9)
        self.assertAlmostEqual(e.z, 0.0, delta=1e-9)

    def test_gimbal_lock_positive(self):
        """pitch = +90° is a gimbal-lock singularity — should not raise, just clamp."""
        q = Quaternion.from_euler_angles(0, 90, 0, degrees=True)
        e = q.to_euler_angles(degrees=True)
        self.assertAlmostEqual(e.y, 90.0, delta=1e-6)

    def test_gimbal_lock_negative(self):
        q = Quaternion.from_euler_angles(0, -90, 0, degrees=True)
        e = q.to_euler_angles(degrees=True)
        self.assertAlmostEqual(e.y, -90.0, delta=1e-6)

    def test_from_euler_produces_unit_quaternion(self):
        q = Quaternion.from_euler_angles(37, -22, 111)
        self.assertAlmostEqual(q.norm(), 1.0, delta=1e-12)


# ===========================================================================
# 8. Rotation matrix round-trip
# ===========================================================================

class TestMatrix(unittest.TestCase):

    def test_identity_matrix(self):
        m = Quaternion.identity().to_matrix()
        np.testing.assert_allclose(m, np.eye(3), atol=1e-12)

    def test_matrix_is_orthonormal(self):
        q = Quaternion.from_euler_angles(30, 45, 60)
        m = q.to_matrix()
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(m)), 1.0, delta=1e-12)

    def _matrix_round_trip(self, roll, pitch, yaw):
        q = Quaternion.from_euler_angles(roll, pitch, yaw)
        q2 = Quaternion.from_matrix(q.to_matrix())
        assert_quat_close(self, q, q2, atol=1e-9)

    def test_matrix_round_trip_0_0_0(self):         self._matrix_round_trip(0, 0, 0)

    def test_matrix_round_trip_90_0_0(self):        self._matrix_round_trip(90, 0, 0)

    def test_matrix_round_trip_0_45_0(self):        self._matrix_round_trip(0, 45, 0)

    def test_matrix_round_trip_0_0_90(self):        self._matrix_round_trip(0, 0, 90)

    def test_matrix_round_trip_30_45_60(self):      self._matrix_round_trip(30, 45, 60)

    def test_matrix_round_trip_neg60_30_neg45(self): self._matrix_round_trip(-60, 30, -45)

    def test_from_matrix_bad_shape(self):
        with self.assertRaisesRegex(ValueError, "3x3"):
            Quaternion.from_matrix(np.eye(4))

    def test_from_matrix_branch_trace_positive(self):
        q = Quaternion.from_euler_angles(0, 0, 0)
        assert_quat_close(self, Quaternion.from_matrix(q.to_matrix()), q, atol=1e-9)

    def test_from_matrix_branch_x_dominant(self):
        q = Quaternion.from_euler_angles(170, 5, 5)
        assert_quat_close(self, Quaternion.from_matrix(q.to_matrix()), q, atol=1e-9)

    def test_from_matrix_branch_y_dominant(self):
        q = Quaternion.from_euler_angles(5, 170, 5)
        assert_quat_close(self, Quaternion.from_matrix(q.to_matrix()), q, atol=1e-9)

    def test_from_matrix_branch_z_dominant(self):
        q = Quaternion.from_euler_angles(5, 5, 170)
        assert_quat_close(self, Quaternion.from_matrix(q.to_matrix()), q, atol=1e-9)


# ===========================================================================
# 9. from_up_forward
# ===========================================================================

class TestFromUpForward(unittest.TestCase):

    def test_default_forward(self):
        up = Vector3(0.0, 1.0, 0.0)
        q = Quaternion.from_up_forward(up)
        m = q.to_matrix()
        # up direction maps to second column of rotation matrix
        np.testing.assert_allclose(m[:, 1], [0.0, 1.0, 0.0], atol=1e-9)

    def test_explicit_forward(self):
        up = Vector3(0.0, 1.0, 0.0)
        forward = Vector3(1.0, 0.0, 0.0)
        q = Quaternion.from_up_forward(up, forward)
        self.assertAlmostEqual(q.norm(), 1.0, delta=1e-12)

    def test_collinear_raises(self):
        up = Vector3(0.0, 0.0, 1.0)
        forward = Vector3(0.0, 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "collinear"):
            Quaternion.from_up_forward(up, forward)

    def test_result_is_unit(self):
        up = Vector3(0.0, 1.0, 0.0)
        forward = Vector3(0.0, 0.0, 1.0)
        q = Quaternion.from_up_forward(up, forward)
        self.assertAlmostEqual(q.norm(), 1.0, delta=1e-12)

    def test_left_handed_differs_from_right_handed(self):
        # (0,1,0) / (0,0,1) is degenerate — both handedness yield identity.
        # Use a non-axis-aligned forward so the cross product direction matters.
        up = Vector3(0.0, 1.0, 0.0)
        forward = Vector3(1.0, 0.0, 0.0)
        q_rh = Quaternion.from_up_forward(up, forward, right_handed=True)
        q_lh = Quaternion.from_up_forward(up, forward, right_handed=False)
        self.assertFalse(q_rh.isclose(q_lh))


# ===========================================================================
# 10. rotate_by_euler
# ===========================================================================

class TestRotateByEuler(unittest.TestCase):

    def test_identity_plus_zero_rotation(self):
        q = Quaternion.identity()
        assert_quat_close(self, q.rotate_by_euler(0, 0, 0), Quaternion.identity())

    def test_cumulative_yaw(self):
        q = Quaternion.from_euler_angles(0, 0, 45)
        q2 = q.rotate_by_euler(0, 0, 45)
        expected = Quaternion.from_euler_angles(0, 0, 90)
        assert_quat_close(self, q2, expected, atol=1e-9)

    def test_radians_mode(self):
        q = Quaternion.identity()
        result = q.rotate_by_euler(0, 0, math.pi / 2, degrees=False)
        expected = Quaternion.from_euler_angles(0, 0, 90, degrees=True)
        assert_quat_close(self, result, expected, atol=1e-9)

    def test_local_frame_order(self):
        """rotate_by_euler applies in the local (body) frame."""
        q_base = Quaternion.from_euler_angles(0, 0, 90)
        delta = Quaternion.from_euler_angles(90, 0, 0)
        expected = q_base * delta
        result = q_base.rotate_by_euler(90, 0, 0)
        assert_quat_close(self, result, expected, atol=1e-9)


# ===========================================================================
# 11. Mathematical identities
# ===========================================================================

class TestMathematicalIdentities(unittest.TestCase):

    def test_unit_quaternion_inverse_equals_conjugate(self):
        q = Quaternion.from_euler_angles(20, 50, -70)
        assert_quat_close(self, q.inverse(), q.conjugate())

    def test_double_negation(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        assert_quat_close(self, -(-q), q)

    def test_norm_preserved_under_product(self):
        q1 = Quaternion.from_euler_angles(10, 20, 30)
        q2 = Quaternion.from_euler_angles(40, 50, 60)
        product = q1 * q2
        self.assertAlmostEqual(product.norm(), q1.norm() * q2.norm(), delta=1e-12)

    def test_conjugate_of_product(self):
        """(q1 * q2)* == q2* * q1*"""
        q1 = Quaternion.from_euler_angles(10, 20, 30)
        q2 = Quaternion.from_euler_angles(40, 50, 60)
        lhs = (q1 * q2).conjugate()
        rhs = q2.conjugate() * q1.conjugate()
        assert_quat_close(self, lhs, rhs)

    def test_norm_of_conjugate(self):
        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertAlmostEqual(q.conjugate().norm(), q.norm(), delta=1e-12)

    def test_identity_is_unit(self):
        self.assertAlmostEqual(Quaternion.identity().norm(), 1.0, delta=1e-12)

    def test_dot_is_symmetric(self):
        q1 = Quaternion(1.0, 2.0, 3.0, 4.0)
        q2 = Quaternion(5.0, 6.0, 7.0, 8.0)
        self.assertAlmostEqual(q1.dot(q2), q2.dot(q1), delta=1e-12)

    def test_rotation_180_twice_is_identity(self):
        q = Quaternion.from_euler_angles(0, 0, 180)
        assert_quat_close(self, q * q, Quaternion.identity(), atol=1e-9)

    def test_matrix_composition_matches_quaternion_product(self):
        q1 = Quaternion.from_euler_angles(10, 20, 30)
        q2 = Quaternion.from_euler_angles(-5, 40, 15)
        m_combined = q1.to_matrix() @ q2.to_matrix()
        q_combined = q1 * q2
        np.testing.assert_allclose(q_combined.to_matrix(), m_combined, atol=1e-12)


class TestRotateVector(unittest.TestCase):

    def test_identity_leaves_vector_unchanged(self):
        q = Quaternion.identity()
        v = Vector3(1.0, 2.0, 3.0)
        assert_vec3_close(self, q.rotate_vector(v), v)

    def test_90_roll_rotates_up_to_forward(self):
        # 90° roll (around X): +Y -> +Z, +Z -> -Y
        q = Quaternion.from_euler_angles(90, 0, 0)
        assert_vec3_close(self, q.rotate_vector(Vector3.up()), Vector3.forward(), atol=1e-9)

    def test_90_pitch_rotates_right_to_backward(self):
        # 90° pitch (around Y): +X -> -Z, +Z -> +X
        q = Quaternion.from_euler_angles(0, 90, 0)
        assert_vec3_close(self, q.rotate_vector(Vector3.right()), Vector3.backward(), atol=1e-9)

    def test_90_yaw_rotates_right_to_up(self):
        # 90° yaw (around Z): +X -> +Y, +Y -> -X
        q = Quaternion.from_euler_angles(0, 0, 90)
        assert_vec3_close(self, q.rotate_vector(Vector3.right()), Vector3.up(), atol=1e-9)

    def test_180_yaw_reverses_right(self):
        # 180° yaw (around Z): +X -> -X
        q = Quaternion.from_euler_angles(0, 0, 180)
        assert_vec3_close(self, q.rotate_vector(Vector3.right()), Vector3.left(), atol=1e-9)

    def test_preserves_vector_length(self):
        q = Quaternion.from_euler_angles(37, -22, 111)
        v = Vector3(1.0, 2.0, 3.0)
        self.assertAlmostEqual(q.rotate_vector(v).norm(), v.norm(), delta=1e-12)

    def test_zero_vector_stays_zero(self):
        q = Quaternion.from_euler_angles(30, 45, 60)
        assert_vec3_close(self, q.rotate_vector(Vector3.zero()), Vector3.zero())

    def test_composition_matches_sequential_rotation(self):
        # Rotating by q1*q2 should equal rotating by q2 then q1
        q1 = Quaternion.from_euler_angles(30, 0, 0)
        q2 = Quaternion.from_euler_angles(0, 45, 0)
        v = Vector3(1.0, 2.0, 3.0)
        combined = (q1 * q2).rotate_vector(v)
        sequential = q1.rotate_vector(q2.rotate_vector(v))
        assert_vec3_close(self, combined, sequential, atol=1e-9)

    def test_inverse_rotation_roundtrip(self):
        q = Quaternion.from_euler_angles(30, 45, 60)
        v = Vector3(1.0, 2.0, 3.0)
        assert_vec3_close(self, q.inverse().rotate_vector(q.rotate_vector(v)), v, atol=1e-9)

    def test_returns_vector3_instance(self):
        q = Quaternion.from_euler_angles(10, 20, 30)
        result = q.rotate_vector(Vector3(1.0, 0.0, 0.0))
        self.assertIsInstance(result, Vector3)


if __name__ == "__main__":
    unittest.main()
