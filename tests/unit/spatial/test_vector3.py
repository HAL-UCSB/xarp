import math
import unittest

import numpy as np
from pydantic import ValidationError

from xarp.spatial import Vector3


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def assert_vec_close(test: unittest.TestCase, a: Vector3, b: Vector3, atol: float = 1e-9) -> None:
    test.assertTrue(a.isclose(b, atol=atol), f"Expected {a} ≈ {b}")


# -----------------------------------------------------------------------
# Construction & Validation
# -----------------------------------------------------------------------

class TestConstructor(unittest.TestCase):
    def test_positional_args(self):
        v = Vector3(1.0, 2.0, 3.0)
        self.assertEqual((v.x, v.y, v.z), (1.0, 2.0, 3.0))

    def test_positional_args_int_coerced(self):
        v = Vector3(1, 2, 3)
        self.assertIsInstance(v.x, float)
        self.assertIsInstance(v.y, float)
        self.assertIsInstance(v.z, float)

    def test_from_list(self):
        v = Vector3([1.0, 2.0, 3.0])
        self.assertEqual((v.x, v.y, v.z), (1.0, 2.0, 3.0))

    def test_from_tuple(self):
        v = Vector3((1.0, 2.0, 3.0))
        self.assertEqual((v.x, v.y, v.z), (1.0, 2.0, 3.0))

    def test_from_numpy(self):
        v = Vector3(np.array([1.0, 2.0, 3.0]))
        self.assertEqual((v.x, v.y, v.z), (1.0, 2.0, 3.0))

    def test_kwargs_still_works(self):
        v = Vector3(x=1.0, y=2.0, z=3.0)
        self.assertEqual((v.x, v.y, v.z), (1.0, 2.0, 3.0))

    def test_positional_wrong_count_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            Vector3(1.0, 2.0)

    def test_sequence_wrong_length_raises(self):
        with self.assertRaises((ValueError, TypeError, IndexError)):
            Vector3([1.0, 2.0])

    def test_sequence_too_long_raises(self):
        with self.assertRaises((ValueError, TypeError, IndexError)):
            Vector3([1.0, 2.0, 3.0, 4.0])

    def test_all_forms_produce_equal_vectors(self):
        expected = Vector3(x=1.0, y=2.0, z=3.0)
        self.assertEqual(Vector3(1.0, 2.0, 3.0), expected)
        self.assertEqual(Vector3([1.0, 2.0, 3.0]), expected)
        self.assertEqual(Vector3((1.0, 2.0, 3.0)), expected)
        self.assertEqual(Vector3(np.array([1.0, 2.0, 3.0])), expected)

    def test_invalid_type_raises(self):
        with self.assertRaises((ValueError, TypeError, ValidationError)):
            Vector3("a", "b", "c")


class TestConstruction(unittest.TestCase):
    def test_basic(self):
        v = Vector3(x=1.0, y=2.0, z=3.0)
        self.assertEqual(v.x, 1.0)
        self.assertEqual(v.y, 2.0)
        self.assertEqual(v.z, 3.0)

    def test_int_inputs_coerced_to_float(self):
        v = Vector3(x=1, y=2, z=3)
        self.assertIsInstance(v.x, float)
        self.assertIsInstance(v.y, float)
        self.assertIsInstance(v.z, float)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            Vector3(x="a", y=0.0, z=0.0)

    def test_missing_field_raises(self):
        with self.assertRaises(ValidationError):
            Vector3(x=1.0, y=2.0)

    def test_from_sequence_list(self):
        self.assertEqual(Vector3.from_sequence([1.0, 2.0, 3.0]), Vector3(x=1.0, y=2.0, z=3.0))

    def test_from_sequence_tuple(self):
        self.assertEqual(Vector3.from_sequence((1.0, 2.0, 3.0)), Vector3(x=1.0, y=2.0, z=3.0))

    def test_from_sequence_numpy(self):
        self.assertEqual(Vector3.from_sequence(np.array([1.0, 2.0, 3.0])), Vector3(x=1.0, y=2.0, z=3.0))

    def test_from_sequence_too_short_raises(self):
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            Vector3.from_sequence([1.0, 2.0])

    def test_from_sequence_too_long_raises(self):
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            Vector3.from_sequence([1.0, 2.0, 3.0, 4.0])


# -----------------------------------------------------------------------
# Immutability
# -----------------------------------------------------------------------

class TestImmutability(unittest.TestCase):
    def test_field_assignment_raises(self):
        v = Vector3(x=1.0, y=2.0, z=3.0)
        with self.assertRaises(Exception):
            v.x = 99.0

    def test_numpy_array_is_copy(self):
        v = Vector3(x=1.0, y=2.0, z=3.0)
        arr = v.to_numpy()
        arr[0] = 999.0
        self.assertEqual(v.x, 1.0)


# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

class TestConstants(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(Vector3.zero(), Vector3(x=0.0, y=0.0, z=0.0))

    def test_one(self):
        self.assertEqual(Vector3.one(), Vector3(x=1.0, y=1.0, z=1.0))

    def test_forward(self):
        self.assertEqual(Vector3.forward(), Vector3(x=0.0, y=0.0, z=1.0))

    def test_backward(self):
        self.assertEqual(Vector3.backward(), Vector3(x=0.0, y=0.0, z=-1.0))

    def test_up(self):
        self.assertEqual(Vector3.up(), Vector3(x=0.0, y=1.0, z=0.0))

    def test_down(self):
        self.assertEqual(Vector3.down(), Vector3(x=0.0, y=-1.0, z=0.0))

    def test_right(self):
        self.assertEqual(Vector3.right(), Vector3(x=1.0, y=0.0, z=0.0))

    def test_left(self):
        self.assertEqual(Vector3.left(), Vector3(x=-1.0, y=0.0, z=0.0))

    def test_direction_constants_are_unit_length(self):
        for v in [Vector3.forward(), Vector3.backward(), Vector3.up(), Vector3.down(), Vector3.right(), Vector3.left()]:
            self.assertAlmostEqual(v.norm(), 1.0)

    def test_opposite_directions_sum_to_zero(self):
        assert_vec_close(self, Vector3.forward() + Vector3.backward(), Vector3.zero())
        assert_vec_close(self, Vector3.up() + Vector3.down(), Vector3.zero())
        assert_vec_close(self, Vector3.right() + Vector3.left(), Vector3.zero())

    def test_forward_and_up_are_orthogonal(self):
        self.assertAlmostEqual(Vector3.forward().dot(Vector3.up()), 0.0)

    def test_forward_and_backward_are_antiparallel(self):
        self.assertAlmostEqual(Vector3.forward().dot(Vector3.backward()), -1.0)

    def test_up_and_down_are_antiparallel(self):
        self.assertAlmostEqual(Vector3.up().dot(Vector3.down()), -1.0)

    def test_right_and_left_are_antiparallel(self):
        self.assertAlmostEqual(Vector3.right().dot(Vector3.left()), -1.0)


# -----------------------------------------------------------------------
# Sequence Protocol
# -----------------------------------------------------------------------

class TestSequenceProtocol(unittest.TestCase):
    def test_len(self):
        self.assertEqual(len(Vector3(x=1.0, y=2.0, z=3.0)), 3)

    def test_unpack(self):
        x, y, z = Vector3(x=1.0, y=2.0, z=3.0)
        self.assertEqual((x, y, z), (1.0, 2.0, 3.0))

    def test_getitem(self):
        v = Vector3(x=1.0, y=2.0, z=3.0)
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 2.0)
        self.assertEqual(v[2], 3.0)

    def test_getitem_out_of_range(self):
        with self.assertRaises(IndexError):
            _ = Vector3(x=1.0, y=2.0, z=3.0)[3]

    def test_negative_index(self):
        self.assertEqual(Vector3(x=1.0, y=2.0, z=3.0)[-1], 3.0)


# -----------------------------------------------------------------------
# Conversion
# -----------------------------------------------------------------------

class TestConversion(unittest.TestCase):
    def test_to_numpy_dtype(self):
        self.assertEqual(Vector3(x=1.0, y=2.0, z=3.0).to_numpy().dtype, np.float64)

    def test_to_numpy_values(self):
        np.testing.assert_array_equal(Vector3(x=1.0, y=2.0, z=3.0).to_numpy(), [1.0, 2.0, 3.0])

    def test_to_list(self):
        self.assertEqual(Vector3(x=1.0, y=2.0, z=3.0).to_list(), [1.0, 2.0, 3.0])

    def test_to_tuple(self):
        self.assertEqual(Vector3(x=1.0, y=2.0, z=3.0).to_tuple(), (1.0, 2.0, 3.0))

    def test_json_roundtrip(self):
        v = Vector3(x=1.0, y=2.0, z=3.0)
        self.assertEqual(Vector3.model_validate_json(v.model_dump_json()), v)

    def test_dict_roundtrip(self):
        v = Vector3(x=1.0, y=2.0, z=3.0)
        self.assertEqual(Vector3.model_validate(v.model_dump()), v)


# -----------------------------------------------------------------------
# Equality & Hashing
# -----------------------------------------------------------------------
class TestEqualityAndHashing(unittest.TestCase):

    def test_equal(self):
        self.assertEqual(Vector3(1.0, 2.0, 3.0), Vector3(1.0, 2.0, 3.0))

    def test_not_equal(self):
        self.assertNotEqual(Vector3(1.0, 2.0, 3.0), Vector3(1.0, 2.0, 4.0))

    def test_not_equal_to_non_vector(self):
        self.assertNotEqual(Vector3(1.0, 2.0, 3.0), (1.0, 2.0, 3.0))

    def test_hash_stable(self):
        v = Vector3(1.0, 2.0, 3.0)
        self.assertEqual(hash(v), hash(v))

    def test_equal_vectors_have_equal_hashes(self):
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(1.0, 2.0, 3.0)
        self.assertEqual(hash(a), hash(b))

    def test_usable_as_dict_key(self):
        d = {Vector3(1.0, 0.0, 0.0): "right"}
        self.assertEqual(d[Vector3(1.0, 0.0, 0.0)], "right")

    def test_usable_in_set(self):
        s = {Vector3(1.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0)}
        self.assertEqual(len(s), 1)

    def test_isclose_true(self):
        self.assertTrue(Vector3(1.0, 0.0, 0.0).isclose(Vector3(1.0 + 1e-10, 0.0, 0.0)))

    def test_isclose_false(self):
        self.assertFalse(Vector3(1.0, 0.0, 0.0).isclose(Vector3(1.1, 0.0, 0.0)))

    def test_isclose_custom_atol_within(self):
        self.assertTrue(Vector3(1.0, 0.0, 0.0).isclose(Vector3(1.05, 0.0, 0.0), atol=0.1))

    def test_isclose_custom_atol_outside(self):
        self.assertFalse(Vector3(1.0, 0.0, 0.0).isclose(Vector3(1.05, 0.0, 0.0), atol=0.01))


# -----------------------------------------------------------------------
# Arithmetic
# -----------------------------------------------------------------------

class TestArithmetic(unittest.TestCase):
    def test_add(self):
        result = Vector3(x=1.0, y=2.0, z=3.0) + Vector3(x=4.0, y=5.0, z=6.0)
        self.assertEqual(result, Vector3(x=5.0, y=7.0, z=9.0))

    def test_add_invalid_type_returns_not_implemented(self):
        self.assertIs(Vector3(x=1.0, y=0.0, z=0.0).__add__(5.0), NotImplemented)

    def test_sub(self):
        result = Vector3(x=4.0, y=5.0, z=6.0) - Vector3(x=1.0, y=2.0, z=3.0)
        self.assertEqual(result, Vector3(x=3.0, y=3.0, z=3.0))

    def test_rsub_correctness(self):
        a = Vector3(x=1.0, y=0.0, z=0.0)
        b = Vector3(x=3.0, y=0.0, z=0.0)
        # b.__rsub__(a) should compute a - b
        self.assertEqual(b.__rsub__(a), Vector3(x=-2.0, y=0.0, z=0.0))

    def test_rsub_not_commutative(self):
        a = Vector3(x=1.0, y=2.0, z=3.0)
        b = Vector3(x=4.0, y=6.0, z=8.0)
        self.assertNotEqual(a - b, b - a)

    def test_mul_scalar(self):
        self.assertEqual(Vector3(x=1.0, y=2.0, z=3.0) * 2.0, Vector3(x=2.0, y=4.0, z=6.0))

    def test_rmul_scalar(self):
        self.assertEqual(2.0 * Vector3(x=1.0, y=2.0, z=3.0), Vector3(x=2.0, y=4.0, z=6.0))

    def test_mul_invalid_type_returns_not_implemented(self):
        self.assertIs(Vector3(x=1.0, y=0.0, z=0.0).__mul__("2"), NotImplemented)

    def test_div_scalar(self):
        self.assertEqual(Vector3(x=2.0, y=4.0, z=6.0) / 2.0, Vector3(x=1.0, y=2.0, z=3.0))

    def test_div_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            _ = Vector3(x=1.0, y=0.0, z=0.0) / 0.0

    def test_neg(self):
        self.assertEqual(-Vector3(x=1.0, y=-2.0, z=3.0), Vector3(x=-1.0, y=2.0, z=-3.0))

    def test_abs_returns_norm(self):
        self.assertAlmostEqual(abs(Vector3(x=3.0, y=0.0, z=4.0)), 5.0)

    def test_add_returns_new_instance(self):
        a = Vector3(x=1.0, y=0.0, z=0.0)
        b = Vector3(x=0.0, y=1.0, z=0.0)
        c = a + b
        self.assertIsNot(c, a)
        self.assertIsNot(c, b)

    def test_arithmetic_chain(self):
        result = (Vector3(x=1.0, y=0.0, z=0.0) + Vector3(x=0.0, y=1.0, z=0.0)) * 3.0
        self.assertEqual(result, Vector3(x=3.0, y=3.0, z=0.0))


# -----------------------------------------------------------------------
# Norm
# -----------------------------------------------------------------------

class TestNorm(unittest.TestCase):
    def test_norm_unit_vector(self):
        self.assertAlmostEqual(Vector3(x=1.0, y=0.0, z=0.0).norm(), 1.0)

    def test_norm_known_value(self):
        self.assertAlmostEqual(Vector3(x=3.0, y=0.0, z=4.0).norm(), 5.0)

    def test_norm_zero_vector(self):
        self.assertAlmostEqual(Vector3.zero().norm(), 0.0)

    def test_norm_squared(self):
        self.assertAlmostEqual(Vector3(x=3.0, y=0.0, z=4.0).norm_squared(), 25.0)

    def test_norm_squared_consistent_with_norm(self):
        v = Vector3(x=1.0, y=2.0, z=3.0)
        self.assertAlmostEqual(v.norm_squared(), v.norm() ** 2)


# -----------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------

class TestNormalized(unittest.TestCase):
    def test_normalized_is_unit(self):
        self.assertAlmostEqual(Vector3(x=3.0, y=0.0, z=4.0).normalized().norm(), 1.0)

    def test_normalized_direction_preserved(self):
        assert_vec_close(self, Vector3(x=3.0, y=0.0, z=0.0).normalized(), Vector3(x=1.0, y=0.0, z=0.0))

    def test_normalize_already_unit(self):
        v = Vector3(x=1.0, y=0.0, z=0.0)
        assert_vec_close(self, v.normalized(), v)

    def test_normalize_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            Vector3.zero().normalized()

    def test_normalize_near_zero_raises(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            Vector3(x=1e-200, y=0.0, z=0.0).normalized()

    def test_normalize_negative(self):
        assert_vec_close(self, Vector3(x=-3.0, y=0.0, z=0.0).normalized(), Vector3(x=-1.0, y=0.0, z=0.0))


# -----------------------------------------------------------------------
# Dot & Cross
# -----------------------------------------------------------------------

class TestDotAndCross(unittest.TestCase):
    def test_dot_orthogonal(self):
        self.assertAlmostEqual(
            Vector3(x=1.0, y=0.0, z=0.0).dot(Vector3(x=0.0, y=1.0, z=0.0)), 0.0
        )

    def test_dot_parallel(self):
        self.assertAlmostEqual(
            Vector3(x=2.0, y=0.0, z=0.0).dot(Vector3(x=3.0, y=0.0, z=0.0)), 6.0
        )

    def test_dot_commutative(self):
        a = Vector3(x=1.0, y=2.0, z=3.0)
        b = Vector3(x=4.0, y=5.0, z=6.0)
        self.assertAlmostEqual(a.dot(b), b.dot(a))

    def test_dot_antiparallel(self):
        self.assertAlmostEqual(
            Vector3(x=1.0, y=0.0, z=0.0).dot(Vector3(x=-1.0, y=0.0, z=0.0)), -1.0
        )

    def test_cross_orthogonal_basis(self):
        assert_vec_close(self, Vector3.right().cross(Vector3.up()), Vector3.forward())
        assert_vec_close(self, Vector3.up().cross(Vector3.forward()), Vector3.right())
        assert_vec_close(self, Vector3.forward().cross(Vector3.right()), Vector3.up())

    def test_cross_anticommutative(self):
        a = Vector3(x=1.0, y=2.0, z=3.0)
        b = Vector3(x=4.0, y=5.0, z=6.0)
        assert_vec_close(self, a.cross(b), -(b.cross(a)))

    def test_cross_parallel_is_zero(self):
        assert_vec_close(self,
                         Vector3(x=2.0, y=0.0, z=0.0).cross(Vector3(x=5.0, y=0.0, z=0.0)),
                         Vector3.zero(),
                         )

    def test_cross_result_orthogonal_to_inputs(self):
        a = Vector3(x=1.0, y=2.0, z=3.0)
        b = Vector3(x=4.0, y=5.0, z=6.0)
        c = a.cross(b)
        self.assertAlmostEqual(c.dot(a), 0.0, places=9)
        self.assertAlmostEqual(c.dot(b), 0.0, places=9)


# -----------------------------------------------------------------------
# Projection
# -----------------------------------------------------------------------

class TestProjection(unittest.TestCase):
    def test_project_onto_parallel(self):
        v = Vector3(x=3.0, y=0.0, z=0.0)
        assert_vec_close(self, v.project_onto(Vector3(x=1.0, y=0.0, z=0.0)), v)

    def test_project_onto_orthogonal(self):
        assert_vec_close(self,
                         Vector3(x=0.0, y=3.0, z=0.0).project_onto(Vector3(x=1.0, y=0.0, z=0.0)),
                         Vector3.zero(),
                         )

    def test_project_onto_zero_raises(self):
        with self.assertRaises(ValueError):
            Vector3(x=1.0, y=0.0, z=0.0).project_onto(Vector3.zero())

    def test_project_onto_plane_on_plane(self):
        v = Vector3(x=3.0, y=0.0, z=4.0)
        assert_vec_close(self, v.project_onto_plane(Vector3.zero(), Vector3.up()), v)

    def test_project_onto_plane_above_plane(self):
        assert_vec_close(self,
                         Vector3(x=3.0, y=5.0, z=4.0).project_onto_plane(Vector3.zero(), Vector3.up()),
                         Vector3(x=3.0, y=0.0, z=4.0),
                         )

    def test_project_onto_plane_offset_origin(self):
        assert_vec_close(self,
                         Vector3(x=1.0, y=5.0, z=1.0).project_onto_plane(
                             Vector3(x=0.0, y=2.0, z=0.0), Vector3.up()
                         ),
                         Vector3(x=1.0, y=2.0, z=1.0),
                         )

    def test_project_onto_plane_zero_normal_raises(self):
        with self.assertRaises(ValueError):
            Vector3(x=1.0, y=1.0, z=1.0).project_onto_plane(Vector3.zero(), Vector3.zero())


# -----------------------------------------------------------------------
# Distance & Lerp
# -----------------------------------------------------------------------

class TestDistanceAndLerp(unittest.TestCase):
    def test_distance_to_self(self):
        v = Vector3(x=1.0, y=2.0, z=3.0)
        self.assertAlmostEqual(v.distance(v), 0.0)

    def test_distance_known(self):
        self.assertAlmostEqual(
            Vector3.zero().distance(Vector3(x=3.0, y=0.0, z=4.0)), 5.0
        )

    def test_distance_symmetric(self):
        a = Vector3(x=1.0, y=2.0, z=3.0)
        b = Vector3(x=4.0, y=5.0, z=6.0)
        self.assertAlmostEqual(a.distance(b), b.distance(a))

    def test_lerp_t0(self):
        a = Vector3(x=0.0, y=0.0, z=0.0)
        b = Vector3(x=10.0, y=0.0, z=0.0)
        assert_vec_close(self, a.lerp(b, 0.0), a)

    def test_lerp_t1(self):
        a = Vector3(x=0.0, y=0.0, z=0.0)
        b = Vector3(x=10.0, y=0.0, z=0.0)
        assert_vec_close(self, a.lerp(b, 1.0), b)

    def test_lerp_midpoint(self):
        assert_vec_close(self,
                         Vector3.zero().lerp(Vector3(x=10.0, y=0.0, z=0.0), 0.5),
                         Vector3(x=5.0, y=0.0, z=0.0),
                         )

    def test_lerp_extrapolation(self):
        assert_vec_close(self,
                         Vector3.zero().lerp(Vector3(x=10.0, y=0.0, z=0.0), 2.0),
                         Vector3(x=20.0, y=0.0, z=0.0),
                         )


# -----------------------------------------------------------------------
# Angle
# -----------------------------------------------------------------------

class TestAngle(unittest.TestCase):
    def test_angle_same_vector(self):
        self.assertAlmostEqual(Vector3.right().angle(Vector3.right()), 0.0)

    def test_angle_orthogonal(self):
        self.assertAlmostEqual(Vector3.right().angle(Vector3.up()), math.pi / 2)

    def test_angle_antiparallel(self):
        self.assertAlmostEqual(Vector3.right().angle(Vector3.left()), math.pi)

    def test_angle_symmetric(self):
        a = Vector3(x=1.0, y=2.0, z=3.0)
        b = Vector3(x=4.0, y=5.0, z=6.0)
        self.assertAlmostEqual(a.angle(b), b.angle(a))

    def test_angle_with_zero_raises(self):
        with self.assertRaises(ValueError):
            Vector3.right().angle(Vector3.zero())

    def test_angle_known_value(self):
        self.assertAlmostEqual(
            Vector3(x=1.0, y=0.0, z=0.0).angle(Vector3(x=1.0, y=1.0, z=0.0)),
            math.pi / 4,
        )


if __name__ == "__main__":
    unittest.main()
