import math
import unittest

import numpy as np
from pydantic import ValidationError

from xarp.spatial import Vector4


def assert_vec_close(test: unittest.TestCase, a: Vector4, b: Vector4, atol: float = 1e-9) -> None:
    test.assertTrue(a.isclose(b, atol=atol), f"Expected {a} ~= {b}")


class TestConstructor(unittest.TestCase):
    def test_positional_args(self):
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        self.assertEqual((v.x, v.y, v.z, v.w), (1.0, 2.0, 3.0, 4.0))

    def test_positional_args_int_coerced(self):
        v = Vector4(1, 2, 3, 4)
        self.assertIsInstance(v.x, float)
        self.assertIsInstance(v.y, float)
        self.assertIsInstance(v.z, float)
        self.assertIsInstance(v.w, float)

    def test_from_sequence_inputs(self):
        expected = Vector4(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(Vector4([1.0, 2.0, 3.0, 4.0]), expected)
        self.assertEqual(Vector4((1.0, 2.0, 3.0, 4.0)), expected)
        self.assertEqual(Vector4(np.array([1.0, 2.0, 3.0, 4.0])), expected)

    def test_kwargs_still_work(self):
        self.assertEqual(Vector4(x=1, y=2, z=3, w=4), Vector4(1.0, 2.0, 3.0, 4.0))

    def test_from_xyzw(self):
        self.assertEqual(Vector4.from_xyzw(1, 2, 3, 4), Vector4(1.0, 2.0, 3.0, 4.0))

    def test_from_sequence(self):
        self.assertEqual(Vector4.from_sequence([1.0, 2.0, 3.0, 4.0]), Vector4(1.0, 2.0, 3.0, 4.0))

    def test_wrong_count_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            Vector4(1.0, 2.0, 3.0)
        with self.assertRaises((ValueError, TypeError, IndexError)):
            Vector4([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "Expected 4"):
            Vector4.from_sequence([1.0, 2.0, 3.0])

    def test_invalid_type_raises(self):
        with self.assertRaises((ValueError, TypeError, ValidationError)):
            Vector4("a", "b", "c", "d")


class TestImmutability(unittest.TestCase):
    def test_field_assignment_raises(self):
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        with self.assertRaises(Exception):
            v.x = 99.0

    def test_numpy_array_is_copy(self):
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        arr = v.to_numpy()
        arr[0] = 999.0
        self.assertEqual(v.x, 1.0)


class TestConstants(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(Vector4.zero(), Vector4(0.0, 0.0, 0.0, 0.0))

    def test_one(self):
        self.assertEqual(Vector4.one(), Vector4(1.0, 1.0, 1.0, 1.0))


class TestSequenceAndConversion(unittest.TestCase):
    def test_len_iter_and_index(self):
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(len(v), 4)
        self.assertEqual(tuple(v), (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(v[3], 4.0)

    def test_conversion(self):
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(v.to_list(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(v.to_tuple(), (1.0, 2.0, 3.0, 4.0))
        np.testing.assert_array_equal(v.to_numpy(), [1.0, 2.0, 3.0, 4.0])

    def test_model_round_trip(self):
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(Vector4.model_validate_json(v.model_dump_json()), v)
        self.assertEqual(Vector4.model_validate(v.model_dump()), v)
        self.assertEqual(v.model_dump(), [1.0, 2.0, 3.0, 4.0])


class TestArithmetic(unittest.TestCase):
    def test_add_subtract(self):
        a = Vector4(1.0, 2.0, 3.0, 4.0)
        b = Vector4(4.0, 3.0, 2.0, 1.0)
        self.assertEqual(a + b, Vector4(5.0, 5.0, 5.0, 5.0))
        self.assertEqual(a - b, Vector4(-3.0, -1.0, 1.0, 3.0))

    def test_scalar_multiply_divide_and_negate(self):
        v = Vector4(1.0, -2.0, 3.0, -4.0)
        self.assertEqual(v * 2.0, Vector4(2.0, -4.0, 6.0, -8.0))
        self.assertEqual(2.0 * v, Vector4(2.0, -4.0, 6.0, -8.0))
        self.assertEqual(v / 2.0, Vector4(0.5, -1.0, 1.5, -2.0))
        self.assertEqual(-v, Vector4(-1.0, 2.0, -3.0, 4.0))

    def test_divide_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            _ = Vector4.one() / 0.0


class TestVectorMath(unittest.TestCase):
    def test_norm_and_dot(self):
        v = Vector4(1.0, 2.0, 3.0, 4.0)
        self.assertAlmostEqual(v.norm(), math.sqrt(30.0))
        self.assertAlmostEqual(v.norm_squared(), 30.0)
        self.assertAlmostEqual(v.dot(Vector4.one()), 10.0)

    def test_normalized(self):
        v = Vector4(2.0, 0.0, 0.0, 0.0).normalized()
        assert_vec_close(self, v, Vector4(1.0, 0.0, 0.0, 0.0))

    def test_normalized_zero_raises(self):
        with self.assertRaises(ValueError):
            Vector4.zero().normalized()

    def test_distance_and_lerp(self):
        self.assertAlmostEqual(Vector4.zero().distance(Vector4(0.0, 0.0, 3.0, 4.0)), 5.0)
        self.assertEqual(Vector4.zero().lerp(Vector4(2.0, 4.0, 6.0, 8.0), 0.5), Vector4(1.0, 2.0, 3.0, 4.0))

    def test_isclose(self):
        self.assertTrue(Vector4(1.0, 0.0, 0.0, 0.0).isclose(Vector4(1.0 + 1e-10, 0.0, 0.0, 0.0)))
        self.assertFalse(Vector4(1.0, 0.0, 0.0, 0.0).isclose(Vector4(1.1, 0.0, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
