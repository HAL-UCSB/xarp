import unittest

from pydantic import ValidationError

from xarp.spatial import Vector2


class TestVector2(unittest.TestCase):
    def test_geometry_and_serialization(self):
        vector = Vector2(x=3, y=4)
        self.assertEqual(vector.to_tuple(), (3, 4))
        self.assertEqual(vector.norm(), 5)
        self.assertEqual(vector + Vector2.one(), Vector2(4, 5))
        self.assertEqual(Vector2.model_validate_json(vector.model_dump_json()), vector)

    def test_copied_vector_supports_geometry(self):
        vector = Vector2(3, 4).model_copy(deep=True)
        self.assertEqual(vector.norm(), 5)

    def test_rejects_wrong_dimension(self):
        with self.assertRaises(ValidationError):
            Vector2.model_validate([1, 2, 3])
