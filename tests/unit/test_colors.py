import unittest

from xarp import colors
from xarp.spatial import Vector4


class TestColorConstants(unittest.TestCase):
    def test_constants_are_vector4_values(self):
        for name in colors.__all__:
            self.assertIsInstance(getattr(colors, name), Vector4)

    def test_common_values(self):
        self.assertEqual(colors.TRANSPARENT, Vector4(0.0, 0.0, 0.0, 0.0))
        self.assertEqual(colors.BLACK, Vector4(0.0, 0.0, 0.0, 1.0))
        self.assertEqual(colors.WHITE, Vector4(1.0, 1.0, 1.0, 1.0))
        self.assertEqual(colors.RED, Vector4(1.0, 0.0, 0.0, 1.0))
        self.assertEqual(colors.GREEN, Vector4(0.0, 1.0, 0.0, 1.0))
        self.assertEqual(colors.BLUE, Vector4(0.0, 0.0, 1.0, 1.0))
        self.assertEqual(colors.YELLOW, Vector4(1.0, 1.0, 0.0, 1.0))
        self.assertEqual(colors.CYAN, Vector4(0.0, 1.0, 1.0, 1.0))
        self.assertEqual(colors.MAGENTA, Vector4(1.0, 0.0, 1.0, 1.0))
        self.assertEqual(colors.GRAY, Vector4(0.5, 0.5, 0.5, 1.0))

    def test_grey_aliases_gray(self):
        self.assertIs(colors.GREY, colors.GRAY)

    def test_constants_are_immutable(self):
        with self.assertRaises(Exception):
            colors.RED.x = 0.5

    def test_constants_are_normalized(self):
        for name in colors.__all__:
            for component in getattr(colors, name):
                self.assertGreaterEqual(component, 0.0)
                self.assertLessEqual(component, 1.0)


if __name__ == "__main__":
    unittest.main()
