import unittest

from pydantic import ValidationError
from xarp.data_models import CameraIntrinsics, DeviceInfo, Hands
from xarp.spatial import Pose, Quaternion, Vector3


# ---------------------------------------------------------------------------
# Hands
# ---------------------------------------------------------------------------

class TestHands(unittest.TestCase):

    def setUp(self):
        self.pose_a = Pose(position=Vector3(1.0, 0.0, 0.0))
        self.pose_b = Pose(position=Vector3(2.0, 0.0, 0.0))
        self.pose_c = Pose(position=Vector3(3.0, 0.0, 0.0))
        self.hands = Hands(
            left=(self.pose_a, self.pose_b),
            right=(self.pose_c,),
        )

    def test_getitem_left(self):
        self.assertEqual(self.hands["left"], (self.pose_a, self.pose_b))

    def test_getitem_right(self):
        self.assertEqual(self.hands["right"], (self.pose_c,))

    def test_getitem_pose_by_index(self):
        self.assertEqual(self.hands["left"][0], self.pose_a)
        self.assertEqual(self.hands["left"][1], self.pose_b)

    def test_getitem_invalid_key_raises(self):
        with self.assertRaises(KeyError):
            _ = self.hands["center"]

    def test_iter_yields_left_then_right(self):
        left, right = self.hands
        self.assertEqual(left, (self.pose_a, self.pose_b))
        self.assertEqual(right, (self.pose_c,))

    def test_iter_order(self):
        sides = list(self.hands)
        self.assertIs(sides[0], self.hands.left)
        self.assertIs(sides[1], self.hands.right)

    def test_frozen_mutation_raises(self):
        with self.assertRaises(ValidationError):
            self.hands.left = ()

    def test_empty_defaults(self):
        h = Hands()
        self.assertEqual(h.left, ())
        self.assertEqual(h.right, ())

    def test_extra_fields_rejected(self):
        with self.assertRaises(ValidationError):
            Hands(left=(), right=(), center=())


if __name__ == "__main__":
    unittest.main()
