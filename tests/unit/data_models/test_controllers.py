"""Controller data contract tests; no Quest capture or networking is exercised."""

import unittest

from pydantic import ValidationError

from xarp.data_models import Controller, ControllerButtons, Controllers
from xarp.spatial import Pose


class TestControllers(unittest.TestCase):
    def test_unavailable_defaults(self):
        controllers = Controllers()
        for side in (controllers.left, controllers.right):
            self.assertFalse(side.connected)
            self.assertFalse(side.tracked)
            self.assertIsNone(side.grip_pose)
            self.assertEqual((side.trigger, side.grip), (0.0, 0.0))
            self.assertEqual(side.thumbstick, (0.0, 0.0))
            self.assertFalse(side.buttons.primary)
            self.assertFalse(side.buttons.secondary)
        self.assertIsNot(controllers.left, controllers.right)

    def test_parse_representative_payload_and_round_trip(self):
        payload = {
            "left": {"connected": True, "tracked": False,
                     "grip_pose": None, "buttons": {"primary": True}},
            "right": {"connected": True, "tracked": True,
                      "grip_pose": {"position": [1, 2, 3],
                                    "rotation": [0, 0, 0, 1]},
                      "trigger": 0.8, "grip": 0.5,
                      "thumbstick": [-1, 1],
                      "buttons": {"primary": False, "secondary": True}},
        }
        controllers = Controllers.model_validate(payload)
        self.assertTrue(controllers.left.buttons.primary)
        self.assertFalse(controllers.left.tracked)
        self.assertIsNone(controllers.left.grip_pose)
        right = controllers.right
        self.assertIsInstance(right.grip_pose, Pose)
        self.assertEqual(tuple(right.grip_pose.position), (1, 2, 3))
        self.assertEqual(tuple(right.grip_pose.rotation), (0, 0, 0, 1))
        self.assertEqual((right.trigger, right.grip), (0.8, 0.5))
        self.assertEqual(right.thumbstick, (-1.0, 1.0))
        self.assertTrue(right.buttons.secondary)
        self.assertEqual(Controllers.model_validate_json(controllers.model_dump_json()),
                         controllers)

    def test_connected_controller_can_have_missing_pose(self):
        controller = Controller(connected=True, tracked=True, grip_pose=None)
        self.assertIsNone(controller.grip_pose)

    def test_analog_boundaries(self):
        for value in (0.0, 1.0):
            with self.subTest(value=value):
                controller = Controller(trigger=value, grip=value)
                self.assertEqual((controller.trigger, controller.grip), (value, value))

    def test_invalid_analog_values_rejected(self):
        for name in ("trigger", "grip"):
            for value in (-0.01, 1.01, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValidationError):
                        Controller(**{name: value})

    def test_invalid_thumbsticks_rejected(self):
        for value in ((-1.01, 0), (0, 1.01), (float("nan"), 0),
                      (0, float("inf")), (0,), (0, 0, 0)):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    Controller(thumbstick=value)

    def test_unknown_fields_rejected_at_each_level(self):
        for payload in ({"center": {}}, {"left": {"unknown": True}},
                        {"right": {"buttons": {"unknown": True}}}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    Controllers.model_validate(payload)

    def test_models_are_frozen(self):
        for model, field, value in (
            (Controllers(), "left", Controller()),
            (Controller(), "trigger", 1.0),
            (ControllerButtons(), "primary", True),
        ):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(ValidationError):
                    setattr(model, field, value)


if __name__ == "__main__":
    unittest.main()
