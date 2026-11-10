"""Controller data contract tests; no Quest capture or networking is exercised."""

import unittest

from pydantic import ValidationError

from xarp.data_models import Controller, ControllerButtons, Controllers
from xarp.spatial import Pose, Vector2


class TestControllers(unittest.TestCase):
    def test_unavailable_defaults(self):
        controllers = Controllers()
        self.assertIsNone(controllers.left)
        self.assertIsNone(controllers.right)
        controller = Controller(pose=Pose())
        self.assertEqual((controller.trigger, controller.grip), (0.0, 0.0))
        self.assertEqual(controller.thumbstick, Vector2.zero())
        self.assertFalse(controller.buttons.primary)
        self.assertFalse(controller.buttons.secondary)

    def test_parse_representative_payload_and_round_trip(self):
        payload = {
            "left": None,
            "right": {"pose": {"position": [1, 2, 3],
                               "rotation": [0, 0, 0, 1]},
                      "trigger": 0.8, "grip": 0.5,
                      "thumbstick": [-1, 1],
                      "buttons": {"primary": False, "secondary": True}},
        }
        controllers = Controllers.model_validate(payload)
        self.assertIsNone(controllers.left)
        right = controllers.right
        self.assertIsInstance(right.pose, Pose)
        self.assertEqual(tuple(right.pose.position), (1, 2, 3))
        self.assertEqual(tuple(right.pose.rotation), (0, 0, 0, 1))
        self.assertEqual((right.trigger, right.grip), (0.8, 0.5))
        self.assertEqual(right.thumbstick, Vector2(-1.0, 1.0))
        self.assertEqual(controllers.model_dump(), payload)
        self.assertTrue(right.buttons.secondary)
        self.assertEqual(Controllers.model_validate_json(controllers.model_dump_json()),
                         controllers)

    def test_present_controller_requires_pose(self):
        for sample in ({}, {"pose": None}):
            with self.subTest(sample=sample), self.assertRaises(ValidationError):
                Controllers.model_validate({"left": sample})

    def test_analog_boundaries(self):
        for value in (0.0, 1.0):
            with self.subTest(value=value):
                controller = Controller(pose=Pose(), trigger=value, grip=value)
                self.assertEqual((controller.trigger, controller.grip), (value, value))

    def test_invalid_analog_values_rejected(self):
        for name in ("trigger", "grip"):
            for value in (-0.01, 1.01, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValidationError):
                        Controller(pose=Pose(), **{name: value})

    def test_invalid_thumbsticks_rejected(self):
        for value in ((-1.01, 0), (0, 1.01), (float("nan"), 0),
                      (0, float("inf")), (0,), (0, 0, 0)):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    Controller(pose=Pose(), thumbstick=value)

    def test_unknown_fields_rejected_at_each_level(self):
        for payload in ({"center": {}}, {"left": {"pose": {}, "unknown": True}},
                        {"right": {"pose": {}, "buttons": {"unknown": True}}}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    Controllers.model_validate(payload)

    def test_models_are_frozen(self):
        for model, field, value in (
            (Controllers(), "left", Controller(pose=Pose())),
            (Controller(pose=Pose()), "trigger", 1.0),
            (ControllerButtons(), "primary", True),
        ):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(ValidationError):
                    setattr(model, field, value)


if __name__ == "__main__":
    unittest.main()
