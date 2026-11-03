"""Controller command contract tests without a device or network connection."""

import unittest

from pydantic import ValidationError

from xarp.commands import Bundle, ResponseMode
from xarp.commands.sensing import ControllersCommand, HeadCommand
from xarp.data_models import Controllers
from xarp.spatial import Pose


class TestControllersCommand(unittest.TestCase):
    def test_serialized_command_name(self):
        self.assertEqual(ControllersCommand().model_dump(), {"cmd": "controllers"})

    def test_command_name_cannot_be_changed(self):
        with self.assertRaises(ValidationError):
            ControllersCommand(cmd="hands")
        command = ControllersCommand()
        with self.assertRaises(ValidationError):
            command.cmd = "hands"

    def test_parse_controller_payload(self):
        result = ControllersCommand().validate_response_value({
            "left": {"connected": True, "tracked": False,
                     "grip_pose": None, "buttons": {"primary": True}},
            "right": {"connected": True, "tracked": True,
                      "grip_pose": {"position": [1, 2, 3],
                                    "rotation": [0, 0, 0, 1]},
                      "trigger": 0.8, "grip": 0.2,
                      "thumbstick": [0.5, -0.5]},
        })
        self.assertIsInstance(result, Controllers)
        self.assertTrue(result.left.buttons.primary)
        self.assertIsNone(result.left.grip_pose)
        self.assertEqual(tuple(result.right.grip_pose.position), (1, 2, 3))
        self.assertEqual(result.right.trigger, 0.8)
        self.assertEqual(result.right.grip, 0.2)
        self.assertEqual(result.right.thumbstick, (0.5, -0.5))

    def test_unavailable_sides(self):
        result = ControllersCommand().validate_response_value({"left": {}, "right": {}})
        self.assertFalse(result.left.connected)
        self.assertFalse(result.right.connected)
        self.assertIsNone(result.right.grip_pose)

    def test_malformed_payload_is_rejected(self):
        for payload in (None, [], {"right": {"trigger": 2}},
                        {"right": {"grip_pose": 42}},
                        {"left": {"buttons": {"unknown": True}}}):
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    ControllersCommand().validate_response_value(payload)

    def test_bundle_serializes_and_validates_with_other_sensing_commands(self):
        for mode in (ResponseMode.SINGLE, ResponseMode.STREAM):
            with self.subTest(mode=mode):
                bundle = Bundle(cmds=[HeadCommand(), ControllersCommand()], mode=mode)
                self.assertEqual(bundle.model_dump()["cmds"],
                                 [{"cmd": "head"}, {"cmd": "controllers"}])
                self.assertEqual(bundle.model_dump()["mode"], mode)
                head, controllers = bundle.validate_response_value([
                    {"position": [0, 1, 0], "rotation": [0, 0, 0, 1]},
                    {"right": {"connected": True, "trigger": 0.7}},
                ])
                self.assertIsInstance(head, Pose)
                self.assertIsInstance(controllers, Controllers)
                self.assertEqual(controllers.right.trigger, 0.7)


if __name__ == "__main__":
    unittest.main()
