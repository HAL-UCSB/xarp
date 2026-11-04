"""Replay the controller camera demo without a headset or server."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image

from xarp.data_models import Controller, Controllers
from xarp.entities import ImageAsset
from xarp.spatial import Pose, Vector3


DEMOS = Path(__file__).resolve().parents[1] / "demos"


def load_demo(name):
    spec = importlib.util.spec_from_file_location(name, DEMOS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with patch.dict(sys.modules, {"camera": load_demo("camera")}):
    demo = load_demo("controller_camera")


def sample(trigger=0, *, left=(-0.25, -0.25, 0.5), right=(0.25, 0.25, 0.5), **invalid):
    return {
        "controllers": Controllers(
            left=Controller(connected=True, tracked=True, trigger=trigger,
                            grip_pose=Pose(position=Vector3(*left))),
            right=Controller(**dict(dict(connected=True, tracked=True, trigger=trigger,
                                        grip_pose=Pose(position=Vector3(*right))), **invalid)),
        ),
        "eye": Pose(),
        "image": ImageAsset.from_obj(Image.new("RGB", (100, 100), "red")),
    }


class Stream:
    def __init__(self, frames):
        self.frames = frames
        self.closed = False

    def __iter__(self):
        yield from self.frames

    def close(self):
        self.closed = True


class XR:
    def __init__(self, frames, fail=False):
        self.stream = Stream(frames)
        self.updates = []
        self.fail = fail

    def sense(self, **options):
        assert options == {"image": True, "eye": True, "controllers": True}
        return self.stream

    def update(self, panel):
        if self.fail:
            raise RuntimeError("update failed")
        self.updates.append(panel.model_copy(update={"transform": panel.transform.model_copy()}))


class TestControllerCamera(unittest.TestCase):
    def replay(self, frames, fail=False):
        xr = XR(frames, fail)
        try:
            demo.app(xr)
        finally:
            self.assertTrue(xr.stream.closed)
        return xr.updates

    def test_live_crop_hysteresis_and_release(self):
        updates = self.replay([sample(1), sample(), sample(0.79), sample(0.8),
                               sample(0.5), sample(0.2)])
        self.assertEqual(len(updates), 2)
        panel = updates[-1]
        self.assertEqual(panel.asset.obj.size, (50, 50))
        self.assertEqual(panel.asset.obj.getpixel((25, 25)), (255, 0, 0))
        self.assertEqual(tuple(panel.transform.position), (0, 0, 0.5))
        self.assertEqual(tuple(panel.transform.rotation), tuple(Pose().rotation))

    def test_either_trigger_release_stops_framing(self):
        for side in ("left", "right"):
            released = sample(1)
            controllers = released["controllers"]
            released["controllers"] = controllers.model_copy(update={
                side: getattr(controllers, side).model_copy(update={"trigger": 0}),
            })
            self.assertEqual(len(self.replay([sample(), released, sample(1), released])), 1)

    def test_tracking_recovery_requires_release(self):
        for invalid in ({"tracked": False}, {"connected": False}, {"grip_pose": None}):
            with self.subTest(invalid=invalid):
                self.assertEqual(len(self.replay([
                    sample(), sample(1), sample(1, **invalid), sample(1), sample(), sample(1),
                ])), 2)

    def test_invalid_corners_are_skipped(self):
        for right in ((0.25, 0.25, -0.5), (0.25, 0.25, 0),
                      (-0.25, 0.25, 0.5), (0.25, -0.25, 0.5)):
            with self.subTest(right=right):
                self.assertEqual(self.replay([sample(), sample(1, right=right)]), [])

    def test_update_failure_closes_stream(self):
        with self.assertRaisesRegex(RuntimeError, "update failed"):
            self.replay([sample(), sample(1)], fail=True)


if __name__ == "__main__":
    unittest.main()
