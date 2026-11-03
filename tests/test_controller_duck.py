"""Replay controller duck manipulation without a headset or network."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

from xarp.data_models import Controller, Controllers
from xarp.entities import DefaultAssets, Element
from xarp.spatial import Pose, Transform, Vector3


spec = importlib.util.spec_from_file_location(
    "controller_duck", Path(__file__).resolve().parents[1] / "demos/controller_3d_manipulation.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def sample(trigger=0, *, left=(1, 0, 0), right=(-1, 0, 0), grip=0, **invalid):
    return {"controllers": Controllers(
        left=Controller(connected=True, tracked=True, trigger=trigger, grip=grip,
                        grip_pose=Pose(position=Vector3(*left))),
        right=Controller(**dict(dict(connected=True, tracked=True, trigger=trigger, grip=grip,
                                    grip_pose=Pose(position=Vector3(*right))), **invalid)),
    )}


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
        assert options == {"controllers": True}
        return self.stream

    def update(self, duck):
        if self.fail:
            raise RuntimeError("update failed")
        self.updates.append(duck.model_copy(update={"transform": duck.transform.model_copy()}))


class TestControllerDuck(unittest.TestCase):
    def replay(self, frames, fail=False):
        xr = XR(frames, fail)
        duck = Element(key="duck", asset=DefaultAssets.cube(),
                       transform=Transform(position=Vector3(0, 1, 0)))
        try:
            demo.manipulate(xr, duck)
        finally:
            self.assertTrue(xr.stream.closed)
        return xr.updates

    def test_translation_rotation_scale_and_release(self):
        updates = self.replay([
            sample(), sample(1),
            sample(0.5, left=(3, 2, 0), right=(3, -2, 0)), sample(),
        ])
        self.assertEqual(len(updates), 2)
        np.testing.assert_allclose(tuple(updates[-1].transform.position), (1, 0, 0), atol=1e-6)
        np.testing.assert_allclose(tuple(updates[-1].transform.scale), (2, 2, 2))
        rotated = updates[-1].transform.rotation.rotate_vector(Vector3.right())
        np.testing.assert_allclose(tuple(rotated), (0, 1, 0), atol=1e-6)

    def test_grips_do_not_grab(self):
        self.assertEqual(self.replay([sample(), sample(grip=1)]), [])

    def test_both_triggers_required_and_either_release_stops(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                released = sample(1)
                controllers = released["controllers"]
                released["controllers"] = controllers.model_copy(update={
                    side: getattr(controllers, side).model_copy(update={"trigger": 0}),
                })
                updates = self.replay([
                    sample(), released, sample(1), released, released,
                ])
                self.assertEqual(len(updates), 1)

    def test_startup_and_tracking_recovery_require_release(self):
        for invalid in ({"tracked": False}, {"connected": False}, {"grip_pose": None}):
            with self.subTest(invalid=invalid):
                updates = self.replay([
                    sample(1), sample(), sample(1), sample(1, **invalid),
                    sample(1), sample(), sample(1),
                ])
                self.assertEqual(len(updates), 2)

    def test_coincident_controllers_do_not_divide_by_zero(self):
        updates = self.replay([sample(), sample(1, left=(0, 0, 0), right=(0, 0, 0)), sample(1)])
        self.assertEqual(len(updates), 1)

    def test_update_error_closes_stream(self):
        with self.assertRaisesRegex(RuntimeError, "update failed"):
            self.replay([sample(), sample(1)], fail=True)


if __name__ == "__main__":
    unittest.main()
