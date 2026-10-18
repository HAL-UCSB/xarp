r"""Replay the proposed synchronous controller demo without Quest or a server.

Run from the repository root:
    .\venv\Scripts\python.exe -m unittest discover -s tests -p test_controller_brush_draft.py -v

These tests exercise the actual demo with real XARP Element/Pose models and
test-only controller samples. They do not implement or verify the proposed
controller protocol, asynchronous API, device capture, or network reliability.
Session changes are injected samples, not real WebSocket reconnections.
"""

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
import unittest

from xarp.spatial import Pose, Vector3


DEMO_PATH = Path(__file__).resolve().parents[1] / "demos" / "controller_brush.py"
spec = importlib.util.spec_from_file_location("controller_brush_draft", DEMO_PATH)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@dataclass
class FakeButtons:
    primary: bool = False
    secondary: bool = False
    primary_press_count: int = 0


@dataclass
class FakeController:
    connected: bool = True
    tracked: bool = True
    grip_pose: Pose | None = field(default_factory=Pose)
    trigger: float = 0.0
    grip: float = 0.0
    thumbstick: tuple[float, float] = (0.0, 0.0)
    buttons: FakeButtons = field(default_factory=FakeButtons)
    tracking_generation: int = 0


@dataclass
class FakeControllers:
    session_id: str
    left: FakeController
    right: FakeController


def sample(trigger=0.0, *, position=(1, 2, 3), connected=True, tracked=True,
           pose_valid=True, generation=0, clear_count=0, session="first"):
    """Build a delivered sample; counters stand in for prior client capture."""
    right = FakeController(
        trigger=trigger, connected=connected, tracked=tracked,
        grip_pose=Pose(position=Vector3(*position)) if pose_valid else None,
        tracking_generation=generation,
    )
    left = FakeController(buttons=FakeButtons(primary_press_count=clear_count))
    return {"controllers": FakeControllers(session, left, right)}


class FakeStream:
    def __init__(self, frames):
        self.frames = frames
        self.index = -1
        self.closed = False

    def __iter__(self):
        for self.index, frame in enumerate(self.frames):
            if isinstance(frame, Exception):
                raise frame
            yield frame

    def close(self):
        self.closed = True


class FakeXR:
    """Record scene commands, copying Elements before the demo mutates them."""
    def __init__(self, frames, *, fail_update=False):
        self.stream = FakeStream(frames)
        self.sense_options = None
        self.actions = []
        self.fail_update = fail_update

    def sense(self, **options):
        self.sense_options = options
        return self.stream

    def update(self, element):
        if self.fail_update:
            raise RuntimeError("simulated scene update failure")
        self.actions.append((self.stream.index, "update", element.model_copy(deep=True)))

    def destroy_element(self, **options):
        self.actions.append((self.stream.index, "clear", options))

    @property
    def paints(self):
        return [(index, element) for index, kind, element in self.actions
                if kind == "update" and element.key.startswith("paint_")]

    @property
    def clears(self):
        return [(index, options) for index, kind, options in self.actions if kind == "clear"]

    @property
    def cursors(self):
        return [(index, element) for index, kind, element in self.actions
                if kind == "update" and element.key == "brush"]


class TestControllerBrushDraft(unittest.TestCase):
    def replay(self, *frames):
        xr = FakeXR(frames)
        demo.app(xr)
        self.assertEqual(xr.sense_options, {"controllers": True})
        self.assertTrue(xr.stream.closed)
        return xr

    def test_move_draw_hold_release(self):
        xr = self.replay(sample(), sample(1, position=(2, 3, 4)),
                         sample(1, position=(3, 4, 5)), sample(0))
        self.assertEqual([i for i, _ in xr.paints], [1, 2])
        self.assertEqual([e.key for _, e in xr.paints], ["paint_0", "paint_1"])
        self.assertEqual([tuple(e.transform.position) for _, e in xr.paints],
                         [(2, 3, 4), (3, 4, 5)])
        self.assertEqual(tuple(xr.cursors[1][1].transform.position), (2, 3, 4))

    def test_trigger_requires_release_at_start_and_uses_hysteresis(self):
        xr = self.replay(sample(1), sample(.2), sample(.79), sample(.8),
                         sample(.5), sample(.2), sample(.5), sample(.8))
        self.assertEqual([i for i, _ in xr.paints], [3, 4, 7])

    def test_clear_wins_and_a_held_count_does_not_clear_twice(self):
        xr = self.replay(sample(), sample(1), sample(1, clear_count=1),
                         sample(1, clear_count=1))
        self.assertEqual(xr.clears, [(2, {"all_elements": True})])
        self.assertEqual([i for i, _ in xr.paints], [1, 3])
        self.assertEqual([e.key for _, e in xr.paints], ["paint_0", "paint_0"])

    def test_counter_jump_clears_once_even_when_current_button_is_released(self):
        xr = self.replay(sample(), sample(clear_count=3), sample(clear_count=3))
        self.assertEqual(xr.clears, [(1, {"all_elements": True})])

    def test_invalid_tracking_hides_cursor_and_requires_release_after_recovery(self):
        xr = self.replay(sample(), sample(1), sample(1, tracked=False, generation=1),
                         sample(1, generation=1), sample(0, generation=1),
                         sample(1, generation=1))
        self.assertEqual([i for i, _ in xr.paints], [1, 5])
        self.assertEqual([(i, e.active) for i, e in xr.cursors],
                         [(0, True), (1, True), (2, False), (3, True), (4, True), (5, True)])

    def test_disconnected_or_missing_pose_cannot_paint(self):
        for invalid in ({"connected": False}, {"pose_valid": False}):
            with self.subTest(invalid=invalid):
                xr = self.replay(sample(), sample(1, **invalid), sample(1),
                                 sample(0), sample(1))
                self.assertEqual([i for i, _ in xr.paints], [4])

    def test_tracking_counter_detects_interruption_between_delivered_frames(self):
        xr = self.replay(sample(), sample(1), sample(1, generation=1),
                         sample(0, generation=1), sample(1, generation=1))
        self.assertEqual([i for i, _ in xr.paints], [1, 4])

    def test_new_session_baselines_old_clear_count_and_requires_trigger_release(self):
        xr = self.replay(sample(), sample(1), sample(1, session="second", clear_count=4),
                         sample(0, session="second", clear_count=4),
                         sample(1, session="second", clear_count=4))
        self.assertEqual([i for i, _ in xr.paints], [1, 4])
        self.assertEqual(xr.clears, [])

    def test_stream_closes_when_scene_update_raises(self):
        xr = FakeXR([sample()], fail_update=True)
        with self.assertRaisesRegex(RuntimeError, "scene update failure"):
            demo.app(xr)
        self.assertTrue(xr.stream.closed)

    def test_stream_closes_when_iteration_raises(self):
        xr = FakeXR([sample(), RuntimeError("simulated disconnect")])
        with self.assertRaisesRegex(RuntimeError, "simulated disconnect"):
            demo.app(xr)
        self.assertTrue(xr.stream.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
