"""Exercise the controller APIs with a validating fake remote, without Quest."""

import asyncio
import threading
from types import SimpleNamespace
import unittest

from xarp.commands import ResponseMode
from xarp.data_models import Controllers, Hands
from xarp.express import AsyncXR, SyncXR


class FakeRemote:
    def __init__(self):
        self.bundles = []
        self.closed = False
        self.fail_stream = False

    async def execute(self, bundle):
        self.bundles.append(bundle)
        payloads = [{"right": {"connected": True, "trigger": 0.8}}
                    if cmd.cmd == "controllers" else {}
                    for cmd in bundle.cmds]
        if bundle.mode == ResponseMode.SINGLE:
            return SimpleNamespace(value=bundle.validate_response_value(payloads))

        async def stream():
            try:
                yield SimpleNamespace(value=bundle.validate_response_value(payloads))
                if self.fail_stream:
                    raise RuntimeError("simulated stream failure")
                yield SimpleNamespace(value=bundle.validate_response_value(payloads))
            finally:
                self.closed = True

        return stream()


class TestAsyncControllerAPI(unittest.IsolatedAsyncioTestCase):
    async def test_single_sample(self):
        remote = FakeRemote()
        result = await AsyncXR(remote).controllers()
        self.assertIsInstance(result, Controllers)
        self.assertEqual(result.right.trigger, 0.8)
        self.assertEqual(remote.bundles[0].mode, ResponseMode.SINGLE)
        self.assertEqual([c.cmd for c in remote.bundles[0].cmds], ["controllers"])

    async def test_mixed_stream_and_explicit_cleanup(self):
        remote = FakeRemote()
        stream = AsyncXR(remote).sense(hands=True, controllers=True, rt=False)
        frame = await anext(stream)
        self.assertEqual(list(frame), ["hands", "controllers"])
        self.assertIsInstance(frame["hands"], Hands)
        self.assertEqual(frame["controllers"].right.trigger, 0.8)
        self.assertEqual([c.cmd for c in remote.bundles[0].cmds],
                         ["hands", "controllers"])
        self.assertEqual(remote.bundles[0].mode, ResponseMode.STREAM)
        self.assertFalse(remote.bundles[0].rt)
        await stream.aclose()
        self.assertTrue(remote.closed)

    async def test_controller_only_stream_completes(self):
        remote = FakeRemote()
        frames = [frame async for frame in AsyncXR(remote).sense(controllers=True)]
        self.assertEqual(len(frames), 2)
        self.assertEqual(list(frames[0]), ["controllers"])
        self.assertIsInstance(frames[0]["controllers"], Controllers)
        self.assertTrue(remote.bundles[0].rt)
        self.assertTrue(remote.closed)

    async def test_stream_error_propagates_and_closes(self):
        remote = FakeRemote()
        remote.fail_stream = True
        stream = AsyncXR(remote).sense(controllers=True)
        await anext(stream)
        with self.assertRaisesRegex(RuntimeError, "simulated stream failure"):
            await anext(stream)
        self.assertTrue(remote.closed)

    async def test_default_sense_does_not_request_controllers(self):
        remote = FakeRemote()
        frames = [frame async for frame in AsyncXR(remote).sense(hands=True)]
        self.assertEqual(list(frames[0]), ["hands"])
        self.assertEqual([c.cmd for c in remote.bundles[0].cmds], ["hands"])

    async def test_empty_sense_sends_no_request(self):
        remote = FakeRemote()
        self.assertEqual([frame async for frame in AsyncXR(remote).sense()], [])
        self.assertEqual(remote.bundles, [])


class TestSyncControllerAPI(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()
        self.remote = FakeRemote()
        self.xr = SyncXR(self.remote, self.loop, self.thread)

    def tearDown(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        self.loop.close()

    def test_single_sample(self):
        result = self.xr.controllers()
        self.assertIsInstance(result, Controllers)
        self.assertEqual(result.right.trigger, 0.8)
        self.assertEqual(self.remote.bundles[0].mode, ResponseMode.SINGLE)

    def test_stream_forwards_options_and_closes(self):
        stream = self.xr.sense(hands=True, controllers=True, rt=False)
        try:
            frame = next(stream)
            self.assertIsInstance(frame["controllers"], Controllers)
            self.assertIsInstance(frame["hands"], Hands)
            self.assertFalse(self.remote.bundles[0].rt)
        finally:
            stream.close()
        self.assertTrue(self.remote.closed)

    def test_stream_exhaustion(self):
        frames = list(self.xr.sense(controllers=True))
        self.assertEqual(len(frames), 2)
        self.assertEqual(list(frames[0]), ["controllers"])
        self.assertTrue(self.remote.closed)


if __name__ == "__main__":
    unittest.main()
