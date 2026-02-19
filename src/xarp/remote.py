import asyncio
from asyncio import QueueShutDown
from contextlib import suppress
from typing import Callable, Awaitable, AsyncGenerator

import msgpack as serializer
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .commands import Bundle, ResponseMode, Cancel, Response, IncomingMessageValidator, IncomingMessage, \
    MessageType, StreamResponse, SingleResponse, Notification
from .settings import settings
from .time import utc_ts


class HeartbeatTimeoutException(Exception):
    def __init__(self, timeout):
        super().__init__(f"Last heartbeat received {timeout} seconds ago.")


class ResponseException(Exception):
    def __init__(self, incoming: Response):
        super().__init__(f"Command {incoming.xid} raised an exception: {incoming.value}")


class LatestQueue:
    """
    O(1) put and get. Drops all but the latest normal frame.
    EOS and error frames are never dropped, they queue separately
    and take priority over any pending value.
    """

    def __init__(self):
        self._event = asyncio.Event()
        self._latest: Response | None = None
        self._terminals: asyncio.Queue[Response] = asyncio.Queue()
        self._shutdown = False

    def put_nowait(self, item: Response) -> None:
        if self._shutdown:
            raise QueueShutDown
        if item.eos or item.error:
            self._terminals.put_nowait(item)
        else:
            self._latest = item  # overwrites previous, dropping it
        self._event.set()

    async def get(self) -> Response:
        while True:
            if self._shutdown:
                raise QueueShutDown
            if not self._terminals.empty():
                return self._terminals.get_nowait()
            if self._latest is not None:
                item, self._latest = self._latest, None
                self._event.clear()
                return item
            self._event.clear()
            await self._event.wait()

    def shutdown(self) -> None:
        self._shutdown = True
        self._event.set()  # unblock any waiter in get()


class RemoteXRClient:

    def __init__(self, ws: WebSocket, heartbeat_timeout: float | None = None):
        self.ws = ws
        self._xid_counter = 0
        self._send_lock = asyncio.Lock()
        self._receive_task: asyncio.Task | None = None
        self._send_heartbeat_task: asyncio.Task | None = None
        self._monitor_heartbeat_task: asyncio.Task | None = None
        self._heartbeat_timeout = heartbeat_timeout if heartbeat_timeout is not None else settings.heartbeat_timeout_secs
        self._pending_singles: dict[int, asyncio.Future[Response]] = dict()
        self._pending_streams: dict[int, asyncio.Queue[Response] | LatestQueue] = dict()
        self.notification_callbacks: list[Callable[..., Awaitable[None]]] = []
        self._last_received_heartbeat = 0

    async def _send(self, model: BaseModel) -> None:
        async with self._send_lock:
            data = model.model_dump()
            data_bytes = serializer.dumps(data)
            await self.ws.send_bytes(data_bytes)

    async def _receive(self) -> IncomingMessage:
        response_bytes = await self.ws.receive_bytes()
        response_data = serializer.loads(response_bytes)
        return IncomingMessageValidator.validate_python(response_data)

    async def start(self) -> None:
        if self._send_heartbeat_task is not None:
            raise RuntimeError('This RemoteXRClient is already running.')
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._send_heartbeat_task = asyncio.create_task(self._send_heartbeat_loop())
        self._monitor_heartbeat_task = asyncio.create_task(self._monitor_heartbeat_loop())

    async def _monitor_heartbeat_loop(self):
        try:
            self._last_received_heartbeat = utc_ts()
            delta = 0
            while delta < self._heartbeat_timeout:
                delta = (utc_ts() - self._last_received_heartbeat) / 1_000
                await asyncio.sleep(self._heartbeat_timeout / 2)
            print(f'[XARP] client disconnected {delta}')
            self.stop(HeartbeatTimeoutException(delta))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._raise_waiters(e)

    async def _send_heartbeat_loop(self):
        try:
            while True:
                heartbeat = Notification()
                await self._send(heartbeat)
                await asyncio.sleep(self._heartbeat_timeout / 2)
        except asyncio.CancelledError:
            raise
        except WebSocketDisconnect as e:
            self._raise_waiters(e)
        except Exception as e:
            self._raise_waiters(e)

    async def _receive_loop(self) -> None:
        try:
            while True:
                incoming = await self._receive()
                self._last_received_heartbeat = utc_ts()

                match incoming.type:
                    case MessageType.NOTIFICATION:
                        incoming: Notification
                        # TODO: slow callbacks can stall _receive_loop
                        if incoming.error:
                            raise ResponseException(incoming)
                        if incoming.value:
                            print(incoming)
                        for callback in list(self.notification_callbacks):
                            await callback(incoming)
                        continue
                    case MessageType.SINGLE_RESPONSE:
                        incoming: SingleResponse
                        fut = self._pending_singles.pop(incoming.xid, None)
                        if fut is not None:
                            if incoming.error:
                                ex = ResponseException(incoming)
                                fut.set_exception(ex)
                            else:
                                fut.set_result(incoming)
                            continue
                    case MessageType.STREAM_RESPONSE:
                        incoming: StreamResponse
                        queue = self._pending_streams.get(incoming.xid, None)
                        if queue is not None:
                            queue.put_nowait(incoming)

        except asyncio.CancelledError:
            raise
        except WebSocketDisconnect as e:
            self._raise_waiters(e)
        except Exception as e:
            self._raise_waiters(e)

    def stop(self, exception: BaseException | None = None) -> None:
        if self._receive_task is not None:
            self._receive_task.cancel()
            self._receive_task = None

        if self._monitor_heartbeat_task is not None:
            self._monitor_heartbeat_task.cancel()
            self._monitor_heartbeat_task = None

        if self._send_heartbeat_task is not None:
            self._send_heartbeat_task.cancel()
            self._send_heartbeat_task = None

        self._raise_waiters(exception)

    def _raise_waiters(self, exception: BaseException) -> None:
        for fut in self._pending_singles.values():
            if not fut.done():
                fut.set_exception(exception)
        self._pending_singles.clear()

        for queue in self._pending_streams.values():
            queue.shutdown()
        self._pending_streams.clear()

    async def __aenter__(self) -> "RemoteXRClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        tasks = [
            self._receive_task,
            self._monitor_heartbeat_task,
            self._send_heartbeat_task
        ]
        if not all(tasks):
            return
        self.stop(exception=exc)

        gather = asyncio.gather(*tasks, return_exceptions=True)
        try:
            await asyncio.wait_for(gather, timeout=self._heartbeat_timeout)
        except asyncio.TimeoutError:
            gather.cancel()
            with suppress(Exception):
                await gather

    def _next_xid(self) -> int:
        self._xid_counter = (self._xid_counter + 1) & 0xFFFFFFFFFFFFFFFF
        return self._xid_counter

    async def execute(self, bundle: Bundle) -> AsyncGenerator[StreamResponse] | SingleResponse | None:

        # ResponseMode.NONE
        if bundle.mode == ResponseMode.NONE:
            bundle.xid = None
            await self._send(bundle)
            return None

        # ResponseMode.SINGLE
        bundle.xid = self._next_xid()
        if bundle.mode == ResponseMode.SINGLE:
            fut = asyncio.get_running_loop().create_future()
            self._pending_singles[bundle.xid] = fut
            await self._send(bundle)
            response = await fut
            response.value = bundle.validate_response_value(response.value)
            return response

        # ResponseMode.STREAM
        cancel_stream = Bundle(
            cmds=[Cancel(target_xid=bundle.xid)],
            mode=ResponseMode.NONE)

        queue: LatestQueue | asyncio.Queue[Response] = (
            LatestQueue() if bundle.rt else asyncio.Queue()
        )
        self._pending_streams[bundle.xid] = queue
        await self._send(bundle)

        async def _async_gen_stream():
            eos = False
            shutdown = False
            try:
                while True:
                    item = await queue.get()
                    if eos := item.eos:
                        break
                    if item.error:
                        raise ResponseException(item)
                    item.value = bundle.validate_response_value(item.value)
                    yield item
            except QueueShutDown:
                shutdown = True
            finally:
                self._pending_streams.pop(bundle.xid, None)
                if not eos and not shutdown:
                    await self.execute(cancel_stream)

        return _async_gen_stream()
