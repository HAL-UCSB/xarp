import asyncio
from asyncio import AbstractEventLoop, CancelledError
from typing import Callable, Any, List, Union, Awaitable, AsyncGenerator

import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from xarp.auth import settings_file_authorization
from xarp.commands.sense import SenseCommand, ImageCommand, DepthCommand, EyeCommand, HandsCommand
from xarp.commands.ui import WriteCommand, SayCommand, ReadCommand
from xarp.data_models.chat import ChatMessage
from xarp.commands import XRCommand, XRResponse, ResponseMode, CancelCommand
from xarp.data_models.entities import Session
from xarp.data_models.data import Hands, Image, SenseResult, DeviceInfo
from xarp.data_models.spatial import Transform, Vector3
from xarp.settings import settings
from xarp.storage import SessionRepository
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem


class RemoteXRClient:

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.notification_callbacks: list[Callable[..., Awaitable[None]]] = []
        self._next_xid = 0
        self._send_lock = asyncio.Lock()
        self._recv_task: asyncio.Task | None = None
        self._pending_singles: dict[int, asyncio.Future[XRResponse]] = dict()
        self._pending_streams: dict[int, asyncio.Queue[XRResponse]] = dict()

    async def start(self) -> None:
        if self._recv_task is not None:
            raise RuntimeError('This RemoteXRClient is already running.')
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def _recv_loop(self) -> None:
        try:
            while True:
                raw = await self.ws.receive_text()
                result = XRResponse.model_validate_json(raw)
                xid = getattr(result, 'xid', None)

                # Notifications
                # TODO: slow callbacks can stall _recv_loop
                if xid is None:
                    for callback in list(self.notification_callbacks):
                        await callback(result)
                    continue

                # Single result
                fut = self._pending_singles.pop(xid, None)
                if fut is not None:
                    if not fut.done():
                        fut.set_result(result)
                    continue

                # Stream
                queue = self._pending_streams.get(xid)
                if queue is not None:
                    if result.value is None or isinstance(result.value, Exception):
                        self._pending_streams.pop(xid)
                    queue.put_nowait(result)

        except asyncio.CancelledError:
            raise
        except WebSocketDisconnect as e:
            self._raise_waiters(e)
        except Exception as e:
            self._raise_waiters(e)

    def stop(self, exception: BaseException | None = None) -> None:
        if self._recv_task is None:
            return
        self._recv_task.cancel()
        self._recv_task = None
        self._raise_waiters(exception)

    def _raise_waiters(self, exception: BaseException | None):
        if exception is None:
            exception = CancelledError()

        for fut in self._pending_singles.values():
            if not fut.done():
                fut.set_exception(exception)
        self._pending_singles.clear()

        for xid, queue in self._pending_streams.items():
            eos = XRResponse(xid=xid, value=None)
            queue.put_nowait(eos)
        self._pending_streams.clear()

    async def __aenter__(self) -> 'RemoteXRClient':
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.stop(exception=exc)

    async def execute(self, command: XRCommand) -> AsyncGenerator | Any | None:
        # no response mode
        if not command.expects_response:
            async with self._send_lock:
                command_json = command.model_dump_json()
                return await self.ws.send_text(command_json)

        # assign the next xid
        self._next_xid = (self._next_xid + 1) & 0xFFFFFFFFFFFFFFFF
        command.xid = self._next_xid
        command_json = command.model_dump_json()

        # single response mode
        if command.response_mode == ResponseMode.SINGLE:
            fut = asyncio.get_running_loop().create_future()
            self._pending_singles[command.xid] = fut
            async with self._send_lock:
                await self.ws.send_text(command_json)
            result = await fut
            return command.validate_response(result.value)

        # stream response mode
        queue: asyncio.Queue[XRResponse] = asyncio.Queue()
        self._pending_streams[command.xid] = queue

        async with self._send_lock:
            await self.ws.send_text(command_json)

        async def stream():
            eos = False
            try:
                while True:
                    item = await queue.get()
                    if item.value is None:
                        eos = True
                        break
                    yield command.validate_response(item.value)
            finally:
                self._pending_streams.pop(command.xid, None)
                if not eos:
                    cancel = CancelCommand(target_xid=command.xid)
                    await self.execute(cancel)

        return stream()


class AsyncXR(RemoteXRClient):

    def __init__(self,
                 ws: WebSocket,
                 session: Session):
        super().__init__(ws)
        self.session = session
        self.logs = []
        self.log_chat = False

    def _log_command(self, command: XRCommand):
        command_message = ChatMessage.from_system(
            command.model_dump_json(),
            # mimetype=model_cls_to_mimetype[type(command)]
        )
        self.session.chat.append(command_message)

    def _log_result(self, result: Union[List[BaseModel], BaseModel]) -> None:
        if result is None:
            return
        result_message = ChatMessage.from_user(
            result.model_dump_json(),
            # mimetype=model_cls_to_mimetype[type(result)]
        )
        self.session.chat.append(result_message)

    async def write(self, *text, title=None, key=None) -> None:
        await self.execute(WriteCommand, *text, title=title, key=key)

    async def say(self, *text, title=None, key=None) -> None:
        await self.execute(SayCommand, *text, title=title, key=key)

    async def sense(self, eye=None, head=None, hands=None, image=None, depth=None) -> SenseResult:
        return await self.execute(SenseCommand, eye=eye, head=head, hands=hands, image=image, depth=depth)

    async def read(self, *text, title=None, key=None) -> str:
        if text or title:
            await self.execute(WriteCommand, *text, title=title, key=key)
        return await self.execute(ReadCommand)

    async def image(self) -> Image:
        return await self.execute(ImageCommand)

    async def depth(self) -> Image:
        return await self.execute(DepthCommand)

    async def eye(self) -> Transform:
        return await self.execute(EyeCommand)

    async def hands(self) -> Hands:
        return await self.execute(HandsCommand)

    def as_mcp_server(self):
        mcp = FastMCP('XARP')
        mcp.add_tool(self.clear)
        mcp.add_tool(self.write)
        mcp.add_tool(self.read)
        mcp.add_tool(self.image)
        mcp.add_tool(self.depth)
        mcp.add_tool(self.eye)
        mcp.add_tool(self.hands)
        mcp.add_tool(self.display)
        mcp.add_tool(self.sphere)
        mcp.add_tool(self.save)
        mcp.add_tool(self.load)
        return mcp


class XR:
    def __init__(self,
                 async_xr: AsyncXR,
                 loop: AbstractEventLoop):
        self.as_async = async_xr
        self._loop = loop

    def _sync(self, async_fn, *args, **kwargs):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self._loop:
            raise RuntimeError("XR._sync called from the same event loop; this will deadlock.")

        coro = async_fn(*args, **kwargs)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()  # TODO: timeout=settings.client_result_timeout

    def write(self, *text, title=None, key=None) -> None:
        self._sync(self.as_async.write, *text, title=title, key=key)

    def say(self, *text, title=None, key=None) -> None:
        self._sync(self.as_async.say, *text, title=title, key=key)

    def sense(self, eye=None, head=None, hands=None, image=None, depth=None):
        return self._sync(self.as_async.sense, eye=eye, head=head, hands=hands, image=image, depth=depth)

    def clear(self):
        self._sync(self.as_async.clear)

    def read(self, text=None, title=None, key=None) -> str:
        return self._sync(self.as_async.read, text, title=title, key=key)

    def image(self) -> Image:
        return self._sync(self.as_async.image)

    def depth(self) -> Image:
        return self._sync(self.as_async.depth)

    def eye(self) -> Transform:
        return self._sync(self.as_async.eye)

    def display(self,
                image: Image = None,
                depth: float = .48725,
                opacity: float = 1,
                eye=None,
                visible=True,
                key=None) -> None:
        self._sync(self.as_async.display, image=image, depth=depth, opacity=opacity, eye=eye, visible=visible, key=key)

    def hands(self) -> Hands:
        return self._sync(self.as_async.hands)

    def sphere(self, position: Vector3, scale=.1, color = (1, 1, 1, 1), key=None) -> None:
        return self._sync(self.as_async.sphere, position, scale=scale, color=color, key=key)

    def save(self, *keys) -> None:
        return self._sync(self.as_async.save, *keys)

    def load(self, *keys) -> None:
        return self._sync(self.as_async.load, *keys)

    def glb(self, data, position) -> None:
        return self._sync(self.as_async.glb, data, position)

    def info(self) -> DeviceInfo:
        return self._sync(self.as_async.info)

    def as_tool(self):
        return [
            self.clear,
            self.write,
            self.read,
            self.image,
            self.depth,
            self.eye,
            self.hands,
            self.display,
            self.sphere,
            self.save,
            self.load,
        ]


async def _websocket_entrypoint(
        xr_app: Callable[[XR], None],
        authorize: Callable[[str], bool],
        session_repository: SessionRepository,
        ws: WebSocket,
        user_id: str,
        session_ts: int = None) -> None:
    """

    :param xr_app:
    :param session_repository:
    :param ws:
    :param user_id:
    :param session_ts:
    :return:
    """
    if not authorize(user_id):
        raise HTTPException(status_code=401, detail="User unauthorized")

    session = None
    if session_ts:
        session = session_repository.get(user_id, session_ts)
    if session is None:
        session = Session(user_id=user_id)

    await ws.accept()
    await ws.send_text(str(session.ts))

    loop = asyncio.get_running_loop()
    xr = XR(ws, loop, session)

    def handle_xr_thread_exceptions():
        try:
            xr_app(xr)
        finally:
            session_repository.save(xr.session)

    await asyncio.to_thread(handle_xr_thread_exceptions)

    await ws.close()


def run_xr_app(
        xr_app: Union[
            Callable[[XR], None],
            Callable[[AsyncXR], Awaitable]],
        auth: Callable[[str], bool] = None,
        session_repo: SessionRepository = None) -> None:
    """

    :param xr_app:
    :param auth:
    :param session_repo:
    :return:
    """
    if auth is None:
        auth = settings_file_authorization

    if session_repo is None:
        session_repo = SessionRepositoryLocalFileSystem(**settings.model_dump())

    app = FastAPI()

    async def entrypoint(
            ws: WebSocket,
            user_id: str,
            session_ts: int = None):
        if not auth(user_id):
            raise HTTPException(status_code=401, detail='User unauthorized')

        session = None
        if session_ts:
            session = session_repo.get(user_id, session_ts)
        if session is None:
            session = Session(user_id=user_id)

        await ws.accept()
        await ws.send_text(str(session.ts))

        async with AsyncXR(ws, session) as xr:
            if asyncio.iscoroutinefunction(xr_app):
                await xr_app(xr)
            else:
                loop = asyncio.get_running_loop()
                sync_xr = XR(xr, loop)
                await asyncio.to_thread(xr_app, sync_xr)

        await ws.close()

    app.add_api_websocket_route(
        settings.ws_route,
        entrypoint)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port)


# def run_agent(xr_app: Callable[[XR, ], None], session_repository: SessionRepository):


def run_xr(xr_app: Callable[[RemoteXRClient], Awaitable]) -> None:
    app = FastAPI()

    async def entrypoint(
            ws: WebSocket,
            user_id: str,
            session_ts: int = None):
        await ws.accept()
        async with RemoteXRClient(ws) as xr:
            await xr_app(xr)
        await ws.close()

    app.add_api_websocket_route(
        settings.ws_route,
        entrypoint)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port)
