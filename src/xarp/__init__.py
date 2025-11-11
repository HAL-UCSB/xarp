import asyncio
from asyncio import AbstractEventLoop
from functools import partial
from typing import Callable, Any, List, Union
import base64

import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from mcp.server.fastmcp import FastMCP

from xarp.settings import settings
from xarp.auth import settings_file_authorization
from xarp.data_models import model_cls_to_mimetype
from xarp.data_models.chat import ChatMessage
from xarp.data_models.entities import Session
from xarp.data_models.app import Hands, Image
from xarp.data_models.spatial import Transform, FloatArrayLike
from xarp.storage import SessionRepository
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.data_models.commands import XRCommand, XRCommandBundle, ClearCommand, WriteCommand, ReadCommand, \
    ImageCommand, DepthCommand, EyeCommand, DisplayCommand, HandsCommand, SphereCommand, Bundlable


class AsyncXR:

    def __init__(self,
                 ws: WebSocket,
                 session: Session):
        self.ws = ws
        self.session = session
        self.logs = []
        self.log_chat = False

    async def _execute(self, command_type, *args, **kwargs) -> Any:
        # skip null keywords
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # XR Command from server
        command = command_type(*args, **kwargs)
        command_json = command.model_dump_json()
        if self.log_chat:
            self._log_command(command)
        await self.ws.send_text(command_json)

        # Log and return result from client
        result_json = await self.ws.receive_text()
        result = command.result(result_json)
        if self.log_chat:
            self._log_result(result)
        return result

    def _log_command(self, command: XRCommand):
        # flatten bundle
        if isinstance(command, XRCommandBundle):
            for subcommand in command.args:
                self._log_command(subcommand)
            return

        command_message = ChatMessage.from_system(
            command.model_dump_json(),
            mimetype=model_cls_to_mimetype[type(command)])
        self.session.chat.append(command_message)

    def _log_result(self, result: Any) -> None:
        if result is None:
            return

        # flatten bundle result
        if isinstance(result, list):
            for sub_result in result:
                self._log_result(sub_result)
            return

        result_message = ChatMessage.from_user(
            result.model_dump_json(),
            mimetype=model_cls_to_mimetype[type(result)])
        self.session.chat.append(result_message)

    async def bundle(self, *cmds: Union[XRCommand, Callable, str]) -> Bundlable:
        if not cmds:
            return []
        args = []
        for cmd in cmds:
            if type(cmd) is XRCommand:
                args.append(cmd)
            else:
                # syntactic sugar: allow bundle(xr.image, xr.depth) instead of bundle('image', 'depth')
                if callable(cmd):
                    cmd = cmd.__name__
                cmd_type = XRCommandBundle.bundle_map[cmd]
                arg = cmd_type()
                args.append(arg)
        return await self._execute(XRCommandBundle, args=args)

    async def clear(self):
        return await self._execute(ClearCommand)

    async def write(self, *text, title=None, key=None) -> None:
        await self._execute(WriteCommand, *text, title=title, key=title)

    async def read(self, *text, title=None) -> str:
        if text or title:
            await self._execute(WriteCommand, *text, title=title, key=title)
        return await self._execute(ReadCommand)

    async def image(self) -> Image:
        return await self._execute(ImageCommand)

    async def depth(self) -> Image:
        return await self._execute(DepthCommand)

    async def eye(self) -> Transform:
        result = await self._execute(EyeCommand)
        model_json = result.model_dump_json()
        chat_message = ChatMessage.from_user(EyeCommand._cmd, model_json)
        self.session.chat.append(chat_message)
        return result

    async def display(self,
                      image: Image = None,
                      depth: float = None,
                      opacity: float = 1.0,
                      eye=None,
                      visible=True,
                      key=None) -> None:
        await self._execute(DisplayCommand, image=image, depth=depth, opacity=opacity, eye=eye, visible=visible,
                            key=key)

    async def hands(self) -> Hands:
        return await self._execute(HandsCommand)

    async def sphere(self,
                     position: FloatArrayLike,
                     scale: float = .1,
                     color: FloatArrayLike = (1, 1, 1, 1),
                     key=None) -> None:
        await self._execute(SphereCommand, position, scale=scale, color=color, key=key)

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
        return mcp


class XR(AsyncXR):
    def __init__(self,
                 ws: WebSocket,
                 loop: AbstractEventLoop,
                 session: Session):
        super().__init__(ws, session)
        self._loop = loop

    def _sync(self, async_fn, *args, **kwargs):
        coro = async_fn(*args, **kwargs)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def bundle(self, *cmds: Union[Callable, str]) -> Bundlable:
        return self._sync(super().bundle, *cmds)

    def clear(self):
        self._sync(super().clear)

    def write(self, *text, title=None) -> None:
        self._sync(super().write, *text, title)

    def read(self, text=None, title=None) -> str:
        return self._sync(super().read, text, title)

    def image(self) -> Image:
        return self._sync(super().image)

    def depth(self) -> Image:
        return self._sync(super().depth)

    def eye(self) -> Transform:
        return self._sync(super().eye)

    def display(self,
                image: Image = None,
                depth: float = None,
                opacity: float = 1.0,
                eye=None,
                visible=True,
                key=None) -> None:
        self._sync(super().display, image=image, depth=depth, opacity=opacity, eye=eye, visible=visible, key=key)

    def hands(self) -> Hands:
        return self._sync(super().hands)

    def sphere(self, position: FloatArrayLike, scale=.1, color: FloatArrayLike = (1, 1, 1, 1), key=None) -> None:
        return self._sync(super().sphere, position, scale=scale, color=color, key=key)

    def as_tool(self):
        return [
            self.clear,
            self.write,
            self.read,
            self.image,
            self.depth,
            self.eye,
            self.hands,
            self.display]


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
        xr_app: Callable[[XR], None],
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

    entrypoint = partial(
        _websocket_entrypoint,
        xr_app,
        auth,
        session_repo)

    app.add_api_websocket_route(
        settings.ws_route,
        entrypoint)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port)

# def run_agent(xr_app: Callable[[XR, ], None], session_repository: SessionRepository):
