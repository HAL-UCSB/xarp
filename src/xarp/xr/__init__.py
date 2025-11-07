import asyncio
import pathlib
from asyncio import AbstractEventLoop
from functools import partial
from typing import Callable, Any, List, Union

import uvicorn
from PIL import Image
from fastapi import FastAPI, WebSocket, HTTPException
from mcp.server.fastmcp import FastMCP

from xarp import settings
from xarp.data_models import Session, Hands, ChatMessage
from xarp.spatial import Transform, FloatArrayLike
from xarp.storage import SessionRepository
from xarp.xr.commands import XRCommand, bundle_map, XRCommandBundle, ClearCommand, WriteCommand, ReadCommand, \
    ImageCommand, DepthCommand, EyeCommand, DisplayCommand, HandsCommand, SphereCommand


class AsyncXR:

    def __init__(self,
                 ws: WebSocket,
                 session: Session):
        self.ws = ws
        self.session = session
        self.logs = []

    async def _execute(self, command_type, *args, **kwargs) -> Any:
        # skip null keywords
        kwargs = {k: v for k, v in kwargs.items() if v}

        command = command_type(*args, **kwargs)
        self.logs.append(command)
        model_json = command.model_dump_json()
        await self.ws.send_json(model_json)

        result_json = await self.ws.receive_json()
        self.logs.append(result_json)
        return command.result(result_json)

    async def bundle(self, *cmds: Union[XRCommand, Callable, str]) -> List:
        if not cmds:
            return []
        args = []
        for cmd in cmds:
            if type(cmd) is XRCommand:
                args.append(cmd)
            else:
                # syntactic sugar
                if callable(cmd):
                    cmd = cmd.__name__
                cmd_type = bundle_map[cmd]
                arg = cmd_type()
                args.append(arg)
        return await self._execute(XRCommandBundle, args=args)

    async def clear(self):
        return await self._execute(ClearCommand)

    async def write(self, *text, title=None, key=None) -> None:
        await self._execute(WriteCommand, *text, title=title, key=title)

    async def read(self, *text, title=None) -> str:
        if text or title:
            await self.write(*text, title=title)
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

    async def display(self, content: bytes, width: int, height: int, depth: float, opacity: float = 1.0, eye=None,
                      key=None) -> None:
        await self._execute(DisplayCommand, content, width, height, depth, opacity=opacity, eye=eye, key=key)

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

    def bundle(self, *cmds: Union[Callable, str]) -> List:
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

    def display(self, content: bytes, width: int, height: int, depth: float, opacity: float = 1.0, eye=None,
                key=None) -> None:
        self._sync(super().display, content, width, height, depth, opacity=opacity, eye=eye, key=key)

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


class XRLogger(AsyncXR):

    def __init__(self, ws: WebSocket, loop: AbstractEventLoop, session: Session):
        super().__init__(ws, session)
        self._loop = loop

    def _get_files_path(self) -> pathlib.Path:
        files_path = pathlib.Path(settings.local_storage / self.session.user_id / str(self.session.ts) / 'files')
        files_path.absolute().mkdir(parents=True, exist_ok=True)
        return files_path

    def _sync(self, async_fn, *args, **kwargs):
        coro = async_fn(*args, **kwargs)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def bundle(self, *cmds: Union[Callable, str]) -> List:
        return self._sync(super().bundle, *cmds)

    def clear(self):
        self._sync(super().clear)

    def write(self, *text, title=None) -> None:
        self._sync(super().write, *text, title)
        message = ChatMessage.from_user(*text)
        self.session.chat.append(message)

    def read(self, text=None, title=None) -> str:
        result = self._sync(super().read, text, title)
        message = ChatMessage.from_user(result)
        self.session.chat.append(message)
        return result

    def image(self) -> Image:
        result = self._sync(super().image)
        image_path = self._get_files_path() / f'{id(result)}.png'
        result.save(str(image_path))
        message = ChatMessage.from_user(ImageCommand._cmd, files=[image_path.absolute().as_uri()])
        self.session.chat.append(message)
        return result

    def depth(self) -> Image:
        result = self._sync(super().depth)
        image_path = self._get_files_path() / f'{id(result)}.png'
        result.save(str(image_path))
        message = ChatMessage.from_user(DepthCommand._cmd, files=[image_path.absolute().as_uri()])
        self.session.chat.append(message)
        return result

    def eye(self) -> Transform:
        result = self._sync(super().eye)
        model_json = result.model_dump_json()
        message = ChatMessage.from_user(EyeCommand._cmd, model_json)
        self.session.chat.append(message)
        return result

    def display(self, content: Any, width: int, height: int, opacity: float = 1,
                eye: Transform = None, key=None) -> None:
        return self._sync(super().display, content, width, height, opacity=opacity, eye=eye, key=key)

    def hands(self) -> Hands:
        result = self._sync(super().hands)
        model_json = result.model_dump_json()
        chat_message = ChatMessage.from_user(HandsCommand._cmd, model_json)
        self.session.chat.append(chat_message)
        return result

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


async def websocket_entrypoint(
        xr_app: Callable[[XR], None],
        session_repository: SessionRepository,
        ws: WebSocket,
        user_id: str,
        session_ts: int = None) -> None:
    if settings.authorized and user_id not in settings.authorized:
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


def run_app(xr_app: Callable[[XR], None], session_repository_type: type[SessionRepository]):
    app = FastAPI()
    session_repository = session_repository_type(**settings.model_dump())
    _websocket_entrypoint = partial(websocket_entrypoint, xr_app, session_repository)

    app.add_api_websocket_route(
        settings.ws_route,
        _websocket_entrypoint)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port)

# def run_agent(xr_app: Callable[[XR, ], None], session_repository: SessionRepository):
