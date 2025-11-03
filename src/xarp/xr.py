import asyncio
import base64
import pathlib
from asyncio import AbstractEventLoop
from functools import partial
from typing import Callable, Any

import uvicorn
from PIL import Image
from fastapi import FastAPI, WebSocket, HTTPException
from mcp.server.fastmcp import FastMCP

from xarp import settings
from xarp.data_models import Session, XRCommand, ChatMessage, Hands
from xarp.spatial import Transform, FloatArrayLike
from xarp.storage import SessionRepository


class AsyncXR:

    def __init__(self, ws: WebSocket, session: Session, session_repository: SessionRepository):
        self.ws = ws
        self.session = session
        self.session_repository = session_repository
        self._chat_log = False

    @property
    def chat_log(self):
        return self._chat_log

    @chat_log.setter
    def chat_log(self, value):
        self._chat_log = value

    def _get_files_path(self) -> pathlib.Path:
        files_path = pathlib.Path(
            settings.local_storage / self.session.user_id / str(self.session.ts) / 'files')
        files_path.absolute().mkdir(parents=True, exist_ok=True)
        return files_path

    async def _send_command(self, cmd, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        cmd = XRCommand(
            cmd=cmd,
            args=args,
            kwargs=kwargs,
        ).model_dump_json()
        if self._chat_log:
            sys_message = ChatMessage.from_system(cmd)
            self.session.chat.append(sys_message)
        await self.ws.send_json(cmd)

    async def clear(self):
        await self._send_command('clear')
        await self.ws.receive()

    async def _write(self, *text, title=None, key=None) -> None:
        title = title or ''
        await self._send_command(
            'write',
            *text,
            title=title,
            key=key)
        await self.ws.receive()
        if self._chat_log:
            assistant_message = ChatMessage.from_assistant(*text)
            self.session.chat.append(assistant_message)
            self.session_repository.save(self.session)

    async def write(self, *text, title=None, key=None) -> None:
        await self._write(*text, title=title, key=key)

    async def _read(self, *text, title=None) -> str:
        await self._write(*text, title=title)
        await self._send_command('read')
        user_text = await self.ws.receive_text()
        if self._chat_log:
            user_message = ChatMessage.from_user(user_text)
            self.session.chat.append(user_message)
            self.session_repository.save(self.session)
        await self.ws.receive()
        return user_text

    async def read(self, text=None, title=None) -> str:
        return await self._read(text, title)

    async def _image(self, cmd: str, pil_img_mode: str) -> Image:
        await self._send_command(cmd)
        image_dict = await self.ws.receive_json()
        pixels = base64.b64decode(image_dict['pixels'])
        size = image_dict['width'], image_dict['height']
        image = Image.frombytes(pil_img_mode, size, pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if self._chat_log:
            image_path = self._get_files_path() / f'{id(image)}.png'
            image.save(str(image_path))
            chat_message = ChatMessage.from_user(cmd, files=[image_path.absolute().as_uri()])
            self.session.chat.append(chat_message)
            self.session_repository.save(self.session)
        await self.ws.receive()
        return image

    async def image(self) -> Image:
        return await self._image('image', 'RGBA')

    async def depth(self) -> Image:
        return await self._image('depth', 'I;16')

    async def eye(self) -> Transform:
        await self._send_command('eye')
        model_dict = await self.ws.receive_json()
        model = Transform.model_validate(model_dict)
        if self._chat_log:
            model_json = model.model_dump_json()
            chat_message = ChatMessage.from_user('eye', model_json)
            self.session.chat.append(chat_message)
            self.session_repository.save(self.session)
        await self.ws.receive()
        return model

    async def display(self, content: bytes, width: int, height: int, depth: float, key=None) -> None:
        await self._send_command('display', content, width, height, depth, key=key)
        await self.ws.receive()

    async def display_eye(self, content: bytes, width: int, height: int, opacity: float = 1,
                          eye: Transform = None, key=None) -> None:
        await self._send_command('display_eye', content, width, height, opacity=opacity, eye=eye, key=key)
        await self.ws.receive()

    async def hands(self) -> Hands:
        await self._send_command('hands')
        model_dict = await self.ws.receive_json()
        model = Hands.model_validate(model_dict)
        if self._chat_log:
            model_json = model.model_dump_json()
            chat_message = ChatMessage.from_user('hands', model_json)
            self.session.chat.append(chat_message)
            self.session_repository.save(self.session)
        await self.ws.receive()
        return model

    async def sphere(self,
                     position: FloatArrayLike,
                     scale: float = .1,
                     color: FloatArrayLike = (1, 1, 1, 1),
                     key=None) -> None:
        await self._send_command('sphere', *position, scale=scale, color=color, key=key)
        await self.ws.receive()
        if self._chat_log:
            self.session_repository.save(self.session)

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
        mcp.add_tool(self.display_eye)
        return mcp


class XR(AsyncXR):

    def __init__(self, ws: WebSocket, loop: AbstractEventLoop, session: Session, session_repository: SessionRepository):
        super().__init__(ws, session, session_repository)
        self._loop = loop

    def _sync(self, async_fn, *args, **kwargs):
        coro = async_fn(*args, **kwargs)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

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

    def display(self, content: Any, width: int, height: int, depth: float, key=None) -> None:
        return self._sync(super().display, content, width, height, depth, key=key)

    def display_eye(self, content: Any, width: int, height: int, opacity: float = 1,
                    eye: Transform = None, key=None) -> None:
        return self._sync(super().display_eye, content, width, height, opacity=opacity, eye=eye, key=key)

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
            self.display,
            self.display_eye]


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
    xr = XR(ws, loop, session, session_repository)

    def handle_xr_thread_exceptions():
        try:
            xr_app(xr)
        finally:
            if xr.chat_log:
                xr.session_repository.save(xr.session)

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
