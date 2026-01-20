import asyncio
import inspect
import socket
import threading
from typing import Any, Callable, Awaitable, Union
from urllib.parse import urlencode

import PIL.Image
import qrcode
import uvicorn
from fastapi import FastAPI, WebSocket

from .express import SyncXR, AsyncXR
from .remote import RemoteXRClient
from .settings import settings

XRApp = Union[
    Callable[[AsyncXR, dict[str, Any]], Awaitable[None]],
    Callable[[SyncXR, dict[str, Any]], None],
]


def run(xr_app: XRApp) -> None:
    app = FastAPI()

    async def entrypoint(ws: WebSocket) -> None:
        await ws.accept()
        query_params = dict(ws.query_params)

        async with RemoteXRClient(ws) as remote:
            if inspect.iscoroutinefunction(xr_app):
                axr = AsyncXR(remote)
                await xr_app(axr, query_params)
            else:
                loop = asyncio.get_running_loop()
                loop_thread = threading.current_thread()
                xr = SyncXR(
                    remote,
                    loop,
                    loop_thread)
                await asyncio.to_thread(xr_app, xr, query_params)
        print("normal session shutdown")

    app.add_api_websocket_route(
        settings.ws_path,
        entrypoint)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        ws_max_size=100 * 1024 ** 2  # 100MB
    )
    print("normal server shutdown")


def show_qrcode_link(protocol="ws", address: str = None, path=None, **query_params) -> PIL.Image.Image:
    if address is None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            address = s.getsockname()[0]

    if path is None:
        path = settings.ws_path
    elif not path.startswith("/"):
        path = "/" + path

    url = f"{protocol}://{address}:{settings.port}{path}"
    if query_params:
        url += "?" + urlencode(query_params, doseq=True)
    print(url)
    img = qrcode.make(f"xarp.websocketurl={url}")
    img.show()
    return img
