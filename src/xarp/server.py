import asyncio
import inspect
import io
import secrets
import socket
import threading
from typing import Any, Callable, Awaitable, Union
from urllib.parse import urlencode

import PIL.Image
import qrcode
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException, Response
from fastapi import WebSocket

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


def serve_pil_image_ephemeral(
        img: Image.Image,
        *,
        ttl_seconds: int = 60,
        host: str = "127.0.0.1",
        port: int = 0,  # 0 => choose an ephemeral free port
        path: str = "/image.png",
        fmt: str = "PNG",
) -> str:
    """
    Returns a local URL that serves `img` for at most `ttl_seconds`, via a FastAPI app.
    Side effect: spins up a uvicorn server in a background thread and shuts it down after TTL.

    Notes:
      - This serves on localhost by default (not publicly reachable).
      - If you set host="0.0.0.0", it may be reachable on your LAN (firewall permitting).
    """
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be > 0")

    # Encode the image once (avoid re-encoding per request)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    payload = buf.getvalue()

    content_type = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
    }.get(fmt.upper(), "application/octet-stream")

    token = secrets.token_urlsafe(16)
    served_path = path if path.startswith("/") else "/" + path

    app = FastAPI()

    @app.get(served_path)
    def get_image(token: str):
        if token != token_expected:
            raise HTTPException(status_code=404, detail="Not found")
        return Response(
            content=payload,
            media_type=content_type,
            headers={"Cache-Control": "no-store"},
        )

    # Freeze expected token in closure safely
    token_expected = token

    # Build uvicorn server programmatically so we can shut it down cleanly
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="off",
    )
    server = uvicorn.Server(config)

    # Run uvicorn in a background thread
    def _run():
        server.run()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Wait until server socket is created and port is known
    # (uvicorn sets server.servers once started)
    import time
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if getattr(server, "servers", None):
            break
        time.sleep(0.01)
    if not getattr(server, "servers", None):
        # couldn't start in time
        server.should_exit = True
        raise RuntimeError("Uvicorn server failed to start")

    # Extract actual bound port (handles port=0)
    # server.servers is a list; each has .sockets
    sockets = server.servers[0].sockets
    actual_port = sockets[0].getsockname()[1]

    # Schedule teardown after TTL
    def _shutdown():
        server.should_exit = True

    timer = threading.Timer(ttl_seconds, _shutdown)
    timer.daemon = True
    timer.start()

    return f"http://{host}:{actual_port}{served_path}?token={token}"
