import asyncio
import inspect
import logging
import socket
import threading
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode

import PIL.Image
import qrcode
import uvicorn
from fastapi import FastAPI, WebSocket

from .express import AsyncXR, SyncXR
from .remote import RemoteXRClient
from .settings import get_settings

log = logging.getLogger(__name__)

XRApp = (
        Callable[[AsyncXR, dict[str, Any]], Awaitable[None]]
        | Callable[[SyncXR, dict[str, Any]], None]
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def run(xr_app: XRApp) -> None:
    """
    Start the XR WebSocket server and block until it shuts down.

    Accepts both async and sync XR app callables.  Sync callables are
    dispatched via asyncio.to_thread so the event loop is never blocked.
    """
    app = FastAPI()
    settings = get_settings()

    @app.websocket(settings.ws_path)
    async def entrypoint(ws: WebSocket) -> None:
        await ws.accept()
        query_params = dict(ws.query_params)

        try:
            async with RemoteXRClient(ws) as remote:
                if inspect.iscoroutinefunction(xr_app):
                    await xr_app(AsyncXR(remote), query_params)
                else:
                    xr = SyncXR(remote, asyncio.get_running_loop(), threading.current_thread())
                    await asyncio.to_thread(xr_app, xr, query_params)
        except Exception:
            log.exception("Unhandled exception in xr_app")
            raise

        log.info("Session closed normally")

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        ws_max_size=100 * 1024 ** 2,  # 100 MB
    )
    log.info("Server shut down normally")


# ---------------------------------------------------------------------------
# QR code helper
# ---------------------------------------------------------------------------

def make_qrcode_image(
        protocol: str = "ws",
        address: str | None = None,
        path: str | None = None,
        show: bool = True,
        qr_code_schema="xarp.websocketurl",
        **query_params: Any,
) -> PIL.Image.Image:
    """
    Build and optionally display a QR code encoding the XR WebSocket URL.

    The QR payload uses the ``xarp.websocketurl=<url>`` scheme so that the
    client app can detect and connect automatically on scan.

    Args:
        protocol:     WebSocket scheme — ``"ws"`` or ``"wss"``.
        address:      Server IP or hostname.  Defaults to the local LAN address.
        path:         WebSocket path.  Defaults to ``settings.ws_path``.
        show:         If True, open the QR image in the default viewer.
        qr_scheme:    The client app detects this prefix to extract and connect to the WebSocket URL.
        **query_params: Additional URL query parameters forwarded to the client.

    Returns:
        A PIL Image of the QR code.
    """

    if address is None:
        address = _local_ip()

    settings = get_settings()
    if path is None:
        path = settings.ws_path
    elif not path.startswith("/"):
        path = "/" + path

    url = f"{protocol}://{address}:{settings.port}{path}"
    if query_params:
        url += "?" + urlencode(query_params, doseq=True)

    log.info("QR code URL: %s", url)

    img = qrcode.make(f"{qr_code_schema}={url}")

    if show:
        img.show()

    return img


def _local_ip() -> str:
    """Return the machine's LAN IP by probing an external address."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
