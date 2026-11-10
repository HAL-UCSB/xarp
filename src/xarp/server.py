"""WebSocket server and connection helpers for XARP applications."""

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

#: Asynchronous or synchronous application callable accepted by :func:`run`.
XRApp = (
        Callable[[AsyncXR, dict[str, Any]], Awaitable[None]]
        | Callable[[SyncXR, dict[str, Any]], None]
)


def run(xr_app: XRApp) -> None:
    """Start the XR WebSocket server and block until it shuts down.

    The callable is invoked once for each accepted client session. Asynchronous
    callables receive :class:`xarp.express.AsyncXR`; synchronous callables
    receive :class:`xarp.express.SyncXR` and run in a worker thread. The second
    argument contains the WebSocket URL query parameters as strings.

    The listening host, port, and WebSocket path come from
    :func:`xarp.settings.get_settings`. This function owns the Uvicorn event loop
    and does not return until the server shuts down.

    Args:
        xr_app: Function with the signature ``(xr, query_params) -> None``. An
            async function may return an awaitable; a synchronous function must
            return normally.

    Raises:
        Exception: Re-raises an exception from ``xr_app`` after logging it.

    Example:
        Run a blocking application::

            def app(xr, query_params):
                xr.write(f"Connected as {query_params.get('name', 'guest')}")

            run(app)
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


def make_qrcode_image(
        protocol: str = "ws",
        address: str | None = None,
        path: str | None = None,
        show: bool = True,
        qr_code_schema="xarp.websocketurl",
        **query_params: Any,
) -> PIL.Image.Image:
    """Build and optionally display a QR code for the XR WebSocket URL.

    The QR payload uses the ``xarp.websocketurl=<url>`` scheme so that the
    client app can detect and connect automatically on scan. Query parameter
    values are encoded with :func:`urllib.parse.urlencode`; sequences are
    expanded into repeated keys.

    Args:
        protocol: WebSocket scheme, normally ``"ws"`` or ``"wss"``.
        address: Server IP address or hostname. Defaults to the machine's LAN
            address.
        path: WebSocket route. Defaults to the configured ``ws_path``. A leading
            slash is added when needed.
        show: Whether to open the QR image with Pillow's default image viewer.
        qr_code_schema: Prefix used by the client to recognize the QR payload.
        **query_params: Additional URL query parameters included in the encoded
            WebSocket URL.

    Returns:
        Pillow image containing the generated QR code.

    Raises:
        OSError: If ``address`` is omitted and the LAN address cannot be
            determined.
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
