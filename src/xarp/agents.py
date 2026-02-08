import asyncio
import inspect
import threading
from sys import argv
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI
from fastmcp import FastMCP
from smolagents import tool, CodeAgent, MultiStepAgent, ActionStep
from smolagents.tools import Tool
from starlette.websockets import WebSocket

from xarp import server
from xarp.entities import ImageAsset
from xarp.express import SyncXR, AsyncXR, SyncSimpleXR, AsyncSimpleXR
from xarp.remote import RemoteXRClient
from xarp.server import show_qrcode_link
from xarp.settings import settings

XRAgentApp = Callable[[SyncXR, MultiStepAgent, dict[str, Any]], None]

_ALLOWED_TOOLS = (
    "info",
    "write",
    "say",
    "read",
    "passthrough",
    "image",
    "virtual_image",
    "depth",
    "eye",
    "head",
    "hands",
    "list_assets",
    "list_elements",
    "destroy_element",
    "create_or_update_glb",
    "create_or_update_label",
    "create_or_update_cube",
    "create_or_update_sphere",
    "create_or_update_image",
    "reconstruction_3d"
)


class ImageAssetToolInterceptor:

    def __init__(self):
        self.intercepted_images = []

    def intercept(self, image_asset_tool: Tool):
        tool_forward = image_asset_tool.forward

        def _wrapper(*args, **kwargs):
            image_asset: ImageAsset = tool_forward(*args, **kwargs)
            self.intercepted_images.append(image_asset.obj)
            n_images = len(self.intercepted_images)
            print("Images to observe:", n_images)
            return image_asset.model_dump()

        image_asset_tool.forward = _wrapper

    def provide_observations(self, step: ActionStep):
        step.observations_images = [img.copy() for img in self.intercepted_images]

    @staticmethod
    def attach_to_agent(agent: MultiStepAgent):
        interceptor = ImageAssetToolInterceptor()
        for _tool in agent.tools.values():
            if _tool.forward.__annotations__["return"] is ImageAsset:
                interceptor.intercept(_tool)
        agent.step_callbacks.register(ActionStep, interceptor.provide_observations)


def _get_public_methods(obj) -> list[tuple[str, Any]]:
    public_methods = []
    for pair in inspect.getmembers(obj, inspect.ismethod):
        name = pair[0]
        if not name.startswith("_") and name in _ALLOWED_TOOLS:
            public_methods.append(pair)
    return public_methods


def as_agent_tools(xr: SyncXR) -> list[Tool]:
    tools = [tool(member) for name, member in _get_public_methods(xr)]
    return tools


def run_xr_agent(xr_agent_app: XRAgentApp, model, **kwargs) -> None:
    async def _with_agent(axr: AsyncXR, params: dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()
        loop_thread = threading.current_thread()
        sxr = SyncSimpleXR(axr.remote, loop, loop_thread)

        agent = CodeAgent(
            tools=as_agent_tools(sxr),
            model=model,
            additional_authorized_imports=[
                "numpy",
                "numpy.linalg",
                "base64",
                "math",
                "time",
            ],
            **kwargs
        )

        xr_prompt = """
        You are in a right-handed coordinate system. +X is right, +Y is up, and +Z is forward.
        Positions and scales are in meters.
        """
        agent.prompt_templates["system_prompt"] = agent.prompt_templates["system_prompt"] + xr_prompt

        ImageAssetToolInterceptor.attach_to_agent(agent)

        await asyncio.to_thread(xr_agent_app, sxr, agent, params)

    server.run(_with_agent)


def run_mcp():
    async def entrypoint(ws: WebSocket) -> None:
        await ws.accept()

        remote = RemoteXRClient(ws)
        await remote.start()
        asxr = AsyncSimpleXR(remote)

        mcp = FastMCP(
            argv[1],
            host="127.0.0.1",
            port=argv[2]
        )

        for name, method in _get_public_methods(asxr):
            mcp.tool(method)

        try:
            await mcp.run_http_async(show_banner=False)
        finally:
            remote.stop()

    app = FastAPI()
    app.add_api_websocket_route(
        settings.ws_path,
        entrypoint)

    show_qrcode_link()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        ws_max_size=100 * 1024 ** 2  # 100MB
    )


if __name__ == '__main__':
    run_mcp()
