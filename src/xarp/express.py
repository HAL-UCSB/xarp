import asyncio
import base64
import secrets
import socket
import threading
import types
from io import BytesIO
from threading import Thread
from typing import Any, Iterator
from typing import AsyncGenerator

import PIL.Image
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastmcp.utilities.types import Image as MCPImage

from xarp.commands import Bundle, ResponseMode
from xarp.commands.entities import (
    ListAssetsCommand,
    DestroyElementCommand,
    DestroyAssetCommand, ListElementsCommand, CreateOrUpdateAssetsCommand, CreateOrUpdateElementCommand,
)
from xarp.commands.info import InfoCommand
from xarp.commands.sensing import (
    ImageCommand,
    EyeCommand,
    HeadCommand,
    HandsCommand,
    DepthCommand,
    VirtualImageCommand,
)
from xarp.commands.ui import WriteCommand, SayCommand, ReadCommand, PassthroughCommand
from xarp.data_models import DeviceInfo, Hands
from xarp.entities import ImageAsset, Asset, Element, GLBAsset, TextAsset, DefaultAssets
from xarp.remote import RemoteXRClient
from xarp.spatial import Pose, Transform, Vector3, Quaternion


class AsyncXR:
    """Async convenience wrapper around a :class:`~xarp.RemoteXRClient` that exposes common XR operations"""

    def __init__(self, remote: RemoteXRClient):
        self.remote = remote

    async def _execute_none(self, *cmds: Any) -> None:
        await self.remote.execute(Bundle(cmds=list(cmds), mode=ResponseMode.NONE))

    async def _execute_single(self, *cmds: Any) -> Any:
        resp = await self.remote.execute(Bundle(cmds=list(cmds), mode=ResponseMode.SINGLE))
        return resp.value[0] if len(cmds) == 1 else resp.value

    # ---- INFO ----

    async def info(self) -> DeviceInfo:
        """Returns remote system information.

        Returns:
            DeviceInfo describing runtime/device capabilities and configuration.
        """
        return await self._execute_single(InfoCommand())

    # ---- UI ----

    async def write(self, text: str, title: str | None = None) -> None:
        """Displays a text message.

        Args:
            text: Message content to display.
            title: Optional title displayed alongside the message.

        Returns:
            None.
        """
        await self._execute_none(WriteCommand(text=text, title=title))

    async def say(self, text: str) -> None:
        """Plays synthesized speech for a text. Resolves when speech playback completes.

        Args:
            text: Message content to display and speak.
        Returns:
            None.
        """
        await self.remote.execute(
            Bundle(cmds=[SayCommand(text=text)], mode=ResponseMode.SINGLE)
        )

    async def read(self) -> str:
        """Prompts the user for text input and returns it.
        Returns:
            The user's entered text.
        """
        return await self._execute_single(ReadCommand())

    async def passthrough(self, transparency: float) -> None:
        """Sets passthrough transparency.
        Args:
            transparency: Ranges from 0.0 to 1.0. 0.0 is fully virtual, 1.0 is full passthrough.

        Returns:
            None.
        """
        await self._execute_none(PassthroughCommand(transparency=transparency))

    # ---- SENSING (SINGLE) ----

    async def image(self) -> ImageAsset:
        """Captures one RGB image of the physical environment.
        Returns:
            ImageResource containing an RGB image from the user's point of view.
        """
        return await self._execute_single(ImageCommand())

    async def virtual_image(self) -> ImageAsset:
        """Captures one RGBA image of the virtual environment.
        Returns:
            ImageResource containing an RGBA render from the user's point of view.
        """
        return await self._execute_single(VirtualImageCommand())

    async def depth(self) -> ImageAsset:
        """Captures one depth frame of the physical environment.
        Returns:
            ImageResource containing a depth image from the user's point of view.
        """
        return await self._execute_single(DepthCommand())

    async def eye(self) -> Pose:
        """Returns the pose of the main camera (camera extrinsics).
        Returns:
            Pose of the device's main camera, typically close to the user's eye pose.
        """
        return await self._execute_single(EyeCommand())

    async def head(self) -> Pose:
        """Returns the pose of the XR device (typically the user's head pose).
        Returns:
            Pose of the headset in the runtime coordinate frame.
        """
        return await self._execute_single(HeadCommand())

    async def hands(self) -> Hands:
        """Returns tracked hand joint poses.
        Returns:
            Hands payload. ``left`` and ``right`` are None when the corresponding hand
            is not currently tracked.
        """
        return await self._execute_single(HandsCommand())

    # ---- SENSING (STREAM) ----

    async def sense(
            self,
            *,
            image: bool = False,
            virtual_image: bool = False,
            depth: bool = False,
            eye: bool = False,
            head: bool = False,
            hands: bool = False,
            rt: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streams selected sensing modalities continuously. Values are produced later by iterating the returned async generator. Cleanup requires an explict call to the aclose to the returned async generator.
        Args:
            image: If True, include RGB physical-camera frames under key ``"image"``.
            virtual_image: If True, include RGBA virtual-render frames under key
                ``"virtual_image"``.
            depth: If True, include depth frames under key ``"depth"``.
            eye: If True, include camera pose under key ``"eye"``.
            head: If True, include headset pose under key ``"head"``.
            hands: If True, include hand-tracking payload under key ``"hands"``.
            rt: If True, yields the latest available sample at read time (dropping
                intermediate frames). If False, yields samples sequentially even if
                delayed.

        Yields:
            Dictionaries mapping enabled modality keys to their corresponding values.
        """
        keys: list[str] = []
        cmds: list[Any] = []

        if image:
            keys.append("image")
            cmds.append(ImageCommand())
        if virtual_image:
            keys.append("virtual_image")
            cmds.append(VirtualImageCommand())
        if depth:
            keys.append("depth")
            cmds.append(DepthCommand())
        if eye:
            keys.append("eye")
            cmds.append(EyeCommand())
        if head:
            keys.append("head")
            cmds.append(HeadCommand())
        if hands:
            keys.append("hands")
            cmds.append(HandsCommand())

        if not cmds:
            return

        stream = await self.remote.execute(Bundle(cmds=cmds, mode=ResponseMode.STREAM, rt=rt))
        try:
            async for item in stream:
                yield dict(zip(keys, item.value, strict=True))
        finally:
            await stream.aclose()

    # ---- ASSETS ----

    async def save(self, asset: Asset) -> None:
        """Stores an asset on the XR device.
        Args:
            asset: asset object to be stored on the device.
        Returns:
            None.
        """
        await self._execute_single(CreateOrUpdateAssetsCommand(assets=[asset]))

    async def list_assets(self) -> list[str]:
        """Lists stored asset keys.
        Returns:
            List of asset identification keys stored on the device.
        """
        return await self._execute_single(ListAssetsCommand())

    async def destroy_asset(self, asset_key: str | None = None, all_assets: bool = False) -> None:
        """Deletes one asset or all assets.
        Args:
            asset_key: Asset key to delete. Ignored if ``all_assets`` is True.
            all_assets: If True, deletes all stored assets.

        Returns:
            None.
        """
        await self._execute_single(DestroyAssetCommand(asset_key=asset_key, all_assets=all_assets))

    async def update(self, element: Element) -> None:
        """
        Creates or updates (upsert) a remote element. Necessary to apply change the state of a remote element instance.
        Args:
            element: Element holding the desired state of virtual elements on the client.
        Returns:
            None.
        """
        await self._execute_single(CreateOrUpdateElementCommand(elements=[element]))

    async def list_elements(self) -> list[str]:
        """Lists existing elements, both active an inactive.
        Returns:
            List of element identification keys.
        """
        return await self._execute_single(ListElementsCommand())

    async def destroy_element(self, element: Element | None = None, all_elements: bool = False) -> None:
        """Destroys one element or all elements.
        Args:
            element: Element to destroy.
            all_elements: If True, destroys all elements.

        Returns:
            None.
        """
        await self._execute_single(
            DestroyElementCommand(
                key=element.key if element is not None else None,
                all_elements=all_elements,
            )
        )


class AsyncGeneratorIterator(Iterator[dict[str, Any]]):
    """Blocking iterator over an async generator, running on a given loop."""

    def __init__(self, agen, loop: asyncio.AbstractEventLoop):
        self._agen = agen
        self._loop = loop
        self._done = False

    def __iter__(self) -> "AsyncGeneratorIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        if self._done:
            raise StopIteration
        fut = asyncio.run_coroutine_threadsafe(self._agen.__anext__(), self._loop)
        try:
            return fut.result()
        except StopAsyncIteration:
            self._done = True
            raise StopIteration

    def close(self) -> None:
        if self._done:
            return
        self._done = True
        asyncio.run_coroutine_threadsafe(self._agen.aclose(), self._loop).result()


class SyncXR(AsyncXR):

    def __init__(self, remote: RemoteXRClient, loop: asyncio.AbstractEventLoop, loop_thread: Thread):
        super().__init__(remote)
        self._loop = loop
        self._loop_thread = loop_thread

    def _sync(self, coro) -> Any:
        if threading.current_thread() is self._loop_thread:
            raise RuntimeError("SyncXR called from its event loop thread leads to deadlock")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    # ---- INFO ----
    def info(self) -> DeviceInfo:
        return self._sync(super().info())

    # ---- UI ----
    def write(self, text: str, title: str | None = None) -> None:
        return self._sync(super().write(text=text, title=title))

    def say(self, text: str) -> None:
        return self._sync(super().say(text=text))

    def read(self) -> str:
        return self._sync(super().read())

    def passthrough(self, transparency: float) -> None:
        return self._sync(super().passthrough(transparency=transparency))

    # ---- SENSING (SINGLE) ----
    def image(self) -> ImageAsset:
        return self._sync(super().image())

    def virtual_image(self) -> ImageAsset:
        return self._sync(super().virtual_image())

    def depth(self) -> ImageAsset:
        return self._sync(super().depth())

    def eye(self) -> Pose:
        return self._sync(super().eye())

    def head(self) -> Pose:
        return self._sync(super().head())

    def hands(self) -> Hands:
        return self._sync(super().hands())

    # ---- SENSING (STREAM) ----
    def sense(
            self,
            *,
            image: bool = False,
            virtual_image: bool = False,
            depth: bool = False,
            eye: bool = False,
            head: bool = False,
            hands: bool = False,
            rt: bool = True,
    ) -> AsyncGeneratorIterator:
        agen = super().sense(
            image=image,
            virtual_image=virtual_image,
            depth=depth,
            eye=eye,
            head=head,
            hands=hands,
            rt=rt,
        )
        return AsyncGeneratorIterator(agen, self._loop)

    # ---- ENTITIES ----
    def save(self, asset: Asset) -> None:
        return self._sync(super().save(asset))

    def list_assets(self) -> list[str]:
        return self._sync(super().list_assets())

    def destroy_asset(self, asset_key: str | None = None, all_assets: bool = False) -> None:
        return self._sync(super().destroy_asset(asset_key=asset_key, all_assets=all_assets))

    def update(self, element: Element) -> None:
        return self._sync(super().update(element))

    def list_elements(self) -> list[str]:
        return self._sync(super().list_elements())

    def destroy_element(self, element: Element | None = None, all_elements: bool = False) -> None:
        return self._sync(super().destroy_element(element=element, all_elements=all_elements))


class AsyncSimpleXR(AsyncXR):

    async def image(self) -> MCPImage:
        """
        Captures one RGB image of the physical environment from the user's point of view.
        Returns:
            An Image in PNG format.
        """
        asset: ImageAsset = await super().image()
        buf = BytesIO()
        asset.obj.save(buf, format="PNG")
        buf.seek(0)
        return MCPImage(data=buf.getvalue(), format="png")

    async def virtual_image(self) -> MCPImage:
        """
        Captures one RGB image of the virtual environment from the user's point of view.
        Returns:
            An Image in PNG format.
        """
        asset = await super().virtual_image()
        buf = BytesIO()
        asset.obj.save(buf, format="PNG")
        buf.seek(0)
        return MCPImage(data=buf.getvalue(), format="png")

    async def depth(self) -> MCPImage:
        """
        Captures one depth frame of the physical environment.
        Returns:
            An image in PNG format.
        """
        asset = await super().depth()
        buf = BytesIO()
        asset.obj.save(buf, format="PNG")
        buf.seek(0)
        return MCPImage(data=buf.getvalue(), format="png")

    async def info(self) -> dict[str, Any]:
        data = await super().info()
        return data.model_dump()

    async def eye(self) -> dict[str, Any]:
        data = await super().eye()
        return data.model_dump()

    async def head(self) -> dict[str, Any]:
        data = await super().head()
        return data.model_dump()

    async def hands(self) -> dict[str, Any]:
        data = await super().hands()
        return data.model_dump()

    async def create_or_update_glb(self, key: str, url: str,
                                   position: tuple[float, float, float] = (0, 0, 0),
                                   euler_angles: tuple[float, float, float] = (0, 0, 0),
                                   scale: tuple[float, float, float] = (1, 1, 1),
                                   color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update a GLB-based element in the scene.

        If an element with the given key already exists, it is replaced.
        Otherwise, a new element is created and added.

        Args:
            key: Unique identifier for the element.
            url: URL to download the GLB asset.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in radians.
            scale: Non-uniform scale factors (x, y, z).
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        response = requests.get(
            url,
            headers={"User-Agent": "python"},
            timeout=10
        )
        response.raise_for_status()
        glb_bytes = response.content

        element = Element(
            key=key,
            asset=GLBAsset(asset_key=f"asset_{key}", raw=glb_bytes),
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.from_xyz(*scale),
            )
        )
        await self.update(element)

    async def create_or_update_label(self, key: str, text: str,
                                     position: tuple[float, float, float] = (0, 0, 0),
                                     euler_angles: tuple[float, float, float] = (0, 0, 0),
                                     color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update a text label element in the scene.

        The label is rendered as a text-based asset positioned in 3D space.

        Args:
            key: Unique identifier for the element.
            text: Text content of the label.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in radians.
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        element = Element(
            key=key,
            asset=TextAsset.from_obj(text),
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.one(),
            )
        )
        self.update(element)

    async def create_or_update_cube(self, key: str,
                                    position: tuple[float, float, float] = (0, 0, 0),
                                    euler_angles: tuple[float, float, float] = (0, 0, 0),
                                    scale: tuple[float, float, float] = (1, 1, 1),
                                    color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update a cube primitive element in the scene.

        Uses the default cube asset.

        Args:
            key: Unique identifier for the element.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in radians.
            scale: Non-uniform scale factors (x, y, z).
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        element = Element(
            key=key,
            asset=DefaultAssets.CUBE,
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.from_xyz(*scale),
            )
        )
        await self.update(element)

    async def create_or_update_sphere(self, key: str,
                                      position: tuple[float, float, float] = (0, 0, 0),
                                      euler_angles: tuple[float, float, float] = (0, 0, 0),
                                      scale: tuple[float, float, float] = (1, 1, 1),
                                      color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update a sphere primitive element in the scene.

        Uses the default sphere asset.

        Args:
            key: Unique identifier for the element.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in radians.
            scale: Non-uniform scale factors (x, y, z).
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        element = Element(
            key=key,
            asset=DefaultAssets.SPHERE,
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.from_xyz(*scale),
            )
        )
        await self.update(element)

    async def create_or_update_image(self, key: str,
                                     base_64: str,
                                     position: tuple[float, float, float] = (0, 0, 0),
                                     euler_angles: tuple[float, float, float] = (0, 0, 0),
                                     scale: tuple[float, float, float] = (1, 1, 1),
                                     color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update an image-based element in the scene.

        The image is decoded from a base64-encoded string and converted to
        an RGBA texture.

        Args:
            key: Unique identifier for the element.
            base_64: Base64-encoded image data.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in radians.
            scale: Non-uniform scale factors (x, y, z).
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        decoded = base64.b64decode(base_64)
        buffer = BytesIO(decoded)
        img = PIL.Image.open(buffer).convert("RGBA")
        element = Element(
            key=key,
            asset=ImageAsset.from_obj(img),
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.from_xyz(*scale),
            )
        )
        await self.update(element)


class SyncSimpleXR(SyncXR):

    def info(self) -> dict[str, Any]:
        return super().info().model_dump()

    def eye(self) -> dict[str, tuple[float, float, float]]:
        return super().eye().model_dump()

    def head(self) -> dict[str, tuple[float, float, float]]:
        return super().head().model_dump()

    def hands(self) -> dict:
        return super().hands().model_dump()

    def create_or_update_glb(self, key: str, raw: bytes,
                             position: tuple[float, float, float] = (0, 0, 0),
                             euler_angles: tuple[float, float, float] = (0, 0, 0),
                             scale: tuple[float, float, float] = (1, 1, 1),
                             color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update a GLB-based element in the scene.

        If an element with the given key already exists, it is replaced.
        Otherwise, a new element is created and added.

        Args:
            key: Unique identifier for the element.
            raw: Raw bytes of the GLB asset.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in degrees.
            scale: Non-uniform scale factors (x, y, z).
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        element = Element(
            key=key,
            asset=GLBAsset(asset_key=f"asset_{key}", raw=raw),
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.from_xyz(*scale),
            )
        )
        self.update(element)

    def create_or_update_label(self, key: str, text: str,
                               position: tuple[float, float, float] = (0, 0, 0),
                               euler_angles: tuple[float, float, float] = (0, 0, 0),
                               color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update a text label element in the scene.

        The label is rendered as a text-based asset positioned in 3D space.

        Args:
            key: Unique identifier for the element.
            text: Text content of the label.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in degrees.
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        element = Element(
            key=key,
            asset=TextAsset.from_obj(text),
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.one(),
            )
        )
        self.update(element)

    def create_or_update_cube(self, key: str,
                              position: tuple[float, float, float] = (0, 0, 0),
                              euler_angles: tuple[float, float, float] = (0, 0, 0),
                              scale: tuple[float, float, float] = (1, 1, 1),
                              color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update a cube primitive element in the scene.

        Uses the default cube asset.

        Args:
            key: Unique identifier for the element.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in degrees.
            scale: Non-uniform scale factors (x, y, z).
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        element = Element(
            key=key,
            asset=DefaultAssets.CUBE,
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.from_xyz(*scale),
            )
        )
        self.update(element)

    def create_or_update_sphere(self, key: str,
                                position: tuple[float, float, float] = (0, 0, 0),
                                euler_angles: tuple[float, float, float] = (0, 0, 0),
                                scale: tuple[float, float, float] = (1, 1, 1),
                                color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update a sphere primitive element in the scene.

        Uses the default sphere asset.

        Args:
            key: Unique identifier for the element.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in degrees.
            scale: Non-uniform scale factors (x, y, z).
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        element = Element(
            key=key,
            asset=DefaultAssets.SPHERE,
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.from_xyz(*scale),
            )
        )
        self.update(element)

    def create_or_update_image(self, key: str,
                               base_64: str,
                               position: tuple[float, float, float] = (0, 0, 0),
                               euler_angles: tuple[float, float, float] = (0, 0, 0),
                               scale: tuple[float, float, float] = (1, 1, 1),
                               color: tuple[float, float, float, float] = (1, 1, 1, 1)) -> None:
        """Create or update an image-based element in the scene.

        The image is decoded from a base64-encoded string and converted to
        an RGBA texture.

        Args:
            key: Unique identifier for the element.
            base_64: Base64-encoded image data.
            position: World-space position (x, y, z).
            euler_angles: Rotation expressed as Euler angles (roll, pitch, yaw),
                in degrees.
            scale: Non-uniform scale factors (x, y, z).
            color: RGBA color multiplier with components in [0.0, 1.0].

        Returns:
            None
        """
        decoded = base64.b64decode(base_64)
        buffer = BytesIO(decoded)
        img = PIL.Image.open(buffer).convert("RGBA")
        element = Element(
            key=key,
            asset=ImageAsset.from_obj(img),
            color=color,
            transform=Transform(
                position=Vector3.from_xyz(*position),
                rotation=Quaternion.from_euler_angles(*euler_angles),
                scale=Vector3.from_xyz(*scale),
            )
        )
        self.update(element)


def copy_public_methods_doc(from_class, to_class):
    for name, member in to_class.__dict__.items():
        if name.startswith("_"):
            continue
        if not isinstance(member, types.FunctionType):
            continue
        src = getattr(from_class, name, None)
        src_doc = getattr(src, "__doc__", None)
        if not src_doc:
            continue
        member.__doc__ = src_doc


copy_public_methods_doc(AsyncXR, SyncXR)


def serve_pil_image_ephemeral(
        img: PIL.Image.Image,
        *,
        ttl_seconds: int = 60,
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
    buf = BytesIO()
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

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        host = s.getsockname()[0]

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
